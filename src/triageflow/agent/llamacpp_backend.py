"""llama.cpp backend — manages a local ``llama-server`` (OpenAI-compatible) + GGUF models.

Design (industry best practice):
  * Chat goes through ``llama-server``'s OpenAI-compatible ``/v1`` endpoint — identical client
    code to every other backend.
  * We own the server process: launch ``llama-server -m <gguf> --host .. --port .. --jinja``,
    health-check ``/health``, and restart it to switch models (llama-server serves one model
    per process). A pre-existing server on the port is reused.
  * Models are single-file GGUF quants downloaded from Hugging Face with real progress; a
    lighter model is offered so a low-end box isn't forced onto the 7B.

The ``llama-server`` binary is installed separately (e.g. ``brew install llama.cpp``); we
detect it and give a clear hint if it's missing — the app still runs deterministically.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..core.config import settings
from .downloads import download_with_resume
from .llm_backend import BackendStatus, ModelOption


@dataclass(frozen=True)
class _GGUF:
    id: str
    repo: str
    filename: str
    label: str
    tier: str
    size_gb: float


_CATALOG: list[_GGUF] = [
    _GGUF("qwen2.5-7b-instruct-q4", "bartowski/Qwen2.5-7B-Instruct-GGUF",
          "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Qwen2.5 7B · standard", "standard", 4.7),
    _GGUF("qwen2.5-3b-instruct-q4", "bartowski/Qwen2.5-3B-Instruct-GGUF",
          "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5 3B · small / faster", "small", 1.9),
    _GGUF("qwen2.5-0.5b-instruct-q4", "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
          "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "Qwen2.5 0.5B · tiny / low-end", "small", 0.4),
]
_BY_ID = {g.id: g for g in _CATALOG}


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._active: str | None = None
        settings.models_dir.mkdir(parents=True, exist_ok=True)

    # --- URLs / paths ---------------------------------------------------------
    def _base(self) -> str:
        return f"http://{settings.llamacpp_host}:{settings.llamacpp_port}"

    def chat_base_url(self) -> str:
        return self._base() + "/v1"

    def gguf_path(self, model_id: str) -> Path:
        g = _BY_ID.get(model_id)
        return settings.models_dir / (g.filename if g else f"{model_id}.gguf")

    # --- runtime binary (detect / self-provision) -----------------------------
    def _provisioned_bin(self) -> Path:
        exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        return settings.home / "bin" / "llamacpp" / exe

    def _bin(self) -> str | None:
        """Resolved llama-server: one on PATH, else one we provisioned under ~/.triageflow."""
        found = shutil.which(settings.llamacpp_bin)
        if found:
            return found
        p = self._provisioned_bin()
        return str(p) if p.is_file() else None

    def runtime_installed(self) -> bool:
        return self._bin() is not None

    def can_provision(self) -> bool:
        # llama.cpp publishes prebuilt binaries only for Windows; elsewhere we guide the user.
        return platform.system() == "Windows"

    def ensure_runtime(self, on_progress: Callable[[str], None] | None = None) -> bool:
        """Make a llama-server binary available. If none is installed, download the prebuilt
        Windows release (resumable). On macOS/Linux there is no official prebuilt binary, so we
        return False and the caller surfaces the one-line install hint."""
        if self.runtime_installed():
            return True
        if not self.can_provision():
            return False
        try:
            arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
            rel = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases",
                            params={"per_page": "10"}, timeout=20, trust_env=False).json()
            asset = None
            for r in rel:
                for a in r.get("assets", []):
                    n = a["name"]
                    if n.endswith(".zip") and "bin-win-cpu" in n and arch in n:
                        asset = a
                        break
                if asset:
                    break
            if not asset:
                return False
            zip_path = settings.home / "bin" / asset["name"]
            if on_progress:
                on_progress(f"downloading llama.cpp runtime ({asset['size'] >> 20} MB)…")
            if not download_with_resume(asset["browser_download_url"], zip_path,
                                        label="llama.cpp runtime", on_progress=on_progress):
                return False
            dest = self._provisioned_bin().parent
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest)
            # flatten: binaries may be nested in a subfolder
            for p in dest.rglob("llama-server*"):
                if p.name in ("llama-server", "llama-server.exe"):
                    target = self._provisioned_bin()
                    if p != target:
                        for f in p.parent.iterdir():  # keep DLLs beside the exe
                            try:
                                shutil.copy2(f, dest / f.name)
                            except (shutil.SameFileError, OSError):
                                pass
                    os.chmod(target, 0o755)
                    break
            zip_path.unlink(missing_ok=True)
            return self.runtime_installed()
        except Exception as e:
            if on_progress:
                on_progress(f"runtime provisioning failed: {type(e).__name__}: {e}")
            return False

    def server_up(self) -> bool:
        try:
            r = httpx.get(self._base() + "/health", timeout=2.0, trust_env=False)
            return r.status_code == 200
        except Exception:
            return False

    def is_installed(self, model_id: str) -> bool:
        return self.gguf_path(model_id).is_file()

    def installed_models(self) -> list[str]:
        return [g.id for g in _CATALOG if self.is_installed(g.id)]

    def catalog(self) -> list[ModelOption]:
        return [ModelOption(g.id, g.label, g.tier, g.size_gb, self.is_installed(g.id))
                for g in _CATALOG]

    def _pick_model(self) -> str | None:
        """Which model to serve: the active one, else the configured default, else any installed."""
        for cand in (self._active, settings.llamacpp_default_model):
            if cand and self.is_installed(cand):
                return cand
        inst = self.installed_models()
        return inst[0] if inst else None

    def ensure_server(self, wait_s: float = 0.0,
                      on_progress: Callable[[str], None] | None = None) -> bool:
        if self.server_up():
            if self._active is None:
                self._active = settings.llamacpp_default_model
            return True
        if not self.runtime_installed() and not self.ensure_runtime(on_progress):
            return False
        model_id = self._pick_model()
        if not model_id:
            return False  # runtime ready, but no model downloaded yet (user picks one)
        return self._start_server_for(model_id, wait_s=max(wait_s, 60.0))

    def _start_server_for(self, model_id: str, wait_s: float = 60.0) -> bool:
        self._stop()
        gguf = self.gguf_path(model_id)
        binary = self._bin()
        if not gguf.is_file() or not binary:
            return False
        cmd = [binary, "-m", str(gguf),
               "--host", settings.llamacpp_host, "--port", str(settings.llamacpp_port),
               "-c", str(settings.llamacpp_ctx), "-ngl", str(settings.llamacpp_ngl), "--jinja"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.server_up():
                self._active = model_id
                return True
            if self._proc.poll() is not None:  # process died
                return False
            time.sleep(0.5)
        return False

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # --- models ---------------------------------------------------------------
    def pull_model(self, model_id: str,
                   on_progress: Callable[[str], None] | None = None) -> bool:
        g = _BY_ID.get(model_id)
        if g is None:
            if on_progress:
                on_progress(f"unknown model: {model_id}")
            return False
        if self.is_installed(model_id):
            return True
        url = f"https://huggingface.co/{g.repo}/resolve/main/{g.filename}?download=true"
        # Resumable + retrying — survives network interruptions (a .part file is resumed).
        return download_with_resume(url, self.gguf_path(model_id), label=g.label,
                                    on_progress=on_progress)

    def set_active_model(self, model_id: str) -> None:
        # Switching model = restart llama-server with the new GGUF (one model per process).
        if self.is_installed(model_id):
            self._start_server_for(model_id)

    def resolve_model(self) -> tuple[str | None, bool]:
        if self.server_up() and self._active:
            return self._active, (self._active != settings.llamacpp_default_model)
        return None, False

    def active_model(self) -> str | None:
        return self.resolve_model()[0]

    def status(self) -> BackendStatus:
        installed = self.runtime_installed()
        up = self.server_up()
        model, fb = self.resolve_model()
        if not installed:
            msg = "llama.cpp runtime not found — the agent runs deterministically meanwhile."
            sysname = platform.system()
            if sysname == "Windows":
                hint = "The llama.cpp runtime will be downloaded automatically — pick a model below."
            elif sysname == "Darwin":
                hint = "Install once with `brew install llama.cpp`, then restart (or use Ollama)."
            else:
                hint = "Install llama.cpp (`llama-server` on PATH) or use Ollama, then restart."
        elif not up:
            msg = "llama.cpp runtime ready — pick a model to download/start below."
            hint = ""
        else:
            msg = "Ready"
            hint = ""
        return BackendStatus(
            backend="llamacpp", runtime_installed=installed, server_up=up, model=model,
            using_fallback=fb, catalog=self.catalog(), message=msg, hint=hint)
