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

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..core.config import settings
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

    # --- runtime / server -----------------------------------------------------
    def runtime_installed(self) -> bool:
        return shutil.which(settings.llamacpp_bin) is not None

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

    def ensure_server(self, wait_s: float = 0.0) -> bool:
        if self.server_up():
            if self._active is None:
                self._active = settings.llamacpp_default_model
            return True
        if not self.runtime_installed():
            return False
        model_id = self._pick_model()
        if not model_id:
            return False  # nothing to serve yet — needs a download
        return self._start_server_for(model_id, wait_s=max(wait_s, 60.0))

    def _start_server_for(self, model_id: str, wait_s: float = 60.0) -> bool:
        self._stop()
        gguf = self.gguf_path(model_id)
        if not gguf.is_file() or not self.runtime_installed():
            return False
        cmd = [settings.llamacpp_bin, "-m", str(gguf),
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
        dest = self.gguf_path(model_id)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = f"https://huggingface.co/{g.repo}/resolve/main/{g.filename}?download=true"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                last = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if on_progress and total and done - last > (20 << 20):  # every ~20MB
                            last = done
                            on_progress(f"downloading {g.label} — {100 * done / total:.0f}% "
                                        f"({done >> 20}/{total >> 20} MB)")
            tmp.rename(dest)
            if on_progress:
                on_progress(f"downloaded {g.label}")
            return True
        except Exception as e:  # cleanup partials so a retry is clean
            tmp.unlink(missing_ok=True)
            if on_progress:
                on_progress(f"download failed: {type(e).__name__}: {e}")
            return False

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
            msg = "llama-server not found — the agent runs deterministically."
            hint = "Install with `brew install llama.cpp` (or build llama.cpp), then restart."
        elif not up:
            msg = "llama-server not running yet — pick a model to download/start below."
            hint = ""
        else:
            msg = "Ready"
            hint = ""
        return BackendStatus(
            backend="llamacpp", runtime_installed=installed, server_up=up, model=model,
            using_fallback=fb, catalog=self.catalog(), message=msg, hint=hint)
