"""llama.cpp backend — manages a local ``llama-server`` (OpenAI-compatible) + GGUF models.

Design (industry best practice):
  * Chat goes through ``llama-server``'s OpenAI-compatible ``/v1`` endpoint — identical client
    code to every other backend.
  * We own the server process: launch ``llama-server -m <gguf> --host .. --port ..``,
    health-check ``/health``, and restart it to switch models (llama-server serves one model
    per process). A pre-existing server on the port is reused; the child is stopped on exit.
  * **Self-provisioning on Windows, macOS and Linux.** If no ``llama-server`` is on PATH, the
    official prebuilt binary for this OS/CPU is downloaded from the llama.cpp GitHub releases
    (resumable), unpacked under ``~/.triageflow/bin`` and verified by actually running it.
    No Homebrew, no compiler, no admin rights.
  * Models are single-file GGUF quants downloaded from Hugging Face with real progress; a
    lighter model is offered so a low-end box isn't forced onto the 7B. A GGUF that Ollama
    already downloaded is reused in place rather than fetched again.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..core.config import settings
from .downloads import download_with_resume
from .llm_backend import BackendStatus, ModelOption

_REPO_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
_REPO_DL = "https://github.com/ggml-org/llama.cpp/releases/download"
# Used only if the GitHub API can't be reached (offline mirror, rate limit): the build that
# the llama.cpp v1.0.0 stable release points at.
PINNED_TAG = "b12359"


@dataclass(frozen=True)
class _GGUF:
    id: str
    repo: str
    filename: str
    label: str
    tier: str
    size_gb: float
    ollama: tuple[str, str]      # (library name, tag) of the same weights in an Ollama store


_CATALOG: list[_GGUF] = [
    _GGUF("qwen2.5-7b-instruct-q4", "bartowski/Qwen2.5-7B-Instruct-GGUF",
          "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Qwen2.5 7B · standard", "standard", 4.7,
          ("qwen2.5", "7b-instruct")),
    _GGUF("qwen2.5-3b-instruct-q4", "bartowski/Qwen2.5-3B-Instruct-GGUF",
          "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5 3B · small / faster", "small", 1.9,
          ("qwen2.5", "3b-instruct")),
    _GGUF("qwen2.5-0.5b-instruct-q4", "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
          "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "Qwen2.5 0.5B · tiny / low-end", "small", 0.4,
          ("qwen2.5", "0.5b-instruct")),
]
_BY_ID = {g.id: g for g in _CATALOG}


def runtime_asset_slug(system: str | None = None, machine: str | None = None) -> str | None:
    """The release-asset suffix for this OS/CPU, e.g. ``macos-arm64.tar.gz`` — or None if
    llama.cpp publishes no CPU build for it (then the user installs llama-server themselves)."""
    sysname = system or platform.system()
    m = (machine or platform.machine() or "").lower()
    arch = ("arm64" if m in ("arm64", "aarch64") else
            "x64" if m in ("x86_64", "amd64", "x64") else None)
    if arch is None:
        return None
    return {"Windows": f"win-cpu-{arch}.zip",
            "Darwin": f"macos-{arch}.tar.gz",
            "Linux": f"ubuntu-{arch}.tar.gz"}.get(sysname)


def _exe_name() -> str:
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_in_use(host: str, port: int) -> bool:
    """True if we could not bind the port ourselves. (Probing with connect() is unreliable:
    a busy listener with a full backlog looks 'free' and llama-server then fails to bind.)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if platform.system() != "Windows":
            # A server that has just exited leaves the port in TIME_WAIT for a while; without
            # this the port would look taken and we would drift to a random one. (Not on
            # Windows, where the same option means "steal a port someone is listening on".)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._active: str | None = None
        self._port: int = settings.llamacpp_port
        self._adopted_pid: int | None = None   # a server WE started in an earlier, killed run
        self._closing = False
        self.last_error: str = ""
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)   # never leave an orphaned llama-server holding RAM

    # --- URLs / paths ---------------------------------------------------------
    def _base(self) -> str:
        return f"http://{settings.llamacpp_host}:{self._port}"

    def chat_base_url(self) -> str:
        return self._base() + "/v1"

    def _ollama_blob(self, model_id: str) -> Path | None:
        """The same weights already downloaded by Ollama, if present (Ollama stores models as
        plain GGUF blobs). Reusing it avoids a second multi-GB download."""
        g = _BY_ID.get(model_id)
        if g is None or not settings.reuse_ollama_models:
            return None
        roots = [Path(os.environ["OLLAMA_MODELS"])] if os.environ.get("OLLAMA_MODELS") else [
            Path.home() / ".ollama" / "models", Path("/usr/share/ollama/.ollama/models")]
        name, tag = g.ollama
        for root in roots:
            try:
                manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
                layers = json.loads(manifest.read_text(encoding="utf-8")).get("layers", [])
                for layer in layers:
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        blob = root / "blobs" / layer["digest"].replace(":", "-")
                        if blob.is_file() and os.access(blob, os.R_OK):
                            return blob
            except (OSError, ValueError, KeyError):
                continue
        return None

    def gguf_path(self, model_id: str) -> Path:
        """Where this model's weights are (our download, else Ollama's copy), or where our
        download will go."""
        g = _BY_ID.get(model_id)
        own = settings.models_dir / (g.filename if g else f"{model_id}.gguf")
        if own.is_file():
            return own
        return self._ollama_blob(model_id) or own

    # --- runtime binary (detect / self-provision) -----------------------------
    def _runtime_dir(self) -> Path:
        return settings.home / "bin" / "llamacpp"

    def _provisioned_bin(self) -> Path:
        """The llama-server we unpacked under ~/.triageflow (kept in its release layout, so
        its shared libraries stay next to it). Returns the default path if none exists yet."""
        root = self._runtime_dir()
        if root.is_dir():
            found = sorted(p for p in root.rglob(_exe_name()) if p.is_file())
            if found:
                return found[0]
        return root / _exe_name()

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
        return runtime_asset_slug() is not None

    def _resolve_asset(self) -> tuple[str, str, int]:
        """(file name, download URL, size) of the right prebuilt binary. Prefers the build the
        latest STABLE release points to; falls back to recent builds, then to a pinned tag."""
        slug = runtime_asset_slug()
        pat = re.compile(rf"^llama-b\d+-bin-{re.escape(slug)}$")
        try:
            with httpx.Client(timeout=20, trust_env=False, follow_redirects=True,
                              headers={"Accept": "application/vnd.github+json"}) as c:
                releases = []
                latest = c.get(f"{_REPO_API}/latest").json()
                for a in latest.get("assets", []):
                    if a.get("name") == "nightly-tag.txt":   # stable -> the build it ships
                        tag = c.get(a["browser_download_url"]).text.strip()
                        if re.fullmatch(r"b\d+", tag):
                            releases.append(c.get(f"{_REPO_API}/tags/{tag}").json())
                releases.append(latest)
                if not any(pat.match(a.get("name", "")) for r in releases
                           for a in r.get("assets", [])):
                    releases += c.get(_REPO_API, params={"per_page": "15"}).json()
                for r in releases:
                    for a in r.get("assets", []) if isinstance(r, dict) else []:
                        if pat.match(a.get("name", "")):
                            return a["name"], a["browser_download_url"], int(a.get("size", 0))
        except Exception:
            pass
        name = f"llama-{PINNED_TAG}-bin-{slug}"
        return name, f"{_REPO_DL}/{PINNED_TAG}/{name}", 0

    @staticmethod
    def _unpack(archive: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                root = dest.resolve()
                for member in z.namelist():        # refuse paths that escape the folder
                    if not (dest / member).resolve().is_relative_to(root):
                        raise ValueError(f"unsafe path in archive: {member}")
                z.extractall(dest)
        else:
            with tarfile.open(archive) as t:
                try:
                    t.extractall(dest, filter="data")  # blocks absolute paths / traversal
                except TypeError:                      # Python 3.11.0-3.11.3: no filter arg
                    root = dest.resolve()
                    for member in t.getmembers():
                        if not (dest / member.name).resolve().is_relative_to(root):
                            raise ValueError(f"unsafe path in archive: {member.name}") from None
                    t.extractall(dest)

    def _runs(self, binary: str) -> tuple[bool, str]:
        """Actually execute the binary — the only real proof it works on THIS machine
        (right CPU, shared libraries present, not blocked by the OS)."""
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True,
                               timeout=60, **_no_window())
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["no output"]
            return False, tail[0]
        return True, ""

    def ensure_runtime(self, on_progress: Callable[[str], None] | None = None) -> bool:
        """Make a working llama-server available. If none is installed, download the official
        prebuilt binary for this OS/CPU (resumable), unpack it and verify that it runs."""
        say = on_progress or (lambda _s: None)
        if self.runtime_installed():
            return True
        if not self.can_provision():
            self.last_error = (f"No prebuilt llama.cpp for {platform.system()} "
                               f"{platform.machine()} — install llama-server manually.")
            return False
        try:
            name, url, size = self._resolve_asset()
            root = self._runtime_dir()
            archive = settings.home / "bin" / name
            say(f"downloading llama.cpp runtime ({size >> 20} MB)…" if size
                else "downloading llama.cpp runtime…")
            if not download_with_resume(url, archive, label="llama.cpp runtime",
                                        on_progress=on_progress):
                self.last_error = "Could not download the llama.cpp runtime (network?)."
                return False
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            self._unpack(archive, root)
            archive.unlink(missing_ok=True)
            binary = self._provisioned_bin()
            if not binary.is_file():
                self.last_error = "The llama.cpp download did not contain llama-server."
                return False
            if platform.system() != "Windows":
                for f in binary.parent.iterdir():   # archives made on Windows drop the +x bit
                    if f.is_file() and not f.suffix:
                        f.chmod(f.stat().st_mode | 0o755)
            ok, why = self._runs(str(binary))
            if not ok:
                shutil.rmtree(root, ignore_errors=True)
                self.last_error = (f"The prebuilt llama.cpp runtime does not run on this "
                                   f"machine ({why}). Install llama-server manually or use Ollama.")
                say(self.last_error)
                return False
            say("llama.cpp runtime ready")
            return True
        except Exception as e:
            self.last_error = f"runtime provisioning failed: {type(e).__name__}: {e}"
            say(self.last_error)
            return False

    def _heal_runtime(self, on_progress: Callable[[str], None] | None = None) -> bool:
        """If the runtime WE unpacked no longer runs (files removed by a cleaner, antivirus
        quarantine, an interrupted update), fetch it again instead of failing every start.
        A llama-server the user installed on PATH is theirs — it is never replaced."""
        binary = self._bin()
        ours = self._provisioned_bin()
        if not binary or not ours.is_file() or Path(binary) != ours:
            return True
        ok, why = self._runs(binary)
        if ok:
            return True
        if on_progress:
            on_progress(f"llama.cpp runtime is damaged ({why}) — downloading it again…")
        shutil.rmtree(self._runtime_dir(), ignore_errors=True)
        return self.ensure_runtime(on_progress)

    # --- server lifecycle -----------------------------------------------------
    def server_up(self) -> bool:
        try:
            r = httpx.get(self._base() + "/health", timeout=2.0, trust_env=False)
            return r.status_code == 200 and "status" in r.text
        except Exception:
            return False

    def _is_llama_server(self) -> bool:
        """A server we did not start is only adopted if it really speaks the model API — some
        other program on the port with its own /health must not be mistaken for the model."""
        try:
            data = httpx.get(self._base() + "/v1/models", timeout=3.0, trust_env=False).json()
            return isinstance(data.get("data"), list)
        except Exception:
            return False

    def _served_model(self) -> str | None:
        """Catalog id of the model a running llama-server is serving (via its alias/path)."""
        try:
            data = httpx.get(self._base() + "/v1/models", timeout=3.0,
                             trust_env=False).json().get("data", [])
        except Exception:
            return None
        for m in data:
            mid = str(m.get("id", ""))
            if mid in _BY_ID:
                return mid
            for g in _CATALOG:
                if g.filename.lower() in mid.lower():
                    return g.id
        return None

    def is_installed(self, model_id: str) -> bool:
        return model_id in _BY_ID and self.gguf_path(model_id).is_file()

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
        ours = self._proc is not None and self._proc.poll() is None
        if not ours and self._adopted_pid is None:
            self._recover_previous_server()
        if self.server_up() and (ours or self._adopted_pid or self._is_llama_server()):
            if self._active is None:
                self._active = self._served_model()
            return True
        if not self.runtime_installed() and not self.ensure_runtime(on_progress):
            return False
        if not self._heal_runtime(on_progress):
            return False
        model_id = self._pick_model()
        if not model_id:
            return False  # runtime ready, but no model downloaded yet (user picks one)
        if on_progress:
            on_progress(f"loading {_BY_ID[model_id].label} into memory…")
        return self._start_server_for(model_id, wait_s=max(wait_s, 180.0))

    def _command(self, binary: str, gguf: Path, model_id: str, minimal: bool) -> list[str]:
        cmd = [binary, "-m", str(gguf), "--host", settings.llamacpp_host,
               "--port", str(self._port), "-c", str(settings.llamacpp_ctx)]
        if not minimal:   # niceties a much older/newer llama-server might not accept
            cmd += ["-ngl", str(settings.llamacpp_ngl), "--alias", model_id, "--jinja"]
        return cmd

    def _start_server_for(self, model_id: str, wait_s: float = 180.0) -> bool:
        self._stop()
        gguf = self.gguf_path(model_id)
        binary = self._bin()
        if not gguf.is_file() or not binary:
            self.last_error = "Model file or llama-server binary is missing."
            return False
        # The configured port may belong to something else (8080 is popular): take a free one.
        self._port = settings.llamacpp_port
        if _port_in_use(settings.llamacpp_host, self._port):
            self._port = _free_port()
        log_path = settings.home / "logs" / "llama-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for minimal in (False, True):
            if self._closing:                     # the app is shutting down: do not (re)start
                return False
            if minimal and _port_in_use(settings.llamacpp_host, self._port):
                self._port = _free_port()         # lost a race for the port: take another
            self._log = open(log_path, "ab")      # kept for diagnosis (`triageflow doctor`)
            self._proc = subprocess.Popen(
                self._command(binary, gguf, model_id, minimal), stdout=self._log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **_no_window())
            deadline = time.time() + wait_s
            while time.time() < deadline:
                if self._closing:                      # shut down while the model was loading
                    self._stop()
                    return False
                proc = self._proc
                if proc is None or proc.poll() is not None:   # died: retry with minimal flags
                    break
                if self.server_up():
                    self._active = model_id
                    self.last_error = ""
                    self._write_pidfile(self._proc.pid)
                    return True
                time.sleep(0.5)
            else:
                self.last_error = f"llama-server did not become ready in {int(wait_s)}s."
                self._stop()
                return False
            self._close_log()
        self.last_error = f"llama-server exited at start-up — see {log_path}"
        self._proc = None
        return False

    def _close_log(self) -> None:
        if self._log:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def close(self) -> None:
        """Final shutdown: stop the server and make sure a load still in progress on another
        thread gives up instead of retrying (which would hold the process open on Ctrl+C)."""
        self._closing = True
        self._stop()

    def _pidfile(self) -> Path:
        return settings.home / "run" / "llama-server.pid"

    def _write_pidfile(self, pid: int) -> None:
        try:
            self._pidfile().parent.mkdir(parents=True, exist_ok=True)
            self._pidfile().write_text(f"{pid} {self._port}", encoding="utf-8")
        except OSError:
            pass

    def _read_pidfile(self) -> tuple[int, int] | None:
        """(pid, port) of the llama-server this app last started, if recorded."""
        try:
            pid, port = self._pidfile().read_text(encoding="utf-8").split()
            return int(pid), int(port)
        except (OSError, ValueError):
            return None

    def _recover_previous_server(self) -> None:
        """If the app was killed last time (terminal closed, crash, power button) the
        llama-server it started is still running and holding gigabytes of RAM. Find it by
        our pid file — on whatever port it was using — and take ownership again: reuse it if
        it answers, otherwise stop it. Either way it no longer outlives the app."""
        rec = self._read_pidfile()
        if rec is None:
            return
        pid, port = rec
        if not _pid_is_llama_server(pid):          # gone (or the pid now belongs to
            self._pidfile().unlink(missing_ok=True)   # something else): nothing to recover
            return
        self._port = port
        if self.server_up() and self._is_llama_server():
            self._adopted_pid = pid
            return
        _terminate_pid(pid)                        # ours, but hung or half-loaded: clear it
        self._pidfile().unlink(missing_ok=True)
        self._port = settings.llamacpp_port

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:                                # a server busy generating can ignore a polite
                self._proc.wait(timeout=4)      # stop for a long time — don't hang the exit
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        elif self._adopted_pid:                 # ours from a previous run: stop it too
            _terminate_pid(self._adopted_pid)
        if self._proc or self._adopted_pid:
            self._pidfile().unlink(missing_ok=True)
        self._proc = None
        self._adopted_pid = None
        self._close_log()

    # --- models ---------------------------------------------------------------
    def pull_model(self, model_id: str,
                   on_progress: Callable[[str], None] | None = None) -> bool:
        say = on_progress or (lambda _s: None)
        g = _BY_ID.get(model_id)
        if g is None:
            say(f"unknown model: {model_id}")
            return False
        if self.is_installed(model_id):
            return True
        dest = settings.models_dir / g.filename
        need = int(g.size_gb * 1.1 * (1 << 30))
        try:
            free = shutil.disk_usage(settings.models_dir).free
        except OSError:
            free = need
        if free < need:
            self.last_error = (f"Not enough disk space for {g.label}: needs about "
                               f"{g.size_gb:.1f} GB, {free / (1 << 30):.1f} GB free. "
                               "Pick a smaller model.")
            say(self.last_error)
            return False
        url = f"https://huggingface.co/{g.repo}/resolve/main/{g.filename}?download=true"
        # Resumable + retrying — survives network interruptions (a .part file is resumed).
        if not download_with_resume(url, dest, label=g.label, on_progress=on_progress):
            self.last_error = "Model download failed — check the connection and try again."
            return False
        with open(dest, "rb") as f:        # an error page saved as .gguf must not be "installed"
            magic = f.read(4)
        if magic != b"GGUF":               # (unlink AFTER closing: Windows locks open files)
            dest.unlink(missing_ok=True)
            self.last_error = "Downloaded file is not a valid GGUF model — try again."
            say(self.last_error)
            return False
        return True

    def set_active_model(self, model_id: str) -> bool:
        # Switching model = restart llama-server with the new GGUF (one model per process).
        return self.is_installed(model_id) and self._start_server_for(model_id)

    def resolve_model(self) -> tuple[str | None, bool]:
        if not self.server_up():
            return None, False
        if self._active is None:
            self._active = self._served_model() or settings.llamacpp_default_model
        return self._active, (self._active != settings.llamacpp_default_model)

    def active_model(self) -> str | None:
        return self.resolve_model()[0]

    def status(self) -> BackendStatus:
        installed = self.runtime_installed()
        up = self.server_up()
        model, fb = self.resolve_model()
        hint = ""
        if not installed:
            msg = "llama.cpp runtime not found — the agent runs deterministically meanwhile."
            if self.can_provision():
                hint = ("The llama.cpp runtime will be downloaded automatically (about 25 MB) "
                        "— pick a model below.")
            else:
                hint = ("No prebuilt llama.cpp for this machine — install `llama-server` on "
                        "PATH (or use Ollama), then restart.")
        elif not up:
            msg = "llama.cpp runtime ready — pick a model to download/start below."
        else:
            msg = "Ready"
        if self.last_error and not up:
            hint = (hint + " " if hint else "") + f"Last error: {self.last_error}"
        return BackendStatus(
            backend="llamacpp", runtime_installed=installed, server_up=up, model=model,
            using_fallback=fb, catalog=self.catalog(), message=msg, hint=hint)


def _pid_is_llama_server(pid: int) -> bool:
    """True only if ``pid`` is alive AND is a llama-server — a recorded pid may have been
    reused by an unrelated program since, and that must never be touched."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10, **_no_window()).stdout
        else:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return "llama-server" in out


def _terminate_pid(pid: int, grace_s: float = 4.0) -> None:
    """Stop a llama-server we do not hold a handle to: ask, wait briefly, then force."""
    if not _pid_is_llama_server(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)            # on Windows this terminates immediately
    except OSError:
        return
    if platform.system() == "Windows":
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _pid_is_llama_server(pid):
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _no_window() -> dict:
    """On Windows, don't flash a console window for the child process."""
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flag} if flag else {}
