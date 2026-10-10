"""Automatic Ollama + open-weights model management.

Responsibilities (so the operator never has to think about it):
  - detect the `ollama` binary
  - make sure the Ollama server is running (start it if not)
  - make sure the configured model is present (pull it if not, with progress)
  - expose a health check and the resolved model name (primary or fallback)

Everything runs locally; patient data never leaves the machine. A large model pull is
the one heavy step — callers decide whether to trigger it (the CLI asks/【--yes】).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..core.config import settings


@dataclass
class ModelStatus:
    ollama_installed: bool
    server_up: bool
    model: str | None          # resolved model that is actually available
    available_models: list[str]
    using_fallback: bool = False
    message: str = ""


def _binary() -> str | None:
    return shutil.which("ollama")


def server_up(host: str | None = None, timeout: float = 2.0) -> bool:
    host = host or settings.ollama_host
    try:
        r = httpx.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_models(host: str | None = None) -> list[str]:
    host = host or settings.ollama_host
    try:
        r = httpx.get(f"{host}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _model_present(name: str, available: list[str]) -> bool:
    # Ollama tags may append :latest; match on prefix before the tag too.
    base = name.split(":")[0]
    return any(m == name or m.split(":")[0] == base for m in available)


def resolve_model() -> tuple[str | None, bool]:
    """Lightweight check for a ready model WITHOUT starting the server (so it never pops up
    the Ollama app or blocks). Returns (model_name_or_None, using_fallback). Safe to call often."""
    if not server_up():
        return None, False
    available = list_models()
    if _model_present(settings.model, available):
        return settings.model, False
    if _model_present(settings.fallback_model, available):
        return settings.fallback_model, True
    return None, False


def start_server(wait_s: float = 25.0) -> bool:
    """Make the Ollama server reachable, if it is not already.

    On macOS we prefer launching the Ollama.app (it manages a persistent server and owns
    the port), which avoids a stray `ollama serve` fighting the app over port 11434. We
    fall back to `ollama serve` elsewhere, or if the app is not installed.
    """
    if server_up():
        return True
    started = False
    app_path = Path("/Applications/Ollama.app")
    if sys.platform == "darwin" and app_path.exists():
        # -g = launch in the background, so it never steals focus / pops the window forward.
        subprocess.run(["open", "-g", "-a", "Ollama"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
        started = True
    if not started:
        if not _binary():
            return False
        subprocess.Popen([_binary(), "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_up():
            return True
        time.sleep(0.5)
    return False


def _models_dir() -> Path:
    """Where Ollama stores manifests and blobs (partial downloads live here)."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def clear_partial_downloads(on_progress: Callable[[str], None] | None = None) -> int:
    """Delete partial/temporary blob files left by a failed or interrupted pull.

    Ollama writes incomplete blobs as `sha256-...-partial` (plus `-partial-*` chunk
    files) under <models>/blobs. Removing them forces a clean re-download; completed
    blobs (which other models may share) are left untouched.
    """
    blobs = _models_dir() / "blobs"
    removed = 0
    if not blobs.is_dir():
        return 0
    for f in blobs.iterdir():
        if "partial" in f.name:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if on_progress and removed:
        on_progress(f"cleared {removed} partial download file(s)")
    return removed


def remove_model(name: str) -> bool:
    """Remove a (possibly broken) model via `ollama rm`, so the next pull starts clean."""
    if not _binary():
        return False
    try:
        subprocess.run([_binary(), "rm", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
        return True
    except Exception:
        return False


def _pull_once(name: str, on_progress: Callable[[str], None] | None = None) -> bool:
    """One streaming pull attempt via the Ollama HTTP API."""
    host = settings.ollama_host
    try:
        with httpx.stream("POST", f"{host}/api/pull",
                          json={"name": name, "stream": True}, timeout=None) as resp:
            if resp.status_code != 200:
                return False
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = msg.get("status", "")
                if on_progress:
                    if msg.get("total") and msg.get("completed") is not None:
                        pct = 100 * msg["completed"] / msg["total"]
                        on_progress(f"{status} {pct:.0f}%")
                    elif status:
                        on_progress(status)
                if msg.get("error"):
                    if on_progress:
                        on_progress(f"error: {msg['error']}")
                    return False
        return _model_present(name, list_models())
    except Exception as e:  # network drop, timeout, server restart…
        if on_progress:
            on_progress(f"pull interrupted: {type(e).__name__}")
        return False


def pull_model(name: str, on_progress: Callable[[str], None] | None = None,
               retries: int = 2, clean_on_fail: bool = True) -> bool:
    """Pull a model, self-healing on failure.

    On a failed/interrupted attempt it clears partial download files (and, on the final
    retry, removes the half-written model) before trying again, so a broken download can
    never wedge the app — it just re-downloads cleanly.
    """
    attempts = retries + 1
    for i in range(attempts):
        if i > 0 and on_progress:
            on_progress(f"retry {i}/{retries} for {name}…")
        if _pull_once(name, on_progress):
            return True
        if clean_on_fail:
            clear_partial_downloads(on_progress)
            if i == attempts - 2:  # before the last attempt, also drop the broken manifest
                remove_model(name)
        time.sleep(1.5 * (i + 1))
    return False


def ensure_ready(pull: bool = False, force_clean: bool = False,
                 on_progress: Callable[[str], None] | None = None) -> ModelStatus:
    """Bring the stack to a ready state. If `pull` is False, report what is missing
    rather than downloading multi-GB weights without consent.

    `force_clean` clears any partial downloads (and drops a broken primary model) before
    pulling — the manual escape hatch when a previous download failed or corrupted.
    """
    log = on_progress or (lambda s: None)
    installed = _binary() is not None
    if not installed:
        return ModelStatus(False, False, None, [], message=(
            "Ollama is not installed. Install it from https://ollama.com/download, "
            "then re-run. (A large model download cannot proceed without it.)"))

    up = server_up()
    if not up:
        log("starting ollama server…")
        up = start_server()
    if not up:
        return ModelStatus(True, False, None, [], message="Could not start the Ollama server.")

    if force_clean:
        log("force-clean requested: clearing partial downloads…")
        clear_partial_downloads(log)
        remove_model(settings.model)

    available = list_models()
    if not force_clean:
        for candidate, is_fallback in ((settings.model, False), (settings.fallback_model, True)):
            if _model_present(candidate, available):
                return ModelStatus(True, True, candidate, available, using_fallback=is_fallback,
                                   message=f"Ready with {candidate}.")

    # Neither present.
    if not pull:
        return ModelStatus(True, True, None, available, message=(
            f"No local model found. Run with --pull to download '{settings.model}' "
            f"(~5 GB) or set TRIAGEFLOW_MODEL to one you already have: {available}"))

    log(f"pulling {settings.model} (first run, ~5 GB)…")
    if pull_model(settings.model, on_progress):
        return ModelStatus(True, True, settings.model, list_models(),
                           message=f"Ready with {settings.model}.")
    log(f"primary pull failed; trying fallback {settings.fallback_model}…")
    if pull_model(settings.fallback_model, on_progress):
        return ModelStatus(True, True, settings.fallback_model, list_models(),
                           using_fallback=True, message=f"Ready with {settings.fallback_model}.")
    return ModelStatus(True, True, None, list_models(),
                       message="Model pull failed. Check connectivity / disk space.")
