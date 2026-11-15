"""Ollama backend — wraps the low-level ollama_manager behind the LLMBackend interface.

Ollama serves many models at once, so switching the active model is just selecting a tag
(and pulling it first if needed). Chat goes through Ollama's OpenAI-compatible ``/v1``.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.config import settings
from . import ollama_manager as om
from .llm_backend import BackendStatus, ModelOption

# (tag, label, tier, size_gb)
_CATALOG = [
    ("qwen2.5:7b-instruct", "Qwen2.5 7B · standard", "standard", 4.7),
    ("qwen2.5:3b-instruct", "Qwen2.5 3B · small / faster", "small", 1.9),
    ("llama3.2:3b", "Llama 3.2 3B · small / faster", "small", 2.0),
]


class OllamaBackend:
    name = "ollama"

    def __init__(self) -> None:
        self._active: str | None = None

    def chat_base_url(self) -> str:
        return settings.ollama_host.rstrip("/") + "/v1"

    def runtime_installed(self) -> bool:
        return om._binary() is not None

    def server_up(self) -> bool:
        return om.server_up()

    def ensure_server(self, wait_s: float = 0.0,
                      on_progress: Callable[[str], None] | None = None) -> bool:
        return True if om.server_up() else om.start_server(wait_s=wait_s)

    def installed_models(self) -> list[str]:
        """What is downloaded: ask the server; if it is not running, read Ollama's own
        manifests from disk — a stopped server must not look like 'nothing downloaded'."""
        return om.list_models() if om.server_up() else om.models_on_disk()

    def is_installed(self, model_id: str) -> bool:
        # Exact tag match for the catalog (the loose base-match in ollama_manager would wrongly
        # call qwen2.5:3b-instruct "installed" just because qwen2.5:7b-instruct is present).
        avail = self.installed_models()
        return model_id in avail or f"{model_id}:latest" in avail

    def catalog(self) -> list[ModelOption]:
        avail = self.installed_models()
        return [ModelOption(mid, label, tier, size,
                            mid in avail or f"{mid}:latest" in avail)
                for (mid, label, tier, size) in _CATALOG]

    def resolve_model(self) -> tuple[str | None, bool]:
        if self._active and self.is_installed(self._active):
            return self._active, (self._active == settings.fallback_model)
        return om.resolve_model()

    def pull_model(self, model_id: str,
                   on_progress: Callable[[str], None] | None = None) -> bool:
        if not om.server_up() and not om.start_server(wait_s=45):
            if on_progress:
                on_progress("Ollama is not running and could not be started — open the "
                            "Ollama app and try again.")
            return False
        if self.is_installed(model_id):       # already on disk: nothing to download
            self._active = model_id
            return True
        ok = om.pull_model(model_id, on_progress=on_progress)
        if ok:
            self._active = model_id
        return ok

    def set_active_model(self, model_id: str) -> bool:
        # Ollama serves many models; selecting a tag is enough — but its server must be up.
        if not om.server_up() and not om.start_server(wait_s=45):
            return False
        self._active = model_id
        return True

    def active_model(self) -> str | None:
        return self.resolve_model()[0]

    def status(self) -> BackendStatus:
        up = om.server_up()
        model, fb = self.resolve_model()
        catalog = self.catalog()
        have = [o for o in catalog if o.installed]
        if model:
            message, hint = "Ready", ""
        elif not self.runtime_installed():
            message, hint = "Ollama is not installed", "Install Ollama from https://ollama.com/download"
        elif not up:
            message = "Ollama is installed but its server is not running yet"
            hint = ((f"Your model ({have[0].label}) is already downloaded — waiting for Ollama "
                     "to start. If it does not, open the Ollama app.") if have else
                    "Waiting for Ollama to start. If it does not, open the Ollama app.")
        elif not have:
            message, hint = "Ollama is running but no model is downloaded", "Pick a model to download."
        else:
            message = "Ollama is running; the model is not loaded yet"
            hint = "Pick the downloaded model from the model menu to use it."
        return BackendStatus(
            backend="ollama", runtime_installed=self.runtime_installed(), server_up=up,
            model=model, using_fallback=fb, catalog=catalog, message=message, hint=hint)
