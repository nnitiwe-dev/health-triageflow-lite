"""Pluggable local-LLM backends behind one OpenAI-compatible interface.

Best practice: the agent never talks to a specific runtime — it talks to an OpenAI-compatible
``/v1/chat/completions`` endpoint. Ollama and llama.cpp's ``llama-server`` (and vLLM, TGI, …)
all speak it, so switching backend is a config change, not a code change. Each backend also
manages its own server lifecycle + model download and exposes a small **catalog** so the UI
can let a nurse switch to a lighter model if the standard one is too heavy for their box.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ModelOption:
    id: str              # backend id (an Ollama tag, or a llama.cpp catalog key)
    label: str           # human label for the UI
    tier: str            # "standard" | "small"
    size_gb: float
    installed: bool = False


@dataclass
class BackendStatus:
    backend: str
    runtime_installed: bool        # the runtime binary/app is present
    server_up: bool
    model: str | None           # active, ready model (or None)
    using_fallback: bool
    catalog: list[ModelOption] = field(default_factory=list)
    message: str = ""
    hint: str = ""                 # install guidance when the runtime is missing


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def chat_base_url(self) -> str: ...
    def runtime_installed(self) -> bool: ...
    def server_up(self) -> bool: ...
    def ensure_server(self, wait_s: float = 0.0,
                      on_progress: Callable[[str], None] | None = None) -> bool: ...
    def installed_models(self) -> list[str]: ...
    def catalog(self) -> list[ModelOption]: ...
    def is_installed(self, model_id: str) -> bool: ...
    def resolve_model(self) -> tuple[str | None, bool]: ...
    def pull_model(self, model_id: str,
                   on_progress: Callable[[str], None] | None = None) -> bool: ...
    def set_active_model(self, model_id: str) -> bool | None: ...
    def active_model(self) -> str | None: ...
    def status(self) -> BackendStatus: ...


def make_chat_model(base_url: str, model: str, *, temperature: float = 0.0,
                    max_tokens: int = 160, timeout: float = 120.0):
    """One chat client for every backend: an OpenAI-compatible client pointed at base_url.
    (Local servers need no real key — a placeholder keeps the SDK happy.)"""
    from langchain_openai import ChatOpenAI
    # max_retries=0: a local server that is slow or down should fail fast into the
    # deterministic fallback, not be retried while a nurse waits.
    return ChatOpenAI(model=model, base_url=base_url.rstrip("/"), api_key="sk-no-auth-local",
                      temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                      max_retries=0)


def get_backend(backend: str | None = None) -> LLMBackend:
    """Resolve the backend. ``auto`` = **llama.cpp**, the project's runtime: it needs no
    daemon or GUI app to be running, sets up its own ~25 MB prebuilt binary on every OS, and
    loads model files Ollama has already downloaded — so nothing large is fetched twice.
    Ollama is chosen automatically only where llama.cpp cannot run (no prebuilt binary for
    the CPU and none on PATH), or explicitly with ``TRIAGEFLOW_LLM_BACKEND=ollama``."""
    from ..core.config import settings
    from .llamacpp_backend import LlamaCppBackend
    from .ollama_backend import OllamaBackend

    name = (backend or settings.llm_backend or "auto").lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "llamacpp":
        return LlamaCppBackend()
    llama = LlamaCppBackend()
    if llama.runtime_installed() or llama.can_provision():
        return llama
    ollama = OllamaBackend()
    return ollama if ollama.runtime_installed() else llama
