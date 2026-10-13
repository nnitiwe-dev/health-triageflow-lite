"""Runtime configuration. All overridable by environment variables so a facility can
pin the model, the data-residency region, and paths without touching code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _normalize_ollama_host(raw: str) -> str:
    """Turn any OLLAMA_HOST value into a valid client URL.

    Ollama's own CLI accepts a bare host (e.g. ``0.0.0.0`` or ``127.0.0.1:11434``) with no
    scheme, and ``0.0.0.0`` is a *listen* address, not a *connect* one. httpx needs a full
    URL and a real host — so add http:// if missing, map 0.0.0.0/localhost to 127.0.0.1, and
    default the port to 11434. A genuine remote host is left as-is.
    """
    raw = (raw or "").strip()
    if not raw:
        return "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    u = urlparse(raw)
    host = (u.hostname or "127.0.0.1")
    if host in ("0.0.0.0", "localhost", "::", "[::]"):
        host = "127.0.0.1"
    port = u.port or 11434
    return f"{u.scheme or 'http'}://{host}:{port}"


@dataclass
class Settings:
    # Which local open-weights runtime to use. "auto" (default) detects what's already on the
    # machine (Ollama or llama.cpp) and uses it — no redundant setup; only if neither is present
    # does it set one up. Force one with "ollama" or "llamacpp". All keep data on the machine.
    llm_backend: str = field(default_factory=lambda: _env("TRIAGEFLOW_LLM_BACKEND", "auto").lower())

    # Open-weights model served locally (data never leaves the machine).
    model: str = field(default_factory=lambda: _env("TRIAGEFLOW_MODEL", "qwen2.5:7b-instruct"))
    fallback_model: str = field(default_factory=lambda: _env("TRIAGEFLOW_FALLBACK_MODEL", "llama3.2:3b"))
    # Use 127.0.0.1 (not "localhost"): the Ollama app binds IPv4 only, and "localhost" can
    # resolve to IPv6 (::1) first. Normalised so a bare/0.0.0.0 OLLAMA_HOST still works.
    ollama_host: str = field(default_factory=lambda: _normalize_ollama_host(
        _env("OLLAMA_HOST", "http://127.0.0.1:11434")))

    # llama.cpp backend settings (only used when llm_backend == "llamacpp").
    llamacpp_bin: str = field(default_factory=lambda: _env("TRIAGEFLOW_LLAMACPP_BIN", "llama-server"))
    llamacpp_host: str = field(default_factory=lambda: _env("TRIAGEFLOW_LLAMACPP_HOST", "127.0.0.1"))
    llamacpp_port: int = field(default_factory=lambda: int(_env("TRIAGEFLOW_LLAMACPP_PORT", "8080")))
    llamacpp_ctx: int = field(default_factory=lambda: int(_env("TRIAGEFLOW_LLAMACPP_CTX", "4096")))
    llamacpp_ngl: int = field(default_factory=lambda: int(_env("TRIAGEFLOW_LLAMACPP_NGL", "999")))
    llamacpp_default_model: str = field(default_factory=lambda: _env("TRIAGEFLOW_LLAMACPP_MODEL", "qwen2.5-7b-instruct-q4"))

    # Data residency guard — recorded in the audit log and shown in the UI.
    data_region: str = field(default_factory=lambda: _env("TRIAGEFLOW_DATA_REGION", "local-only"))

    # Optional FHIR/EMR integration (OFF unless a base URL is set). Point this at a facility's
    # IN-COUNTRY FHIR R4 server for a pilot, or a public sandbox for a synthetic-data demo.
    # When empty, the app runs fully local with no network dependency (data residency default).
    fhir_base_url: str = field(default_factory=lambda: _env("TRIAGEFLOW_FHIR_BASE_URL", "").rstrip("/"))

    # Paths
    home: Path = field(default_factory=lambda: Path(_env("TRIAGEFLOW_HOME", str(Path.home() / ".triageflow"))))
    protocols_dir: Path = field(default_factory=lambda: Path(
        _env("TRIAGEFLOW_PROTOCOLS", str(Path(__file__).resolve().parents[3] / "protocols"))))
    # Where llama.cpp GGUF files are stored/downloaded.
    models_dir: Path = field(default_factory=lambda: Path(
        _env("TRIAGEFLOW_MODELS_DIR", str(Path.home() / ".triageflow" / "models"))))

    # Web
    host: str = field(default_factory=lambda: _env("TRIAGEFLOW_WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("TRIAGEFLOW_WEB_PORT", "8000")))

    def __post_init__(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)


settings = Settings()
