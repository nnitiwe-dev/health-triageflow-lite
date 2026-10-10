"""Runtime configuration. All overridable by environment variables so a facility can
pin the model, the data-residency region, and paths without touching code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Settings:
    # Open-weights model served locally by Ollama (data never leaves the machine).
    model: str = field(default_factory=lambda: _env("TRIAGEFLOW_MODEL", "qwen2.5:7b-instruct"))
    fallback_model: str = field(default_factory=lambda: _env("TRIAGEFLOW_FALLBACK_MODEL", "llama3.2:3b"))
    # Use 127.0.0.1 (not "localhost"): the Ollama app binds IPv4 only, and "localhost" can
    # resolve to IPv6 (::1) first, which makes the server look unreachable.
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))

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

    # Web
    host: str = field(default_factory=lambda: _env("TRIAGEFLOW_WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("TRIAGEFLOW_WEB_PORT", "8000")))

    def __post_init__(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)


settings = Settings()
