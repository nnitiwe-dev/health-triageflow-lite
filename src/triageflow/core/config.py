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
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))

    # Data residency guard — recorded in the audit log and shown in the UI.
    data_region: str = field(default_factory=lambda: _env("TRIAGEFLOW_DATA_REGION", "local-only"))

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
