"""OLLAMA_HOST normalization — a bare or 0.0.0.0 host must become a valid client URL."""

import pytest

from triageflow.core.config import _normalize_ollama_host as norm


@pytest.mark.parametrize("raw,expected", [
    ("", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),            # listen-all → connect to loopback
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),    # bare host:port gets a scheme
    ("localhost", "http://127.0.0.1:11434"),          # avoid IPv6 localhost pitfall
    ("http://localhost:11434", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("192.168.1.5:11434", "http://192.168.1.5:11434"),  # genuine remote preserved
    ("https://fhir.example.org:443", "https://fhir.example.org:443"),  # scheme/port preserved
])
def test_normalize_ollama_host(raw, expected):
    assert norm(raw) == expected
