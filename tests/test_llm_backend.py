"""Tests for the pluggable LLM backend abstraction (no server/network needed)."""

import pytest

from triageflow.agent.llamacpp_backend import LlamaCppBackend
from triageflow.agent.llm_backend import ModelOption, get_backend, make_chat_model
from triageflow.agent.ollama_backend import OllamaBackend
from triageflow.core.config import settings


def test_get_backend_selects_by_name():
    # Explicit names are deterministic. ("auto"/None is environment-dependent — it picks an
    # installed runtime — so it's covered separately by the monkeypatched tests below.)
    assert isinstance(get_backend("ollama"), OllamaBackend)
    assert isinstance(get_backend("llamacpp"), LlamaCppBackend)


def test_make_chat_model_is_openai_compatible():
    llm = make_chat_model("http://127.0.0.1:11434/v1", "qwen2.5:7b-instruct", max_tokens=42)
    # langchain-openai exposes the resolved base URL + model
    assert str(llm.openai_api_base).rstrip("/").endswith("/v1")
    assert llm.model_name == "qwen2.5:7b-instruct"
    assert llm.max_tokens == 42


def test_ollama_backend_catalog_shape():
    cat = OllamaBackend().catalog()
    assert cat and all(isinstance(o, ModelOption) for o in cat)
    ids = {o.id for o in cat}
    assert "qwen2.5:7b-instruct" in ids
    assert any(o.tier == "small" for o in cat)  # a lighter option is offered


def test_ollama_chat_base_url_has_v1():
    assert OllamaBackend().chat_base_url().endswith("/v1")


def test_llamacpp_catalog_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    b = LlamaCppBackend()
    cat = b.catalog()
    assert {o.id for o in cat} >= {"qwen2.5-7b-instruct-q4", "qwen2.5-3b-instruct-q4"}
    # nothing downloaded yet
    assert all(o.installed is False for o in cat)
    assert b.is_installed("qwen2.5-7b-instruct-q4") is False
    # path points into the models dir with a .gguf name
    p = b.gguf_path("qwen2.5-3b-instruct-q4")
    assert p.parent == tmp_path and p.name.endswith(".gguf")
    # a dropped-in file flips the installed flag
    p.write_bytes(b"gguf")
    assert b.is_installed("qwen2.5-3b-instruct-q4") is True


def test_llamacpp_status_without_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _b: None)  # pretend llama-server missing
    st = LlamaCppBackend().status()
    assert st.backend == "llamacpp"
    assert st.runtime_installed is False
    assert "llama.cpp" in st.hint.lower() or "llama-server" in st.message.lower()


@pytest.mark.parametrize("tier_present", ["standard", "small"])
def test_both_backends_offer_tiers(tier_present, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    for b in (OllamaBackend(), LlamaCppBackend()):
        assert any(o.tier == tier_present for o in b.catalog())


def test_auto_prefers_installed_ollama(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: True)
    assert isinstance(get_backend("auto"), OllamaBackend)


def test_auto_falls_back_to_llamacpp_when_no_ollama(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: True)
    assert isinstance(get_backend("auto"), LlamaCppBackend)


def test_auto_uses_llamacpp_when_neither_installed(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: False)
    assert isinstance(get_backend("auto"), LlamaCppBackend)  # it can self-provision


def test_llamacpp_provisioned_bin_under_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    b = LlamaCppBackend()
    assert b._provisioned_bin().name.startswith("llama-server")
    assert tmp_path in b._provisioned_bin().parents


@pytest.mark.parametrize("osname,needle", [
    ("Windows", "automatically"), ("Darwin", "brew install"), ("Linux", "PATH")])
def test_llamacpp_status_hint_per_os(osname, needle, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("platform.system", lambda: osname)
    monkeypatch.setattr("shutil.which", lambda _b: None)
    st = LlamaCppBackend().status()
    assert needle.lower() in st.hint.lower()
