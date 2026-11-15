"""Tests for the pluggable LLM backend abstraction (no server/network needed)."""

import os

import pytest

from triageflow.agent.llamacpp_backend import PINNED_TAG, LlamaCppBackend, runtime_asset_slug
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


def test_auto_is_llamacpp_when_both_are_installed(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: True)
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: True)
    assert isinstance(get_backend("auto"), LlamaCppBackend)   # the project's runtime


def test_auto_is_llamacpp_even_if_only_ollama_is_installed(monkeypatch):
    """llama.cpp sets itself up and reuses Ollama's model files; it does not depend on the
    Ollama app being open — the cause of 'model not found' when Ollama was not running."""
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "can_provision", lambda self: True)
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: True)
    assert isinstance(get_backend("auto"), LlamaCppBackend)


def test_auto_falls_back_to_ollama_only_where_llamacpp_cannot_run(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "can_provision", lambda self: False)
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: True)
    assert isinstance(get_backend("auto"), OllamaBackend)


def test_auto_uses_llamacpp_when_neither_installed(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "runtime_installed", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: False)
    assert isinstance(get_backend("auto"), LlamaCppBackend)  # it can self-provision


# ---- Ollama: a stopped server must not look like "no models downloaded" ---------------------

def _fake_ollama_store(root, tags):
    for tag in tags:
        name, _, version = tag.partition(":")
        f = root / "manifests" / "registry.ollama.ai" / "library" / name / version
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}", encoding="utf-8")


def test_ollama_models_are_found_on_disk_when_its_server_is_down(tmp_path, monkeypatch):
    from triageflow.agent import ollama_manager as om
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    _fake_ollama_store(tmp_path, ["qwen2.5:7b-instruct", "llama3.2:3b", "llama3.2:latest"])
    monkeypatch.setattr(om, "server_up", lambda *a, **k: False)
    monkeypatch.setattr(om, "list_models", lambda *a, **k: [])       # what a dead server gives
    b = OllamaBackend()
    assert om.models_on_disk() == ["llama3.2:3b", "llama3.2:latest", "qwen2.5:7b-instruct"]
    assert b.is_installed("qwen2.5:7b-instruct") and not b.is_installed("qwen2.5:3b-instruct")
    installed = {o.id: o.installed for o in b.catalog()}
    assert installed == {"qwen2.5:7b-instruct": True, "qwen2.5:3b-instruct": False,
                         "llama3.2:3b": True}
    monkeypatch.setattr(b, "runtime_installed", lambda: True)
    st = b.status()
    assert st.model is None and "already downloaded" in st.hint


def test_ollama_with_no_store_reports_nothing_installed(tmp_path, monkeypatch):
    from triageflow.agent import ollama_manager as om
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "missing"))
    monkeypatch.setattr(om, "server_up", lambda *a, **k: False)
    assert om.models_on_disk() == []
    assert not OllamaBackend().is_installed("qwen2.5:7b-instruct")


def test_ollama_uses_the_server_list_when_it_is_running(tmp_path, monkeypatch):
    from triageflow.agent import ollama_manager as om
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    _fake_ollama_store(tmp_path, ["qwen2.5:7b-instruct"])
    monkeypatch.setattr(om, "server_up", lambda *a, **k: True)
    monkeypatch.setattr(om, "list_models", lambda *a, **k: ["llama3.2:3b"])
    assert OllamaBackend().installed_models() == ["llama3.2:3b"]


def test_ollama_select_fails_cleanly_if_its_server_cannot_start(monkeypatch):
    from triageflow.agent import ollama_manager as om
    monkeypatch.setattr(om, "server_up", lambda *a, **k: False)
    monkeypatch.setattr(om, "start_server", lambda *a, **k: False)
    b, seen = OllamaBackend(), []
    assert b.set_active_model("qwen2.5:7b-instruct") is False
    assert b.pull_model("qwen2.5:7b-instruct", on_progress=seen.append) is False
    assert "open the Ollama app" in seen[0]


def test_llamacpp_does_not_adopt_a_foreign_server_on_its_port(tmp_path, monkeypatch):
    """Something else answering /health on the port is not the model server."""
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(LlamaCppBackend, "server_up", lambda self: True)
    monkeypatch.setattr(LlamaCppBackend, "_is_llama_server", lambda self: False)
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: True)
    assert LlamaCppBackend().ensure_server() is False       # no model of ours to start either


def test_llamacpp_provisioned_bin_under_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    b = LlamaCppBackend()
    assert b._provisioned_bin().name.startswith("llama-server")
    assert tmp_path in b._provisioned_bin().parents


# ---- cross-platform runtime provisioning ----------------------------------------------------

@pytest.mark.parametrize("system,machine,slug", [
    ("Windows", "AMD64", "win-cpu-x64.zip"),
    ("Windows", "ARM64", "win-cpu-arm64.zip"),
    ("Darwin", "x86_64", "macos-x64.tar.gz"),
    ("Darwin", "arm64", "macos-arm64.tar.gz"),
    ("Linux", "x86_64", "ubuntu-x64.tar.gz"),
    ("Linux", "aarch64", "ubuntu-arm64.tar.gz"),
    ("Linux", "riscv64", None),          # no prebuilt -> manual install hint, not a crash
    ("FreeBSD", "x86_64", None),
])
def test_runtime_asset_for_every_supported_platform(system, machine, slug):
    assert runtime_asset_slug(system, machine) == slug


@pytest.mark.parametrize("osname", ["Windows", "Darwin", "Linux"])
def test_status_hint_promises_automatic_setup_on_every_os(osname, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr("platform.system", lambda: osname)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("shutil.which", lambda _b: None)
    st = LlamaCppBackend().status()
    assert "automatically" in st.hint and "brew" not in st.hint.lower()


def test_status_hint_on_unsupported_cpu_is_a_manual_install(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr("platform.machine", lambda: "riscv64")
    monkeypatch.setattr("shutil.which", lambda _b: None)
    b = LlamaCppBackend()
    assert b.can_provision() is False
    assert "llama-server" in b.status().hint
    assert b.ensure_runtime() is False and "No prebuilt" in b.last_error


class _FakeGitHub:
    """Minimal stand-in for httpx.Client against the GitHub releases API."""

    def __init__(self, routes):
        self.routes = routes

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kw):
        for suffix, payload in self.routes.items():
            if url.endswith(suffix):
                if isinstance(payload, Exception):
                    raise payload
                return type("R", (), {"json": lambda self, p=payload: p,
                                      "text": payload if isinstance(payload, str) else ""})()
        raise RuntimeError(f"unexpected URL {url}")


def _asset(name):
    return {"name": name, "browser_download_url": f"https://example.invalid/{name}", "size": 1 << 20}


def test_resolve_asset_follows_the_stable_release_pointer(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    routes = {
        "/releases/latest": {"assets": [{"name": "nightly-tag.txt",
                                         "browser_download_url": "https://x.invalid/nightly-tag.txt"}]},
        "nightly-tag.txt": "b777\n",
        "/releases/tags/b777": {"assets": [
            _asset("llama-b777-bin-ubuntu-vulkan-x64.tar.gz"),     # must NOT be picked
            _asset("llama-b777-bin-ubuntu-x64.tar.gz"),
            _asset("llama-b777-bin-win-cpu-x64.zip")]},
    }
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.httpx.Client", _FakeGitHub(routes))
    name, url, size = LlamaCppBackend()._resolve_asset()
    assert name == "llama-b777-bin-ubuntu-x64.tar.gz" and size == 1 << 20


def test_resolve_asset_falls_back_to_the_pinned_build_when_github_is_unreachable(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.httpx.Client",
                        _FakeGitHub({"/releases/latest": OSError("offline")}))
    name, url, _ = LlamaCppBackend()._resolve_asset()
    assert name == f"llama-{PINNED_TAG}-bin-win-cpu-x64.zip"
    assert url.endswith(f"/download/{PINNED_TAG}/{name}")


def _fake_runtime_archive(path, *, exit_code=0, nested="llama-b1/"):
    """A release-shaped archive whose 'llama-server' is a tiny script (POSIX only)."""
    import io
    import tarfile
    script = f"#!/bin/sh\necho 'version: fake'\nexit {exit_code}\n".encode()
    with tarfile.open(path, "w:gz") as t:
        for name, data, mode in ((nested + "llama-server", script, 0o644),   # +x bit lost
                                 (nested + "libllama.so", b"lib", 0o644)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            t.addfile(info, io.BytesIO(data))


posix_only = pytest.mark.skipif(os.name == "nt", reason="fake runtime is a shell script")


@posix_only
def test_ensure_runtime_downloads_unpacks_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr("shutil.which", lambda _b: None)
    monkeypatch.setattr(LlamaCppBackend, "_resolve_asset",
                        lambda self: ("llama-b1-bin-ubuntu-x64.tar.gz", "https://x.invalid/a", 1))

    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        _fake_runtime_archive(dest)
        return True
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume", fake_download)
    b, seen = LlamaCppBackend(), []
    assert b.ensure_runtime(on_progress=seen.append) is True
    binary = b._provisioned_bin()
    assert binary.is_file() and os.access(binary, os.X_OK)          # +x restored
    assert (binary.parent / "libllama.so").is_file()                # libs stay beside it
    assert b.runtime_installed() and "llama.cpp runtime ready" in seen
    assert not list((tmp_path / "bin").glob("*.tar.gz"))            # archive cleaned up


@posix_only
def test_runtime_that_does_not_run_is_rejected_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _b: None)
    monkeypatch.setattr(LlamaCppBackend, "_resolve_asset",
                        lambda self: ("llama-b1-bin-ubuntu-x64.tar.gz", "https://x.invalid/a", 1))

    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        _fake_runtime_archive(dest, exit_code=127)      # e.g. missing shared library
        return True
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume", fake_download)
    b = LlamaCppBackend()
    assert b.ensure_runtime() is False
    assert not b.runtime_installed() and "does not run on this machine" in b.last_error


@posix_only
def test_damaged_runtime_is_downloaded_again_before_starting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr("shutil.which", lambda _b: None)
    monkeypatch.setattr(LlamaCppBackend, "_resolve_asset",
                        lambda self: ("llama-b1-bin-ubuntu-x64.tar.gz", "https://x.invalid/a", 1))
    calls = []

    def fake_download(url, dest, **kw):
        calls.append(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _fake_runtime_archive(dest)
        return True
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume", fake_download)
    b = LlamaCppBackend()
    assert b.ensure_runtime() and len(calls) == 1
    assert b._heal_runtime() is True and len(calls) == 1          # healthy: left alone
    b._provisioned_bin().write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")   # damage it
    seen = []
    assert b._heal_runtime(on_progress=seen.append) is True and len(calls) == 2
    assert any("damaged" in m for m in seen) and b._runs(b._bin())[0]


def test_failed_runtime_download_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr("shutil.which", lambda _b: None)
    monkeypatch.setattr(LlamaCppBackend, "_resolve_asset",
                        lambda self: ("llama-b1-bin-x.zip", "https://x.invalid/a", 1))
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume",
                        lambda *a, **kw: False)
    b = LlamaCppBackend()
    assert b.ensure_runtime() is False and "download" in b.last_error.lower()


def test_unpack_zip_keeps_layout_and_refuses_path_traversal(tmp_path):
    import zipfile
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("llama-server.exe", "MZ")
        z.writestr("llama.dll", "dll")
    LlamaCppBackend._unpack(good, tmp_path / "out")
    assert (tmp_path / "out" / "llama-server.exe").is_file()
    assert (tmp_path / "out" / "llama.dll").is_file()
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError):
        LlamaCppBackend._unpack(evil, tmp_path / "out2")
    assert not (tmp_path / "escaped.txt").exists()


# ---- models ---------------------------------------------------------------------------------

def test_model_already_downloaded_by_ollama_is_reused(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    root = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "7b-instruct"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:aaa"},
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:bbb"}]}),
        encoding="utf-8")
    (root / "blobs").mkdir()
    (root / "blobs" / "sha256-bbb").write_bytes(b"GGUF....")
    b = LlamaCppBackend()
    assert b.is_installed("qwen2.5-7b-instruct-q4")
    assert b.gguf_path("qwen2.5-7b-instruct-q4") == root / "blobs" / "sha256-bbb"
    assert not b.is_installed("qwen2.5-3b-instruct-q4")
    monkeypatch.setattr(settings, "reuse_ollama_models", False)      # opt-out works
    assert not b.is_installed("qwen2.5-7b-instruct-q4")


def test_downloaded_file_that_is_not_a_gguf_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)

    def fake_download(url, dest, **kw):
        dest.write_bytes(b"<html>rate limited</html>")
        return True
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume", fake_download)
    b = LlamaCppBackend()
    assert b.pull_model("qwen2.5-0.5b-instruct-q4") is False
    assert not b.is_installed("qwen2.5-0.5b-instruct-q4") and "GGUF" in b.last_error


def test_model_download_refused_when_disk_is_too_full(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr("shutil.disk_usage",
                        lambda _p: type("U", (), {"free": 100 << 20})())
    called = []
    monkeypatch.setattr("triageflow.agent.llamacpp_backend.download_with_resume",
                        lambda *a, **kw: called.append(1) or True)
    b = LlamaCppBackend()
    assert b.pull_model("qwen2.5-7b-instruct-q4") is False
    assert not called and "disk space" in b.last_error


def test_unknown_model_id_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    b = LlamaCppBackend()
    assert b.pull_model("no-such-model") is False and not b.is_installed("no-such-model")


# ---- server process lifecycle (a tiny fake llama-server) ------------------------------------

_FAKE_SERVER = """#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
if "--jinja" in a and "{reject_jinja}" == "yes":
    sys.exit(2)                                   # an old build that rejects a newer flag
port = int(a[a.index("--port") + 1])
alias = a[a.index("--alias") + 1] if "--alias" in a else "model.gguf"
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        body = json.dumps({{"status": "ok"}} if self.path == "/health"
                          else {{"data": [{{"id": alias}}]}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _install_fake_server(tmp_path, monkeypatch, reject_jinja="no"):
    import sys

    from triageflow.agent.llamacpp_backend import _free_port
    binary = tmp_path / "llama-server"
    binary.write_text(_FAKE_SERVER.format(python=sys.executable, reject_jinja=reject_jinja),
                      encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "llamacpp_bin", str(binary))
    monkeypatch.setattr(settings, "llamacpp_port", _free_port())
    b = LlamaCppBackend()
    b.gguf_path("qwen2.5-3b-instruct-q4").write_bytes(b"GGUF")
    return b


@posix_only
def test_server_starts_reports_its_model_and_stops(tmp_path, monkeypatch):
    b = _install_fake_server(tmp_path, monkeypatch)
    try:
        assert b.ensure_server() is True
        assert b.resolve_model()[0] == "qwen2.5-3b-instruct-q4"
        assert b._served_model() == "qwen2.5-3b-instruct-q4"       # via --alias
        assert b.status().message == "Ready"
        assert (tmp_path / "logs" / "llama-server.log").is_file()
    finally:
        b._stop()
    assert b.server_up() is False and b.resolve_model() == (None, False)


@posix_only
def test_server_moves_to_a_free_port_when_the_configured_one_is_taken(tmp_path, monkeypatch):
    import socket
    b = _install_fake_server(tmp_path, monkeypatch)
    with socket.socket() as squatter:                # something else owns "our" port
        squatter.bind(("127.0.0.1", settings.llamacpp_port))
        squatter.listen(1)
        try:
            assert b.ensure_server() is True
            assert b._port != settings.llamacpp_port
            assert str(b._port) in b.chat_base_url()
        finally:
            b._stop()


@posix_only
def test_server_retries_with_minimal_flags_if_the_binary_rejects_the_optional_ones(
        tmp_path, monkeypatch):
    b = _install_fake_server(tmp_path, monkeypatch, reject_jinja="yes")
    try:
        assert b.ensure_server() is True and b.server_up()
    finally:
        b._stop()


def test_server_start_without_a_model_is_a_clean_no(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "home", tmp_path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(LlamaCppBackend, "runtime_installed", lambda self: True)
    monkeypatch.setattr(LlamaCppBackend, "server_up", lambda self: False)
    assert LlamaCppBackend().ensure_server() is False


# ---- downloads ------------------------------------------------------------------------------

def test_download_replaces_an_existing_destination(tmp_path, monkeypatch):
    """Path.rename() fails on Windows when the target exists; the downloader must not."""
    from triageflow.agent import downloads

    class _Stream:
        status_code, headers = 200, {"content-length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=0):
            yield b"GGUF"
    monkeypatch.setattr(downloads.httpx, "stream", lambda *a, **kw: _Stream())
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"old")
    assert downloads.download_with_resume("https://x.invalid/m", dest) is True
    assert dest.read_bytes() == b"GGUF" and not dest.with_suffix(".gguf.part").exists()
