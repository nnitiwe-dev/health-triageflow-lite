"""Real end-to-end check of the llama.cpp runtime on THIS machine (used by CI on every OS).

    python evals/runtime_smoke.py

It runs the app's own code path with nothing mocked: provision the official llama-server for
this OS/CPU (if not already present), start it with a 1 MB test model, wait for it to become
healthy, generate a few tokens, and stop it. Exits non-zero with the reason if any step fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.environ.setdefault("TRIAGEFLOW_HOME", tempfile.mkdtemp(prefix="triageflow-smoke-"))
os.environ["TRIAGEFLOW_MODELS_DIR"] = tempfile.mkdtemp(prefix="triageflow-smoke-models-")
os.environ["TRIAGEFLOW_REUSE_OLLAMA_MODELS"] = "0"

import httpx  # noqa: E402

from triageflow.agent.downloads import download_with_resume  # noqa: E402
from triageflow.agent.llamacpp_backend import LlamaCppBackend  # noqa: E402

TINY = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
MODEL_ID = "qwen2.5-0.5b-instruct-q4"      # the tiny file stands in for a catalog entry


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    say = lambda m: print(f"  … {m}", flush=True)  # noqa: E731
    b = LlamaCppBackend()
    print("1. runtime", flush=True)
    if not b.ensure_runtime(on_progress=say):
        print(f"FAILED: {b.last_error}")
        return 1
    print(f"   binary: {b._bin()}")

    print("2. test model", flush=True)
    gguf = b.gguf_path(MODEL_ID)
    if not download_with_resume(TINY, gguf, label="1 MB test model", on_progress=say):
        print("FAILED: could not download the test model")
        return 1

    print("3. start server", flush=True)
    try:
        if not b.ensure_server(on_progress=say):
            log = Path(os.environ["TRIAGEFLOW_HOME"]) / "logs" / "llama-server.log"
            tail = log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else ""
            print(f"FAILED: {b.last_error}\n{tail}")
            return 1
        print(f"   up on port {b._port}; serving: {b._served_model()}")
        if b._served_model() != MODEL_ID:
            print("FAILED: the server does not report the model we started")
            return 1

        print("4. generate", flush=True)
        r = httpx.post(b._base() + "/completion", timeout=120, trust_env=False,
                       json={"prompt": "Once upon a time", "n_predict": 8})
        text = r.json().get("content", "") if r.status_code == 200 else ""
        if not text.strip():
            print(f"FAILED: no text generated (HTTP {r.status_code}: {r.text[:200]})")
            return 1
        print(f"   generated: {text.strip()!r}")
    finally:
        b._stop()
    if b.server_up():
        print("FAILED: the server is still running after stop")
        return 1
    print("5. stopped cleanly\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
