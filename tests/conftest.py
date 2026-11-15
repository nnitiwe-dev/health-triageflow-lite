"""Hermetic test environment — set BEFORE triageflow is imported (settings read env once).

Nothing a test does may touch the real ``~/.triageflow`` (audit log, provisioned binaries,
models) or pick up models from a real Ollama install on the developer's machine.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="triageflow-test-")
os.environ["TRIAGEFLOW_HOME"] = _TMP
os.environ["TRIAGEFLOW_MODELS_DIR"] = os.path.join(_TMP, "models")
os.environ["OLLAMA_MODELS"] = os.path.join(_TMP, "ollama-models")
os.environ.pop("TRIAGEFLOW_LLM_BACKEND", None)
os.environ.pop("TRIAGEFLOW_FHIR_BASE_URL", None)
