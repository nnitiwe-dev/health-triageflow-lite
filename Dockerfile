# syntax=docker/dockerfile:1
#
# Container image for a public cloud demo (built for linux/amd64, e.g. AWS App Runner).
#
# The model and the llama.cpp runtime are BAKED IN — not downloaded at start-up — because a
# cloud instance has no persistent disk and may restart or scale at any time; re-downloading
# ~5 GB on every cold start would be slow and fragile. They live in their own build stage so a
# change to the application code never re-downloads them. See docs/DEPLOY_AWS.md.
#
# Override the model at build time, e.g.:
#   docker build --build-arg MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf \
#                --build-arg MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF \
#                --build-arg MODEL_ID=qwen2.5-3b-instruct-q4 .

ARG MODEL_REPO=bartowski/Qwen2.5-7B-Instruct-GGUF
ARG MODEL_FILE=Qwen2.5-7B-Instruct-Q4_K_M.gguf
ARG MODEL_ID=qwen2.5-7b-instruct-q4
ARG LLAMA_BUILD=b12359

# ---- assets: model weights + llama.cpp server, in a cache-stable layer ----------------------
FROM debian:bookworm-slim AS assets
ARG MODEL_REPO
ARG MODEL_FILE
ARG LLAMA_BUILD
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /assets
# Official prebuilt llama-server for linux x64 (pinned), unpacked in the layout the app expects.
RUN mkdir -p bin/llamacpp \
    && curl -fSL -o llama.tgz \
       "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" \
    && tar -xzf llama.tgz -C bin/llamacpp \
    && rm llama.tgz \
    && find bin/llamacpp -name llama-server -exec chmod 0755 {} +
# The GGUF model, verified to really be a GGUF file (not an error page).
RUN mkdir -p models \
    && curl -fSL -o "models/${MODEL_FILE}" \
       "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}?download=true" \
    && test "$(head -c 4 "models/${MODEL_FILE}")" = "GGUF"

# ---- deps: the Python virtualenv ------------------------------------------------------------
FROM python:3.12-slim-bookworm AS deps
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime --------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
# Node is for the borrowed filesystem MCP (npx). The app falls back to reading the protocol
# file directly if it is absent, so this is for a faithful two-MCP-server demo, not required.
# libgomp1: the OpenMP runtime llama-server links against (not in the slim base image).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 nodejs npm \
    && npm install -g @modelcontextprotocol/server-filesystem \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/* /root/.npm

ARG MODEL_ID
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PATH=/app/.venv/bin:$PATH \
    HOME=/home/app \
    TRIAGEFLOW_HOME=/home/app/.triageflow \
    TRIAGEFLOW_MODELS_DIR=/home/app/.triageflow/models \
    TRIAGEFLOW_LLM_BACKEND=llamacpp \
    TRIAGEFLOW_LLAMACPP_MODEL=${MODEL_ID} \
    TRIAGEFLOW_REUSE_OLLAMA_MODELS=0 \
    TRIAGEFLOW_WEB_HOST=0.0.0.0 \
    TRIAGEFLOW_DATA_REGION="cloud demo · synthetic data only"

RUN useradd -m -u 10001 app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY src ./src
COPY protocols ./protocols
COPY pyproject.toml uv.lock README.md ./
COPY --from=assets --chown=app:app /assets/bin    /home/app/.triageflow/bin
COPY --from=assets --chown=app:app /assets/models /home/app/.triageflow/models

USER app
EXPOSE 8000
# The web server answers /api/status as soon as it is up; the model loads in the background.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=10 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4).status==200 else 1)"
# Honour $PORT if the platform sets one (App Runner uses the port configured on the service).
CMD ["sh", "-c", "exec python -m triageflow.cli serve --host 0.0.0.0 --port ${PORT:-8000}"]
