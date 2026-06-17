# syntax=docker/dockerfile:1
# ── arro-nlp-frontend ─────────────────────────────────────────────────────────
# INTERNAL SERVICE — never publish ports to host.
# Reachable only via Docker internal network (arro-net) from arro-cve-search.
#
# Startup sequence:
#   1. uv installs deps from frozen lockfile
#   2. `arro-nlp-frontend` CLI starts uvicorn (factory mode)
#   3. lifespan loads Embedder (sentence-transformers or OpenAI)
#   4. GET /health returns 200 → container marked healthy
#
# Cold boot: ~90s (HF model download ~90MB)
# Warm boot with volume: ~5s (model already cached)

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl needed for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# uv — pinned via digest in CI, latest here for simplicity
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Deps layer (cached until pyproject.toml or uv.lock changes) ───────────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── App source ────────────────────────────────────────────────────────────────
# README.md needed by hatchling during build (validate_fields checks existence)
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Remove CUDA packages on non-GPU builds
RUN /app/.venv/bin/pip uninstall -y nvidia-cusparselt-cu13 2>/dev/null || true

# ── Runtime user + dirs ───────────────────────────────────────────────────────
# /app/data  → SQLite document store (store_db_path default: ./data/documents.sqlite)
# HF_HOME    → sentence-transformers model cache (mount as named volume in compose)
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /data /app/data /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /data /app /home/appuser/.cache

USER appuser

# ── Runtime env defaults ──────────────────────────────────────────────────────
# HOST=0.0.0.0  REQUIRED — uvicorn must bind all interfaces inside Docker.
#               If 127.0.0.1, other containers cannot reach this service.
# PORT=8000     internal only, never published via `ports:` in compose.
# HF_HOME       must match the volume mount path in docker-compose.yml.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    EMBED_BACKEND=local \
    EMBED_MODEL=all-MiniLM-L6-v2 \
    EMBEDDER_MODEL_PATH="" \
    EMBED_SCALE_FACTOR=1.0 \
    OPENAI_API_KEY="" \
    ARRO_SERVER_URL="http://arro-server:8000" \
    HF_HOME=/home/appuser/.cache/huggingface

# ── Expose (internal only) ────────────────────────────────────────────────────
# EXPOSE is documentation. The compose file enforces no `ports:` for this service.
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# /health → 200 only after embedder weights are loaded.
# start_period=90s covers cold HF download.
# With warm hf-model-cache volume, healthy in ~10s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Uses the installed CLI script from pyproject.toml [project.scripts].
# Equivalent to: uvicorn arro_nlp_frontend.main:create_app --factory
CMD ["uv", "run", "arro-nlp-frontend"]
