# syntax=docker/dockerfile:1
# ── arro-nlp-frontend ─────────────────────────────────────────────────────────
# INTERNAL ONLY. Never publish ports. Reachable only via arro-net.
# Loads sentence-transformers embedding model at startup.
# Entrypoint: `arro-nlp-frontend` CLI → arro_nlp_frontend.main:run → uvicorn

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser/.cache

USER appuser

# CRITICAL: HOST must be 0.0.0.0 — settings.host drives uvicorn bind
# PORT 8000 is EXPOSE only — compose must NOT add `ports:`
ENV HOST=0.0.0.0 \
    PORT=8000 \
    EMBED_BACKEND=local \
    EMBED_MODEL=all-MiniLM-L6-v2 \
    EMBEDDER_MODEL_PATH="" \
    EMBED_SCALE_FACTOR=1.0 \
    HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

# /health returns 200 only after embedder is fully loaded (model weights in RAM)
# start_period=90s: ~90MB HF download on cold boot; ~5s on warm cache volume
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "arro-nlp-frontend"]
