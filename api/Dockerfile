# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install third-party dependencies first for layer caching.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra api --no-install-project

# Install the local nameage library and api package non-editably into .venv.
COPY src ./src
COPY api ./api
RUN uv sync --frozen --no-dev --extra api --no-editable


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NAMEAGE_DATA_DIR="/data" \
    MPLCONFIGDIR="/tmp/matplotlib"

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "100", "--timeout-keep-alive", "5"]
