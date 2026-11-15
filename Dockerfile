# syntax=docker/dockerfile:1
# The Kindling API server. Ollama stays on the host: Docker on macOS has no GPU / Metal access.
#   docker compose --profile app up -d --build        (see docs/deploy.md)
#
# The project lives at /app and is installed editable (uv's default), so `kindling db upgrade`
# finds alembic.ini and migrations/ next to src/, exactly as in a checkout.

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /app
# Dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY README.md alembic.ini models.example.toml ./
COPY migrations ./migrations
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin kindling
WORKDIR /app
# Code and venv stay root-owned (read-only to the server); only these folders are writable.
COPY --from=build /app /app
COPY --chmod=755 deploy/entrypoint.sh deploy/healthcheck.py /app/deploy/
RUN mkdir -p /app/workspace /app/evals/results && chown kindling:kindling /app/workspace /app/evals/results
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KINDLING_HOST=0.0.0.0 \
    KINDLING_PORT=8000
USER kindling
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["python", "/app/deploy/healthcheck.py"]
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
