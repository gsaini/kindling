#!/bin/sh
# Container entrypoint: bring the database schema up to date, then serve.
# Runs from /app, where alembic.ini and migrations/ live (see the Dockerfile).
set -eu
cd /app

if [ -d /app/models.toml ]; then
    # Docker made an empty directory because ./models.toml doesn't exist on the host.
    echo "entrypoint: no models.toml on the host; serving the single model from KINDLING_MODEL" >&2
fi

if [ "${KINDLING_STORE:-postgres}" = "postgres" ]; then
    echo "entrypoint: applying database migrations" >&2
    kindling db upgrade
fi

# exec, so the server is PID 1 and gets SIGTERM directly for a graceful shutdown.
exec kindling serve --host 0.0.0.0 "$@"
