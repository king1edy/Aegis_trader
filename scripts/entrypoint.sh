#!/bin/bash
set -e

echo "=== Aegis Trading — Container Startup ==="

# ── Run Alembic migrations ────────────────────────────────────────────────
# The working directory is /app/src (set by Dockerfile), but alembic.ini
# lives one level up at /app.  We run from /app so the `prepend_sys_path`
# directive in alembic.ini adds both "." and "src" to sys.path.
echo "Running database migrations..."
cd /app

# Wait for Postgres to be ready (up to 30 seconds)
for i in $(seq 1 30); do
    if python -c "
import sys, asyncio, sqlalchemy
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
async def check():
    e = create_async_engine('${DATABASE_URL}'.replace('postgresql://', 'postgresql+asyncpg://'), pool_pre_ping=True)
    async with e.connect() as c:
        await c.execute(text('SELECT 1'))
    await e.dispose()
asyncio.run(check())
" 2>/dev/null; then
        echo "Database is ready."
        break
    fi
    echo "Waiting for database... ($i/30)"
    sleep 1
done

# Run migrations — tolerate failure so the app can still start in CSV-only mode
if alembic upgrade head; then
    echo "Migrations applied successfully."
else
    echo "WARNING: Migrations failed — running in degraded mode."
fi

# ── Start the application ─────────────────────────────────────────────────
echo "Starting Aegis Trading server..."
cd /app/src
exec uvicorn main:app \
    --host "${EA_LOG_SERVER_HOST:-0.0.0.0}" \
    --port "${EA_LOG_SERVER_PORT:-8000}" \
    --no-access-log
