# Deployment Guide

## Deployment Targets

Aegis supports:

- local Python runtime
- Docker Compose stack

Core deployment artifacts:

- `docker-compose.yml`
- `Dockerfile`
- `scripts/entrypoint.sh`
- `alembic.ini`

## Local Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- MT5 runtime if using direct connector mode

### Steps

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
cd src
uvicorn main:app --reload
```

## Docker Compose Stack

Services:

- `trading_app`
- `postgres` (TimescaleDB image)
- `redis`

Ports:

- app: `8000`
- postgres: `5432`
- redis: `6379`

Startup command:

```bash
docker-compose up -d
```

## Container Startup Sequence

Entrypoint behavior (`scripts/entrypoint.sh`):

1. wait for DB connectivity
2. run `alembic upgrade head`
3. continue in degraded mode if migration fails
4. start uvicorn

This is intentionally fail-soft for CSV-first logging continuity.

## Environment Variables

Source defaults are defined in `src/core/config.py`.

### Application

- `APP_NAME`
- `APP_ENV`
- `DEBUG`
- `LOG_LEVEL`

### Auth and Security

- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

### Database and Cache

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `DATABASE_URL`
- `REDIS_HOST`
- `REDIS_PORT`
- `REDIS_DB`
- `REDIS_URL`

### EA/Webhook Runtime

- `EA_MODE`
- `EA_LOG_SERVER_HOST`
- `EA_LOG_SERVER_PORT`
- `TRADE_LOG_CSV_PATH`

### Broker Connectivity

- `BROKER_MODE`
- `MT5_BRIDGE_URL`
- `MT5_BRIDGE_API_KEY`
- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`
- `MT5_PATH`

### Notification

- `TELEGRAM_BOT_TOKEN`

### Observability

- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_PROTOCOL`
- `OTEL_SERVICE_NAME`
- `OTEL_RESOURCE_ATTRIBUTES`
- `OTEL_LOGS_ENABLED`
- `OTEL_TRACES_ENABLED`
- `OTEL_METRICS_ENABLED`

## Production Checklist

1. rotate `JWT_SECRET_KEY` to strong random value
2. set `APP_ENV=production`
3. set `DEBUG=false`
4. run behind TLS termination proxy
5. enforce backup cadence for Postgres and logs
6. monitor migration status and app health endpoint
7. protect API key and secret distribution process

## Migrations Operations

```bash
alembic history
alembic current
alembic upgrade head
alembic downgrade -1
```

## Backup and Recovery

Recommended:

- regular Postgres logical backups (`pg_dump`)
- durable storage for `logs/MTFTR_TradeLog.csv`

CSV log can act as last-resort event journal if DB path is degraded.

## Scaling Considerations

- single instance is adequate for early usage
- bridge mode may become bottleneck under high webhook throughput
- add infra-level horizontal scaling only after validating state and idempotency patterns

## Source Citations

- [docker-compose.yml](../docker-compose.yml#L1)
- [docker-compose.yml](../docker-compose.yml#L3)
- [Dockerfile](../Dockerfile#L1)
- [Dockerfile](../Dockerfile#L44)
- [scripts/entrypoint.sh](../scripts/entrypoint.sh#L34)
- [scripts/entrypoint.sh](../scripts/entrypoint.sh#L43)
- [src/core/config.py](../src/core/config.py#L36)

## Related Docs

- `docs/INTEGRATIONS.md`
- `docs/API_REFERENCE.md`
- `docs/DATABASE_SCHEMA.md`
