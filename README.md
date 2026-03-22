# Aegis Trading Platform

A multi-tenant trading journal and strategy execution platform for MetaTrader 5.

## Purpose

Aegis is a SaaS-ready platform with three subscription tiers:

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Aegis Journal | EA mode -- traders install the EA, Aegis journals and analyzes |
| **Tier 2** | Aegis Pro | Journal + risk management enforcement |
| **Tier 3** | Aegis Autopilot | Full Python Bridge strategy execution |

## Project Structure

```
Aegis_trader/
├── alembic/               # Database migrations (Alembic async)
│   ├── env.py             # Migration environment (reads DB URL from settings)
│   └── versions/          # 6-step migration chain (001-006)
├── alembic.ini            # Alembic config (sys.path includes src/)
├── docker-compose.yml     # Container orchestration (postgres, redis, app)
├── Dockerfile             # Application container (entrypoint runs migrations)
├── requirements.txt       # Python dependencies
│
├── src/                   # Application root (CWD for uvicorn)
│   ├── main.py            # FastAPI entry point (lifespan, router wiring)
│   ├── trading_system.py  # TradingSystem class (active trading mode)
│   ├── auth/              # Authentication & subscriptions
│   │   ├── models.py      # User + ApiKey SQLAlchemy models
│   │   ├── subscription_models.py  # Subscription, UserSettings, RateLimits
│   │   ├── security.py    # Password hashing (bcrypt), JWT, API key hashing
│   │   ├── dependencies.py# FastAPI deps: get_current_user, get_tenant_id
│   │   └── router.py      # Register, login, /me (enriched), API key CRUD
│   ├── settings/          # Per-user settings management
│   │   ├── loader.py      # TradingConfig dataclass + DB loader
│   │   ├── schemas.py     # Pydantic request/response models
│   │   └── router.py      # GET/PATCH /api/settings, subscription, rate-limits
│   ├── core/              # Config (env-based), logging (structlog+OTEL), rate limiter
│   ├── database/          # SQLAlchemy ORM (16 tables), tenant-scoped repository
│   ├── trade_logging/     # EA event receiver (POST /trade, CSV + DB)
│   ├── journal/           # MT5 poller, analyzer, dashboard router
│   ├── execution/         # Broker connectivity (MT5 direct/bridge/paper)
│   ├── strategies/        # MTFTR strategy, indicators, position manager
│   ├── risk/              # Risk checker, position sizer, risk monitor
│   ├── notifications/     # Telegram alerts
│   └── backtesting/       # Strategy validation
│
├── scripts/
│   ├── entrypoint.sh      # Docker entrypoint (migrations + uvicorn)
│   └── init_db.sql        # PostgreSQL extensions + helper functions
├── docs/
│   └── ARCHITECTURE.md    # Full system architecture documentation
└── tests/                 # Test suites
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL 15+ (localhost:5432 or via Docker)
- Redis 7+ (for rate limiting)
- MetaTrader 5 (Windows only, for live trading / poller)

### Docker (recommended)

```bash
# Start everything (migrations run automatically via entrypoint)
docker-compose up -d

# Check logs
docker-compose logs -f trading_app
```

The entrypoint script waits for Postgres, runs `alembic upgrade head`, then starts uvicorn. If migrations fail, the app still starts in CSV-only mode.

### Local Development

```bash
# 1. Clone and configure
git clone <repository>
cd Aegis_trader
cp .env.example .env  # Edit with your credentials

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Run database migrations
alembic upgrade head

# 4. Start the server
cd src && uvicorn main:app --reload
```

### Running Modes

| Mode | `EA_MODE` | Description |
|------|-----------|-------------|
| **EA mode** | `true` (default) | Passive logging server -- receives trade events from the MT5 EA |
| **Trading mode** | `false` | Full strategy engine -- Python executes the MTFTR strategy via MT5 bridge |

## API Endpoints

### Public

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /` | Trade journal dashboard |
| `POST /api/auth/register` | Create new account |
| `POST /api/auth/login` | Login (returns JWT + enriched profile) |

### Authenticated (JWT)

| Endpoint | Description |
|----------|-------------|
| `GET /api/auth/me` | Current user + subscription + rate limits |
| `POST /api/auth/api-keys` | Generate EA API key |
| `GET /api/auth/api-keys` | List API keys |
| `DELETE /api/auth/api-keys/{id}` | Revoke an API key |
| `GET /trades` | Recent trade events |
| `GET /trades/summary` | Trade summary stats |
| `GET /api/journal/*` | Journal analytics |
| `GET /api/settings` | Full user settings |
| `PATCH /api/settings` | Update settings (partial) |
| `GET /api/settings/subscription` | Subscription details |
| `GET /api/settings/rate-limits` | Current tier's rate limits |

### EA Webhook (API Key)

| Endpoint | Description |
|----------|-------------|
| `POST /trade` | EA webhook -- receives trade events (`X-API-Key` header) |

## Authentication

### Multi-Tenant Architecture

Every user is a tenant (`tenant_id == user.id`). All data is isolated per tenant -- trades, deals, snapshots, performance metrics, and tags are scoped by `tenant_id` on every query.

### Auth Flows

- **Dashboard (browser)**: JWT via `Authorization: Bearer <token>` -- obtained from `POST /api/auth/login`
- **EA webhook (MT5)**: API key via `X-API-Key: <key>` header -- generated from the dashboard or `POST /api/auth/api-keys`
- **Health check**: No authentication required

### EA Integration

Add the API key header to your MT5 EA's `WebRequest` call:
```mql5
input string InpFastAPIKey = "";  // Aegis API Key
// In SendTradeEvent(): add "X-API-Key: " + InpFastAPIKey + "\r\n" to headers
```

## Configuration

### Two-Layer Config System

**Layer 1 -- Infrastructure (env-based, `core/config.py`):**

These are server-level settings that don't vary per user.

| Variable | Description | Default |
|----------|-------------|---------|
| `EA_MODE` | EA mode or trading mode | `true` |
| `JWT_SECRET_KEY` | Secret for signing JWTs | `change-me-in-production...` |
| `DEFAULT_TENANT_ID` | User UUID for TradingSystem (single-tenant) | None |
| `POSTGRES_HOST/PORT/DB` | PostgreSQL connection | `localhost:5432/trading_db` |
| `REDIS_HOST/PORT` | Redis connection | `localhost:6379` |
| `BROKER_MODE` | Connection mode (auto/direct/bridge/paper) | `auto` |
| `TELEGRAM_BOT_TOKEN` | Bot token (secret, stays in env) | `""` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | SigNoz OTLP endpoint | `http://localhost:4318` |

**Layer 2 -- Per-User Trading Config (DB-backed, `settings/loader.py`):**

Trading settings (risk rules, strategy params, notifications, sessions) are stored in the `user_settings` DB table. The `.env` values serve as seed defaults for new users.

```
Registration → creates user_settings row with column defaults
Dashboard PATCH /api/settings → updates per-user settings
TradingSystem.initialize() → get_trading_config(user_id) loads from DB
```

| DB Column | Maps To | Description |
|-----------|---------|-------------|
| `max_daily_drawdown_pct` | `max_drawdown_percent` | DB stores 5.00, config uses 0.05 |
| `max_lot_size` | `max_lot_size` | Position size cap |
| `max_open_positions` | `max_open_positions` | Concurrent position limit |
| `max_daily_trades` | `max_daily_trades` | Daily trade cap |
| `allowed_symbols` | `default_symbol` | First symbol used as default |
| `strategy_params` (JSONB) | All `mtftr_*` attrs | Strategy-specific parameters |
| `telegram_enabled` | `telegram_enabled` | Per-user notification toggle |

### Subscriptions & Rate Limiting

Registration creates a `journal` tier subscription (trialing). Rate limits are enforced by Redis-backed sliding window middleware:

| Tier | API/min | API/day | Webhook/min | Strategies | Accounts |
|------|---------|---------|-------------|------------|----------|
| Journal | 30 | 5,000 | 60 | 1 | 1 |
| Pro | 120 | 20,000 | 300 | 10 | 3 |
| Autopilot | 300 | 50,000 | 600 | 25 | 5 |

If Redis is unavailable, rate limiting is silently disabled (graceful degradation).

## Database

### Migrations (Alembic)

Schema is managed by Alembic. In Docker, migrations run automatically via `scripts/entrypoint.sh`.

| Migration | Description |
|-----------|-------------|
| `001_baseline` | Stamp for pre-existing databases |
| `002_users_api_keys` | `users` and `api_keys` tables |
| `003_add_tenant_id` | `tenant_id` (nullable) on 10 tenant-scoped tables |
| `004_seed_user` | Create admin user, assign orphaned data |
| `005_tenant_not_null` | Enforce `tenant_id NOT NULL` |
| `006_subscriptions_settings_ratelimits` | `subscriptions`, `user_settings`, `rate_limits` + seed data |

```bash
alembic upgrade head     # Apply all
alembic history          # View chain
```

### Schema (16 tables)

| Table | Tenant-scoped | Description |
|-------|:---:|-------------|
| `users` | -- | User accounts |
| `api_keys` | -- | API keys (FK -> users) |
| `subscriptions` | -- | Per-user tier + billing state |
| `user_settings` | -- | Per-user trading configuration |
| `rate_limits` | -- | Reference: one row per tier |
| `trades` | Yes | Complete trade lifecycle records |
| `partial_closes` | Yes | Partial close events |
| `trade_modifications` | Yes | SL/TP modification history |
| `journal_deals` | Yes | Raw MT5 deal audit log |
| `setup_tags` | Yes | User-defined trade categorization |
| `account_snapshots` | Yes | Periodic account state captures |
| `daily_performance` | Yes | Aggregated daily metrics |
| `signals` | Yes | Generated signals (executed or not) |
| `system_events` | Yes | Audit trail |
| `trading_pauses` | Yes | When and why trading was paused |
| `price_bars` | No | Shared market data |

## Behavioral Safeguards

1. **Manual Override Disabled**: By default, no manual trades or modifications
2. **Consecutive Loss Pause**: Trading pauses after N consecutive losses
3. **Daily Loss Limit**: Stops trading when daily loss exceeds threshold
4. **Drawdown Protection**: Pauses at maximum drawdown level
5. **Trade Cooldown**: Minimum time between trades
6. **Rate Limiting**: Per-user API rate limits based on subscription tier

## Monitoring

### SigNoz Observability

Logs and traces are exported via OpenTelemetry to SigNoz (runs in a separate compose stack).

```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=aegis-trading
OTEL_LOGS_ENABLED=true
OTEL_TRACES_ENABLED=true
```

### Telegram Alerts

Per-user notification preferences (trade opens/closes, daily summaries, drawdown warnings) are configured via `PATCH /api/settings`. The bot token stays in `.env` as a secret.

## Important Notes

1. **Demo First**: Always test thoroughly on demo before live deployment
2. **Set JWT_SECRET_KEY**: Use `openssl rand -hex 32` in production
3. **Risk Management**: User settings define risk limits; env values are seed defaults only
4. **Docker Migrations**: The entrypoint auto-runs `alembic upgrade head` on container start
5. **Backtest**: Validate any strategy changes with comprehensive backtesting

## License

Private - For personal use only.
