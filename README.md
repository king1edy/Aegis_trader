# Aegis Trading Platform

A multi-tenant trading journal and strategy execution platform for MetaTrader 5.

## Purpose

Aegis is a SaaS-ready platform with three tiers:

| Tier | Name | Description |
|------|------|-------------|
| **Tier 1** | Aegis Journal | EA mode — traders install the EA, Aegis journals and analyzes |
| **Tier 2** | Aegis Pro | Journal + risk management enforcement |
| **Tier 3** | Aegis Autopilot | Full Python Bridge strategy execution |

**Phase 1 MVP** delivers Tier 1: users register, authenticate, connect their EA via API key, and get an isolated journal + analytics dashboard.

## Project Structure

```
Aegis_trader/
├── alembic/               # Database migrations (Alembic async)
│   ├── env.py             # Migration environment (reads DB URL from settings)
│   └── versions/          # 5-step migration chain
├── alembic.ini            # Alembic config (sys.path includes src/)
├── docker-compose.yml     # Container orchestration
├── Dockerfile             # Application container
├── requirements.txt       # Python dependencies
│
├── src/                   # Application root (CWD for uvicorn)
│   ├── main.py            # FastAPI entry point (module-level app)
│   ├── trading_system.py  # TradingSystem class (active trading mode)
│   ├── auth/              # Authentication (JWT, API keys, user management)
│   │   ├── models.py      # User + ApiKey SQLAlchemy models
│   │   ├── security.py    # Password hashing (bcrypt), JWT, API key hashing
│   │   ├── dependencies.py# FastAPI deps: get_current_user, get_tenant_id
│   │   └── router.py      # POST /api/auth/{register,login}, API key CRUD
│   ├── core/              # Configuration, logging (structlog + OTEL), exceptions
│   ├── database/          # SQLAlchemy ORM (13 tables), tenant-scoped repository
│   ├── trade_logging/     # EA event receiver (POST /trade, CSV + DB)
│   ├── journal/           # MT5 poller, analyzer, dashboard router
│   ├── execution/         # Broker connectivity (MT5 direct/bridge/paper)
│   ├── strategies/        # MTFTR strategy, indicators, position manager
│   ├── risk/              # Risk checker, position sizer, risk monitor
│   ├── data/              # Market data handling
│   ├── notifications/     # Telegram alerts
│   └── backtesting/       # Strategy validation
│
├── scripts/               # Database migration & setup verification
└── tests/                 # Test suites
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL (localhost:5432 or via Docker)
- MetaTrader 5 (Windows only, for live trading / poller)

### Setup

1. **Clone and configure:**
   ```bash
   git clone <repository>
   cd Aegis_trader
   cp .env.example .env
   # Edit .env with your credentials and settings
   ```

2. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```
   This creates all tables (including `users`, `api_keys`) and seeds a default admin user (`admin@aegis.local`). Set `SEED_ADMIN_PASSWORD` env var before running to choose the admin password (default: `changeme123!`).

4. **Start the server:**
   ```bash
   cd src && uvicorn main:app --reload
   ```

5. **Or use Docker:**
   ```bash
   docker-compose up -d
   ```

### Running Modes

The system operates in two modes, controlled by the `EA_MODE` environment variable:

| Mode | `EA_MODE` | Description |
|------|-----------|-------------|
| **EA mode** | `true` (default) | Passive logging server — receives trade events from the MT5 EA via HTTP POST |
| **Trading mode** | `false` | Full strategy engine — Python executes the MTFTR strategy via MT5 bridge |

### Access Points

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | None | Health check |
| `GET /` | None (JS handles auth) | Trade journal dashboard |
| `POST /trade` | API key (`X-API-Key` header) | EA webhook — receives trade events |
| `GET /trades` | JWT (`Authorization: Bearer`) | Recent trade events |
| `GET /trades/summary` | JWT | Trade summary stats |
| `GET /api/journal/*` | JWT | Journal analytics endpoints |
| `POST /api/auth/register` | None | Create new account |
| `POST /api/auth/login` | None | Login (returns JWT) |
| `GET /api/auth/me` | JWT | Current user profile |
| `POST /api/auth/api-keys` | JWT | Generate EA API key |
| `GET /api/auth/api-keys` | JWT | List API keys |
| `DELETE /api/auth/api-keys/{id}` | JWT | Revoke an API key |

## Authentication

### Multi-Tenant Architecture

Every user is a tenant (`tenant_id == user.id`). All data is isolated per tenant — trades, deals, snapshots, performance metrics, and tags are scoped by `tenant_id` on every query.

### Auth Flows

- **Dashboard (browser)**: JWT via `Authorization: Bearer <token>` — obtained from `POST /api/auth/login`
- **EA webhook (MT5)**: API key via `X-API-Key: <key>` header — generated from the dashboard or `POST /api/auth/api-keys`
- **Health check**: No authentication required

### EA Integration

Add the API key header to your MT5 EA's `WebRequest` call:
```mql5
input string InpFastAPIKey = "";  // Aegis API Key
// In SendTradeEvent(): add "X-API-Key: " + InpFastAPIKey + "\r\n" to headers
```

## Database

### Migrations (Alembic)

Schema is managed by Alembic. The migration chain:

| Migration | Description |
|-----------|-------------|
| `001_baseline` | Stamp for pre-existing databases |
| `002_users_api_keys` | `users` and `api_keys` tables |
| `003_add_tenant_id` | `tenant_id` (nullable) on 10 tenant-scoped tables |
| `004_seed_user` | Create admin user, assign orphaned data |
| `005_tenant_not_null` | Enforce `tenant_id NOT NULL` |

```bash
# Apply all migrations
alembic upgrade head

# Stamp existing DB without running migrations
alembic stamp 001_baseline

# View migration history
alembic history
```

### Schema (13 tables)

| Table | Tenant-scoped | Description |
|-------|:---:|-------------|
| `users` | — | User accounts |
| `api_keys` | — | API keys (FK → users) |
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

## Configuration

All configuration is via environment variables. See `.env.example` for the complete list.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `EA_MODE` | EA mode (passive) or trading mode (active) | `true` |
| `JWT_SECRET_KEY` | Secret for signing JWTs | `change-me-in-production...` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token lifetime | `1440` (24h) |
| `DEFAULT_TENANT_ID` | Tenant UUID for JournalPoller (self-hosted mode) | None (poller disabled) |
| `MT5_LOGIN` | MT5 account number | Required for trading |
| `MT5_PASSWORD` | MT5 password | Required for trading |
| `MT5_SERVER` | Broker server | `Exness-MT5Trial` |
| `MAX_RISK_PER_TRADE` | Risk per trade (decimal) | `0.01` (1%) |
| `MAX_DAILY_RISK` | Max daily risk | `0.03` (3%) |
| `MAX_TRADES_PER_DAY` | Trade limit | `3` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `trading_db` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | SigNoz OTLP endpoint | `http://localhost:4318` |

## Behavioral Safeguards

1. **Manual Override Disabled**: By default, no manual trades or modifications
2. **Consecutive Loss Pause**: Trading pauses after N consecutive losses
3. **Daily Loss Limit**: Stops trading when daily loss exceeds threshold
4. **Drawdown Protection**: Pauses at maximum drawdown level
5. **Trade Cooldown**: Minimum time between trades

## Development

### Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Development server (with hot reload)
cd src && uvicorn main:app --reload

# Or run directly
cd src && python main.py
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/ tests/
ruff src/ tests/
mypy src/
```

## Monitoring

### SigNoz Observability

Logs and traces are exported via OpenTelemetry to SigNoz.

```bash
# Clone and start SigNoz
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy
docker-compose -f docker/clickhouse-setup/docker-compose.yaml up -d
```

Configure in `.env`:
```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=aegis-trading
OTEL_LOGS_ENABLED=true
OTEL_TRACES_ENABLED=true
```

### Telegram Alerts

Configure notifications for trade opens/closes, daily summaries, drawdown warnings, and system errors.

## Important Notes

1. **Demo First**: Always test thoroughly on demo before live deployment
2. **Set JWT_SECRET_KEY**: Use `openssl rand -hex 32` in production
3. **Risk Management**: Never exceed configured risk limits
4. **Logs**: Review logs regularly to understand system behavior
5. **Backtest**: Validate any strategy changes with comprehensive backtesting

## License

Private - For personal use only.
