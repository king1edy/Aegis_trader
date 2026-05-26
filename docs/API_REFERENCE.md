# API Reference

Comprehensive HTTP API catalog for Aegis Trader.

## Base and Auth Conventions

- Base URL: deployment-specific host and port
- Content type: `application/json` unless using OAuth form post
- JWT auth header: `Authorization: Bearer <token>`
- API key auth header: `X-API-Key: <key>`

## Authentication Flows

### Flow A: Dashboard/User Session

1. Register with `POST /api/auth/register` or login with `POST /api/auth/login`
2. Receive JWT access token
3. Use JWT for protected routes (journal, settings, key management)

### Flow B: Machine Webhooks (EA or TradingView)

1. Create API key with `POST /api/auth/api-keys`
2. Store full key securely (returned once)
3. Send webhook requests with `X-API-Key`

## Endpoint Index

| Router | Method | Path | Auth |
|---|---|---|---|
| Auth | POST | `/api/auth/register` | None |
| Auth | POST | `/api/auth/login` | None (OAuth2 form) |
| Auth | GET | `/api/auth/me` | JWT |
| Auth | POST | `/api/auth/api-keys` | JWT |
| Auth | GET | `/api/auth/api-keys` | JWT |
| Auth | DELETE | `/api/auth/api-keys/{key_id}` | JWT |
| EA | GET | `/health` | None |
| EA | POST | `/trade` | API key |
| EA | GET | `/trades` | JWT |
| EA | GET | `/trades/summary` | JWT |
| Journal | GET | `/api/journal/stats` | JWT |
| Journal | GET | `/api/journal/trades` | JWT |
| Journal | GET | `/api/journal/trades/open` | JWT |
| Journal | PATCH | `/api/journal/trades/{trade_id}` | JWT |
| Journal | GET | `/api/journal/deals` | JWT |
| Journal | GET | `/api/journal/analysis/sessions` | JWT |
| Journal | GET | `/api/journal/analysis/hours` | JWT |
| Journal | GET | `/api/journal/analysis/days` | JWT |
| Journal | GET | `/api/journal/analysis/setups` | JWT |
| Journal | GET | `/api/journal/analysis/symbols` | JWT |
| Journal | GET | `/api/journal/analysis/direction` | JWT |
| Journal | GET | `/api/journal/equity` | JWT |
| Journal | GET | `/api/journal/tags` | JWT |
| Journal | POST | `/api/journal/tags` | JWT |
| Journal UI | GET | `/` | None |
| Settings | GET | `/api/settings` | JWT |
| Settings | PATCH | `/api/settings` | JWT |
| Settings | GET | `/api/settings/subscription` | JWT |
| Settings | GET | `/api/settings/rate-limits` | JWT |
| TradingView | POST | `/webhook/tradingview` | API key |

## Auth Router

Base prefix: `/api/auth`

### POST /api/auth/register

Creates a new user and returns JWT.

Request body:

```json
{
  "email": "user@example.com",
  "username": "trader1",
  "password": "strong-password"
}
```

Response 201:

```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "username": "trader1",
  "access_token": "jwt",
  "token_type": "bearer"
}
```

Errors:

- 409 if email or username already exists

### POST /api/auth/login

OAuth2 password grant (`application/x-www-form-urlencoded`).

Form fields:

- `username` (email or username)
- `password`

Response 200:

```json
{
  "access_token": "jwt",
  "token_type": "bearer"
}
```

Errors:

- 401 invalid credentials
- 403 deactivated account

### GET /api/auth/me

Returns enriched profile including subscription and tier rate limits.

Response 200 includes:

- user identity fields
- `subscription` object
- `rate_limits` object

### POST /api/auth/api-keys

Creates a new API key.

Request:

```json
{ "name": "MT5 Terminal A" }
```

Response 201:

```json
{
  "id": "uuid",
  "name": "MT5 Terminal A",
  "key_prefix": "abcd1234",
  "full_key": "returned-once",
  "created_at": "iso-timestamp"
}
```

### GET /api/auth/api-keys

Lists API keys for current user (no full key returned).

### DELETE /api/auth/api-keys/{key_id}

Revokes an API key by setting inactive flag.

Response: 204

## EA Event Router

### GET /health

Health and runtime metadata.

Response:

```json
{
  "status": "ok",
  "version": "3.0.0",
  "events_in_memory": 10,
  "csv_file": "logs/MTFTR_TradeLog.csv",
  "server_time": "iso-timestamp"
}
```

### POST /trade

Ingests EA event payload (API-key protected).

Schema mirrors `TradeEvent` model.

Request sample:

```json
{
  "timestamp": "2026-05-26T12:00:00Z",
  "event": "OPEN",
  "ticket": 123456,
  "symbol": "XAUUSD",
  "direction": "LONG",
  "method": "EMA Bounce",
  "session": "London",
  "entry": 2400.0,
  "sl": 2395.0,
  "tp1": 2405.0,
  "tp2": 2410.0,
  "lots": 0.1,
  "risk_pct": 1.0,
  "sl_dist": 5.0,
  "exit_price": null,
  "pnl": null,
  "rr": null,
  "outcome": "OPEN",
  "d1_bias": "LONG",
  "h4_bias": "LONG",
  "pos_state": 0,
  "balance": 10000.0,
  "equity": 10000.0,
  "note": ""
}
```

Response 201:

```json
{ "status": "logged", "ticket": 123456, "event": "OPEN" }
```

### GET /trades

Returns tenant-scoped events newest first.

Query params:

- `ticket`
- `event`
- `direction`
- `outcome`
- `limit` (default 200)

### GET /trades/summary

Returns summary aggregates and breakdowns:

- win/loss counts
- win rate
- PnL aggregates
- method/session/direction breakdown maps

## Journal Router

### GET /api/journal/stats

Returns overall closed-trade stats.

Response example:

```json
{
  "total_trades": 124,
  "open_trades": 2,
  "wins": 71,
  "losses": 45,
  "breakevens": 8,
  "win_rate": 0.5726,
  "net_pnl": 1845.5,
  "gross_profit": 3920.0,
  "gross_loss": 2074.5,
  "profit_factor": 1.89,
  "avg_rr": 1.47,
  "avg_pnl": 14.88,
  "ea_trades": 98,
  "manual_trades": 26
}
```

### GET /api/journal/trades

Paginated trade listing.

Query params:

- `page` default 1
- `per_page` default 50 (max 500)
- optional filters: `symbol`, `direction`, `session`, `setup`, `source`, `status`

Response shape:

```json
{
  "total": 0,
  "page": 1,
  "per_page": 50,
  "pages": 0,
  "items": []
}
```

Full response example:

```json
{
  "total": 124,
  "page": 1,
  "per_page": 2,
  "pages": 62,
  "items": [
    {
      "id": "6e2f8a43-96c8-4d44-b14a-4a5f67bdf2ad",
      "ticket": 123456,
      "symbol": "XAUUSD",
      "direction": "BUY",
      "status": "CLOSED",
      "trade_source": "ea",
      "entry_price": 2400.0,
      "exit_price": 2407.5,
      "entry_time": "2026-05-26T09:30:00+00:00",
      "exit_time": "2026-05-26T11:10:00+00:00",
      "lot_size": 0.1,
      "stop_loss": 2395.0,
      "take_profit_1": 2405.0,
      "profit_loss": 75.0,
      "outcome": "WIN",
      "exit_reason": "TP2_HIT",
      "risk_reward_actual": 1.5,
      "trading_session": "London",
      "hour_of_day": 9,
      "day_of_week": 1,
      "setup_tag": "EMA Bounce",
      "journal_notes": "Clean continuation",
      "mt5_position_id": 90001,
      "strategy_name": "MTFTR_EA"
    },
    {
      "id": "2ce4c31d-2eb6-4d7b-af18-594f49ec4af8",
      "ticket": 123457,
      "symbol": "XAUUSD",
      "direction": "SELL",
      "status": "OPEN",
      "trade_source": "manual",
      "entry_price": 2412.0,
      "exit_price": null,
      "entry_time": "2026-05-26T12:15:00+00:00",
      "exit_time": null,
      "lot_size": 0.08,
      "stop_loss": 2416.5,
      "take_profit_1": 2408.0,
      "profit_loss": null,
      "outcome": null,
      "exit_reason": null,
      "risk_reward_actual": null,
      "trading_session": "New_York",
      "hour_of_day": 12,
      "day_of_week": 1,
      "setup_tag": null,
      "journal_notes": null,
      "mt5_position_id": 90002,
      "strategy_name": "Manual"
    }
  ]
}
```

### GET /api/journal/trades/open

Returns open and partially-closed positions.

Response example:

```json
[
  {
    "id": "2ce4c31d-2eb6-4d7b-af18-594f49ec4af8",
    "ticket": 123457,
    "symbol": "XAUUSD",
    "direction": "SELL",
    "status": "OPEN",
    "trade_source": "manual",
    "entry_price": 2412.0,
    "exit_price": null,
    "entry_time": "2026-05-26T12:15:00+00:00",
    "exit_time": null,
    "lot_size": 0.08,
    "stop_loss": 2416.5,
    "take_profit_1": 2408.0,
    "profit_loss": null,
    "outcome": null,
    "exit_reason": null,
    "risk_reward_actual": null,
    "trading_session": "New_York",
    "hour_of_day": 12,
    "day_of_week": 1,
    "setup_tag": null,
    "journal_notes": null,
    "mt5_position_id": 90002,
    "strategy_name": "Manual"
  }
]
```

### PATCH /api/journal/trades/{trade_id}

Annotates trade with setup tag and/or notes.

Request:

```json
{
  "setup_tag": "EMA Bounce",
  "journal_notes": "Good entry, late exit"
}
```

Response example:

```json
{
  "status": "ok",
  "trade_id": "6e2f8a43-96c8-4d44-b14a-4a5f67bdf2ad"
}
```

### GET /api/journal/deals

Paginated raw MT5 deal log.

Response example:

```json
{
  "total": 312,
  "page": 1,
  "per_page": 2,
  "pages": 156,
  "items": [
    {
      "id": "8ddfe206-e983-49c8-8b42-9319b9d7fd27",
      "deal_id": 700001,
      "position_id": 90001,
      "symbol": "XAUUSD",
      "deal_time": "2026-05-26T09:30:03+00:00",
      "deal_type": "BUY",
      "entry_type": "IN",
      "volume": 0.1,
      "price": 2400.0,
      "commission": -0.8,
      "swap": 0.0,
      "profit": 0.0,
      "exit_reason": null,
      "comment": "MTFTR open"
    },
    {
      "id": "f78f6c43-66d0-4521-b9ce-d0d8c84d37bb",
      "deal_id": 700002,
      "position_id": 90001,
      "symbol": "XAUUSD",
      "deal_time": "2026-05-26T11:10:00+00:00",
      "deal_type": "SELL",
      "entry_type": "OUT",
      "volume": 0.1,
      "price": 2407.5,
      "commission": -0.8,
      "swap": -0.1,
      "profit": 75.0,
      "exit_reason": "TP",
      "comment": "MTFTR close"
    }
  ]
}
```

### Analysis Endpoints

- `/api/journal/analysis/sessions`
- `/api/journal/analysis/hours`
- `/api/journal/analysis/days`
- `/api/journal/analysis/setups`
- `/api/journal/analysis/symbols`
- `/api/journal/analysis/direction`

Response examples:

`GET /api/journal/analysis/sessions`

```json
[
  {
    "session": "London",
    "total": 60,
    "wins": 38,
    "win_rate": 0.6333,
    "net_pnl": 1240.5,
    "avg_rr": 1.62
  }
]
```

`GET /api/journal/analysis/hours`

```json
[
  {
    "hour": 9,
    "total": 24,
    "wins": 15,
    "win_rate": 0.625,
    "net_pnl": 410.0
  }
]
```

`GET /api/journal/analysis/days`

```json
[
  {
    "day": 1,
    "day_name": "Tue",
    "total": 22,
    "wins": 13,
    "win_rate": 0.5909,
    "net_pnl": 305.5
  }
]
```

`GET /api/journal/analysis/setups`

```json
[
  {
    "setup": "EMA Bounce",
    "total": 45,
    "wins": 30,
    "win_rate": 0.6667,
    "net_pnl": 980.0,
    "avg_rr": 1.71
  }
]
```

`GET /api/journal/analysis/symbols`

```json
[
  {
    "symbol": "XAUUSD",
    "total": 120,
    "wins": 70,
    "win_rate": 0.5833,
    "net_pnl": 1810.0
  }
]
```

`GET /api/journal/analysis/direction`

```json
[
  {
    "direction": "BUY",
    "total": 64,
    "wins": 39,
    "win_rate": 0.6094,
    "net_pnl": 1040.0
  },
  {
    "direction": "SELL",
    "total": 56,
    "wins": 31,
    "win_rate": 0.5536,
    "net_pnl": 770.0
  }
]
```

### GET /api/journal/equity

Returns recent account snapshot time-series.

Query params:

- `limit` default 500, min 10, max 5000

Response example:

```json
[
  {
    "time": "2026-05-26T10:00:00+00:00",
    "balance": 10000.0,
    "equity": 10012.5,
    "floating_pl": 12.5
  },
  {
    "time": "2026-05-26T10:18:00+00:00",
    "balance": 10000.0,
    "equity": 10005.0,
    "floating_pl": 5.0
  }
]
```

### Tags Endpoints

- GET `/api/journal/tags`
- POST `/api/journal/tags`

`GET /api/journal/tags` response example:

```json
[
  {
    "id": 1,
    "name": "EMA Bounce",
    "color": "#3B82F6",
    "description": "Price bounces off EMA"
  }
]
```

Create-tag request:

```json
{
  "name": "Break and Retest",
  "color": "#3B82F6",
  "description": "optional"
}
```

`POST /api/journal/tags` response example:

```json
{
  "id": 7,
  "name": "Break and Retest",
  "color": "#3B82F6"
}
```

### GET /

Returns embedded dashboard HTML (not included in OpenAPI schema).

## Settings Router

Base prefix: `/api/settings`

### GET /api/settings

Returns full `UserSettingsResponse` model.

### PATCH /api/settings

Partial update; only supplied fields are applied.

Patch supports MT5, risk, strategy, notification, and UI preference fields.

### GET /api/settings/subscription

Returns current tier and billing-period metadata.

### GET /api/settings/rate-limits

Returns active rate limit envelope for user tier.

## TradingView Router

### POST /webhook/tradingview

Consumes `TradingViewAlert` payload.

Actions:

- `BUY` or `SELL` routes to open handler
- `CLOSE` routes to close handler and requires `trade_id`

Request sample:

```json
{
  "action": "BUY",
  "symbol": "XAUUSD",
  "price": 2400.0,
  "stop_loss": 2395.0,
  "take_profit": 2406.0,
  "take_profit_2": 2412.0,
  "quantity": 0.1,
  "strategy_name": "TV-MTFTR",
  "timeframe": "15",
  "filters": {
    "ema200_trend": "bullish",
    "rsi": 47.2,
    "session": "london"
  },
  "note": "alert message"
}
```

CLOSE sample:

```json
{
  "action": "CLOSE",
  "symbol": "XAUUSD",
  "trade_id": "uuid",
  "exit_price": 2407.5,
  "pnl": 75.0
}
```

## CSV Column Reference (EA Logging)

CSV columns (in order) used by trade event logging:

- `timestamp`
- `event`
- `ticket`
- `symbol`
- `direction`
- `method`
- `session`
- `entry`
- `sl`
- `tp1`
- `tp2`
- `lots`
- `risk_pct`
- `sl_dist`
- `exit_price`
- `pnl`
- `rr`
- `outcome`
- `d1_bias`
- `h4_bias`
- `pos_state`
- `balance`
- `equity`
- `note`

## Tier and Rate Limit Classes

Current tier envelopes:

| Tier | API/min | API/day | Webhook/min | Backtests/day | Strategies | Accounts |
|---|---:|---:|---:|---:|---:|---:|
| journal | 30 | 5000 | 60 | 2 | 1 | 1 |
| pro | 120 | 20000 | 300 | 20 | 10 | 3 |
| autopilot | 300 | 50000 | 600 | 100 | 25 | 5 |

## Source Anchors

- [src/main.py](../src/main.py#L121)
- [src/auth/router.py](../src/auth/router.py#L105)
- [src/auth/dependencies.py](../src/auth/dependencies.py#L45)
- [src/settings/router.py](../src/settings/router.py#L59)
- [src/settings/schemas.py](../src/settings/schemas.py#L12)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L69)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L98)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/journal/router.py](../src/journal/router.py#L70)
- [src/journal/analyzer.py](../src/journal/analyzer.py#L69)
- [src/journal/analyzer.py](../src/journal/analyzer.py#L389)
- [src/journal/analyzer.py](../src/journal/analyzer.py#L464)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/webhooks/tv_schema.py](../src/webhooks/tv_schema.py#L15)
