# User Guide

This guide helps traders onboard and operate Aegis Trader end-to-end.

## 1. Create Your Account

1. Open the app host URL.
2. Register with email, username, and password.
3. Log in to receive a session token in the dashboard flow.

API equivalents:

- `POST /api/auth/register`
- `POST /api/auth/login`

## 2. Generate Your API Key (Required for EA and TradingView)

1. Open API key management from your authenticated session.
2. Create a key with a descriptive name.
3. Copy and store the full key immediately (shown once).

API equivalent:

- `POST /api/auth/api-keys`

## 3. Dashboard Tour

Main dashboard route:

- `GET /`

Key capabilities:

- summary stats
- trade list and filters
- open trades section
- analysis panels by session/hour/day/setup/symbol/direction
- equity curve
- setup tag management

## 4. Connect MT5 EA

### EA Settings

In your EA inputs, set:

- FastAPI URL to your `/trade` endpoint
- API key for `X-API-Key` authentication

### Typical Workflow

1. Attach EA to symbol chart.
2. Place or manage trades.
3. Verify events are logged and visible in dashboard and journal APIs.

Verification endpoints:

- `GET /health`
- `GET /trades`
- `GET /trades/summary`

## 5. Connect TradingView Alerts

Use webhook URL:

- `/webhook/tradingview`

Auth header:

- `X-API-Key`

Alert JSON must follow `TradingViewAlert` schema.

Minimum BUY/SELL example:

```json
{
  "action": "BUY",
  "symbol": "XAUUSD",
  "price": 2400.0,
  "stop_loss": 2395.0,
  "take_profit": 2406.0,
  "quantity": 0.1,
  "strategy_name": "TV-MTFTR"
}
```

CLOSE requires `trade_id`.

## 6. Configure Risk Rules

Use settings endpoints:

- `GET /api/settings`
- `PATCH /api/settings`

High-impact fields:

- `max_daily_drawdown_pct`
- `max_consecutive_losses`
- `max_lot_size`
- `max_open_positions`
- `max_daily_trades`
- `pause_on_rule_breach`

Suggested conservative starter profile:

- daily drawdown 5%
- max consecutive losses 3
- pause on breach enabled

## 7. Configure Notifications

User-level fields in settings:

- `telegram_chat_id`
- `telegram_enabled`
- `notify_on_trade_open`
- `notify_on_trade_close`
- `notify_on_daily_summary`
- `notify_on_risk_breach`

Global bot token is managed by operator environment configuration.

## 8. Tag Trades and Review Analytics

### Tagging

- Create tags: `POST /api/journal/tags`
- Annotate trade: `PATCH /api/journal/trades/{trade_id}`

### Analytics Routes

- `/api/journal/analysis/sessions`
- `/api/journal/analysis/hours`
- `/api/journal/analysis/days`
- `/api/journal/analysis/setups`
- `/api/journal/analysis/symbols`
- `/api/journal/analysis/direction`

## 9. Understand Trading Pauses

If limits are breached, trading may be blocked temporarily.

Common triggers:

- daily loss limit reached
- drawdown limit reached
- consecutive loss threshold reached
- margin risk conditions

Check your settings and risk notifications to understand what triggered the pause.

## 10. Troubleshooting

### I do not see new trades

- verify API key is active
- verify webhook URL is correct
- check `/health`
- check dashboard filters and tenant session

### TradingView CLOSE fails

- ensure `trade_id` is provided
- verify action is exactly `CLOSE`

### Rate limit errors

- verify current tier via `/api/settings/subscription`
- inspect limits via `/api/settings/rate-limits`

### Telegram not sending

- verify operator configured bot token
- verify your `telegram_chat_id`
- verify `telegram_enabled=true`

## Screenshot Placeholders

- [Placeholder] Login screen
- [Placeholder] API key creation modal
- [Placeholder] Dashboard summary cards
- [Placeholder] Trade table tagging action
- [Placeholder] Settings risk panel
- [Placeholder] TradingView alert config screen

## Source Citations

- [src/auth/router.py](../src/auth/router.py#L105)
- [src/auth/router.py](../src/auth/router.py#L226)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/settings/router.py](../src/settings/router.py#L59)
- [src/settings/schemas.py](../src/settings/schemas.py#L12)
- [src/journal/router.py](../src/journal/router.py#L70)
- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)

## Related Docs

- `docs/API_REFERENCE.md`
- `docs/RISK_MANAGEMENT.md`
- `docs/INTEGRATIONS.md`
- `docs/STRATEGY.md`
