# Integrations

## Integration Surfaces

Aegis supports multiple ingress and execution-adjacent paths:

- MT5 EA webhook ingestion
- TradingView webhook ingestion
- MT5 direct connector (Windows)
- MT5 bridge connector (container/non-Windows)
- paper broker mode
- Telegram notifications
- OpenTelemetry export to SigNoz
- Redis for rate limiting

## MT5 EA Webhook

Endpoint:

- `POST /trade`

Auth:

- API key via `X-API-Key`

Payload model:

- `TradeEvent` in `src/trade_logging/trade_event_server.py`

Data path:

1. write CSV row
2. append in-memory store
3. persist to DB
4. send notifications

## TradingView Webhook

Endpoint:

- `POST /webhook/tradingview`

Auth:

- API key via `X-API-Key`

Payload model:

- `TradingViewAlert` in `src/webhooks/tv_schema.py`

Action routing:

- `BUY` and `SELL` -> open flow
- `CLOSE` -> close flow and requires `trade_id`

Example payload:

```json
{
  "action": "BUY",
  "symbol": "XAUUSD",
  "price": 2400.0,
  "stop_loss": 2395.0,
  "take_profit": 2406.0,
  "quantity": 0.1,
  "strategy_name": "TV-MTFTR",
  "timeframe": "15",
  "filters": {
    "ema200_trend": "bullish",
    "rsi": 47.2,
    "session": "london"
  }
}
```

## Broker Connection Modes

Configured in `src/core/config.py` via `BROKER_MODE`.

Supported values:

- `auto`
- `direct`
- `bridge`
- `paper`

### Direct Mode

- Uses local MT5 runtime
- Best fit for Windows host deployments

### Bridge Mode

- Uses bridge API (`MT5_BRIDGE_URL`)
- Best fit for Docker/Linux hosting with MT5 on separate Windows node

### Paper Mode

- Simulation mode for non-live execution testing

## Telegram Notifications

Notification facade:

- `src/notifications/service.py`

Channel implementation:

- `src/notifications/telegram.py`

Config:

- bot token stays in infra env settings
- user-level toggles stored in `user_settings`

Notification classes include:

- trade opened/closed
- signal generated/rejected
- risk warnings

## OpenTelemetry and SigNoz

Key settings in `src/core/config.py`:

- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_PROTOCOL`
- `OTEL_SERVICE_NAME`
- `OTEL_LOGS_ENABLED`
- `OTEL_TRACES_ENABLED`
- `OTEL_METRICS_ENABLED`

Compose note indicates SigNoz stack is external and receives telemetry via OTLP.

## Redis Integration

Purpose:

- rate limiting middleware support

Behavior:

- normal operation when Redis is available
- graceful degradation if Redis is unavailable

## Integration Checklist

1. Create user and JWT session.
2. Generate API key.
3. Validate `/health`.
4. Send test event to `/trade` or `/webhook/tradingview`.
5. Confirm event appears in journal endpoints.
6. Confirm expected notification and logs.

## Source Citations

- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L98)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/webhooks/tv_schema.py](../src/webhooks/tv_schema.py#L15)
- [src/execution/broker_factory.py](../src/execution/broker_factory.py#L16)
- [src/notifications/service.py](../src/notifications/service.py#L18)
- [src/core/config.py](../src/core/config.py#L257)
- [src/main.py](../src/main.py#L139)

## Related Docs

- `docs/API_REFERENCE.md`
- `docs/DEPLOYMENT.md`
- `docs/USER_GUIDE.md`
- `src/execution/broker_factory.py`
- `src/webhooks/tv_router.py`
- `src/trade_logging/trade_event_server.py`
