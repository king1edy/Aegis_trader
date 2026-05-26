# Features and Tier Matrix

Current features and capability boundaries mapped to Journal, Pro, and Autopilot tiers.

## Tier Comparison

| Capability | Journal | Pro | Autopilot |
|---|---:|---:|---:|
| JWT authentication and profile endpoints | Yes | Yes | Yes |
| API key creation and revocation | Yes | Yes | Yes |
| EA webhook ingestion at POST /trade | Yes | Yes | Yes |
| TradingView webhook ingestion at POST /webhook/tradingview | Yes | Yes | Yes |
| Dashboard and journal analytics routes | Yes | Yes | Yes |
| Per-user settings persistence | Yes | Yes | Yes |
| Rate-limit envelope | Base | Higher | Highest |
| Max API requests per minute | 30 | 120 | 300 |
| Max API requests per day | 5000 | 20000 | 50000 |
| Max webhook events per minute | 60 | 300 | 600 |
| Max backtests per day | 2 | 20 | 100 |
| Max strategies | 1 | 10 | 25 |
| Max connected accounts | 1 | 3 | 5 |
| Risk enforcement usage | Basic visibility | Full enforcement | Full enforcement |
| Direct execution workflow intent | No | Limited | Primary advanced tier |

Source for limits: subscription and rate limit models plus migration seed data.

## Integration Matrix

| Integration Path | Journal | Pro | Autopilot | Notes |
|---|---:|---:|---:|---|
| MT5 EA webhook | Yes | Yes | Yes | Requires API key in X-API-Key header |
| TradingView webhook | Yes | Yes | Yes | POST /webhook/tradingview |
| MT5 direct connector | Optional | Optional | Optional | Broker mode auto/direct |
| MT5 bridge client | Optional | Optional | Optional | Broker mode bridge |
| Paper broker | Optional | Optional | Optional | Broker mode paper |

## Notification Matrix

| Event Type | Channel | Availability |
|---|---|---|
| Trade open | Telegram | User-configurable in settings |
| Trade close | Telegram | User-configurable in settings |
| Daily summary | Telegram | User-configurable in settings |
| Risk breach warning | Telegram | User-configurable in settings |

## Risk Control Matrix

| Control | Layer | Trigger Type | Result |
|---|---|---|---|
| Max daily trades | RiskChecker | Pre-trade | Reject new trade |
| Max open positions | RiskChecker | Pre-trade | Reject new trade |
| Daily loss limit | RiskChecker and RiskMonitor | Pre-trade and continuous | Pause trading |
| Drawdown limit | RiskChecker and RiskMonitor | Pre-trade and continuous | Pause trading |
| Consecutive losses | RiskChecker | Pre-trade | Start pause |
| Margin warning | RiskMonitor | Continuous | Warning and possible block |
| Emergency margin close | RiskMonitor | Continuous | Emergency callback path |

## Practical Competitive Framing

| Dimension | Aegis | Raw MT5 | Journal-only Tools |
|---|---|---|---|
| Unified journaling plus ingestion | Yes | Partial | Partial |
| API-first backend | Yes | No | Varies |
| Built-in tier/rate model | Yes | No | Rare |
| Tenant-scoped persistence | Yes | No | Varies |
| Risk pause automation | Yes | Manual | Usually external |

## Cross-References

- [PRD.md](PRD.md)
- [USE_CASES.md](USE_CASES.md)
- [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md)

## Source Citations

- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L139)
- [alembic/versions/006_subscriptions_settings_ratelimits.py](../alembic/versions/006_subscriptions_settings_ratelimits.py#L136)
- [src/main.py](../src/main.py#L139)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L18)
