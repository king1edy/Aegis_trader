# Glossary

Shared terminology for product, engineering, operations, and end users.

## Trading Terms

| Term | Meaning |
|---|---|
| Pip | Smallest standardized price move in many FX instruments. |
| Tick | Smallest observed market price change. |
| Lot | Position sizing unit in MT5. |
| Risk:Reward (R:R) | Ratio of potential profit to potential loss on a trade. |
| Drawdown | Decline from an account peak balance or equity. |
| Balance | Closed PnL account value, excluding floating PnL. |
| Equity | Balance plus floating PnL on open positions. |
| Margin Level | Equity divided by margin used, shown as a percent. |
| Swing High/Low | Recent local price extremes used for structure and stops. |
| Session | Market time block such as Asian, London, or New York. |
| EMA | Exponential moving average. |
| Hull MA | Hull moving average, a smoothed trend indicator. |
| RSI | Relative strength index momentum oscillator. |
| ATR | Average true range, volatility measure. |
| Partial Close | Closing part of a position while leaving the remainder open. |
| Trailing Stop | A stop loss that moves with favorable price movement. |
| Prop Firm | A firm that funds traders under rule-based constraints. |

## Aegis Terms

| Term | Meaning |
|---|---|
| Tenant | A single user boundary for data isolation. In current design, one user equals one tenant. |
| Tier | Subscription plan: journal, pro, or autopilot. |
| EA Mode | Passive ingestion mode where MT5 EA sends events to Aegis. |
| Trading Mode | Deprecated direct execution mode in Python when EA_MODE is false. |
| Broker Mode | Connection mode selected by broker factory: auto, direct, bridge, or paper. |
| MTFTR | Multi-timeframe trend rider strategy used by EA and Python mirror. |
| Setup Tag | User-defined category attached to trades for analytics. |
| Signal Source | Origin of signal, including TRADINGVIEW and strategy-defined values. |
| Position State | Lifecycle marker such as initial, tp1_hit, tp2_hit, trailing. |
| RiskChecker | Pre-trade gate that validates limits before execution. |
| RiskMonitor | Background service that continuously enforces drawdown, loss, and margin thresholds. |
| Journal Deals | Raw MT5 deal audit records stored for reproducibility. |
| Trading Pause | System-enforced or manual pause with reason and duration metadata. |

## Source Citations

- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L47)
- [src/database/models.py](../src/database/models.py#L82)
- [src/database/models.py](../src/database/models.py#L500)
- [src/database/models.py](../src/database/models.py#L420)
- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L31)
