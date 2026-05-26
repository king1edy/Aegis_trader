# Risk Management

## Layered Risk Architecture

Aegis applies multiple risk layers, each with a distinct responsibility.

1. Pre-trade gate (`RiskChecker`)
2. Continuous monitor (`RiskMonitor`)
3. Position sizing (`PositionSizer`)
4. Trading pause persistence (`trading_pauses` + system events)

Core sources:

- `src/risk/risk_checker.py`
- `src/risk/risk_monitor.py`
- `src/risk/position_sizer.py`
- `src/database/models.py` (`trading_pauses`)

## Pre-Trade Controls (RiskChecker)

`RiskChecker.check_all_limits()` validates a candidate signal before execution.

Checks include:

- active trading pause
- max trades per day
- max open positions
- daily loss threshold
- maximum drawdown threshold
- consecutive losses threshold
- margin availability

If a limit is breached, the signal is rejected with a structured reason and can trigger pause or notification behavior.

## Continuous Controls (RiskMonitor)

`RiskMonitor` runs background checks on a cadence (default around 18 seconds).

It tracks:

- peak balance and peak equity
- current balance and equity
- daily and weekly starting balances
- daily and weekly loss percentages
- margin level

Threshold breaches set trading blocked state and can trigger emergency close callbacks.

## RiskMonitor Default Thresholds

From `RiskMonitorConfig` defaults:

| Setting | Default |
|---|---:|
| `check_interval_seconds` | 18.0 |
| `max_balance_drawdown_pct` | 10.0 |
| `max_equity_drawdown_pct` | 15.0 |
| `max_daily_loss_pct` | 5.0 |
| `max_weekly_loss_pct` | 10.0 |
| `min_margin_level_pct` | 150.0 |
| `emergency_margin_level_pct` | 100.0 |

## Config Defaults Used Across Risk Logic

From `src/core/config.py` seed defaults:

| Setting | Default |
|---|---:|
| `max_risk_per_trade` | 0.01 |
| `max_daily_risk` | 0.03 |
| `max_drawdown_percent` | 0.10 |
| `max_trades_per_day` | 3 |
| `max_open_positions` | 2 |
| `max_daily_trades` | 3 |
| `max_daily_loss_percent` | 0.03 |
| `min_margin_level` | 200.0 |
| `max_consecutive_losses` | 3 |
| `pause_duration_hours` | 4 |
| `cooldown_after_loss_minutes` | 30 |

Note: runtime values can be overridden per user via `user_settings` and settings APIs.

## Position Sizing Math

`PositionSizer` uses fixed-fractional sizing.

Core formulas:

$$
\text{risk\_amount} = \text{account\_balance} \times \text{risk\_percent}
$$

$$
\text{sl\_distance\_pips} = \frac{|\text{entry\_price} - \text{stop\_loss}|}{\text{point}}
$$

$$
\text{lot\_size} = \frac{\text{risk\_amount}}{\text{sl\_distance\_pips} \times \text{tick\_value}}
$$

Final lot size is normalized to broker min/max/step constraints.

### Worked Example

Assume:

- balance = 10,000
- risk_percent = 0.01 (1%)
- entry = 2400.00
- stop_loss = 2398.00
- point = 0.01
- tick_value = 1.00

Then:

- risk_amount = 100
- sl_distance_pips = 200
- raw lot = 100 / (200 x 1) = 0.50 lots

After symbol normalization and bounds, this becomes broker-compliant final lot.

## Drawdown Semantics

### Balance Drawdown

Computed from peak historical balance to current balance.

### Equity Drawdown

Computed from peak historical equity to current equity.

### Daily and Weekly Loss

Computed from reset anchors:

- daily reset at UTC date rollover
- weekly reset on week boundary

`RiskMonitor` updates these anchors and re-evaluates loss percentages continuously.

## Margin Protection

Two-level behavior:

- warning/block threshold when margin level drops below minimum threshold
- emergency threshold for severe margin risk triggering emergency callback path

Defaults in monitor config:

- warning boundary at 150%
- emergency boundary at 100%

## Trading Pause Model

Pause records are persisted in `trading_pauses` with:

- `start_time`
- `end_time`
- `reason`
- trigger and threshold values
- auto/manual flag
- notes

Pause reasons may include:

- `daily_loss_limit`
- `max_drawdown`
- `consecutive_losses`
- margin-related risk events

## Per-User Overrides

`user_settings` supports user-specific risk preferences, including:

- `max_daily_drawdown_pct`
- `max_consecutive_losses`
- `max_lot_size`
- `max_open_positions`
- `max_daily_trades`
- `pause_on_rule_breach`
- allowed sessions and symbols

These values are read via settings loader and API endpoints.

## Suggested Prop-Firm Profile

A common strict profile for evaluation-style trading:

- daily loss: 5%
- total drawdown: 10%
- low max open positions
- pause on breach enabled
- conservative risk per trade (1% or less)

Aegis supports this through configurable checker and monitor thresholds plus persisted pause records.

## Source Citations

- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L18)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L47)
- [src/risk/position_sizer.py](../src/risk/position_sizer.py#L28)
- [src/core/config.py](../src/core/config.py#L283)
- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L72)
- [src/database/models.py](../src/database/models.py#L420)

## Related Docs

- `docs/STRATEGY.md`
- `docs/PRD.md`
- `docs/FEATURES.md`
- `src/risk/risk_checker.py`
- `src/risk/risk_monitor.py`
- `src/risk/position_sizer.py`
