# 🏛️ Aegis Trader — v2 Architecture

> **Multi-asset, multi-broker, discipline-first.**
>
> Status: Draft v0.1 — May 2026
> Author: Architecture working session
> Predecessor: `docs/ARCHITECTURE.md` (v1.1, Feb 2026)

---

## 0. Framing

This is a **refactor + extension** of v1, not a rebuild.

v1 already gives us: multi-tenant DB, JWT/API-key auth, subscription tiers, settings system, alembic migrations, rate limiting, SigNoz observability, Telegram notifications, backtesting engine, EA mode + Trading mode, `BrokerInterface` ABC, `BaseStrategy` ABC. That's months of work and it's good work. **None of it gets thrown out.**

What v2 changes:

1. **Asset class is now a first-class concept.** Users choose FX, Synthetics, or both. Strategies, brokers, risk profiles, and lot specs are all asset-class-aware.
2. **The broker layer becomes truly pluggable.** MT5 stays. Deriv joins. The interface gets cleaned of MT5-isms.
3. **Data ingestion is decoupled from broker.** Deriv streams ticks over WebSocket; MT5 polls. v1's tick-loop assumes polling. v2 supports both.
4. **Strategy templates become a published, versioned artifact.** Users don't bring code. They subscribe to backtested templates with disclosed metrics.
5. **The product positioning is locked.** AegisTrader is a discipline + risk + attribution system. It does not promise profit. Strategy templates publish their drawdowns honestly.

What v2 does **not** do (scope discipline):

- No new auth system
- No new migrations system (extend the existing 006 chain)
- No new observability stack
- No microservices split (monolith with internal modules stays, for now)
- No custom strategy code from users (templates only, in v2)

---

## 1. Product Tenets (non-negotiable)

These are the constraints every architectural decision is checked against.

1. **No profit guarantees.** Anywhere. Marketing, UI, API responses, docs. Strategy templates publish backtest metrics with full drawdown disclosure.
2. **Forced demo before live.** No template can be enabled in live trading until the user has forward-tested it on demo for ≥ N trades (configured per template, default 50).
3. **Edge before infra.** A strategy template is not shipped until it has a documented edge measured against ≥ 6 months of real tick data, with realistic costs modeled.
4. **Max 6 supported instruments at launch.** Quality over quantity. Each one has a properly backtested template, documented edge, published metrics.
5. **Risk Fortress is centralized and cannot be bypassed.** Every order goes through it. No exceptions.
6. **Tenant isolation is enforced at the DB layer.** No code path can leak across tenants.
7. **Observability for everything.** Every signal, order, fill, risk event, backtest run is traced + logged + metered via OTel → SigNoz.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          AEGIS TRADER v2                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                       PRESENTATION LAYER                             │    │
│  │  FastAPI (REST) + Streamlit Dashboard + EA Webhook + Telegram        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     APPLICATION LAYER                                │    │
│  │  Auth │ Settings │ Subscriptions │ Templates │ Backtest │ Journal    │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                      TRADING CORE                                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │    │
│  │  │   Signal     │  │     Risk     │  │  Execution   │                │    │
│  │  │  Generator   │─►│   Fortress   │─►│  Dispatcher  │                │    │
│  │  └──────▲───────┘  └──────────────┘  └──────┬───────┘                │    │
│  │         │                                    │                       │    │
│  │  ┌──────┴───────────────────────────────────▼───────┐                │    │
│  │  │         ASSET-CLASS REGISTRY                     │                │    │
│  │  │  FX Module │ Synth Module │ (future modules)     │                │    │
│  │  └──────▲───────────────────────────────────▲───────┘                │    │
│  │         │                                    │                       │    │
│  │  ┌──────┴────────────┐              ┌────────┴───────────┐           │    │
│  │  │  Data Ingestion   │              │  Broker Adapters   │           │    │
│  │  │  - MT5 Poller     │              │  - MT5 Connector   │           │    │
│  │  │  - Deriv WS       │              │  - Deriv API       │           │    │
│  │  └───────────────────┘              └────────────────────┘           │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    PERSISTENCE LAYER                                 │    │
│  │  Postgres (source of truth) │ Redis (hot state) │ MinIO (cold)       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                   OBSERVABILITY (existing)                           │    │
│  │  OTel → SigNoz → Logs + Traces + Metrics                             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Abstractions

### 3.1 Asset Class

A new first-class concept. Drives instrument metadata, lot specs, supported brokers, strategy templates, risk model defaults.

```python
class AssetClass(str, Enum):
    FX = "fx"            # MT5-routed, conventional FX pairs + metals
    SYNTHETIC = "synth"  # Deriv-routed, synthetic indices
    # Future: CRYPTO, EQUITIES, etc.
```

Stored on `instruments`, `strategy_templates`, `user_strategy_configs`, and `subscriptions.allowed_asset_classes`.

### 3.2 Broker Adapter (refactored `BrokerInterface`)

v1 has `BrokerInterface` but it's MT5-shaped. v2 generalizes it. Two distinct capability surfaces every adapter must expose:

```python
class MarketDataAdapter(ABC):
    """Pull or stream price data."""
    @abstractmethod
    async def get_ohlcv(symbol, timeframe, count) -> List[Bar]: ...

    @abstractmethod
    async def stream_ticks(symbol) -> AsyncIterator[Tick]: ...
    # ^ NEW in v2 — required because Deriv is push-based

    @abstractmethod
    async def get_symbol_info(symbol) -> SymbolInfo: ...


class ExecutionAdapter(ABC):
    """Place, modify, close orders."""
    @abstractmethod
    async def open_position(...) -> TradeResult: ...

    @abstractmethod
    async def close_position(ticket) -> TradeResult: ...

    @abstractmethod
    async def modify_position(ticket, sl, tp) -> TradeResult: ...

    @abstractmethod
    async def get_open_positions() -> List[Position]: ...

    @abstractmethod
    async def get_account_info() -> AccountInfo: ...
```

| Adapter           | MarketData | Execution | Notes                                   |
|-------------------|:----------:|:---------:|-----------------------------------------|
| `MT5Connector`    | ✅ poll    | ✅        | Existing v1, refactor for cleanliness    |
| `MT5APIClient`    | ✅ poll    | ✅        | Existing v1, no change                   |
| `DerivAdapter`    | ✅ stream  | ✅        | **NEW** — WebSocket for both             |
| `PaperBroker`     | ✅         | ✅        | Existing, used for forced demo period    |

**Key change:** `stream_ticks` is added as a required method. MT5 adapter implements it by polling at 1s intervals and yielding bars; Deriv implements it natively via WebSocket. Strategy code does not care which.

### 3.3 Strategy Template

This is the biggest conceptual change in v2.

In v1, a strategy is a Python class. In v2, a strategy template is a **published, versioned, backtested artifact** that:

- Has a unique `template_id` and `version`
- Belongs to an asset class
- Has documented entry/exit logic
- Has published backtest metrics (Sharpe, max DD, win rate, expectancy, average trades/month, recommended capital, recommended risk per trade)
- Has minimum demo trades before live (`min_demo_trades`)
- Has a code module that implements `BaseStrategy`

Users **subscribe** to a template and configure their per-user params (within ranges the template allows). They don't write code.

```
strategy_templates
├── boom_1000_spike_drift_v1     (asset_class=SYNTHETIC, broker=deriv)
├── crash_1000_spike_drift_v1    (asset_class=SYNTHETIC, broker=deriv)
├── vol_75_mean_reversion_v1     (asset_class=SYNTHETIC, broker=deriv)
├── xauusd_mtftr_v2              (asset_class=FX, broker=mt5)  [migrated from v1]
└── ...                          [max 6 at launch]
```

### 3.4 Signal → Risk → Execution pipeline

Every signal flows through one canonical pipeline. This is enforced.

```
[1] Signal Generator (per template, per tenant)
        │ emits TradingSignal
        ▼
[2] Risk Fortress
        │ checks: max daily loss, drawdown, consec losses,
        │ open positions, cooldown, tenant kill switch,
        │ asset-class-specific limits, template demo gate
        │
        ├──► REJECTED → log risk_event, notify user, end
        │
        └──► APPROVED → emit ApprovedOrder
        ▼
[3] Execution Dispatcher
        │ routes to correct broker adapter by template.broker
        ▼
[4] Broker Adapter (MT5 or Deriv)
        │ executes → returns TradeResult
        ▼
[5] Journal
        │ persist trade, emit OTel span, send Telegram alert
```

**Critical property:** there is no path from Signal Generator to Broker Adapter that bypasses Risk Fortress. The Execution Dispatcher accepts only `ApprovedOrder` objects, which can only be produced by Risk Fortress.

---

## 4. Data Model Changes

v1's 16 tables stay. v2 adds the following:

### 4.1 New tables

| Table                       | Tenant-scoped | Purpose                                                                  |
|-----------------------------|:-------------:|--------------------------------------------------------------------------|
| `asset_classes`             | --            | Reference: FX, Synthetic, etc. with display names + metadata             |
| `instruments`               | --            | The (≤6) supported instruments. Asset class, broker, lot specs, status   |
| `brokers`                   | --            | Reference: MT5, Deriv, with API endpoints + capabilities                 |
| `strategy_templates`        | --            | The published, backtested templates                                       |
| `strategy_template_versions`| --            | Version history with published metrics                                    |
| `backtest_runs`             | Yes           | User-initiated backtest jobs (or system-published baseline runs)         |
| `backtest_results`          | Yes           | Trades + equity curve + metrics from a run                               |
| `user_strategy_configs`     | Yes           | Per-user instance of a template (params, demo/live state, account)       |
| `user_broker_credentials`   | Yes           | Encrypted per-user broker tokens (Deriv API token, MT5 creds)            |
| `demo_progress`             | Yes           | Tracks demo trades per user_strategy_config toward demo gate             |
| `tick_archive_index`        | --            | Index into MinIO Parquet partitions                                       |

### 4.2 Changes to existing tables

| Table              | Change                                                                                       |
|--------------------|----------------------------------------------------------------------------------------------|
| `subscriptions`    | Add `allowed_asset_classes` (array of asset class enums)                                     |
| `user_settings`    | Add `default_asset_class`, `per_asset_class_risk_overrides` (JSONB)                          |
| `trades`           | Add `template_id`, `template_version`, `asset_class`, `broker`, `user_strategy_config_id`    |
| `signals`          | Add same set as `trades`                                                                     |
| `daily_performance`| Partition rollups by `asset_class` and `broker`                                              |

### 4.3 Migration approach

Continue the alembic chain from `006`. New migrations:

```
007_asset_classes_brokers_instruments  -- reference tables, seed data
008_strategy_templates                  -- templates + versions
009_user_strategy_configs               -- per-user instances + demo progress
010_user_broker_credentials             -- encrypted creds
011_backtest_runs_results              -- backtest persistence
012_extend_trades_signals              -- add template/asset metadata
013_tick_archive_index                 -- MinIO index
```

Each migration is reversible. Each ships with a data-fix migration if needed (e.g., backfill `asset_class=FX` on existing `trades`).

---

## 5. Multi-Asset User Journey

### 5.1 Onboarding

```
1. Register (existing v1 flow)
2. Pick asset class(es) → FX, Synth, both
   (gated by subscription tier — Journal: 1, Pro: 1, Autopilot: both)
3. For each asset class, connect a broker
   - FX → enter MT5 creds (via existing v1 flow)
   - Synth → enter Deriv API token (read + trade scopes)
4. Browse strategy templates (filtered by allowed asset classes)
5. Subscribe to template → creates user_strategy_config in DEMO state
6. Trade demo until min_demo_trades hit → eligible for LIVE
7. Promote to LIVE → orders route to real broker
```

### 5.2 State machine: `user_strategy_config.state`

```
DRAFT ──► DEMO ──► DEMO_PASSED ──► LIVE
             │                       │
             └─► DEMO_FAILED         └─► PAUSED ──► LIVE
                  (user re-configures)    (risk fortress)
                       ▲                     │
                       └─────────────────────┘
```

Transitions are append-only events to `system_events` for audit.

### 5.3 Risk overrides per asset class

Synthetic indices behave differently from FX. Default risk profiles should differ. Users can override within bounds:

| Setting                  | FX default | Synth default | Hard max  |
|--------------------------|-----------:|--------------:|----------:|
| Risk per trade           | 1.0%       | 0.5%          | 2.0%      |
| Max daily loss           | 3.0%       | 2.0%          | 5.0%      |
| Max open positions       | 3          | 1             | 5         |
| Max consec losses → pause| 3          | 2             | 5         |
| Trade cooldown (min)     | 30         | 5             | --        |

---

## 6. Deriv Integration — New Module

Implementation lives in `src/execution/deriv/` and `src/data/deriv/`.

### 6.1 Components

```
src/execution/deriv/
├── adapter.py          # DerivAdapter implementing Market+Execution interfaces
├── ws_client.py        # Async WebSocket client (reconnect, heartbeat, auth)
├── api_models.py       # Pydantic models for Deriv WS messages
├── symbol_mapping.py   # Deriv symbol codes → Aegis instrument metadata
└── auth.py             # Token-based auth, per-tenant token resolution

src/data/deriv/
├── tick_stream.py      # Subscribes to ticks, fans out to Redis + Postgres + MinIO
├── tick_archive.py     # Periodic Parquet flush to MinIO
└── history_loader.py   # Pull historical ticks for backtesting
```

### 6.2 Tick flow

```
Deriv WS ──► ws_client ──► tick_stream ──┬──► Redis (latest tick, key: tick:{tenant}:{symbol})
                                          ├──► Strategy Generator (async consumer)
                                          ├──► Postgres (hot table: ticks_recent, 24h retention)
                                          └──► MinIO Parquet (cold archive, partitioned by date+symbol)
```

### 6.3 Execution flow

```
Approved Order ──► DerivAdapter.open_position
                       │
                       └─► Deriv API: /buy or /sell with appropriate contract type
                              │
                              └─► fill or rejection
                                     │
                                     └─► Journal + Telegram + OTel span
```

### 6.4 Supported Deriv instruments at launch

Pick from this candidate list, validated through Phase 1 backtests:

| Symbol         | Aegis name              | Why considered                        |
|----------------|-------------------------|---------------------------------------|
| `BOOM1000`     | Boom 1000               | Spike-counting edge, mechanical       |
| `CRASH1000`    | Crash 1000              | Mirror of Boom, same logic            |
| `R_75`         | Volatility 75 Index     | Liquid, behaves like FX               |
| `R_100`        | Volatility 100 Index    | Higher vol, similar mechanics         |
| `R_75_1s`      | Volatility 75 (1s)      | Lower lot mins, better for small accts|
| `stpRNG`       | Step Index              | Fixed step size, patience play        |

**Target:** ship with 2–3 of these initially. The other 3–4 slots reserved for proven FX templates (XAUUSD MTFTR migrated from v1, plus any others that pass v2 backtest standards).

---

## 7. Strategy Template Lifecycle

```
[1] Hypothesis → notebook (data exploration, edge probe)
[2] Backtest → ≥ 6 months realistic tick data, costs modeled
[3] Code → implement BaseStrategy subclass under src/strategies/templates/
[4] Internal review → drawdown, Sharpe, expectancy meet thresholds
[5] Publish → strategy_templates row + version + published metrics
[6] Demo-only release → users can subscribe in DEMO state
[7] Forward test on demo accounts ≥ 30 days
[8] If demo results match backtest → enable LIVE promotion
[9] Ongoing monitoring → real performance tracked vs published metrics
[10] If real performance drifts > X% from backtest → freeze + investigate
```

Published metrics on every template:

- Backtest period, instrument, data source
- Trades count, win rate, expectancy in R, average trade duration
- Sharpe, Sortino, max drawdown (absolute + relative), Calmar
- Recovery factor, profit factor
- Worst losing streak
- Recommended capital floor + recommended risk-per-trade
- Average monthly trades
- **A plain-English honest paragraph** describing how this strategy loses money

---

## 8. Risk Fortress (centralized risk gate)

v1 has risk checks but they live partly in the trading loop and partly in the strategy. v2 centralizes into one module that every order must pass through.

Gates evaluated in order. Short-circuits on first failure:

1. **Tenant kill switch** (`trading_pauses` active)
2. **Subscription tier permits this asset class**
3. **Template state is LIVE** (not DEMO, PAUSED, DEMO_FAILED)
4. **Per-asset-class risk overrides honored** (risk %, daily loss, open count)
5. **Account-level guards** (margin level, equity drawdown, balance drawdown)
6. **Time-based** (cooldown, max trades per day, allowed sessions for FX)
7. **Per-template hard limits** (max position size, max simultaneous in same template)
8. **Sanity** (SL distance > minimum, lot rounded to broker step)

Every gate decision is logged to `risk_events` with full context. Rejections are emitted as Telegram alerts (configurable per user).

---

## 9. Migration Plan v1 → v2

Phased. Each phase is shippable on its own. No big-bang cutover.

### Phase A — Abstraction refactor (no new features)
- Refactor `BrokerInterface` into `MarketDataAdapter` + `ExecutionAdapter`
- Add `stream_ticks` requirement; implement MT5 stub by polling
- Introduce `AssetClass` enum, default everything to FX
- Add asset class column to `instruments`, `trades`, `signals`
- All v1 functionality preserved, all tests pass
- **Risk:** Low. **Time:** ~1 week.

### Phase B — Strategy templates as data
- Migrate `xauusd_mtftr_v2` into the new template structure
- Backfill template_id on existing trades/signals
- Update settings UI to show user the template they're running
- **Risk:** Low. **Time:** ~1 week.

### Phase C — Deriv adapter + Synth module
- Build `DerivAdapter`, WS client, tick stream
- Wire into existing Strategy → Risk → Execution pipeline
- No live trading yet — connector only, paper trades only
- **Risk:** Medium. **Time:** ~2 weeks.

### Phase D — First Synth strategy template
- Boom 1000 spike-drift template (or whatever survives Phase 1 notebook validation)
- Published metrics from real backtest
- Demo-only mode
- **Risk:** Highest. This is the actual edge work. **Time:** ~3–4 weeks.

### Phase E — Multi-asset onboarding UX
- User picks asset class(es), connects brokers, subscribes to templates
- Demo gate enforcement in UI
- Per-asset risk overrides
- **Risk:** Medium. **Time:** ~2 weeks.

### Phase F — Live for Synth
- Promotion to LIVE state for synth templates that passed demo
- Tiny-size launch with internal users first
- Monitor for 30 days minimum before broader rollout
- **Risk:** Medium-high. **Time:** ~ongoing.

---

## 10. What's Explicitly NOT in v2

To prevent scope creep:

- ❌ Custom user-supplied strategy code (templates only)
- ❌ Crypto, equities, or other asset classes
- ❌ Mobile app (web/Streamlit only)
- ❌ Social/copy-trading features
- ❌ Brokers other than MT5 + Deriv
- ❌ Microservices split (monolith stays)
- ❌ Replace Postgres / Redis / MinIO / Kafka / SigNoz — your existing stack is correct
- ❌ Strategy marketplace where 3rd parties publish templates
- ❌ Any form of "guaranteed profit" language anywhere in the product

---

## 11. Open Questions (need decision)

1. **Subscription tier scope.** Do existing Journal/Pro/Autopilot tiers stay as-is, or do they need new asset-class entitlements? Current proposal: Journal = 1 asset class, Pro = 1 asset class with risk enforcement, Autopilot = both asset classes with automation.
2. **Deriv API token storage.** Encrypt at rest with Fernet (symmetric) or use Vault/AWS KMS? Recommend Fernet for v2 launch, plan KMS migration later.
3. **Tick archive retention.** How long do we keep raw ticks in MinIO? Recommend 18 months (covers a typical backtest window), then aggregate to OHLCV.
4. **Demo broker for synth.** Deriv has a built-in demo account — use that, or run our own paper-trading sim? Recommend Deriv's demo for realism, fall back to internal sim if their demo API has restrictions.
5. **Frontend.** Stick with Streamlit for the dashboard, or invest in a proper React SPA? Recommend Streamlit through Phase E, evaluate after.
6. **EA mode for synth.** Deriv doesn't have EAs in the MT5 sense. Equivalent for synth = scheduled webhook or always-on subscription. Need to design analogous "EA mode" for synth users who want passive journaling only.

---

## 12. First Concrete Tasks

This is what unblocks Phase A immediately:

1. Create a branch `feature/v2-architecture-doc` with this file under `docs/`.
2. Open issue: "Refactor BrokerInterface into MarketDataAdapter + ExecutionAdapter" (Phase A).
3. Open issue: "Introduce AssetClass enum and add columns to instruments/trades/signals" (Phase A).
4. Spin up notebook project `aegis-boom-research/` — separate repo for edge exploration. Outputs feed into Phase D, not into AegisTrader directly until edge is proven.
5. Review and decide on Open Questions #1–6 above.

---

*End of v2 architecture draft. Iterate.*
