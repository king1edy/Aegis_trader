# 🏛️ Aegis Trader System Architecture

> **Comprehensive Technical Documentation**
>
> Last Updated: February 2026 (v1.1)

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Breakdown](#component-breakdown)
   - [Entry Point](#1️⃣-entry-point---srcmainpy)
   - [Configuration](#2️⃣-configuration---srccoreconfgipy)
   - [Broker Execution Layer](#3️⃣-broker-execution-layer---srcexecution)
   - [Strategy Layer](#4️⃣-strategy-layer---srcstrategies)
   - [Risk Management](#5️⃣-risk-management---srcrisk)
   - [Notification System](#6️⃣-notification-system---srcnotifications)
   - [Database Layer](#7️⃣-database-layer---srcdatabase)
   - [Infrastructure](#8️⃣-infrastructure---docker-setup)
   - [Backtesting Engine](#9️⃣-backtesting-engine)
4. [Complete Data Flow](#complete-data-flow)
5. [Behavioral Safeguards](#behavioral-safeguards)
6. [Monitoring & Observability](#monitoring--observability)
7. [Key Touch Points Summary](#key-touch-points-summary)

---

## Overview

**Aegis Trader** is a professional-grade automated trading system for **XAUUSD (Gold)** using **MetaTrader 5**. The core philosophy is **removing human emotion from trading** - manual interventions are disabled by default.

### Key Principles

- **Full Automation**: The system should be trusted completely
- **Disciplined Execution**: Consistent strategy execution without emotional interference
- **Risk-First Design**: Multiple safeguards prevent destructive behavior
- **Multi-Timeframe Analysis**: 4H trend → 1H confirmation → 15M entry
- **Real-Time Notifications**: Telegram alerts for all trading events

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AEGIS TRADER SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   BROKER    │◄──►│   STRATEGY   │◄──►│    RISK     │◄──►│  DATABASE  │  │
│  │  EXECUTION  │    │    ENGINE    │    │ MANAGEMENT  │    │   LAYER    │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                   │                 │          │
│         │                  │                   │                 │          │
│         ▼                  ▼                   ▼                 ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       MAIN TRADING LOOP                             │   │
│  │  1. Check risk monitor (every tick)                                 │   │
│  │  2. Manage existing positions (every tick)                          │   │
│  │  3. Check for new 15M bar → Generate signals                        │   │
│  │  4. Validate signals against risk limits                            │   │
│  │  5. Execute approved signals                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NOTIFICATION LAYER                               │   │
│  │  📱 Telegram alerts for trades, signals, risks, and system events   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   BACKGROUND RISK MONITOR                            │   │
│  │  🛡️ Always-on drawdown protection (runs every 18 seconds)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1️⃣ Entry Point - `src/main.py`

The `TradingSystem` class orchestrates all components:

| Component | Role |
|-----------|------|
| `broker` | Connection to MetaTrader 5 |
| `data_manager` | Multi-timeframe price data caching |
| `indicator_calc` | Technical indicator calculations |
| `session_filter` | London/NY session filtering |
| `strategy` | MTFTR trading logic |
| `position_manager` | Manages open positions (TPs, trailing stops) |
| `position_sizer` | Calculates lot sizes based on risk % |
| `risk_checker` | Validates all risk limits |
| `risk_monitor` | **NEW** Background drawdown monitoring |
| `notifier` | **NEW** Telegram notification service |

#### Main Loop Flow

```
initialize() → run() → _trading_iteration() [loops forever]
                              │
                              ├─ Check risk_monitor.is_trading_blocked
                              │
                              ├─ PHASE 1: manage_positions() (every tick)
                              │
                              └─ PHASE 2: On new 15M bar:
                                    ├─ strategy.analyze() → TradingSignal
                                    ├─ notifier.notify_signal_generated()
                                    ├─ risk_checker.check_all_limits()
                                    ├─ position_sizer.calculate_lot_size()
                                    ├─ _execute_signal()
                                    └─ notifier.notify_trade_opened()
```

#### Key Methods

- `initialize()` - Sets up all components, connects to broker, starts risk monitor
- `run()` - Main trading loop with error handling
- `_trading_iteration()` - Single iteration of the loop
- `_execute_signal()` - Places trade and records to database
- `_on_risk_breach()` - **NEW** Handles risk threshold breaches
- `_emergency_close_all()` - **NEW** Emergency position liquidation
- `shutdown()` - Graceful cleanup with notification

---

### 2️⃣ Configuration - `src/core/config.py`

Centralized settings via **Pydantic Settings** (loaded from `.env`):

#### Application Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `app_env` | Environment (development/staging/production) | development |
| `debug` | Enable debug mode | False |
| `log_level` | Logging level | INFO |

#### MT5 Connection

| Setting | Description | Default |
|---------|-------------|---------|
| `mt5_login` | MT5 account number | Required |
| `mt5_password` | MT5 password | Required |
| `mt5_server` | Broker server name | Exness-MT5Trial |
| `broker_mode` | Connection mode (auto/direct/bridge/paper) | auto |
| `mt5_bridge_url` | API bridge URL for Docker | http://host.docker.internal:8001 |

#### Risk Management

| Setting | Description | Default |
|---------|-------------|---------|
| `max_risk_per_trade` | Risk per trade (decimal) | 0.01 (1%) |
| `max_daily_risk` | Maximum daily risk | 0.03 (3%) |
| `max_drawdown_percent` | Maximum drawdown before shutdown | 0.10 (10%) |
| `max_trades_per_day` | Daily trade limit | 3 |
| `min_trade_interval_minutes` | Cooldown between trades | 60 |
| `max_daily_loss_percent` | Max daily loss percentage | 0.03 (3%) |
| `min_margin_level` | Minimum margin level % | 200.0 |

#### Position Limits

| Setting | Description | Default |
|---------|-------------|---------|
| `max_open_positions` | Max concurrent positions | 2 |
| `max_daily_trades` | Max trades per day | 3 |
| `default_lot_size` | Default lot size | 0.01 |
| `max_lot_size` | Maximum lot size | 0.05 |

#### Session Filtering (GMT)

| Setting | Description | Default |
|---------|-------------|---------|
| `london_session_start` | London open | 07:00 |
| `london_session_end` | London close | 12:00 |
| `ny_session_start` | New York open | 13:00 |
| `ny_session_end` | New York close | 16:00 |

#### MTFTR Strategy Parameters

| Setting | Description | Default |
|---------|-------------|---------|
| `mtftr_enabled` | Enable MTFTR strategy | True |
| `mtftr_ema_200` | 4H trend EMA period | 200 |
| `mtftr_ema_50` | 1H confirmation EMA | 50 |
| `mtftr_ema_21` | 15M entry EMA | 21 |
| `mtftr_hull_55` | 4H Hull MA period | 55 |
| `mtftr_hull_34` | 1H Hull MA period | 34 |
| `mtftr_tp1_rr` | TP1 risk:reward | 1.0 |
| `mtftr_tp2_rr` | TP2 risk:reward | 2.0 |
| `mtftr_tp1_close_percent` | Close at TP1 | 0.50 (50%) |
| `mtftr_tp2_close_percent` | Close at TP2 | 0.30 (30%) |
| `mtftr_trail_percent` | Trail remainder | 0.20 (20%) |

#### Telegram Notifications *(NEW)*

| Setting | Description | Default |
|---------|-------------|---------|
| `telegram_enabled` | Enable Telegram notifications | False |
| `telegram_bot_token` | Telegram bot API token | "" |
| `telegram_chat_id` | Telegram chat/channel ID | "" |
| `notify_on_trade_open` | Notify when trades open | True |
| `notify_on_trade_close` | Notify when trades close | True |
| `notify_on_signal_generated` | Notify on new signals | True |
| `notify_on_daily_summary` | Send daily summary | True |
| `notify_on_error` | Notify on errors | True |
| `notify_on_drawdown_warning` | Notify on drawdown warnings | True |

#### Behavioral Safeguards

| Setting | Description | Default |
|---------|-------------|---------|
| `enable_manual_override` | Allow manual intervention | **False** |
| `max_consecutive_losses` | Losses before pause | 3 |
| `pause_duration_hours` | Pause length | 4 |
| `cooldown_after_loss_minutes` | Post-loss cooldown | 30 |

---

### 3️⃣ Broker Execution Layer - `src/execution/`

#### Broker Factory (`broker_factory.py`)

Auto-selects the appropriate broker connector:

```
broker_mode = "auto"
    │
    ├─ Windows + MT5 available → MT5Connector (direct)
    ├─ Docker/Linux → MT5APIClient (bridge to Windows host)
    └─ Fallback → PaperTradingBroker (simulation)
```

#### MT5 Connector (`mt5_connector.py`)

Defines the `BrokerInterface` abstract base class:

```python
class BrokerInterface(ABC):
    @abstractmethod
    async def connect() -> bool
    
    @abstractmethod
    async def disconnect() -> None
    
    @abstractmethod
    async def get_account_info() -> AccountInfo
    
    @abstractmethod
    async def get_symbol_info(symbol: str) -> SymbolInfo
    
    @abstractmethod
    async def get_current_tick(symbol: str) -> Tick
    
    @abstractmethod
    async def get_ohlcv(symbol: str, timeframe: str, count: int) -> pd.DataFrame
    
    @abstractmethod
    async def open_position(...) -> TradeResult
    
    @abstractmethod
    async def close_position(ticket: int) -> TradeResult
    
    @abstractmethod
    async def modify_position(ticket: int, sl: float, tp: float) -> TradeResult
    
    @abstractmethod
    async def get_open_positions() -> List[Position]
```

#### Data Classes

| Class | Purpose |
|-------|---------|
| `OrderDirection` | BUY / SELL enum |
| `SymbolInfo` | Symbol details (digits, spread, lot sizes, tick value) |
| `AccountInfo` | Account state (balance, equity, margin, leverage) |
| `Position` | Open position details |
| `Tick` | Current bid/ask prices |
| `TradeResult` | Execution result (success, ticket, price, error) |

#### Connector Implementations

| Connector | File | Use Case |
|-----------|------|----------|
| `MT5Connector` | `mt5_connector.py` | Direct Windows MT5 connection |
| `MT5APIClient` | `mt5_api_client.py` | HTTP client to bridge server |
| `MT5APIBridge` | `mt5_api_bridge.py` | FastAPI server exposing MT5 |
| `PaperTradingBroker` | `paper_broker.py` | Simulation mode |

---

### 4️⃣ Strategy Layer - `src/strategies/`

#### MTFTR Strategy (`mtftr.py`)

**Multi-Timeframe Trend Rider** - the core trading logic:

```
┌────────────────────────────────────────────────────────────┐
│                    MTFTR ANALYSIS FLOW                      │
├────────────────────────────────────────────────────────────┤
│  4H Timeframe  │  Trend Direction (EMA200, Hull55)         │
│       ↓        │                                            │
│  1H Timeframe  │  Trend Confirmation (EMA50, Hull34)       │
│       ↓        │                                            │
│  15M Timeframe │  Entry Trigger (EMA21, candlestick        │
│                │  patterns, structure breaks)               │
└────────────────────────────────────────────────────────────┘
```

##### Entry Methods

| Method | Description |
|--------|-------------|
| **Method A** | EMA bounce with reversal candle (engulfing, pin bar) |
| **Method B** | Structure break above/below swing points |

##### Exit Strategy

| Level | Action |
|-------|--------|
| TP1 (1:1 R:R) | Close 50%, move SL to breakeven |
| TP2 (2:1 R:R) | Close 30% |
| Trail (20%) | Trail on 1H EMA50, exit on Hull flip |
| Time Limit | Close all after 8 hours if no TP hit |

##### Configuration Class

```python
@dataclass
class MTFTRConfig(StrategyConfig):
    # Indicator periods
    ema_200: int = 200
    ema_50: int = 50
    ema_21: int = 21
    hull_55: int = 55
    hull_34: int = 34
    rsi_period: int = 14
    atr_period: int = 14
    
    # Risk parameters
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp1_close_percent: float = 0.50
    tp2_close_percent: float = 0.30
    trail_percent: float = 0.20
    
    # Entry filters
    min_rsi_long: float = 40.0
    max_rsi_long: float = 55.0
    min_rsi_short: float = 45.0
    max_rsi_short: float = 60.0
```

#### Data Manager (`data_manager.py`)

Manages price data across multiple timeframes:

| Feature | Description |
|---------|-------------|
| **Caching** | TTL-based cache to avoid repeated broker calls |
| **New Bar Detection** | Prevents duplicate signals per timeframe |
| **Freshness Validation** | Raises `StaleDataError` if data too old |
| **Auto Invalidation** | Clears cache when TTL expires |

```python
manager = MultiTimeframeDataManager(broker, cache_ttl_seconds=60)
df = await manager.get_data("XAUUSD", "H4", count=250)
is_new = await manager.is_new_bar("XAUUSD", "M15")
```

#### Indicator Calculator (`indicators.py`)

Calculates all technical indicators:

| Indicator | Method | Description |
|-----------|--------|-------------|
| EMA | `calculate_ema()` | Exponential Moving Average |
| Hull MA | `calculate_hull_ma()` | Hull Moving Average (custom) |
| WMA | `calculate_wma()` | Weighted Moving Average |
| RSI | `calculate_rsi()` | Relative Strength Index |
| ATR | `calculate_atr()` | Average True Range |
| Swings | `find_swings()` | Swing high/low detection |

All calculations verified against MQL5 reference implementation.

#### Pattern Recognizer (`patterns.py`)

Candlestick reversal patterns for entry confirmation:

| Pattern | Method | Criteria |
|---------|--------|----------|
| Bullish Engulfing | `is_bullish_engulfing()` | Bullish candle engulfs prior bearish |
| Bearish Engulfing | `is_bearish_engulfing()` | Bearish candle engulfs prior bullish |
| Hammer/Pin Bar | `is_hammer()` | Long lower wick ≥2× body, body in upper 60% |
| Shooting Star | `is_shooting_star()` | Long upper wick ≥2× body, body in lower 40% |

#### Session Filter (`filters/session_filter.py`)

Ensures trades only occur during high-liquidity sessions:

| Session | Hours (GMT) |
|---------|-------------|
| London | 07:00 - 12:00 |
| New York | 13:00 - 16:00 |

```python
filter = SessionFilter(london_start="07:00", london_end="12:00", ...)
is_tradeable, session = await filter.is_tradeable_time()
# Returns (True, "london") or (False, "outside_hours")
```

#### Position Manager (`position_manager.py`)

Manages open positions with state machine:

```
INITIAL → TP1_HIT → TP2_HIT → TRAILING → CLOSED
    │         │          │         │
    │         │          │         └─ Exit on Hull flip
    │         │          └─ Update trailing SL on 1H bars
    │         └─ Close 50%, SL → Breakeven
    └─ Check time limit (8 hours)
```

**NEW**: Now accepts optional `notifier` parameter to send trade close notifications.

| State | TP1 Action | TP2 Action | Trail Action |
|-------|------------|------------|--------------|
| `initial` | Close 50%, SL→BE | - | - |
| `tp1_hit` | - | Close 30% | - |
| `tp2_hit` | - | - | Update SL on 1H bars |
| `trailing` | - | - | Exit on Hull flip |

#### Base Strategy (`base_strategy.py`)

Abstract base class for all strategies:

```python
@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    direction: OrderDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_final: Optional[float]
    confidence: float  # 0.0 to 1.0
    reason: str
    market_context: Dict[str, Any]
    strategy_data: Dict[str, Any]
```

---

### 5️⃣ Risk Management - `src/risk/`

#### Risk Checker (`risk_checker.py`)

Validates signals against **ALL** limits before execution:

| Check | Method | Limit |
|-------|--------|-------|
| Trading Pause | `_check_trading_pause()` | Active pause blocks trading |
| Daily Trades | `_check_daily_trade_limit()` | Max 3 per day |
| Open Positions | `_check_open_positions_limit()` | Max 2 concurrent |
| Daily Loss | `_check_daily_loss_limit()` | Max 3% loss |
| Drawdown | `_check_drawdown_limit()` | Max 10% |
| Consecutive Losses | `_check_consecutive_losses()` | 3 → triggers 4hr pause |
| Margin | `_check_margin_availability()` | Min 200% level |

**NEW**: Now accepts optional `notifier` parameter to send risk warnings.

```python
can_trade, reason = await risk_checker.check_all_limits(signal, account)
if not can_trade:
    logger.info(f"Signal rejected: {reason}")
    await notifier.notify_signal_rejected(...)
```

#### Position Sizer (`position_sizer.py`)

Calculates lot size using fixed-fractional risk:

```
risk_amount = account_balance × risk_percent (1%)
sl_distance_pips = |entry - stop_loss| / pip_size
lot_size = risk_amount / (sl_distance_pips × tick_value)
→ Normalize to broker's min/max/step
```

```python
@dataclass
class PositionSizeResult:
    lot_size: float
    risk_amount: float
    sl_distance_pips: float
    risk_percent_actual: float
    normalized_lot: float
```

#### Risk Monitor (`risk_monitor.py`) *(NEW)*

**Background always-on monitoring** for drawdown protection. Runs independently every ~18 seconds.

```python
@dataclass
class RiskMonitorConfig:
    check_interval_seconds: float = 18.0
    max_balance_drawdown_pct: float = 10.0  # Max % from peak balance
    max_equity_drawdown_pct: float = 15.0   # Max % from peak equity
    max_daily_loss_pct: float = 5.0         # Max daily loss %
    max_weekly_loss_pct: float = 10.0       # Max weekly loss %
    min_margin_level_pct: float = 150.0     # Warning level
    emergency_margin_level_pct: float = 100.0  # Emergency close all
```

##### Risk State Tracking

```python
@dataclass 
class RiskState:
    peak_balance: float
    peak_equity: float
    daily_start_balance: float
    weekly_start_balance: float
    current_balance: float
    current_equity: float
    margin_level: float
    balance_drawdown_pct: float
    equity_drawdown_pct: float
    daily_loss_pct: float
    weekly_loss_pct: float
    is_trading_blocked: bool
    block_reason: Optional[str]
```

##### Features

| Feature | Description |
|---------|-------------|
| **Peak Tracking** | Tracks highest balance/equity reached |
| **Daily Reset** | Resets daily start balance at midnight UTC |
| **Weekly Reset** | Resets weekly start balance on Monday |
| **Trading Block** | Sets `is_trading_blocked = True` on breach |
| **Emergency Close** | Triggers `on_emergency_close` callback at critical levels |
| **Notifications** | Calls `on_threshold_breach` callback for alerts |

##### Enforcement Priority

1. Emergency margin (<100%) → **Close all positions immediately**
2. Balance drawdown (>10%) → Block trading
3. Equity drawdown (>15%) → Block trading
4. Daily loss (>5%) → Block trading
5. Weekly loss (>10%) → Block trading
6. Margin warning (<150%) → Block trading

---

### 6️⃣ Notification System - `src/notifications/` *(NEW)*

Centralized notification system for real-time trading alerts via Telegram.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NOTIFICATION SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌────────────────────────┐     │
│  │  NotificationService │───►│   MessageFormatter     │     │
│  │     (Singleton)      │    │   (Rich Emoji Style)   │     │
│  └──────────────────────┘    └────────────────────────┘     │
│            │                                                 │
│            ▼                                                 │
│  ┌──────────────────────┐                                   │
│  │   TelegramNotifier   │                                   │
│  │   (python-telegram-  │                                   │
│  │       bot)           │                                   │
│  └──────────────────────┘                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### NotificationService (`service.py`)

Central facade for all notification channels (singleton pattern):

```python
from src.notifications import NotificationService

# Get singleton instance
notifier = NotificationService.get_instance(settings)

# Trade notifications
await notifier.notify_trade_opened(...)
await notifier.notify_trade_closed(...)

# Signal notifications
await notifier.notify_signal_generated(...)
await notifier.notify_signal_rejected(...)

# Risk notifications
await notifier.notify_risk_warning(...)

# System notifications
await notifier.notify_system_status("started", {...})
await notifier.notify_daily_summary(...)
```

#### TelegramNotifier (`telegram.py`)

Async Telegram sender using `python-telegram-bot`:

| Method | Trigger |
|--------|---------|
| `notify_trade_opened()` | Trade opened |
| `notify_trade_closed()` | Trade closed (full or partial) |
| `notify_signal_generated()` | Signal generated |
| `notify_signal_rejected()` | Signal rejected by risk checks |
| `notify_risk_warning()` | Drawdown/margin warning |
| `notify_system_status()` | System start/stop/reconnect |
| `notify_daily_summary()` | End-of-day summary |
| `send_custom_message()` | Custom messages |

#### MessageFormatter (`message_formatter.py`)

Rich emoji-styled Markdown messages:

```
🟢 *BUY XAUUSD*

💵 *Entry:* `2050.50`
🛡️ *SL:* `2040.00`
🎯 *TP1:* `2075.00`
🎯 *TP2:* `2100.00`

📦 *Size:* `0.10` lots
📊 *R:R:* `1:2.5`
🎯 *Confidence:* `85%`

💡 _EMA bounce with bullish engulfing_

🕐 _2026-02-08 14:30:00 UTC_
```

##### Emoji Reference

| Emoji | Usage |
|-------|-------|
| 🟢 | BUY direction |
| 🔴 | SELL direction |
| 💰 | Profit |
| 💸 | Loss |
| ⚠️ | Warning |
| 📊 | Signal/Chart |
| 🚀 | System started |
| 🛑 | System stopped |
| 🔄 | Reconnected |
| 🎯 | Target/Confidence |
| 🛡️ | Stop loss/Protection |
| 💵 | Money/Entry |

---

### 7️⃣ Database Layer - `src/database/`

#### Database Stack

- **PostgreSQL 15** with **TimescaleDB** extension
- **Async SQLAlchemy** for non-blocking queries
- **UUID primary keys** for distributed safety

#### Models (`models.py`)

| Model | Purpose |
|-------|---------|
| `Trade` | Complete trade lifecycle (signal → close) |
| `PartialClose` | Records partial exits (TP1, TP2) |
| `TradeModification` | Audit trail for SL/TP changes |
| `Signal` | All generated signals (executed or not) |
| `DailyPerformance` | Daily P&L aggregates |
| `AccountSnapshot` | Periodic account state |
| `TradingPause` | Trading pause events |
| `SystemEvent` | General system events |

##### Trade Model (Core)

```python
class Trade(Base):
    # Identification
    id: UUID
    ticket: int  # Broker ticket
    symbol: str
    order_type: OrderType  # BUY/SELL
    status: TradeStatus  # PENDING→OPEN→CLOSED
    
    # Signal info
    signal_source: SignalSource
    strategy_name: str
    signal_time: datetime
    
    # Entry
    entry_price: Decimal
    entry_time: datetime
    lot_size: Decimal
    
    # Risk levels
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal
    trailing_stop: Decimal
    position_state: str  # "initial"→"tp1_hit"→"tp2_hit"→"trailing"
    
    # Results
    exit_price: Decimal
    exit_time: datetime
    profit_loss: Decimal
    outcome: TradeOutcome  # WIN/LOSS/BREAKEVEN
    
    # Context (JSONB)
    market_context: dict
    strategy_data: dict
```

##### Trade Status Flow

```
PENDING → OPEN → PARTIALLY_CLOSED → CLOSED
              ↘        CANCELLED / REJECTED
```

#### Repository (`repository.py`)

Clean data access layer:

| Repository | Methods |
|------------|---------|
| `TradeRepository` | `create()`, `get_by_id()`, `get_by_ticket()`, `get_open_trades()`, `get_recent_trades()`, `count_today_trades()` |
| `SignalRepository` | `save()`, `get_unexecuted()`, `get_by_date_range()` |
| `PerformanceRepository` | `get_daily_stats()`, `get_weekly_stats()`, `calculate_drawdown()` |
| `SystemRepository` | `get_active_pause()`, `create_pause()`, `log_event()` |

```python
async with get_session() as session:
    trade_repo = TradeRepository(session)
    open_trades = await trade_repo.get_open_trades(symbol="XAUUSD")
```

---

### 8️⃣ Infrastructure - Docker Setup

#### Services (`docker-compose.yml`)

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER SERVICES                          │
├─────────────────────────────────────────────────────────────┤
│  trading_app   │  Main Python trading application           │
│  postgres      │  TimescaleDB for time-series data          │
│  redis         │  Caching and real-time state               │
│  grafana       │  Monitoring dashboards (:3000)             │
│  prometheus    │  Metrics collection (:9090)                │
└─────────────────────────────────────────────────────────────┘
```

#### Network Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| `BROKER_MODE` | bridge | Connect via HTTP to Windows host |
| `MT5_BRIDGE_URL` | http://host.docker.internal:8001 | MT5 API bridge |
| `DATABASE_URL` | postgresql://...@postgres:5432/... | Container networking |
| `REDIS_URL` | redis://redis:6379/0 | Container networking |

#### Volumes

| Volume | Purpose |
|--------|---------|
| `postgres_data` | Persistent trade database |
| `redis_data` | Redis persistence |
| `grafana_data` | Dashboard configs |
| `prometheus_data` | Metrics history |
| `./src` | Live code mounting |
| `./logs` | Application logs |
| `./data` | Historical data |

#### Running MT5 Bridge (Windows Host)

Since MT5 only runs on Windows, run the bridge on your Windows machine:

```bash
python -m src.execution.mt5_api_bridge
```

This exposes MT5 functionality via HTTP on port 8001.

---

### 9️⃣ Backtesting Engine - `src/backtesting/engine.py`

Event-driven backtester for strategy validation:

#### Features

- Multi-timeframe data simulation
- Realistic spread and slippage modeling
- Position management with partial closes
- Comprehensive trade logging and metrics

#### Configuration

```python
@dataclass
class BacktestConfig:
    # Account
    initial_balance: float = 10000.0
    leverage: int = 100
    
    # Execution (realistic)
    spread_pips: float = 1.5
    slippage_pips: float = 0.3
    commission_per_lot: float = 7.0
    
    # Position management (optimized)
    tp1_close_percent: float = 0.40  # 40% at TP1
    tp2_close_percent: float = 0.30  # 30% at TP2
    trail_percent: float = 0.30      # Trail 30%
    
    # Risk (more aggressive for backtesting)
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_risk: float = 0.06      # 6%
    max_drawdown: float = 0.20        # 20%
    max_trades_per_day: int = 6
    max_open_trades: int = 3
```

#### Backtesting Files

| File | Purpose |
|------|---------|
| `engine.py` | Core backtesting engine |
| `run_backtest.py` | CLI runner script |
| `data_provider.py` | Historical data loading |
| `strategy_simulator.py` | Strategy simulation wrapper |
| `test_data_generator.py` | Synthetic data generation |

#### Running a Backtest

```bash
python -m src.backtesting.run_backtest
```

Output files:
- `backtest_trades_XAUUSD_{dates}.csv` - All trades
- `backtest_equity_XAUUSD_{dates}.csv` - Equity curve

---

## Complete Data Flow

### Trading Iteration Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRADING ITERATION FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

0. RISK MONITOR CHECK (First)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  if risk_monitor.is_trading_blocked:                                │
   │      logger.warning("Trading blocked", reason=block_reason)         │
   │      await asyncio.sleep(60)  # Wait longer when blocked            │
   │      return                                                         │
   └─────────────────────────────────────────────────────────────────────┘

1. POSITION MANAGEMENT (Every Tick)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  PositionManager.manage_positions()                                 │
   │    ├─ For each open trade:                                          │
   │    │   ├─ Get current tick (bid/ask)                               │
   │    │   ├─ Check time limit (8 hours)                               │
   │    │   ├─ Check TP1 hit → partial close 50%, SL → BE               │
   │    │   │   └─ notifier.notify_trade_closed(is_partial=True)        │
   │    │   ├─ Check TP2 hit → partial close 30%                        │
   │    │   │   └─ notifier.notify_trade_closed(is_partial=True)        │
   │    │   └─ Update trailing stop on 1H bars                          │
   └─────────────────────────────────────────────────────────────────────┘

2. SIGNAL GENERATION (On New 15M Bar)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  DataManager.is_new_bar("XAUUSD", "M15")                           │
   │    ↓                                                                │
   │  MTFTRStrategy.analyze("XAUUSD")                                   │
   │    ├─ SessionFilter.is_tradeable_time() → London/NY only           │
   │    ├─ DataManager.get_data() for H4, H1, M15                       │
   │    ├─ IndicatorCalculator.calculate_all()                          │
   │    │   ├─ EMA 200/50/21                                            │
   │    │   ├─ Hull MA 55/34                                            │
   │    │   ├─ RSI, ATR                                                 │
   │    │   └─ Swing highs/lows                                         │
   │    ├─ Analyze 4H trend (EMA200, Hull55)                            │
   │    ├─ Confirm 1H alignment (EMA50, Hull34)                         │
   │    ├─ Check 15M entry trigger                                      │
   │    │   ├─ Method A: EMA bounce + reversal candle                   │
   │    │   └─ Method B: Structure break                                │
   │    ├─ PatternRecognizer (engulfing, pin bar, etc.)                 │
   │    └─ Calculate SL/TP levels → TradingSignal                       │
   │                                                                     │
   │  notifier.notify_signal_generated(...)  ← NEW                      │
   └─────────────────────────────────────────────────────────────────────┘

3. RISK VALIDATION
   ┌─────────────────────────────────────────────────────────────────────┐
   │  RiskChecker.check_all_limits(signal, account)                     │
   │    ├─ Check trading pause                                          │
   │    ├─ Check daily trade limit (3)                                  │
   │    ├─ Check open positions (2 max)                                 │
   │    ├─ Check daily loss (3% max)                                    │
   │    ├─ Check drawdown (10% max)                                     │
   │    ├─ Check consecutive losses (3 → pause)                         │
   │    └─ Check margin availability (200%+)                            │
   │  → Returns (True, None) or (False, "reason")                       │
   │                                                                     │
   │  if not can_trade:                                                 │
   │      notifier.notify_signal_rejected(...)  ← NEW                   │
   └─────────────────────────────────────────────────────────────────────┘

4. POSITION SIZING
   ┌─────────────────────────────────────────────────────────────────────┐
   │  PositionSizer.calculate_lot_size()                                │
   │    ├─ risk_amount = balance × 1%                                   │
   │    ├─ sl_pips = |entry - sl| / point                               │
   │    ├─ lot_size = risk_amount / (sl_pips × tick_value)              │
   │    └─ Normalize to broker limits (min/max/step)                    │
   └─────────────────────────────────────────────────────────────────────┘

5. EXECUTION
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Broker.open_position(symbol, direction, volume, sl, tp, comment)  │
   │    ├─ MT5 places market order                                      │
   │    └─ Returns TradeResult (success, ticket, price)                 │
   │                                                                     │
   │  Save to Database:                                                  │
   │    ├─ Trade record (ticket, entry, SL, TPs, state)                 │
   │    └─ Signal record (was_executed=True)                            │
   │                                                                     │
   │  notifier.notify_trade_opened(...)  ← NEW                          │
   └─────────────────────────────────────────────────────────────────────┘

BACKGROUND: RISK MONITOR (Every ~18 seconds)
   ┌─────────────────────────────────────────────────────────────────────┐
   │  RiskMonitor._check_risk_state()                                   │
   │    ├─ Get current balance/equity/margin                            │
   │    ├─ Update peaks (if higher)                                     │
   │    ├─ Check daily/weekly period resets                             │
   │    ├─ Calculate drawdowns                                          │
   │    └─ Enforce limits:                                              │
   │        ├─ Emergency margin (<100%) → close all, block              │
   │        ├─ Balance DD (>10%) → block                                │
   │        ├─ Equity DD (>15%) → block                                 │
   │        ├─ Daily loss (>5%) → block                                 │
   │        ├─ Weekly loss (>10%) → block                               │
   │        └─ Margin warning (<150%) → block                           │
   │                                                                     │
   │  on breach:                                                        │
   │    ├─ _on_risk_breach(reason, state)                              │
   │    │   └─ notifier.notify_system_status("risk_breach", {...})     │
   │    └─ _emergency_close_all(reason)  [if emergency]                │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## Behavioral Safeguards

| Safeguard | Implementation | Default |
|-----------|----------------|---------|
| Manual Override Disabled | `enable_manual_override = False` | **Disabled** |
| Consecutive Loss Pause | After 3 losses → 4 hour pause | 3 losses |
| Daily Loss Limit | Stops trading at threshold | 3% |
| Drawdown Protection | Pauses at max drawdown | 10% |
| Trade Cooldown | Minimum time between trades | 60 min |
| Session Filter | Only trade during active sessions | London/NY |
| Max Positions | Limits concurrent exposure | 2 |
| Max Daily Trades | Prevents overtrading | 3 |
| Margin Check | Ensures sufficient margin | 200%+ |
| **Background Risk Monitor** | Always-on drawdown protection | ~18s checks |
| **Emergency Close All** | Closes positions at critical margin | <100% margin |
| **Telegram Alerts** | Real-time notifications | Configurable |

---

## Monitoring & Observability

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Trading dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| PostgreSQL | localhost:5432 | Trade database |
| Redis | localhost:6379 | State caching |
| **Telegram** | Via Bot | Real-time alerts |

### Logging

Structured logging via `structlog`:

```python
logger.info(
    "Trade opened successfully",
    ticket=result.ticket,
    direction=signal.direction.value,
    entry=result.price,
    sl=signal.stop_loss,
    lot_size=lot_size
)
```

Log files: `./logs/`

### Notification Events

| Event | When | Content |
|-------|------|---------|
| System Started | On initialization | Version, balance, broker mode |
| System Stopped | On shutdown | Graceful stop message |
| Trade Opened | Trade executed | Entry, SL, TP, lot size, R:R |
| Trade Closed | Position closed | Entry, exit, P/L, duration |
| Partial Close | TP1/TP2 hit | Partial %, realized P/L |
| Signal Generated | Strategy signal | Direction, entry, SL, TP, confidence |
| Signal Rejected | Risk check failed | Rejection reason |
| Risk Breach | Threshold exceeded | Drawdown %, margin level |
| Daily Summary | End of day | Total trades, win rate, net P/L |

---

## Key Touch Points Summary

| # | Component | File | Purpose |
|---|-----------|------|---------|
| 1 | Entry Point | `src/main.py` | `TradingSystem.run()` |
| 2 | Config | `src/core/config.py` | Pydantic Settings from `.env` |
| 3 | Broker Factory | `src/execution/broker_factory.py` | Auto-select MT5/Bridge/Paper |
| 4 | Data Manager | `src/strategies/data_manager.py` | Cached multi-timeframe data |
| 5 | Indicators | `src/strategies/indicators.py` | EMA, Hull, RSI, ATR, Swings |
| 6 | Strategy | `src/strategies/mtftr.py` | Trend analysis + entry triggers |
| 7 | Session Filter | `src/strategies/filters/session_filter.py` | London/NY only |
| 8 | Patterns | `src/strategies/patterns.py` | Candlestick patterns |
| 9 | Risk Checker | `src/risk/risk_checker.py` | All limits validated |
| 10 | **Risk Monitor** | `src/risk/risk_monitor.py` | **Background drawdown protection** |
| 11 | Position Sizer | `src/risk/position_sizer.py` | 1% risk calculation |
| 12 | MT5 Connector | `src/execution/mt5_connector.py` | Place orders |
| 13 | Position Manager | `src/strategies/position_manager.py` | TPs, trailing, exits |
| 14 | Database | `src/database/repository.py` | Trade persistence |
| 15 | Models | `src/database/models.py` | Data structures |
| 16 | **Notification Service** | `src/notifications/service.py` | **Central notification facade** |
| 17 | **Telegram Notifier** | `src/notifications/telegram.py` | **Telegram message sending** |
| 18 | **Message Formatter** | `src/notifications/message_formatter.py` | **Rich emoji formatting** |
| 19 | Backtesting | `src/backtesting/engine.py` | Historical simulation |

---

## Quick Reference

### Start Trading (Docker)

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. On Windows host, start MT5 bridge
python -m src.execution.mt5_api_bridge

# 3. Check logs
docker-compose logs -f trading_app
```

### Run Backtest

```bash
python -m src.backtesting.run_backtest
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# MT5 Configuration
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=Exness-MT5Trial

# Risk Settings
MAX_RISK_PER_TRADE=0.01
ENABLE_MANUAL_OVERRIDE=false

# Telegram Notifications (NEW)
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-1001234567890
NOTIFY_ON_TRADE_OPEN=true
NOTIFY_ON_TRADE_CLOSE=true
NOTIFY_ON_SIGNAL_GENERATED=true
NOTIFY_ON_DAILY_SUMMARY=true
```

### Set Up Telegram Bot

1. Create a bot via [@BotFather](https://t.me/botfather)
2. Get the bot token
3. Create a group/channel and add the bot
4. Get the chat ID (use [@userinfobot](https://t.me/userinfobot))
5. Configure in `.env`

---

*Document generated for Aegis Trader v1.1.0*
*Last updated: February 2026*
