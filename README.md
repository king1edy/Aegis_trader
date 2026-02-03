# Trading Automation System

A professional-grade automated trading system for XAUUSD using MetaTrader 5.

## 🎯 Purpose

This system is designed to remove manual intervention from trading entirely. The primary goal is consistent, disciplined execution of proven strategies without emotional interference.

**Key Principle**: The system should be trusted completely. Manual overrides are disabled by default because historical data shows that automated execution consistently outperforms manual interventions.

## 📁 Project Structure

```
trading_system/
├── docker-compose.yml      # Container orchestration
├── Dockerfile              # Application container
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── prometheus.yml          # Metrics configuration
│
├── src/
│   ├── main.py            # Application entry point
│   ├── core/              # Configuration, logging, exceptions
│   ├── database/          # Models and data access
│   ├── execution/         # Broker connectivity (MT5)
│   ├── strategies/        # Trading strategies
│   ├── risk/              # Risk management
│   ├── data/              # Market data handling
│   ├── api/               # REST API & webhooks
│   ├── notifications/     # Telegram alerts
│   └── backtesting/       # Strategy validation
│
├── scripts/               # Database & utility scripts
├── tests/                 # Test suites
└── dashboards/            # Grafana dashboards
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- MetaTrader 5 (Windows only for live trading)

### Setup

1. **Clone and configure:**
   ```bash
   git clone <repository>
   cd trading_system
   cp .env.example .env
   # Edit .env with your MT5 credentials and settings
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **Check status:**
   ```bash
   docker-compose logs -f trading_app
   ```

### Access Points

- **Grafana Dashboard**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **API (when implemented)**: http://localhost:8000

## ⚙️ Configuration

All configuration is via environment variables. See `.env.example` for the complete list.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MT5_LOGIN` | MT5 account number | Required |
| `MT5_PASSWORD` | MT5 password | Required |
| `MT5_SERVER` | Broker server | Exness-MT5Trial |
| `MAX_RISK_PER_TRADE` | Risk per trade (decimal) | 0.01 (1%) |
| `MAX_DAILY_RISK` | Max daily risk | 0.03 (3%) |
| `MAX_TRADES_PER_DAY` | Trade limit | 3 |
| `ENABLE_MANUAL_OVERRIDE` | Allow manual intervention | false |

## 🛡️ Behavioral Safeguards

The system includes multiple safeguards to prevent destructive manual intervention:

1. **Manual Override Disabled**: By default, no manual trades or modifications
2. **Consecutive Loss Pause**: Trading pauses after N consecutive losses
3. **Daily Loss Limit**: Stops trading when daily loss exceeds threshold
4. **Drawdown Protection**: Pauses at maximum drawdown level
5. **Trade Cooldown**: Minimum time between trades

## 📊 Database Schema

The system uses PostgreSQL with TimescaleDB for efficient time-series data:

- `trades`: Complete trade lifecycle records
- `signals`: All generated signals (executed or not)
- `account_snapshots`: Periodic account state captures
- `daily_performance`: Aggregated daily metrics
- `system_events`: Audit trail
- `trading_pauses`: When and why trading was paused

## 🔧 Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally (requires running PostgreSQL and Redis)
python -m src.main
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Formatting
black src/ tests/

# Linting
ruff src/ tests/

# Type checking
mypy src/
```

## 📈 Monitoring

### Grafana Dashboard

The included Grafana dashboard shows:
- Account balance/equity over time
- Win rate and profit factor
- Trade distribution by strategy
- Drawdown tracking
- System health metrics

### Alerts

Configure Telegram notifications for:
- Trade opens/closes
- Daily summaries
- Drawdown warnings
- System errors

## 🚧 Development Phases

### Phase 1: Foundation ✅
- Docker environment
- Database models
- MT5 connection
- Logging infrastructure

### Phase 2: Strategy Engine (Next)
- Indicator calculations
- Hull Suite strategy
- Session filtering
- Signal generation

### Phase 3: Risk Management
- Position sizing (ATR-based)
- Daily limits
- Drawdown protection
- Behavioral safeguards

### Phase 4: Execution & Monitoring
- Order execution engine
- Trade lifecycle management
- Telegram notifications
- Grafana dashboard

### Phase 5: Backtesting
- Historical data pipeline
- Backtest engine
- Walk-forward optimization
- Performance reporting

## ⚠️ Important Notes

1. **Trust the System**: Your historical data shows EAs outperform manual trading
2. **Demo First**: Always test thoroughly on demo before live deployment
3. **Risk Management**: Never exceed configured risk limits
4. **Logs**: Review logs regularly to understand system behavior
5. **Backtest**: Validate any strategy changes with comprehensive backtesting

## 📝 License

Private - For personal use only.

---

*Remember: The best trade is often no trade. Let the system wait for high-probability setups.*
