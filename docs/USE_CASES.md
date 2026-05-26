# Use Cases

Detailed persona scenarios for Aegis Trader.

## Persona 1: Retail Trader

### Scenario 1.1: Install EA and view live journal

- Preconditions:
  - User account exists and user is logged in
  - User has generated an API key
  - MT5 terminal allows WebRequest to Aegis host
- Steps:
  1. Configure EA with FastAPI URL and API key
  2. Attach EA to chart
  3. Open and close a test trade
  4. Open dashboard at root route
  5. Inspect journal trade list and summary widgets
- Expected outcome:
  - Trade event accepted at POST /trade
  - Trade appears in journal endpoints and dashboard UI
- Observable artifacts:
  - CSV row in logs/MTFTR_TradeLog.csv
  - Trade row in trades table
  - Updated summary payload from /trades/summary

### Scenario 1.2: Tag setup and analyze win rate by setup

- Preconditions:
  - User has at least 10 closed trades
- Steps:
  1. Create setup tags using POST /api/journal/tags
  2. Patch selected trades with setup_tag
  3. Query /api/journal/analysis/setups
- Expected outcome:
  - Setup-level performance breakdown is returned
- Observable artifacts:
  - Rows in setup_tags table
  - Updated trades.setup_tag values

### Scenario 1.3: Hit daily loss limit and observe pause

- Preconditions:
  - Risk limits configured in user settings
- Steps:
  1. Generate losses until daily threshold is breached
  2. Attempt another trade signal
- Expected outcome:
  - Trade is rejected by risk controls
  - Trading pause entry is created
- Observable artifacts:
  - trading_pauses row with reason daily_loss_limit
  - Risk warning notification if enabled

## Persona 2: EA Developer

### Scenario 2.1: Generate and rotate API keys

- Preconditions:
  - Valid JWT session
- Steps:
  1. Create key via POST /api/auth/api-keys
  2. Configure key in EA headers
  3. Revoke stale key via DELETE /api/auth/api-keys/{id}
- Expected outcome:
  - New key works immediately
  - Revoked key no longer authenticates webhook calls
- Observable artifacts:
  - api_keys rows and activation flags

### Scenario 2.2: Add strategy variant metadata

- Preconditions:
  - Existing trade ingestion is functional
- Steps:
  1. Add strategy_name and strategy_data fields in EA payload
  2. Send events to /trade
  3. Filter analytics by strategy name
- Expected outcome:
  - New variant is segmentable in analytics and repository queries
- Observable artifacts:
  - strategy_name and strategy_data stored on trades

## Persona 3: TradingView Signal User

### Scenario 3.1: Configure Pine alert to webhook endpoint

- Preconditions:
  - TradingView alerting enabled
  - Secret format aligned with backend schema
- Steps:
  1. Create TradingView alert JSON payload
  2. Set webhook URL to /webhook/tradingview
  3. Trigger test alert
- Expected outcome:
  - Webhook returns created response
  - Signal and/or trade attribution records include TRADINGVIEW source
- Observable artifacts:
  - webhook request logs
  - signal source markers in database rows

## Persona 4: Operator / Admin

### Scenario 4.1: Deploy stack and verify health

- Preconditions:
  - Docker and docker-compose available
- Steps:
  1. Launch stack with docker-compose up
  2. Check migrations applied by entrypoint
  3. Call /health
- Expected outcome:
  - Services become healthy
  - App accepts auth and webhook traffic
- Observable artifacts:
  - container logs
  - alembic version at head

### Scenario 4.2: Observe degraded mode when Redis is down

- Preconditions:
  - Platform running with middleware enabled
- Steps:
  1. Stop Redis
  2. Continue API requests
- Expected outcome:
  - API remains available
  - Rate limiting degrades gracefully
- Observable artifacts:
  - warning logs for Redis connectivity
  - successful endpoint responses

## Persona 5: Prop-Firm Candidate

### Scenario 5.1: Enforce strict challenge risk profile

- Preconditions:
  - User has Pro or Autopilot tier and settings access
- Steps:
  1. Configure strict drawdown and daily loss limits
  2. Enable pause-on-breach behavior
  3. Trade across sessions
- Expected outcome:
  - Rule violations auto-pause trading
  - Journal provides evidence for discipline tracking
- Observable artifacts:
  - user_settings threshold values
  - trading_pauses entries
  - daily performance records

## Cross-References

- [PRD.md](PRD.md)
- [FEATURES.md](FEATURES.md)
- [USER_GUIDE.md](USER_GUIDE.md)

## Source Citations

- [src/auth/router.py](../src/auth/router.py#L105)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/journal/router.py](../src/journal/router.py#L76)
- [src/settings/router.py](../src/settings/router.py#L59)
- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L47)
