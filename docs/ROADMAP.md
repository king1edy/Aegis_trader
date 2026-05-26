# Roadmap

## Now (Shipped)

Current shipped platform capabilities include:

- multi-tenant auth and API key model
- tiered subscription and rate-limit framework
- EA webhook ingestion and CSV/DB persistence
- TradingView webhook ingestion path
- journal analytics API and dashboard UI route
- risk checking, risk monitoring, and trading pause persistence
- settings API for per-user risk/strategy/notification preferences
- Dockerized deployment with Alembic migrations

## Next 1-2 Quarters

### Product and Platform

- complete end-to-end journal/dashboard regression suite
- harden TradingView open/close lifecycle paths and idempotency checks
- strengthen tier upgrade and billing lifecycle integration
- improve docs and operational runbooks for onboarding at scale

### Strategy and Analytics

- introduce explicit signal scorer module with weighted factor transparency
- add richer setup attribution and cohort analytics
- expand multi-symbol workflow beyond current defaults

### Operations

- improve observability dashboards and alerting playbooks
- formalize deployment validation and rollback checklists

## Vision (12+ Months)

- ML-assisted signal quality scoring and feedback loops
- mobile companion experience for monitoring and control
- social or team collaboration features with strict governance boundaries
- broader broker ecosystem integrations beyond MT5-first footprint
- advanced backtester UX and strategy experimentation tooling
- packaged rule templates for prop-firm style constraints and reporting

## Prioritization Principles

- preserve risk-first behavior over growth-only feature pressure
- avoid cross-module complexity spikes without testability gains
- keep tenant isolation and auditability as non-negotiable constraints

## Success Signals

- faster activation to first logged trade
- improved retention for journal-active users
- increased tier upgrades with low support burden
- reduced operational incidents per release

## Source Citations

- [src/main.py](../src/main.py#L142)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/journal/router.py](../src/journal/router.py#L70)
- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L47)
- [docker-compose.yml](../docker-compose.yml#L1)
- [scripts/entrypoint.sh](../scripts/entrypoint.sh#L34)

## Related Docs

- `docs/PRD.md`
- `docs/VISION.md`
- `docs/FEATURES.md`
