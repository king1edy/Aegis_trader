# Aegis Trader Documentation

Aegis Trader is a multi-tenant MT5 trading platform with three tiers: Journal, Pro, and Autopilot. This documentation set is organized for four audiences: product stakeholders, end-user traders, developers, and operators.

## Start Here By Role

- Trader: [USER_GUIDE.md](USER_GUIDE.md)
- Product or business stakeholder: [VISION.md](VISION.md) then [PRD.md](PRD.md)
- Developer or contributor: [ARCHITECTURE.md](ARCHITECTURE.md), [API_REFERENCE.md](API_REFERENCE.md), [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)
- Integrator: [INTEGRATIONS.md](INTEGRATIONS.md)
- Operator or SRE: [DEPLOYMENT.md](DEPLOYMENT.md), [TESTING.md](TESTING.md)

## Document Map

| Document | Primary Audience | Purpose |
|---|---|---|
| [VISION.md](VISION.md) | Stakeholders | Product thesis, market problem, positioning |
| [PRD.md](PRD.md) | Product, engineering | Requirements, scope, goals, risks, metrics |
| [USE_CASES.md](USE_CASES.md) | Product, UX, support | Persona-based end-to-end scenarios |
| [FEATURES.md](FEATURES.md) | Sales, product | Tier matrix and capability comparisons |
| [USER_GUIDE.md](USER_GUIDE.md) | Traders | Hands-on setup and usage walkthroughs |
| [STRATEGY.md](STRATEGY.md) | Traders, developers | MTFTR strategy details and EA/Python mapping |
| [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) | Traders, product | Risk controls, thresholds, pause logic |
| [API_REFERENCE.md](API_REFERENCE.md) | Developers, integrators | Endpoint-by-endpoint API reference |
| [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) | Developers, operators | Tables, relationships, migrations |
| [INTEGRATIONS.md](INTEGRATIONS.md) | Integrators | MT5, TradingView, Telegram, observability |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Operators | Local and production runbooks |
| [TESTING.md](TESTING.md) | Developers | Test layers, commands, coverage intent |
| [ROADMAP.md](ROADMAP.md) | Stakeholders | Current state, near-term, long-term direction |
| [GLOSSARY.md](GLOSSARY.md) | All | Shared trading and Aegis terminology |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers | Deep technical architecture and internals |

## Current Ground Truth Anchors

The documents in this folder are expected to stay aligned with:

- [src/main.py](../src/main.py#L121) for app wiring and route composition
- [src/core/config.py](../src/core/config.py#L36) for environment and runtime settings
- [src/database/models.py](../src/database/models.py#L82) for ORM schema definitions
- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L31) and [alembic/versions/006_subscriptions_settings_ratelimits.py](../alembic/versions/006_subscriptions_settings_ratelimits.py#L136) for tier and limit source-of-truth
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L69) and [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27) for ingestion contracts

## Status

This documentation expansion is complete for the planned wave. The completed set is:

- [GLOSSARY.md](GLOSSARY.md)
- [VISION.md](VISION.md)
- [USE_CASES.md](USE_CASES.md)
- [FEATURES.md](FEATURES.md)
- [PRD.md](PRD.md)
- [STRATEGY.md](STRATEGY.md)
- [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md)
- [API_REFERENCE.md](API_REFERENCE.md)
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)
- [INTEGRATIONS.md](INTEGRATIONS.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [TESTING.md](TESTING.md)
- [USER_GUIDE.md](USER_GUIDE.md)
- [ROADMAP.md](ROADMAP.md)

Any future additions can be treated as iterative improvements and maintenance updates.
