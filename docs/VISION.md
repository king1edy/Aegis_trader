# Product Vision

## Problem

Retail algorithmic traders often combine disconnected tools:

- MT5 for execution
- Spreadsheets or screenshots for journaling
- External alerting tools for notifications
- Manual discipline for risk control

This creates an operational gap between strategy design, execution, and accountability. Traders can execute fast, but they struggle to enforce consistent rules, observe behavior over time, and audit what happened across systems.

## Why Aegis Exists

Aegis combines execution-adjacent ingestion, journaling, and risk enforcement in one platform:

- Single source of truth for trade events and deal history
- Multi-tenant architecture so each trader has isolated data
- Tiered subscription model from journaling to autopilot workflows
- Dual ingestion path: MT5 EA events and TradingView webhooks
- Built-in observability hooks and operational controls

## Market Focus

Initial focus is XAUUSD workflows in MT5-centric environments where:

- MT5 is already the execution surface
- traders need stronger process discipline
- teams need shared visibility into performance and risk behavior

The product architecture is not hard-wired to one symbol, but current strategy defaults and user settings are optimized for this path.

## Positioning

### Against raw MT5

Aegis adds:

- structured journaling and analytics endpoints
- tenant-scoped data model
- API key and JWT auth model
- configurable risk controls and pause semantics

### Against journal-only products

Aegis adds:

- direct EA ingestion endpoint
- TradingView signal ingress
- strategy-aware lifecycle state handling
- operations/deployment support for self-hosted workflows

### Against charting-plus-broker glue

Aegis adds:

- normalized event storage and historical analysis
- robust subscription and rate-limit primitives
- backend-first API surface for integrations

## Mission

Help discretionary and systematic traders operate with risk-first discipline by making trade data, controls, and feedback loops explicit and actionable.

## Product Principles

- Risk first before signal frequency
- Transparent data model over opaque black-box behavior
- Integration optionality: MT5 direct, bridge, paper, and webhook ingress
- Tenant isolation as a default architecture property
- Progressive capability by tier, not by fragmented products

## Long-Term Direction

- Multi-symbol and multi-asset expansion beyond XAUUSD-first defaults
- Stronger signal quality scoring and explainability
- Mobile-friendly operator and trader experiences
- Additional broker ecosystem support through pluggable execution layers
- Team and social workflows where governance remains risk-centered

## Source Citations

- [src/main.py](../src/main.py#L121)
- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L31)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L326)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/journal/router.py](../src/journal/router.py#L70)

## Related Docs

- [PRD.md](PRD.md)
- [USE_CASES.md](USE_CASES.md)
- [FEATURES.md](FEATURES.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
