# Product Requirements Document (PRD)

## 1. Executive Summary

Aegis Trader is a multi-tenant trading platform that unifies event ingestion, journaling, analytics, and risk controls for MT5-centric workflows. It supports three subscription tiers (journal, pro, autopilot), dual signal ingestion (EA webhook plus TradingView webhook), and backend capabilities for authentication, settings, and operational visibility.

The product solves a core problem: traders execute quickly but often lack consistent discipline tooling, reproducible trade records, and integrated risk guardrails.

## 2. Product Goals and Non-Goals

### Goals

- Provide a single backend for trade ingestion, journaling, and analytics
- Enforce configurable risk constraints with both pre-trade and continuous checks
- Support multi-tenant isolation for all user-owned trading artifacts
- Enable integration flexibility across MT5 direct, bridge, paper, and webhook pathways
- Offer clear tiered value progression from journal to autopilot capabilities

### Non-Goals (Current Scope)

- Broker-side regulatory compliance automation
- Tax reporting and jurisdiction-specific reporting packs
- Fully managed social copy-trading platform
- Native mobile app parity in current release line
- Multi-broker parity across all non-MT5 ecosystems

## 3. Personas

### Persona A: Retail Trader

Needs:

- Quick onboarding from MT5 to visible journal results
- Clear risk boundaries with minimal setup overhead
- Performance breakdown by sessions, setups, and symbols

Success criteria:

- First trade appears in dashboard within initial setup session
- User can explain losses and wins with stored context

### Persona B: EA Developer

Needs:

- Stable API key lifecycle
- Predictable webhook contract and response behavior
- Extensible fields for strategy metadata

Success criteria:

- Can rotate keys without downtime
- Can segment analytics by strategy metadata

### Persona C: TradingView User

Needs:

- Reliable alert-to-backend path
- Payload schema clarity
- Source attribution in downstream analytics

Success criteria:

- Test alert accepted and visible in records

### Persona D: Prop-Firm Candidate

Needs:

- Strict daily and total risk constraints
- Automatic pause semantics
- Audit trail proving rule compliance behavior

Success criteria:

- System blocks rule-violating conditions before additional damage

### Persona E: Operator/Admin

Needs:

- Repeatable deploy and startup process
- Visibility into health and migration state
- Safe degraded behavior for dependency outages

Success criteria:

- Stack reaches healthy state with predictable checks

## 4. Primary Use Cases

Canonical scenarios are documented in [USE_CASES.md](USE_CASES.md). Key workflows include:

- Account registration and first JWT session
- API key generation and EA ingestion handshake
- Trade journaling with setup tagging and analytics
- TradingView alert ingestion and source attribution
- Risk threshold breach and pause lifecycle
- Docker-based deployment with health verification

## 5. Functional Requirements

### 5.1 Authentication and Tenant Boundary

- FR-001: System shall support account registration with unique email or username.
- FR-002: System shall issue JWT tokens for authenticated dashboard/API usage.
- FR-003: System shall support user-managed API keys for machine webhook calls.
- FR-004: System shall enforce tenant-scoped access to user-owned resources.
- FR-005: System shall return enriched user profile containing subscription and tier limits.

### 5.2 Ingestion and Journaling

- FR-010: System shall accept EA trade events at POST /trade using API key auth.
- FR-011: System shall expose health endpoint for liveness checks.
- FR-012: System shall support TradingView webhook ingestion at POST /webhook/tradingview.
- FR-013: System shall persist trade records with strategy and market context fields.
- FR-014: System shall keep CSV journal continuity for event traceability.

### 5.3 Journal Analytics and Dashboard

- FR-020: System shall provide journal analytics routes for sessions, hours, days, setups, symbols, and direction.
- FR-021: System shall support trade list retrieval and open trade views.
- FR-022: System shall allow updating selected trade journaling attributes.
- FR-023: System shall provide setup tag CRUD for categorization workflows.

### 5.4 Settings and Tier-Aware Controls

- FR-030: System shall provide per-user settings retrieval and patch endpoints.
- FR-031: System shall persist risk and notification preferences in user settings.
- FR-032: System shall expose subscription and rate-limit details to authenticated users.
- FR-033: System shall initialize default subscription and user settings at registration.

### 5.5 Risk Management

- FR-040: System shall validate new signals against configurable pre-trade constraints.
- FR-041: System shall continuously monitor drawdown, margin, and periodic loss thresholds.
- FR-042: System shall persist trading pause records with reason and trigger metadata.
- FR-043: System shall support warning and emergency callbacks for critical conditions.

### 5.6 Notifications

- FR-050: System shall support Telegram-based notifications with per-user toggles.
- FR-051: System shall emit risk warning notifications when configured thresholds are near or breached.

### 5.7 Deployment and Operations

- FR-060: System shall support containerized startup with migration execution before app launch.
- FR-061: System shall start in best-effort mode when database connectivity is unavailable.
- FR-062: System shall apply Redis-backed rate limits when Redis is available.
- FR-063: System shall degrade gracefully when Redis is unavailable.

## 6. Non-Functional Requirements

### 6.1 Reliability and Availability

- NFR-001: Core auth, health, and ingestion routes should remain available under normal dependency health.
- NFR-002: Dependency failures should fail soft where explicitly designed (for example Redis rate limiter fallback).

### 6.2 Data Integrity and Durability

- NFR-010: Trade events should be persisted to durable storage with tenant ownership metadata.
- NFR-011: Migration-managed schema must remain source of truth for table lifecycle.
- NFR-012: Journal data model must preserve auditability across trades, deals, modifications, and pauses.

### 6.3 Security and Isolation

- NFR-020: JWT and API key authentication must enforce route-level access patterns.
- NFR-021: Tenant-scoped data retrieval should be default behavior in repositories and routes.
- NFR-022: Sensitive credentials and secrets must be environment-configured, not hardcoded.

### 6.4 Performance

- NFR-030: Journal APIs should support practical dashboard usage with paged or bounded responses.
- NFR-031: Risk monitor interval should remain near configured cadence (default approx. 18 seconds).

### 6.5 Observability

- NFR-040: System should emit structured logs for startup, ingest, risk, and failure paths.
- NFR-041: OpenTelemetry export configuration should be available for traces/logs/metrics integration.

## 7. Tier Matrix and Commercial Packaging

| Tier | Description | API/min | API/day | Webhook/min | Backtests/day | Strategies | Accounts |
|---|---|---:|---:|---:|---:|---:|---:|
| Journal | Journal and analytics baseline | 30 | 5000 | 60 | 2 | 1 | 1 |
| Pro | Higher limits plus stronger operational posture | 120 | 20000 | 300 | 20 | 10 | 3 |
| Autopilot | Highest throughput and scaling envelope | 300 | 50000 | 600 | 100 | 25 | 5 |

Source of truth: tier models and migration seed values.

## 8. Success Metrics

### Activation Metrics

- New user reaches first logged trade within onboarding window
- API key creation conversion after registration

### Engagement and Retention Metrics

- Weekly active users touching journal analytics endpoints
- Frequency of tagging and analysis usage by active traders

### Risk and Behavior Metrics

- Number of prevented trades due to risk constraints
- Time to recovery after risk pause event

### Revenue and Packaging Metrics

- Journal to Pro conversion rate
- Pro to Autopilot conversion rate
- Churn by tier and usage profile

## 9. Constraints and Risks

### Technical Constraints

- Direct MT5 execution paths are constrained by Windows and MT5 runtime compatibility
- Bridge path introduces additional network and operational dependency

### Product Risks

- Misconfigured risk settings may lead to false confidence
- Aggressive default limits can reduce user perceived autonomy
- Schema and docs drift if migration and docs updates are not synchronized

### Operational Risks

- Redis outage reduces abuse protection and fair-use enforcement
- Database outage can force reduced functionality modes

## 10. Out of Scope

- Full broker compliance automation and legal reporting tooling
- Tax forms and regional accounting exports
- End-to-end social trading marketplace features
- Full native mobile applications in current phase

## 11. Open Questions

- What exact commercial entitlements differentiate Pro from Autopilot beyond throughput and scale limits?
- Should there be explicit SLA tiers for webhook processing latency and support response time?
- What is the preferred roadmap for multi-symbol defaults beyond XAUUSD-first onboarding?
- How should billing lifecycle states map to graceful feature degradation behaviors?

## 12. Source of Truth Anchors

This PRD is aligned to current implementation anchors:

- [src/main.py](../src/main.py#L121)
- [src/auth/router.py](../src/auth/router.py#L105)
- [src/auth/subscription_models.py](../src/auth/subscription_models.py#L31)
- [src/settings/router.py](../src/settings/router.py#L59)
- [src/trade_logging/trade_event_server.py](../src/trade_logging/trade_event_server.py#L98)
- [src/webhooks/tv_router.py](../src/webhooks/tv_router.py#L27)
- [src/journal/router.py](../src/journal/router.py#L70)
- [src/risk/risk_checker.py](../src/risk/risk_checker.py#L30)
- [src/risk/risk_monitor.py](../src/risk/risk_monitor.py#L18)
- [src/core/config.py](../src/core/config.py#L36)
- [src/database/models.py](../src/database/models.py#L82)
- [alembic/versions/006_subscriptions_settings_ratelimits.py](../alembic/versions/006_subscriptions_settings_ratelimits.py#L17)
- [alembic/versions/007_tradingview_support.py](../alembic/versions/007_tradingview_support.py#L16)

## 13. Related Docs

- [VISION.md](VISION.md)
- [USE_CASES.md](USE_CASES.md)
- [FEATURES.md](FEATURES.md)
- [GLOSSARY.md](GLOSSARY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
