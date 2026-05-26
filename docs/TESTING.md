# Testing Strategy

## Test Philosophy

Aegis testing is organized by intent:

- fast logic validation
- integration of collaborating components
- backtest reproducibility checks
- quick smoke confidence

Pytest markers from `pyproject.toml`:

- `unit`
- `integration`
- `backtest_validation`
- `smoke`

## Test Layout

Top-level test directories:

- `tests/unit/`
- `tests/integration/`
- `tests/backtest_validation/`

Shared fixtures:

- `tests/conftest.py`

## Running Tests

### Run all

```bash
pytest
```

### Marker-specific runs

```bash
pytest -m unit
pytest -m integration
pytest -m backtest_validation
pytest -m smoke
```

### Coverage run

```bash
pytest --cov=src --cov-report=term-missing
```

Coverage config currently omits `src/main.py` and reports missing lines.

## Layer Details

### Unit

Focus:

- pure logic and deterministic helpers
- no external service dependencies

Typical targets:

- risk math
- indicator transforms
- mapper and formatter functions

### Integration

Focus:

- repository interactions
- router/service interactions
- auth dependency behaviors with realistic flows

### Backtest Validation

Focus:

- strategy simulation correctness
- reproducibility across historical data windows
- validation outputs against expected patterns

### Smoke

Focus:

- importability and startup-adjacent sanity checks
- high-signal low-runtime checks for CI gate confidence

## Recommended Coverage Targets

Suggested rolling targets:

- unit-critical modules: 80%+
- API and repository path coverage: 70%+
- risk and execution controls: high branch coverage on failure paths

## Current Gaps to Track

Potential under-tested areas to prioritize:

- end-to-end TradingView close flow edge cases
- degraded-mode behavior under DB or Redis partial outage
- migration compatibility tests for future schema changes
- strategy parity regression tests across EA and Python mirror assumptions

## Adding New Tests

For new endpoints:

1. add request/response contract tests
2. add auth boundary tests (JWT/API key/unauth)
3. add tenant-scope tests for data isolation

For new strategy logic:

1. add deterministic unit tests for indicator and filter gates
2. add scenario tests with known bar fixtures
3. add backtest validation snapshots for key market regimes

## CI Recommendations

- run `smoke` and `unit` on every PR
- run `integration` on PR and main
- run `backtest_validation` on scheduled cadence and release branches

## Source Citations

- [pyproject.toml](../pyproject.toml#L1)
- [pyproject.toml](../pyproject.toml#L5)
- [pyproject.toml](../pyproject.toml#L12)
- [tests](../tests)

## Related Docs

- `docs/STRATEGY.md`
- `docs/API_REFERENCE.md`
- `docs/DATABASE_SCHEMA.md`
