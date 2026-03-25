# Contributing to Aegis Trader

Thank you for your interest in contributing! Please read this guide carefully before opening any pull requests.

---

## Branching Strategy

This repository uses a **three-tier branching model**:

```
feature/<name>  ──►  dev  ──►  main
```

| Branch | Purpose | Who can push directly |
|--------|---------|----------------------|
| `main` | Production-ready releases | Nobody – PRs from `dev` only |
| `dev` | Integration / staging branch | `king1edy` only |
| `feature/*`, `fix/*`, `chore/*` | Day-to-day development | Any contributor |

---

## Workflow for Contributors

### 1. Create a feature branch from `dev`

Always branch off the latest `dev`:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/<short-description>
```

Use one of these prefixes:

| Prefix | When to use |
|--------|-------------|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `chore/` | Maintenance, refactoring, docs |
| `hotfix/` | Urgent production fixes (still targets `dev` first) |

### 2. Commit and push your feature branch

```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/<short-description>
```

### 3. Open a Pull Request targeting `dev`

- **Base branch must be `dev`** – never open a PR directly to `main`.
- Fill in the PR template completely.
- Link any relevant issues.
- Ensure all CI checks pass before requesting a review.

### 4. Review and merge into `dev`

- At least one approval from `king1edy` is required.
- Squash or merge commit – follow the reviewer's guidance.

### 5. Release: `dev` → `main`

Only `king1edy` opens the PR from `dev` to `main` when a release is ready.  
Direct pushes to `main` are **blocked**; all changes must arrive via this PR.

---

## Branch Protection Rules (enforced by GitHub)

| Rule | `main` | `dev` |
|------|--------|-------|
| Require PR before merging | ✅ | ✅ |
| Only `dev` can be merged into `main` | ✅ (Ruleset) | N/A |
| Direct push allowed | ❌ | `king1edy` only |
| Force pushes allowed | ❌ | ❌ |
| Branch deletion allowed | ❌ | ❌ |

---

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short description>

[optional body]
[optional footer]
```

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

---

## Code Quality

Before pushing, run the local quality checks:

```bash
# Formatting
black src/ tests/

# Linting
ruff src/ tests/

# Type checking
mypy src/

# Tests
pytest tests/ -v
```

All checks must pass for a PR to be mergeable.

---

## Questions?

Open a [GitHub Discussion](../../discussions) or ping `king1edy` in the PR.
