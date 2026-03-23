<!--
  ⚠️  BASE BRANCH CHECK  ⚠️
  ─────────────────────────────────────────────────────────────────────────────
  • Feature / fix / chore work  →  base branch must be  `dev`
  • Release (dev → main)        →  only opened by @king1edy
  ─────────────────────────────────────────────────────────────────────────────
  If this PR targets `main` and the head branch is NOT `dev`, it will be
  rejected. Please change the base branch to `dev` before submitting.
-->

## Summary

<!-- Briefly describe what this PR does and why. -->

## Type of Change

<!-- Put an x inside the brackets that apply: [x] -->

- [ ] `feature` – new functionality
- [ ] `fix` – bug fix
- [ ] `chore` – maintenance / refactor / docs
- [ ] `release` – merging `dev` into `main` (maintainer only)

## Related Issues

<!-- Closes #<issue-number> -->

## Changes Made

<!--
  List the main changes:
  - Added X
  - Updated Y
  - Removed Z
-->

## Testing

- [ ] Existing tests pass (`pytest tests/ -v`)
- [ ] New tests added / updated (if applicable)
- [ ] Manually tested the affected functionality

## Checklist

- [ ] My branch is based on `dev` (not `main`)
- [ ] Code follows the style guidelines (`black`, `ruff`, `mypy`)
- [ ] Self-reviewed my own code
- [ ] No secrets or credentials committed
- [ ] Documentation updated (if applicable)
