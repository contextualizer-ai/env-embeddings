# CI/CD and Quality Assurance Setup

This document describes the comprehensive CI/CD and code quality setup for the env-embeddings project.

## Overview

The project now has a complete quality assurance pipeline that runs:
- **Tests** with coverage reporting
- **Type checking** with mypy
- **Linting** with ruff
- **Dependency checking** with deptry

These checks run:
1. **Locally** via `just test`
2. **On commit** via pre-commit hooks
3. **On push** via pre-push hooks
4. **On CI** via GitHub Actions

## Quick Start

### Install Pre-commit Hooks

```bash
# Install pre-commit tool (if not already installed)
uv tool install pre-commit

# Install the hooks
pre-commit install
pre-commit install --hook-type pre-push
```

### Run All Quality Checks

```bash
just test
```

This runs:
1. `pytest` with coverage (minimum 50% coverage)
2. `mypy` for type checking
3. `ruff` for linting
4. `deptry` for dependency analysis

## Test Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `just test` | Run all quality checks (pytest-cov + mypy + ruff + deptry) |
| `just pytest-cov` | Run tests with coverage and timing |
| `just pytest` | Run tests only (no coverage) |
| `just mypy` | Run type checking |
| `just ruff` | Run linting |
| `just deptry` | Check for unused/missing dependencies |

### Advanced Commands

| Command | Description |
|---------|-------------|
| `just test-full` | Run all checks including integration tests |
| `just pytest-integration` | Run integration tests |
| `just doctest` | Run doctests in src/ |

## Test Coverage

Current coverage: **9%** (baseline from existing tests)

Coverage reports:
- **Terminal**: Shows missing lines after test run
- **HTML**: Generated in `htmlcov/` directory
- **Minimum**: 50% coverage required for pre-push

View HTML coverage:
```bash
open htmlcov/index.html
```

## Pre-commit Hooks

### Installed Hooks

**On every commit:**
- `check-toml` - Validate TOML files
- `check-yaml` - Validate YAML files
- `end-of-file-fixer` - Ensure files end with newline
- `trailing-whitespace` - Remove trailing whitespace
- `yamllint` - Lint YAML files
- `codespell` - Spell checking
- `typos` - Typo detection
- `ruff` - Linting with auto-fix
- `ruff-format` - Code formatting
- `uv-lock` - Keep uv.lock in sync
- `mypy` - Type checking
- `pytest` - Run tests

**On push:**
- `pytest-cov` - Run tests with 50% minimum coverage

### Manual Pre-commit Run

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run mypy --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify
```

## GitHub Actions

The CI pipeline runs on:
- **Push to main**
- **Pull requests**

Workflow: `.github/workflows/main.yaml`

Matrix testing across Python versions:
- 3.10
- 3.11
- 3.12
- 3.13

## Configuration Files

### pytest (`pytest.ini` or `pyproject.toml`)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### mypy (`mypy.ini`)
Key settings:
- Type checking for `src/` and `tests/`
- Ignores missing imports for external libraries without type stubs:
  - pandas, pytest, typer, diskcache, tqdm, ols_client, global_land_mask, ee

### ruff (`pyproject.toml`)
- Modern, fast Python linter
- Replaces flake8, isort, and more
- Auto-fixes many issues

### deptry (`pyproject.toml`)
```toml
[tool.deptry]
per_rule_ignores = {
  DEP002 = ["linkml-runtime", "numpy", "requests-cache", "global-land-mask"],
  DEP003 = ["typing_extensions"]
}
```

## Dependencies

### Development Dependencies

All testing tools are installed as dev dependencies:
```bash
uv sync --group dev
```

Installed tools:
- `pytest` - Testing framework
- `pytest-cov` - Coverage plugin
- `mypy` - Type checker
- `ruff` - Linter/formatter
- `deptry` - Dependency checker

## Troubleshooting

### "Failed to spawn: pytest"

**Problem**: Virtual environment may be corrupt or pointing to wrong path

**Solution**:
```bash
rm -rf .venv
uv sync --group dev
```

### Mypy can't find type stubs

**Problem**: External libraries missing type information

**Solution**: Add to `mypy.ini`:
```ini
[mypy-package_name.*]
ignore_missing_imports = True
```

### Deptry false positives

**Problem**: Legitimate dependencies flagged as unused

**Solution**: Add to `pyproject.toml`:
```toml
[tool.deptry]
per_rule_ignores = {DEP002 = ["package_name"]}
```

### Coverage too low

**Problem**: Tests don't cover enough code

**Solution**: Write more tests or adjust minimum:
```bash
# Adjust minimum in .pre-commit-config.yaml
entry: uv run pytest --cov=src/env_embeddings --cov-fail-under=40
```

## Best Practices

### 1. Run tests before committing
```bash
just test
```

### 2. Let pre-commit auto-fix issues
Pre-commit will auto-fix many issues (trailing whitespace, formatting, etc.)

### 3. Don't skip hooks
Only use `--no-verify` in emergencies

### 4. Keep coverage high
Aim for >80% coverage on new code

### 5. Fix type errors
Don't just add `# type: ignore` - fix the actual issue

### 6. Update dependencies regularly
```bash
uv lock --upgrade
```

## Example Workflow

```bash
# Make changes to code
vim src/env_embeddings/earth_engine.py

# Run tests locally
just test

# Tests pass! Stage changes
git add src/env_embeddings/earth_engine.py

# Commit (pre-commit hooks run automatically)
git commit -m "Add retry logic to earth engine"

# Push (pre-push hooks run automatically including coverage check)
git push origin feature-branch

# GitHub Actions runs on the PR
# All checks must pass before merge
```

## Summary

✅ **Complete test suite** with coverage reporting
✅ **Type safety** with mypy
✅ **Code quality** with ruff
✅ **Dependency health** with deptry
✅ **Pre-commit hooks** for immediate feedback
✅ **CI/CD** with GitHub Actions
✅ **No workarounds** - clean `uv run` commands

The setup ensures high code quality without being overly restrictive. All checks are fast and provide clear error messages.
