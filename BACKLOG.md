# commodity-backtester — Issue Backlog

Issues are ordered by **recruiter impact ÷ implementation effort**. All issues below have been converted into actual code in this repository.

---

## Issue #1 — [CRITICAL] Remove committed virtual environment from git

**Labels:** `bug` `infra` `priority:critical`

### Description

The `backtest/` directory is a committed Python virtual environment (contains `Lib/site-packages/`, `Scripts/`, `pyvenv.cfg`). This means the repository carries ~2.3 MB of third-party library source code that should never be tracked. Consequences:

- `git clone` is unnecessarily slow
- Dependency versions are frozen inside a platform-specific binary tree
- Any `pip install` changes to the venv will produce thousands of diffs

### Acceptance Criteria

- [ ] `backtest/` is removed from git tracking via `git rm -r --cached`
- [ ] `backtest/` and any other venv-like directories are added to `.gitignore`
- [ ] `git status` shows a clean tree after removal
- [ ] `requirements.txt` remains the single source of truth for dependencies

---

## Issue #2 — [CRITICAL] Replace expiring contract tickers with continuous futures

**Labels:** `bug` `data` `priority:critical`

### Description

`constants.py` hardcodes front-month contract tickers (`ZSN25.CBT`, `ZCN25.CBT`, etc.) with human-readable names like `"Soybean - july 25"`. These tickers expire in July 2025. After expiry:

- `yfinance.download()` returns an empty DataFrame
- The app silently shows "No data found" with no explanation
- All downstream logic breaks

Continuous (perpetual) futures tickers — `ZS=F`, `ZC=F`, etc. — automatically roll to the nearest active contract and never expire.

### Acceptance Criteria

- [ ] All tickers in `constants.py` are replaced with their `=F` continuous equivalents
- [ ] Commodity display names are generic (no month/year suffix)
- [ ] Contract sizes and conversion factors are updated to match continuous contract specs
- [ ] `yahoo_quotes()` returns non-empty data for any date range since 2000

---

## Issue #3 — [CRITICAL] Fix broken relative imports in `src/` modules

**Labels:** `bug` `priority:critical`

### Description

`src/utils.py` and `src/data_loader.py` use bare module imports:

```python
from constants import commodities_dict, tickers   # ❌ breaks when imported from app.py
```

When Python resolves these from `app.py`, it cannot find a top-level module named `constants`. The correct pattern for a package is either:

```python
from .constants import commodities_dict, tickers  # ✅ relative import
```

or ensuring `src/` is a proper package with all imports prefixed with `src.`.

### Acceptance Criteria

- [ ] All imports within `src/` use relative syntax (`from .module import ...`)
- [ ] `app.py` imports are unchanged (`from src.constants import ...`)
- [ ] `python -c "from src.strategy import backtest"` executes without error
- [ ] No `ImportError` or `ModuleNotFoundError` at runtime

---

## Issue #4 — Add `pyproject.toml` with ruff, mypy, and pytest configuration

**Labels:** `infra` `dx` `priority:high`

### Description

The project has no unified tooling configuration. Linting, type-checking, and test settings live in different places (or not at all). A `pyproject.toml` is the modern Python standard that consolidates all tool configuration into one file, signals maturity to reviewers, and enables one-command quality checks.

### Acceptance Criteria

- [ ] `pyproject.toml` exists at repo root
- [ ] `[tool.ruff]` section configured with line-length, target Python version, and relevant rule sets (`E`, `F`, `I`, `N`, `UP`)
- [ ] `[tool.mypy]` section configured with `strict = true`, `python_version = "3.11"`
- [ ] `[tool.pytest.ini_options]` section replaces `pytest.ini` (or `pytest.ini` is removed)
- [ ] `ruff check src/` exits 0
- [ ] `mypy src/` exits 0

---

## Issue #5 — Add `requirements-dev.txt` with testing and linting dependencies

**Labels:** `infra` `dx` `priority:high`

### Description

There is no separation between runtime dependencies (`requirements.txt`) and development/testing dependencies. Developers and CI need pytest, ruff, mypy, and pytest-cov installed, but production deployments (Streamlit Cloud) do not.

### Acceptance Criteria

- [ ] `requirements-dev.txt` exists and includes: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pandas-stubs`, `types-requests`
- [ ] `requirements.txt` contains only runtime dependencies (no dev tools)
- [ ] `README.md` documents `pip install -r requirements-dev.txt` for local development

---

## Issue #6 — Add GitHub Actions CI pipeline (lint + typecheck + test)

**Labels:** `ci` `infra` `priority:high`

### Description

The project has `pytest.ini` and test files visible on GitHub but no workflow to run them automatically. A CI badge on the README is the single most trusted signal of code quality for technical recruiters and hiring managers.

Pipeline should include:
1. `ruff check` — fast linting
2. `mypy src/` — static type validation
3. `pytest --cov=src` — test execution with coverage

### Acceptance Criteria

- [ ] `.github/workflows/ci.yml` triggers on `push` and `pull_request` to `main`
- [ ] Pipeline runs on `ubuntu-latest`, Python 3.11
- [ ] Pipeline has three jobs: `lint`, `typecheck`, `test`
- [ ] Coverage report is uploaded (Codecov or summary comment)
- [ ] README has a CI status badge linked to the workflow

---

## Issue #7 — Add type hints to all `src/` modules

**Labels:** `quality` `typing` `priority:high`

### Description

No function in `src/` has type annotations. For a quant/data-engineering role, type safety is a non-negotiable signal of production-readiness. Annotations also enable mypy to catch logic errors (e.g., optional vs non-optional returns, wrong dict key types) before runtime.

### Acceptance Criteria

- [ ] All public functions in `strategy.py`, `utils.py`, `data_loader.py`, `visualization.py`, `constants.py` have full parameter and return type annotations
- [ ] `mypy src/ --strict` passes with zero errors
- [ ] Type stubs (`pandas-stubs`, `types-requests`) are installed for external libs

---

## Issue #8 — Vectorize the backtest engine (replace `iterrows` with pandas ops)

**Labels:** `performance` `quality` `priority:medium`

### Description

`strategy.py` uses `for _, row in df_filtered.iterrows()` to generate trade signals. On a 20-year daily series (~5,000 rows), this is ~50–100× slower than equivalent vectorized pandas/numpy operations. The ratio strategy can be fully expressed as a vectorized state machine using `np.where` and `shift()`.

Beyond performance, vectorized code is easier to test (each step is a pure Series transformation), easier to read, and demonstrates pandas proficiency.

### Acceptance Criteria

- [ ] `backtest()` for the `"ratio"` strategy uses no Python-level loops over rows
- [ ] Output `df_trades` is identical to the previous implementation (verified by a regression test)
- [ ] Runtime for a 20-year backtest is under 100 ms (measurable via `pytest-benchmark` or `%timeit`)
- [ ] Mean reversion stub is preserved and documented as `NotImplementedError`

---

## Issue #9 — Add extended performance metrics (Sortino, Calmar, Profit Factor)

**Labels:** `feature` `quant` `priority:medium`

### Description

`backtest_performance()` currently computes a basic Sharpe ratio using trade-level PnL, which is not the standard methodology (Sharpe should use daily returns, not per-trade returns). Additionally, Sortino ratio, Calmar ratio, and Profit Factor are standard quant metrics expected by any trading desk or quant researcher reviewing the output.

### Acceptance Criteria

- [ ] Sharpe ratio is recomputed on **daily** equity curve returns (not per-trade PnL)
- [ ] **Sortino ratio** is added (downside deviation denominator)
- [ ] **Calmar ratio** is added (annualized return / max drawdown)
- [ ] **Profit Factor** is added (gross profit / gross loss)
- [ ] **Recovery Factor** is added (total profit / max drawdown)
- [ ] All new metrics appear in the `backtest_performance()` DataFrame output
- [ ] Each metric has a docstring explaining its formula

---

## Issue #10 — Add unit tests for core engine modules

**Labels:** `testing` `quality` `priority:medium`

### Description

The GitHub API reports a `tests/` directory, but it does not exist locally — it was never committed. Without tests, there is no safety net for refactoring, and CI cannot report a coverage number. This is the highest-visibility gap for a technical reviewer.

Tests should cover the deterministic core: signal generation, PnL calculation, and performance metrics. Streamlit UI rendering is excluded.

### Acceptance Criteria

- [ ] `tests/` directory exists and is tracked in git
- [ ] `tests/test_strategy.py` covers: correct buy/sell signal generation, no trades when ratio never crosses threshold, open position at end of series
- [ ] `tests/test_utils.py` covers: correct round-trip PnL, zero PnL on flat prices, MTM calculation for open positions
- [ ] `tests/test_metrics.py` covers: Sharpe/Sortino/Calmar/profit-factor on synthetic equity curves
- [ ] `tests/conftest.py` contains shared fixtures (synthetic price DataFrame)
- [ ] `pytest --cov=src` reports ≥ 80% coverage across `src/`

---

## Issue #11 — Update README with CI badge, screenshots, and developer setup

**Labels:** `docs` `priority:medium`

### Description

The README currently has only a Streamlit app badge and no CI status, no screenshots of the running app, and no developer setup instructions (how to run tests, how to lint). For any technical reviewer who lands on the repo, the first 10 seconds of the README determines whether they read further.

### Acceptance Criteria

- [ ] CI status badge is present and linked to the GitHub Actions workflow
- [ ] At least one screenshot of the running Streamlit app (results table + chart) is embedded
- [ ] "Developer Setup" section explains: clone → venv → `pip install -r requirements-dev.txt` → `pytest` → `ruff check src/`
- [ ] "Architecture" section has an updated project structure tree reflecting new layout
- [ ] All badges render correctly on GitHub (not broken image links)
