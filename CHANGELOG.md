# Changelog

All notable changes to the CBPS Python package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Minimum Python version raised to 3.10 (Python 3.9 reached end-of-life October 2025)
- Added Python 3.13 support
- `[all]` extra no longer includes `dev` and `docs` dependencies (user-facing only)
- Replaced invalid PyPI classifier `Topic :: Scientific/Engineering :: Statistics` with `Topic :: Scientific/Engineering :: Information Analysis`
- Removed unused `setuptools_scm` from build dependencies
- Separated `[test]` optional dependencies from `[dev]` for cleaner CI installs
- Rewrote `examples/README.md` and `docs/tutorials/index.rst` to reflect the
  three replication notebooks actually shipped with the package
  (Imai & Ratkovic 2014/2015, Fong, Hazlett & Imai 2018)

### Added
- mypy hook in pre-commit configuration
- `numpydoc` added to docs dependencies for NumPy-style docstring support
- `pre-commit` added to dev dependencies
- Bandit security scanning in pre-commit hooks
- McCabe complexity checking in flake8 configuration (max-complexity=15)
- `Framework :: Pytest` PyPI classifier
- `sphinx-copybutton` extension for documentation code blocks
- `numpydoc` and `sphinx.ext.todo` Sphinx extensions for improved documentation rendering
- LaTeX packages in ReadTheDocs configuration for reliable PDF builds
- `statistics` and `count` options in flake8 configuration for summary output
- Complete API reference under `docs/api/` generated via Sphinx autodoc
- `Optimal CBPS` section in `docs/theory.rst` covering Fan et al. (2022)
- Fan et al. (2022) entry added to README.md References/BibTeX sections and `CITATION.cff`

### Fixed
- Removed E501 from flake8 extend-ignore to properly enforce line length via black
- Fixed truncated `[tool.black]` include regex pattern in pyproject.toml
- Set `nbsphinx_allow_errors = False` in Sphinx configuration for production quality
- Removed redundant `Download` URL from pyproject.toml project URLs
- Updated copyright year range in documentation configuration
- Unified GitHub repository URLs across README, docs, and CITATION.cff to
  use `https://github.com/gorgeousfish/CBPS-py`
- Removed hardcoded local `sys.path` insertion from `examples/run_replication.py`
- Corrected `docs/implementation_notes.rst` description of automatic treatment
  type detection (only strict 0/1 integer arrays are auto-detected as binary;
  3–4 level discrete treatments require explicit `pd.Categorical` conversion)
- Aligned `CITATION.cff` release date with the `0.1.0` entry in the changelog
- Corrected README note claiming that `two_step` is ignored when
  `method='exact'`; the two flags are independent — `two_step` still
  controls whether the balance-loss optimizer uses an analytical gradient
- Completed `CBMSM()` docstring `init` parameter listing by adding the
  previously-documented-only-at-the-low-level `'CBPS'` option
- Added a runnable scikit-learn integration example (Pipeline +
  IPW-weighted outcome regression) to README and
  `docs/advanced_usage.rst`, along with a clear note that
  `CBPSEstimator.predict_proba`/`predict` only return training-sample
  fitted values (so default-scored `GridSearchCV`/`cross_val_score` do not
  yield meaningful test-fold scores)
- Corrected the `hdCBPS()` docstring in `cbps/__init__.py`: `seed` is
  now documented as controlling the LASSO cross-validation fold
  assignment (matching the `README.md`, `docs/implementation_notes.rst`
  and the low-level `cbps.highdim.hdcbps.hdCBPS` docstring); the
  `verbose` parameter is now correctly flagged as reserved for future
  use (instead of advertising three non-existent verbosity levels)
- Rewrote the Optimal CBPS section of `docs/theory.rst` so the moment
  conditions match Fan et al. (2022, Eq. 3.3) and the Python
  implementation in `cbps/core/cbps_optimal.py`: the two conditions are
  `(T/π - (1-T)/(1-π)) h_1(X) = 0` (balancing `baseline_formula`) and
  `(T/π - 1) h_2(X) = 0` (balancing `diff_formula`), with
  `h_1 ≈ E[Y(0)|X]` and `h_2 ≈ E[Y(1)-Y(0)|X]`
- Replaced the `pip install cbps-python` instructions in `README.md`,
  `docs/installation.rst`, `docs/index.rst` and
  `docs/api/diagnostics.rst` with a GitHub-based install
  (`pip install "cbps-python @ git+https://github.com/gorgeousfish/CBPS-py.git"`)
  because the package is not yet published to PyPI, and noted how to
  switch back to the shorter PyPI form once the package is released
- Commented out the PyPI version badge in `README.md` (wrapped in an
  HTML comment with a restore hint) because `cbps-python` is not yet
  published to PyPI and the badge was rendering as a broken
  "package not found" image
- Fixed Read the Docs build failure under
  `sphinx -W --keep-going` (`fail_on_warning: true`): the build was
  failing because Sphinx autodoc could not import
  `cbps.sklearn.CBPSEstimator` (RTD only installed the `[docs]` extra,
  so `scikit-learn` was missing and the unguarded
  `from sklearn.base import ...` in `cbps/sklearn/estimator.py` raised
  `ImportError`, which the autosummary logic surfaced as a warning).
  Two complementary fixes:
  * `.readthedocs.yaml` now installs the `sklearn` and `plots` extras
    in addition to `docs`, so `cbps.sklearn.CBPSEstimator` and the
    `cbps.diagnostics.plots` helpers render with their real signatures
  * `docs/conf.py` declares
    `autodoc_mock_imports = ['glmnetforpython', 'rpy2']` so RTD does not
    need a Fortran toolchain to build documentation for `cbps.hdCBPS`
    or the optional rpy2-backed validation helper

## [0.1.0] - 2025-12-04

### Added

#### Core Estimation Algorithms
- **CBPS** — Main function for binary/multi-valued/continuous treatments
  - Binary treatments (0/1): ATT and ATE estimation with GMM and over-identified GMM
  - Multi-valued treatments (3–4 levels): Multinomial logit propensity scores
  - Continuous treatments: Generalized propensity score (GPS)
  - Automatic treatment type detection: arrays with exactly two unique values in
    ``{0, 1}`` (integer, boolean, or float) are routed to the binary backend;
    3–4 level discrete treatments require explicit ``pd.Categorical`` conversion
  - Formula interface with automatic intercept handling
  - SVD preprocessing for numerical stability
  - Over-identified GMM with J-statistic
- **Optimal CBPS (oCBPS)** — Doubly-robust estimation (Fan et al. 2022)
- **CBMSM** — Marginal structural models for longitudinal data
- **npCBPS** — Nonparametric CBPS using empirical likelihood
- **hdCBPS** — High-dimensional CBPS with LASSO variable selection
- **CBIV** — CBPS for instrumental variables

#### Inference Methods
- **AsyVar** — Sandwich variance estimator for CBPS coefficients
- **vcov_outcome** — Robust variance estimation for weighted outcome regression

#### Diagnostic and Visualization Tools
- **balance** — Covariate balance assessment (SMD for discrete, correlation for continuous)
- **summary** — Statistical summary with coefficient table and J-statistic
- **plot** — Balance plots, weight distribution plots for binary and continuous treatments

#### Data and Examples
- LaLonde NSW dataset, Blackwell dataset, continuous treatment simulation data
- Three replication notebooks (and equivalent scripts) covering Imai &
  Ratkovic (2014), Fong, Hazlett & Imai (2018), and Imai & Ratkovic (2015)

#### scikit-learn Integration
- **CBPSEstimator** — sklearn-compatible wrapper with `fit()`, `predict_proba()`, `predict()`, `get_weights()`

#### Development Infrastructure
- Testing framework with pytest (2000+ tests, target ≥80% coverage)
- Code quality tools: black, isort, flake8, mypy, pre-commit, bandit
- Documentation: Sphinx + ReadTheDocs with PDF/ePub output

### Numerical Accuracy
- Core algorithms validated to ±1e-6 precision against R CBPS package v0.23
- Benchmark tests using LaLonde and Blackwell datasets

### References

1. Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *JRSS-B*, 76(1), 243–263.
2. Fan, J., et al. (2022). Optimal covariate balancing conditions in propensity score estimation. *JBES*, 41(1), 97–110.
3. Imai, K., & Ratkovic, M. (2015). Robust estimation of inverse probability weights for marginal structural models. *JASA*, 110(511), 1013–1023.
4. Fong, C., Hazlett, C., & Imai, K. (2018). Covariate balancing propensity score for a continuous treatment. *AOAS*, 12(1), 156–177.
5. Ning, Y., Peng, S., & Imai, K. (2020). Robust estimation of causal effects via a high-dimensional covariate balancing propensity score. *Biometrika*, 107(3), 533–554.

[Unreleased]: https://github.com/gorgeousfish/CBPS-py/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gorgeousfish/CBPS-py/releases/tag/v0.1.0
