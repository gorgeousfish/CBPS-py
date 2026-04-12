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

### Added
- Documentation build check in CI pipeline
- mypy hook in pre-commit configuration
- `numpydoc` added to docs dependencies for NumPy-style docstring support
- `pre-commit` added to dev dependencies
- Package build verification step in CI workflow
- Bandit security scanning in pre-commit hooks
- McCabe complexity checking in flake8 configuration (max-complexity=15)
- CodeQL security scanning workflow for automated vulnerability detection
- `Framework :: Pytest` PyPI classifier
- `sphinx-copybutton` extension for documentation code blocks
- `numpydoc` and `sphinx.ext.todo` Sphinx extensions for improved documentation rendering
- Version tag validation step in publish workflow
- `timeout-minutes` for all CI jobs to prevent hanging builds
- LaTeX packages in ReadTheDocs configuration for reliable PDF builds
- `statistics` and `count` options in flake8 configuration for summary output

### Fixed
- Codecov action token configuration updated to recommended `with: token:` syntax
- Removed E501 from flake8 extend-ignore to properly enforce line length via black
- Fixed truncated `[tool.black]` include regex pattern in pyproject.toml
- Set `nbsphinx_allow_errors = False` in Sphinx configuration for production quality
- Removed redundant `Download` URL from pyproject.toml project URLs
- Updated copyright year range in documentation configuration

## [0.1.0] - 2025-12-04

### Added

#### Core Estimation Algorithms
- **CBPS** — Main function for binary/multi-valued/continuous treatments
  - Binary treatments (0/1): ATT and ATE estimation with GMM and over-identified GMM
  - Multi-valued treatments (3–4 levels): Multinomial logit propensity scores
  - Continuous treatments: Generalized propensity score (GPS)
  - Automatic treatment type detection for integer arrays (≤4 unique values)
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
- 13 Python example scripts, 4 Jupyter notebook tutorials

#### scikit-learn Integration
- **CBPSEstimator** — sklearn-compatible wrapper with `fit()`, `predict_proba()`, `predict()`, `get_weights()`

#### Development Infrastructure
- Testing framework with pytest (400+ tests, >80% coverage)
- Code quality tools: black, isort, flake8, mypy
- CI/CD: GitHub Actions with multi-OS, multi-Python testing
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

[Unreleased]: https://github.com/gorgeousfish/cbps-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gorgeousfish/cbps-python/releases/tag/v0.1.0
