# CBPS Python Test Suite

## Overview

Comprehensive test suite for the CBPS (Covariate Balancing Propensity Score)
Python package, organized for JOSS publication review. The suite validates
numerical accuracy against five peer-reviewed papers and provides unit,
integration, and edge-case coverage for all public modules.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures, markers, warning filters
├── test_imports.py          # Import validation for all public modules
├── test_api.py              # Top-level API interface tests
│
├── binary/                  # Binary treatment CBPS
│   ├── test_unit.py         # Unit tests (GMM, weights, V-matrix, vcov)
│   ├── test_integration.py  # Integration tests (fit pipeline, LaLonde)
│   └── test_edge_cases.py   # Edge cases (collinearity, extreme propensities)
│
├── continuous/              # Continuous treatment CBPS
│   └── test_continuous.py
│
├── core/                    # Core module (CBPSResult, initialization)
│   └── test_core.py
│
├── datasets/                # Built-in datasets (LaLonde)
│   └── test_datasets.py
│
├── diagnostics/             # Balance diagnostics and visualization
│   └── test_diagnostics.py
│
├── highdim/                 # High-dimensional CBPS (Ning et al. 2020)
│   └── test_hdcbps.py
│
├── inference/               # Statistical inference (sandwich SE, CI)
│   └── test_inference.py
│
├── integration/             # Cross-module integration tests
│   └── test_pipeline.py
│
├── iv/                      # Instrumental variable CBPS
│   └── test_cbiv.py
│
├── monte_carlo/             # Paper reproduction Monte Carlo tests
│   ├── conftest.py          # DGP implementations and MC utilities
│   ├── paper_constants.py   # Numerical targets (single source of truth)
│   ├── test_imai2014.py     # Imai & Ratkovic (2014) JRSSB
│   ├── test_ir2015.py       # Imai & Ratkovic (2015) JASA
│   ├── test_fong2018.py     # Fong et al. (2018) AoAS
│   ├── test_ning2020.py     # Ning et al. (2020) Biometrika
│   └── test_fan2022.py      # Fan et al. (2022) JBES
│
├── msm/                     # Marginal structural model CBPS
│   └── test_cbmsm.py
│
├── multitreat/              # Multi-valued treatment CBPS
│   └── test_multitreat.py
│
├── nonparametric/           # Nonparametric CBPS (CBGPS)
│   └── test_npcbps.py
│
├── optimal/                 # Optimal CBPS (Fan et al. 2022)
│   └── test_ocbps.py
│
├── sklearn/                 # scikit-learn estimator integration
│   └── test_estimator.py
│
└── utils/                   # Utility functions
    └── test_utils.py
```

## Paper-to-Test Mapping

| Paper | Module Tests | Monte Carlo |
|-------|-------------|-------------|
| Imai & Ratkovic (2014) JRSSB | `binary/`, `iv/`, `multitreat/` | `monte_carlo/test_imai2014.py` |
| Imai & Ratkovic (2015) JASA | `msm/` | `monte_carlo/test_ir2015.py` |
| Fong et al. (2018) AoAS | `continuous/`, `nonparametric/` | `monte_carlo/test_fong2018.py` |
| Ning et al. (2020) Biometrika | `highdim/` | `monte_carlo/test_ning2020.py` |
| Fan et al. (2022) JBES | `optimal/`, `inference/` | `monte_carlo/test_fan2022.py` |

## Running Tests

```bash
# All fast tests (CI/CD)
pytest tests/ -m "not slow" --tb=short

# All tests including Monte Carlo
pytest tests/ -v

# Specific module
pytest tests/binary/ -v
pytest tests/monte_carlo/test_imai2014.py -v

# By marker
pytest tests/ -m unit
pytest tests/ -m paper_reproduction
pytest tests/ -m edge_case

# With coverage
pytest tests/ --cov=cbps --cov-report=html -v
```

## Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual functions/classes |
| `integration` | End-to-end integration tests |
| `slow` | Tests with runtime > 10 seconds |
| `paper_reproduction` | Monte Carlo paper reproduction tests |
| `edge_case` | Edge case and boundary condition tests |
| `numerical` | Numerical precision validation tests |

## Tolerance Principles

Tolerances are calibrated using Monte Carlo Standard Error (MC SE):

```
MC SE = SD / sqrt(n_sims)
Tolerance = 3 * MC SE  (99.7% CI)
```

For unit/numerical tests, tolerances are based on machine epsilon and
expected numerical precision of each operation (see `conftest.py` for
the `Tolerances` class).

## References

1. Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
   JRSSB 76(1), 243-263. DOI: 10.1111/rssb.12027

2. Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability
   Weights for Marginal Structural Models. JASA 110(511), 1013-1023.
   DOI: 10.1080/01621459.2014.956872

3. Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
   Score for a Continuous Treatment. AoAS 12(1), 156-177.
   DOI: 10.1214/17-AOAS1101

4. Ning, Y., Peng, S., and Imai, K. (2020). Robust Estimation of Causal Effects
   via a High-Dimensional Covariate Balancing Propensity Score. Biometrika
   107(3), 533-554. DOI: 10.1093/biomet/asaa020

5. Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022). Optimal
   Covariate Balancing Conditions in Propensity Score Estimation. JBES 41(1),
   97-110. DOI: 10.1080/07350015.2021.2002159
