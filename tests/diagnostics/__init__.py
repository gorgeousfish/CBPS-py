"""
Diagnostics Test Suite
======================

This package contains comprehensive tests for the CBPS diagnostics module,
including covariate balance assessment, plotting functions, and MSM diagnostics.

Test Modules:
    - test_balance.py: Covariate balance assessment tests for binary/multi-valued treatments
    - test_balance_cbmsm.py: Balance diagnostics for marginal structural models (CBMSM)
    - test_balance_enhanced.py: Enhanced balance reporting and format tests
    - test_plots.py: Visualization tests for balance plots (DIAG-PLOT-001 to DIAG-PLOT-025)
    - test_init.py: Module initialization and export tests

Test Categories:
    - Unit tests: Basic balance computation and SMD calculation
    - Numerical tests: Balance improvement verification
    - Visualization tests: Plot generation and rendering
    - Edge cases: Boundary conditions and error handling

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

    Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability
    Weights for Marginal Structural Models. Journal of the American Statistical
    Association 110(511), 1013-1023.
"""
