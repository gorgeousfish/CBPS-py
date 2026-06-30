"""
High-Dimensional CBPS Test Suite
================================

This package contains tests for the high-dimensional CBPS (hdCBPS) implementation,
which handles scenarios where the number of covariates may exceed the sample size.

The hdCBPS algorithm from Ning et al. (2020) combines LASSO regularization with
covariate balancing to achieve doubly robust estimation in high-dimensional
settings. The four-step procedure includes:
    1. LASSO for propensity score variable selection
    2. LASSO for outcome model variable selection
    3. GMM covariate balancing with selected variables
    4. Treatment effect estimation with variance estimation

Test Modules:
    - test_main.py: Main hdCBPS algorithm tests (HD-001 to HD-050)
    - test_gmm_loss.py: GMM loss function tests (GMMLOSS-001 to GMMLOSS-025)
    - test_lasso_utils.py: LASSO regularization utility tests (LASSO-001 to LASSO-030)
    - test_weight_funcs.py: Weight function tests (HDWT-001 to HDWT-020)

Test Categories:
    - Unit tests: Individual function validation
    - Numerical tests: R package comparison and paper formula verification
    - Integration tests: End-to-end workflow with high-dimensional data
    - Edge cases: p > n scenarios, sparse signal recovery

References:
    Ning, Y., Peng, S., and Imai, K. (2020). Robust Estimation of Causal Effects
    via a High-Dimensional Covariate Balancing Propensity Score. Biometrika
    107(3), 533-554. https://doi.org/10.1093/biomet/asaa020

    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
