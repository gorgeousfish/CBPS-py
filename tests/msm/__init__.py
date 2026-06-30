"""
CBMSM Test Suite
================

This package contains comprehensive tests for the Covariate Balancing
Propensity Score for Marginal Structural Models (CBMSM) implementation.

CBMSM extends the CBPS methodology to handle longitudinal/panel data with
time-varying treatments. It estimates inverse probability weights that
balance both baseline and time-varying covariates across treatment histories,
enabling robust estimation of causal effects under the sequential ignorability
assumption.

Test Modules:
    - test_cbmsm.py: Core CBMSM functionality tests (CBMSM-001 to CBMSM-025)

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Integration tests: End-to-end workflow with panel data
    - Numerical tests: Weight computation and balance verification
    - Edge cases: Boundary conditions and error handling

References:
    Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability
    Weights for Marginal Structural Models. Journal of the American Statistical
    Association 110(511), 1013-1023.
    https://doi.org/10.1080/01621459.2014.956872

    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
