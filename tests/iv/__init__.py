"""
CBIV Test Suite
===============

This package contains comprehensive tests for the Covariate Balancing
Propensity Score for Instrumental Variable Estimates (CBIV) implementation.

CBIV extends the CBPS methodology to handle instrumental variable settings
with one-sided or two-sided noncompliance, enabling robust estimation of
local average treatment effects (LATE).

Test Modules:
    - test_cbiv.py: Core CBIV functionality tests (CBIV-001 to CBIV-035)

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Compliance probability estimation accuracy
    - Integration tests: End-to-end workflow with IV data
    - Edge cases: Boundary conditions and error handling

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

    Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability
    Weights for Marginal Structural Models. Journal of the American Statistical
    Association 110(511), 1013-1023.
"""
