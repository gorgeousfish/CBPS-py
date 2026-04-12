"""
Optimal CBPS (oCBPS) Test Suite
===============================

This package contains comprehensive tests for the optimal Covariate Balancing
Propensity Score (oCBPS) implementation based on Fan et al. (2022).

The optimal CBPS extends the standard CBPS by incorporating dual balancing
conditions that achieve semiparametric efficiency bounds for average treatment
effect estimation. This provides doubly robust estimators with improved
efficiency compared to standard CBPS methods.

Test Modules:
    - test_ocbps.py: Core oCBPS functionality tests (OCBPS-001 to OCBPS-025)

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Dual balancing condition verification
    - Integration tests: End-to-end workflow with simulated data
    - Edge cases: Boundary conditions and error handling

References:
    Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
    Optimal Covariate Balancing Conditions in Propensity Score Estimation.
    Journal of Business & Economic Statistics 41(1), 97-110.
    https://doi.org/10.1080/07350015.2021.2002159

    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
