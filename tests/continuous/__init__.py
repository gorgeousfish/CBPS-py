"""
Continuous Treatment CBPS Test Suite
====================================

This package contains comprehensive tests for the continuous treatment CBPS
(Covariate Balancing Propensity Score) implementation, which extends the
standard CBPS framework to handle continuous treatment variables through
the Generalized Propensity Score (GPS) methodology.

Test Modules:
    - test_continuous.py: Core continuous CBPS functionality tests (CT-001 to CT-025)
    - conftest.py: Shared fixtures and tolerance configurations

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: GPS weight computation accuracy
    - Integration tests: End-to-end workflow with simulated data
    - Edge cases: Boundary conditions and numerical stability

Methodology:
    The continuous CBPS estimates the generalized propensity score by maximizing
    covariate balance. The method involves:
    1. Cholesky whitening of covariates with sample weights
    2. Log-space normal density computation for numerical stability
    3. GMM optimization with multiple starting values
    4. Coefficient inverse transformation from whitened to original space
"""
