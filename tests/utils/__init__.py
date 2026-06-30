"""
Utils Test Suite
================

This package contains comprehensive tests for the CBPS utility modules,
covering formula parsing, weight computation, numerical operations, and validation.

Test Modules:
    - test_formula.py: Formula parsing tests (patsy interface)
    - test_weights.py: Weight computation tests (ATE/ATT weights)
    - test_numerics.py: Numerical utilities tests (pseudoinverse, matrix operations)
    - test_validation.py: Input validation tests
    - test_helpers.py: Helper function tests (data preprocessing, encoding)
    - test_variance_transform.py: Variance transformation tests (SVD, multitreat)
    - test_r_compat.py: R compatibility layer tests
    - test_init.py: Module initialization and export tests

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Numerical accuracy and precision verification
    - R compatibility tests: Cross-language consistency
    - Edge cases: Boundary conditions and error handling

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
