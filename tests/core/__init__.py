"""
Core Test Suite
===============

This package contains comprehensive tests for the CBPS core module components,
including the CBPSResults class and core fit functions.

Test Modules:
    - test_results.py: CBPSResults and CBPSSummary class tests (RES-001 to RES-050)
        - Initialization and attribute tests
        - Prediction method tests
        - Summary and string representation tests
        - Variance-covariance matrix tests
    - test_core_init.py: Core module initialization and export tests
        - Module-level exports verification
        - API signature tests
        - Fit function availability tests

Test Categories:
    - Unit tests: Class initialization, method functionality
    - Property tests: Property getters and computed attributes
    - Prediction tests: predict() method with various inputs
    - Summary tests: summary() method and CBPSSummary class
    - Edge cases: Error handling and boundary conditions

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
