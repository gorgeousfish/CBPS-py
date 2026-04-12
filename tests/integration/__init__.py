"""
CBPS Integration Test Suite
===========================

This package contains end-to-end integration tests that verify complete
CBPS workflows from data preparation to treatment effect estimation.

Test Modules:
    - test_cbps_pipeline.py: Complete CBPS pipeline tests (INT-001 to INT-020)
        - Binary treatment pipeline
        - Continuous treatment pipeline
        - Multi-valued treatment pipeline
        - Cross-treatment comparison
    - test_cross_module.py: Cross-module integration tests
        - CBPS + diagnostics integration
        - CBPS + inference integration
        - Complete analysis pipeline

Test Categories:
    - Pipeline tests: Complete CBPS workflow validation
    - Cross-module tests: Integration between CBPS and diagnostics/inference
    - Real data tests: Validation with LaLonde and other datasets

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""
