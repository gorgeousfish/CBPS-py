"""
Multi-Valued Treatment CBPS Test Suite
======================================

This package contains tests for the multi-valued treatment CBPS implementation,
which handles treatments with more than two levels using multinomial logistic
regression for propensity score estimation.

The multi-valued treatment CBPS extends the binary CBPS framework to categorical
treatments with 3 or more levels. It estimates generalized propensity scores
using multinomial logistic regression while optimizing covariate balance across
all treatment groups simultaneously.

Test Modules:
    - test_multitreat.py: Multi-valued treatment CBPS tests (MT-001 to MT-020)
        - 3-level treatment tests (cbps_3treat_fit)
        - 4-level treatment tests (cbps_4treat_fit)
        - Weight computation and balance assessment

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Multinomial coefficient estimation accuracy
    - Integration tests: End-to-end workflow with simulated data
    - Edge cases: Imbalanced treatment groups, small sample sizes

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    https://doi.org/10.1111/rssb.12027
"""
