"""
Nonparametric CBPS Test Suite
=============================

This package contains comprehensive tests for the nonparametric CBPS (npCBPS)
implementation based on empirical likelihood methods.

Test Modules:
    - test_taylor_approx.py: Taylor approximation functions (llog, llogp)
    - test_cholesky_whitening.py: Cholesky whitening transformation
    - test_empirical_likelihood.py: Empirical likelihood optimization
    - test_constraint_matrix.py: Constraint matrix construction
    - test_npcbps_fit.py: Main optimization flow
    - test_npcbps_api.py: High-level API tests
    - test_npcbps_results.py: NPCBPSResults class tests
    - test_npcbps_balance.py: Balance diagnostics tests
    - test_npcbps_plots.py: Visualization tests
    - test_npcbps_edge_cases.py: Edge case and boundary tests
    - test_npcbps_regression.py: Regression tests for known bugs

Test Categories:
    - Unit tests: Individual function verification
    - Numerical tests: R package comparison, mathematical properties
    - Integration tests: End-to-end workflow validation
    - Edge case tests: Small samples, high dimensions, near-singular matrices

References:
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics 12(1), 156-177.
"""
