"""
CBPS Test Suite
===============

Comprehensive tests for the CBPS (Covariate Balancing Propensity Score)
Python package, implementing methods from five foundational papers in
causal inference methodology.

This test suite is designed for academic software validation, providing
thorough coverage of all CBPS methods with numerical precision benchmarks
against the original R implementation.

Test Modules
------------
Core Methods:
    - binary/: Binary treatment CBPS (ATT/ATE estimation)
    - continuous/: Continuous treatment CBPS (GPS-based)
    - multitreat/: Multi-valued treatment CBPS

Advanced Methods:
    - msm/: Marginal Structural Models (CBMSM)
    - iv/: Instrumental Variables (CBIV)
    - optimal/: Optimal CBPS (oCBPS)
    - nonparametric/: Nonparametric CBPS (npCBPS)
    - highdim/: High-dimensional CBPS (hdCBPS)

Supporting Modules:
    - inference/: Statistical inference (variance estimation)
    - sklearn/: scikit-learn API compatibility

Test Organization
-----------------
Within each module, tests are organized by type:
    - unit/: Unit tests for individual functions
    - integration/: End-to-end integration tests
    - edge_cases/: Boundary condition tests
    - numerical/: Paper formula verification tests
    - regression/: Regression tests for known bugs

Test Markers
------------
Tests can be filtered using pytest markers:
    - @pytest.mark.unit: Fast unit tests
    - @pytest.mark.integration: Integration tests
    - @pytest.mark.numerical: Numerical precision tests
    - @pytest.mark.paper_reproduction: Paper reproduction tests
    - @pytest.mark.slow: Slow tests (>10 seconds)
    - @pytest.mark.edge_case: Edge case tests

Usage
-----
Run all tests:
    pytest tests/ -v

Skip slow tests:
    pytest tests/ -m "not slow"

Run specific module:
    pytest tests/binary/ -v

Run with coverage:
    pytest tests/ --cov=cbps --cov-report=term-missing
"""
