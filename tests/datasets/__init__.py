"""
Datasets Test Suite
===================

This package contains comprehensive tests for the CBPS dataset loaders,
covering the LaLonde, Blackwell, and simulated datasets.

Test Modules:
    - test_datasets.py: Dataset loading and integrity tests (DS-001 to DS-025)
        - LaLonde NSW/CPS/PSID dataset tests
        - Blackwell longitudinal dataset tests
        - Continuous treatment simulation tests
        - npCBPS simulation dataset tests
    - test_init.py: Module initialization and export tests

Test Categories:
    - Unit tests: Basic loading functionality
    - Data integrity tests: Column presence, data types, value ranges
    - Cross-dataset tests: Consistency across related datasets

References:
    LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
    programs with experimental data. American Economic Review, 76(4), 604-620.

    Blackwell, M. (2013). A Framework for Dynamic Causal Inference in Political
    Science. American Journal of Political Science, 57(2), 504-520.
"""
