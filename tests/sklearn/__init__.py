"""
Scikit-learn Integration Test Suite
===================================

This package contains tests for the scikit-learn compatible CBPS estimator,
verifying compliance with sklearn API conventions and integration with
sklearn pipelines.

The CBPSEstimator class wraps the core CBPS functionality in a scikit-learn
compatible interface, enabling:
    - Integration with sklearn Pipelines for preprocessing
    - Hyperparameter tuning via GridSearchCV/RandomizedSearchCV
    - Cross-validation for model selection
    - Compatibility with sklearn model selection utilities

Test Modules:
    - test_estimator.py: sklearn API compliance tests (SK-001 to SK-020)
        - fit/predict/predict_proba method tests
        - Pipeline integration tests
        - GridSearchCV compatibility tests
        - get_params/set_params tests

Test Categories:
    - Unit tests: Individual method validation
    - Integration tests: Pipeline and cross-validation workflows
    - API compliance tests: sklearn estimator checks

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

    scikit-learn estimator development guidelines:
    https://scikit-learn.org/stable/developers/develop.html
"""
