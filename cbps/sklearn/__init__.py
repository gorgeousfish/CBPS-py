"""
scikit-learn Integration
========================

This module provides scikit-learn compatible wrappers for CBPS estimators,
enabling seamless integration with the sklearn ecosystem.

Classes
-------
CBPSEstimator
    A scikit-learn compatible wrapper for discrete treatment CBPS that
    inherits from ``BaseEstimator`` and ``ClassifierMixin``.

Features
--------
- Full compatibility with sklearn's ``Pipeline`` and ``FeatureUnion``
- ``fit``, ``predict``, ``predict_proba``, ``score`` and ``get_weights``
  surface the CBPS fit through the standard sklearn API
- Access to CBPS weights through ``get_weights()`` for downstream analysis

Limitations
-----------
- Supports discrete treatments with 2-4 levels; for continuous treatments
  use ``cbps.CBPS()`` directly
- Out-of-sample prediction is not implemented: ``predict_proba`` and
  ``predict`` only return the stored training-sample propensity scores and
  raise ``ValueError`` on arrays with a different sample count. As a
  consequence, ``GridSearchCV``, ``RandomizedSearchCV`` and
  ``cross_val_score`` with default scoring do **not** produce meaningful
  test-fold scores; pass a custom ``scoring`` callable that evaluates
  ``get_weights()`` on the refit estimator if hyperparameter search is
  desired. For out-of-sample propensity-score prediction, use
  ``cbps.CBPS().predict(newdata=...)`` on the wrapped ``cbps_result_``
- Only array interface is available; formula interface requires ``cbps.CBPS()``

See Also
--------
cbps.CBPS : Main CBPS function with full feature support.

References
----------
.. [1] Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
   Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
   https://doi.org/10.1111/rssb.12027
"""

from cbps.sklearn.estimator import CBPSEstimator

__all__ = ['CBPSEstimator']
