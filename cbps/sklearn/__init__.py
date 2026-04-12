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
- Hyperparameter tuning via ``GridSearchCV`` and ``RandomizedSearchCV``
- Cross-validation support for model selection
- Access to CBPS weights through ``get_weights()`` for downstream analysis

Limitations
-----------
- Supports discrete treatments with 2-4 levels; for continuous treatments
  use ``cbps.CBPS()`` directly
- Out-of-sample prediction via ``predict_proba()`` is not implemented;
  for prediction on new data, use ``cbps.CBPS().predict(newdata=...)``
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
