scikit-learn Integration
=========================

The ``cbps.sklearn`` subpackage exposes a scikit-learn compatible wrapper
around the binary / multi-valued CBPS estimator. The wrapper inherits from
``BaseEstimator`` and ``ClassifierMixin`` and can be used inside
``Pipeline`` / ``FeatureUnion`` constructs.

.. note::
   The sklearn wrapper only supports discrete treatments with 2-4 levels.
   For continuous treatments, use :func:`cbps.CBPS` directly.

.. warning::
   ``CBPSEstimator.predict_proba`` and ``CBPSEstimator.predict`` return the
   stored training-sample propensity scores only and raise ``ValueError``
   on arrays with a different sample count. As a consequence,
   ``GridSearchCV`` / ``RandomizedSearchCV`` / ``cross_val_score`` with
   default scoring do **not** produce meaningful test-fold scores. Provide
   a custom scorer (for example, a callable that inspects
   ``estimator.get_weights()`` on the refit estimator) if hyperparameter
   search is required, or call
   :meth:`cbps.core.results.CBPSResults.predict` for out-of-sample
   propensity-score prediction.

Estimator
---------

.. autoclass:: cbps.sklearn.CBPSEstimator
   :members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_score_request, set_predict_request, set_predict_proba_request
