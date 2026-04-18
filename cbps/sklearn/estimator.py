"""
scikit-learn Compatible CBPS Estimator
======================================

This module provides a scikit-learn compatible wrapper for the CBPS estimator,
allowing the binary / multi-valued CBPS fit to sit inside a
:class:`~sklearn.pipeline.Pipeline` alongside standard preprocessors.

The wrapper exposes CBPS functionality through the standard sklearn API
(``fit``, ``predict``, ``predict_proba``, ``score``) while preserving access
to CBPS-specific outputs such as the propensity-score weights needed for
inverse probability weighting.

.. note::
   ``predict_proba`` and ``predict`` only return the stored training-sample
   propensity scores and raise ``ValueError`` on arrays with a different
   sample count. As a result, ``GridSearchCV`` / ``RandomizedSearchCV`` /
   ``cross_val_score`` with default scoring do **not** produce meaningful
   test-fold scores. Use a custom scorer driven by ``get_weights()`` if
   hyperparameter search is required, or fall back to
   :meth:`cbps.core.results.CBPSResults.predict` for out-of-sample
   propensity-score prediction.

References
----------
.. [1] Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
   Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
   https://doi.org/10.1111/rssb.12027
"""

from typing import Optional
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CBPSEstimator(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible wrapper for Covariate Balancing Propensity Score.

    This estimator wraps the CBPS methodology as a scikit-learn compatible
    classifier so the fit can sit inside a :class:`~sklearn.pipeline.Pipeline`
    alongside standard preprocessors. ``GridSearchCV`` and ``cross_val_score``
    with default scoring are **not** supported because ``predict_proba`` /
    ``predict`` only return the stored training-sample propensity scores
    (see the Notes section below).

    CBPS estimates propensity scores by simultaneously optimizing treatment
    prediction and covariate balance through the Generalized Method of Moments
    (GMM) framework.

    Parameters
    ----------
    att : {0, 1, 2}, default=1
        Target estimand for causal inference:

        - 0: Average Treatment Effect (ATE)
        - 1: Average Treatment Effect on the Treated (ATT), second level as treated
        - 2: ATT with first level as treated

        Multi-valued treatments (3-4 levels) only support att=0 (ATE).
    method : {'over', 'exact'}, default='over'
        GMM estimation method:

        - 'over': Over-identified GMM combining score function and balance conditions
        - 'exact': Just-identified GMM using balance conditions only
    two_step : bool, default=True
        If True, uses two-step GMM with pre-computed weight matrix (faster).
        If False, uses continuous updating GMM (better finite-sample properties).
    iterations : int, default=1000
        Maximum number of optimization iterations.
    standardize : bool, default=True
        If True, normalizes weights to sum to 1 within each treatment group.
        If False, returns Horvitz-Thompson weights.
    sample_weights : array-like of shape (n_samples,), optional
        Survey sampling weights. Defaults to uniform weights.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the model has been fitted.
    cbps_result_ : CBPSResults
        Complete CBPS result object containing coefficients, diagnostics,
        and convergence information.
    classes_ : ndarray of shape (n_classes,)
        Unique treatment levels observed during fitting.
    n_features_in_ : int
        Number of features seen during fit (excludes auto-added intercept).

    Notes
    -----
    **Limitations**

    - Supports array interface only; for formula interface use ``cbps.CBPS()``
    - Supports discrete treatments with 2-4 levels; for continuous treatments
      use ``cbps.CBPS()`` directly
    - ``predict_proba()`` returns stored training fitted values only; for
      prediction on new data, access ``cbps_result_.predict(newdata=...)``

    **Propensity Score Output**

    - Binary treatment: ``fitted_values`` is 1D array of shape (n,) representing P(T=1)
    - Multi-valued treatment (3-4 levels): ``fitted_values`` is 2D array of shape
      (n, K) where each row is a probability distribution over K treatment levels

    **Multi-valued Treatment**

    For treatments with 3-4 levels, the wrapper automatically converts numeric
    arrays to ``pd.Categorical`` to trigger multi-valued discrete CBPS (using
    multinomial logistic regression per Imai and Ratkovic 2014, Section 4.1).

    References
    ----------
    .. [1] Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
       Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
       https://doi.org/10.1111/rssb.12027

    Examples
    --------
    Basic usage with binary treatment:

    >>> from cbps.sklearn import CBPSEstimator
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde()
    >>> X = df[['age', 'educ', 're74', 're75']].values
    >>> y = df['treat'].values
    >>> est = CBPSEstimator(att=1, method='over')
    >>> est.fit(X, y)  # doctest: +ELLIPSIS
    CBPSEstimator(...)
    >>> weights = est.get_weights()
    >>> weights.shape
    (445,)

    Integration with sklearn Pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('cbps', CBPSEstimator(att=1))
    ... ])
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)
    """
    
    def __init__(
        self,
        att: int = 1,
        method: str = 'over',
        two_step: bool = True,
        iterations: int = 1000,
        standardize: bool = True,
        sample_weights: Optional[np.ndarray] = None
    ):
        # CBPS core parameters (array interface only)
        self.att = att
        self.method = method
        self.two_step = two_step
        self.iterations = iterations
        self.standardize = standardize
        self.sample_weights = sample_weights

    def fit(self, X, y):
        """Fit the CBPS model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix. An intercept column is automatically added
            if not present.

        y : array-like of shape (n_samples,)
            Treatment assignment vector with 2-4 unique discrete values.
            For 3-4 levels, numeric arrays are automatically converted to
            ``pd.Categorical`` to use multi-valued discrete CBPS.

        Returns
        -------
        self : CBPSEstimator
            Fitted estimator.

        Raises
        ------
        ValueError
            If X is not 2-dimensional.
            If y is not 1-dimensional.
            If X and y have different numbers of samples.
            If y has fewer than 2 or more than 4 unique values.
            If ``att != 0`` for treatments with 3-4 levels.
        """
        from cbps import CBPS
        import pandas as pd

        X = np.asarray(X)
        
        # Preserve original y for CBPS (may be pd.Categorical for multi-valued)
        y_original = y
        y_array = np.asarray(y)  # For validation only

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")

        if y_array.ndim != 1:
            raise ValueError(f"y must be a 1D array, got {y_array.ndim}D")

        if X.shape[0] != len(y_array):
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]} samples, "
                f"y has {len(y_array)} samples"
            )

        n_unique = len(np.unique(y_array))

        if n_unique < 2:
            raise ValueError(
                "Treatment variable must have at least 2 unique values"
            )

        if n_unique > 4:
            raise ValueError(
                f"CBPSEstimator supports discrete treatments with 2-4 levels. "
                f"Received {n_unique} unique values. For continuous treatments, "
                f"use cbps.CBPS() directly."
            )

        if n_unique >= 3 and self.att != 0:
            raise ValueError(
                f"Multi-valued treatment ({n_unique} levels) requires att=0 (ATE). "
                f"ATT estimation is only available for binary treatments."
            )

        # For multi-valued treatment, ensure categorical type
        # This triggers multi-valued discrete CBPS instead of continuous
        if n_unique >= 3 and not isinstance(y_original, pd.Categorical):
            y_original = pd.Categorical(y_original)

        # sklearn convention: store input feature count
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y_array)

        # Fit CBPS model (pass original y to preserve Categorical type)
        self.cbps_result_ = CBPS(
            treatment=y_original,
            covariates=X,
            att=self.att,
            method=self.method,
            two_step=self.two_step,
            iterations=self.iterations,
            standardize=self.standardize,
            sample_weights=self.sample_weights
        )

        self.fitted_ = True

        # Expose sklearn-standard coefficient attributes
        coefs = self.cbps_result_.coefficients
        if coefs.ndim == 2 and coefs.shape[1] == 1:
            # Binary treatment: (k, 1) -> intercept + coef_
            self.intercept_ = float(coefs[0, 0])
            self.coef_ = coefs[1:, 0]
        elif coefs.ndim == 2 and coefs.shape[1] > 1:
            # Multi-valued treatment: (k, J-1)
            self.intercept_ = coefs[0, :]
            self.coef_ = coefs[1:, :]
        else:
            self.intercept_ = float(coefs.ravel()[0])
            self.coef_ = coefs.ravel()[1:]

        return self

    def predict_proba(self, X):
        """Return estimated propensity scores for observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix. Must have the same number of samples as the
            training data. The actual values are not used; this parameter
            exists for sklearn API compatibility.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Propensity score matrix. For binary treatment, column 0 contains
            P(T=0) and column 1 contains P(T=1). For multi-valued treatment,
            each column k contains P(T=k).

        Raises
        ------
        ValueError
            If the number of samples in X differs from the training set size.

        Warns
        -----
        UserWarning
            Always issued to remind that this method returns stored fitted
            values rather than predictions on new data.
        """
        check_is_fitted(self, 'fitted_')

        X = np.asarray(X)
        n_samples_X = X.shape[0]
        n_samples_train = len(self.cbps_result_.fitted_values)

        if n_samples_X != n_samples_train:
            raise ValueError(
                f"Sample count mismatch: X has {n_samples_X} samples, but the "
                f"model was fitted on {n_samples_train} samples. "
                f"predict_proba() only returns fitted values for training data."
            )

        warnings.warn(
            "predict_proba() returns stored fitted values from training. "
            "For prediction on new data, use self.cbps_result_.predict(newdata=...).",
            UserWarning,
            stacklevel=2
        )

        fitted_values = self.cbps_result_.fitted_values

        if len(self.classes_) == 2:
            # Binary: fitted_values is P(T=1), convert to (n, 2) matrix
            proba = np.column_stack([1 - fitted_values, fitted_values])
        else:
            # Multi-valued: fitted_values is already (n, K) matrix
            proba = fitted_values

        return proba

    def predict(self, X):
        """Predict treatment assignment based on maximum propensity score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix. Must match the training data sample count.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted treatment class for each observation, determined by
            the treatment level with highest estimated propensity.

        Notes
        -----
        This method returns the treatment level with the maximum estimated
        propensity score for each observation. It is provided for sklearn
        API compatibility but has limited practical utility since CBPS
        propensity scores are estimated for weighting purposes, not
        classification.

        See Also
        --------
        predict_proba : Return probability estimates.
        get_weights : Return IPW weights (primary CBPS output).
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_weights(self):
        """Return inverse probability weights for causal effect estimation.

        Returns
        -------
        weights : ndarray of shape (n_samples,)
            Covariate balancing weights. When ``standardize=True`` (default),
            weights sum to 1 within each treatment group. Otherwise,
            Horvitz-Thompson weights are returned.

        Notes
        -----
        These weights are the primary output of CBPS estimation, designed
        for use in weighted outcome regressions or Horvitz-Thompson estimators
        to obtain unbiased estimates of causal effects.

        For ATE estimation (``att=0``), all observations receive positive
        weights. For ATT estimation (``att=1`` or ``att=2``), control group
        observations are reweighted to match the treated group's covariate
        distribution.

        Examples
        --------
        >>> est = CBPSEstimator(att=1).fit(X, y)  # doctest: +SKIP
        >>> weights = est.get_weights()  # doctest: +SKIP
        >>> # Use weights in outcome regression
        >>> from sklearn.linear_model import LinearRegression
        >>> outcome_model = LinearRegression()  # doctest: +SKIP
        >>> outcome_model.fit(X, outcome, sample_weight=weights)  # doctest: +SKIP
        """
        check_is_fitted(self, 'fitted_')
        return self.cbps_result_.weights

