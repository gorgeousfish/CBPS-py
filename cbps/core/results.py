"""
Result Classes for Covariate Balancing Propensity Score Estimation

================================================================

This module implements the primary result containers for CBPS estimators,
providing a unified interface for accessing estimation results, conducting
statistical inference, and performing diagnostic assessments.

The module contains two main classes:

- :class:`CBPSResults`: Main result object containing all fitted model components
- :class:`CBPSSummary`: Statistical summary with coefficient table and diagnostics

These classes implement a comprehensive statistical modeling interface with
methods for inference, prediction, and diagnostic evaluation, maintaining
compatibility with established statistical software conventions while following
Python best practices.

Mathematical Framework
---------------------

The CBPS estimator solves the generalized method of moments (GMM) optimization
problem:

    min_β ḡ(β)' Σ^(-1) ḡ(β)

where ḡ(β) is the sample average of moment conditions combining:

1. Score function: ∂ℓ(β)/∂β for treatment prediction
2. Balance conditions: (T_i - e(X_i,β))X_i for covariate balance

The resulting weights are:

    w_i = T_i/e(X_i,β) - (1-T_i)/(1-e(X_i,β))

which satisfy the moment conditions E[w_i X_i] = 0 when correctly specified.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
https://doi.org/10.1111/rssb.12027
"""

from typing import Optional, List
import numpy as np
import scipy.stats
import warnings


class CBPSResults:
    """
    Result object from CBPS estimation.
    
    This class encapsulates all outputs from the CBPS fitting procedure,
    providing a unified interface for accessing coefficients, weights,
    fitted propensity scores, and diagnostic statistics.
    
    Attributes
    ----------
    coefficients : ndarray, shape (k, 1) or (k, n_treats-1)
        Estimated propensity score model coefficients.
        For binary treatment: (k, 1) matrix.
        For multi-valued treatment: (k, n_treats-1) matrix.
    fitted_values : ndarray, shape (n,)
        Fitted propensity scores for each observation.
    weights : ndarray, shape (n,)
        Optimal inverse probability weights for causal effect estimation.
    linear_predictor : ndarray, shape (n,)
        Linear predictor X @ beta before link function transformation.
    y : ndarray, shape (n,)
        Treatment assignment vector.
    x : ndarray, shape (n, k)
        Covariate matrix including intercept.
    J : float
        Hansen J-statistic for the GMM over-identification test.
    mle_J : float
        J-statistic evaluated at MLE estimates (baseline comparison).
    deviance : float
        Model deviance (-2 * log-likelihood).
    converged : bool
        Whether the optimization algorithm converged successfully.
    var : ndarray, shape (k, k)
        Variance-covariance matrix of coefficients (sandwich estimator).
    coef_names : list of str
        Names of coefficients extracted from the model formula.
    call_info : str
        String representation of the function call.
    formula : str or None
        The model formula used for fitting (if formula interface was used).
    att : int or None
        Target estimand: 0 for ATE, 1 for ATT.
    method : str or None
        Estimation method: 'over' (over-identified) or 'exact' (just-identified).
    standardize : bool or None
        Whether weights are standardized to sum to sample size.
    two_step : bool or None
        Whether two-step GMM estimator was used.
    sigmasq : float or None
        Residual variance estimate (continuous treatment only).
    Ttilde : ndarray or None
        Standardized treatment (zero mean, unit variance) for continuous treatment.
        Used by vcov_outcome for variance estimation.
    Xtilde : ndarray or None
        Cholesky-whitened covariates for continuous treatment.
        Used by vcov_outcome for variance estimation.
    beta_tilde : ndarray or None
        Coefficients in whitened space for continuous treatment.
    sigmasq_tilde : float or None
        Variance in whitened space for continuous treatment.
    treat_names : list of str or None
        Treatment level names for multi-valued treatment.
        Example: ['Control', 'Low', 'High'] for 3-valued treatment.
    na_action : dict or None
        Missing value handling information containing:
        - 'method': handling method ('omit', 'fail', 'ignore')
        - 'n_dropped': number of dropped observations (only for method='omit')

    Examples
    --------
    >>> fit = CBPS('treat ~ age + educ', data=lalonde, att=1)
    >>> summ = fit.summary()  # Compute summary statistics
    >>> print(summ)  # Print full coefficient table
    >>> vcov_mat = fit.vcov()  # Get variance-covariance matrix
    >>> print(fit)  # Concise output
    """
    
    def __init__(
        self,
        # Core estimation results
        coefficients: np.ndarray,
        fitted_values: np.ndarray,
        weights: np.ndarray,
        linear_predictor: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        J: float,
        mle_J: float,
        deviance: float,
        converged: bool,
        var: np.ndarray,
        nulldeviance: Optional[float] = None,
        
        # Metadata
        coef_names: Optional[List[str]] = None,
        call_info: Optional[str] = None,
        formula: Optional[str] = None,
        data: Optional[object] = None,
        terms: Optional[object] = None,
        model: Optional[object] = None,
        xlevels: Optional[dict] = None,
        
        # Input parameters
        att: Optional[int] = None,
        method: Optional[str] = None,
        standardize: Optional[bool] = None,
        two_step: Optional[bool] = None,
        
        # Continuous treatment specific
        sigmasq: Optional[float] = None,
        Ttilde: Optional[np.ndarray] = None,
        Xtilde: Optional[np.ndarray] = None,
        beta_tilde: Optional[np.ndarray] = None,
        sigmasq_tilde: Optional[float] = None,
        stabilizers: Optional[np.ndarray] = None,

        # Multi-valued treatment specific
        treat_names: Optional[List[str]] = None,

        # Missing data handling
        na_action: Optional[dict] = None
    ):
        """
        Initialize CBPS result object.
        
        Parameters
        ----------
        coefficients : ndarray
            Coefficient matrix, shape (k, 1) for binary or (k, n_treats-1) for multi-valued.
        fitted_values : ndarray
            Fitted propensity scores, shape (n,).
        weights : ndarray
            Optimal IPW weights, shape (n,).
        linear_predictor : ndarray
            Linear predictor X @ beta, shape (n,).
        y : ndarray
            Treatment vector (original), shape (n,).
        x : ndarray
            Covariate matrix (with intercept), shape (n, k).
        J : float
            Hansen J-statistic (over-identification test).
        mle_J : float
            MLE baseline J-statistic.
        deviance : float
            Negative 2 times log-likelihood.
        converged : bool
            Optimization convergence status.
        var : ndarray
            Coefficient variance-covariance matrix, shape (k, k).
        nulldeviance : float, optional
            Null model deviance for pseudo R-squared calculation.
        coef_names : list, optional
            Coefficient names from formula.
        call_info : str, optional
            Call information string.
        formula : str, optional
            Model formula string.
        
        Notes
        -----
        All parameters are typically passed from internal fitting routines
        and should not be constructed manually by users.
        
        The coefficients must be a 2D matrix:
        - Binary treatment: (k, 1)
        - 3-valued treatment: (k, 2)
        - 4-valued treatment: (k, 3)
        """
        # Core estimation results
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.weights = weights
        self.linear_predictor = linear_predictor
        self.y = y
        self.x = x
        self.J = J
        self.mle_J = mle_J
        self.deviance = deviance
        self.nulldeviance = nulldeviance
        self.converged = converged
        self.var = var
        
        # Metadata
        self.call_info = call_info or "CBPS()"
        self.call = call_info or "CBPS()"
        self.coef_names = coef_names or self._default_coef_names()
        self.formula = formula
        self.data = data
        self.terms = terms
        self.model = model
        self.xlevels = xlevels
        
        # Input parameters
        self.att = att
        self.method = method
        self.standardize = standardize
        self.two_step = two_step
        
        # Continuous treatment specific
        self.sigmasq = sigmasq
        self.Ttilde = Ttilde
        self.Xtilde = Xtilde
        self.beta_tilde = beta_tilde
        self.sigmasq_tilde = sigmasq_tilde
        self.stabilizers = stabilizers
        
        # Multi-valued treatment specific
        self.treat_names = treat_names

        # Missing data handling
        self.na_action = na_action

        # Validate coefficients shape
        if self.coefficients.ndim != 2:
            raise ValueError(
                f"coefficients must be 2D array, got shape {self.coefficients.shape}. "
                f"Expected (k, 1) for binary or (k, n_treats-1) for multi-valued."
            )
    
    def _default_coef_names(self) -> List[str]:
        """Generate default coefficient names when none are provided."""
        k = self.coefficients.shape[0]
        if k == 0:
            return []
        return ["Intercept"] + [f"X{i}" for i in range(1, k)]
    
    def vcov(self) -> np.ndarray:
        """
        Return the variance-covariance matrix of the estimated coefficients.
        
        Returns
        -------
        ndarray, shape (k, k)
            Variance-covariance matrix computed using the sandwich estimator.
        
        Raises
        ------
        ValueError
            If the variance matrix was not computed during fitting (var is None).
        
        Warns
        -----
        UserWarning
            If the condition number exceeds 1e10, indicating potential
            near-collinearity that may affect standard error reliability.
        
        Notes
        -----
        This method directly returns the stored variance matrix computed during
        fitting using the sandwich formula. It does not recompute the matrix.
        
        The variance matrix is computed as:
        ``vcov = (G' W G)^{-1} G' W Omega W' G (G' W G)^{-1}``
        
        where G is the gradient matrix, W is the weighting matrix, and Omega
        is the covariance of the moment conditions.
        """
        if self.var is None:
            raise ValueError(
                "Variance-covariance matrix not computed. "
                "This may indicate a fitting error."
            )
        
        # Check condition number to detect near-collinearity
        try:
            cond_number = np.linalg.cond(self.var)
            if cond_number > 1e10:
                warnings.warn(
                    f"Variance-covariance matrix has high condition number ({cond_number:.2e}). "
                    f"This suggests near-collinearity among covariates. "
                    f"Standard errors may be unreliable. "
                    f"Consider:\n"
                    f"  1. Removing highly correlated covariates\n"
                    f"  2. Using regularization (e.g., hdCBPS)\n"
                    f"  3. Checking for perfect collinearity with np.linalg.matrix_rank(X)",
                    UserWarning,
                    stacklevel=2
                )
        except np.linalg.LinAlgError:
            warnings.warn(
                "Failed to compute condition number of variance-covariance matrix. "
                "Matrix may be singular or near-singular.",
                UserWarning,
                stacklevel=2
            )
        
        return self.var
    
    @property
    def residuals(self) -> np.ndarray:
        """
        Model residuals (observed minus fitted values).
        
        Returns
        -------
        ndarray
            Residual vector or matrix depending on treatment type:
            
            - Binary: y - fitted_values, shape (n,)
            - Continuous: standardized residuals in whitened space, shape (n,)
            - Multi-valued: one-hot encoded y minus fitted probabilities, shape (n, k)
        """
        # Continuous treatment
        if self.Ttilde is not None:
            return self.Ttilde - self.linear_predictor.ravel()
        
        # Binary treatment
        if self.fitted_values.ndim == 1 or (self.fitted_values.ndim == 2 and self.fitted_values.shape[1] == 1):
            return self.y - self.fitted_values.ravel()
            
        # Multi-valued treatment
        n_samples = len(self.y)
        n_classes = self.fitted_values.shape[1]
        y_onehot = np.zeros((n_samples, n_classes))
        
        try:
            y_int = self.y.astype(int)
            if y_int.min() >= 0 and y_int.max() < n_classes:
                y_onehot[np.arange(n_samples), y_int] = 1
                return y_onehot - self.fitted_values
        except Exception:
            pass
            
        raise NotImplementedError(
            "Residuals not supported for this multi-valued treatment format"
        )

    @property
    def pseudo_r2(self) -> Optional[float]:
        """
        McFadden's pseudo R-squared measure of model fit.
        
        Returns
        -------
        float or None
            Pseudo R² = 1 - deviance / null_deviance.
            Returns None if null deviance is unavailable or zero.
        
        Notes
        -----
        The pseudo R² measures improvement over the null (intercept-only) model:
        
        - 0: No improvement over null model
        - 1: Perfect fit
        - Typical range: 0.05-0.40 for logistic models
        
        The null model contains only the intercept and predicts all observations
        with probability equal to the sample mean.
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=data)
        >>> print(f"Pseudo R²: {fit.pseudo_r2:.4f}")
        """
        if self.nulldeviance is None or self.nulldeviance == 0:
            return None
        return 1.0 - self.deviance / self.nulldeviance
    
    def balance(self, **kwargs):
        """
        Compute covariate balance statistics.
        
        This is a convenience method that calls the standalone ``balance()``
        function. Both ``fit.balance()`` and ``balance(fit)`` are supported,
        allowing users to choose either object-oriented or functional style.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to ``balance()``:
            
            - enhanced : bool, default=False
                Whether to return enhanced diagnostic information.
            - threshold : float, default=0.1
                Imbalance threshold (SMD or correlation) for flagging covariates.
            - covariate_names : list, optional
                Covariate names for enhanced output.
        
        Returns
        -------
        dict
            Dictionary containing balance statistics. Keys depend on the
            treatment type:

            - Binary / multi-valued: ``'balanced'`` and ``'original'``
              (standardised mean differences).
            - Continuous: ``'balanced'`` and ``'unweighted'``
              (absolute Pearson correlations).

            See ``cbps.balance()`` documentation for full details. For CBMSM
            results, use ``CBMSMResults.balance()`` which instead returns
            capitalised keys (``'Balanced'`` / ``'Unweighted'`` / ``'StatBal'``)
            to mirror the R CBPS package.
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=df, att=1)
        >>> 
        >>> # Method 1: Standalone function
        >>> from cbps import balance
        >>> bal = balance(fit)
        >>> 
        >>> # Method 2: Object method (Python style)
        >>> bal = fit.balance()
        >>> 
        >>> # Both methods produce identical results
        """
        from cbps import balance as balance_func
        
        cbps_dict = {
            'weights': self.weights,
            'x': self.x,
            'y': self.y,
            'fitted_values': self.fitted_values,
            'coefficients': self.coefficients
        }
        
        return balance_func(cbps_dict, **kwargs)
    
    @property
    def coef(self) -> np.ndarray:
        """
        Coefficient vector (1D convenience accessor).
        
        Returns
        -------
        ndarray, shape (k,)
            Coefficient vector (1D), extracted from the coefficients matrix.
            
        Notes
        -----
        This is a convenience property providing a 1D view of coefficients.
        
        - For binary treatment: returns ``coefficients[:, 0]`` (1D)
        - For multi-valued treatment: returns ``coefficients[:, 0]`` (first contrast)
        
        The full coefficient matrix is still accessible via ``fit.coefficients``.
        
        Comparison with other Python packages:
        
        - statsmodels: ``result.params`` (1D)
        - sklearn: ``model.coef_`` (may be 2D)
        - CBPS: ``fit.coef`` (1D, this property) + ``fit.coefficients`` (full 2D)
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=df, att=1)
        >>> fit.coef  # Convenient 1D access
        array([0.123, 0.456, -0.789])
        >>> fit.coefficients  # Full 2D matrix
        array([[0.123],
               [0.456],
               [-0.789]])
        """
        if self.coefficients.ndim == 1:
            return self.coefficients
        else:
            return self.coefficients[:, 0] if self.coefficients.shape[1] == 1 else self.coefficients.ravel()
    
    @property
    def fitted(self) -> np.ndarray:
        """
        Alias for fitted_values (alternative accessor).
        
        Returns
        -------
        ndarray
            Fitted propensity scores, equivalent to ``fitted_values``.
            
        Notes
        -----
        This is an alias for ``fitted_values`` for convenience.
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=df)
        >>> # The following are equivalent
        >>> fv1 = fit.fitted_values
        >>> fv2 = fit.fitted
        >>> np.allclose(fv1, fv2)
        True
        """
        return self.fitted_values
    
    @property
    def J_stat(self) -> float:
        """
        Alias for J (Hansen's J-statistic).
        
        Returns
        -------
        float
            The GMM over-identification test statistic.
            
        Notes
        -----
        The J-statistic is used for the GMM over-identification test.
        Under the null hypothesis of correct model specification, J is
        asymptotically chi-squared distributed with degrees of freedom
        equal to the number of over-identifying restrictions.
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=df)
        >>> j1 = fit.J        # Original attribute
        >>> j2 = fit.J_stat   # Alias
        >>> assert j1 == j2
        """
        return self.J
    
    @property
    def sigma_squared(self) -> Optional[float]:
        """
        Residual variance estimate (continuous treatment only).
        
        Returns
        -------
        float or None
            Variance estimate for continuous treatment models.
            Returns None for binary or multi-valued treatments.
            
        Notes
        -----
        Only available for continuous treatment CBPS. For binary and
        multi-valued treatments, this property returns None.
        
        Examples
        --------
        >>> # Continuous treatment
        >>> fit_cont = CBPS('dose ~ age + educ', data=df)
        >>> sigma2 = fit_cont.sigma_squared
        >>>
        >>> # Binary treatment
        >>> fit_bin = CBPS('treat ~ age + educ', data=df)
        >>> assert fit_bin.sigma_squared is None
        """
        return getattr(self, 'sigmasq', None)
    
    def predict(self, newdata=None, type='response'):
        """
        Predict propensity scores for new data.
        
        Parameters
        ----------
        newdata : DataFrame, ndarray, or None
            New data for prediction. If None, returns fitted values from
            the training data.
            
            - DataFrame: Required when the model was fit with the formula
              interface; the stored patsy ``DesignInfo`` is used to build
              the new design matrix (intercept added automatically).
            - ndarray: Must have the same number of columns as ``self.x``,
              **including the intercept column** if one was added during
              fitting. The column ordering must also match the design matrix
              used at fitting time. Passing a matrix that is missing the
              intercept will raise ``ValueError``.
        
        type : {'response', 'link'}, default='response'
            Type of prediction:
            
            - 'response': Probabilities/expected values (after link function)
            - 'link': Linear predictor X @ beta (before transformation)
        
        Returns
        -------
        ndarray
            Predicted values. Shape depends on treatment type:
            
            - Binary: (n_new,) probabilities
            - Multi-valued: (n_new, n_levels) probabilities
            - Continuous: (n_new,) conditional means
        
        Raises
        ------
        ValueError
            If type is invalid or newdata dimensions do not match.
        
        Notes
        -----
        **Treatment type handling:**
        
        - Binary treatment: logistic link (expit)
        - Continuous treatment: identity link
        - Multi-valued treatment: multinomial logistic (softmax)
        
        **Formula vs array interface:**
        
        - Formula interface: uses patsy DesignInfo to rebuild design matrix
        - Array interface: directly uses newdata as covariate matrix
        
        Examples
        --------
        >>> # Train model
        >>> fit = CBPS('treat ~ x1 + x2', data=train_df)
        >>>
        >>> # Predict new data
        >>> pred = fit.predict(test_df)
        >>>
        >>> # Predict linear predictor
        >>> linear_pred = fit.predict(test_df, type='link')
        >>>
        >>> # Get training data fitted values
        >>> fitted = fit.predict()  # Equivalent to fit.fitted_values
        """
        valid_types = {'response', 'link'}
        if type not in valid_types:
            raise ValueError(
                f"Invalid type: '{type}'. Must be one of {valid_types}."
            )
        
        if newdata is None:
            if type == 'response':
                return self.fitted_values
            elif type == 'link':
                return self.linear_predictor
        
        X_new = self._prepare_newdata(newdata)
        linear_pred = X_new @ self.coefficients
        
        if type == 'link':
            if linear_pred.ndim == 2 and linear_pred.shape[1] == 1:
                return linear_pred.ravel()
            return linear_pred
        elif type == 'response':
            return self._apply_link_function(linear_pred)
    
    def _prepare_newdata(self, newdata):
        """Prepare design matrix from new data for prediction."""
        import pandas as pd
        
        if self.formula is not None and self.terms is not None:
            if not isinstance(newdata, pd.DataFrame):
                raise TypeError(
                    f"When using formula interface, newdata must be a DataFrame. "
                    f"Got {type(newdata).__name__}."
                )
            
            try:
                from patsy import dmatrix
                X_new_df = dmatrix(self.terms, newdata, return_type='dataframe')
                X_new = X_new_df.values
            except Exception as e:
                raise ValueError(
                    f"Failed to build design matrix from newdata using formula '{self.formula}'. "
                    f"Error: {str(e)}\n"
                    f"Make sure newdata contains all variables used in the formula."
                ) from e
        else:
            X_new = np.asarray(newdata)
            if X_new.ndim == 1:
                X_new = X_new.reshape(1, -1)
            
            if X_new.shape[1] != self.x.shape[1]:
                raise ValueError(
                    f"newdata has {X_new.shape[1]} columns, "
                    f"but model was trained with {self.x.shape[1]} columns. "
                    f"Expected shape: (n_new, {self.x.shape[1]})"
                )
        
        return X_new
    
    def _apply_link_function(self, linear_pred):
        """Apply inverse link function to convert linear predictor to response scale."""
        coef_shape = self.coefficients.shape
        
        # Binary treatment: logistic link
        if len(coef_shape) == 2 and coef_shape[1] == 1:
            from scipy.special import expit
            return expit(linear_pred).ravel()
        
        # Continuous treatment: identity link
        elif self.sigmasq is not None:
            return linear_pred.ravel()
        
        # Multi-valued treatment: multinomial logistic (softmax)
        elif len(coef_shape) == 2 and coef_shape[1] > 1:
            exp_pred = np.exp(linear_pred)
            denom = 1.0 + exp_pred.sum(axis=1, keepdims=True)
            prob_baseline = 1.0 / denom
            prob_others = exp_pred / denom
            return np.column_stack([prob_baseline, prob_others])
        
        else:
            raise ValueError(
                f"Cannot determine treatment type from coefficients shape {coef_shape}. "
                f"Expected (k, 1) for binary/continuous or (k, K-1) for multi-valued."
            )
    
    
    def plot_deviance_residuals(self, ax=None, **kwargs):
        """
        Plot deviance residual diagnostics (binary treatment only).
        
        Generates a 2x2 panel of diagnostic plots:
        
        1. Residuals vs Fitted: Check for non-linearity and heteroscedasticity
        2. Q-Q Plot: Assess normality of residuals
        3. Scale-Location: Check homoscedasticity assumption
        4. Residuals vs Leverage: Identify influential observations
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object for plotting. If None, creates a new figure.
        **kwargs : dict
            Additional arguments passed to matplotlib plotting functions.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : ndarray of matplotlib.axes.Axes
            Array of axes objects (2x2).
            
        Raises
        ------
        ValueError
            If treatment is not binary or required data is missing.
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            raise ImportError(
                "matplotlib and scipy are required for plotting. "
                "Install with: pip install matplotlib scipy"
            )
        
        if not hasattr(self, 'y') or self.y is None:
            raise ValueError("Deviance residuals plot requires y (treatment) data")
        
        y_binary = np.asarray(self.y).ravel()
        unique_y = np.unique(y_binary)
        if len(unique_y) != 2:
            raise ValueError(
                f"Deviance residuals plot only available for binary treatment. "
                f"Found {len(unique_y)} unique treatment values."
            )
        
        # Compute deviance residuals
        fitted_values = np.asarray(self.fitted_values).ravel()
        
        eps = 1e-10
        fitted_safe = np.clip(fitted_values, eps, 1 - eps)
        
        sign = np.where(y_binary == 1, 1, -1)
        deviance_resid = sign * np.sqrt(-2 * (
            y_binary * np.log(fitted_safe) +
            (1 - y_binary) * np.log(1 - fitted_safe)
        ))
        
        # Standardized residuals
        std_resid = deviance_resid / np.std(deviance_resid)
        
        # Create 2x2 subplot grid
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
        else:
            fig = ax.figure
            axes = [ax]
            if len(axes) < 4:
                raise ValueError("Need 4 axes for diagnostic plots. Pass ax=None to create new figure.")
        
        # Panel 1: Residuals vs Fitted
        axes[0].scatter(fitted_values, deviance_resid, alpha=0.5, **kwargs)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
        
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(deviance_resid, fitted_values, frac=0.3)
            axes[0].plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='LOWESS')
            axes[0].legend()
        except ImportError:
            pass
        
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Deviance Residuals')
        axes[0].set_title('Residuals vs Fitted')
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Q-Q Plot
        scipy.stats.probplot(deviance_resid, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Scale-Location
        sqrt_std_resid = np.sqrt(np.abs(std_resid))
        axes[2].scatter(fitted_values, sqrt_std_resid, alpha=0.5, **kwargs)
        
        # Add LOWESS smoother
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(sqrt_std_resid, fitted_values, frac=0.3)
            axes[2].plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='LOWESS')
            axes[2].legend()
        except ImportError:
            pass
        
        axes[2].set_xlabel('Fitted values')
        axes[2].set_ylabel('√|Standardized Residuals|')
        axes[2].set_title('Scale-Location')
        axes[2].grid(True, alpha=0.3)
        
        # Panel 4: Residuals vs Leverage
        leverage = fitted_values * (1 - fitted_values)
        axes[3].scatter(leverage, std_resid, alpha=0.5, **kwargs)
        axes[3].axhline(y=0, color='r', linestyle='--', linewidth=1)
        
        # Mark high-influence points
        cook_threshold = 4 / len(y_binary)
        high_influence = np.abs(std_resid) * leverage > cook_threshold
        if np.any(high_influence):
            axes[3].scatter(leverage[high_influence], std_resid[high_influence], 
                          color='red', s=100, alpha=0.7, label='High influence')
            axes[3].legend()
        
        axes[3].set_xlabel('Leverage')
        axes[3].set_ylabel('Standardized Residuals')
        axes[3].set_title('Residuals vs Leverage')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot(self, kind='deviance', **kwargs):
        """
        Generate diagnostic plots for the CBPS fit.
        
        Parameters
        ----------
        kind : {'deviance'}, default='deviance'
            Type of diagnostic plot to generate.
            Currently only 'deviance' (residual diagnostics) is supported.
        **kwargs : dict
            Additional arguments passed to the plotting function.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes or array of Axes
        
        Raises
        ------
        ValueError
            If an unknown plot kind is specified.
        """
        if kind == 'deviance':
            return self.plot_deviance_residuals(**kwargs)
        else:
            raise ValueError(
                f"Unknown plot kind: '{kind}'. "
                f"Available options: 'deviance'"
            )
    @staticmethod
    def _symnum(pval: np.ndarray) -> List[str]:
        """Convert p-values to significance symbols."""
        symbols = []
        for p in pval:
            if p < 0.001:
                symbols.append('***')
            elif p < 0.01:
                symbols.append('**')
            elif p < 0.05:
                symbols.append('*')
            elif p < 0.1:
                symbols.append('.')
            else:
                symbols.append(' ')
        return symbols
    
    def summary(self) -> 'CBPSSummary':
        """
        Compute and return a statistical summary of the CBPS fit.
        
        Returns
        -------
        CBPSSummary
            Summary object containing coefficient table with estimates,
            standard errors, z-values, p-values, and significance codes.
        
        Raises
        ------
        ValueError
            If the variance-covariance matrix was not computed (var is None),
            standard errors cannot be calculated.
        
        Notes
        -----
        Key implementation details:
        
        1. Standard errors are computed from the diagonal of the variance matrix
        2. z-values are computed as coefficient / standard error
        3. p-values are two-sided: p = 2 * (1 - Phi(abs(z)))
        4. Row names differ for binary vs multi-valued treatment
        
        Examples
        --------
        >>> fit = CBPS('treat ~ age + educ', data=lalonde, att=1)
        >>> summ = fit.summary()
        >>> print(summ)  # Formatted coefficient table
        >>> summ.coef    # Coefficient estimates
        >>> summ.se      # Standard errors
        >>> summ.pvalues # Two-sided p-values
        """
        if self.var is None:
            raise ValueError(
                "Variance-covariance matrix required for summary. "
                "Cannot compute standard errors."
            )
        
        std_err = np.sqrt(np.diag(self.var))
        coef = self.coefficients.ravel()
        z_value = coef / std_err
        p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_value)))
        coef_table = np.column_stack([coef, std_err, z_value, p_value])
        
        significance = self._symnum(p_value)
        
        if self.coefficients.shape[1] == 1:
            row_names = self.coef_names
        else:
            row_names = self._format_multitreat_names()
        
        return CBPSSummary(
            call=self.call_info,
            coef_table=coef_table,
            coef_names=row_names,
            significance=significance,
            J=self.J,
            deviance=self.deviance,
            sigmasq=self.sigmasq,
            y=self.y,
            fitted_values=self.fitted_values,
            weights=self.weights,
            converged=self.converged,
        )
    
    def _format_multitreat_names(self) -> List[str]:
        """Format coefficient names for multi-valued treatment display."""
        row_names = []
        n_row, n_col = self.coefficients.shape
        
        if self.treat_names is not None and len(self.treat_names) >= n_col:
            level_names = self.treat_names[:n_col]
        else:
            level_names = [f"Level{i}" for i in range(n_col)]
        
        for i in range(n_col):
            for j in range(n_row):
                row_names.append(f"{level_names[i]}: {self.coef_names[j]}")
        
        return row_names
    
    def __str__(self) -> str:
        """Return formatted string representation of the CBPS fit."""
        digits = 3
        output = f"\nCall:\n  {self.call_info}\n\n"
        
        if self.coefficients.size > 0:
            output += "Coefficients:\n"
            coef_str = np.array2string(
                self.coefficients, 
                precision=digits, 
                suppress_small=True
            )
            output += coef_str + "\n"
        else:
            output += "No coefficients\n\n"
        
        if self.sigmasq is not None:
            output += f"\nSigma-Squared: {self.sigmasq}\n"
        
        output += f"Residual Deviance:\t{self.deviance:.{digits}g}\n"
        output += f"J-Statistic:\t\t{self.J:.{digits}g}\n"
        output += f"Log-Likelihood:\t{-0.5 * self.deviance:.{digits}g}\n"
        
        # Diagnostics block
        output += f"\nDiagnostics:\n"
        output += f"  Converged:              {'Yes' if self.converged else 'No'}\n"
        
        if self.weights is not None:
            w = self.weights
            output += f"  Weight Summary:\n"
            output += f"    Min: {w.min():10.4f}    Max: {w.max():10.4f}    Mean: {w.mean():8.4f}\n"
            ess = (w.sum() ** 2) / (w ** 2).sum()
            output += f"  Effective Sample Size:  {ess:.1f}\n"
        
        return output
    
    def __repr__(self) -> str:
        """Return concise representation for interactive display."""
        return (f"CBPSResults(n={len(self.y)}, k={self.coefficients.shape[0]}, "
                f"J={self.J:.6f}, converged={self.converged})")


class CBPSSummary:
    """
    Summary object from CBPS estimation.
    
    Contains the coefficient table with estimates, standard errors,
    z-values, p-values, and significance codes. This object is returned
    by the ``summary()`` method of ``CBPSResults``.
    
    Attributes
    ----------
    call : str
        String representation of the fitting call.
    coefficients : ndarray, shape (k, 4)
        Coefficient table with columns: Estimate, Std. Error, z value, Pr(>z).
    coef_names : list of str
        Names of coefficients (row labels).
    significance : list of str
        Significance codes for each coefficient ('***', '**', '*', '.', ' ').
    J : float
        Hansen J-statistic for over-identification test.
    deviance : float
        Model deviance (-2 * log-likelihood).
    sigmasq : float or None
        Residual variance (continuous treatment only, None for binary/multi-valued).
    
    Examples
    --------
    >>> fit = CBPS('treat ~ age + educ', data=lalonde, att=1)
    >>> summ = fit.summary()
    >>> print(summ)           # Formatted table
    >>> summ.coef             # Coefficient estimates
    >>> summ.se               # Standard errors
    >>> summ.zvalues          # z-statistics
    >>> summ.pvalues          # Two-sided p-values
    """
    
    def __init__(
        self,
        call: str,
        coef_table: np.ndarray,
        coef_names: List[str],
        significance: List[str],
        J: float,
        deviance: float,
        sigmasq: Optional[float] = None,
        y: Optional[np.ndarray] = None,
        fitted_values: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        converged: Optional[bool] = None,
    ):
        """
        Initialize summary object.
        
        Parameters
        ----------
        call : str
            Call information string.
        coef_table : ndarray
            Coefficient table (k × 4 matrix).
        coef_names : list
            Coefficient name list.
        significance : list
            Significance symbols list.
        J : float
            J-statistic.
        deviance : float
            Model deviance.
        sigmasq : float, optional
            Sigma squared (continuous treatment only, default None).
        y : ndarray, optional
            Treatment variable (for computing deviance residuals).
        fitted_values : ndarray, optional
            Fitted propensity scores (for computing deviance residuals).
        weights : ndarray, optional
            Estimated weights for diagnostics output.
        converged : bool, optional
            Whether the optimization converged.
        """
        self.call = call
        self.coefficients = coef_table
        self.coef_names = coef_names
        self.significance = significance
        self.J = J
        self.deviance = deviance
        self.sigmasq = sigmasq
        self.y = y
        self.fitted_values = fitted_values
        self.weights = weights
        self.converged = converged
    
    def __str__(self) -> str:
        """Return formatted summary table."""
        output = f"\nCall:\n  {self.call}\n"
        
        # Deviance residuals for binary treatment
        if self.y is not None and self.fitted_values is not None:
            unique_y = np.unique(self.y)
            if len(unique_y) == 2:
                fitted = self.fitted_values.ravel()
                y_binary = self.y.ravel()
                
                eps = 1e-10
                fitted_safe = np.clip(fitted, eps, 1 - eps)
                sign = np.where(y_binary == 1, 1, -1)
                
                deviance_resid = sign * np.sqrt(-2 * (
                    y_binary * np.log(fitted_safe) +
                    (1 - y_binary) * np.log(1 - fitted_safe)
                ))
                
                percentiles = np.percentile(deviance_resid, [0, 25, 50, 75, 100])
                output += "\nDeviance Residuals:\n"
                output += f"    Min      1Q  Median      3Q     Max\n"
                output += f"{percentiles[0]:7.4f} {percentiles[1]:7.4f} {percentiles[2]:7.4f} {percentiles[3]:7.4f} {percentiles[4]:7.4f}\n"
        
        output += "\nCoefficients:\n"
        output += f"{'':20s} {'Estimate':>12s} {'Std. Error':>12s} {'z value':>10s} {'Pr(>|z|)':>12s}\n"
        
        for i, name in enumerate(self.coef_names):
            row = self.coefficients[i]
            sig = self.significance[i]
            output += (f"{name:20s} {row[0]:12.6f} {row[1]:12.6f} "
                      f"{row[2]:10.3f} {row[3]:12.3e} {sig}\n")
        
        output += "---\n"
        output += "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"
        
        if self.sigmasq is not None:
            output += f"\nSigma-Squared: {self.sigmasq}\n"
        
        output += f"\nJ - statistic:  {self.J}\n"
        output += f"Log-Likelihood: {-0.5 * self.deviance}\n"
        
        # Diagnostics block
        if self.converged is not None:
            output += f"\nDiagnostics:\n"
            output += f"  Converged:              {'Yes' if self.converged else 'No'}\n"
            
            if self.weights is not None:
                w = self.weights
                output += f"  Weight Summary:\n"
                output += f"    Min: {w.min():10.4f}    Max: {w.max():10.4f}    Mean: {w.mean():8.4f}\n"
                ess = (w.sum() ** 2) / (w ** 2).sum()
                output += f"  Effective Sample Size:  {ess:.1f}\n"
        
        return output
    
    def __repr__(self) -> str:
        """Return concise representation."""
        return f"CBPSSummary(k={len(self.coef_names)}, J={self.J:.6f})"
    
    @property
    def coef(self) -> np.ndarray:
        """
        Coefficient estimates (convenience property).
        
        Returns
        -------
        ndarray
            Coefficient vector, equivalent to ``self.coefficients[:, 0]``.
        
        Examples
        --------
        >>> summ = fit.summary()
        >>> summ.coef  # Convenient access
        array([...])
        >>> summ.coefficients[:, 0]  # Original access (still supported)
        array([...])
        """
        return self.coefficients[:, 0]
    
    @property
    def se(self) -> np.ndarray:
        """
        Standard errors of coefficient estimates (convenience property).
        
        Returns
        -------
        ndarray
            Standard error vector, equivalent to ``self.coefficients[:, 1]``.
        
        Notes
        -----
        Aligns with statsmodels API: ``fit.bse`` (standard error of coefficients).
        
        Examples
        --------
        >>> summ.se  # Convenient access
        array([...])
        """
        return self.coefficients[:, 1]
    
    @property
    def zvalues(self) -> np.ndarray:
        """
        Z-statistics for coefficient estimates (convenience property).
        
        Returns
        -------
        ndarray
            z-statistic vector, equivalent to ``self.coefficients[:, 2]``.
        
        Notes
        -----
        Aligns with statsmodels API: ``fit.tvalues`` (t-statistic, z for large samples).
        
        Examples
        --------
        >>> summ.zvalues  # Convenient access
        array([...])
        """
        return self.coefficients[:, 2]
    
    @property
    def pvalues(self) -> np.ndarray:
        """
        Two-sided p-values for coefficient estimates (convenience property).
        
        Returns
        -------
        ndarray
            p-value vector, equivalent to ``self.coefficients[:, 3]``.
        
        Notes
        -----
        Two-sided test: p = 2 * (1 - Phi(abs(z)))
        
        Aligns with statsmodels API: ``fit.pvalues``.
        
        Examples
        --------
        >>> summ.pvalues  # Convenient access
        array([...])
        """
        return self.coefficients[:, 3]