"""
High-Dimensional Covariate Balancing Propensity Score (hdCBPS)
==============================================================

This module implements the hdCBPS estimator for robust causal inference in
high-dimensional settings where the number of covariates may exceed the
sample size (p >> n).

Algorithm
---------
The hdCBPS algorithm proceeds in four steps:

**Step 1 (Propensity Score LASSO)**: Obtain initial propensity score estimates
via penalized M-estimation with cross-validated regularization parameter.

**Step 2 (Outcome Model LASSO)**: Fit penalized outcome models separately for
treated and control groups to identify relevant predictors of the outcome.

**Step 3 (Covariate Balancing)**: Calibrate propensity score coefficients by
minimizing the GMM objective that balances covariates selected in Step 2. This
achieves the weak covariate balancing property (Equation 9 in the paper).

**Step 4 (Treatment Effect Estimation)**: Compute ATE/ATT using the
Horvitz-Thompson estimator and derive asymptotic standard errors using the
sandwich variance estimator (Equation 11 in the paper).

Key Properties
--------------
- **Double Robustness**: Root-n consistent and asymptotically normal when
  either the propensity score model or outcome model is correctly specified
  (Propositions 1 and 2 in the paper).
- **Semiparametric Efficiency**: Achieves the semiparametric efficiency bound
  under correct specification of both models (Theorem 1 in the paper).
- **Sample Boundedness**: Estimated treatment effects lie within the range
  of observed outcomes due to the covariate balancing constraint (Remark 2
  in the paper).

References
----------
Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score. Biometrika,
107(3), 533-554. https://doi.org/10.1093/biomet/asaa020
"""

import warnings
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .lasso_utils import (cv_glmnet, predict_glmnet_fortran, select_variables, 
                          HAS_GLMNETFORPYTHON)
from .weight_funcs import ate_wt_func, ate_wt_nl_func, att_wt_func, att_wt_nl_func
from .gmm_loss import gmm_func, att_gmm_func

# Optimization options
NELDER_MEAD_OPTIONS = {'maxfev': 50000}


class HDCBPSSummary:
    """Summary object for HDCBPSResults.

    Returned by :meth:`HDCBPSResults.summary`. Provides a structured
    representation of hdCBPS estimation results that can be printed
    via ``print()`` or ``str()``.
    """

    def __init__(
        self,
        ATE: float,
        ATT: Optional[float],
        s: float,
        w: Optional[float],
        fitted_values: Optional[np.ndarray],
        fitted_y: Optional[np.ndarray],
        fitted_x: Optional[np.ndarray],
        converged: bool,
        iterations_used: Optional[Dict[str, int]],
        selected_indices_treat: Optional[np.ndarray],
        selected_indices_control: Optional[np.ndarray],
        n_selected_treat: Optional[int],
        n_selected_control: Optional[int],
    ):
        self.ATE = ATE
        self.ATT = ATT
        self.s = s
        self.w = w
        self.fitted_values = fitted_values
        self.fitted_y = fitted_y
        self.fitted_x = fitted_x
        self.converged = converged
        self.iterations_used = iterations_used
        self.selected_indices_treat = selected_indices_treat
        self.selected_indices_control = selected_indices_control
        self.n_selected_treat = n_selected_treat
        self.n_selected_control = n_selected_control

    def __str__(self) -> str:
        """Return formatted summary text (matches HDCBPSResults.__str__())."""
        output = "\nhdCBPS Estimation Results\n"
        output += "=" * 60 + "\n"

        if self.fitted_y is not None:
            output += f"Sample size: {len(self.fitted_y)}\n"
        if self.fitted_x is not None:
            output += f"Covariates: {self.fitted_x.shape[1]}\n"
        output += f"Converged: {'Yes' if self.converged else 'No'}\n"

        output += f"\nTreatment Effects:\n"
        output += f"  ATE (Average Treatment Effect): {self.ATE:.6f}\n"
        output += f"  Standard Error: {self.s:.6f}\n"
        output += f"  95% CI: [{self.ATE - 1.96*self.s:.6f}, {self.ATE + 1.96*self.s:.6f}]\n"

        if self.ATT is not None:
            output += f"\n  ATT (Average Treatment on Treated): {self.ATT:.6f}\n"
            if self.w is not None:
                output += f"  Standard Error: {self.w:.6f}\n"
                output += f"  95% CI: [{self.ATT - 1.96*self.w:.6f}, {self.ATT + 1.96*self.w:.6f}]\n"

        if self.fitted_values is not None:
            output += f"\nPropensity Scores:\n"
            output += f"  Min: {self.fitted_values.min():.4f}\n"
            output += f"  Max: {self.fitted_values.max():.4f}\n"
            output += f"  Mean: {self.fitted_values.mean():.4f}\n"

        if self.iterations_used is not None:
            output += f"\nIterations:\n"
            for key, val in self.iterations_used.items():
                output += f"  {key}: {val}\n"

        if self.selected_indices_treat is not None or self.selected_indices_control is not None:
            output += f"\nVariable Selection (LASSO):\n"
            if self.n_selected_treat is not None:
                output += f"  Treatment group: {self.n_selected_treat} variables selected\n"
                if len(self.selected_indices_treat) <= 10:
                    output += f"    Indices: {self.selected_indices_treat.tolist()}\n"
                else:
                    output += f"    Indices: {self.selected_indices_treat[:5].tolist()}...{self.selected_indices_treat[-3:].tolist()} (showing first 5 and last 3)\n"
            if self.n_selected_control is not None:
                output += f"  Control group: {self.n_selected_control} variables selected\n"
                if len(self.selected_indices_control) <= 10:
                    output += f"    Indices: {self.selected_indices_control.tolist()}\n"
                else:
                    output += f"    Indices: {self.selected_indices_control[:5].tolist()}...{self.selected_indices_control[-3:].tolist()} (showing first 5 and last 3)\n"

        output += "=" * 60 + "\n"
        return output

    def __repr__(self) -> str:
        return f"HDCBPSSummary(ATE={self.ATE:.6f}, converged={self.converged})"


class HDCBPSResults:
    """
    Results container for high-dimensional CBPS estimation.

    This class stores the output of :func:`hdCBPS`, including treatment effect
    estimates, standard errors, propensity scores, and diagnostic information.

    Attributes
    ----------
    ATE : float
        Average Treatment Effect estimate computed using the Horvitz-Thompson
        estimator: :math:`\\hat{\\mu}_1 - \\hat{\\mu}_0`.
    ATT : float or None
        Average Treatment Effect on the Treated. Only computed if ``ATT=1``
        was specified in the call to :func:`hdCBPS`.
    s : float
        Standard error of ATE computed using the sandwich variance estimator
        (Equation 11 in Ning et al., 2020).
    w : float or None
        Standard error of ATT (if ATT was estimated).
    fitted_values : np.ndarray, shape (n,)
        Calibrated propensity scores from Step 3 of the algorithm.
    weights : np.ndarray, shape (n,)
        Inverse probability weights for outcome modeling.
    coefficients1 : np.ndarray
        Calibrated propensity score coefficients (treatment group optimization).
    coefficients0 : np.ndarray
        Calibrated propensity score coefficients (control group optimization).
    fitted_y : np.ndarray, shape (n,)
        Outcome variable used in estimation.
    fitted_x : np.ndarray, shape (n, p+1)
        Covariate matrix with intercept column.
    converged : bool
        Whether the GMM optimization converged within tolerance.
    iterations_used : dict
        Number of Nelder-Mead iterations for each optimization step.
    selected_indices_treat : np.ndarray
        Indices of variables selected by LASSO for the treated outcome model.
    selected_indices_control : np.ndarray
        Indices of variables selected by LASSO for the control outcome model.
    n_selected_treat : int
        Number of variables selected for the treated outcome model.
    n_selected_control : int
        Number of variables selected for the control outcome model.
    data : pd.DataFrame
        Original input data.
    formula : str
        Model formula specification.
    terms : object
        Patsy design information.
    call : str
        String representation of the function call.
    na_action : dict or None
        Missing value handling information.
    """

    def __init__(self):
        self.ATE: Optional[float] = None
        self.ATT: Optional[float] = None
        self.s: Optional[float] = None
        self.w: Optional[float] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.coefficients1: Optional[np.ndarray] = None
        self.coefficients0: Optional[np.ndarray] = None
        self.fitted_y: Optional[np.ndarray] = None
        self.fitted_x: Optional[np.ndarray] = None
        self.test1: Optional[np.ndarray] = None
        self.test0: Optional[np.ndarray] = None
        self.converged: Optional[bool] = None
        self.iterations_used: Optional[Dict[str, int]] = None
        self.weights: Optional[np.ndarray] = None
        
        # Variable selection info
        self.selected_indices_treat: Optional[np.ndarray] = None
        self.selected_indices_control: Optional[np.ndarray] = None
        self.n_selected_treat: Optional[int] = None
        self.n_selected_control: Optional[int] = None
        
        # Metadata
        self.data: Optional[pd.DataFrame] = None
        self.formula: Optional[str] = None
        self.terms: Optional[object] = None
        self.call: Optional[str] = None
        self.na_action: Optional[dict] = None

        # Debug data (stored in private dict; access via result._debug['key'])
        self._debug: Dict[str, Any] = {}

    def __getattr__(self, name: str):
        """Backward compatibility: emit deprecation warning when accessing debug data via legacy attribute names."""
        if name.startswith('debug_') and '_debug' in self.__dict__:
            warnings.warn(
                f"Direct access to '{name}' is deprecated. Use result._debug['{name}'] instead."
                " This attribute will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            if name in self._debug:
                return self._debug[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        """Ensure dir() does not expose debug_* attributes."""
        default = super().__dir__()
        return [attr for attr in default if not attr.startswith('debug_')]
    
    @property
    def ate(self) -> Optional[float]:
        """Lowercase alias for ATE (Python naming convention)."""
        return self.ATE

    @property
    def se_ate(self) -> Optional[float]:
        """Standard error of ATE (alias for ``s``)."""
        return self.s

    @property
    def att(self) -> Optional[float]:
        """Lowercase alias for ATT (Python naming convention)."""
        return self.ATT

    @property
    def se_att(self) -> Optional[float]:
        """Standard error of ATT (alias for ``w``)."""
        return self.w

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """
        Alias for coefficients0 to maintain API consistency with other CBPS results.
        
        Returns
        -------
        np.ndarray
            Coefficients for the control group model.
        """
        return self.coefficients0
    
    @property
    def x(self) -> Optional[np.ndarray]:
        """Alias for fitted_x."""
        return self.fitted_x
    
    @property
    def y(self) -> Optional[np.ndarray]:
        """Alias for fitted_y."""
        return self.fitted_y
    
    def vcov(self):
        """
        Variance-covariance matrix of coefficients.
        
        Raises
        ------
        ValueError
            hdCBPS does not provide a standard variance-covariance matrix for coefficients
            due to the nature of LASSO variable selection. Standard errors for ATE and ATT
            are available via the 's' and 'w' attributes.
        """
        raise ValueError(
            "hdCBPS does not provide a variance-covariance matrix for coefficients "
            "due to the nature of LASSO variable selection. "
            "Standard errors for ATE and ATT are available as 's' and 'w' attributes."
        )

    def summary(self) -> 'HDCBPSSummary':
        """Generate a formatted summary of hdCBPS estimation results.

        Returns
        -------
        HDCBPSSummary
            Summary object with ``__str__`` method. Use ``print(result.summary())``
            to display the formatted text.
        """
        return HDCBPSSummary(
            ATE=self.ATE,
            ATT=self.ATT,
            s=self.s,
            w=self.w,
            fitted_values=self.fitted_values,
            fitted_y=self.fitted_y,
            fitted_x=self.fitted_x,
            converged=self.converged,
            iterations_used=self.iterations_used,
            selected_indices_treat=self.selected_indices_treat,
            selected_indices_control=self.selected_indices_control,
            n_selected_treat=self.n_selected_treat,
            n_selected_control=self.n_selected_control,
        )

    def __repr__(self) -> str:
        if self.ATE is not None:
            return f"HDCBPSResults(ATE={self.ATE:.6f}, s={self.s:.6f}, converged={self.converged})"
        else:
            return "HDCBPSResults(not fitted)"
    
    def __str__(self) -> str:
        if self.ATE is None:
            return "HDCBPSResults(not fitted)"
        
        output = "\nhdCBPS Estimation Results\n"
        output += "=" * 60 + "\n"
        
        # Basic Info
        if self.fitted_y is not None:
            output += f"Sample size: {len(self.fitted_y)}\n"
        if self.fitted_x is not None:
            output += f"Covariates: {self.fitted_x.shape[1]}\n"
        output += f"Converged: {'Yes' if self.converged else 'No'}\n"
        
        # Treatment Effects
        output += f"\nTreatment Effects:\n"
        output += f"  ATE (Average Treatment Effect): {self.ATE:.6f}\n"
        output += f"  Standard Error: {self.s:.6f}\n"
        output += f"  95% CI: [{self.ATE - 1.96*self.s:.6f}, {self.ATE + 1.96*self.s:.6f}]\n"
        
        if self.ATT is not None:
            output += f"\n  ATT (Average Treatment on Treated): {self.ATT:.6f}\n"
            if self.w is not None:
                output += f"  Standard Error: {self.w:.6f}\n"
                output += f"  95% CI: [{self.ATT - 1.96*self.w:.6f}, {self.ATT + 1.96*self.w:.6f}]\n"
        
        # Propensity Scores
        if self.fitted_values is not None:
            output += f"\nPropensity Scores:\n"
            output += f"  Min: {self.fitted_values.min():.4f}\n"
            output += f"  Max: {self.fitted_values.max():.4f}\n"
            output += f"  Mean: {self.fitted_values.mean():.4f}\n"
        
        # Iterations
        if self.iterations_used is not None:
            output += f"\nIterations:\n"
            for key, val in self.iterations_used.items():
                output += f"  {key}: {val}\n"
        
        # Variable Selection
        if self.selected_indices_treat is not None or self.selected_indices_control is not None:
            output += f"\nVariable Selection (LASSO):\n"
            if self.n_selected_treat is not None:
                output += f"  Treatment group: {self.n_selected_treat} variables selected\n"
                if len(self.selected_indices_treat) <= 10:
                    output += f"    Indices: {self.selected_indices_treat.tolist()}\n"
                else:
                    output += f"    Indices: {self.selected_indices_treat[:5].tolist()}...{self.selected_indices_treat[-3:].tolist()} (showing first 5 and last 3)\n"
            if self.n_selected_control is not None:
                output += f"  Control group: {self.n_selected_control} variables selected\n"
                if len(self.selected_indices_control) <= 10:
                    output += f"    Indices: {self.selected_indices_control.tolist()}\n"
                else:
                    output += f"    Indices: {self.selected_indices_control[:5].tolist()}...{self.selected_indices_control[-3:].tolist()} (showing first 5 and last 3)\n"
        
        output += "=" * 60 + "\n"
        
        return output


def hdCBPS(
    formula: str,
    data: pd.DataFrame,
    y: Union[str, np.ndarray],
    ATT: int = 0,
    iterations: int = 1000,
    method: str = 'linear',
    seed: Optional[int] = None,
    na_action: Optional[str] = None,
    verbose: int = 0
) -> HDCBPSResults:
    """
    High-dimensional covariate balancing propensity score estimation.

    Estimate average treatment effects in high-dimensional settings (p >> n)
    using the hdCBPS methodology of Ning, Peng, and Imai (2020). The method
    combines LASSO variable selection with covariate balancing to achieve
    double robustness and semiparametric efficiency.

    Parameters
    ----------
    formula : str
        Model formula specifying treatment variable and covariates in Patsy
        format. The left-hand side should be the binary treatment variable.

        Example: ``'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'``

    data : pd.DataFrame
        Dataset containing all variables referenced in the formula and
        the outcome variable.
    y : str or np.ndarray
        Outcome variable. Can be either:

        - Column name (str) in ``data``
        - NumPy array of outcome values

    ATT : int, default=0
        Target estimand:

        - 0: Average Treatment Effect (ATE)
        - 1: Average Treatment Effect on the Treated (ATT)

    iterations : int, default=1000
        Maximum number of Nelder-Mead iterations for the GMM optimization
        in Step 3.
    method : str, default='linear'
        Outcome model family:

        - ``'linear'``: Gaussian (continuous outcomes)
        - ``'binomial'``: Logistic (binary outcomes)
        - ``'poisson'``: Poisson (count outcomes)

    seed : int, optional
        Random seed for reproducibility of cross-validation folds.
    na_action : str, optional
        Missing value handling:

        - ``None`` or ``'warn'``: Drop missing rows with warning (default)
        - ``'drop'``: Drop missing rows silently
        - ``'fail'``: Raise error if missing values present

    verbose : int, default=0
        Verbosity level (currently unused, reserved for future use).

    Returns
    -------
    HDCBPSResults
        Results object containing:

        - ``ATE``: Average treatment effect estimate
        - ``s``: Standard error of ATE
        - ``ATT``: ATT estimate (if ``ATT=1``)
        - ``w``: Standard error of ATT (if ``ATT=1``)
        - ``fitted_values``: Calibrated propensity scores
        - ``weights``: Inverse probability weights
        - ``converged``: Convergence status
        - ``selected_indices_*``: LASSO-selected variable indices

    Raises
    ------
    ImportError
        If glmnetforpython is not installed.
    ValueError
        If ``iterations`` is not a positive integer, or if ``na_action='fail'``
        and missing values are present.

    Examples
    --------
    >>> from cbps import hdCBPS
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>> result = hdCBPS(
    ...     formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    ...     data=df,
    ...     y='re78',
    ...     ATT=0,
    ...     method='linear'
    ... )
    >>> print(f"ATE: {result.ATE:.4f} (SE: {result.s:.4f})")

    Notes
    -----
    The algorithm achieves double robustness: the estimator is consistent
    and asymptotically normal if either the propensity score model or the
    outcome model is correctly specified (Propositions 1 and 2 in the paper).

    References
    ----------
    Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal
    effects via a high-dimensional covariate balancing propensity score.
    Biometrika, 107(3), 533-554.
    """
    if iterations is None:
        iterations = 1000
    
    if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise ValueError(
            f"iterations must be a positive integer, got {iterations}. Default is 1000."
        )
    
    if seed is not None:
        np.random.seed(seed)
    
    # Handle missing values
    n_orig = len(data)
    if data.isnull().any().any():
        if na_action == 'fail':
            raise ValueError(
                f"Data contains missing values and na_action='fail'. "
                f"Found {data.isnull().sum().sum()} missing values."
            )
        data = data.dropna()
        if na_action != 'drop':
            n_dropped = n_orig - len(data)
            warnings.warn(
                f"Removed {n_dropped} observations ({n_dropped/n_orig*100:.1f}%) with missing values.",
                UserWarning
            )
    
    from cbps.utils.formula import parse_formula
    from patsy import dmatrices
    
    data_original = data.copy()
    
    _, X_design = dmatrices(formula, data, return_type='dataframe')
    terms_obj = X_design.design_info
    
    treat, X = parse_formula(formula, data)
    
    if isinstance(y, str):
        y_array = data[y].values
    else:
        y_array = np.asarray(y)
    
    if y_array is None or len(y_array) == 0:
        raise ValueError("Outcome variable y is required and cannot be None or empty.")
    
    y_var = np.var(y_array)
    if y_var < 1e-10:
        warnings.warn(
            f"Outcome variable y has near-zero variance ({y_var:.2e}). ATE estimates may be close to zero.",
            UserWarning
        )
    
    # Preprocess X: remove zero-variance columns and add intercept
    X_nonzero_var = X[:, X.std(axis=0) > 0]
    X_with_intercept = np.column_stack([np.ones(len(X_nonzero_var)), X_nonzero_var])
    
    result = hdCBPS_fit(
        x=X_with_intercept,
        y=y_array,
        treat=treat,
        ATT=ATT,
        iterations=iterations,
        method=method
    )
    
    result.data = data_original
    result.formula = formula
    result.terms = terms_obj
    result.call = f"hdCBPS(formula='{formula}', data=..., y=..., ATT={ATT}, method='{method}')"
    result.na_action = None
    
    return result


def hdCBPS_fit(
    x: np.ndarray,
    y: np.ndarray,
    treat: np.ndarray,
    ATT: int,
    iterations: int = 1000,
    method: str = "linear"
) -> HDCBPSResults:
    """
    Core fitting routine for hdCBPS estimation.

    This function implements the four-step hdCBPS algorithm on preprocessed
    data. It is called internally by :func:`hdCBPS` after formula parsing
    and missing value handling.

    Parameters
    ----------
    x : np.ndarray, shape (n, p+1)
        Covariate matrix with intercept column prepended.
    y : np.ndarray, shape (n,)
        Outcome variable.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).
    ATT : int
        Estimand selection:

        - 0: Estimate ATE only
        - 1: Estimate both ATE and ATT

    iterations : int, default=1000
        Maximum Nelder-Mead iterations for GMM optimization.
    method : str, default='linear'
        Outcome model family: ``'linear'``, ``'binomial'``, or ``'poisson'``.

    Returns
    -------
    HDCBPSResults
        Fitted results object with treatment effect estimates and diagnostics.

    Notes
    -----
    The algorithm steps are:

    1. LASSO for outcome models (treated and control groups separately)
    2. LASSO for propensity score model
    3. GMM optimization to calibrate propensity scores
    4. Horvitz-Thompson estimation and variance calculation
    """
    n, p = x.shape
    
    y1hat = y[treat == 1]
    x1hat = x[treat == 1, :]
    y0hat = y[treat == 0]
    x0hat = x[treat == 0, :]
    
    # ===================================================================
    # Steps 1-2: LASSO Variable Selection
    # Paper: Step 1 = propensity score (Eq.5), Step 2 = outcome model (Eq.6)
    # Implementation: outcome models first, then propensity score
    # (order is interchangeable when w_2(u)=1, i.e., unweighted LASSO)
    # ===================================================================
    
    # Outcome model LASSO (Step 2 in paper, Equation 6 with w_2=1)
    # Fit separately for treated and control groups to select predictive covariates
    if method == "linear":
        # Gaussian family for continuous outcomes
        model1, coef1, lambda_min_1 = cv_glmnet(x1hat, y1hat, family='gaussian', intercept=True)
        model0, coef0, lambda_min_0 = cv_glmnet(x0hat, y0hat, family='gaussian', intercept=True)
    elif method == "binomial":
        # Logistic family for binary outcomes (Section 4 of paper)
        # Note: intercept=False for outcome model in R's cv.glmnet
        # cv_glmnet returns full coefficient vector including intercept position (set to 0)
        model1, coef1, lambda_min_1 = cv_glmnet(x1hat, y1hat, family='binomial', intercept=False)
        model0, coef0, lambda_min_0 = cv_glmnet(x0hat, y0hat, family='binomial', intercept=False)
    elif method == "poisson":
        # Poisson family for count outcomes (Section 4 of paper)
        # Note: intercept=False for outcome model
        model1, coef1, lambda_min_1 = cv_glmnet(x1hat, y1hat, family='poisson', intercept=False)
        model0, coef0, lambda_min_0 = cv_glmnet(x0hat, y0hat, family='poisson', intercept=False)
    else:
        raise ValueError(f"method '{method}' not supported")

    # Propensity score LASSO (Step 1 in paper, Equation 5)
    # Logistic regression with L1 penalty for initial propensity score coefficients
    modelb, coef_b, lambda_min_b = cv_glmnet(x, treat, family='binomial', intercept=True)
    
    using_glmnetforpython = HAS_GLMNETFORPYTHON
    
    # Extract selected variable indices: S_tilde in paper notation (Equation 7)
    S1 = select_variables(coef1)  # Support of treated outcome model coefficients
    S0 = select_variables(coef0)  # Support of control outcome model coefficients
    
    # ===================================================================
    # Step 3: Covariate Balancing Calibration (Equation 7)
    # Minimize ||g_n(gamma)||^2 to achieve weak covariate balance
    # ===================================================================
    
    tol = 1e-5  # Convergence tolerance for GMM objective
    kk0 = 0
    kk1 = 0
    ATT_kk0 = 0
    diff0 = 1.0
    diff1 = 1.0
    ATT_diff0 = 1.0
    
    def gmm_loss0(beta_curr):
        return gmm_func(
            beta_curr, S=S0, tt=0, X_gmm=x, method=method,
            cov1_coef=coef1, cov0_coef=coef0, treat=treat, beta_ini=coef_b
        )
    
    def gmm_loss1(beta_curr):
        return gmm_func(
            beta_curr, S=S1, tt=1, X_gmm=x, method=method,
            cov1_coef=coef1, cov0_coef=coef0, treat=treat, beta_ini=coef_b
        )
    
    # ===================================================================
    # Step 3.1: ATE Calibration
    # Optimize gamma separately for treated (mu_1) and control (mu_0)
    # ===================================================================
    
    if method == "linear":
        # Control group: calibrate to balance X_{S0}
        if len(S0) > 0:
            beta0_init = coef_b[S0]
            beta0_result = minimize(gmm_loss0, beta0_init, method='Nelder-Mead',
                                  options=NELDER_MEAD_OPTIONS)

            while diff0 > tol and kk0 < iterations:
                beta0_result = minimize(gmm_loss0, beta0_result.x, method='Nelder-Mead',
                                      options=NELDER_MEAD_OPTIONS)
                diff0 = beta0_result.fun
                kk0 += 1

            beta0_par = beta0_result.x
        else:
            beta0_par = np.array([])
            diff0 = 0.0

        # Treatment group optimization
        if len(S1) > 0:
            beta1_init = coef_b[S1]
            beta1_result = minimize(gmm_loss1, beta1_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)

            while diff1 > tol and kk1 < iterations:
                beta1_result = minimize(gmm_loss1, beta1_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                diff1 = beta1_result.fun
                kk1 += 1

            beta1_par = beta1_result.x
        else:
            beta1_par = np.array([])
            diff1 = 0.0

        w_curr1 = ate_wt_func(beta1_par, S1, 1, x, coef_b, treat)
        w_curr0 = ate_wt_func(beta0_par, S0, 0, x, coef_b, treat)

    else:
        # Nonlinear (binomial/poisson) optimization
        # Control group
        if len(S0) > 0:
            beta0_init = coef_b[np.r_[0, S0]]
            beta0_result = minimize(gmm_loss0, beta0_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)

            while diff0 > tol and kk0 < iterations:
                beta0_result = minimize(gmm_loss0, beta0_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                diff0 = beta0_result.fun
                kk0 += 1

            beta0_par = beta0_result.x
        else:
            beta0_init = coef_b[[0]]
            beta0_result = minimize(gmm_loss0, beta0_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)

            while diff0 > tol and kk0 < iterations:
                beta0_result = minimize(gmm_loss0, beta0_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                diff0 = beta0_result.fun
                kk0 += 1

            beta0_par = beta0_result.x

        # Treatment group
        if len(S1) > 0:
            beta1_init = coef_b[np.r_[0, S1]]
            beta1_result = minimize(gmm_loss1, beta1_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)

            while diff1 > tol and kk1 < iterations:
                beta1_result = minimize(gmm_loss1, beta1_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                diff1 = beta1_result.fun
                kk1 += 1

            beta1_par = beta1_result.x
        else:
            beta1_init = coef_b[[0]]
            beta1_result = minimize(gmm_loss1, beta1_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)

            while diff1 > tol and kk1 < iterations:
                beta1_result = minimize(gmm_loss1, beta1_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                diff1 = beta1_result.fun
                kk1 += 1

            beta1_par = beta1_result.x

        w_curr1 = ate_wt_nl_func(beta1_par, S1, 1, x, coef_b, treat)
        w_curr0 = ate_wt_nl_func(beta0_par, S0, 0, x, coef_b, treat)

    ATE = (1.0 / n) * (
        y1hat.T @ (w_curr1[treat == 1] + 1.0) +
        y0hat.T @ (w_curr0[treat == 0] - 1.0)
    )

    # ===================================================================
    # Step 3.2: ATT Optimization Loop
    # ===================================================================

    if ATT != 0:
        def ATT_gmm_loss(beta_curr):
            return att_gmm_func(
                beta_curr, S=S0, X_gmm=x, method=method,
                cov0_coef=coef0, treat=treat, beta_ini=coef_b
            )
        
        if method == "linear":
            if len(S0) > 0:
                ATT_beta0_init = coef_b[S0]
                ATT_beta0_result = minimize(ATT_gmm_loss, ATT_beta0_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                
                while ATT_diff0 > tol and ATT_kk0 < iterations:
                    ATT_beta0_result = minimize(ATT_gmm_loss, ATT_beta0_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                    ATT_diff0 = ATT_beta0_result.fun
                    ATT_kk0 += 1
                
                ATT_beta0_par = ATT_beta0_result.x
            else:
                ATT_beta0_par = np.array([])
                ATT_diff0 = 0.0
            
            ATT_w_curr0 = att_wt_func(ATT_beta0_par, S0, x, coef_b, treat)
        
        else:
            if len(S0) > 0:
                ATT_beta0_init = coef_b[np.r_[0, S0]]
            else:
                ATT_beta0_init = coef_b[[0]]
            
            ATT_beta0_result = minimize(ATT_gmm_loss, ATT_beta0_init, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
            
            while ATT_diff0 > tol and ATT_kk0 < iterations:
                ATT_beta0_result = minimize(ATT_gmm_loss, ATT_beta0_result.x, method='Nelder-Mead', options=NELDER_MEAD_OPTIONS)
                ATT_diff0 = ATT_beta0_result.fun
                ATT_kk0 += 1
            
            ATT_beta0_par = ATT_beta0_result.x
            
            ATT_w_curr0 = att_wt_nl_func(ATT_beta0_par, S0, x, coef_b, treat)
        
        ATT_value = (
            (1.0 / np.sum(treat)) * np.sum(y1hat) -
            (1.0 / np.sum(ATT_w_curr0[treat == 0])) * (y0hat.T @ ATT_w_curr0[treat == 0])
        )

        X_full = np.column_stack([np.ones(n), x])
        ATT_beta_0 = coef_b.copy()
        if method == "linear":
            if len(S0) > 0:
                ATT_beta_0[S0] = ATT_beta0_par
        else:
            if len(S0) > 0:
                ATT_beta_0[np.r_[0, S0]] = ATT_beta0_par
            else:
                ATT_beta_0[[0]] = ATT_beta0_par

        ATT_theta_0 = X_full @ ATT_beta_0
        ATT_r_yhatb0 = 1.0 - 1.0 / (1.0 + np.exp(ATT_theta_0))
    else:
        ATT_value = None
        ATT_w_curr0 = None
        ATT_r_yhatb0 = None

    if ATT != 0:
        converged = (diff0 <= tol and diff1 <= tol and ATT_diff0 <= tol)
        iterations_used = {'kk0': kk0, 'kk1': kk1, 'ATT_kk0': ATT_kk0}
    else:
        converged = (diff0 <= tol and diff1 <= tol)
        iterations_used = {'kk0': kk0, 'kk1': kk1}
    
    # ===================================================================
    # Step 4: Variance Estimation
    # Extended from Equation 11 in Ning et al. (2020) to estimate Var(ATE):
    #   V_hat = (1/n) * [delta_K + term1 + term2]
    # where:
    #   delta_K = sum((alpha1'X - alpha0'X - ATE)^2)  [prediction variance]
    #   term1 = sum((Y - alpha1'X)^2 / pi^2) for treated  [treated residuals]
    #   term2 = sum((Y - alpha0'X)^2 / (1-pi)^2) for control  [control residuals]
    # Standard error: s = sqrt(V_hat / n)
    # ===================================================================

    if method == "linear":
        X_full = np.column_stack([np.ones(n), x])
        beta_0 = coef_b.copy()
        if len(S0) > 0:
            beta_0[S0] = beta0_par
        theta_0 = X_full @ beta_0
        r_yhatb0 = 1.0 - 1.0 / (1.0 + np.exp(theta_0))

        beta_1 = coef_b.copy()
        if len(S1) > 0:
            beta_1[S1] = beta1_par
        theta_1 = X_full @ beta_1
        r_yhatb1 = 1.0 - 1.0 / (1.0 + np.exp(theta_1))

        if using_glmnetforpython:
            r_yhat1 = predict_glmnet_fortran(model1, x1hat)
            r_yhat0 = predict_glmnet_fortran(model0, x0hat)
            r_yhat1full = predict_glmnet_fortran(model1, x)
            r_yhat0full = predict_glmnet_fortran(model0, x)
        else:
            r_yhat1 = x1hat @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(len(y1hat), coef1[0])
            r_yhat0 = x0hat @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(len(y0hat), coef0[0])
            r_yhat1full = x @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(n, coef1[0])
            r_yhat0full = x @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(n, coef0[0])

    else:
        X_full = np.column_stack([np.ones(n), x])
        beta_0 = coef_b.copy()
        if len(S0) > 0:
            beta_0[np.r_[0, S0]] = beta0_par
        else:
            beta_0[[0]] = beta0_par
        theta_0 = X_full @ beta_0
        r_yhatb0 = 1.0 - 1.0 / (1.0 + np.exp(theta_0))

        beta_1 = coef_b.copy()
        if len(S1) > 0:
            beta_1[np.r_[0, S1]] = beta1_par
        else:
            beta_1[[0]] = beta1_par
        theta_1 = X_full @ beta_1
        r_yhatb1 = 1.0 - 1.0 / (1.0 + np.exp(theta_1))

        if using_glmnetforpython:
            r_yhat1 = predict_glmnet_fortran(model1, x1hat)
            r_yhat0 = predict_glmnet_fortran(model0, x0hat)
            r_yhat1full = predict_glmnet_fortran(model1, x)
            r_yhat0full = predict_glmnet_fortran(model0, x)
        else:
            if method == "binomial":
                eta1 = x1hat @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(len(y1hat), coef1[0])
                eta0 = x0hat @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(len(y0hat), coef0[0])
                eta1full = x @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(n, coef1[0])
                eta0full = x @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(n, coef0[0])
                r_yhat1 = 1.0 / (1.0 + np.exp(-eta1))
                r_yhat0 = 1.0 / (1.0 + np.exp(-eta0))
                r_yhat1full = 1.0 / (1.0 + np.exp(-eta1full))
                r_yhat0full = 1.0 / (1.0 + np.exp(-eta0full))
            else:  # poisson
                eta1 = x1hat @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(len(y1hat), coef1[0])
                eta0 = x0hat @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(len(y0hat), coef0[0])
                eta1full = x @ coef1[1:] + coef1[0] if len(coef1) > 1 else np.full(n, coef1[0])
                eta0full = x @ coef0[1:] + coef0[0] if len(coef0) > 1 else np.full(n, coef0[0])
                r_yhat1 = np.exp(eta1)
                r_yhat0 = np.exp(eta0)
                r_yhat1full = np.exp(eta1full)
                r_yhat0full = np.exp(eta0full)

    # Compute variance components (extended from Equation 11 for ATE)
    # Following R implementation exactly:
    #   delta_K <- sum((r_yhat1full - r_yhat0full - rep(ATE, n))^2)
    #   sigma_1 <- sum((r_yhat1 - y1hat)^2/r_yhatb1[treat == 1])/n  # scalar
    #   sigma_0 <- sum((r_yhat0 - y0hat)^2/(1 - r_yhatb0[treat == 0]))/n  # scalar
    #   s = sqrt((delta_K + sum(sigma_1/r_yhatb1) + sum(sigma_0/r_yhatb0))/n)/sqrt(n)
    delta_K = np.sum((r_yhat1full - r_yhat0full - ATE)**2)
    
    # Treatment group variance term: denominator is π̂² (squared)
    term1 = np.sum((r_yhat1 - y1hat)**2 / (r_yhatb1[treat == 1]**2))
    # Control group variance term: denominator is (1-π̂)² (squared)
    term2 = np.sum((r_yhat0 - y0hat)**2 / ((1.0 - r_yhatb0[treat == 0])**2))
    
    V_hat = (delta_K + term1 + term2) / n
    s_value = np.sqrt(V_hat) / np.sqrt(n)  # Standard error of ATE

    if ATT != 0:
        ATT_delta_K = np.sum(ATT_r_yhatb0 * (r_yhat1full - r_yhat0full - ATT_value)**2)
        term_treat = np.sum(ATT_r_yhatb0[treat == 1] * (r_yhat1 - y1hat)**2)
        term_control = np.sum(
            ATT_r_yhatb0[treat == 0]**2 * (r_yhat0 - y0hat)**2 /
            (1.0 - ATT_r_yhatb0[treat == 0])
        )
        w_value = (n / np.sum(treat)) * np.sqrt(
            (ATT_delta_K + term_treat + term_control) / n
        ) / np.sqrt(n)
    else:
        w_value = None
    
    # ===================================================================
    # Compute Calibrated Propensity Scores
    # Recover pi from weights: W = T/pi - 1 => pi = T/(W+1)
    # ===================================================================

    fitted_values = np.ones(n)
    fitted_values[treat == 1] = 1.0 / (w_curr1[treat == 1] + 1.0)  # pi for treated
    fitted_values[treat == 0] = 1.0 - 1.0 / (1.0 - w_curr0[treat == 0])  # pi for control
    
    # ===================================================================
    # Construct Result Object
    # ===================================================================
    
    if method == "linear":
        beta_0_final = coef_b.copy()
        if len(S0) > 0:
            beta_0_final[S0] = beta0_par

        beta_1_final = coef_b.copy()
        if len(S1) > 0:
            beta_1_final[S1] = beta1_par
    else:
        beta_0_final = coef_b.copy()
        if len(S0) > 0:
            beta_0_final[np.r_[0, S0]] = beta0_par
        else:
            beta_0_final[[0]] = beta0_par

        beta_1_final = coef_b.copy()
        if len(S1) > 0:
            beta_1_final[np.r_[0, S1]] = beta1_par
        else:
            beta_1_final[[0]] = beta1_par
    
    result = HDCBPSResults()
    result.ATE = ATE
    result.ATT = ATT_value
    result.s = s_value
    result.w = w_value
    result.fitted_values = fitted_values
    result.coefficients1 = beta_1_final
    result.coefficients0 = beta_0_final
    result.fitted_y = y
    result.fitted_x = x
    result.test1 = w_curr1
    result.test0 = w_curr0
    result.converged = converged
    result.iterations_used = iterations_used
    
    pi = fitted_values
    if ATT == 0:
        weights = treat / pi + (1 - treat) / (1 - pi)
    elif ATT == 1:
        weights = treat + (1 - treat) * pi / (1 - pi)
    else:
        weights = treat * (1 - pi) / pi + (1 - treat)
    result.weights = weights
    
    result.selected_indices_treat = S1
    result.selected_indices_control = S0
    result.n_selected_treat = len(S1)
    result.n_selected_control = len(S0)

    if ATT != 0:
        result.ATT_w_curr0 = ATT_w_curr0

    # Debug information
    result._debug['debug_r_yhat1'] = r_yhat1
    result._debug['debug_r_yhat0'] = r_yhat0
    result._debug['debug_r_yhat1full'] = r_yhat1full
    result._debug['debug_r_yhat0full'] = r_yhat0full
    result._debug['debug_r_yhatb1'] = r_yhatb1
    result._debug['debug_r_yhatb0'] = r_yhatb0
    result._debug['debug_delta_K'] = delta_K
    result._debug['debug_sigma_1'] = term1
    result._debug['debug_sigma_0'] = term2
    result._debug['debug_y1hat'] = y1hat
    result._debug['debug_y0hat'] = y0hat
    result._debug['debug_coef1'] = coef1
    result._debug['debug_coef0'] = coef0
    result._debug['debug_coef_b'] = coef_b
    result._debug['debug_lambda_min_1'] = lambda_min_1
    result._debug['debug_lambda_min_0'] = lambda_min_0
    result._debug['debug_lambda_min_b'] = lambda_min_b

    return result

