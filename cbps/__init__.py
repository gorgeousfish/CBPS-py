"""
Covariate Balancing Propensity Score (CBPS)
===========================================

A comprehensive Python implementation of the covariate balancing propensity score
methodology for causal inference from observational studies.

The CBPS approach revolutionizes propensity score estimation by directly incorporating
covariate balance conditions into the estimation procedure [1]_. Unlike traditional
propensity score methods that solely maximize the likelihood of treatment assignment,
CBPS estimates propensity scores by solving moment conditions that simultaneously
optimize covariate balance between treatment groups while maintaining predictive power.

This innovative approach is implemented through the generalized method of moments
(GMM) framework, where the objective function seamlessly integrates the score function
for treatment prediction with moment conditions ensuring covariate balance. The resulting
estimator achieves superior finite-sample balance performance while preserving the
double robustness properties of conventional propensity score methods.

Methodological Framework
------------------------

For a binary treatment :math:`T \\in \\{0,1\\}` and covariates :math:`X`, the CBPS
estimator :math:`\\hat{\\beta}` solves the following moment conditions:

.. math::
    \\frac{1}{n} \\sum_{i=1}^n \\psi_i(\\beta) = 0

where the moment function :math:`\\psi_i(\\beta)` combines:

1. **Score function**: :math:`\\psi_i^{(1)}(\\beta) = T_i - e(X_i,\\beta)`
2. **Balance conditions**: :math:`\\psi_i^{(2)}(\\beta) = T_i X_i - e(X_i,\\beta) X_i`

with :math:`e(X_i,\\beta)` denoting the propensity score model.

Key Features
------------

* **Binary Treatments**: Robust estimation of average treatment effects (ATE) and
  average treatment effects on the treated (ATT) using logistic models [1]_

* **Multi-valued Treatments**: Seamless extension to categorical treatments via
  multinomial logistic regression supporting treatments with three or four levels

* **Continuous Treatments**: Generalized propensity scores for continuous
  treatment variables using flexible parametric distributions [2]_

* **High-dimensional Settings**: State-of-the-art regularization through LASSO
  when the number of covariates exceeds the sample size, with automatic variable
  selection and valid post-selection inference [3]_

* **Nonparametric Estimation**: Empirical likelihood methods that completely
  avoid parametric modeling assumptions about the propensity score [4]_

* **Longitudinal Data**: Marginal structural models for time-varying treatments
  with time-dependent confounding, extending causal inference to complex study designs [5]_

* **Instrumental Variables**: Comprehensive support for treatment noncompliance
  and instrumental variable assignment scenarios [6]_

Implementation Highlights
--------------------------

- **Automatic Treatment Detection**: Intelligent recognition of binary, multi-valued,
  and continuous treatments based on data characteristics
- **Dual Interface Design**: Both intuitive patsy formula interface and efficient
  NumPy array interface for different usage patterns
- **Advanced GMM Options**: Two-step and continuous updating GMM estimators for
  different precision and speed requirements
- **Numerical Stability**: Robust optimization with enhanced convergence diagnostics
  and graceful failure handling
- **High Precision**: Maintains ±1e-6 numerical accuracy for core algorithms,
  ensuring reproducible research results

References
----------
.. [1] Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
   Journal of the Royal Statistical Society, Series B 76(1), 243-263.
   https://doi.org/10.1111/rssb.12027

.. [2] Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
   score for a continuous treatment: Application to the efficacy of political
   advertisements. The Annals of Applied Statistics 12(1), 156-177.
   https://doi.org/10.1214/17-AOAS1101

.. [3] Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
   via a high-dimensional covariate balancing propensity score. Biometrika 107(3),
   533-554. https://doi.org/10.1093/biomet/asaa020

.. [4] Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
   score for general treatment regimes. Journal of the American Statistical
   Association 113(523), 1316-1329. https://doi.org/10.1080/01621459.2017.1385465

.. [5] Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
   weights for marginal structural models. Journal of the American Statistical
   Association 110(511), 1013-1023. https://doi.org/10.1080/01621459.2014.956872

.. [6] Fong, C. (2018). Robust and efficient estimation of causal effects with
   calibrated covariate balance. Unpublished manuscript.

License
-------
AGPL-3.0

Copyright (c) 2025-2026 Cai Xuanyu, Xu Wenli
"""

from typing import Any, Optional, Union, Dict
import warnings
import pandas as pd
import numpy as np

__version__ = "0.1.0"

from cbps.core.results import CBPSResults, CBPSSummary
from cbps.core.cbps_binary import cbps_binary_fit

__all__ = [
    "CBPS",
    "cbps_fit",
    "CBMSM",
    "cbmsm_fit",
    "npCBPS",
    "npCBPS_fit",
    "hdCBPS",
    "CBIV",
    "AsyVar",
    "balance",
    "vcov_outcome",
    "plot_cbps",
    "plot_cbps_continuous",
    "plot_cbmsm",
    "plot_npcbps",
]


def _add_balance_labels(balance_result: Dict[str, np.ndarray], cbps_dict: Dict[str, Any],
                        coef_names: Optional[list], is_continuous: bool) -> Dict[str, pd.DataFrame]:
    """
    Attach covariate labels to balance assessment statistics.

    This internal function transforms balance statistics from numpy arrays to
    labeled pandas DataFrames, facilitating interpretation of balance diagnostics.
    The labeling convention varies by treatment type to reflect the appropriate
    balance metrics.

    Parameters
    ----------
    balance_result : Dict[str, np.ndarray]
        Balance statistics computed from either discrete or continuous treatment
        models. Dictionary contains keys for weighted ('balanced') and unweighted
        ('original' for discrete, 'unweighted' for continuous) statistics.
    cbps_dict : Dict[str, Any]
        Fitted CBPS estimator object containing treatment assignment data and
        model specifications necessary for label generation.
    coef_names : list or None
        Names of covariate variables excluding the intercept term. When None,
        generic covariate labels are generated automatically.
    is_continuous : bool
        Indicator flag for continuous treatment models, which determines the
        appropriate column labeling convention.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mirroring the input structure but with DataFrame objects
        containing properly labeled rows (covariates) and columns (balance
        statistics).

    Notes
    -----
    Column labeling follows treatment-specific conventions:

    * **Discrete treatments**: Statistics include treatment means and standardized
      mean differences, with columns labeled as "treatment.mean" and
      "treatment.std.mean"
    * **Continuous treatments**: Statistics focus on correlation coefficients,
      with the single column labeled as "corr"

    The output follows standard balance table conventions, with rows for covariates
    and columns for treatment-specific statistics.
    """
    # Extract original numpy arrays
    balanced_array = balance_result['balanced']
    original_key = 'unweighted' if is_continuous else 'original'
    original_array = balance_result[original_key]

    # Generate row names (covariate names)
    n_covars = balanced_array.shape[0]
    if coef_names is not None and len(coef_names) == n_covars:
        row_names = coef_names
    else:
        # Fall back to default names
        row_names = [f"X{i+1}" for i in range(n_covars)]
    
    # Generate column names
    if is_continuous:
        # Continuous treatment: single correlation column
        col_names_balanced = ['corr']
        col_names_original = ['corr']
    else:
        # Discrete treatment: mean and standardized mean for each level
        treats = pd.Categorical(cbps_dict['y'])
        treat_levels = treats.categories
        n_treats = len(treat_levels)

        # Generate all mean columns first, then all standardized mean columns
        col_names = []
        for level in treat_levels:
            # Format treatment levels: remove decimal point for integers
            if isinstance(level, (int, np.integer)):
                level_str = str(int(level))
            elif isinstance(level, (float, np.floating)) and level == int(level):
                level_str = str(int(level))  # 0.0 → "0", 1.0 → "1"
            else:
                level_str = str(level)
            col_names.append(f"{level_str}.mean")
        for level in treat_levels:
            # Apply same formatting logic
            if isinstance(level, (int, np.integer)):
                level_str = str(int(level))
            elif isinstance(level, (float, np.floating)) and level == int(level):
                level_str = str(int(level))
            else:
                level_str = str(level)
            col_names.append(f"{level_str}.std.mean")
        col_names_balanced = col_names
        col_names_original = col_names
    
    # Convert to DataFrame
    balanced_df = pd.DataFrame(
        balanced_array,
        columns=col_names_balanced,
        index=row_names
    )
    
    original_df = pd.DataFrame(
        original_array,
        columns=col_names_original,
        index=row_names
    )
    
    # Return dictionary with DataFrames
    return {
        'balanced': balanced_df,
        original_key: original_df
    }


def _check_overlap_violation(
    cbps_result: Any,
    is_continuous: bool,
    threshold: float = 0.05
) -> None:
    """
    Assess potential violations of the overlap assumption in propensity scores.

    The overlap assumption, also known as the common support condition, requires
    that all units have non-zero probability of receiving each treatment level.
    This diagnostic function identifies potential violations by detecting extreme
    propensity score values that may indicate perfect separation, quasi-complete
    separation, or substantial lack of overlap between treatment groups.

    Parameters
    ----------
    cbps_result : CBPSResults
        Fitted CBPS estimator object containing estimated propensity scores.
    is_continuous : bool
        Logical indicator distinguishing between discrete and continuous
        treatment models. Overlap assessment differs by treatment type.
    threshold : float, default=0.05
        Proportion threshold for triggering warnings about extreme values.
        The default 0.05 corresponds to 5% of the sample.

    Notes
    -----
    The overlap assumption is fundamental for causal inference with propensity
    scores. Formally, it requires that for all covariate values :math:`X`,
    :math:`0 < \\Pr(T = t | X) < 1` for all treatment levels :math:`t`.

    Extreme value detection follows treatment-specific conventions:

    * **Discrete treatments**: Propensity scores below 0.01 or above 0.99 are
      flagged as extreme, indicating potential lack of overlap
    * **Continuous treatments**: The check is skipped as fitted values represent
      probability densities rather than probabilities in [0,1]

    Violations of overlap can lead to:
    - Infinite or unstable coefficient estimates
    - Large variance in treatment effect estimates
    - Dependence on model extrapolation beyond the data support

    References
    ----------
    .. [1] King, G. and Zeng, L. (2001). Logistic regression in rare events data.
       Political Analysis, 9(2), 137-163.
    .. [2] Firth, D. (1993). Bias reduction of maximum likelihood estimates.
       Biometrika, 80(1), 27-38.
    .. [3] Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
       Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    if is_continuous:
        # For continuous treatments, fitted_values are probability densities
        # rather than probabilities in [0,1]. Skip overlap check.
        return

    # Check discrete treatments for extreme propensity scores
    fitted_vals = cbps_result.fitted_values

    # Handle multi-treat case where fitted_values may be 2D
    if fitted_vals.ndim == 2:
        # Multi-treatment: check each column
        probs = fitted_vals
    else:
        # Binary treatment: 1D array
        probs = fitted_vals.ravel()
    
    # Define extreme values: < 0.01 or > 0.99
    extreme_low = 0.01
    extreme_high = 0.99

    # Calculate proportion of extreme values
    if probs.ndim == 1:
        # Binary treatment
        n_extreme = np.sum((probs < extreme_low) | (probs > extreme_high))
    else:
        # Multi-treatment: count if any column has extreme values
        n_extreme = np.sum(np.any((probs < extreme_low) | (probs > extreme_high), axis=1))
    
    n_total = len(cbps_result.y)
    extreme_ratio = n_extreme / n_total
    
    if extreme_ratio > threshold:
        warnings.warn(
            f"Potential overlap violation detected: {extreme_ratio:.1%} of observations "
            f"have extreme propensity scores (< {extreme_low} or > {extreme_high}). "
            f"This may indicate:\n"
            f"  - Perfect or quasi-complete separation\n"
            f"  - Severe violation of the overlap assumption\n"
            f"  - Possible numerical instability in coefficient estimates\n\n"
            f"Recommendations:\n"
            f"  - Check covariate balance diagnostics\n"
            f"  - Consider removing or combining problematic covariates\n"
            f"  - Use regularization methods (e.g., hdCBPS) if appropriate\n"
            f"  - Verify that treatment groups have sufficient covariate overlap\n\n"
            f"Theory: CBPS assumes 0 < Pr(T|X) < 1 for all X (Imai & Ratkovic 2014, Assumption 1).",
            UserWarning,
            stacklevel=3
        )


def _validate_finite_inputs(
    treat: np.ndarray,
    X: np.ndarray,
    func_name: str = "CBPS"
) -> None:
    """
    Validate input data for numerical finiteness.

    This preprocessing function ensures that treatment and covariate data contain
    only finite values, checking for the presence of NaN (Not a Number) or
    infinite values that would compromise the optimization algorithm. The validation
    adapts to different data types, gracefully handling categorical and string
    variables which cannot contain numerical infinities.

    Parameters
    ----------
    treat : np.ndarray
        Treatment assignment variable of shape (n,). May be numeric for binary
        or continuous treatments, or categorical/string for multi-valued
        treatments.
    X : np.ndarray
        Covariate matrix of shape (n, k) containing predictor variables.
        Must contain only finite numeric values for model estimation.
    func_name : str, default="CBPS"
        Name of the calling function used to generate informative error
        messages for debugging purposes.

    Raises
    ------
    ValueError
        Raised when either the treatment variable or covariate matrix contains
        NaN or infinite values. The error message includes the count of
        problematic values and suggests data cleaning strategies.

    Notes
    -----
    The function implements type-aware validation:

    * **Numeric treatments**: Full finiteness check with detailed error reporting
    * **Categorical treatments**: Validation skipped as categories cannot be
      infinite or NaN
    * **String treatments**: Validation skipped for the same reason

    For covariates, all columns must be finite as missing or infinite values
    would break the numerical optimization routines used in CBPS estimation.
    """
    # Check treatment variable
    # Skip isfinite check for categorical/string types (strings cannot have inf/nan)
    treat_is_categorical = (
        hasattr(treat, 'categories') or 
        (hasattr(treat, 'dtype') and hasattr(treat.dtype, 'categories'))
    )
    treat_is_string = (
        hasattr(treat, 'dtype') and 
        (treat.dtype.kind in ('U', 'O', 'S'))  # U=unicode, O=object, S=bytes
    )
    
    # Check treatment variable (skip string/categorical types)
    if not treat_is_string and not treat_is_categorical:
        # Attempt to convert to numeric type for validation
        try:
            treat_numeric = np.asarray(treat, dtype=np.float64)
            if not np.all(np.isfinite(treat_numeric)):
                n_inf = np.isinf(treat_numeric).sum()
                n_nan = np.isnan(treat_numeric).sum()
                raise ValueError(
                    f"{func_name}: Treatment variable contains {n_nan} NaN and {n_inf} Inf value(s). "
                    f"Inf values typically indicate data errors (division by zero, numerical overflow, "
                    f"or incorrect feature engineering). Please clean your data before calling {func_name}. "
                    f"Consider: data.dropna() or data[np.isfinite(data).all(axis=1)]"
                )
        except (ValueError, TypeError):
            # Cannot convert to numeric type (e.g., strings), skip isfinite check
            pass
    
    # Check covariates
    if not np.all(np.isfinite(X)):
        n_inf = np.isinf(X).sum()
        n_nan = np.isnan(X).sum()
        # Identify columns containing inf/nan values
        bad_cols = np.where(~np.all(np.isfinite(X), axis=0))[0]
        raise ValueError(
            f"{func_name}: Covariates contain {n_nan} NaN and {n_inf} Inf value(s) "
            f"in column(s) {bad_cols.tolist()}. "
            f"Inf values typically indicate data errors (e.g., log(0), division by zero). "
            f"Please clean your data before calling {func_name}."
        )


def _has_intercept(X: np.ndarray) -> bool:
    """
    Detect whether the covariate matrix includes an intercept term.

    This function determines if the first column of the design matrix represents
    an intercept term (a column of ones). The formula interface automatically
    includes an intercept, while the array interface requires explicit handling.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Design matrix containing covariates and potentially an intercept term.
        The matrix should be in the format expected by CBPS estimation functions.

    Returns
    -------
    bool
        True if the first column consists entirely of ones (within numerical
        precision), False otherwise.

    Notes
    -----
    The detection uses np.allclose with default tolerances to account for
    floating-point representation errors. Values such as 1.0000001 or 0.9999999
    are correctly identified as intercept terms.

    This function is essential for:
    - Proper handling of model specifications across interfaces
    - Avoiding duplicate intercept terms in model fitting
    - Maintaining numerical stability in optimization

    Examples
    --------
    >>> import numpy as np
    >>> X_with_intercept = np.column_stack([np.ones(100), np.random.normal(size=(100, 3))])
    >>> _has_intercept(X_with_intercept)
    True
    >>> X_no_intercept = np.random.normal(size=(100, 3))
    >>> _has_intercept(X_no_intercept)
    False
    """
    if X.shape[1] == 0:
        return False
    return np.allclose(X[:, 0], 1.0)


def _apply_svd_preprocessing(X: np.ndarray) -> tuple:
    """
    Apply SVD preprocessing to covariate matrix for numerical stability.

    This function performs singular value decomposition preprocessing to improve
    numerical stability in multi-valued treatment models.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Covariate matrix with intercept in first column.

    Returns
    -------
    X_svd : np.ndarray, shape (n, k)
        SVD-orthogonalized matrix (first k columns of U matrix).
    svd_info : dict
        Dictionary containing SVD information needed for inverse transform:
        - 'V': V matrix from SVD
        - 'd': Singular values
        - 'x_sd': Standard deviations for standardization
        - 'x_mean': Means for standardization
        - 'U': Complete U matrix

    Notes
    -----
    Creates a copy of input matrix to avoid modifying original data.
    """
    # Create a copy to avoid modifying input
    X_work = X.copy()
    X_orig = X_work.copy()  # Save original unstandardized copy

    # Standardize X (excluding intercept column)
    x_sd = X_work[:, 1:].std(axis=0, ddof=1)
    x_mean = X_work[:, 1:].mean(axis=0)
    X_work[:, 1:] = (X_work[:, 1:] - x_mean) / x_sd

    # SVD decomposition
    U, s, Vt = np.linalg.svd(X_work, full_matrices=True)
    V_matrix = Vt.T  # NumPy returns Vt, R returns V

    # Save SVD information for inverse transform
    svd_info = {
        'V': V_matrix,
        'd': s,
        'x_sd': x_sd,
        'x_mean': x_mean,
        'U': U,
        'X_standardized': X_work.copy()  # Save standardized X for debugging
    }

    # Replace X with U matrix (first k columns)
    X_svd = U[:, :X_orig.shape[1]]  # Take first k columns

    return X_svd, svd_info


def _apply_svd_inverse_transform(beta_svd: np.ndarray, svd_info: dict) -> np.ndarray:
    """
    Apply inverse SVD transform to coefficient matrix.

    Transforms coefficients from SVD-orthogonalized space back to original
    covariate space.

    Parameters
    ----------
    beta_svd : np.ndarray, shape (k, K-1)
        Coefficient matrix in SVD space.
    svd_info : dict
        SVD information dictionary returned by preprocessing function.

    Returns
    -------
    beta_transformed : np.ndarray, shape (k, K-1)
        Coefficient matrix in original covariate space.

    Notes
    -----
    Transformation steps:
    1. SVD inverse transform: beta = V @ diag(d_inv) @ beta_svd
    2. Reverse standardization (except intercept): beta[1:,:] /= x_sd
    3. Adjust intercept: beta[0,:] -= x_mean @ beta[1:,:]
    """
    # Singular value truncation
    d_inv = svd_info['d'].copy()
    d_inv[d_inv > 1e-5] = 1.0 / d_inv[d_inv > 1e-5]
    d_inv[d_inv <= 1e-5] = 0

    # Apply inverse SVD transform to coefficients
    beta_transformed = svd_info['V'] @ np.diag(d_inv) @ beta_svd

    # Reverse standardization (except intercept)
    beta_transformed[1:, :] = beta_transformed[1:, :] / svd_info['x_sd'][:, None]

    # Adjust intercept
    beta_transformed[0, :] = beta_transformed[0, :] - svd_info['x_mean'] @ beta_transformed[1:, :]

    return beta_transformed


# Whitelist of allowed kwargs to pass through to scipy.optimize.minimize
_SCIPY_ALLOWED_KWARGS = {
    'callback',  # Optimization callback function
    'tol',       # Tolerance for termination
    'options',   # Options dictionary for optimizer
    'bal_gtol',  # Gradient tolerance for balance optimization (R-matching)
    'gmm_gtol',  # Gradient tolerance for GMM optimization (R-matching)
}

def _detect_treatment_type(
    treat: np.ndarray,
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    treat_col_name: Optional[str] = None
) -> tuple[bool, bool, bool]:
    """
    Detect the type of treatment variable for parameter validation and routing.

    Parameters
    ----------
    treat : np.ndarray
        Treatment variable array.
    formula : str, optional
        Formula string (used for column name extraction).
    data : pd.DataFrame, optional
        Data frame (used for checking categorical types).
    treat_col_name : str, optional
        Treatment column name (if known).

    Returns
    -------
    tuple of bool
        (is_categorical, is_binary_01, is_continuous) where:
        - is_categorical: True if pandas Categorical type
        - is_binary_01: True if binary 0/1 numeric values
        - is_continuous: True if continuous (non-binary, non-categorical)

    Notes
    -----
    Detection logic:
    1. If pandas Categorical → (True, False, False)
    2. If unique values are {0, 1} → (False, True, False)
    3. Otherwise → (False, False, True)

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([0, 1, 0, 1])
    >>> is_cat, is_bin, is_cont = _detect_treatment_type(treat)
    >>> print(is_bin, is_cont)
    True False
    """
    # Ensure treat is a numpy array
    treat_array = np.asarray(treat).ravel()

    # Step 1: Check if pandas Categorical type
    is_categorical = False
    
    if data is not None and treat_col_name is not None:
        # Check original column type from data
        if treat_col_name in data.columns:
            is_categorical = (
                isinstance(data[treat_col_name].dtype, pd.CategoricalDtype) or
                isinstance(data[treat_col_name], pd.Categorical)
            )
    elif hasattr(treat, 'cat'):
        # Directly passed Series might have .cat attribute
        is_categorical = True
    elif isinstance(treat, pd.Categorical):
        is_categorical = True
    
    # Step 2: If not categorical, check if binary 0/1
    is_binary_01 = False
    if not is_categorical:
        treat_unique = np.unique(treat_array)
        is_binary_01 = (
            len(treat_unique) == 2 and
            set(treat_unique) <= {0, 1, 0.0, 1.0, False, True}
        )
    
    # Step 3: Determine if continuous treatment
    is_continuous = (
        not is_categorical and
        not is_binary_01 and
        np.issubdtype(treat_array.dtype, np.number)
    )
    
    return is_categorical, is_binary_01, is_continuous


def CBPS(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    treatment: Optional[np.ndarray] = None,
    covariates: Optional[np.ndarray] = None,
    att: int = 1,
    method: str = 'over',
    two_step: bool = True,
    standardize: bool = True,
    sample_weights: Optional[np.ndarray] = None,
    baseline_formula: Optional[str] = None,
    diff_formula: Optional[str] = None,
    iterations: int = 1000,
    theoretical_exact: bool = False,
    na_action: str = 'warn',
    verbose: int = 0,
    ATT: Optional[int] = None,
    twostep: Optional[bool] = None,
    **kwargs
) -> CBPSResults:
    """
    Covariate Balancing Propensity Score (CBPS) Estimation

    Estimates propensity scores such that both covariate balance and prediction
    of treatment assignment are simultaneously maximized. The method avoids
    the iterative process between model fitting and balance checking by
    implementing both objectives simultaneously.

    Supports binary, multi-valued (3-4 levels), and continuous treatments.

    Parameters
    ----------
    formula : str, optional
        A symbolic description of the model to be fitted. The formula should
        be of the form ``treatment ~ covariate1 + covariate2 + ...``.
        Either ``formula`` and ``data`` or ``treatment`` and ``covariates``
        must be provided.
    data : pd.DataFrame, optional
        A data frame containing the variables in the model. Required when
        using the formula interface.
    treatment : np.ndarray, optional
        Treatment vector. Required when using the array interface instead of
        the formula interface.
    covariates : np.ndarray, optional
        Covariate matrix. Required when using the array interface. Should not
        include an intercept column (it will be added automatically).
    att : int, default 1
        Target estimand. 0 for ATE (average treatment effect), 1 for ATT
        with the second level as treated, 2 for ATT with the first level as
        treated. For non-binary treatments, only ATE is available.
    ATT : int, optional
        Deprecated. Use lowercase ``att`` instead.
    method : {'over', 'exact'}, default 'over'
        Estimation method. 'over' for over-identified GMM (combines propensity
        score likelihood and covariate balancing conditions), 'exact' for
        exactly-identified GMM (covariate balancing conditions only).
    two_step : bool, default True
        If True, uses the two-step GMM estimator (faster). If False, uses
        the continuous-updating GMM estimator (better finite sample properties).
    twostep : bool, optional
        Alias for ``two_step`` parameter. Use ``two_step`` for consistency
        with Python naming conventions.
    standardize : bool, default True
        If True, normalizes weights to sum to 1 within each treatment group
        (or to 1 for the entire sample with continuous treatments). If False,
        returns Horvitz-Thompson weights.
    sample_weights : np.ndarray, optional
        Survey sampling weights for the observations. If None, defaults to
        equal weights of 1 for each observation.
    baseline_formula : str, optional
        Formula for covariates in the baseline outcome model E(Y(0)|X). Used only
        for optimal CBPS (iCBPS) with binary treatments.
    diff_formula : str, optional
        Formula for covariates in the treatment effect difference model
        E(Y(1)-Y(0)|X). Used only for optimal CBPS (iCBPS) with binary treatments.
    iterations : int, default 1000
        Maximum number of iterations for the optimization algorithm.
    theoretical_exact : bool, default False
        When method='exact', uses direct equation solver for exact GMM solution.
        If False, uses balance loss optimization (default behavior).
    na_action : {'warn', 'fail', 'ignore'}, default 'warn'
        How to handle missing values. 'warn' removes observations with missing
        values and issues a warning, 'fail' raises an error, 'ignore' uses
        patsy's default behavior.
    verbose : int, default 0
        Verbosity level. 0 for silent output, 1 for basic progress, 2 for
        detailed iteration information.
    **kwargs
        Additional parameters passed to the optimization routine.

    Returns
    -------
    CBPSResults
        A fitted CBPS object containing:
        - coefficients: estimated propensity score coefficients
        - fitted.values: estimated propensity scores
        - weights: covariate balancing weights
        - converged: convergence status
        - j_statistic: J-statistic for overidentification test

    Raises
    ------
    ValueError
        If required inputs are missing or invalid, or if the model cannot be
        estimated (e.g., perfect collinearity, insufficient sample size).

    Notes
    -----
    **Treatment Type Detection**

    - Binary treatments: Automatically detected for integer arrays with ≤4 unique values
    - Multi-valued treatments: Must be converted to ``pd.Categorical`` before fitting
    - Continuous treatments: Automatically detected for floating-point arrays or >4 unique values

    **Estimation Methods**

    - The 'over' method combines likelihood-based score functions with covariate
      balance constraints in an over-identified GMM framework
    - The 'exact' method uses only covariate balancing conditions (exactly-identified)

    **Weight Standardization**

    - When standardize=True, weights sum to 1 within each treatment group
    - When standardize=False, returns Horvitz-Thompson weights

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    https://doi.org/10.1111/rssb.12027

    Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
    Optimal Covariate Balancing Conditions in Propensity Score Estimation.
    Journal of Business & Economic Statistics, 41(1), 97-110.
    
    Examples
    --------
    >>> import cbps
    >>> from cbps.datasets import load_lalonde
    >>> # Load LaLonde job training data
    >>> data = load_lalonde(dehejia_wahba_only=True)
    >>> # Estimate CBPS for ATT
    >>> fit = cbps.CBPS('treat ~ age + educ + black + hisp', data=data, att=1)
    >>> print(fit.summary())
    >>> # Access weights for downstream analysis
    >>> weights = fit.weights

    """
    # Handle twostep parameter alias for compatibility
    if twostep is not None:
        # Use twostep value if provided (overrides two_step)
        two_step = twostep

    # Parameter validation
    # att must be 0, 1, or 2
    # att=0: ATE, att=1: ATT (T=1 as treated), att=2: ATT (T=0 as treated)
    # Check type first, then value range (TypeError before ValueError)
    if not isinstance(att, (int, np.integer)):
        raise TypeError(
            f"att must be an integer (0, 1, or 2), got type {type(att).__name__}: {att}"
        )
    if att not in [0, 1, 2]:
        raise ValueError(
            f"Invalid att parameter: {att}\n\n"
            f"att must be 0, 1, or 2:\n"
            f"  att=0: ATE (Average Treatment Effect) for entire population\n"
            f"  att=1: ATT (Average Treatment effect on the Treated, T=1 as treated)\n"
            f"  att=2: ATT (Average Treatment effect on the Treated, T=0 as treated)\n\n"
            f"You provided: att={att}"
        )
    
    # Handle legacy uppercase ATT parameter for backward compatibility
    if ATT is not None:
        # Validate ATT parameter
        if not isinstance(ATT, (int, np.integer)) or ATT not in [0, 1]:
            raise ValueError(
                f"Invalid ATT parameter: {ATT}\n\n"
                f"ATT must be either 0 or 1:\n"
                f"  ATT=0: ATE (Average Treatment Effect)\n"
                f"  ATT=1: ATT (Average Treatment effect on the Treated)\n\n"
                f"You provided: ATT={ATT} (type: {type(ATT).__name__})"
            )
        
        if att == 1:  # att is default value, user didn't explicitly set it
            att = ATT
            warnings.warn(
                f"Using deprecated parameter name 'ATT={ATT}'. "
                f"Please use lowercase 'att={ATT}' instead for consistency with Python naming conventions.",
                DeprecationWarning,
                stacklevel=2
            )
        else:
            # User set both att and ATT with different values
            warnings.warn(
                f"Both 'att={att}' and 'ATT={ATT}' were specified. Using 'att={att}'. "
                f"Please use only 'att' parameter (lowercase) to avoid confusion.",
                UserWarning
            )
    
    # Validate kwargs to prevent confusing scipy errors
    if kwargs:
        invalid_kwargs = set(kwargs.keys()) - _SCIPY_ALLOWED_KWARGS
        if invalid_kwargs:
            # Check if this is a common error (uppercase parameter names)
            suggestions = []
            for invalid_key in invalid_kwargs:
                # Check for case confusion
                if invalid_key.upper() == 'ATT':
                    suggestions.append(f"  - Did you mean 'att' (lowercase) instead of '{invalid_key}'?")
                elif invalid_key.lower() == 'standardize':
                    suggestions.append(f"  - Did you mean 'standardize' (correct spelling) instead of '{invalid_key}'?")
                elif invalid_key.lower() == 'method':
                    suggestions.append(f"  - Did you mean 'method' (lowercase) instead of '{invalid_key}'?")
            
            error_msg = (
                f"CBPS() got unexpected keyword argument(s): {sorted(invalid_kwargs)}\n\n"
                f"Valid scipy.optimize parameters are: {sorted(_SCIPY_ALLOWED_KWARGS)}\n"
            )
            if suggestions:
                error_msg += "\nCommon mistakes:\n" + "\n".join(suggestions)
            else:
                error_msg += (
                    "\nNote: CBPS parameters (att, method, standardize, etc.) should be "
                    "specified as named arguments, not in **kwargs."
                )
            
            raise TypeError(error_msg)
    
    # Mutual exclusivity check: formula and treatment cannot both be specified
    if formula is not None and treatment is not None:
        raise ValueError(
            "Cannot specify both 'formula' and 'treatment' parameters. "
            "Please use either:\n"
            "  1. Formula interface: CBPS(formula='treat ~ X1 + X2', data=df)\n"
            "  2. Array interface: CBPS(treatment=treat_array, covariates=X_array)\n"
            f"\nReceived:\n"
            f"  formula = {repr(formula)}\n"
            f"  treatment = {'<array>' if treatment is not None else 'None'}"
        )
    
    # Validate iterations parameter
    if not isinstance(iterations, (int, np.integer)):
        raise TypeError(
            f"iterations must be an integer, got {type(iterations).__name__}. "
            f"Received: iterations={iterations}"
        )
    if iterations < 1:
        raise ValueError(
            f"iterations must be ≥1 (at least one optimization step required). "
            f"Received: iterations={iterations}"
        )
    if iterations > 100000:
        warnings.warn(
            f"iterations={iterations} is very large and may take a long time. "
            f"Consider using a smaller value (default is 1000).",
            UserWarning
        )
    
    # Validate att parameter
    if not isinstance(att, (int, np.integer)):
        raise TypeError(
            f"att must be an integer (0, 1, or 2), got {type(att).__name__}. "
            f"Received: att={att}"
        )
    if att not in (0, 1, 2):
        raise ValueError(
            f"att must be 0 (ATE), 1 (ATT treated=level2), or 2 (ATT treated=level1). "
            f"Received: att={att}\n\n"
            f"Explanation:\n"
            f"  att=0: Average Treatment Effect (ATE) for entire population\n"
            f"  att=1: Average Treatment effect on the Treated (ATT), second level as treated\n"
            f"  att=2: ATT with first level as treated"
        )
    
    # Validate method parameter
    valid_methods = {'over', 'exact'}
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__}. "
            f"Received: method={method}"
        )
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}. "
            f"Received: method='{method}'\n\n"
            f"Explanation:\n"
            f"  method='over': Over-identified GMM (score + balance conditions, recommended)\n"
            f"  method='exact': Exactly-identified GMM (balance conditions only)\n\n"
            f"Note: method is case-sensitive, use lowercase only."
        )
    
    # Validate theoretical_exact compatibility with method parameter
    if theoretical_exact and method != 'exact':
        warnings.warn(
            f"theoretical_exact=True only works with method='exact'. "
            f"Current method='{method}' does not use this parameter. "
            f"The theoretical_exact parameter will be ignored.\n\n"
            f"To use theoretical_exact, set method='exact'.",
            UserWarning
        )
    
    # Validate verbose parameter
    if not isinstance(verbose, (int, np.integer)):
        raise TypeError(
            f"verbose must be an integer (0, 1, or 2), got {type(verbose).__name__}. "
            f"Received: verbose={verbose}"
        )
    if verbose not in (0, 1, 2):
        raise ValueError(
            f"verbose must be 0 (silent), 1 (basic), or 2 (detailed). "
            f"Received: verbose={verbose}"
        )
    
    # Validate two_step parameter
    if not isinstance(two_step, bool):
        raise TypeError(
            f"two_step must be a boolean (True or False), got {type(two_step).__name__}. "
            f"Received: two_step={two_step}\n\n"
            f"Hint: Use True or False, not 1 or 0."
        )
    
    # Note: method='exact' and two_step=True is a valid combination.
    # In R's CBPS package, method='exact' sets bal.only=TRUE (only balance
    # conditions used for optimization), while twostep independently controls
    # whether analytical gradient is used in balance optimization.
    # twostep=TRUE → analytical gradient; twostep=FALSE → numerical gradient.
    # These two parameters are orthogonal and should NOT override each other.
    
    # Validate standardize parameter
    if not isinstance(standardize, bool):
        raise TypeError(
            f"standardize must be a boolean (True or False), got {type(standardize).__name__}. "
            f"Received: standardize={standardize}\n\n"
            f"Hint: Use True or False, not 1 or 0."
        )
    
    # Step 1: Formula path vs array path
    na_action_info = None  # Track missing value handling info
    
    # Initialize metadata variables (needed for all code paths)
    data_original = None
    terms_obj = None
    model_frame = None
    xlevels_obj = None

    if formula is not None:
        # Formula interface path
        
        # Validate data parameter type
        if data is None:
            raise ValueError(
                "data parameter is required when using formula interface. "
                "Please provide a pandas DataFrame containing the variables in your formula."
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame when using formula interface. "
                f"Got: {type(data).__name__}. "
                f"If you have a dict, convert it: pd.DataFrame(your_dict). "
                f"Or use the array interface: CBPS(treatment=..., covariates=...)"
            )
        
        # Validate formula type
        if not isinstance(formula, str):
            raise TypeError(
                f"formula must be a string, got {type(formula).__name__}. "
                f"Received: formula={formula}\n\n"
                f"Example of correct formula: 'treat ~ age + educ + black'"
            )
        
        # Validate formula format
        if '~' not in formula:
            raise ValueError(
                f"Formula must contain '~' to separate treatment from covariates. "
                f"Got: '{formula}'. "
                f"Example: 'treat ~ age + educ + black'"
            )
        
        # Step 1.1: Handle missing values
        # Extract columns involved in formula
        treat_col = formula.split('~')[0].strip()
        covar_cols = [col.strip() for col in formula.split('~')[1].split('+')]

        # Use exact column matching (avoid substring matching issues)
        relevant_cols = [treat_col] + covar_cols
        # Filter out columns not in data (handles I() and other functions)
        relevant_cols = [col for col in relevant_cols if col in data.columns]

        # Validate na_action parameter value
        valid_na_actions = {'warn', 'fail', 'ignore', 'omit'}
        if na_action not in valid_na_actions:
            raise ValueError(
                f"Invalid na_action='{na_action}'. "
                f"Valid options are: {', '.join(repr(x) for x in sorted(valid_na_actions))}. "
                f"Note: 'omit' is an alias for 'warn'."
            )
        
        # Alias mapping: 'omit' maps to 'warn'
        if na_action == 'omit':
            na_action = 'warn'
        
        # Check for missing values
        n_missing = data[relevant_cols].isna().any(axis=1).sum()
        if n_missing > 0:
            if na_action == 'fail':
                raise ValueError(
                    f"Missing values detected in {n_missing} observations. "
                    f"Set na_action='warn' to remove them, or handle missing values before calling CBPS()."
                )
            elif na_action == 'warn':
                from cbps.utils.helpers import handle_missing
                data_clean, n_dropped = handle_missing(data, relevant_cols)
                data = data_clean
                na_action_info = {'method': 'omit', 'n_dropped': n_dropped}
            elif na_action == 'ignore':
                # Ignore mode: silently remove missing values, still record info
                data_clean = data.dropna(subset=relevant_cols)
                n_dropped = len(data) - len(data_clean)
                data = data_clean
                na_action_info = {'method': 'ignore', 'n_dropped': n_dropped}

        from patsy import dmatrices, PatsyError
        from cbps.utils.formula import _convert_r_formula_to_patsy
        
        # Support dot formula (treat ~ .)
        # Expands 'y ~ .' to 'y ~ x1 + x2 + ...' since Patsy doesn't support dot syntax
        if isinstance(formula, str) and '~' in formula:
            parts = formula.split('~')
            if len(parts) == 2 and parts[1].strip() == '.':
                if data is None:
                    raise ValueError("Data must be provided when using dot formula ('~ .')")
                
                # Parse treatment variable name
                treat_part = parts[0].strip()
                
                # Extract real column name (handle C() or factor())
                import re
                real_treat_col = treat_part
                c_match = re.match(r'C\(([^)]+)\)', treat_part)
                factor_match = re.match(r'factor\(([^)]+)\)', treat_part)
                if c_match:
                    real_treat_col = c_match.group(1).strip()
                elif factor_match:
                    real_treat_col = factor_match.group(1).strip()
                    
                # Get all other columns
                other_cols = [c for c in data.columns if c != real_treat_col]
                if not other_cols:
                    raise ValueError("No covariates found in data (only treatment column exists)")
                    
                # Rebuild formula
                # Quote column names with spaces or special characters using Q()
                def _quote_if_needed(col):
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                        return f"Q('{col}')"
                    return col
                    
                rhs = ' + '.join([_quote_if_needed(c) for c in other_cols])
                formula = f"{treat_part} ~ {rhs}"

        # Convert R formula syntax to patsy
        formula = _convert_r_formula_to_patsy(formula)
        
        # Extract treatment variable from original data to avoid Patsy's one-hot encoding
        treat_col_name = formula.split('~')[0].strip()
        
        # Support C() and factor() syntax for explicit categorical specification
        import re
        
        # Detect C() or factor() wrapper
        categorical_from_formula = False
        c_match = re.match(r'C\(([^)]+)\)', treat_col_name)
        factor_match = re.match(r'factor\(([^)]+)\)', treat_col_name)
        
        if c_match:
            # 'C(treat)' -> 'treat'
            real_treat_col = c_match.group(1).strip()
            categorical_from_formula = True
        elif factor_match:
            # 'factor(treat)' -> 'treat'
            real_treat_col = factor_match.group(1).strip()
            categorical_from_formula = True
        else:
            # Plain column name
            real_treat_col = treat_col_name
        
        # Save treatment category names for summary display
        treat_categories_from_formula = None
        if real_treat_col in data.columns:
            # Extract treatment variable from original data (preserve Categorical type)
            treat_orig_series = data[real_treat_col]
            
            # If formula uses C() or factor(), force categorical flag
            if categorical_from_formula:
                is_treat_categorical = True
                # Convert to categorical if not already
                if not isinstance(treat_orig_series.dtype, pd.CategoricalDtype):
                    treat_orig_series = pd.Categorical(treat_orig_series)
                    treat_categories_from_formula = list(treat_orig_series.categories)
                # Extract numeric codes
                treat = treat_orig_series.codes if hasattr(treat_orig_series, 'codes') else treat_orig_series.cat.codes.to_numpy()
                if treat_categories_from_formula is None:
                    treat_categories_from_formula = list(treat_orig_series.categories if isinstance(treat_orig_series, pd.Categorical) else treat_orig_series.cat.categories)
            else:
                # Detect if categorical (check priority to avoid duplicate conversion)
                is_treat_categorical = (
                    isinstance(treat_orig_series.dtype, pd.CategoricalDtype) or
                    isinstance(treat_orig_series, pd.Categorical)
                )
                
                # Auto-convert string treatment to categorical
                if not is_treat_categorical:
                    treat = treat_orig_series.to_numpy()  # Convert to numpy array
                    
                    # Detect string/object type
                    if treat.dtype == object or pd.api.types.is_string_dtype(treat):
                        # Auto-convert to categorical
                        treat_orig_series = pd.Categorical(treat_orig_series)
                        treat = treat_orig_series.codes  # Convert to numeric codes
                        is_treat_categorical = True
                        # Save original category names
                        treat_categories_from_formula = list(treat_orig_series.categories)
                        warnings.warn(
                                f"Treatment variable '{real_treat_col}' is string/object type. "
                            f"Automatically converting to categorical with levels: {treat_categories_from_formula}.",
                            UserWarning
                        )
                else:
                    # Already categorical, extract numeric codes
                    # Note: Categorical Series to_numpy() returns original category values
                    # We need numeric codes instead
                    if hasattr(treat_orig_series, 'cat'):
                        treat = treat_orig_series.cat.codes.to_numpy()
                        # Save original category names
                        treat_categories_from_formula = list(treat_orig_series.cat.categories)
                    elif isinstance(treat_orig_series, pd.Categorical):
                        treat = treat_orig_series.codes
                        # Save original category names
                        treat_categories_from_formula = list(treat_orig_series.categories)
                    else:
                        treat = treat_orig_series.to_numpy()
            
            # Treatment type detection rules:
            #   - categorical/factor → discrete CBPS
            #   - numeric → continuous CBPS
            #   - only 0/1 binary values are auto-converted to factor
            if not is_treat_categorical:
                # Check for 0/1 binary values
                treat_unique = np.unique(treat)
                n_unique = len(treat_unique)
                is_binary_01 = (
                    n_unique == 2 and
                    set(treat_unique) <= {0, 1, 0.0, 1.0, False, True}
                )
                # Warn for float type binary treatment
                if is_binary_01 and np.issubdtype(treat.dtype, np.floating):
                    warnings.warn(
                        "Treatment variable is numeric (float) with only 2 unique values. "
                        "Interpreting as binary treatment. "
                        "Consider using int or Categorical type for clarity.",
                        UserWarning
                    )
                if is_binary_01:
                    is_treat_categorical = True
        else:
            raise ValueError(
                f"Treatment column '{real_treat_col}' not found in data.\n"
                f"Original formula: {formula}\n"
                f"Available columns: {list(data.columns)}"
            )
        
        # Handle C() or factor() on left-hand side of formula
        # Patsy encodes C(treat) as multiple dummy columns, but CBPS expects single vector
        # Solution: only use patsy for RHS, extract y from original data
        if categorical_from_formula:
            # Already extracted treat_orig_series from data
            # Construct RHS-only formula for patsy
            from patsy import dmatrix
            formula_rhs = '~' + formula.split('~')[1]
            try:
                X_design = dmatrix(formula_rhs, data, return_type='dataframe')
            except Exception as e:
                raise ValueError(
                    f"Failed to parse formula right-hand side: '{formula_rhs}'\n"
                    f"Error: {type(e).__name__}: {str(e)[:200]}"
                ) from e
        else:
            # Standard formula, use dmatrices to parse both sides
            # Wrap patsy errors with user-friendly messages
            try:
                _, X_design = dmatrices(formula, data, return_type='dataframe')
            except PatsyError as e:
                # Convert patsy-specific errors to friendlier messages
                raise ValueError(
                    f"Invalid formula syntax: '{formula}'\n"
                    f"Patsy error: {str(e)[:200]}\n\n"
                    f"Common issues:\n"
                    f"  - Undefined variables or functions\n"
                    f"  - Syntax errors in I() expressions\n"
                    f"  - Missing columns in data\n\n"
                    f"Formula format: 'treatment ~ covariate1 + covariate2 + ...'\n"
                    f"Examples:\n"
                    f"  - 'treat ~ age + educ + black'\n"
                    f"  - 'treat ~ age + I(age**2) + educ'\n"
                    f"  - 'treat ~ C(country) + income'"
                ) from e
            except NameError as e:
                # Function or variable undefined
                raise ValueError(
                    f"Invalid formula: '{formula}'\n"
                    f"Error: {str(e)}\n\n"
                    f"Make sure all variables exist in your data and all functions are defined.\n"
                    f"Available columns: {list(data.columns)}"
                ) from e
            except KeyError as e:
                # Column does not exist
                raise ValueError(
                    f"Invalid formula: '{formula}'\n"
                    f"Column not found in data: {str(e)}\n\n"
                    f"Available columns: {list(data.columns)}"
                ) from e
            except Exception as e:
                # Other unexpected errors
                raise ValueError(
                    f"Failed to parse formula: '{formula}'\n"
                    f"Error: {type(e).__name__}: {str(e)[:200]}\n\n"
                    f"Please check your formula syntax and data."
                ) from e
        
        X = X_design.values
        
        # Save terms object for predict() and update() methods
        terms_obj = X_design.design_info  # Patsy's DesignInfo object
        
        # Extract factor levels for predict() validation
        xlevels_dict = {}
        if hasattr(X_design, 'design_info') and hasattr(X_design.design_info, 'factor_infos'):
            for factor_name, factor_info in X_design.design_info.factor_infos.items():
                # Check if categorical variable
                if factor_info.type == 'categorical' and hasattr(factor_info, 'categories'):
                    # Extract variable name (remove EvalFactor wrapper)
                    var_name_str = str(factor_name)
                    # Handle C(var_name) format
                    if 'C(' in var_name_str and ')' in var_name_str:
                        var_name = var_name_str.split('C(')[1].split(')')[0]
                    else:
                        var_name = var_name_str
                    xlevels_dict[var_name] = list(factor_info.categories)
        xlevels_obj = xlevels_dict if xlevels_dict else None
        
        # Save original data
        data_original = data.copy()
        
        # Reorder columns: Intercept → regular vars (formula order) → I() function cols
        
        all_cols = list(X_design.columns)
        intercept_cols = [c for c in all_cols if c == 'Intercept']
        i_func_cols = [c for c in all_cols if c.startswith('I(') and c != 'Intercept']
        regular_cols = [c for c in all_cols if c not in intercept_cols and c not in i_func_cols]
        
        # Construct model frame containing all formula variables (after NA removal)
        model_cols = [real_treat_col]
        for col in regular_cols + i_func_cols:
            # Only include columns that exist in data (exclude I() expressions etc.)
            if col in data.columns:
                model_cols.append(col)
        model_frame = data[model_cols].copy() if len(model_cols) > 0 else data.copy()
        
        # Standard ordering: Intercept, regular vars, I() functions
        ordered_cols = intercept_cols + regular_cols + i_func_cols
        
        # Get column indices and reorder X and coef_names
        col_indices = [all_cols.index(c) for c in ordered_cols]
        X = X[:, col_indices]
        
        # Standardize column names to standard statistical modeling conventions
        # Format: "(Intercept)", "age", "I(re75 == 0)TRUE" (convert patsy's [T.True] suffix)
        coef_names = []
        for name in ordered_cols:
            if name == 'Intercept':
                coef_names.append('(Intercept)')  # Standard intercept notation
            elif '[T.True]' in name:
                # Remove patsy's [T.True] suffix, replace with TRUE
                coef_names.append(name.replace('[T.True]', 'TRUE'))
            elif '[T.False]' in name:
                coef_names.append(name.replace('[T.False]', 'FALSE'))
            else:
                coef_names.append(name)
        
        # Sync sample_weights dimensions when na_action removes rows
        if sample_weights is not None:
            original_sample_weights = sample_weights
            # If sample_weights is Series/DataFrame, use data index to select rows
            if isinstance(sample_weights, (pd.Series, pd.DataFrame)):
                if isinstance(sample_weights, pd.DataFrame):
                    sample_weights = sample_weights.iloc[:, 0].values
                else:
                    sample_weights = sample_weights.loc[data.index].values
            else:
                # If numpy array, check dimension match
                sample_weights = np.asarray(sample_weights)
                if len(sample_weights) != len(treat):
                    # Dimension mismatch with array type, cannot auto-sync
                    warnings.warn(
                        f"sample_weights length ({len(original_sample_weights)}) does not match "
                        f"the number of valid observations after removing missing values ({len(treat)}). "
                        f"Setting sample_weights to None (equal weights). "
                        f"To avoid this, provide sample_weights as a pandas Series with matching index, "
                        f"or handle missing values before calling CBPS().",
                        UserWarning
                    )
                    sample_weights = None
    elif treatment is not None and covariates is not None:
        # Array interface path
        treat_original = treatment  # Save for type detection
        
        # Convert to numpy array (required by core algorithms)
        if isinstance(treatment, (pd.Series, pd.Categorical)):
            treat = np.asarray(treatment).ravel()
        else:
            treat = np.asarray(treatment).ravel()
        X = np.asarray(covariates)
        
        # Validate covariates dimensions (must be 2D)
        if X.ndim == 0:
            raise ValueError(
                f"covariates must be a 2D array with shape (n_samples, n_features). "
                f"Got a scalar (0-dimensional array).\n"
                f"Expected shape: ({len(treat)}, k) where k >= 1.\n"
                f"If you have a single covariate, reshape it: X.reshape(-1, 1)"
            )
        elif X.ndim == 1:
            raise ValueError(
                f"covariates must be a 2D array with shape (n_samples, n_features). "
                f"Got a 1D array with shape {X.shape}.\n"
                f"Expected shape: ({len(treat)}, k) where k >= 1.\n\n"
                f"To fix this:\n"
                f"  - If you have a single covariate: X.reshape(-1, 1)\n"
                f"  - If you passed the transposed matrix: X.T\n\n"
                f"Current shapes:\n"
                f"  treatment: {treat.shape}\n"
                f"  covariates: {X.shape}"
            )
        elif X.ndim > 2:
            raise ValueError(
                f"covariates must be a 2D array with shape (n_samples, n_features). "
                f"Got a {X.ndim}-dimensional array with shape {X.shape}.\n"
                f"Expected shape: ({len(treat)}, k) where k >= 1."
            )
        
        # Validate treatment and covariates have matching lengths
        if len(treat) != X.shape[0]:
            raise ValueError(
                f"Treatment and covariates must have the same number of samples.\n"
                f"  treatment length: {len(treat)}\n"
                f"  covariates rows: {X.shape[0]}\n\n"
                f"Please ensure treatment and covariates come from the same dataset."
            )
        
        # Auto-add intercept column if not present
        if not _has_intercept(X):
            if verbose > 0:
                warnings.warn(
                    "Intercept column not detected. Adding intercept to covariates matrix. "
                    "To suppress this warning, manually add intercept: "
                    "np.column_stack([np.ones(n), X])",
                    UserWarning
                )
            X = np.column_stack([np.ones(len(treat)), X])
        
        # Generate default column names
        if isinstance(covariates, pd.DataFrame):
            coef_names = covariates.columns.tolist()
            # If intercept was added, prepend "Intercept" to column names
            if not _has_intercept(np.asarray(covariates)):
                coef_names = ["Intercept"] + coef_names
        else:
            k = X.shape[1]
            coef_names = ["Intercept"] + [f"X{i}" for i in range(1, k)]
    else:
        raise ValueError(
            "Must provide either 'formula' and 'data', or 'treatment' and 'covariates'"
        )
    
    # Step 1.5: Dual formula parsing (oCBPS path)
    baseline_X = None
    diff_X = None
    
    # Check if baseline/diff formula is provided
    has_baseline_or_diff = (baseline_formula is not None or diff_formula is not None)
    
    if has_baseline_or_diff:
        # Check data parameter first
        if data is None:
            raise ValueError(
                "The data parameter is required when using baseline_formula or diff_formula.\n"
                "These parameters require access to the original DataFrame to parse formulas."
            )
        
        # Extract treatment variable and detect type
        treat_for_check = None
        treat_col_name_for_check = None
        
        if formula is not None:
            # Formula path: extract treatment from data
            treat_col_name_raw = formula.split('~')[0].strip()
            
            # Handle C() and factor() syntax
            import re
            c_match = re.match(r'C\(([^)]+)\)', treat_col_name_raw)
            factor_match = re.match(r'factor\(([^)]+)\)', treat_col_name_raw)
            
            if c_match:
                treat_col_name_for_check = c_match.group(1).strip()
            elif factor_match:
                treat_col_name_for_check = factor_match.group(1).strip()
            else:
                treat_col_name_for_check = treat_col_name_raw
            
            if treat_col_name_for_check in data.columns:
                treat_for_check = data[treat_col_name_for_check].to_numpy()
        elif treatment is not None:
            # Array path: use treatment directly
            treat_for_check = treatment
        
        # Call unified treatment type detection function
        if treat_for_check is not None:
            is_cat, is_bin, is_cont = _detect_treatment_type(
                treat_for_check,
                formula=formula,
                data=data,
                treat_col_name=treat_col_name_for_check
            )
            
            # Reject continuous treatment immediately (takes priority over XOR check)
            if is_cont:
                raise ValueError(
                    "baseline_formula and diff_formula are only supported for binary treatments.\n"
                    "Optimal CBPS is not defined for continuous treatments.\n"
                    "\n"
                    "Reference:\n"
                    "  Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., & Yang, X. (2022).\n"
                    "  Optimal Covariate Balancing Conditions in Propensity Score Estimation.\n"
                    "  Journal of Business & Economic Statistics, 41(1), 97-110.\n"
                    "\n"
                    "For continuous treatments, use the standard CBPS without baseline/diff formulas."
                )
        
        # Passed continuous treatment check, now check XOR (binary treatment only)
        if (baseline_formula is None) != (diff_formula is None):
            raise ValueError(
                "Both baseline_formula and diff_formula must be specified together, or neither.\n"
                f"Currently: baseline_formula={'provided' if baseline_formula else 'None'}, "
                f"diff_formula={'provided' if diff_formula else 'None'}.\n"
                "\n"
                "Either specify both formulas to use iCBPS (Optimal CBPS), or leave both as None."
            )
        
        # Dual formula parsing
        from patsy import dmatrix
        
        # Parse baseline formula
        baseline_X_raw = dmatrix(baseline_formula, data, return_type='dataframe').values
        # Filter zero-variance columns (intercept with sd=0 will be removed)
        baseline_X = baseline_X_raw[:, baseline_X_raw.std(axis=0, ddof=1) > 0]
        
        # Parse diff formula
        diff_X_raw = dmatrix(diff_formula, data, return_type='dataframe').values
        # Filter zero-variance columns
        diff_X = diff_X_raw[:, diff_X_raw.std(axis=0, ddof=1) > 0]

    # Step 1.5.5a: Basic dimension and sample size checks (must execute first)
    n = len(treat)
    
    # Handle empty array (n=0)
    if n == 0:
        raise ValueError(
            "Treatment array is empty (n=0). "
            "CBPS requires at least 10 observations for valid inference."
        )
    
    # Zero variance check takes priority over sample size check
    # Check if treatment variable has variance (all values identical)
    if n > 1:  # Only check variance when n > 1
        # Get unique value count (works for all types)
        unique_vals = np.unique(treat)
        n_unique = len(unique_vals)
        
        if n_unique == 1:
            # All values identical, cannot estimate propensity score
            raise ValueError(
                f"Treatment variable has zero variance. "
                f"All {n} observations have the same treatment value (treat={unique_vals[0]}). "
                f"CBPS requires variation in the treatment variable to estimate propensity scores. "
                f"Please check your data for errors or use a different subset with treatment variation."
            )
        
        # For numeric types, also check if std is too small (near-constant)
        # Skip Categorical/string types (cannot compute std)
        is_categorical = hasattr(treat, 'categories') or (
            hasattr(treat, 'dtype') and hasattr(treat.dtype, 'categories')
        )
        is_string_dtype = (
            hasattr(treat, 'dtype') and 
            (treat.dtype.kind == 'U' or treat.dtype.kind == 'O' or treat.dtype.kind == 'S')
        )
        
        if not is_categorical and not is_string_dtype and n_unique > 1:
            try:
                treat_numeric = np.asarray(treat, dtype=np.float64)
                treat_std = np.std(treat_numeric, ddof=1)
                if treat_std == 0 or np.isclose(treat_std, 0):
                    # Numeric treatment with zero std but multiple unique values (rare)
                    raise ValueError(
                        f"Treatment variable has zero or near-zero variance (std={treat_std:.2e}). "
                        f"CBPS requires sufficient treatment variation for stable estimation."
                    )
            except (ValueError, TypeError):
                # Cannot convert to numeric, skip std check (handled in type detection)
                pass
    
    # Reject n<10 (statistically meaningless)
    if n < 10:
        raise ValueError(
            f"Sample size (n={n}) too small for CBPS (minimum: n ≥ 10). "
            f"CBPS relies on asymptotic (large-sample) theory for valid inference. "
            f"With n<10, standard errors and confidence intervals are completely invalid. "
            f"Current sample provides insufficient degrees of freedom for reliable estimation."
        )
    
    # Step 1.5.5b: Input validation - detect inf/nan values
    try:
        _validate_finite_inputs(treat, X, func_name="CBPS")
    except ValueError as e:
        # Provide friendlier error message for formula interface
        if formula is not None:
            raise ValueError(
                f"{e}\n"
                f"Formula used: '{formula}'\n"
                f"Hint: Check your data for log(0), division by zero, or missing values."
            ) from e
        else:
            raise
    
    # Step 1.6: Zero-variance covariate filtering
    # Auto-drop zero-variance columns (except intercept) for numerical stability
    if X.shape[1] > 1:  # If there are columns besides intercept
        # Compute std for all columns except intercept
        x_sd = X[:, 1:].std(axis=0, ddof=1)
        const_threshold = 1e-10
        non_const_mask = x_sd > const_threshold

        # Check if any constant columns need to be dropped
        n_const_cols = np.sum(~non_const_mask)
        if n_const_cols > 0:
            # Record dropped column names if available
            if 'coef_names' in locals() and len(coef_names) == X.shape[1]:
                const_col_names = [coef_names[i+1] for i, is_const in enumerate(~non_const_mask) if is_const]
                warnings.warn(
                    f"Dropping {n_const_cols} constant covariate(s) with zero variance: "
                    f"{const_col_names}.",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"Dropping {n_const_cols} constant covariate(s) with zero variance.",
                    UserWarning
                )

            # Keep intercept + non-constant columns
            X = np.column_stack([X[:, 0], X[:, 1:][:, non_const_mask]])

            # Update column names accordingly
            if 'coef_names' in locals() and len(coef_names) == X.shape[1] + n_const_cols:
                coef_names = [coef_names[0]] + [coef_names[i+1] for i, is_non_const in enumerate(non_const_mask) if is_non_const]
    
    # Reject intercept-only model (CBPS requires covariates to balance)
    if X.shape[1] <= 1:
        raise ValueError(
            f"CBPS requires at least one covariate (non-intercept) for covariate balancing.\n"
            f"Formula '{formula if formula else 'array input'}' resulted in design matrix with only intercept.\n\n"
            f"Explanation:\n"
            f"  CBPS = Covariate Balancing Propensity Score\n"
            f"  Without covariates, there is nothing to balance.\n\n"
            f"Theoretical reference:\n"
            f"  Imai & Ratkovic (2014) Equation 8 requires covariates X_i for balance conditions.\n\n"
            f"Please add covariates to your formula, for example:\n"
            f"  'treat ~ age + education + income'\n"
            f"  'treat ~ x1 + x2 + I(x1**2)'\n\n"
            f"Current design matrix shape: {X.shape}"
        )

    # Step 1.7: Rank check for collinearity detection
    rank_X = np.linalg.matrix_rank(X)
    k = X.shape[1]
    
    if rank_X < k:
        # Provide detailed error message to help diagnose the issue
        raise ValueError(
            f"Covariate matrix X is not full rank (rank={rank_X} < {k}). "
            f"This indicates perfect collinearity among covariates. "
            f"Possible causes:\n"
            f"  - Linear combinations (e.g., X2 = 2*X1 + 3)\n"
            f"  - Duplicate columns (e.g., X2 = X1)\n"
            f"  - Redundant interactions or polynomial terms\n"
            f"Please remove or combine collinear covariates. "
            f"Use variance inflation factor (VIF) or correlation matrix to diagnose."
        )
    
    # Optional: Condition number warning for near-collinearity
    # High condition number indicates X'X is near-singular
    cond_num = np.linalg.cond(X)
    if cond_num > 1e10:
        warnings.warn(
            f"Covariate matrix X has very high condition number ({cond_num:.2e}). "
            f"This suggests near-collinearity, which may cause numerical instability. "
            f"Consider:\n"
            f"  - Removing highly correlated covariates (check correlation matrix)\n"
            f"  - Centering and scaling variables\n"
            f"  - Using regularization (hdCBPS for high-dimensional settings)",
            UserWarning
        )

    # Step 1.8: Relative sample size check
    k = X.shape[1]
    
    # Warn for small samples (10 ≤ n < 30)
    if n < 30:
        warnings.warn(
            f"Small sample size (n={n}, recommended minimum: n ≥ 30). "
            f"CBPS standard errors rely on asymptotic normality which may not hold well for small samples. "
            f"Consider:\n"
            f"  - Using bootstrap for more reliable confidence intervals\n"
            f"  - Reporting results with appropriate caution\n"
            f"  - Collecting more data if possible",
            UserWarning
        )
    
    # Warn for low n/k ratio (insufficient relative sample size)
    if n <= k + 5:
        warnings.warn(
            f"Sample size (n={n}) small relative to number of parameters (k={k}). "
            f"Ratio n/k={n/k:.2f} is low (recommended: n/k ≥ 5). "
            f"Consider reducing the number of covariates for more stable estimates.",
            UserWarning
        )

    # Step 2: Construct call_info
    if formula is not None:
        call_info = (f"CBPS(formula='{formula}', data=<DataFrame>, "
                    f"att={att}, method='{method}', two_step={two_step})")
    else:
        call_info = (f"CBPS(treatment=<array>, covariates=<array>, "
                    f"att={att}, method='{method}')")
    
    # Step 3: Treatment type detection and routing
    # Important: detect factor/categorical before numeric
    # (pd.Categorical.dtype can be int64, causing misclassification)
    
    # Debug output for treatment type detection
    if verbose > 1:
        print(f"DEBUG: Treatment type detection")
        if formula is not None:
            print(f"  Formula path: is_treat_categorical={is_treat_categorical}")
        print(f"  treat unique values: {np.unique(treat)}")
        print(f"  treat dtype: {treat.dtype}")
    
    # Discrete treatment detection (factor/categorical takes priority)
    if formula is not None:
        # Formula path: use saved is_treat_categorical
        is_factor = is_treat_categorical
        if verbose > 1:
            print(f"  Formula path: is_factor={is_factor}")
    else:
        # Array path: detect Categorical or 0/1 binary values
        # Only 0/1 binary is auto-converted to factor (other numeric stays continuous)
        treat_unique = np.unique(treat)
        n_unique = len(treat_unique)

        # Check for 0/1 binary
        is_binary_01 = (
            n_unique == 2 and
            set(treat_unique) <= {0, 1, 0.0, 1.0, False, True}
        )
        
        # Warn for float-type binary treatment
        if is_binary_01 and np.issubdtype(treat_original.dtype, np.floating):
            warnings.warn(
                "Treatment variable is numeric (float) with only 2 unique values. "
                "Interpreting as binary treatment. "
                "Consider using int or Categorical type for clarity.",
                UserWarning
            )

        is_factor = (
            isinstance(treat_original, pd.Categorical) or
            hasattr(treat_original, 'cat') or
            is_binary_01  # Only 0/1 binary auto-detected as discrete
        )
    
    # Continuous treatment detection
    # Numeric and not factor = continuous (regardless of unique value count)
    is_continuous = (
        not is_factor and
        np.issubdtype(treat.dtype, np.number)
    )

    if is_continuous:
        # Warn if numeric treatment has few unique values (may be discrete)
        n_unique = len(np.unique(treat))
        if n_unique <= 4:
            warnings.warn(
                f"Treatment vector is numeric with {n_unique} unique values. "
                f"Interpreting as a continuous treatment. "
                f"To solve for a binary or multi-valued treatment, convert treat to categorical "
                f"(e.g., pd.Categorical(treat) or treat.astype('category')).",
                UserWarning
            )
        
        # Continuous treatment does not support ATT, warn and ignore
        if att != 0:
            warnings.warn(
                f"ATT parameter (att={att}) is not supported for continuous treatments. "
                f"Continuous CBPS only estimates the Average Treatment Effect (ATE). "
                f"The att parameter will be ignored. "
                f"\n\nReason: ATT (Average Treatment Effect on the Treated) requires a binary "
                f"distinction between 'treated' and 'control' groups, which does not exist "
                f"for continuous treatments. "
                f"\n\nTheoretical reference: Fong, Hazlett & Imai (2018, Annals of Applied Statistics) "
                f"define stabilized weights for continuous treatments that estimate ATE only. "
                f"\n\nNote: For non-binary treatments, only the ATE is available.",
                UserWarning
            )
        
        # Call continuous CBPS
        from cbps.core.cbps_continuous import cbps_continuous_fit
        
        # Apply SVD preprocessing
        X_orig = X.copy()  # Save original X for inverse transform
        X_svd, svd_info = _apply_svd_preprocessing(X)
        
        # Compute rank and XprimeX_inv in SVD space
        k = np.linalg.matrix_rank(X_svd)
        if k < X_svd.shape[1]:
            raise ValueError("X is not full rank")
        
        # Compute XprimeX_inv in SVD space
        if sample_weights is None:
            sample_weights_norm = np.ones(len(treat))
        else:
            sample_weights_norm = sample_weights / np.mean(sample_weights)
        
        sw_sqrt_X = np.sqrt(sample_weights_norm)[:, None] * X_svd
        XprimeX = sw_sqrt_X.T @ sw_sqrt_X
        from cbps.core.cbps_binary import _r_ginv
        XprimeX_inv = _r_ginv(XprimeX)
        
        result_dict = cbps_continuous_fit(
            treat, X_svd,  # Pass SVD-preprocessed X
            method=method,
            two_step=two_step,
            iterations=iterations,
            standardize=standardize,
            sample_weights=sample_weights,
            verbose=verbose
        )
        
        # SVD inverse transform
        beta_svd = result_dict['coefficients']
        if beta_svd.ndim == 1:
            beta_svd = beta_svd.reshape(-1, 1)
        beta_transformed = _apply_svd_inverse_transform(beta_svd, svd_info)
        result_dict['coefficients'] = beta_transformed
        
        # Update x to original X
        result_dict['x'] = X_orig
        
        # Wrap in CBPSResults
        result = CBPSResults(
            **result_dict,
            coef_names=coef_names,
            call_info=call_info,
            formula=formula,
            data=data_original if formula is not None else None,
            terms=terms_obj if formula is not None else None,
            model=model_frame if formula is not None else None,
            xlevels=xlevels_obj if formula is not None else None,
            att=att,
            method=method,
            standardize=standardize,
            two_step=two_step
        )
        
        return result
    
    # Discrete treatment routing
    # Detect treatment levels (prioritize saved category names from formula interface)
    if formula is not None and 'treat_categories_from_formula' in locals() and treat_categories_from_formula is not None:
        treat_levels = np.array(treat_categories_from_formula)
    elif isinstance(treat, pd.Categorical):
        treat_levels = treat.categories.values
    elif hasattr(treat, 'cat'):  # pandas Series with categorical dtype
        treat_levels = treat.cat.categories.values
    else:
        treat_levels = np.unique(treat)
    
    # Sort treat_levels for consistent baseline (MNLogit uses treat_levels[0] as baseline)
    treat_levels = np.sort(treat_levels)
    
    # Re-encode if treat uses categorical codes to align with sorted levels
    if formula is not None and ('treat_orig_series' in locals()):
        if hasattr(treat_orig_series, 'cat') or isinstance(treat_orig_series, pd.Categorical):
            # Re-encode: map original values to sorted indices
            if isinstance(treat_orig_series, pd.Categorical):
                treat_original_values = np.asarray(treat_orig_series)
            else:
                treat_original_values = treat_orig_series.to_numpy()
            value_to_sorted_index = {val: i for i, val in enumerate(treat_levels)}
            treat = np.array([value_to_sorted_index[val] for val in treat_original_values])
    
    no_treats = len(treat_levels)
    
    # Validate treatment level count
    if no_treats > 4:
        raise ValueError(
            "Parametric CBPS is not implemented for more than 4 treatment values. "
            "Consider using a continuous value."
        )
    if no_treats < 2:
        raise ValueError("Treatment must take more than one value")
    
    # theoretical_exact not supported for multi-valued treatments
    if no_treats >= 3 and theoretical_exact:
        raise ValueError(
            f"theoretical_exact=True is not supported for multi-valued treatments ({no_treats} levels). "
            f"theoretical_exact is an experimental feature for binary treatments only.\n\n"
            f"Please set theoretical_exact=False (default) or use binary treatment."
        )
    
    # Multi-valued treatment ATT handling (only ATE supported for 3+ levels)
    if no_treats >= 3 and att != 0:
        warnings.warn(
            f"Multi-valued treatment ({no_treats} levels) only supports att=0 (ATE). "
            f"ATT parameter (att={att}) will be overridden to att=0.\n\n"
            f"Reason: ATT requires a binary distinction between 'treated' and 'control'. "
            f"With {no_treats} levels, there is no single 'treated' group.\n\n"
            f"Reference: Imai & Ratkovic (2014), JRSS-B, Section 4.1.",
            UserWarning
        )
        att = 0  # Force ATE
    
    # Binary treatment routing
    if no_treats == 2:
        # Handle att=2 encoding reversal
        from cbps.utils.helpers import encode_treatment_factor

        # Save original treat for result object
        treat_original_for_results = treat.copy() if isinstance(treat, np.ndarray) else treat

        # oCBPS path check - must be done BEFORE encoding to prevent att=2 reversal
        is_ocbps_path = baseline_X is not None and diff_X is not None
        
        # For oCBPS, force att=0 BEFORE encoding to match R behavior
        att_for_encoding = att
        if is_ocbps_path and att != 0:
            warnings.warn(
                f"CBPSOptimal only supports att=0 (ATE). "
                f"Received att={att}, forcing to att=0. "
                f"Treatment encoding will NOT be reversed.",
                UserWarning
            )
            att_for_encoding = 0  # Force ATE encoding for oCBPS

        # Apply ATT encoding logic for binary treatment
        if formula is not None and 'treat_orig_series' in locals() and is_treat_categorical:
            # Formula path: use original categorical series
            treat_encoded, treat_levels_ordered, treat_orig = encode_treatment_factor(treat_orig_series, att_for_encoding, verbose=verbose)
        else:
            # Array path or treat is already numeric
            treat_encoded, treat_levels_ordered, treat_orig = encode_treatment_factor(treat, att_for_encoding, verbose=verbose)

        # Update treat to encoded 0/1 array
        treat = treat_encoded

        # Normalize att to 0 or 1 (encoding already handles att=2 reversal)
        # att=0 → 0 (ATE), att=1 → 1 (ATT), att=2 → 1 (ATT with reversed encoding)
        # For oCBPS, att_for_encoding is always 0, so att_normalized will be 0
        att_normalized = 0 if att_for_encoding == 0 else 1

        # oCBPS routing
        if is_ocbps_path:
            # oCBPS path - only supports ATE (att=0)
            # Warning already issued above if att != 0

            # Force ATT=0 for oCBPS
            from cbps.core.cbps_optimal import cbps_optimal_2treat
            result_dict = cbps_optimal_2treat(
                treat, X, baseline_X, diff_X,
                iterations=iterations,
                att=0,  # Force to 0
                standardize=standardize
            )
        elif baseline_X is not None or diff_X is not None:
            # Only one of baseline_X/diff_X provided - invalid for oCBPS
            raise ValueError(
                "For oCBPS (optimal CBPS), both baseline_formula and diff_formula "
                "(or baseline_X and diff_X) must be provided. "
                f"Received: baseline={'provided' if baseline_X is not None else 'None'}, "
                f"diff={'provided' if diff_X is not None else 'None'}. "
                "Either provide both for oCBPS, or neither for standard CBPS."
            )
        else:
            # Standard CBPS path
            # Apply SVD preprocessing (matching R package CBPSMain.R lines 307-314)
            X_orig_binary = X.copy()
            X_svd_binary, svd_info_binary = _apply_svd_preprocessing(X)

            # Compute rank check in SVD space
            k_binary = np.linalg.matrix_rank(X_svd_binary)
            if k_binary < X_svd_binary.shape[1]:
                raise ValueError("X is not full rank")

            # Compute XprimeX_inv in SVD space
            if sample_weights is None:
                sw_norm_binary = np.ones(len(treat))
            else:
                sw_norm_binary = sample_weights / np.mean(sample_weights)
            sw_sqrt_X_binary = np.sqrt(sw_norm_binary)[:, None] * X_svd_binary
            XprimeX_binary = sw_sqrt_X_binary.T @ sw_sqrt_X_binary
            from cbps.core.cbps_binary import _r_ginv
            XprimeX_inv_binary = _r_ginv(XprimeX_binary)

            result_dict = cbps_binary_fit(
                treat, X_svd_binary,  # Pass SVD-transformed X
                att=att_normalized,
                method=method,
                two_step=two_step,
                standardize=standardize,
                sample_weights=sample_weights,
                iterations=iterations,
                XprimeX_inv=XprimeX_inv_binary,
                theoretical_exact=theoretical_exact,
                verbose=verbose,
                # R-matching optimizer tolerances (only set if user hasn't specified)
                bal_gtol=kwargs.pop('bal_gtol', 1e-6),
                gmm_gtol=kwargs.pop('gmm_gtol', 1e-10),
                **kwargs
            )

            # SVD inverse transform for coefficients
            # R: beta.opt = V %*% diag(d.inv) %*% coef(output)
            # R: beta.opt[-1,] = beta.opt[-1,] / x.sd
            # R: beta.opt[1,] = beta.opt[1,] - x.mean %*% beta.opt[-1,]
            beta_svd_binary = result_dict['coefficients']  # (k, 1)
            beta_transformed_binary = _apply_svd_inverse_transform(
                beta_svd_binary, svd_info_binary
            )
            result_dict['coefficients'] = beta_transformed_binary

            # SVD inverse transform for variance-covariance matrix
            # R: Dx.inv %*% ginv(X.orig'X.orig) %*% X.orig' %*% X_svd %*% V %*%
            #    ginv(diag(d)) %*% var %*% ginv(diag(d)) %*% V' %*% X_svd' %*%
            #    X.orig %*% ginv(X.orig'X.orig) %*% Dx.inv
            variance_svd = result_dict['var']
            x_sd = svd_info_binary['x_sd']
            x_mean = svd_info_binary['x_mean']
            V_mat = svd_info_binary['V']
            d_vals = svd_info_binary['d']
            X_svd_mat = X_svd_binary  # U matrix

            # Dx_inv in R is diag(c(1, x.sd)) — note: R's naming is misleading
            Dx = np.diag(np.concatenate([[1.0], x_sd]))

            # d_inv for variance transform
            d_inv_var = d_vals.copy()
            d_inv_var[d_inv_var > 1e-5] = 1.0 / d_inv_var[d_inv_var > 1e-5]
            d_inv_var[d_inv_var <= 1e-5] = 0.0

            # Build transform matrix A:
            # A = Dx %*% ginv(X.orig'X.orig) %*% X.orig' %*% X_svd %*% V %*% diag(d_inv)
            XoXo_inv = _r_ginv(X_orig_binary.T @ X_orig_binary)
            A = (Dx @ XoXo_inv @ X_orig_binary.T @ X_svd_mat
                 @ V_mat @ np.diag(d_inv_var))

            # var_transformed = A %*% variance %*% A'
            result_dict['var'] = A @ variance_svd @ A.T

            # Restore original X (fitted_values and weights are preserved by SVD)
            result_dict['x'] = X_orig_binary
    
    # 3-level treatment routing
    elif no_treats == 3:
        from cbps.core.cbps_multitreat import cbps_3treat_fit

        # Convert method to bal_only flag
        bal_only = (method == 'exact')

        # Apply SVD preprocessing
        X_orig = X.copy()  # Save original X
        X_svd, svd_info = _apply_svd_preprocessing(X)

        # Compute rank and XprimeX_inv
        k = np.linalg.matrix_rank(X_svd)
        if k < X_svd.shape[1]:
            raise ValueError("X is not full rank")

        # Compute XprimeX_inv in SVD space
        if sample_weights is None:
            sample_weights_norm = np.ones(len(treat))
        else:
            sample_weights_norm = sample_weights / np.mean(sample_weights)

        sw_sqrt_X = np.sqrt(sample_weights_norm)[:, None] * X_svd
        XprimeX = sw_sqrt_X.T @ sw_sqrt_X
        from cbps.core.cbps_binary import _r_ginv
        XprimeX_inv = _r_ginv(XprimeX)

        # Call 3-level fit in SVD space
        result_dict = cbps_3treat_fit(
            treat=treat,
            X=X_svd,  # SVD-orthogonalized matrix
            method=method,
            k=k,
            XprimeX_inv=XprimeX_inv,
            bal_only=bal_only,
            iterations=iterations,
            standardize=standardize,
            two_step=two_step,
            sample_weights=sample_weights,
            treat_levels=treat_levels,
            verbose=verbose
        )

        # SVD inverse transform
        beta_svd = result_dict['coefficients']  # (k, 2)
        beta_transformed = _apply_svd_inverse_transform(beta_svd, svd_info)

        # Update coefficients in result_dict
        result_dict['coefficients'] = beta_transformed
        result_dict['x'] = X_orig  # Restore original X
        
        # Recompute fitted_values and linear_predictor with original X and transformed beta
        theta_transformed = X_orig @ beta_transformed  # (n, 2)
        
        # Recompute softmax probabilities (numerically stable)
        from cbps.core.cbps_multitreat import PROBS_MIN, _compute_softmax_probs_3treat
        probs_transformed = _compute_softmax_probs_3treat(theta_transformed, PROBS_MIN)
        
        # Update result_dict
        result_dict['fitted_values'] = probs_transformed
        result_dict['linear_predictor'] = theta_transformed
        
        # Add treat_names for result object
        treat_names = [str(level) for level in treat_levels]
    
    # 4-level treatment routing
    elif no_treats == 4:
        from cbps.core.cbps_multitreat import cbps_4treat_fit

        bal_only = (method == 'exact')

        # Apply SVD preprocessing
        X_orig = X.copy()  # Save original X
        X_svd, svd_info = _apply_svd_preprocessing(X)

        # Compute rank and XprimeX_inv
        k = np.linalg.matrix_rank(X_svd)
        if k < X_svd.shape[1]:
            raise ValueError("X is not full rank")

        if sample_weights is None:
            sample_weights_norm = np.ones(len(treat))
        else:
            sample_weights_norm = sample_weights / np.mean(sample_weights)

        sw_sqrt_X = np.sqrt(sample_weights_norm)[:, None] * X_svd
        XprimeX = sw_sqrt_X.T @ sw_sqrt_X
        from cbps.core.cbps_binary import _r_ginv
        XprimeX_inv = _r_ginv(XprimeX)

        # Call 4-level fit in SVD space
        result_dict = cbps_4treat_fit(
            treat=treat,
            X=X_svd,  # SVD-orthogonalized matrix
            method=method,
            k=k,
            XprimeX_inv=XprimeX_inv,
            bal_only=bal_only,
            iterations=iterations,
            standardize=standardize,
            two_step=two_step,
            sample_weights=sample_weights,
            treat_levels=treat_levels,
            verbose=verbose
        )

        # SVD inverse transform
        beta_svd = result_dict['coefficients']  # (k, 3)
        beta_transformed = _apply_svd_inverse_transform(beta_svd, svd_info)

        # Update result_dict
        result_dict['coefficients'] = beta_transformed
        result_dict['x'] = X_orig
        
        # Recompute fitted_values and linear_predictor with original X and transformed beta
        theta_transformed = X_orig @ beta_transformed  # (n, 3)
        
        # Recompute softmax probabilities (numerically stable)
        from cbps.core.cbps_multitreat import PROBS_MIN, _compute_softmax_probs_4treat
        probs_transformed = _compute_softmax_probs_4treat(theta_transformed, PROBS_MIN)
        
        # Update result_dict
        result_dict['fitted_values'] = probs_transformed
        result_dict['linear_predictor'] = theta_transformed
        
        # Add treat_names for result object
        treat_names = [str(level) for level in treat_levels]
    
    # Step 4: Wrap in CBPSResults object
    if no_treats in [3, 4]:
        result = CBPSResults(
            **result_dict,
            coef_names=coef_names,
            call_info=call_info,
            formula=formula,
            na_action=na_action_info,
            data=data_original,
            terms=terms_obj,
            model=model_frame,
            xlevels=xlevels_obj,
            treat_names=treat_names,
            att=att,
            method=method,
            standardize=standardize,
            two_step=two_step
        )
    else:
        result = CBPSResults(
            **result_dict,
            coef_names=coef_names,
            call_info=call_info,
            formula=formula,
            na_action=na_action_info,
            data=data_original if formula is not None else None,
            terms=terms_obj if formula is not None else None,
            model=model_frame if formula is not None else None,
            xlevels=xlevels_obj if formula is not None else None,
            att=att,
            method=method,
            standardize=standardize,
            two_step=two_step
        )
    
    # Check for overlap violation
    _check_overlap_violation(result, is_continuous)

    return result


def cbps_fit(
    treat: Union[np.ndarray, pd.Series, pd.Categorical],
    X: np.ndarray,
    method: str = 'over',
    att: int = 1,
    two_step: bool = True,
    standardize: bool = True,
    iterations: int = 1000,
    sample_weights: Optional[np.ndarray] = None,
    baseline_X: Optional[np.ndarray] = None,
    diff_X: Optional[np.ndarray] = None,
    verbose: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Low-level CBPS fitting function (type detection and routing).
    
    Performs treatment type detection, SVD preprocessing, routes to specific
    algorithm, applies SVD inverse transform, and returns raw dict (not wrapped
    in CBPSResults object).

    Parameters
    ----------
    treat : np.ndarray or pd.Series or pd.Categorical, shape (n,)
        Treatment variable.
        - pd.Categorical or pd.Series with categorical dtype: discrete treatment
        - np.ndarray (int/float): numeric treatment (0/1 auto-converted to factor)
    X : np.ndarray, shape (n, k)
        Covariate matrix, first column is intercept (all ones).
    method : {'over', 'exact'}, default='over'
        'over': over-identified GMM (default)
        'exact': exactly identified
    att : int, {0, 1}, default=1
        0: ATE (Average Treatment Effect)
        1: ATT (Average Treatment Effect on Treated)
    two_step : bool, default=True
        Whether to use two-step estimation.
    standardize : bool, default=True
        Whether to standardize.
    iterations : int, default=1000
        Maximum iterations.
    sample_weights : np.ndarray, optional
        Sample weights (observation-level).
    baseline_X : np.ndarray, optional
        Baseline outcome covariate matrix for oCBPS.
    diff_X : np.ndarray, optional
        Treatment effect covariate matrix for oCBPS.
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed).
    **kwargs
        Additional arguments passed to underlying algorithm.
    
    Returns
    -------
    dict
        Dictionary containing all fitting results:
        - 'coefficients': coefficient matrix
        - 'fitted_values': fitted propensity scores
        - 'weights': inverse probability weights
        - 'y': treatment variable
        - 'x': covariate matrix (original space)
        - 'converged': convergence status
        - 'J': J statistic
        - 'var': variance-covariance matrix
        - other fields vary by treatment type
    
    Notes
    -----
    Difference from CBPS main function:
    - cbps_fit is low-level API, accepts numpy arrays instead of formulas
    - Returns dict instead of CBPSResults object
    - Handles SVD preprocessing and inverse transform
    - More flexible, suitable for advanced users
    
    SVD preprocessing workflow:
    1. Standardize X (except intercept)
    2. SVD decomposition: X = U·D·V'
    3. Use orthogonalized U as new X
    4. Call underlying algorithm (in SVD space)
    5. Inverse transform coefficients and variance back to original space
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbps import cbps_fit
    >>> 
    >>> # Prepare data
    >>> n = 100
    >>> treat = np.array([0, 1] * 50)
    >>> X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    >>> 
    >>> # Call low-level API
    >>> result = cbps_fit(treat, X, method='over', att=1)
    >>> print(result['coefficients'])
    >>> print(result['weights'])
    
    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    https://doi.org/10.1111/rssb.12027
    """
    from cbps.core.cbps_binary import cbps_binary_fit, _r_ginv
    from cbps.core.cbps_multitreat import cbps_3treat_fit, cbps_4treat_fit
    from cbps.core.cbps_continuous import cbps_continuous_fit
    from cbps.core.cbps_optimal import cbps_optimal_2treat
    
    # Step 1: 0/1 binary special handling
    # Numeric 0/1 auto-converted to factor
    is_factor = False
    treat_array = treat
    
    if isinstance(treat, pd.Categorical):
        is_factor = True
        treat_array = treat.codes  # Numeric codes
        treat_categories = treat.categories
    elif hasattr(treat, 'cat'):  # pd.Series with categorical dtype
        is_factor = True
        treat_array = treat.cat.codes.to_numpy()
        treat_categories = treat.cat.categories
    elif isinstance(treat, pd.Series):
        treat_array = treat.to_numpy()
    else:
        treat_array = np.asarray(treat)
    
    # Detect 0/1 binary
    if not is_factor and np.issubdtype(treat_array.dtype, np.number):
        treat_unique = np.unique(treat_array)
        if len(treat_unique) == 2 and set(treat_unique) <= {0, 1, 0.0, 1.0, False, True}:
            # Auto-convert to factor
            treat = pd.Categorical(treat_array)
            is_factor = True
            treat_array = treat.codes
            treat_categories = treat.categories
    
    # Step 2: Method parameter conversion
    bal_only = (method == 'exact')
    
    # Step 3: Variable name handling
    # Column names for X (for result output)
    names_X = [f"X{i}" if i > 0 else "(Intercept)" for i in range(X.shape[1])]
    # Mark zero-variance columns as "(Intercept)"
    x_sd_check = X.std(axis=0, ddof=1)
    for i in range(X.shape[1]):
        if x_sd_check[i] < 1e-10:
            names_X[i] = "(Intercept)"
    
    # Step 4: SVD preprocessing (non-oCBPS path only)
    # oCBPS requires both baseline_X and diff_X; if only one is provided,
    # we'll raise an error later in the routing logic
    X_orig = X.copy()
    svd_info = None
    
    if baseline_X is None and diff_X is None:  # Non-oCBPS path
        # Apply SVD preprocessing
        X_svd, svd_info = _apply_svd_preprocessing(X)
        X_for_algo = X_svd
    else:
        # oCBPS path (or partial - will be validated later): no SVD preprocessing
        X_for_algo = X
    
    # Step 5: Rank check and XprimeX_inv
    k = np.linalg.matrix_rank(X_for_algo)
    if k < X_for_algo.shape[1]:
        raise ValueError("X is not full rank")
    
    # Compute weighted XprimeX_inv
    if sample_weights is None:
        sample_weights = np.ones(len(treat_array))
    
    w_sqrt = np.sqrt(sample_weights)
    X_weighted = w_sqrt[:, None] * X_for_algo
    XprimeX_inv = _r_ginv(X_weighted.T @ X_weighted)
    
    # Step 6: Treatment type detection and routing
    output = None
    
    if is_factor:
        # Discrete treatment path
        no_treats = len(treat_categories)
        
        # Validate treatment count
        if no_treats > 4:
            raise ValueError(
                "Parametric CBPS is not implemented for more than 4 treatment values. "
                "Consider using a continuous treatment."
            )
        if no_treats < 2:
            raise ValueError("Treatment must take more than one value")
        
        # Route to appropriate algorithm
        if no_treats == 2:
            # Binary treatment
            if baseline_X is not None and diff_X is not None:
                # oCBPS path
                if att != 0:
                    warnings.warn(
                        f"CBPSOptimal only supports att=0 (ATE). "
                        f"Received att={att}, forcing to att=0.",
                        UserWarning
                    )
                output = cbps_optimal_2treat(
                    treat=treat_array,
                    X=X_for_algo,  # oCBPS uses original X
                    baseline_X=baseline_X,
                    diff_X=diff_X,
                    iterations=iterations,
                    att=0,  # oCBPS forces att=0 (ATE only)
                    standardize=standardize
                )
            elif baseline_X is not None or diff_X is not None:
                # Only one of baseline_X/diff_X provided - invalid for oCBPS
                raise ValueError(
                    "For oCBPS (optimal CBPS), both baseline_X and diff_X must be provided. "
                    f"Received: baseline_X={'provided' if baseline_X is not None else 'None'}, "
                    f"diff_X={'provided' if diff_X is not None else 'None'}. "
                    "Either provide both for oCBPS, or neither for standard CBPS."
                )
            else:
                # Standard binary CBPS
                output = cbps_binary_fit(
                    treat=treat_array,
                    X=X_for_algo,  # SVD space X
                    att=att,
                    method=method,
                    two_step=two_step,
                    iterations=iterations,
                    standardize=standardize,
                    sample_weights=sample_weights,
                    XprimeX_inv=XprimeX_inv,
                    
                    verbose=verbose
                )
        
        elif no_treats == 3:
            # 3-level treatment
            output = cbps_3treat_fit(
                treat=treat_array,
                X=X_for_algo,
                method=method,
                k=k,
                XprimeX_inv=XprimeX_inv,
                bal_only=bal_only,
                iterations=iterations,
                standardize=standardize,
                two_step=two_step,
                sample_weights=sample_weights,
                treat_levels=treat_categories.to_numpy() if hasattr(treat_categories, 'to_numpy') else np.array(list(treat_categories)),
                verbose=verbose
            )
        
        elif no_treats == 4:
            # 4-level treatment
            output = cbps_4treat_fit(
                treat=treat_array,
                X=X_for_algo,
                method=method,
                k=k,
                XprimeX_inv=XprimeX_inv,
                bal_only=bal_only,
                iterations=iterations,
                standardize=standardize,
                two_step=two_step,
                sample_weights=sample_weights,
                treat_levels=treat_categories.to_numpy() if hasattr(treat_categories, 'to_numpy') else np.array(list(treat_categories)),
                verbose=verbose
            )
    
    elif np.issubdtype(treat_array.dtype, np.number):
        # Continuous treatment path
        # Warn if ≤4 unique values (may be discrete)
        n_unique = len(np.unique(treat_array))
        if n_unique <= 4:
            warnings.warn(
                f"Treatment vector is numeric with {n_unique} unique values. "
                f"Interpreting as a continuous treatment. "
                f"To solve for a binary or multi-valued treatment, make treat a factor.",
                UserWarning
            )
        
        output = cbps_continuous_fit(
            treat=treat_array,
            X=X_for_algo,
            method=method,
            two_step=two_step,
            iterations=iterations,
            standardize=standardize,
            sample_weights=sample_weights,
            verbose=verbose
        )
    
    else:
        raise ValueError("Treatment must be either a factor or numeric")
    
    # Step 7: SVD inverse transform (non-oCBPS path only)
    if svd_info is not None:
        # Inverse transform coefficients
        beta_svd = output['coefficients']
        beta_orig = _apply_svd_inverse_transform(beta_svd, svd_info)
        
        # Update output
        output['coefficients'] = beta_orig
        output['x'] = X_orig  # Replace with original X
        
        # Variance inverse transform
        from cbps.utils.variance_transform import apply_variance_svd_inverse_transform
        
        # Infer treatment type from coefficients shape
        k = X_orig.shape[1]
        coef_shape = beta_orig.shape
        
        # Determine is_factor and no_treats
        # If coefficients is (k, K-1) shape, it's K-level treatment
        if len(coef_shape) == 2 and coef_shape[1] > 1:
            is_factor_inferred = True
            no_treats_inferred = coef_shape[1] + 1  # K-1 cols → K-level treatment
        elif len(coef_shape) == 2 and coef_shape[1] == 1:
            # (k, 1) may be binary or continuous
            is_factor_inferred = is_factor if 'is_factor' in locals() else False
            no_treats_inferred = 2 if is_factor_inferred else None
        else:
            # (k,) shape, may be binary or continuous
            is_factor_inferred = is_factor if 'is_factor' in locals() else False
            no_treats_inferred = 2 if is_factor_inferred else None
        
        variance_svd = output['var']
        variance_orig = apply_variance_svd_inverse_transform(
            variance_svd=variance_svd,
            svd_info=svd_info,
            X_orig=X_orig,
            X_svd=X_for_algo,
            is_factor=is_factor_inferred,
            no_treats=no_treats_inferred
        )
        output['var'] = variance_orig
        
        if verbose > 0:
            print(f"cbps_fit: SVD inverse transform done, coef shape={beta_orig.shape}, var shape={variance_orig.shape}")
    
    # Add method field
    output['method'] = method
    
    return output


def cbmsm_fit(
    treat: np.ndarray,
    X: np.ndarray,
    id: np.ndarray,
    time: np.ndarray,
    type: str = "MSM",
    twostep: bool = True,
    msm_variance: str = "approx",
    time_vary: bool = False,
    init: str = "opt",
    sample_weights: Optional[np.ndarray] = None,
    iterations: Optional[int] = None,
    **kwargs: Any
) -> 'CBMSMResults':
    """
    CBMSM Matrix Interface (Low-Level Fitting Function)
    
    This is the low-level matrix interface for CBMSM, accepting preprocessed
    matrix inputs. For most users, the formula interface CBMSM() is recommended.
    
    Parameters
    ----------
    treat : np.ndarray, shape (N*T,)
        Treatment vector for N units over T periods.
    X : np.ndarray, shape (N*T, p)
        Covariate matrix (including intercept column).
    id : np.ndarray, shape (N*T,)
        Unit identifiers.
    time : np.ndarray, shape (N*T,)
        Time period identifiers.
    type : str, default="MSM"
        Weight type ('MSM' or 'MultiBin').
    twostep : bool, default=True
        Whether to use two-step estimation.
    msm_variance : str, default="approx"
        Variance estimation method ('approx' or 'full').
    time_vary : bool, default=False
        Whether coefficients vary with time.
    init : str, default="opt"
        Initialization method ('opt', 'glm', 'CBPS').
    sample_weights : np.ndarray, optional
        Observation weights.
    iterations : int, optional
        Maximum iterations.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    CBMSMResults
        CBMSM fitting result object.
    
    See Also
    --------
    CBMSM : Formula interface (recommended)
    
    Examples
    --------
    >>> from cbps import cbmsm_fit
    >>> import numpy as np
    >>> # Prepare matrix data
    >>> treat = np.array([0, 1, 0, 1, 0, 1])
    >>> X = np.column_stack([np.ones(6), np.random.randn(6, 2)])
    >>> id_vec = np.array([1, 2, 3, 1, 2, 3])
    >>> time_vec = np.array([1, 1, 1, 2, 2, 2])
    >>> result = cbmsm_fit(treat, X, id_vec, time_vec)
    """
    from cbps.msm.cbmsm import cbmsm_fit as _cbmsm_fit
    return _cbmsm_fit(
        treat=treat, X=X, id=id, time=time,
        type=type, twostep=twostep, msm_variance=msm_variance,
        time_vary=time_vary, init=init, sample_weights=sample_weights,
        iterations=iterations, **kwargs
    )


def CBMSM(
    formula: str,
    id: Union[str, pd.Series, np.ndarray],
    time: Union[str, pd.Series, np.ndarray],
    data: pd.DataFrame,
    type: str = "MSM",
    twostep: bool = True,
    msm_variance: str = "approx",
    time_vary: bool = False,
    init: str = "opt",
    iterations: Optional[int] = None,
    **kwargs: Any
) -> 'CBMSMResults':
    """
    Covariate Balancing Propensity Score for Marginal Structural Models.

    Estimates inverse probability of treatment weights for longitudinal data
    with time-varying treatments and confounders. Designed for panel data where
    treatment effects unfold over multiple time periods.

    Parameters
    ----------
    formula : str
        Treatment model formula (e.g., 'treat ~ x1 + x2 + x3').
        The same covariates are used for all time periods. Data should be
        sorted by time within each unit.
    id : str or array-like
        Unit identifier column name (str) or ID array identifying individuals
        in the panel data.
    time : str or array-like
        Time column name (str) or time array identifying the temporal ordering
        of observations.
    data : pd.DataFrame
        DataFrame containing treatment, covariates, ID, and time variables.
    type : {'MSM', 'MultiBin'}, default='MSM'
        Weight type:
        - 'MSM': Marginal structural model weights (default)
        - 'MultiBin': Multiple binary treatment weights
    twostep : bool, default=True
        Whether to use two-step estimation (faster with MLE initialization).
        - True: Estimate parameters for each period separately, then combine
        - False: Estimate all parameters simultaneously (single-step)
    msm_variance : {'approx', 'full', None}, default='approx'
        Variance estimation method:
        - 'approx': Approximate variance (fast, recommended)
        - 'full': Full sandwich variance (accurate but slower)
        - None: Do not compute variance
    time_vary : bool, default=False
        Whether treatment model coefficients vary across time:
        - False: Time-invariant model (shared coefficients across periods)
        - True: Time-varying model (independent coefficients per period)
    init : {'opt', 'glm'}, default='opt'
        Initialization method:
        - 'opt': Use both CBPS and GLM starting values, select best balance
        - 'glm': Use only GLM starting values
    iterations : int, optional
        Maximum number of optimization iterations.
    **kwargs
        Additional parameters passed to the underlying implementation.

    Returns
    -------
    CBMSMResults
        CBMSM fitted result object containing:
        - weights: MSM weight array (unit-level)
        - fitted_values: Propensity scores for each period
        - converged: Convergence status
        - coefficients: Estimated model coefficients
        
    Examples
    --------
    Estimate MSM weights using panel data:
    
    >>> from cbps import CBMSM
    >>> from cbps.datasets import load_blackwell
    >>> data = load_blackwell()
    >>> fit = CBMSM('d.gone.neg ~ d.gone.neg.l1 + camp.length',
    ...             id='demName', time='time', data=data, type='MSM')
    >>> print(f"Weights shape: {fit.weights.shape}")
    
    Notes
    -----
    **Data Requirements**: Must be a balanced panel where each id appears
    exactly once at each time period.
    
    References
    ----------
    Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability 
    Weights for Marginal Structural Models. Journal of the American Statistical 
    Association, 110(511), 1013-1023. https://doi.org/10.1080/01621459.2014.956872
    
    See Also
    --------
    CBPS : Covariate balancing propensity score for cross-sectional data
    """
    from cbps.msm.cbmsm import CBMSM as _CBMSM
    # Handle two_step alias
    if 'two_step' in kwargs and twostep is True:
        twostep = kwargs.pop('two_step')
        
    return _CBMSM(
        formula=formula, id=id, time=time, data=data,
        type=type, twostep=twostep, msm_variance=msm_variance,
        time_vary=time_vary, init=init, iterations=iterations,
        **kwargs
    )


def npCBPS(
    formula: str,
    data: pd.DataFrame,
    na_action: Optional[str] = None,
    corprior: Optional[float] = None,
    print_level: int = 0,
    seed: Optional[int] = None,
    verbose: int = 0,
    **kwargs: Any
) -> 'NPCBPSResults':
    """
    Nonparametric Covariate Balancing Propensity Score.

    Estimates weights directly using the empirical likelihood framework,
    without requiring a parametric propensity score model specification.

    Parameters
    ----------
    formula : str
        Model formula specifying treatment and covariates (e.g., 'treat ~ age + educ').
    data : pd.DataFrame
        DataFrame containing the treatment and covariate variables.
    corprior : float, default=None
        Prior standard deviation σ controlling the weighted correlation between
        covariates and treatment, where η ~ N(0, σ²I).
        Note: corprior is the standard deviation σ, not the variance σ².
        
        Default (None): Automatically set to 0.1/n (sample-size adaptive).
          - Small sample (n=10): corprior ≈ 0.01
          - Medium sample (n=100): corprior ≈ 0.001
          - Large sample (n=1000): corprior ≈ 0.0001
        
        Reference: Fong, Hazlett & Imai (2018) Section 3.3.4
    print_level : int, default=0
        Diagnostic output verbosity level.
    seed : int, optional
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level for progress messages.
    **kwargs : Any
        Additional parameters passed to the underlying optimization routine.

    Returns
    -------
    NPCBPSResults
        Fitted result object containing:
        - weights: Estimated empirical likelihood weights
        - eta: Weighted correlations (balance diagnostics)
        - sumw0: Sum of weights (should be ≈ 1, tolerance ±5%)
        - log_el, log_p_eta: Log empirical likelihood and prior density

    Notes
    -----
    The empirical likelihood optimization is non-convex, which may lead to
    different local optima across implementations. Convergence quality should
    be verified by checking that sumw0 ≈ 1.0 (within 5% tolerance).

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing
    Propensity Score for a Continuous Treatment. The Annals of Applied
    Statistics 12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

    Examples
    --------
    >>> from cbps import npCBPS
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>> fit = npCBPS('treat ~ age + educ', data=df, corprior=0.01)
    >>> # Verify convergence
    >>> assert abs(fit.sumw0 - 1.0) < 0.05, "Weight sum should be close to 1"
    """
    from cbps.nonparametric.npcbps import npCBPS as _npCBPS, npCBPS_fit
    # verbose parameter is accepted for API consistency but not passed to underlying function
    # The underlying npCBPS_fit uses print_level to control output
    _ = verbose  # Mark parameter as processed to avoid linter warnings
    return _npCBPS(
        formula=formula, data=data, na_action=na_action,
        corprior=corprior, print_level=print_level, seed=seed,
        **kwargs
    )


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
) -> 'HDCBPSResults':
    """
    High-Dimensional Covariate Balancing Propensity Score estimation.

    Implements covariate balancing propensity score methodology for high-dimensional
    settings where the number of covariates substantially exceeds the sample
    size (d >> n). The approach combines LASSO variable selection with covariate
    balancing constraints to achieve valid causal effect estimation.

    Parameters
    ----------
    formula : str
        Model formula specifying treatment and covariates.
        Example: 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
    data : pd.DataFrame
        Dataset containing all variables specified in the formula.
    y : str or np.ndarray
        Outcome variable name or array. Used for variable selection in the
        high-dimensional framework.
    ATT : int, default 0
        Target estimand: 0 for ATE (average treatment effect), 1 for ATT
        (average treatment effect on the treated).
    iterations : int, default 1000
        Maximum number of iterations for the optimization algorithm.
    method : {'linear', 'binomial', 'poisson'}, default 'linear'
        Type of outcome model for variable selection:
        - 'linear': Linear regression model
        - 'binomial': Logistic regression model
        - 'poisson': Poisson regression model
    seed : int, optional
        Random seed for reproducibility. Note: Current implementation uses
        deterministic LASSO, so this parameter does not affect results.
    na_action : {None, 'warn', 'drop', 'fail'}, optional
        How to handle missing values:
        - None or 'warn': Remove missing observations with warning
        - 'drop': Remove missing observations silently
        - 'fail': Raise an error for missing values
    verbose : int, default 0
        Verbosity level for output:
        - 0: Silent mode
        - 1: Basic iteration information
        - 2: Detailed debugging information

    Returns
    -------
    HDCBPSResults
        Result object containing:
        - ATE: Estimated average treatment effect
        - ATT: Estimated average treatment effect on the treated
        - s: Selected variables
        - fitted_values: Estimated propensity scores
        - coefficients0: LASSO coefficients for control group (T=0)
        - coefficients1: LASSO coefficients for treatment group (T=1)
        - coefficients: Alias for coefficients0 (for API consistency)

    Notes
    -----
    The high-dimensional CBPS methodology extends the original CBPS approach
    to settings with many covariates by incorporating variable selection. The
    algorithm selects a subset of covariates that are predictive of both the
    treatment and outcome while maintaining covariate balance.

    Unlike standard CBPS which has one set of coefficients, hdCBPS estimates
    two LASSO models (one for each treatment level) to achieve variable
    selection in the high-dimensional setting.

    References
    ----------
    Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
    via a high-dimensional covariate balancing propensity score. Biometrika
    107(3), 533-554. https://doi.org/10.1093/biomet/asaa020

    Examples
    --------
    >>> from cbps import hdCBPS
    >>> from cbps.datasets import load_lalonde
    >>> # Load high-dimensional data
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>>
    >>> # Fit high-dimensional CBPS
    >>> result = hdCBPS(
    ...     formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    ...     data=df,
    ...     y='re78',  # Outcome variable
    ...     ATT=0,  # Estimate ATE
    ...     method='linear'
    ... )
    >>>
    >>> # View results
    >>> print(f"ATE: {result.ATE:.4f}")
    >>> print(f"Selected variables: {len(result.s)}")
    >>> print(f"Converged: {result.converged}")
    """
    from cbps.highdim.hdcbps import hdCBPS as _hdCBPS
    return _hdCBPS(formula, data, y, ATT, iterations, method, seed, na_action, verbose)


def CBIV(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    Tr: Optional[np.ndarray] = None,
    Z: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    iterations: int = 1000,
    method: str = "over",
    twostep: bool = True,
    twosided: bool = True,
    probs_min: float = 1e-6,
    warn_clipping: bool = True,
    clipping_warn_threshold: float = 0.05,
    verbose: int = 0,
    **kwargs: Any
) -> 'CBIVResults':
    """
    Covariate Balancing Propensity Score for Instrumental Variables.

    Estimates propensity scores for compliers in instrumental variable settings
    with treatment noncompliance. This method is designed for encouragement
    designs where randomized assignment (instrument) affects treatment uptake
    but may not guarantee compliance.

    Parameters
    ----------
    formula : str, optional
        IV formula in the format "treatment ~ covariates | instrument".
        Example: "treat ~ x1 + x2 | z". Intercept is added automatically.
    data : pd.DataFrame, optional
        DataFrame containing the variables specified in formula.
        Required when using formula interface.
    Tr : np.ndarray, shape (n,), optional
        Binary treatment variable (0/1). Required for matrix interface.
    Z : np.ndarray, shape (n,), optional
        Binary instrument variable (0/1). Required for matrix interface.
    X : np.ndarray, shape (n, p), optional
        Pre-treatment covariate matrix (without intercept). Required for
        matrix interface.
    iterations : int, default=1000
        Maximum number of optimization iterations.
    method : str, default="over"
        Estimation method:

        - 'over': Over-identified GMM (propensity score + balance conditions)
        - 'exact': Just-identified GMM (balance conditions only)
        - 'mle': Maximum likelihood estimation (propensity score only)
    twostep : bool, default=True
        Whether to use two-step GMM estimation. If False, uses continuously
        updating GMM which has better finite-sample properties but is slower.
    twosided : bool, default=True
        Whether to allow two-sided noncompliance:

        - True: Allows compliers, always-takers, and never-takers
        - False: One-sided noncompliance (compliers and never-takers only)
    probs_min : float, default=1e-6
        Probability clipping bound. Compliance probabilities are constrained
        to the interval [probs_min, 1-probs_min].
    warn_clipping : bool, default=True
        Whether to issue a warning when the proportion of clipped compliance
        probabilities exceeds the threshold.
    clipping_warn_threshold : float, default=0.05
        Minimum clipping proportion (between 0 and 1) that triggers a warning.
    verbose : int, default=0
        Verbosity level. 0=silent, 1=basic info, 2=detailed diagnostics.

    Returns
    -------
    CBIVResults
        Result object containing coefficients, fitted values, weights, and
        diagnostic statistics.

    Notes
    -----
    The method implements principal stratification with three compliance types:

    - **Compliers**: Units who take treatment when encouraged (Z=1) and do not
      take treatment when not encouraged (Z=0)
    - **Always-takers**: Units who always take treatment regardless of Z
    - **Never-takers**: Units who never take treatment regardless of Z

    The Complier Average Causal Effect (CACE) is identified under standard IV
    assumptions (exclusion restriction, monotonicity, non-zero first stage).

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society: Series B, 76(1), 243-263.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cbps import CBIV
    >>> # Formula interface
    >>> df = pd.DataFrame({
    ...     'treat': np.random.binomial(1, 0.5, 100),
    ...     'z': np.random.binomial(1, 0.5, 100),
    ...     'x1': np.random.randn(100),
    ...     'x2': np.random.randn(100)
    ... })
    >>> fit = CBIV(formula="treat ~ x1 + x2 | z", data=df)
    >>> print(fit.coefficients.shape)
    >>>
    >>> # Matrix interface
    >>> Tr = np.random.binomial(1, 0.5, 100)
    >>> Z = np.random.binomial(1, 0.5, 100)
    >>> X = np.random.randn(100, 2)
    >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=True)
    >>> print(fit.fitted_values.shape)
    """
    from cbps.iv.cbiv import CBIV as _CBIV
    return _CBIV(
        formula=formula, data=data, Tr=Tr, Z=Z, X=X,
        iterations=iterations, method=method, twostep=twostep,
        twosided=twosided, probs_min=probs_min, warn_clipping=warn_clipping,
        clipping_warn_threshold=clipping_warn_threshold, verbose=verbose,
        **kwargs
    )


def AsyVar(
    Y: np.ndarray,
    Y_1_hat: Optional[np.ndarray] = None,
    Y_0_hat: Optional[np.ndarray] = None,
    CBPS_obj: Optional[Union[Dict[str, Any], 'CBPSResults']] = None,
    method: str = "CBPS",
    X: Optional[np.ndarray] = None,
    TL: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    CI: float = 0.95,
    use_observed_y: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Asymptotic Variance and Confidence Intervals for ATE.

    Estimates the asymptotic variance of the average treatment effect obtained
    using CBPS or optimal CBPS (oCBPS) methods. This function computes valid
    confidence intervals that properly account for the uncertainty in propensity
    score estimation.

    Parameters
    ----------
    Y : np.ndarray
        Observed outcome values.
    Y_1_hat : np.ndarray, optional
        Predicted outcomes under treatment. If None, will be automatically fitted.
    Y_0_hat : np.ndarray, optional
        Predicted outcomes under control. If None, will be automatically fitted.
    CBPS_obj : dict or CBPSResults, optional
        Fitted CBPS object. Required for the CBPS variance estimation path.
    method : str, default="CBPS"
        Variance estimation method: 'CBPS' (standard) or 'oCBPS' (optimal).
    X : np.ndarray, optional
        Covariate matrix (first column must be intercept).
    TL : np.ndarray, optional
        Treatment indicator variable (1=treated, 0=control).
    pi : np.ndarray, optional
        Propensity score vector.
    mu : float, optional
        Average treatment effect estimate.
    CI : float, default=0.95
        Confidence level for the confidence interval.
    use_observed_y : bool, default=False
        Sigma_mu computation method:
        
        - False (default): Use predicted values Y_1_hat, Y_0_hat.
          This matches R CBPS package behavior and is recommended.
        - True: Use observed Y values. This is an experimental option
          not implemented in the R package.

    Returns
    -------
    dict
        Dictionary containing (snake_case keys are preferred):

        - 'mu_hat' (or 'mu.hat'): ATE estimate
        - 'asy_var' (or 'asy.var'): Asymptotic variance of sqrt(n) * (mu_hat - mu)
        - 'var': Finite-sample variance = asy_var / n
        - 'std_err' (or 'std.err'): Standard error = sqrt(var)
        - 'ci_mu_hat' (or 'CI.mu.hat'): Confidence interval [lower, upper]

        R-style dot-separated keys (e.g., 'mu.hat') are retained as
        backward-compatible aliases and point to the same value objects.

    References
    ----------
    Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
    Optimal covariate balancing conditions in propensity score estimation.
    Journal of Business & Economic Statistics, 41(1), 97-110.
    https://doi.org/10.1080/07350015.2021.2002159

    Examples
    --------
    >>> from cbps import CBPS, AsyVar
    >>> from cbps.datasets import load_lalonde
    >>> data = load_lalonde()
    >>> fit = CBPS('treat ~ age + educ + black + hisp', data=data, att=0)
    >>> result = AsyVar(Y=data['re78'].values, CBPS_obj=fit, method="oCBPS")
    >>> print(f"ATE: {result['mu.hat']:.3f} (SE: {result['std.err']:.3f})")
    """
    from cbps.inference.asyvar import asy_var

    # Check for CBPS_obj in kwargs for backward compatibility
    if CBPS_obj is None and 'CBPS_obj' in kwargs:
        CBPS_obj = kwargs['CBPS_obj']

    # Convert CBPSResults object to dict format if necessary
    if CBPS_obj is not None and hasattr(CBPS_obj, 'fitted_values'):
        cbps_dict = {
            'x': CBPS_obj.x,
            'y': CBPS_obj.y,
            'fitted_values': CBPS_obj.fitted_values,
            'coefficients': CBPS_obj.coefficients
        }
        # Include residuals if available
        if hasattr(CBPS_obj, 'residuals'):
            cbps_dict['residuals'] = CBPS_obj.residuals
        CBPS_obj = cbps_dict

    result = asy_var(
        Y=Y, Y_1_hat=Y_1_hat, Y_0_hat=Y_0_hat, CBPS_obj=CBPS_obj,
        method=method, X=X, TL=TL, pi=pi, mu=mu, CI=CI,
        use_observed_y=use_observed_y, **kwargs
    )

    # Add snake_case aliases (retain original R-style keys)
    key_mapping = {
        'mu.hat': 'mu_hat',
        'asy.var': 'asy_var',
        'CI.mu.hat': 'ci_mu_hat',
        'std.err': 'std_err',
    }
    for old_key, new_key in key_mapping.items():
        if old_key in result:
            result[new_key] = result[old_key]

    return result


def balance(cbps_obj, enhanced: bool = False, threshold: float = 0.1,
            covariate_names: Optional[list] = None, *args: Any, **kwargs: Any):
    """
    Assess covariate balance before and after CBPS weighting.

    Computes balance statistics to evaluate the effectiveness of propensity score
    estimation in achieving covariate balance between treatment groups. This
    is a fundamental diagnostic tool for causal inference analyses.

    Parameters
    ----------
    cbps_obj : dict or CBPSResults or NPCBPSResults
        Fitted CBPS object containing the estimation results. Must include:
        - weights: final CBPS weights
        - x: covariate matrix
        - y: treatment variable
        Supports CBPS, CBPSContinuous, and npCBPS objects.
    enhanced : bool, default False
        If False, returns basic balance statistics format.
        If True, returns enhanced diagnostics including:
        - Improvement percentages
        - Summary statistics
        - Text-based diagnostic report
    threshold : float, default 0.1
        Threshold for determining covariate imbalance (used when enhanced=True).
        Standard threshold: SMD < 0.1 indicates excellent balance (Stuart 2010).
    covariate_names : list, optional
        List of covariate names for generating detailed reports. Used when enhanced=True.

    Returns
    -------
    dict
        If enhanced=False (default):
        - balanced: balance statistics after weighting
        - original/unweighted: baseline unweighted statistics

        If enhanced=True (enhanced diagnostics):
        Contains above keys plus:
        - smd_weighted/abs_corr_weighted: weighted SMDs or correlations
        - smd_unweighted/abs_corr_unweighted: unweighted SMDs or correlations
        - improvement_pct: percentage improvement in balance
        - n_imbalanced_before/after: number of imbalanced covariates
        - summary: dictionary with summary statistics
        - report: text-based diagnostic report

    Notes
    -----
    **Balance Metrics:**
    - Binary/multi-valued treatments: Standardized mean differences (SMDs)
    - Continuous treatments: Absolute Pearson correlations
    - For npCBPS, routes to appropriate function based on treatment type

    **Interpretation Guidelines:**
    - SMD < 0.1: Excellent balance
    - SMD 0.1-0.25: Moderate imbalance
    - SMD > 0.25: Severe imbalance
    - For correlations: closer to 0 indicates better balance

    The enhanced diagnostic mode provides comprehensive assessment following
    best practices in the causal inference literature.

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    https://doi.org/10.1111/rssb.12027

    Stuart, E.A. (2010). "Matching methods for causal inference: A review and
    a look forward." Statistical Science 25(1), 1-21.

    Austin, P.C. (2009). "Some methods of propensity-score matching resulted
    in substantial bias in examining the effects of medical interventions."
    Statistics in Medicine 28(25), 3083-3107.

    Examples
    --------
    >>> import cbps
    >>> # Fit CBPS model
    >>> fit = cbps.CBPS('treat ~ age + education + income', data=df)
    >>>
    >>> # Basic balance assessment (R-compatible)
    >>> bal = cbps.balance(fit)
    >>> print("Balance after weighting:", bal['balanced'])
    >>> print("Balance before weighting:", bal['original'])
    >>>
    >>> # Enhanced diagnostics with detailed report
    >>> bal_enh = cbps.balance(fit, enhanced=True, threshold=0.1)
    >>> print(bal_enh['report'])
    >>> print(f"Mean SMD after: {bal_enh['summary']['mean_smd_after']:.3f}")
    >>> print(f"Imbalanced covariates: {bal_enh['n_imbalanced_after']}")
    """
    from cbps.diagnostics.balance import (
        balance_cbps, balance_cbps_continuous,
        balance_cbps_enhanced, balance_cbps_continuous_enhanced
    )
    from cbps.nonparametric.npcbps import NPCBPSResults

    # Extract covariate names for DataFrame labeling
    # Skip intercept column
    coef_names_for_balance = None
    if isinstance(cbps_obj, CBPSResults):
        if hasattr(cbps_obj, 'coef_names') and cbps_obj.coef_names is not None:
            # Skip intercept column
            coef_names_for_balance = [name for name in cbps_obj.coef_names if name not in ['(Intercept)', 'Intercept']]
    elif isinstance(cbps_obj, NPCBPSResults):
        # Extract covariate names from NPCBPSResults.terms (patsy DesignInfo)
        if hasattr(cbps_obj, 'terms') and cbps_obj.terms is not None:
            try:
                coef_names_for_balance = [name for name in cbps_obj.terms.column_names
                                          if name not in ['Intercept', '(Intercept)']]
            except AttributeError:
                pass
    
    # Detect object type and route to appropriate function
    if isinstance(cbps_obj, CBPSResults):
        # Convert to dict format
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'fitted_values': cbps_obj.fitted_values
        }
    elif isinstance(cbps_obj, NPCBPSResults):
        # npCBPS result object
        # Route to appropriate balance function based on treatment type
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'log_el': cbps_obj.log_el,  # Include log_el to identify npCBPS
        }
        # Detect continuous treatment
        # Handle CategoricalDtype separately (always discrete)
        y_array = cbps_obj.y
        is_categorical = hasattr(y_array, 'dtype') and hasattr(y_array.dtype, 'name') and 'category' in str(y_array.dtype).lower()
        is_continuous = False
        if not is_categorical:
            try:
                is_continuous = np.issubdtype(y_array.dtype, np.number) and len(np.unique(y_array)) > 4
            except TypeError:
                # If dtype check fails, treat as discrete
                is_continuous = False
        
        if is_continuous:
            # Continuous treatment path
            if enhanced:
                result = balance_cbps_continuous_enhanced(cbps_dict, threshold, covariate_names)
            else:
                result = balance_cbps_continuous(cbps_dict, *args, **kwargs)
            # Add row/column labels
            return _add_balance_labels(result, cbps_dict, coef_names_for_balance, is_continuous=True)
        else:
            # Discrete treatment path
            if enhanced:
                result = balance_cbps_enhanced(cbps_dict, threshold, covariate_names)
            else:
                result = balance_cbps(cbps_dict, *args, **kwargs)
            # Add row/column labels
            return _add_balance_labels(result, cbps_dict, coef_names_for_balance, is_continuous=False)
    elif hasattr(cbps_obj, '__class__') and cbps_obj.__class__.__name__ == 'CBMSMResults':
        # CBMSM result object support
        from cbps.diagnostics.balance_cbmsm_addon import balance_cbmsm
        
        # Convert to dict format
        cbmsm_dict = {
            'y': cbps_obj.y,
            'x': cbps_obj.x,
            'weights': cbps_obj.weights,
            'glm_weights': cbps_obj.glm_weights,
            'id': cbps_obj.id,
            'time': cbps_obj.time
        }
        
        # Call CBMSM-specific balance function
        result = balance_cbmsm(cbmsm_dict)
        
        # Note: CBMSM return format differs from CBPS (includes StatBal)
        return result
    else:
        cbps_dict = cbps_obj

    # Detect continuous treatment (via fitted_values dimension)
    # Continuous: fitted_values is 1D array
    # Discrete: fitted_values is 2D array or scalar
    if 'fitted_values' in cbps_dict:
        fv = cbps_dict['fitted_values']
        if isinstance(fv, np.ndarray) and fv.ndim == 1 and len(np.unique(cbps_dict['y'])) > 4:
            # Continuous treatment path
            if enhanced:
                result = balance_cbps_continuous_enhanced(cbps_dict, threshold, covariate_names)
            else:
                result = balance_cbps_continuous(cbps_dict, *args, **kwargs)
            # Add row/column labels
            return _add_balance_labels(result, cbps_dict, coef_names_for_balance, is_continuous=True)

    # Default: discrete treatment path
    if enhanced:
        result = balance_cbps_enhanced(cbps_dict, threshold, covariate_names)
    else:
        result = balance_cbps(cbps_dict, *args, **kwargs)
    # Add row/column labels
    return _add_balance_labels(result, cbps_dict, coef_names_for_balance, is_continuous=False)


# Import vcov_outcome
from cbps.inference.vcov_outcome import vcov_outcome

# Import plot functions
from cbps.diagnostics.plots import plot_cbps, plot_cbps_continuous, plot_cbmsm, plot_npcbps

# Import npCBPS_fit low-level interface
from cbps.nonparametric.npcbps import npCBPS_fit
