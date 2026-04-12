"""
Input Validation Utilities

This module provides centralized input validation for CBPS estimators,
ensuring consistent error handling and user-friendly error messages
across all model classes.

The validation functions check for common issues such as:

- Empty or insufficient sample sizes
- Dimension mismatches between treatment and covariates
- Missing or infinite values
- Zero-variance treatment variables
- Improperly shaped covariate matrices

All validation errors include descriptive messages with the calling
module name, making it easy to identify the source of issues.

Functions
---------
validate_cbps_input
    Comprehensive input validation for CBPS estimators.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import numpy as np
from typing import Optional


def validate_cbps_input(
    treat: np.ndarray, 
    X: np.ndarray, 
    min_observations: int = 2,
    module_name: str = "CBPS",
    check_treatment_variance: bool = True
) -> None:
    """
    Validate treatment and covariate arrays for CBPS estimation.
    
    Performs comprehensive validation of input arrays before CBPS fitting,
    providing informative error messages that identify the specific issue
    and suggest remediation steps.
    
    Parameters
    ----------
    treat : np.ndarray
        Treatment variable, shape (n,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    min_observations : int, default=2
        Minimum required sample size.
    module_name : str, default="CBPS"
        Name of calling module for error message prefixes.
    check_treatment_variance : bool, default=True
        If True, verify treatment has non-zero variance.
        Set to False for binary treatments where variance check
        is handled separately.
    
    Raises
    ------
    ValueError
        If any validation check fails. The error message includes:
        
        - The module name prefix for easy identification
        - A description of the specific issue
        - The actual values that caused the error
        - Suggested remediation steps
    
    Notes
    -----
    The following checks are performed in order:
    
    1. Treatment array is non-empty
    2. Covariate matrix is 2-dimensional
    3. Sample size >= min_observations
    4. Treatment and covariate row counts match
    5. Covariate matrix has >= 1 column
    6. No NaN or Inf values in treatment
    7. No NaN or Inf values in covariates
    8. Treatment has non-zero variance (if check_treatment_variance=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> from cbps.utils.validation import validate_cbps_input
    >>> 
    >>> # Valid input passes silently
    >>> treat = np.array([0, 1, 0, 1])
    >>> X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
    >>> validate_cbps_input(treat, X)
    >>> 
    >>> # Dimension mismatch raises informative error
    >>> try:
    ...     validate_cbps_input(np.array([0, 1]), X)
    ... except ValueError as e:
    ...     print("Validation failed")
    Validation failed
    """
    # Check 1: Empty array (highest priority to avoid len() errors)
    if not isinstance(treat, np.ndarray):
        treat = np.asarray(treat)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    
    n_treat = len(treat)
    
    if n_treat == 0:
        raise ValueError(
            f"{module_name}: Treatment array is empty (n=0). "
            f"At least {min_observations} observation(s) are required to estimate "
            f"the propensity score."
        )
    
    # Check 2: Covariate matrix dimensions (check before accessing shape[0])
    if X.ndim == 0:
        raise ValueError(
            f"{module_name}: Covariate input is a scalar. "
            f"Expected a 2-dimensional array with shape (n_observations, n_covariates)."
        )
    
    if X.ndim == 1:
        raise ValueError(
            f"{module_name}: Covariate matrix X is 1-dimensional with shape {X.shape}. "
            f"Expected a 2-dimensional array with shape (n_observations, n_covariates). "
            f"If you have a single covariate, use X.reshape(-1, 1)."
        )
    
    if X.ndim > 2:
        raise ValueError(
            f"{module_name}: Covariate matrix X has {X.ndim} dimensions with shape {X.shape}. "
            f"Expected a 2-dimensional array with shape (n_observations, n_covariates)."
        )
    
    # Now safe to access X.shape[0]
    n_X = X.shape[0]
    
    # Check 3: Insufficient sample size
    if n_treat < min_observations:
        raise ValueError(
            f"{module_name}: Treatment array has only {n_treat} observation(s). "
            f"At least {min_observations} observations are required to estimate "
            f"the propensity score and its variance."
        )
    
    # Check 4: Sample size mismatch
    if n_X != n_treat:
        raise ValueError(
            f"{module_name}: Sample size mismatch between treatment and covariates. "
            f"Treatment has {n_treat} observations, but covariates have {n_X} rows. "
            f"Both arrays must have the same number of observations."
        )
    
    # Check 5: Covariate column count
    if X.shape[1] == 0:
        raise ValueError(
            f"{module_name}: Covariate matrix has 0 columns. "
            f"At least 1 column (e.g., intercept) is required."
        )
    
    # Check 6: NaN/Inf in treatment variable
    if np.any(~np.isfinite(treat)):
        n_nan = np.sum(np.isnan(treat))
        n_inf = np.sum(np.isinf(treat))
        raise ValueError(
            f"{module_name}: Treatment contains {n_nan} NaN value(s) and {n_inf} Inf value(s). "
            f"Please remove or impute missing/infinite values before calling CBPS. "
            f"Consider using data.dropna() or df[df.isfinite().all(axis=1)]."
        )
    
    # Check 7: NaN/Inf in covariates
    if np.any(~np.isfinite(X)):
        n_nan = np.sum(np.isnan(X))
        n_inf = np.sum(np.isinf(X))
        raise ValueError(
            f"{module_name}: Covariates contain {n_nan} NaN value(s) and {n_inf} Inf value(s). "
            f"Please remove or impute missing/infinite values before calling CBPS. "
            f"Consider using data.dropna() or df[df.isfinite().all(axis=1)]."
        )
    
    # Check 8: Treatment variance (only for continuous treatments)
    if check_treatment_variance:
        treat_std = np.std(treat, ddof=1)
        if treat_std == 0 or not np.isfinite(treat_std):
            treat_unique = np.unique(treat)
            raise ValueError(
                f"{module_name}: Treatment variable has zero variance (all values are identical). "
                f"Found only 1 unique value: {treat_unique}. "
                f"CBPS requires variation in the treatment to estimate propensity scores. "
                f"Please check your data or consider using a different treatment definition."
            )

