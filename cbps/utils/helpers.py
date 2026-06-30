"""
Data Preprocessing and Validation Utilities

This module provides helper functions for data preprocessing tasks
commonly needed before CBPS estimation, including sample weight
normalization, input validation, missing value handling, and
treatment variable encoding.

Functions
---------
normalize_sample_weights
    Normalize sampling weights to sum to the sample size.
validate_arrays
    Validate treatment and covariate array dimensions and types.
handle_missing
    Remove observations with missing values.
encode_treatment_factor
    Convert categorical treatment to numeric encoding.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def normalize_sample_weights(
    sample_weights: Optional[np.ndarray],
    n: int
) -> np.ndarray:
    """
    Normalize sampling weights to sum to the sample size.

    Applies the transformation ``sw = sw / mean(sw)`` to ensure
    ``sum(sw) = n``, which is required for proper weighted estimation.

    Parameters
    ----------
    sample_weights : np.ndarray or None
        Original sampling weights, shape (n,).
        If None, returns uniform weights (all ones).
    n : int
        Number of observations (target sum).

    Returns
    -------
    np.ndarray
        Normalized weights satisfying ``sum(weights) = n``, dtype=float64.

    Raises
    ------
    ValueError
        If all weights are zero, any weights are negative, or
        normalization produces unexpected results.

    Warns
    -----
    UserWarning
        If any weights are exactly zero (valid but noteworthy).

    Notes
    -----
    Zero weights are permitted for trimmed or survey designs, but a
    warning is issued since those observations are effectively excluded.

    Examples
    --------
    >>> import numpy as np
    >>> sw = np.array([0.5, 1.0, 1.5, 2.0])
    >>> sw_norm = normalize_sample_weights(sw, n=4)
    >>> bool(np.isclose(sw_norm.sum(), 4.0))
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Step 1: Handle None case
    if sample_weights is None:
        return np.ones(n, dtype=np.float64)
    
    # Step 2: Convert to float64 array
    sw = np.asarray(sample_weights, dtype=np.float64)
    
    # Check for all-zero weights and provide informative error
    if not np.any(sw > 0):  # All weights <= 0
        if np.all(sw == 0):
            raise ValueError(
                f"sample_weights cannot be all zeros. "
                f"Received weights with sum={sw.sum():.1f} for n={n} observations."
            )
        else:
            raise ValueError(
                f"sample_weights must contain at least one positive value. "
                f"Received weights with sum={sw.sum():.6f} (all non-positive)."
            )
    
    # Step 3: Check for negative weights (not allowed)
    if np.any(sw < 0):
        raise ValueError(
            f"sample_weights must be non-negative (>= 0). "
            f"Found {(sw < 0).sum()} negative weights."
        )
    
    # Allow zero weights with warning (valid for survey designs and trimmed weights)
    if np.any(sw == 0):
        n_zeros = (sw == 0).sum()
        warnings.warn(
            f"sample_weights contains {n_zeros} zero values. "
            f"These observations will be effectively excluded from the analysis. "
            f"This is valid for trimmed weights or survey designs, but verify this is intentional.",
            UserWarning,
            stacklevel=3
        )
    
    # Step 4: Normalize by dividing by mean
    sw = sw / sw.mean()
    
    # Step 5: Verify sum equals n (tolerance 1e-10)
    sum_sw = sw.sum()
    # Use ValueError instead of assert for better error messages
    if not np.isclose(sum_sw, n, atol=1e-10):
        raise ValueError(
            f"Internal error in weight normalization: "
            f"normalized weights sum to {sum_sw:.10f} instead of {n}. "
            f"Difference: {abs(sum_sw - n):.2e}. This should not happen."
        )
    
    return sw


def validate_arrays(
    treat: np.ndarray,
    X: np.ndarray,
    check_rank: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize treatment and covariate arrays.

    Performs dimension checking, type conversion, missing value detection,
    and optional rank verification for the design matrix.

    Parameters
    ----------
    treat : np.ndarray
        Treatment vector, shape (n,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    check_rank : bool, default=True
        If True, verify X has full column rank.

    Returns
    -------
    treat : np.ndarray
        Validated treatment vector, dtype=float64.
    X : np.ndarray
        Validated covariate matrix, dtype=float64.

    Raises
    ------
    ValueError
        If dimensions mismatch, arrays contain NaN, or X is rank-deficient
        (when check_rank=True).

    Notes
    -----
    Full column rank is required for the GMM optimization to have a
    unique solution. Rank deficiency typically indicates collinear
    covariates that should be removed.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 0, 1, 0])
    >>> X = np.array([[1, 25], [1, 30], [1, 35], [1, 40]])
    >>> treat_v, X_v = validate_arrays(treat, X)
    >>> treat_v.dtype
    dtype('float64')

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Dimension check
    if len(treat) != X.shape[0]:
        raise ValueError(
            f"treat length {len(treat)} != X rows {X.shape[0]}"
        )
    
    # Type conversion (enforce float64)
    treat = np.asarray(treat, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    # Missing value check
    if np.isnan(treat).any() or np.isnan(X).any():
        raise ValueError(
            "Arrays contain NaN values. Use handle_missing() first."
        )
    
    # Full rank check
    if check_rank:
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            raise ValueError(
                f"X is not full rank: rank={rank} < ncol={X.shape[1]}"
            )
    
    return treat, X


def handle_missing(
    data: pd.DataFrame,
    relevant_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Remove observations with missing values.

    Performs listwise deletion of rows containing NA/NaN in the
    specified columns, with a warning indicating how many rows
    were removed.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    relevant_cols : list of str, optional
        Columns to check for missing values.
        If None, checks all columns.

    Returns
    -------
    data_clean : pd.DataFrame
        DataFrame with missing-value rows removed.
        Original index is preserved.
    n_dropped : int
        Number of rows removed.

    Warns
    -----
    UserWarning
        If any rows were dropped, indicates the count.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'treat': [1, 0, np.nan, 1],
    ...     'age': [25, 30, 35, np.nan]
    ... })
    >>> df_clean, n_drop = handle_missing(df)
    >>> len(df_clean)
    2
    >>> n_drop
    2

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    if relevant_cols is None:
        relevant_cols = data.columns.tolist()
    
    # Drop rows containing NA (preserve index)
    data_clean = data.dropna(subset=relevant_cols, inplace=False)
    n_dropped = len(data) - len(data_clean)
    
    # Warning message (matches na.omit behavior)
    if n_dropped > 0:
        warnings.warn(
            f"Removed {n_dropped} observations with missing values",
            UserWarning
        )
    
    return data_clean, n_dropped


def encode_treatment_factor(
    treat: Union[pd.Series, np.ndarray],
    att: int,
    verbose: int = 1
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Encode categorical treatment variable to binary 0/1.

    Converts a two-level categorical treatment to numeric encoding,
    with the encoding direction controlled by the ATT parameter.

    Parameters
    ----------
    treat : pd.Series or np.ndarray
        Categorical treatment variable with exactly 2 levels.
    att : int
        Target estimand controlling encoding:
        
        - 0: ATE - second level (alphabetically) becomes 1
        - 1: ATT - second level as treated group
        - 2: ATT - first level as treated group (inverts encoding)
        
    verbose : int, default=1
        If > 0, print information about which level is treated.

    Returns
    -------
    treat_numeric : np.ndarray
        Binary treatment vector (0/1), dtype=float64.
    levels : list
        Sorted list of the two factor levels.
    treat_orig : np.ndarray
        Original treatment values (for reference).

    Notes
    -----
    The encoding follows alphabetical ordering of levels:
    
    - Levels are sorted, so ``['control', 'treatment']`` becomes
      ``[0, 1]`` with 'treatment' = 1
    - ATT=2 inverts this, making the first level (alphabetically) = 1

    Examples
    --------
    >>> import pandas as pd
    >>> treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
    >>> treat_num, levels, _ = encode_treatment_factor(treat, att=1, verbose=0)
    >>> [float(x) for x in treat_num]
    [0.0, 1.0, 0.0, 1.0]
    >>> levels
    ['control', 'treatment']

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Save original values
    treat_orig = np.asarray(treat).copy()
    
    # Extract levels (sorted)
    if isinstance(treat, pd.Series):
        if hasattr(treat, 'cat'):
            levels = treat.cat.categories.tolist()
        else:
            levels = sorted(treat.unique())
    else:
        levels = sorted(np.unique(treat))
    
    # ATT encoding: second level becomes 1, first level becomes 0
    treat_numeric = (treat_orig == levels[1]).astype(int)
    
    # ATT=2: Invert treatment assignment
    if att == 2:
        treat_numeric = 1 - treat_numeric
    
    # Print ATT information (controlled by verbose parameter)
    if verbose > 0:
        if att == 1:
            print(
                f"Finding ATT with T={levels[1]} as the treatment. "
                f"Set ATT=2 to find ATT with T={levels[0]} as the treatment"
            )
        elif att == 2:
            print(
                f"Finding ATT with T={levels[0]} as the treatment. "
                f"Set ATT=1 to find ATT with T={levels[1]} as the treatment"
            )
        # ATT=0: No message (ATE scenario)
    
    return treat_numeric.astype(np.float64), levels, treat_orig

