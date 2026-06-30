"""
Variance Matrix Transformation Utilities

This module provides functions to transform variance-covariance matrices
from SVD-orthogonalized parameter space back to the original covariate space.
This transformation is essential for valid statistical inference after SVD
preprocessing of the design matrix.

When SVD is applied to the design matrix X for numerical stability, the
estimated coefficients live in a transformed space. To obtain standard errors
in the original covariate space, the variance matrix must be back-transformed
using the inverse of the SVD transformation.

The transformation formula for a variance matrix V in SVD space is:

    V_orig = D_x^{-1} @ (X'X)^{-1} @ X' @ X_svd @ V_d^{-1} @ V_svd @ V_d^{-1} @
             V_svd' @ X_svd' @ X @ (X'X)^{-1} @ D_x^{-1}

where D_x is the standardization matrix and V_d contains inverse singular values.

Functions
---------
transform_variance_binary
    Transform variance for binary treatment models.
transform_variance_3treat
    Transform variance for 3-level treatment models.
transform_variance_4treat
    Transform variance for 4-level treatment models.
transform_variance_continuous
    Transform variance for continuous treatment models.
apply_variance_svd_inverse_transform
    Dispatch function selecting appropriate transform based on treatment type.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import numpy as np
from typing import Dict, Any, Optional


def _r_ginv(A: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """
    Compute Moore-Penrose generalized inverse via SVD.
    
    Internal function used for variance matrix transformations.
    Singular values below the tolerance threshold are set to zero
    in the inversion to ensure numerical stability.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix, shape (m, n).
    tol : float, optional
        Relative singular value truncation threshold.
        If None, defaults to sqrt(machine_epsilon).
    
    Returns
    -------
    np.ndarray
        Generalized inverse, shape (n, m).
    """
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps)  # R: sqrt(.Machine$double.eps)
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    
    threshold = max(tol * s[0], 0.0)
    positive = s > threshold
    
    if np.all(positive):
        s_inv = 1.0 / s
        return Vt.T @ np.diag(s_inv) @ U.T
    if not np.any(positive):
        return np.zeros((A.shape[1], A.shape[0]))
    
    s_pos = s[positive]
    U_pos = U[:, positive]
    V_pos = Vt.T[:, positive]
    return V_pos @ np.diag(1.0 / s_pos) @ U_pos.T


def transform_variance_binary(
    variance_svd: np.ndarray,
    Dx_inv: np.ndarray,
    X_orig: np.ndarray,
    X_svd: np.ndarray,
    V: np.ndarray,
    d_inv: np.ndarray
) -> np.ndarray:
    """
    Transform variance matrix from SVD space for binary treatment models.
    
    Applies the inverse SVD transformation to convert the variance-covariance
    matrix from the orthogonalized parameter space back to the original
    covariate space, enabling proper inference on the original coefficients.
    
    Parameters
    ----------
    variance_svd : np.ndarray
        Variance matrix in SVD space, shape (k, k).
    Dx_inv : np.ndarray
        Inverse standardization diagonal matrix, shape (k, k).
        Contains [1, sd(x_1), sd(x_2), ...] on the diagonal.
    X_orig : np.ndarray
        Original design matrix with intercept, shape (n, k).
    X_svd : np.ndarray
        SVD-transformed design matrix, shape (n, k).
    V : np.ndarray
        Right singular vectors from X's SVD, shape (k, k).
    d_inv : np.ndarray
        Inverse singular values with small values zeroed, shape (k,).
    
    Returns
    -------
    np.ndarray
        Variance matrix in original covariate space, shape (k, k).
    
    See Also
    --------
    apply_variance_svd_inverse_transform : High-level dispatch function.
    """
    k = variance_svd.shape[0]
    
    # Compute generalized inverse of X_orig' @ X_orig
    XorigT_Xorig_inv = _r_ginv(X_orig.T @ X_orig)
    
    # Diagonal matrix of inverse singular values
    D_inv = np.diag(d_inv)
    
    # Complete transformation formula
    # Dx.inv @ ginv(X'X) @ X' @ Xsvd @ V @ D^-1 @ Var @ D^-1 @ V' @ Xsvd' @ X @ ginv(X'X) @ Dx.inv
    var_transformed = (
        Dx_inv @ 
        XorigT_Xorig_inv @ 
        X_orig.T @ 
        X_svd @ 
        V @ 
        D_inv @ 
        variance_svd @ 
        D_inv @ 
        V.T @ 
        X_svd.T @ 
        X_orig @ 
        XorigT_Xorig_inv @ 
        Dx_inv
    )
    
    return var_transformed


def transform_variance_3treat(
    variance_svd: np.ndarray,
    Dx_inv: np.ndarray,
    X_orig: np.ndarray,
    X_svd: np.ndarray,
    V: np.ndarray,
    d_inv: np.ndarray
) -> np.ndarray:
    """
    Transform variance matrix from SVD space for 3-level treatment models.
    
    For multinomial treatment with 3 levels, the variance matrix has a
    2x2 block structure corresponding to the (J-1) = 2 treatment contrasts.
    Each k x k block is transformed independently using the same formula
    as binary treatment.
    
    Parameters
    ----------
    variance_svd : np.ndarray
        Variance matrix in SVD space, shape (2k, 2k).
    Dx_inv : np.ndarray
        Inverse standardization diagonal matrix, shape (k, k).
    X_orig : np.ndarray
        Original design matrix with intercept, shape (n, k).
    X_svd : np.ndarray
        SVD-transformed design matrix, shape (n, k).
    V : np.ndarray
        Right singular vectors from X's SVD, shape (k, k).
    d_inv : np.ndarray
        Inverse singular values with small values zeroed, shape (k,).
    
    Returns
    -------
    np.ndarray
        Variance matrix in original covariate space, shape (2k, 2k).
    
    See Also
    --------
    transform_variance_binary : Single-block transformation formula.
    """
    k = X_orig.shape[1]
    
    # Decompose into 4 blocks
    var_1_1 = variance_svd[0:k, 0:k]
    var_1_2 = variance_svd[0:k, k:2*k]
    var_2_1 = variance_svd[k:2*k, 0:k]
    var_2_2 = variance_svd[k:2*k, k:2*k]
    
    # Compute common transformation matrices
    XorigT_Xorig_inv = _r_ginv(X_orig.T @ X_orig)
    D_inv = np.diag(d_inv)
    
    # Transform each block independently
    trans_var_1_1 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_1_1 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_1_2 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_1_2 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_2_1 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_2_1 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_2_2 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_2_2 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    
    # Reassemble full variance matrix
    var_transformed = np.block([
        [trans_var_1_1, trans_var_1_2],
        [trans_var_2_1, trans_var_2_2]
    ])
    
    return var_transformed


def transform_variance_4treat(
    variance_svd: np.ndarray,
    Dx_inv: np.ndarray,
    X_orig: np.ndarray,
    X_svd: np.ndarray,
    V: np.ndarray,
    d_inv: np.ndarray
) -> np.ndarray:
    """
    Transform variance matrix from SVD space for 4-level treatment models.
    
    For multinomial treatment with 4 levels, the variance matrix has a
    3x3 block structure corresponding to the (J-1) = 3 treatment contrasts.
    Each k x k block is transformed independently using the same formula
    as binary treatment.
    
    Parameters
    ----------
    variance_svd : np.ndarray
        Variance matrix in SVD space, shape (3k, 3k).
    Dx_inv : np.ndarray
        Inverse standardization diagonal matrix, shape (k, k).
    X_orig : np.ndarray
        Original design matrix with intercept, shape (n, k).
    X_svd : np.ndarray
        SVD-transformed design matrix, shape (n, k).
    V : np.ndarray
        Right singular vectors from X's SVD, shape (k, k).
    d_inv : np.ndarray
        Inverse singular values with small values zeroed, shape (k,).
    
    Returns
    -------
    np.ndarray
        Variance matrix in original covariate space, shape (3k, 3k).
    
    See Also
    --------
    transform_variance_binary : Single-block transformation formula.
    """
    k = X_orig.shape[1]
    
    # Decompose into 9 blocks
    var_1_1 = variance_svd[0:k, 0:k]
    var_1_2 = variance_svd[0:k, k:2*k]
    var_1_3 = variance_svd[0:k, 2*k:3*k]
    var_2_1 = variance_svd[k:2*k, 0:k]
    var_2_2 = variance_svd[k:2*k, k:2*k]
    var_2_3 = variance_svd[k:2*k, 2*k:3*k]
    var_3_1 = variance_svd[2*k:3*k, 0:k]
    var_3_2 = variance_svd[2*k:3*k, k:2*k]
    var_3_3 = variance_svd[2*k:3*k, 2*k:3*k]
    
    # Compute common transformation matrices
    XorigT_Xorig_inv = _r_ginv(X_orig.T @ X_orig)
    D_inv = np.diag(d_inv)
    
    # Transform each block independently
    trans_var_1_1 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_1_1 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_1_2 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_1_2 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_1_3 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_1_3 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_2_1 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_2_1 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_2_2 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_2_2 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_2_3 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_2_3 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_3_1 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_3_1 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_3_2 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_3_2 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    trans_var_3_3 = (
        Dx_inv @ XorigT_Xorig_inv @ X_orig.T @ X_svd @ V @ D_inv @ 
        var_3_3 @ D_inv @ V.T @ X_svd.T @ X_orig @ XorigT_Xorig_inv @ Dx_inv
    )
    
    # Reassemble full variance matrix
    var_transformed = np.block([
        [trans_var_1_1, trans_var_1_2, trans_var_1_3],
        [trans_var_2_1, trans_var_2_2, trans_var_2_3],
        [trans_var_3_1, trans_var_3_2, trans_var_3_3]
    ])
    
    return var_transformed


def transform_variance_continuous(
    variance_svd: np.ndarray,
    Dx_inv: np.ndarray,
    X_orig: np.ndarray,
    X_svd: np.ndarray,
    V: np.ndarray,
    d_inv: np.ndarray
) -> np.ndarray:
    """
    Transform variance matrix from SVD space for continuous treatment models.
    
    For continuous treatments, the coefficient vector has the same dimension
    as the binary case (k parameters), so the same transformation formula
    applies directly.
    
    Parameters
    ----------
    variance_svd : np.ndarray
        Variance matrix in SVD space, shape (k, k).
    Dx_inv : np.ndarray
        Inverse standardization diagonal matrix, shape (k, k).
    X_orig : np.ndarray
        Original design matrix with intercept, shape (n, k).
    X_svd : np.ndarray
        SVD-transformed design matrix, shape (n, k).
    V : np.ndarray
        Right singular vectors from X's SVD, shape (k, k).
    d_inv : np.ndarray
        Inverse singular values with small values zeroed, shape (k,).
    
    Returns
    -------
    np.ndarray
        Variance matrix in original covariate space, shape (k, k).
    
    See Also
    --------
    transform_variance_binary : Underlying transformation implementation.
    """
    # Continuous treatment uses same inverse transform as binary
    return transform_variance_binary(
        variance_svd, Dx_inv, X_orig, X_svd, V, d_inv
    )


def apply_variance_svd_inverse_transform(
    variance_svd: np.ndarray,
    svd_info: Dict[str, Any],
    X_orig: np.ndarray,
    X_svd: np.ndarray,
    is_factor: bool,
    no_treats: int
) -> np.ndarray:
    """
    Apply SVD inverse transformation to variance matrix.
    
    High-level dispatch function that selects the appropriate inverse
    transformation method based on the treatment type and applies it
    to convert the variance matrix from SVD space to original space.
    
    Parameters
    ----------
    variance_svd : np.ndarray
        Variance matrix in SVD space.
        Shape depends on treatment type: (k, k) for binary/continuous,
        (2k, 2k) for 3-level, (3k, 3k) for 4-level.
    svd_info : dict
        SVD preprocessing information containing:
        
        - ``'d'`` : Singular values from SVD, shape (k,)
        - ``'V'`` : Right singular vectors, shape (k, k)
        - ``'x_mean'`` : Column means of original X (excluding intercept)
        - ``'x_sd'`` : Column standard deviations of original X
        
    X_orig : np.ndarray
        Original design matrix with intercept, shape (n, k).
    X_svd : np.ndarray
        SVD-transformed design matrix, shape (n, k).
    is_factor : bool
        True for discrete (factor) treatment, False for continuous.
    no_treats : int
        Number of treatment levels.
        2, 3, or 4 for discrete treatments; ignored for continuous.
    
    Returns
    -------
    np.ndarray
        Variance matrix in original covariate space.
    
    Notes
    -----
    The transformation handles different treatment types:
    
    - **Binary** (no_treats=2): Single k x k block transformation
    - **3-level** (no_treats=3): Four k x k blocks forming 2k x 2k matrix
    - **4-level** (no_treats=4): Nine k x k blocks forming 3k x 3k matrix
    - **Continuous**: Same as binary (single k x k block)
    
    Examples
    --------
    This function is typically called internally by CBPS variance methods:
    
    >>> import numpy as np
    >>> # svd_info, X_orig, X_svd are obtained from CBPS fitting
    >>> # var_orig = apply_variance_svd_inverse_transform(
    >>> #     var_svd, svd_info, X_orig, X_svd, is_factor=True, no_treats=2
    >>> # )
    """
    k = X_orig.shape[1]
    
    # Construct Dx_inv: diag([1, x_sd[0], x_sd[1], ...])
    x_sd = svd_info['x_sd']
    Dx_inv = np.diag(np.concatenate([[1.0], x_sd]))
    
    # Construct d_inv: inverse of singular values (values <= 1e-5 set to 0)
    d_inv = svd_info['d'].copy()
    d_inv[d_inv > 1e-5] = 1.0 / d_inv[d_inv > 1e-5]
    d_inv[d_inv <= 1e-5] = 0.0
    
    V = svd_info['V']
    
    # Select inverse transform method based on treatment type
    if is_factor and no_treats == 2:
        # Binary treatment
        var_transformed = transform_variance_binary(
            variance_svd, Dx_inv, X_orig, X_svd, V, d_inv
        )
    elif is_factor and no_treats == 3:
        # 3-level treatment
        var_transformed = transform_variance_3treat(
            variance_svd, Dx_inv, X_orig, X_svd, V, d_inv
        )
    elif is_factor and no_treats == 4:
        # 4-level treatment
        var_transformed = transform_variance_4treat(
            variance_svd, Dx_inv, X_orig, X_svd, V, d_inv
        )
    else:
        # Continuous treatment
        var_transformed = transform_variance_continuous(
            variance_svd, Dx_inv, X_orig, X_svd, V, d_inv
        )
    
    return var_transformed
