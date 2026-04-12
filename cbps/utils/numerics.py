"""
Numerical Linear Algebra Utilities

This module provides numerically stable implementations of matrix operations
commonly used in CBPS estimation, including pseudoinverses, matrix rank
computation, and symmetry utilities.

The pseudoinverse functions implement tolerance-based singular value truncation
to handle rank-deficient matrices that arise in high-dimensional settings or
with collinear covariates.

Functions
---------
r_ginv_like
    Moore-Penrose pseudoinverse with configurable tolerance.
pinv_match_r
    Pseudoinverse using NumPy with matched tolerance.
pinv_symmetric_psd
    Specialized pseudoinverse for symmetric positive semi-definite matrices.
numeric_rank
    Effective numerical rank via singular value decomposition.
symmetrize
    Force matrix symmetry by averaging with transpose.
max_asymmetry
    Measure of matrix asymmetry (infinity norm of A - A.T).
is_symmetric
    Check if matrix is symmetric within tolerance.

References
----------
Golub, G. H. and Van Loan, C. F. (2013). Matrix Computations (4th ed.).
Johns Hopkins University Press.
"""

import numpy as np
import scipy.linalg as la
from typing import Optional


def r_ginv_like(X: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """
    Compute Moore-Penrose pseudoinverse with tolerance-based truncation.

    Uses SVD decomposition with a threshold rule for singular value truncation:
    singular values below the tolerance are set to zero in the inversion.

    Parameters
    ----------
    X : np.ndarray
        Input matrix to pseudo-invert, shape (m, n).
    tol : float, optional
        Absolute tolerance for singular value truncation.
        If None, uses: max(m, n) * max(singular_values) * machine_epsilon.

    Returns
    -------
    np.ndarray
        Pseudoinverse of X, shape (n, m).

    Notes
    -----
    The tolerance rule follows the standard numerical convention:
    
        tol = max(dim(X)) * sigma_max * eps
    
    where sigma_max is the largest singular value and eps is machine epsilon.
    This ensures robustness against numerical rank deficiency.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_pinv = r_ginv_like(X)
    >>> # Verify pseudoinverse property: X @ X_pinv @ X ≈ X
    >>> assert np.allclose(X @ X_pinv @ X, X, atol=1e-10)
    """
    X = np.asarray(X)
    # Compute SVD once to control tolerance exactly and avoid SciPy defaults
    U, s, Vt = la.svd(X, full_matrices=False, lapack_driver='gesdd')
    if tol is None:
        eps = np.finfo(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64).eps
        tol = max(X.shape) * s.max(initial=0.0) * eps
    # Invert with truncation
    with np.errstate(divide='ignore', invalid='ignore'):
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


def r_ginv_rcond(X: np.ndarray) -> float:
    """
    Compute the relative condition number for pseudoinverse truncation.

    Converts the absolute tolerance rule to a relative cutoff suitable for
    NumPy/SciPy pinv functions.

    Parameters
    ----------
    X : np.ndarray
        Input matrix for which to compute rcond.

    Returns
    -------
    float
        Relative condition number: max(dim(X)) * machine_epsilon.

    Notes
    -----
    The relationship between absolute and relative tolerances is:
    
        absolute_tol = rcond * sigma_max
        rcond = max(m, n) * eps
    
    where sigma_max is the largest singular value.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> rcond = r_ginv_rcond(X)
    >>> assert rcond > 0
    """
    X = np.asarray(X)
    eps = np.finfo(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64).eps
    return max(X.shape) * eps


def pinv_match_r(X: np.ndarray) -> np.ndarray:
    """
    Compute pseudoinverse using NumPy with standard tolerance.

    A convenience wrapper around numpy.linalg.pinv that applies
    the standard tolerance rule for singular value truncation.

    Parameters
    ----------
    X : np.ndarray
        Input matrix to pseudo-invert.

    Returns
    -------
    np.ndarray
        Pseudoinverse of X.

    See Also
    --------
    r_ginv_like : Direct SVD-based implementation with custom tolerance.
    r_ginv_rcond : Computes the rcond value used here.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> X_pinv = pinv_match_r(X)
    >>> assert np.allclose(X @ X_pinv @ X, X, atol=1e-10)
    """
    return np.linalg.pinv(np.asarray(X), rcond=r_ginv_rcond(X))


def pinv_symmetric_psd(X: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """
    Compute pseudoinverse for symmetric positive semi-definite matrices.

    Uses eigenvalue decomposition instead of SVD, exploiting symmetry for
    improved numerical stability and efficiency. Small or negative eigenvalues
    (arising from numerical noise) are clipped to zero.

    Parameters
    ----------
    X : np.ndarray
        Symmetric input matrix to pseudo-invert, shape (n, n).
    tol : float, optional
        Absolute tolerance for eigenvalue truncation.
        If None, uses: n * max(eigenvalues) * machine_epsilon.

    Returns
    -------
    np.ndarray
        Symmetric pseudoinverse of X, shape (n, n).

    Notes
    -----
    For a symmetric matrix X = Q Λ Q^T, the pseudoinverse is:
    
        X^+ = Q Λ^+ Q^T
    
    where Λ^+ has diagonal entries 1/λ_i for λ_i > tol, and 0 otherwise.

    The input matrix is symmetrized as 0.5*(X + X^T) before decomposition
    to handle minor floating-point asymmetries.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a symmetric positive definite matrix
    >>> A = np.array([[4, 2], [2, 3]])
    >>> A_pinv = pinv_symmetric_psd(A)
    >>> assert np.allclose(A @ A_pinv @ A, A, atol=1e-10)
    >>> assert np.allclose(A_pinv, A_pinv.T, atol=1e-14)  # Result is symmetric
    """
    X = np.asarray(X)
    # Symmetrize defensively to counter FP drift
    X = 0.5 * (X + X.T)
    # Eigen-decomposition for symmetric matrices
    w, Q = la.eigh(X)
    if tol is None:
        eps = np.finfo(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64).eps
        tol = max(X.shape) * float(np.max(w, initial=0.0)) * eps
    # Invert with clipping
    w_inv = np.where(w > tol, 1.0 / w, 0.0)
    return (Q * w_inv) @ Q.T


def numeric_rank(X: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Compute effective numerical rank via singular value decomposition.

    The numerical rank counts singular values exceeding the tolerance threshold,
    providing a robust measure of matrix rank that accounts for floating-point
    precision limitations.

    Parameters
    ----------
    X : np.ndarray
        Input matrix, shape (m, n).
    tol : float, optional
        Absolute tolerance for singular value truncation.
        If None, uses: max(m, n) * max(singular_values) * machine_epsilon.

    Returns
    -------
    int
        Number of singular values exceeding the tolerance.

    Notes
    -----
    Unlike numpy.linalg.matrix_rank, this function uses a tolerance rule
    that scales with the matrix dimensions, providing more consistent
    behavior across different problem sizes.

    Examples
    --------
    >>> import numpy as np
    >>> # Full rank matrix
    >>> X = np.array([[1, 2], [3, 4]])
    >>> assert numeric_rank(X) == 2
    >>> 
    >>> # Rank deficient matrix
    >>> Y = np.array([[1, 2], [2, 4]])  # Second row is 2x first row
    >>> assert numeric_rank(Y) == 1
    """
    X = np.asarray(X)
    s = la.svd(X, compute_uv=False, lapack_driver='gesdd')
    if tol is None:
        eps = np.finfo(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64).eps
        tol = max(X.shape) * s.max(initial=0.0) * eps
    return int(np.sum(s > tol))


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Force matrix symmetry by averaging with its transpose.

    Computes 0.5 * (A + A^T), which projects any square matrix onto
    the space of symmetric matrices.

    Parameters
    ----------
    A : np.ndarray
        Square matrix, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric matrix, shape (n, n).

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_sym = symmetrize(A)
    >>> assert np.allclose(A_sym, A_sym.T)
    >>> assert np.allclose(A_sym, [[1, 2.5], [2.5, 4]])
    """
    A = np.asarray(A)
    return 0.5 * (A + A.T)


def max_asymmetry(A: np.ndarray) -> float:
    """
    Compute the maximum asymmetry of a matrix.

    Returns the infinity norm of (A - A^T), measuring how far
    the matrix deviates from perfect symmetry.

    Parameters
    ----------
    A : np.ndarray
        Square matrix, shape (n, n).

    Returns
    -------
    float
        Maximum absolute difference: max|A_ij - A_ji|.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2], [2.001, 4]])
    >>> asym = max_asymmetry(A)
    >>> assert np.isclose(asym, 0.001)
    """
    A = np.asarray(A)
    return float(np.max(np.abs(A - A.T)))


def is_symmetric(A: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Check if a matrix is symmetric within tolerance.

    Parameters
    ----------
    A : np.ndarray
        Square matrix to check, shape (n, n).
    atol : float, default=1e-12
        Absolute tolerance for asymmetry.

    Returns
    -------
    bool
        True if max|A_ij - A_ji| <= atol.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2], [2, 4]])
    >>> assert is_symmetric(A)
    >>> 
    >>> B = np.array([[1, 2], [3, 4]])
    >>> assert not is_symmetric(B)
    """
    return max_asymmetry(A) <= atol
