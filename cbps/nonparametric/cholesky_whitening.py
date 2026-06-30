"""
Cholesky Whitening Transform for Nonparametric CBPS.

This module implements covariate whitening via Cholesky decomposition,
transforming covariates to have zero mean, unit variance, and zero
correlation. This preprocessing step is essential for the empirical
likelihood formulation in npCBPS.

Mathematical Background
-----------------------
The whitening transform orthogonalizes covariates as described in
Section 3.1 of Fong, Hazlett, and Imai (2018):

.. math::

    X_i^* = S_X^{-1/2}(X_i - \\bar{X})

where :math:`\\bar{X}` is the sample mean and :math:`S_X` is the sample
covariance matrix. The Cholesky decomposition provides a numerically
stable way to compute :math:`S_X^{-1/2}`.

After whitening, :math:`\\text{Cov}(X^*) = I_K` (identity matrix), which
simplifies the covariate balancing constraints in the empirical likelihood
optimization.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. The Annals of Applied Statistics, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101

See Section 3.1 for the notation and Section 3.3.1 for the nonparametric
formulation.
"""

import numpy as np
import scipy.linalg


def cholesky_whitening(X: np.ndarray, verify: bool = True) -> np.ndarray:
    """
    Transform covariates to have identity covariance matrix.

    Applies a two-step whitening procedure using Cholesky decomposition:

    1. **Decorrelation**: :math:`X' = X \\cdot \\text{inv}(\\text{chol}(S_X))`
       where :math:`S_X` is the sample covariance matrix.
    2. **Standardization**: Center to zero mean and scale to unit variance.

    The result satisfies :math:`\\text{Cov}(X^*) = I_K`, which is required
    for the covariate balancing constraints in npCBPS.

    Parameters
    ----------
    X : np.ndarray of shape (n, k)
        Covariate matrix with n observations and k variables.
    verify : bool, default=True
        If True, verify that the output covariance equals the identity
        matrix within numerical tolerance. Raises AssertionError on failure.

    Returns
    -------
    np.ndarray of shape (n, k)
        Whitened covariate matrix satisfying:

        - Column means are zero
        - Column standard deviations are one
        - Covariance matrix equals identity

    Raises
    ------
    AssertionError
        If ``verify=True`` and the whitening verification fails.
    numpy.linalg.LinAlgError
        If the covariance matrix is not positive definite.

    Notes
    -----
    **Algorithm details:**

    The Cholesky decomposition factorizes :math:`S_X = L L^T` where L is
    lower triangular. This implementation uses the upper triangular form
    :math:`S_X = U^T U` via ``scipy.linalg.cholesky(..., lower=False)``.

    The whitening transform is then :math:`X' = X \\cdot U^{-1}`, followed
    by standardization to ensure exact zero mean and unit variance.

    **Verification criteria (tolerance 1e-10):**

    - Diagonal of :math:`\\text{Cov}(X^*)` equals 1
    - Off-diagonal elements equal 0
    - Column means equal 0

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Section 3.1 describes the
    whitening notation :math:`X_i^* = S_X^{-1/2}(X_i - \\bar{X})`.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>> X_white = cholesky_whitening(X)
    >>> cov = np.cov(X_white.T, ddof=1)
    >>> np.allclose(cov, np.eye(3), atol=1e-10)
    True
    >>> np.allclose(X_white.mean(axis=0), 0, atol=1e-10)
    True
    """
    n, k = X.shape

    # Step 1: Cholesky whitening
    # Compute unbiased covariance estimate
    cov_X = np.cov(X.T, ddof=1)

    # Cholesky decomposition returns upper triangular matrix
    chol_upper = scipy.linalg.cholesky(cov_X, lower=False)

    # Apply whitening transform
    X_white_step1 = X @ np.linalg.inv(chol_upper)

    # Step 2: Full standardization (center=True, scale=True)
    # Ensures zero mean and unit variance
    X_white = (X_white_step1 - X_white_step1.mean(axis=0)) / X_white_step1.std(axis=0, ddof=1)

    # Whitening verification (optional, enabled by default)
    if verify:
        cov_white = np.cov(X_white.T, ddof=1)

        # Single variable case: cov returns 0-dim scalar, reshape to (1,1)
        if k == 1:
            cov_white = cov_white.reshape(1, 1)

        # Check diagonal elements are close to 1
        diagonal = np.diag(cov_white)
        if not np.allclose(diagonal, 1.0, atol=1e-10):
            raise AssertionError(
                f"Whitening failed: cov(X_white) diagonal not close to 1\n"
                f"Diagonal values: {diagonal}\n"
                f"Expected: [1, 1, ..., 1]"
            )

        # Check off-diagonal elements are close to 0
        off_diagonal_max = np.max(np.abs(cov_white - np.eye(k)))
        if off_diagonal_max > 1e-10:
            raise AssertionError(
                f"Whitening failed: cov(X_white) off-diagonal elements too large\n"
                f"Maximum off-diagonal absolute value: {off_diagonal_max}\n"
                f"Expected: approximately 0 (tolerance 1e-10)"
            )

        # Check overall covariance matrix
        if not np.allclose(cov_white, np.eye(k), atol=1e-10):
            raise AssertionError(
                f"Whitening failed: cov(X_white) not close to identity matrix I\n"
                f"Maximum deviation: {np.max(np.abs(cov_white - np.eye(k)))}"
            )

    return X_white


def verify_whitening(X: np.ndarray, X_white: np.ndarray, atol: float = 1e-10) -> dict:
    """
    Compute diagnostic metrics for whitening quality.

    This function provides detailed verification of the whitening transform
    beyond the basic checks in :func:`cholesky_whitening`.

    Parameters
    ----------
    X : np.ndarray of shape (n, k)
        Original covariate matrix (unused, kept for API consistency).
    X_white : np.ndarray of shape (n, k)
        Whitened covariate matrix to verify.
    atol : float, default=1e-10
        Absolute tolerance for numerical comparisons.

    Returns
    -------
    dict
        Verification metrics with keys:

        - **cov_is_identity** : bool
            True if covariance matrix equals identity within tolerance.
        - **mean_is_zero** : bool
            True if all column means are zero within tolerance.
        - **std_is_one** : bool
            True if all column standard deviations are one within tolerance.
        - **max_cov_deviation** : float
            Maximum absolute deviation of covariance from identity matrix.
        - **condition_number** : float
            Condition number of the whitened matrix (measures numerical stability).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3)
    >>> X_white = cholesky_whitening(X)
    >>> metrics = verify_whitening(X, X_white)
    >>> metrics['cov_is_identity']
    True
    >>> metrics['max_cov_deviation'] < 1e-10
    True
    """
    k = X.shape[1]

    # Compute covariance matrix
    cov_white = np.cov(X_white.T, ddof=1)

    # Compute mean and standard deviation
    mean_white = X_white.mean(axis=0)
    std_white = X_white.std(axis=0, ddof=1)

    # Verification metrics
    cov_is_identity = np.allclose(cov_white, np.eye(k), atol=atol)
    mean_is_zero = np.allclose(mean_white, 0, atol=atol)
    std_is_one = np.allclose(std_white, 1, atol=atol)
    max_cov_deviation = np.max(np.abs(cov_white - np.eye(k)))

    # Condition number (measures numerical stability)
    condition_number = np.linalg.cond(X_white)

    return {
        'cov_is_identity': cov_is_identity,
        'mean_is_zero': mean_is_zero,
        'std_is_one': std_is_one,
        'max_cov_deviation': max_cov_deviation,
        'condition_number': condition_number
    }
