"""
Variance Adjustment for Weighted Outcome Regression
====================================================

Sandwich variance estimator for weighted least squares regression using
CBPS weights from continuous treatment models. Adjusts standard errors to
account for estimation uncertainty in the generalized propensity score.

The methodology follows Section 3.2 of Fong, Hazlett, and Imai (2018),
which derives the asymptotic variance of the weighted least squares
estimator by viewing it as a method of moments estimator based on the
combined moment conditions for propensity score estimation and outcome
regression.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. The Annals of Applied Statistics,
    12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

Newey, W. K. and McFadden, D. (1994). Large sample estimation and
    hypothesis testing. In Handbook of Econometrics, Vol. IV, 2111-2245.
"""

from typing import Union
import numpy as np
from cbps.core.results import CBPSResults


def vcov_outcome(
    cbps_fit: CBPSResults,
    Y: np.ndarray,
    Z: np.ndarray,
    delta: np.ndarray,
    tol: float = 1e-5,
    lambda_: float = 0.01
) -> np.ndarray:
    """
    Compute adjusted variance-covariance matrix for weighted outcome regression.

    Adjusts standard errors to account for uncertainty in CBPS weight
    estimation when using continuous treatment weights. Implements the
    asymptotic variance formula from Section 3.2 of Fong, Hazlett, and
    Imai (2018), treating the weighted regression as a GMM estimator.

    Parameters
    ----------
    cbps_fit : CBPSResults
        Fitted continuous treatment CBPS object with attributes: Ttilde
        (standardized treatment), Xtilde (whitened covariates), beta_tilde,
        sigmasq_tilde, and weights.
    Y : array-like of shape (n,)
        Outcome variable.
    Z : array-like of shape (n, q)
        Outcome model design matrix (including treatment and intercept).
    delta : array-like of shape (q,)
        WLS coefficients from the weighted outcome regression.
    tol : float, default=1e-5
        Condition number tolerance. If the smallest singular value of M
        divided by the largest is below tol, regularization is applied.
    lambda_ : float, default=0.01
        Ridge regularization constant added to diagonal of M when
        ill-conditioned.

    Returns
    -------
    V : ndarray of shape (q, q)
        Adjusted variance-covariance matrix for delta.

    Raises
    ------
    ValueError
        If cbps_fit lacks continuous treatment attributes or dimensions
        are incompatible.

    See Also
    --------
    asy_var : Variance estimation for binary treatment ATE.

    Notes
    -----
    The variance formula accounts for estimation uncertainty in both the
    propensity score parameters (beta, sigma^2) and the outcome regression
    coefficients (delta). The sandwich estimator follows Newey and McFadden
    (1994, Theorem 6.1).

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from cbps import CBPS, vcov_outcome
    >>> fit = CBPS('T ~ X1 + X2 + X3', data=df, att=False)
    >>> Z = sm.add_constant(df[['T', 'X1', 'X2']])
    >>> wls = sm.WLS(df['Y'], Z, weights=fit.weights).fit()
    >>> V_adj = vcov_outcome(fit, df['Y'], Z, wls.params)
    >>> se_adj = np.sqrt(np.diag(V_adj))
    """
    # Input validation
    if not hasattr(cbps_fit, 'Ttilde') or cbps_fit.Ttilde is None:
        raise ValueError(
            "cbps_fit must be a continuous treatment CBPS object with Ttilde "
            "attribute. For binary treatments, use asy_var() instead."
        )
    if not hasattr(cbps_fit, 'Xtilde') or cbps_fit.Xtilde is None:
        raise ValueError("cbps_fit missing Xtilde attribute")
    if not hasattr(cbps_fit, 'beta_tilde') or cbps_fit.beta_tilde is None:
        raise ValueError("cbps_fit missing beta_tilde attribute")
    if not hasattr(cbps_fit, 'sigmasq_tilde') or cbps_fit.sigmasq_tilde is None:
        raise ValueError("cbps_fit missing sigmasq_tilde attribute")

    # Extract attributes
    Xtilde = cbps_fit.Xtilde
    Ttilde = cbps_fit.Ttilde
    w = cbps_fit.weights
    beta_tilde = cbps_fit.beta_tilde
    sigmasq_tilde = cbps_fit.sigmasq_tilde

    # Convert to numpy arrays
    Y = np.asarray(Y).ravel()
    Z = np.asarray(Z)
    delta = np.asarray(delta).ravel()

    # Dimension validation and shape normalization
    N = len(Y)
    Ttilde = np.asarray(Ttilde).reshape(-1)
    w = np.asarray(w).reshape(-1)
    Xtilde = np.asarray(Xtilde)
    Z = np.asarray(Z)
    if Xtilde.ndim != 2:
        raise ValueError("Xtilde must be a 2D matrix")
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix")
    if Xtilde.shape[0] != N and Xtilde.shape[1] == N:
        Xtilde = Xtilde.T
    if Z.shape[0] != N and Z.shape[1] == N:
        Z = Z.T
    if len(Ttilde) != N:
        raise ValueError(f"Ttilde length ({len(Ttilde)}) does not match Y length ({N})")
    if Xtilde.shape[0] != N:
        raise ValueError(f"Xtilde row count ({Xtilde.shape[0]}) does not match Y length ({N})")
    if Z.shape[0] != N:
        raise ValueError(f"Z row count ({Z.shape[0]}) does not match Y length ({N})")
    if len(delta) != Z.shape[1]:
        raise ValueError(f"delta length ({len(delta)}) does not match Z column count ({Z.shape[1]})")
    if len(w) != N:
        raise ValueError(f"weights length ({len(w)}) does not match Y length ({N})")

    # Parameter validation
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if tol > 1.0:
        import warnings
        warnings.warn(
            f"tol={tol} > 1 triggers regularization unconditionally",
            UserWarning
        )
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")

    # Dimensions: K = number of covariates in propensity model, P = outcome model
    K = Xtilde.shape[1]
    P = Z.shape[1]
    Sdelta = np.zeros((P, P))
    Stheta = np.zeros((P, K+1))

    # Residuals from propensity and outcome models
    eps_beta = Ttilde - Xtilde @ beta_tilde
    eps_delta = Y - Z @ delta

    # M-matrix: Jacobian of moment conditions (Section 3.2, Fong et al. 2018)
    M11 = np.mean(-2/sigmasq_tilde * eps_beta[:, None] * Xtilde, axis=0)
    M12 = np.mean(-1/sigmasq_tilde**2 * eps_beta**2)
    M22 = np.mean(
        (1/(2*sigmasq_tilde) * w * (1 - 1/sigmasq_tilde * eps_beta**2) * Ttilde)[:, None] * Xtilde,
        axis=0
    )

    # Compute M21, Sdelta, Stheta via accumulation
    M21 = np.zeros((K, K))
    for i in range(N):
        M21 += (-1/sigmasq_tilde * w[i] * Ttilde[i] * eps_beta[i]) * np.outer(Xtilde[i], Xtilde[i]) / N
        Sdelta -= w[i] * np.outer(Z[i], Z[i]) / N
        Stheta += np.hstack([
            -1/sigmasq_tilde * w[i] * eps_beta[i] * eps_delta[i] * np.outer(Z[i], Xtilde[i]),
            (1/(2*sigmasq_tilde) * w[i] * (1 - 1/sigmasq_tilde * eps_beta[i]**2) * eps_delta[i] * Z[i])[:, None]
        ]) / N

    # Assemble M-matrix
    M = np.vstack([
        np.hstack([M11, [M12]]),
        np.hstack([M21, M22[:, None]])
    ])

    # Ridge regularization if M is ill-conditioned
    sv = np.linalg.svd(M, compute_uv=False)
    cond_num = sv[0] / sv[-1]
    if cond_num > (1/tol):
        M = M + lambda_ * np.eye(M.shape[0])

    # Sandwich variance estimator (Section 3.2, Fong et al. 2018)
    s = (w * eps_delta)[:, None] * Z
    mtheta = np.hstack([
        ((1/sigmasq_tilde) * (eps_beta**2) - 1)[:, None],
        (w * Ttilde)[:, None] * Xtilde
    ])
    assert mtheta.shape == (N, K+1), f"mtheta shape mismatch: {mtheta.shape}"

    M_inv = np.linalg.inv(M)
    inner = np.zeros((P, P))
    for i in range(N):
        inner_part = s[i] - Stheta @ M_inv @ mtheta[i]
        inner += np.outer(inner_part, inner_part) / N

    Sdelta_inv = np.linalg.inv(Sdelta)
    V = Sdelta_inv @ inner @ Sdelta_inv.T / N

    return V

