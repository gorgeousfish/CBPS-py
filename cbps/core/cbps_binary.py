"""
Binary Treatment Covariate Balancing Propensity Score

This module implements the CBPS algorithm for binary treatments, supporting
both exactly-identified and over-identified generalized method of moments
(GMM) estimation.

The covariate balancing propensity score (CBPS) methodology estimates
propensity scores that optimize covariate balance while maintaining good
prediction of treatment assignment.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import warnings
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

from ..utils.weights import standardize_weights
from ..utils.helpers import normalize_sample_weights

# Constants
PROBS_MIN = 1e-6  # Minimum probability clipping threshold


def _r_ginv(X: np.ndarray, tol: float = None) -> np.ndarray:
    """
    Compute the Moore-Penrose pseudoinverse with numerical stability.

    This function computes the pseudoinverse using a tolerance based on the
    square root of machine epsilon to determine which singular values to retain.

    Parameters
    ----------
    X : np.ndarray
        Input matrix for which to compute the pseudoinverse.
    tol : float, optional
        Tolerance parameter for singular value selection. Default is
        sqrt(machine_epsilon) ≈ 1.49e-08. Singular values d are kept if
        d > tol * max(d).

    Returns
    -------
    np.ndarray
        The Moore-Penrose pseudoinverse of X.

    Notes
    -----
    The implementation follows a three-branch logic:
    1. If all singular values are positive: compute full pseudoinverse
    2. If no singular values are positive: return zero matrix
    3. If some singular values are positive: compute partial pseudoinverse
    """
    # Default tolerance: sqrt(machine epsilon) ≈ 1.49e-08
    if tol is None:
        machine_eps = np.finfo(float).eps  # 2.220446049250313e-16
        tol = np.sqrt(machine_eps)  # ≈ 1.490116119384766e-08
    
    # Compute reduced SVD decomposition (matches R's svd() default behavior)
    # For X with shape (m, n):
    #   - U has shape (m, min(m,n))
    #   - d has shape (min(m,n),)
    #   - Vt has shape (min(m,n), n)
    Xsvd_u, Xsvd_d, Xsvd_vt = np.linalg.svd(X, full_matrices=False)
    Xsvd_v = Xsvd_vt.T  # NumPy returns V^T, transpose to get V

    # If no singular values or maximum is extremely small (< machine eps),
    # return zero matrix to avoid numerical amplification
    if len(Xsvd_d) == 0 or Xsvd_d[0] < np.finfo(float).eps:
        return np.zeros((X.shape[1], X.shape[0]))

    # Determine which singular values to retain: d > max(tol * d[0], 0)
    # This matches R's MASS::ginv tolerance formula
    tol_threshold = max(tol * Xsvd_d[0], 0.0)
    Positive = Xsvd_d > tol_threshold

    # Compute pseudoinverse based on retained singular values
    # Formula: X+ = V @ diag(1/d) @ U.T (for retained singular values)
    if np.all(Positive):
        # All singular values retained: V @ diag(1/d) @ U.T
        X_pinv = Xsvd_v @ np.diag(1.0 / Xsvd_d) @ Xsvd_u.T
    elif not np.any(Positive):
        # All singular values truncated: return zero matrix
        X_pinv = np.zeros((X.shape[1], X.shape[0]))
    else:
        # Partial retention: V[:, pos] @ diag(1/d[pos]) @ U[:, pos].T
        Xsvd_v_pos = Xsvd_v[:, Positive]
        Xsvd_d_pos = Xsvd_d[Positive]
        Xsvd_u_pos = Xsvd_u[:, Positive]
        X_pinv = Xsvd_v_pos @ np.diag(1.0 / Xsvd_d_pos) @ Xsvd_u_pos.T
    
    return X_pinv


def _att_wt_func(
    beta_curr: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray
) -> np.ndarray:
    """
    Compute Average Treatment Effect on the Treated (ATT) weights.

    This function implements the ATT weight function that assigns weights
    to observations based on their estimated propensity scores. The weights
    are constructed to balance covariates between treated and control groups.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates, shape (k,).
    X : np.ndarray
        Covariate matrix including intercept, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights summing to n, shape (n,).

    Returns
    -------
    np.ndarray
        ATT weights, possibly containing negative values for control units.
        Shape (n,). The calling function should take absolute values.

    Notes
    -----
    The ATT weight formula is:
    w_i = (n/n_t) * (T_i - π_i) / (1 - π_i)

    where n_t is the weighted sum of treated units and π_i is the estimated
    propensity score. Treated units receive positive weights while control
    units receive negative weights, reflecting the ATT estimand.
    """
    # Compute weighted sample sizes
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    n = n_c + n_t
    
    # Compute propensity scores
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)
    
    # Clip probabilities to avoid numerical instability
    probs_curr = np.minimum(1 - PROBS_MIN, probs_curr)
    probs_curr = np.maximum(PROBS_MIN, probs_curr)
    
    # ATT weight formula: w = (n/n_t) * (T - pi) / (1 - pi)
    w1 = (n / n_t) * (treat - probs_curr) / (1 - probs_curr)
    
    return w1


def _compute_V_matrix(
    X: np.ndarray,
    probs_curr: np.ndarray,
    sample_weights: np.ndarray,
    treat: np.ndarray,
    att: int,
    n: int
) -> np.ndarray:
    """
    Compute the covariance matrix V for GMM estimation.

    This function computes the covariance matrix of moment conditions
    used in the generalized method of moments (GMM) estimation of CBPS.
    The matrix structure differs between ATE and ATT estimation.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix, shape (n, k).
    probs_curr : np.ndarray
        Current propensity score estimates, shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.
    n : int
        Number of observations.

    Returns
    -------
    np.ndarray
        The Moore-Penrose pseudoinverse of the covariance matrix V,
        shape (2k, 2k) where k is the number of covariates.

    Notes
    -----
    The V matrix has a 2x2 block structure combining score and balance
    moment conditions. For ATT estimation, the matrix includes scaling
    factors involving the ratio of treated to total observations.
    """
    sw_sqrt = np.sqrt(sample_weights)
    
    if att:
        # ATT: weighted covariate matrices with propensity score scaling
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs_curr * (1 - probs_curr))[:, None]
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs_curr / (1 - probs_curr))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs_curr)[:, None]
        
        # Block covariance matrix with ATT scaling factors
        n_treat = np.sum(treat == 1)
        V11 = (1 / n) * (X_1.T @ X_1) * n / n_treat
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * n / n_treat
        V21 = V12  # Symmetric
        V22 = (1 / n) * (X_2.T @ X_2) * n**2 / n_treat**2
    else:
        # ATE: weighted covariate matrices
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs_curr * (1 - probs_curr))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs_curr * (1 - probs_curr))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        # Block covariance matrix without scaling
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V21 = V12  # Symmetric
        V22 = (1 / n) * (X_2.T @ X_2)
    
    # Assemble 2x2 block matrix
    V = np.block([[V11, V12],
                  [V21, V22]])
    
    # Compute Moore-Penrose pseudoinverse
    inv_V = _r_ginv(V)
    
    # Verify symmetry of covariance matrix
    assert np.allclose(V, V.T, atol=1e-15), "V matrix must be symmetric"
    
    return inv_V


def _gmm_func(
    beta_curr: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    inv_V: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute the GMM objective function and covariance matrix.

    This function evaluates the generalized method of moments objective
    combining score conditions and covariate balancing conditions.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient vector, shape (k,).
    inv_V : np.ndarray or None
        Precomputed inverse covariance matrix. If None, it will be computed.

    Returns
    -------
    dict
        Dictionary containing:
        - 'loss': float, GMM loss (quadratic form gbar' @ inv_V @ gbar)
        - 'inv_V': ndarray, pseudoinverse of the covariance matrix V

    Notes
    -----
    When two_step=True, the inverse covariance matrix is precomputed and
    passed in; when two_step=False, it is recomputed at each iteration.
    """
    n = len(treat)
    
    # Compute propensity scores
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)
    
    # Clip probabilities for numerical stability
    probs_curr = np.minimum(1 - PROBS_MIN, probs_curr)
    probs_curr = np.maximum(PROBS_MIN, probs_curr)
    probs_curr = probs_curr.ravel()
    
    # Compute weights based on estimand type
    if att:
        w_curr = _att_wt_func(beta_curr, X, treat, sample_weights)
    else:
        # ATE weight: 1 / (pi - 1 + T) = T/pi - (1-T)/(1-pi)
        w_curr = 1 / (probs_curr - 1 + treat)
    
    # Construct moment conditions
    # Balance condition: weighted covariate means
    w_curr_del = (1 / n) * (sample_weights[:, None] * X).T @ w_curr
    w_curr_del = w_curr_del.ravel()
    
    # Combine score and balance conditions
    score_cond = (1 / n) * (sample_weights[:, None] * X).T @ (treat - probs_curr)
    gbar = np.concatenate([score_cond.ravel(), w_curr_del])
    
    # Compute covariance matrix if not provided
    if inv_V is None:
        inv_V = _compute_V_matrix(X, probs_curr, sample_weights, treat, att, n)
    
    # GMM loss: quadratic form
    loss = float(gbar.T @ inv_V @ gbar)
    
    return {'loss': loss, 'inv_V': inv_V}


def _gmm_loss(
    beta: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    inv_V: Optional[np.ndarray]
) -> float:
    """
    Compute the GMM objective function value.

    This function evaluates the generalized method of moments objective
    function that combines the propensity score likelihood and covariate
    balancing conditions.

    Parameters
    ----------
    beta : np.ndarray
        Coefficient vector, shape (k,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.
    inv_V : np.ndarray, optional
        Precomputed inverse covariance matrix. If None, computes it.

    Returns
    -------
    float
        The GMM objective function value.
    """
    return _gmm_func(beta, X, treat, sample_weights, att, inv_V)['loss']


def _gmm_gradient(
    beta_curr: np.ndarray,
    inv_V: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int
) -> np.ndarray:
    """
    Compute the analytical gradient of the GMM objective function.

    This function calculates the analytical gradient of the GMM objective
    with respect to the coefficient vector, following the R CBPS package
    implementation exactly.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates, shape (k,).
    inv_V : np.ndarray
        Inverse covariance matrix, shape (2k, 2k).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.

    Returns
    -------
    np.ndarray
        Gradient vector, shape (k,).

    Notes
    -----
    The gradient is computed as: grad = 2 * dgbar @ inv_V @ gbar
    
    where dgbar is the Jacobian of the moment conditions gbar with respect
    to beta. The formula differs between ATE and ATT estimation.

    For ATE:
        dgbar = [-1/n * X' * diag(sw * pi * (1-pi)) * X,
                 -1/n * X' * diag(sw * (T-pi)^2 / (pi*(1-pi))) * X]
    
    For ATT:
        dw = -n/n_t * pi / (1-pi), with dw[treat==1] = 0
        dgbar = [1/n * X' * diag(-sw * pi * (1-pi)) * X,
                 1/n * X' * diag(dw * sw) * X]
    
    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    n = len(treat)
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    
    # Compute propensity scores
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)
    probs_curr = np.clip(probs_curr, PROBS_MIN, 1 - PROBS_MIN)
    
    # Compute weights based on estimand type
    if att:
        w_curr = _att_wt_func(beta_curr, X, treat, sample_weights)
    else:
        w_curr = 1 / (probs_curr - 1 + treat)
    
    # Compute gbar (moment conditions)
    w_curr_del = (1 / n) * (sample_weights[:, None] * X).T @ w_curr
    w_curr_del = w_curr_del.ravel()
    score_cond = (1 / n) * (sample_weights[:, None] * X).T @ (treat - probs_curr)
    gbar = np.concatenate([score_cond.ravel(), w_curr_del])
    
    # Compute dgbar (Jacobian of moment conditions)
    if att:
        # ATT gradient formula
        # 
        # The ATT weight function is: w = (n/n_t) * (T - pi) / (1 - pi)
        # For T=0 (control): w = -n/n_t * pi / (1-pi)
        # For T=1 (treated): w = n/n_t
        #
        # dw/d_beta for T=0:
        #   w = -n/n_t * pi / (1-pi)
        #   dw/d_pi = -n/n_t * [(1-pi) + pi] / (1-pi)^2 = -n/n_t / (1-pi)^2
        #   d_pi/d_beta = pi * (1-pi) * X
        #   dw/d_beta = dw/d_pi * d_pi/d_beta = -n/n_t * pi / (1-pi) * X
        #
        # For T=1: dw/d_beta = 0 (constant)
        dw = -n / n_t * probs_curr / (1 - probs_curr)
        dw[treat == 1] = 0
        
        # Score condition derivative: 1/n * X' * diag(-sw * pi * (1-pi)) * X
        dgbar_score = (1 / n) * (X * (-sample_weights * probs_curr * (1 - probs_curr))[:, None]).T @ X
        
        # Balance condition derivative: 1/n * X' * diag(dw * sw) * X
        # Note: R code uses 1/n.t here, but mathematically correct is 1/n.
        # The derivative of gbar_balance = (1/n) * X' * sw * w_ATT w.r.t. beta
        # gives (1/n) * X' * diag(sw * dw) * X. R's 1/n.t has an extra n/n_t
        # factor. BFGS is robust to such gradient scaling, so R still converges.
        # We use the mathematically correct 1/n for better numerical gradient match.
        dgbar_balance = (1 / n) * (X * (dw * sample_weights)[:, None]).T @ X
    else:
        # ATE gradient formula from R code
        # Score condition derivative: -1/n * X' * diag(sw * pi * (1-pi)) * X
        dgbar_score = (-1 / n) * (X * (sample_weights * probs_curr * (1 - probs_curr))[:, None]).T @ X
        
        # Balance condition derivative: -1/n * X' * diag(sw * (T-pi)^2 / (pi*(1-pi))) * X
        balance_weight = sample_weights * (treat - probs_curr)**2 / (probs_curr * (1 - probs_curr))
        dgbar_balance = (-1 / n) * (X * balance_weight[:, None]).T @ X
    
    # Combine into full Jacobian: dgbar has shape (k, 2k)
    dgbar = np.hstack([dgbar_score, dgbar_balance])
    
    # Compute gradient: 2 * dgbar @ inv_V @ gbar
    grad = 2 * dgbar @ inv_V @ gbar
    
    return grad


def _bal_loss(
    beta_curr: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    XprimeX_inv: np.ndarray,
    att: int
) -> float:
    """
    Balance loss function (covariate balancing only).

    This function implements the balance component of the CBPS objective
    function, focusing solely on achieving covariate balance between
    treatment groups without considering prediction of treatment assignment.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates, shape (k,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    XprimeX_inv : np.ndarray
        Inverse of X'X matrix, pre-computed for efficiency, shape (k, k).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.

    Returns
    -------
    float
        Balance loss value (absolute quadratic form).

    Notes
    -----
    Key differences between balance loss and GMM loss:
    
    - Balance loss uses absolute value: |ḡ' (X'WX)^{-1} ḡ| where ḡ = X'Ww
    - GMM loss uses quadratic form without absolute value: ḡ' Σ^{-1} ḡ
    - Weight computation includes 1/n scaling factor
    
    Here W = diag(sample_weights) is the sample weight matrix.
    """
    n = len(treat)
    
    # Compute propensity scores with numerical clipping
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)
    probs_curr = np.clip(probs_curr, PROBS_MIN, 1 - PROBS_MIN)

    # Compute weights with 1/n scaling factor
    if att:
        w_curr = (1 / n) * _att_wt_func(beta_curr, X, treat, sample_weights)
    else:
        w_curr = (1 / n) * (1 / (probs_curr - 1 + treat))

    # Compute weighted covariate sum
    Xprimew = (sample_weights[:, None] * X).T @ w_curr  # (k,) vector

    # Balance loss: quadratic form with absolute value
    loss = np.abs(Xprimew.T @ XprimeX_inv @ Xprimew)
    
    return float(loss)


def _bal_gradient(
    beta_curr: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    XprimeX_inv: np.ndarray,
    att: int
) -> np.ndarray:
    """
    Analytical gradient of the balance loss function.

    This function computes the analytical gradient of the balance component
    of the CBPS objective function, following the R CBPS package implementation
    exactly. The use of analytical gradient is critical because the balance
    loss contains an absolute value function.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates, shape (k,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    XprimeX_inv : np.ndarray
        Inverse of X'X matrix, shape (k, k).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.

    Returns
    -------
    np.ndarray
        Gradient vector, shape (k,).
    
    Notes
    -----
    The R implementation uses a sign adjustment to handle the absolute value:
    
        out = sapply(2*dw%*%X%*%XprimeX.inv%*%Xprimew, 
                     function(x) ifelse((x>0 & loss1>0) | (x<0 & loss1<0), 
                                        abs(x), -abs(x)))
    
    This ensures the gradient points in the correct direction for minimizing
    the absolute value of the quadratic form.

    For ATE:
        dw = 1/n * t(-X * (T-pi)^2 / (pi*(1-pi)))
    
    For ATT:
        dw2 = -n/n_t * pi / (1-pi), with dw2[treat==1] = 0
        dw = 1/n * t(X * dw2)

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    n = len(treat)
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    
    # Compute propensity scores
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)
    probs_curr = np.clip(probs_curr, PROBS_MIN, 1 - PROBS_MIN)
    
    # Compute weights with 1/n scaling factor
    if att:
        w_curr = (1 / n) * _att_wt_func(beta_curr, X, treat, sample_weights)
    else:
        w_curr = (1 / n) * (1 / (probs_curr - 1 + treat))
    
    # Compute dw (derivative of weights with respect to beta)
    if att:
        # ATT: dw2 = -n/n_t * pi / (1-pi), with dw2[treat==1] = 0
        dw2 = -n / n_t * probs_curr / (1 - probs_curr)
        dw2[treat == 1] = 0
        # dw has shape (k, n): dw = 1/n * X.T * dw2
        dw = (1 / n) * (X * dw2[:, None]).T
    else:
        # ATE: dw = 1/n * t(-X * (T-pi)^2 / (pi*(1-pi)))
        dw_weight = -(treat - probs_curr)**2 / (probs_curr * (1 - probs_curr))
        dw = (1 / n) * (X * dw_weight[:, None]).T
    
    # Compute Xprimew = X' @ (w_curr * sample_weights)
    Xprimew = X.T @ (w_curr * sample_weights)  # shape (k,)
    
    # Compute loss1 = Xprimew' @ XprimeX_inv @ Xprimew (scalar)
    loss1 = Xprimew.T @ XprimeX_inv @ Xprimew
    
    # Compute raw gradient: 2 * dw @ X @ XprimeX_inv @ Xprimew
    # dw has shape (k, n), X has shape (n, k)
    # dw @ X has shape (k, k)
    # (dw @ X @ XprimeX_inv @ Xprimew) has shape (k,)
    raw_grad = 2 * dw @ X @ XprimeX_inv @ Xprimew
    
    # Apply sign adjustment for absolute value (R's sapply logic)
    # ifelse((x>0 & loss1>0) | (x<0 & loss1<0), abs(x), -abs(x))
    # This means: if x and loss1 have the same sign, use abs(x); otherwise use -abs(x)
    grad = np.where(
        ((raw_grad > 0) & (loss1 > 0)) | ((raw_grad < 0) & (loss1 < 0)),
        np.abs(raw_grad),
        -np.abs(raw_grad)
    )
    
    return grad

def _vmmin_bfgs(
    b0: np.ndarray,
    fn: Callable,
    gr: Optional[Callable],
    maxit: int = 10000,
    abstol: float = -np.inf,
    reltol: float = np.sqrt(np.finfo(float).eps),
    trace: bool = False,
    nREPORT: int = 10,
) -> scipy.optimize.OptimizeResult:
    """
    R's vmmin BFGS optimizer, faithfully translated from C source.

    This is a line-by-line translation of R's ``vmmin`` function from
    ``src/appl/optim.c`` (the backend of ``optim(..., method="BFGS")``).
    It uses a simple Armijo backtracking line search and a relative-
    tolerance convergence criterion, which differ fundamentally from
    scipy's Strong-Wolfe / gradient-norm approach.

    Parameters
    ----------
    b0 : np.ndarray
        Initial parameter vector, shape (n,).
    fn : callable
        Objective function ``fn(b) -> float``.
    gr : callable or None
        Gradient function ``gr(b) -> np.ndarray`` of shape (n,).
        If None, uses forward finite differences with step size 1e-3,
        matching R's ``optim`` default behavior (``fmingr`` in optim.c).
    maxit : int
        Maximum number of BFGS iterations (default 10000, R default).
    abstol : float
        Absolute tolerance on function value (default -inf, R default).
    reltol : float
        Relative tolerance on function value change
        (default ``sqrt(eps) ≈ 1.49e-8``, R default).
    trace : bool
        If True, print iteration information (default False).
    nREPORT : int
        Report every *nREPORT* iterations when *trace* is True.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result object with fields ``x``, ``fun``, ``nit``, ``nfev``,
        ``njev``, ``success``, ``message``.

    Notes
    -----
    Constants hard-coded to match R exactly:

    * ``stepredn = 0.2``  – step reduction factor in backtracking
    * ``acctol   = 0.0001`` – Armijo sufficient-decrease parameter
    * ``reltest  = 10.0``  – used to detect "no change" in parameters

    Convergence criterion (R ``reltol``):
        ``|f_new - f_old| > reltol * (|f_old| + reltol)``

    References
    ----------
    J.C. Nash, *Compact Numerical Methods for Computers*, 2nd ed.
    R Core Team, ``src/appl/optim.c`` (vmmin).
    """
    # If no analytical gradient provided, use R's default numerical gradient.
    # R's fmingr in optim.c uses forward finite differences with ndeps=1e-3.
    if gr is None:
        _ndeps = 1e-3
        def gr(b):
            """Forward finite difference gradient, matching R's fmingr."""
            f0 = fn(b)
            g = np.empty_like(b)
            for i in range(len(b)):
                b_pert = b.copy()
                b_pert[i] += _ndeps
                g[i] = (fn(b_pert) - f0) / _ndeps
            return g
    # ---- constants (must match R exactly) ----
    STEPREDN = 0.2
    ACCTOL = 0.0001
    RELTEST = 10.0

    n = len(b0)
    b = b0.astype(float).copy()

    if maxit <= 0:
        f = fn(b)
        return scipy.optimize.OptimizeResult(
            x=b, fun=f, nit=0, nfev=1, njev=0,
            success=True, message="maxit <= 0, returning initial value",
        )

    # All parameters are free (mask = all True).
    # In R, l[] maps free-parameter indices; here every index is free,
    # so l[i] = i and the indirection is a no-op.

    # ---- allocate working arrays ----
    g = np.empty(n)          # gradient
    t = np.empty(n)          # search direction
    X = np.empty(n)          # saved parameters
    c = np.empty(n)          # saved gradient
    # B: lower-triangular BFGS Hessian-inverse approximation
    # B[i][j] stored for j <= i  (symmetric, only lower triangle kept)
    B = np.zeros((n, n))

    # ---- initial evaluation ----
    f = fn(b)
    if not np.isfinite(f):
        raise ValueError("initial value in vmmin is not finite")
    if trace:
        print(f"initial  value {f}")
    Fmin = f
    funcount = 1
    gradcount = 1
    g[:] = gr(b)
    iter_ = 1
    ilast = gradcount

    while True:
        # ---- Hessian reset when needed ----
        if ilast == gradcount:
            B[:, :] = 0.0
            for i in range(n):
                B[i, i] = 1.0

        # ---- save current state ----
        X[:] = b
        c[:] = g

        # ---- compute search direction  t = -B g ----
        gradproj = 0.0
        for i in range(n):
            s = 0.0
            for j in range(i + 1):        # j <= i
                s -= B[i, j] * g[j]
            for j in range(i + 1, n):      # j > i  (symmetric access)
                s -= B[j, i] * g[j]
            t[i] = s
            gradproj += s * g[i]

        if gradproj < 0.0:
            # ---- downhill: backtracking line search ----
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if RELTEST + X[i] == RELTEST + b[i]:
                        count += 1
                if count < n:
                    f = fn(b)
                    funcount += 1
                    accpoint = (
                        np.isfinite(f)
                        and f <= Fmin + gradproj * steplength * ACCTOL
                    )
                    if not accpoint:
                        steplength *= STEPREDN
                if count == n or accpoint:
                    break

            enough = (
                f > abstol
                and abs(f - Fmin) > reltol * (abs(Fmin) + reltol)
            )
            if not enough:
                count = n
                Fmin = f

            if count < n:
                # ---- making progress ----
                Fmin = f
                g[:] = gr(b)
                gradcount += 1
                iter_ += 1

                # prepare for BFGS update
                for i in range(n):
                    t[i] = steplength * t[i]   # actual step
                    c[i] = g[i] - c[i]         # gradient change
                D1 = float(t @ c)

                if D1 > 0:
                    # ---- BFGS Hessian-inverse update ----
                    # Compute X_tmp = B @ c  (reuse X array as in R)
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(i + 1):
                            s += B[i, j] * c[j]
                        for j in range(i + 1, n):
                            s += B[j, i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1

                    for i in range(n):
                        for j in range(i + 1):
                            B[i, j] += (
                                D2 * t[i] * t[j]
                                - X[i] * t[j]
                                - t[i] * X[j]
                            ) / D1
                else:
                    # D1 <= 0: curvature condition violated → reset
                    ilast = gradcount
            else:
                # ---- no progress ----
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:
            # ---- uphill search direction ----
            count = 0
            if ilast == gradcount:
                count = n          # already reset → give up
            else:
                ilast = gradcount  # reset Hessian

        if trace and (iter_ % nREPORT == 0):
            print(f"iter{iter_:4d} value {f}")

        if iter_ >= maxit:
            break

        # ---- periodic restart ----
        if gradcount - ilast > 2 * n:
            ilast = gradcount

        if count == n and ilast == gradcount:
            break

    if trace:
        print(f"final  value {Fmin}")
        if iter_ < maxit:
            print("converged")
        else:
            print(f"stopped after {iter_} iterations")

    success = iter_ < maxit
    return scipy.optimize.OptimizeResult(
        x=b,
        fun=Fmin,
        nit=iter_,
        nfev=funcount,
        njev=gradcount,
        success=success,
        message="converged" if success else f"stopped after {iter_} iterations",
    )



def _glm_init(
    treat: np.ndarray,
    X: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    gmm_loss_func: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize GLM coefficients through six-step optimization.

    This function computes initial values for the CBPS optimization by
    fitting a standard GLM model and then optimizing the scaling factor
    alpha to minimize the GMM loss function.

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    sample_weights : np.ndarray
        Normalized sampling weights, shape (n,).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.
    gmm_loss_func : Callable
        GMM loss function for alpha scaling optimization.
    
    Returns
    -------
    beta_init : np.ndarray
        Initial coefficients after GLM fitting and alpha scaling.
    beta_glm : np.ndarray
        Original GLM coefficients, used for computing MLE J-statistic.

    Notes
    -----
    Six-step initialization process:
    1. Fit GLM model with warnings suppressed
    2. Set NA coefficients to 0 (first pass)
    3. Sequential probability clipping
    4. Extract coefficients and handle NA (second pass)
    5. Optimize alpha scaling factor in [0.8, 1.1]
    """
    # Step 1: GLM fitting with warnings suppressed
    # Note: GLM doesn't use weights parameter here; sample_weights are used only in GMM steps.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(treat, X, family=Binomial())
        glm_fit = model.fit(tol=1e-8, maxiter=25)  # Standard IRLS algorithm

        # When the design matrix induces quasi-complete separation (a
        # common situation for highly expressive specifications such as
        # polynomial expansions on structurally different groups), the
        # unpenalized IRLS step can fail to converge and returns
        # coefficients whose fitted probabilities are all pinned near 0/1.
        # Such a warm start drives the downstream GMM optimization to a
        # degenerate basin. As a defensive fallback we recompute the
        # initial coefficients with a vanishingly small L2 penalty
        # (``alpha=1e-6``), which is numerically indistinguishable from
        # the unpenalized MLE in well-posed cases but remains well
        # defined under separation. This only affects the initial values
        # used to seed the GMM; the CBPS estimator itself is unchanged.
        beta_candidate = np.asarray(glm_fit.params, dtype=float)
        probs_candidate = np.asarray(glm_fit.fittedvalues, dtype=float)
        extreme_ratio = float(np.mean(
            (probs_candidate < 1e-3) | (probs_candidate > 1 - 1e-3)
        ))
        iteration_failed = (
            not getattr(glm_fit, "converged", True)
            or not np.all(np.isfinite(beta_candidate))
            or extreme_ratio > 0.5
        )
        if iteration_failed:
            try:
                reg_fit = sm.GLM(treat, X, family=Binomial()).fit_regularized(
                    alpha=1e-6, L1_wt=0.0, maxiter=200
                )
                reg_params = np.asarray(reg_fit.params, dtype=float)
                if np.all(np.isfinite(reg_params)):
                    glm_fit = reg_fit  # adopt ridge-penalized warm start
            except Exception:
                # If even the penalized IRLS fails, keep the original fit
                # and let the downstream NaN-handling / alpha scaling deal
                # with it.
                pass

    # Step 2: Handle NA coefficients (first pass)
    beta_glm = np.asarray(glm_fit.params, dtype=float).copy()
    beta_glm[np.isnan(beta_glm)] = 0

    # Step 3: Probability clipping
    # ``fit_regularized`` returns a GLMResultsWrapper without ``fittedvalues``
    # on some statsmodels versions; fall back to ``predict`` when needed.
    if hasattr(glm_fit, "fittedvalues"):
        probs_glm = np.asarray(glm_fit.fittedvalues, dtype=float).copy()
    else:
        probs_glm = np.asarray(glm_fit.predict(X), dtype=float).copy()
    probs_glm = np.clip(probs_glm, PROBS_MIN, 1 - PROBS_MIN)

    # Step 4: Extract coefficients and handle NA (second pass)
    beta_curr = beta_glm.copy()
    beta_curr[np.isnan(beta_curr)] = 0

    # Step 5: Alpha scaling optimization (1D search for optimal scaling factor)
    alpha_func = lambda alpha: gmm_loss_func(beta_curr * alpha)
    result = scipy.optimize.minimize_scalar(
        alpha_func, bounds=(0.8, 1.1), method='bounded'
    )
    beta_curr = beta_curr * result.x

    # Return: scaled coefficients and original GLM coefficients (for MLE J-statistic)
    return beta_curr, beta_glm


def _compute_moment_conditions(
    beta: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    n: int
) -> np.ndarray:
    """
    Compute CBPS moment conditions (covariate balance conditions).

    Implements the moment conditions from Imai & Ratkovic (2014) JRSS-B:
    - Equation (10): ATE balance condition
    - Equation (11): ATT balance condition

    Parameters
    ----------
    beta : np.ndarray
        Coefficient vector, shape (k,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment vector (0/1), shape (n,).
    sample_weights : np.ndarray
        Sample weights, shape (n,).
    att : int
        Estimand: 0=ATE, 1=ATT (T=1 is treated), 2=ATT (T=0 is treated).
    n : int
        Sample size.

    Returns
    -------
    np.ndarray
        k-dimensional moment condition vector.
        For just-identified GMM: moments should be approximately zero.

    Notes
    -----
    This is the core of just-identified GMM: k equations for k unknowns.
    The theoretical requirement is to solve moments = 0 directly.
    """
    theta = X @ beta
    pi = scipy.special.expit(theta)
    pi = np.clip(pi, PROBS_MIN, 1 - PROBS_MIN)
    
    # Compute weights based on estimand (Equations 10/11 in the paper)
    if att == 1:
        # ATT Equation (11): w = (n/n_1) * (T - pi) / (1 - pi)
        n_treated = np.sum(treat * sample_weights)
        w = (n / n_treated) * (treat - pi) / (1 - pi)
    elif att == 2:
        # ATT with reversed treatment (T=0 is treated)
        n_control = np.sum((1 - treat) * sample_weights)
        w = (n / n_control) * (treat - pi) / pi
    else:
        # ATE Equation (10): w = (T - pi) / (pi * (1 - pi))
        w = (treat - pi) / (pi * (1 - pi))
    
    # Moment conditions (covariate balance)
    moments = (sample_weights[:, None] * X).T @ w / n
    
    return moments


def _solve_moment_equations(
    beta_init: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    n: int,
    iterations: int = 1000
) -> Tuple[np.ndarray, bool, np.ndarray, str]:
    """
    Solve moment equations directly (theoretically correct just-identified GMM).

    This implementation follows the GMM framework:
    - Hansen (1982) GMM: Just-identified = solve E[g(X, theta)] = 0
    - Imai & Ratkovic (2014) Equations (10)/(11): Balance conditions

    Parameters
    ----------
    beta_init : np.ndarray
        Initial values (from GLM or balance optimization), shape (k,).

    Returns
    -------
    beta_opt : np.ndarray
        Optimal coefficients satisfying moment = 0.
    success : bool
        Whether convergence was achieved.
    moments_final : np.ndarray
        Final moment values (should be approximately zero).
    method : str
        Solver method used.

    Notes
    -----
    Advantages over balance loss optimization:
    1. Theoretically correct: directly corresponds to just-identified GMM
    2. Numerical precision: can achieve machine precision (~1e-15)
    3. Computational efficiency: typically faster

    Solver strategy:
    1. First try 'hybr' (hybrid Powell, robust and fast)
    2. Fall back to 'lm' (Levenberg-Marquardt, more robust but slower)
    3. If both fail, return failure status
    """
    from scipy.optimize import root
    
    def moment_eq(beta):
        """Moment equations: k equations for k unknowns."""
        return _compute_moment_conditions(beta, X, treat, sample_weights, att, n)
    
    # Primary solver: hybrid Powell method (fast and robust)
    result = root(
        moment_eq,
        x0=beta_init,
        method='hybr',
        options={'xtol': 1e-12, 'maxfev': iterations * 10}
    )
    
    if result.success:
        moments_final = moment_eq(result.x)
        return result.x, True, moments_final, 'hybr'
    
    # Fallback: Levenberg-Marquardt (more robust)
    try:
        result = root(
            moment_eq,
            x0=beta_init,
            method='lm',
            options={'xtol': 1e-12, 'maxiter': iterations * 5}
        )
        
        if result.success:
            moments_final = moment_eq(result.x)
            return result.x, True, moments_final, 'lm'
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    
    # Both solvers failed: return initial values with failure status
    moments_final = moment_eq(beta_init)
    return beta_init, False, moments_final, 'failed'


def _optimize_balance(
    gmm_init: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    XprimeX_inv: np.ndarray,
    att: int,
    two_step: bool,
    iterations: int,
    bal_only: bool = False,
    **kwargs
) -> scipy.optimize.OptimizeResult:
    """
    Optimize balance loss to find initial values for GMM.

    Uses R's vmmin BFGS algorithm (simple Armijo backtracking line search
    with reltol convergence) to exactly replicate R CBPS package behavior.

    Parameters
    ----------
    gmm_init : np.ndarray
        GLM-initialized coefficients, shape (k,).
    bal_only : bool
        Whether this is just-identified mode (method='exact').
    **kwargs
        Additional arguments passed through from CBPS wrapper.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Balance optimization result object.

    Notes
    -----
    The analytical gradient is required for reliable optimization because
    the balance loss function contains an absolute value, which has
    discontinuous derivatives at zero. Numerical gradients perform poorly
    in this case.
    """
    bal_loss_func = lambda b: _bal_loss(b, X, treat, sample_weights, XprimeX_inv, att)
    bal_grad_func = lambda b: _bal_gradient(b, X, treat, sample_weights, XprimeX_inv, att)

    verbose = kwargs.get('verbose', False)

    # R CBPS package only provides analytical gradient for balance optimization
    # when twostep=TRUE. For continuous updating (twostep=FALSE), R uses
    # numerical gradients (finite differences via optim's default behavior).
    gr_func = bal_grad_func if two_step else None

    # Use R's vmmin BFGS (faithful translation of R's optim(..., method="BFGS"))
    # This ensures identical convergence behavior: simple Armijo backtracking
    # line search + reltol convergence criterion.
    opt_bal = _vmmin_bfgs(
        gmm_init,
        fn=bal_loss_func,
        gr=gr_func,
        maxit=iterations,
        trace=verbose,
    )

    return opt_bal


def _optimize_gmm_dual_init(
    gmm_init: np.ndarray,
    beta_bal: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    this_inv_V: np.ndarray,
    two_step: bool,
    iterations: int,
    **kwargs
) -> scipy.optimize.OptimizeResult:
    """
    Perform GMM optimization with dual initialization strategy.

    Runs GMM optimization from two starting points (GLM-initialized and
    balance-optimized) and returns the result with lower objective value.
    Uses R's vmmin BFGS algorithm for exact replication.

    Parameters
    ----------
    gmm_init : np.ndarray
        GLM-initialized coefficients (after alpha scaling), shape (k,).
    beta_bal : np.ndarray
        Balance-optimized coefficients, shape (k,).
    this_inv_V : np.ndarray
        Precomputed inverse covariance matrix (for two-step GMM).

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result with lower objective value.

    Notes
    -----
    The dual initialization strategy improves robustness by exploring
    different regions of the parameter space. When two_step=True, analytical
    gradients are used following the R CBPS package implementation.
    """
    verbose = kwargs.get('verbose', False)

    if two_step:
        # Two-step GMM optimization using analytical gradients (R-compatible)
        def gmm_loss_with_inv_V(b):
            return _gmm_loss(b, X, treat, sample_weights, att, this_inv_V)

        def gmm_grad_with_inv_V(b):
            return _gmm_gradient(b, this_inv_V, X, treat, sample_weights, att)

        gmm_glm_init = _vmmin_bfgs(
            gmm_init,
            fn=gmm_loss_with_inv_V,
            gr=gmm_grad_with_inv_V,
            maxit=iterations,
            trace=verbose,
        )
        gmm_bal_init = _vmmin_bfgs(
            beta_bal,
            fn=gmm_loss_with_inv_V,
            gr=gmm_grad_with_inv_V,
            maxit=iterations,
            trace=verbose,
        )
    else:
        # Continuous updating GMM optimization
        # R CBPS package does NOT provide analytical gradients for continuous
        # updating (twostep=FALSE). It relies on numerical differentiation
        # via optim's default finite-difference method. This is because
        # _gmm_gradient treats inv_V as fixed, which is only valid for
        # two-step GMM where V is pre-computed. In continuous updating,
        # V is recomputed at each iteration, making the fixed-V gradient
        # only an approximation.
        def gmm_loss_continuous(b):
            return _gmm_loss(b, X, treat, sample_weights, att, None)

        gmm_glm_init = _vmmin_bfgs(
            gmm_init,
            fn=gmm_loss_continuous,
            gr=None,
            maxit=iterations,
            trace=verbose,
        )
        gmm_bal_init = _vmmin_bfgs(
            beta_bal,
            fn=gmm_loss_continuous,
            gr=None,
            maxit=iterations,
            trace=verbose,
        )

    # Return the result with lower objective value
    if gmm_glm_init.fun < gmm_bal_init.fun:
        return gmm_glm_init
    else:
        return gmm_bal_init


def _compute_final_weights(
    beta_opt: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    standardize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute final propensity scores and inverse probability weights.

    Parameters
    ----------
    beta_opt : np.ndarray
        Optimized coefficient vector, shape (k,).
    X : np.ndarray
        Covariate matrix, shape (n, k).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    sample_weights : np.ndarray
        Sampling weights, shape (n,).
    att : int
        Estimand type: 0 for ATE, 1 for ATT.
    standardize : bool
        Whether to normalize weights to sum to sample size.

    Returns
    -------
    probs_opt : np.ndarray
        Final propensity scores, shape (n,).
    w_opt : np.ndarray
        Final inverse probability weights (standardized and incorporating
        sample_weights), shape (n,).

    Notes
    -----
    The weight computation follows these steps:
    1. Compute propensity scores from optimized coefficients
    2. Compute initial IPW weights (ATT or ATE formula)
    3. Standardize weights if requested
    4. Incorporate sampling weights
    """
    # Compute propensity scores from optimized coefficients
    theta_opt = X @ beta_opt
    probs_opt_raw = scipy.special.expit(theta_opt)
    probs_opt = np.clip(probs_opt_raw, PROBS_MIN, 1 - PROBS_MIN)
    
    # Detect separation issues (propensity scores at boundaries)
    n_clipped_low = np.sum(probs_opt_raw <= PROBS_MIN)
    n_clipped_high = np.sum(probs_opt_raw >= 1 - PROBS_MIN)
    n_boundary = n_clipped_low + n_clipped_high
    
    if n_boundary > 0:
        import warnings
        boundary_pct = 100.0 * n_boundary / len(treat)
        
        # Check for extreme coefficients (may indicate separation)
        extreme_coef_threshold = 10.0
        extreme_coef_mask = np.abs(beta_opt) > extreme_coef_threshold
        has_extreme_coefs = np.any(extreme_coef_mask)
        
        # Issue graduated warnings based on severity
        if boundary_pct >= 100.0:
            # Complete separation: highest level warning
            msg = (
                f"\n{'='*70}\n"
                f" COMPLETE SEPARATION DETECTED - RESULTS UNRELIABLE \n"
                f"{'='*70}\n"
                f"All {len(treat)} observations have fitted propensity scores at boundaries:\n"
                f"  • {n_clipped_low} observations with π ≤ 1e-6 (predicted control)\n"
                f"  • {n_clipped_high} observations with π ≥ 1-1e-6 (predicted treated)\n"
                f"\nThis indicates covariates PERFECTLY predict treatment assignment,\n"
                f"violating the positivity assumption: 0 < P(T=1|X) < 1.\n"
            )
            if has_extreme_coefs:
                extreme_vals = beta_opt[extreme_coef_mask]
                extreme_indices = np.where(extreme_coef_mask)[0]
                msg += f"\nExtreme coefficients detected:\n"
                for idx, val in zip(extreme_indices, extreme_vals):
                    msg += f"  β[{idx}] = {val:.2f}\n"
            msg += (
                f"\n Inverse probability weights will be infinite or zero.\n"
                f" Treatment effect estimates will be biased and unreliable.\n"
                f"\nRecommended actions:\n"
                f"  1. Remove perfect predictors from the model\n"
                f"  2. Use penalized/regularized methods (e.g., hdCBPS with LASSO)\n"
                f"  3. Use Firth logistic regression for separation\n"
                f"  4. Re-examine data collection and study design\n"
                f"\nNote: CBPS will return results but they should NOT be used.\n"
                f"{'='*70}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            
        elif boundary_pct >= 50.0:
            # Quasi-separation: high level warning
            msg = (
                f"\n{'='*70}\n"
                f" QUASI-SEPARATION DETECTED - CAUTION ADVISED \n"
                f"{'='*70}\n"
                f"{n_boundary} observations ({boundary_pct:.1f}%) at probability boundaries:\n"
                f"  • {n_clipped_low} with π ≤ 1e-6, {n_clipped_high} with π ≥ 1-1e-6\n"
                f"\nThis suggests near-perfect prediction of treatment, which may lead to:\n"
                f"  • Extreme propensity score weights\n"
                f"  • Unstable variance estimates\n"
                f"  • Sensitivity to model specification\n"
            )
            if has_extreme_coefs:
                msg += f"\nExtreme coefficients detected (|β| > {extreme_coef_threshold}).\n"
            msg += (
                f"\nProceed with caution. Consider:\n"
                f"  1. Checking for multicollinearity or perfect predictors\n"
                f"  2. Trimming extreme propensity scores\n"
                f"  3. Using penalized methods\n"
                f"{'='*70}"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            
        elif boundary_pct >= 10.0:
            # Moderate boundary cases: general warning
            msg = (
                f"Potential separation: {n_boundary} observations ({boundary_pct:.1f}%) "
                f"have fitted propensity scores at boundaries [<1e-6 or >1-1e-6] "
                f"(low: {n_clipped_low}, high: {n_clipped_high}). "
                f"This may indicate (quasi-)separation. "
                f"Consider checking for perfect predictors or using regularization."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        else:
            # Minor boundary cases (<10%): brief warning
            msg = (
                f"{n_boundary} observations ({boundary_pct:.1f}%) at probability boundaries. "
                f"This is usually acceptable if <10%, but check covariate balance."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
    
    # Compute initial IPW weights
    if att:
        # ATT weights
        w_opt = np.abs(_att_wt_func(beta_opt, X, treat, sample_weights))
    else:
        # ATE weights
        w_opt = np.abs(1 / (probs_opt - 1 + treat))
    
    # Standardize weights and incorporate sampling weights
    w_opt = standardize_weights(w_opt, treat, probs_opt, sample_weights, att, standardize)
    
    return probs_opt, w_opt


def _compute_diagnostics(
    beta_opt: np.ndarray,
    beta_glm: np.ndarray,
    probs_opt: np.ndarray,
    treat: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    two_step: bool,
    this_inv_V: np.ndarray,
    X: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute J-statistic, deviance, and null deviance.

    Returns
    -------
    J_opt : float
        J-statistic (GMM loss, over-identification test).
    mle_J : float
        MLE baseline J (computed with GLM coefficients).
    deviance : float
        Negative 2 times weighted log-likelihood.
    nulldeviance : float
        Null model deviance (intercept-only model).

    Notes
    -----
    The J-statistic can be used to test the over-identifying restrictions
    in the GMM framework. Under the null hypothesis of correct specification,
    J ~ chi-squared with degrees of freedom equal to the number of
    over-identifying restrictions.
    """
    # Compute J-statistic based on two-step or continuous updating
    if two_step:
        J_opt = _gmm_func(beta_opt, X, treat, sample_weights, att, inv_V=this_inv_V)['loss']
    else:
        J_opt = _gmm_func(beta_opt, X, treat, sample_weights, att, inv_V=None)['loss']
    
    # Compute MLE baseline J-statistic using GLM coefficients
    if two_step:
        mle_J = _gmm_func(beta_glm, X, treat, sample_weights, att, inv_V=this_inv_V)['loss']
    else:
        mle_J = _gmm_func(beta_glm, X, treat, sample_weights, att, inv_V=None)['loss']
    
    # Deviance: negative 2 times weighted log-likelihood
    deviance = -2 * np.sum(
        treat * sample_weights * np.log(probs_opt) +
        (1 - treat) * sample_weights * np.log(1 - probs_opt)
    )
    
    # Null deviance: intercept-only model with predicted probability = sample mean
    treat_mean = np.average(treat, weights=sample_weights)
    treat_mean = np.clip(treat_mean, 1e-10, 1 - 1e-10)  # Prevent log(0)
    nulldeviance = -2 * np.sum(
        treat * sample_weights * np.log(treat_mean) +
        (1 - treat) * sample_weights * np.log(1 - treat_mean)
    )
    
    return J_opt, mle_J, deviance, nulldeviance


def _compute_vcov(
    beta_opt: np.ndarray,
    probs_opt: np.ndarray,
    treat: np.ndarray,
    X: np.ndarray,
    sample_weights: np.ndarray,
    att: int,
    bal_only: bool,
    XprimeX_inv: np.ndarray,
    this_inv_V: np.ndarray,
    two_step: bool,
    n: int
) -> np.ndarray:
    """
    Compute sandwich variance-covariance matrix.

    Returns
    -------
    np.ndarray
        Coefficient variance-covariance matrix, shape (k, k).

    Notes
    -----
    Implements the sandwich estimator (Newey & McFadden 1994, Eq. 6.17):
        Var(beta_hat) = (G'WG)^-1 G'W Omega W'G (G'WG)^-1

    Processing steps:
    1. Construct G matrix (gradients) and W1 matrix (moment conditions)
    2. Assemble G and W matrices based on identification mode
    3. Compute variance using sandwich formula
    """
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    
    # Score condition components (shared by ATT/ATE)
    XG_1 = -X * (probs_opt * (1 - probs_opt))[:, None] * sample_weights[:, None]
    XW_1 = X * (treat - probs_opt)[:, None] * np.sqrt(sample_weights)[:, None]
    
    # Balance condition components (ATT/ATE branches)
    if att:
        # ATT branch
        XW_2 = X * _att_wt_func(beta_opt, X, treat, sample_weights)[:, None] * sample_weights[:, None]
        dw2 = -n / n_t * probs_opt / (1 - probs_opt)
        dw2[treat == 1] = 0  # Zero derivative for treated units
        XG_2 = X * dw2[:, None] * sample_weights[:, None]
    else:
        # ATE branch
        XW_2 = X * (1 / (probs_opt - 1 + treat))[:, None] * np.sqrt(sample_weights)[:, None]
        XG_2 = -X * ((treat - probs_opt)**2 / (probs_opt * (1 - probs_opt)))[:, None] * sample_weights[:, None]
    
    # Assemble G and W matrices based on identification mode
    if bal_only:  # method='exact'
        # Balance conditions only
        G = (XG_2.T @ X) / n
        W1 = XW_2.T
        W = XprimeX_inv
    else:  # method='over'
        # Score + balance conditions
        G = np.hstack([(XG_1.T @ X), (XG_2.T @ X)]) / n
        W1 = np.vstack([XW_1.T, XW_2.T])
        
        # Select W matrix based on estimation method
        if two_step:
            W = this_inv_V  # Reuse precomputed
        else:
            W = _gmm_func(beta_opt, X, treat, sample_weights, att, inv_V=None)['inv_V']
    
    # Sandwich formula
    Omega = (W1 @ W1.T) / n  # Moment condition covariance
    GWG = G @ W @ G.T
    GWGinv = _r_ginv(GWG)  # Moore-Penrose pseudoinverse
    GWGinvGW = GWGinv @ G @ W
    vcov = GWGinvGW @ Omega @ GWGinvGW.T
    
    return vcov


def cbps_binary_fit(
    treat: np.ndarray,
    X: np.ndarray,
    att: int = 1,
    method: str = 'over',
    two_step: bool = True,
    standardize: bool = True,
    sample_weights: Optional[np.ndarray] = None,
    iterations: int = 1000,
    XprimeX_inv: Optional[np.ndarray] = None,
    verbose: int = 0,
    **kwargs
) -> Dict:
    """
    Estimate covariate balancing propensity scores for binary treatments.

    Implements the covariate balancing propensity score (CBPS) methodology
    for binary treatment assignments using generalized method of moments
    (GMM) estimation. The function simultaneously optimizes covariate balance
    and prediction of treatment assignment.

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator vector coded as 0/1, shape (n,).
    X : np.ndarray
        Covariate matrix including intercept column, shape (n, k).
        The intercept should be the first column.
    att : int, default 1
        Target estimand for estimation:
        - 0: Average treatment effect (ATE)
        - 1: Average treatment effect on the treated (ATT) with treatment=1
        - 2: Average treatment effect on the treated (ATT) with treatment=0
    method : {'over', 'exact'}, default 'over'
        GMM estimation method:
        - 'over': Over-identified GMM combining likelihood and balance conditions
        - 'exact': Exactly-identified GMM using balance conditions only
    two_step : bool, default True
        GMM estimator type:
        - True: Two-step GMM with pre-computed weight matrix (faster)
        - False: Continuous-updating GMM with iterative weight updates
    standardize : bool, default True
        Weight standardization:
        - True: Weights sum to 1 within each treatment group
        - False: Return Horvitz-Thompson weights
    sample_weights : np.ndarray, optional
        Sampling weights for observations. If None, defaults to equal weights.
    iterations : int, default 1000
        Maximum number of iterations for the optimization algorithm.
    XprimeX_inv : np.ndarray, optional
        Pre-computed inverse of X'X matrix for balance loss computation.
    theoretical_exact : bool, default False (passed via **kwargs)
        Only applicable when method='exact':
        - True: Direct equation solver for moment conditions (precision ~1e-15)
        - False: Balance loss optimization (R-compatible, precision ~1e-6)
    **kwargs
        Additional arguments passed to scipy.optimize.minimize.

    Returns
    -------
    dict
        Fitted CBPS object containing:
        - coefficients: Estimated propensity score coefficients, shape (k, 1)
        - fitted_values: Estimated propensity scores, shape (n,)
        - weights: CBPS weights for causal effect estimation, shape (n,)
        - J: J-statistic for overidentification test
        - var: Asymptotic variance-covariance matrix, shape (k, k)
        - converged: Boolean convergence indicator
        - mle_J: Maximum likelihood J-statistic
        - deviance: Model deviance
        - linear_predictor: Linear predictor values (X @ coefficients)
        - y: Treatment indicator vector
        - x: Covariate matrix

    Notes
    -----
    The algorithm implements the following key steps:
    1. Initial MLE estimation for starting values
    2. Balance loss optimization for initial GMM values
    3. GMM optimization to satisfy both score and balance conditions
    4. Final weight computation and diagnostics

    For ATT estimation, weights are constructed to balance covariates between
    the treated group and the weighted control group. For ATE estimation,
    weights balance all groups simultaneously.

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    https://doi.org/10.1111/rssb.12027
    """
    n = len(treat)
    bal_only = (method == 'exact')
    
    # Note: SVD preprocessing is applied in the CBPS() main function before
    # calling this function, matching R package's CBPSMain.R behavior.
    # The X passed here may already be SVD-transformed (U matrix).
    X_orig = X.copy()
    
    # Full rank check
    k = np.linalg.matrix_rank(X)
    if k < X.shape[1]:
        raise ValueError(f"X is not full rank: rank={k} < ncol={X.shape[1]}")
    
    # Step 1: Normalize sample weights
    sample_weights = normalize_sample_weights(sample_weights, n)
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    
    # Compute XprimeX_inv
    if XprimeX_inv is None:
        sw_sqrt_X = np.sqrt(sample_weights)[:, None] * X
        XprimeX = sw_sqrt_X.T @ sw_sqrt_X
        XprimeX_inv = _r_ginv(XprimeX)
    
    # Step 2: GLM initialization
    gmm_loss_func_for_init = lambda b: _gmm_loss(b, X, treat, sample_weights, att, None)
    beta_init, beta_glm = _glm_init(
        treat, X, sample_weights, att, gmm_loss_func_for_init
    )
    
    # Step 3: Pre-compute inverse covariance matrix (for two-step GMM)
    gmm_init = beta_init
    gmm_result_init = _gmm_func(gmm_init, X, treat, sample_weights, att, inv_V=None)
    this_inv_V = gmm_result_init['inv_V']
    
    # Step 4: Balance loss optimization for initial values
    if verbose >= 1:
        print(f"[CBPS Binary] Starting balance optimization (max_iter={iterations})...")
    
    opt_bal = _optimize_balance(
        gmm_init, X, treat, sample_weights, XprimeX_inv, att,
        two_step, iterations, bal_only=bal_only, **kwargs
    )
    
    if verbose >= 1:
        print(f"[CBPS Binary] Balance optimization complete: loss={opt_bal.fun:.6f}, converged={opt_bal.success}")
    beta_bal = opt_bal.x  # Extract balance-optimized coefficients
    
    # Step 5: GMM optimization (for method='over') or exact moment solving
    if bal_only:
        # For just-identified GMM, user can choose:
        # - theoretical_exact=True: Use equation solver (precision ~1e-15)
        # - theoretical_exact=False: Use balance loss (R-compatible, precision ~1e-6)
        
        use_theoretical_exact = kwargs.get('theoretical_exact', False)
        
        if use_theoretical_exact:
            # Direct moment equation solving (theoretically correct)
            beta_opt, root_success, moments_final, solver_method = _solve_moment_equations(
                beta_bal,  # Use balance-optimized result as initial value
                X, treat, sample_weights, att, n, iterations
            )
            
            max_moment = np.max(np.abs(moments_final))
            
            if root_success:
                if max_moment < 1e-8:
                    # Perfect convergence to theoretical precision
                    pass
                else:
                    # Solver converged but moment not satisfied (rare)
                    warnings.warn(
                        f"theoretical_exact=True: Equation solver converged but moment={max_moment:.2e}, "
                        f"below theoretical requirement <1e-10. Consider better variable preprocessing.",
                        UserWarning
                    )
            else:
                # Equation solver failed, fall back to balance optimization
                warnings.warn(
                    f"theoretical_exact=True: Equation solver failed ({solver_method}), "
                    f"falling back to balance loss optimization result.",
                    UserWarning
                )
                beta_opt = beta_bal
            
            # Update opt1 object for interface compatibility
            opt1 = opt_bal
            opt1.x = beta_opt
        else:
            # R-compatible implementation: balance loss optimization
            opt1 = opt_bal
            
            # Check moment convergence
            moments_final = _compute_moment_conditions(
                opt1.x, X, treat, sample_weights, att, n
            )
            max_moment = np.max(np.abs(moments_final))
            
            # Note: For method='exact', the J-statistic is computed using over-identified
            # GMM conditions (score + balance). This means J > 0 even for just-identified
            # models, reflecting the degree to which score conditions are violated.
            
            if max_moment > 1e-6:
                warnings.warn(
                    f"method='exact': Moment conditions converged to {max_moment:.2e}, "
                    f"below theoretical GMM precision <1e-10. This is a known limitation "
                    f"of balance loss optimization.\n"
                    f"For exact moment=0 satisfaction (~1e-15 precision), "
                    f"use theoretical_exact=True in CBPS() call.",
                    UserWarning
                )
    else:
        if verbose >= 1:
            print(f"[CBPS Binary] Starting GMM optimization with dual initialization...")
        
        opt1 = _optimize_gmm_dual_init(
            gmm_init, beta_bal, X, treat, sample_weights, att,
            this_inv_V, two_step, iterations, **kwargs
        )
        
        if verbose >= 1:
            print(f"[CBPS Binary] GMM optimization complete: J={opt1.fun:.6f}, converged={opt1.success}")
    
    # Step 6: Final probabilities and weights
    beta_opt = opt1.x
    probs_opt, w_opt = _compute_final_weights(
        beta_opt, X, treat, sample_weights, att, standardize
    )
    
    # Step 7: Compute J-statistic, deviance, and null deviance
    J_opt, mle_J, deviance, nulldeviance = _compute_diagnostics(
        beta_opt, beta_glm, probs_opt, treat, sample_weights,
        att, two_step, this_inv_V, X
    )
    
    # Note: For method='exact', the J-statistic is computed using over-identified
    # GMM conditions. Theoretically, J should be 0 for just-identified models,
    # but the full GMM conditions provide a useful diagnostic.
    
    # Step 8: Variance-covariance matrix
    vcov = _compute_vcov(
        beta_opt, probs_opt, treat, X, sample_weights, att,
        bal_only, XprimeX_inv, this_inv_V, two_step, n
    )
    
    # Step 9: Construct return dictionary
    output = {
        'coefficients': beta_opt.reshape(-1, 1),  # (k, 1) column vector
        'fitted_values': probs_opt,
        'linear_predictor': X @ beta_opt,
        'deviance': deviance,
        'nulldeviance': nulldeviance,
        'weights': w_opt,
        'y': treat,
        'x': X_orig,
        'converged': opt1.success,
        'J': J_opt,
        'var': vcov,
        'mle_J': mle_J
    }
    
    return output

