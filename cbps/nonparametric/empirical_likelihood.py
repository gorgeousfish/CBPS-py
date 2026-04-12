"""
Empirical Likelihood Optimization for Nonparametric CBPS.

This module implements the dual optimization approach for empirical
likelihood estimation described in Section 3.3.2 of Fong, Hazlett,
and Imai (2018).

The key insight is dimension reduction: instead of optimizing over
n weights directly, we optimize over (2K+1) Lagrange multipliers
:math:`\\gamma`, then recover weights via:

.. math::

    w_i = \\frac{1}{1 - \\gamma^T g(X_i^*, T_i^*)}

Key Functions
-------------
- :func:`log_elgiven_eta`: Objective function for :math:`\\gamma`
  optimization given the weighted correlation :math:`\\eta`.
- :func:`get_w`: Recover weights and check convergence.
- :func:`log_post`: Penalized likelihood for the outer :math:`\\alpha`
  line search.

Mathematical Background
-----------------------
The Lagrangian for the constrained likelihood maximization (Section 3.3.2)
leads to the dual problem:

.. math::

    \\underset{\\gamma}{\\text{argmax}} \\sum_{i=1}^n
    \\log(1 - \\gamma^T(g_i - \\eta))

where :math:`g_i = (X_i^* T_i^*, X_i^*, T_i^*)^T` is the constraint vector
and :math:`\\eta` is the allowed finite-sample imbalance.

Note: The ordering of components in :math:`g_i` follows the implementation
rather than the paper's notation :math:`(X_i^*, T_i^*, X_i^* T_i^*)^T`.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. The Annals of Applied Statistics, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101

Owen, A.B. (2001). Empirical Likelihood. Chapman & Hall/CRC.
"""

from typing import Dict, Union
import numpy as np
import scipy.optimize

from .taylor_approx import llog


def log_elgiven_eta(
    gamma: np.ndarray,
    eta: np.ndarray,
    z: np.ndarray,
    eps: float,
    ncon_cor: int,
    n: int
) -> float:
    """
    Dual objective function for empirical likelihood optimization.

    Computes the negative log empirical likelihood as a function of the
    Lagrange multipliers :math:`\\gamma`, given the allowed imbalance
    :math:`\\eta`. This is the inner optimization in the npCBPS algorithm.

    The dual formulation (Equation 9 in Fong et al., 2018) reduces the
    problem from n-dimensional weight optimization to (2K+1)-dimensional
    :math:`\\gamma` optimization.

    Parameters
    ----------
    gamma : np.ndarray of shape (ncon,)
        Lagrange multiplier vector to optimize.
    eta : np.ndarray of shape (ncon_cor,)
        Allowed weighted correlation vector :math:`\\eta`.
    z : np.ndarray of shape (n, ncon)
        Constraint matrix :math:`(X^* T^*, X^*, T^*)`.
    eps : float
        Threshold for Taylor approximation in :func:`llog`, typically 1/n.
    ncon_cor : int
        Number of correlation constraints (K for continuous treatment,
        K*(J-1) for J-level factor treatment).
    n : int
        Sample size.

    Returns
    -------
    float
        Negative log empirical likelihood (to be minimized).

    Notes
    -----
    **Mathematical formulation:**

    The objective is derived from Equation 9 in Section 3.3.3:

    .. math::

        -\\sum_{i=1}^n \\log(1 - \\gamma^T(g_i - \\eta))

    Equivalently, with the scaling convention used in this implementation:

    .. math::

        -\\sum_{i=1}^n \\text{llog}(n + \\gamma^T(\\eta - z_i))

    where :func:`llog` provides numerical stability for small arguments.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Equation 9, Section 3.3.3.
    """
    ncon = z.shape[1]

    # Extend eta to ncon dimensions (pad with zeros)
    eta_long = np.concatenate([eta, np.zeros(ncon - ncon_cor)])

    # Broadcast eta_long to matrix (ncon × n)
    eta_mat = eta_long[:, None] @ np.ones((1, n))

    # Core formula: arg is a 1 × n row vector
    arg = n + gamma.T @ (eta_mat - z.T)

    # Empirical likelihood
    log_el = -np.sum(llog(arg, eps))

    return log_el


def get_w(
    eta: np.ndarray,
    z: np.ndarray,
    sumw_tol: float,
    eps: float,
    ncon_cor: int,
    n: int
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute optimal weights given the allowed imbalance eta.

    This function performs the inner optimization: given :math:`\\eta`,
    find the optimal :math:`\\gamma` via BFGS, then recover weights using
    the formula from Owen (2001):

    .. math::

        w_i = \\frac{1}{1 - \\gamma^T(g_i - \\eta)}

    A convergence check verifies that :math:`\\sum w_i \\approx 1`.

    Parameters
    ----------
    eta : np.ndarray of shape (ncon_cor,)
        Allowed weighted correlation vector.
    z : np.ndarray of shape (n, ncon)
        Constraint matrix.
    sumw_tol : float
        Tolerance for weight sum convergence. If :math:`|1 - \\sum w_i|`
        exceeds this threshold, a penalty is added to the likelihood.
        Typical values: 0.05 for final weights, 0.001 during optimization.
    eps : float
        Threshold for Taylor approximation, typically 1/n.
    ncon_cor : int
        Number of correlation constraints.
    n : int
        Sample size.

    Returns
    -------
    dict
        Dictionary with keys:

        - **w** : np.ndarray of shape (n,)
            Unnormalized weights (before normalization to sum to n).
        - **sumw** : float
            Sum of weights (ideally close to 1).
        - **log_el** : float
            Log empirical likelihood, possibly with penalty if sumw
            deviates from 1.
        - **el_gamma** : np.ndarray of shape (ncon_cor,)
            Optimal Lagrange multipliers for correlation constraints.

    Notes
    -----
    **Convergence penalty:**

    When :math:`|1 - \\sum w_i| > \\text{sumw\\_tol}`:

    .. math::

        \\text{log\\_el} = -\\sum \\log(w_i / \\sum w_i)
                          - 10^4 \\cdot (1 + |1 - \\sum w_i|)

    This penalty guides the outer optimization away from :math:`\\eta`
    values that lead to poor weight recovery.

    References
    ----------
    Owen, A.B. (2001). Empirical Likelihood. Chapman & Hall/CRC.
    """
    ncon = z.shape[1]

    # Initialize gamma = 0
    gam_init = np.zeros(ncon)

    # BFGS optimization
    result = scipy.optimize.minimize(
        log_elgiven_eta,
        gam_init,
        args=(eta, z, eps, ncon_cor, n),
        method='BFGS'
    )
    gam_opt = result.x

    # Recover weights
    eta_long = np.concatenate([eta, np.zeros(ncon - ncon_cor)])
    eta_mat = eta_long[:, None] @ np.ones((1, n))
    arg_temp = n + gam_opt.T @ (eta_mat - z.T)

    # w = 1 / arg_temp
    w = 1 / arg_temp.flatten()
    sum_w = w.sum()

    # Normalize weights
    w_scaled = w / sum_w

    # Convergence check
    if abs(1 - sum_w) <= sumw_tol:
        # Pass: weight sum is close enough to 1
        log_el = -np.sum(np.log(w_scaled))
    else:
        # Fail: add penalty term
        log_el = -np.sum(np.log(w_scaled)) - 10**4 * (1 + abs(1 - sum_w))

    # Return results
    return {
        'w': w,
        'sumw': sum_w,
        'log_el': log_el,
        'el_gamma': gam_opt[:ncon_cor]
    }


def log_post(
    par: float,
    eta_to_be_scaled: np.ndarray,
    eta_prior_sd: np.ndarray,
    z: np.ndarray,
    eps: float,
    sumw_tol: float,
    ncon_cor: int,
    n: int
) -> float:
    """
    Penalized log-likelihood for the outer line search.

    Computes the objective for Equation 10 in Section 3.3.3 of Fong et al.
    (2018):

    .. math::

        \\log f(X^*, T^* | \\eta) + \\log f(\\eta)

    where :math:`\\eta = \\alpha \\cdot \\eta_0` is parameterized by the
    scalar :math:`\\alpha \\in [0, 1]`, and :math:`\\eta_0` is the initial
    (unweighted) correlation.

    Parameters
    ----------
    par : float
        Scaling parameter :math:`\\alpha` in the range [0, 1].
        At :math:`\\alpha = 0`, exact balance is enforced.
        At :math:`\\alpha = 1`, the initial imbalance is retained.
    eta_to_be_scaled : np.ndarray of shape (ncon_cor,)
        Base correlation vector :math:`\\eta_0` to be scaled.
    eta_prior_sd : np.ndarray of shape (ncon_cor,)
        Prior standard deviation :math:`\\sigma` for :math:`\\eta`, where
        :math:`\\eta \\sim N(0, \\sigma^2 I_K)`. This equals the ``corprior``
        parameter.
    z : np.ndarray of shape (n, ncon)
        Constraint matrix.
    eps : float
        Threshold for Taylor approximation.
    sumw_tol : float
        Weight sum tolerance (typically 0.001 during line search).
    ncon_cor : int
        Number of correlation constraints.
    n : int
        Sample size.

    Returns
    -------
    float
        Log posterior (penalized likelihood) for maximization.

    Notes
    -----
    **Prior specification (Section 3.3.3):**

    The penalty assumes :math:`\\eta \\sim N(0, \\sigma^2 I_K)`:

    .. math::

        \\log f(\\eta) = -\\frac{K}{2}\\log(2\\pi\\sigma^2)
                        - \\frac{\\eta^T \\eta}{2\\sigma^2}

    The ``corprior`` parameter corresponds to :math:`\\sigma`. Smaller
    values enforce tighter balance constraints at the cost of potentially
    more extreme weights.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Section 3.3.3: A penalized
    imbalance approach. Equation 10.
    """
    # Scale eta
    eta_now = par * eta_to_be_scaled

    # Compute prior log density
    log_p_eta = np.sum(
        -0.5 * np.log(2 * np.pi * eta_prior_sd**2)
        - eta_now**2 / (2 * eta_prior_sd**2)
    )

    # Compute empirical likelihood
    el_out = get_w(eta_now, z, sumw_tol, eps, ncon_cor, n)

    # Compute posterior density
    c = 1
    log_post_value = el_out['log_el'] + c * log_p_eta

    # Return log posterior for maximization
    return log_post_value
