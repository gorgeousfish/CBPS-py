"""
GMM Loss Functions for High-Dimensional CBPS
=============================================

This module implements the Generalized Method of Moments (GMM) loss functions
used in Step 3 of the hdCBPS algorithm for covariate balance calibration.

The GMM objective minimizes the squared norm of the covariate balancing
moment conditions (Equation 7 in Ning et al., 2020):

.. math::

    \\tilde{\\gamma} = \\arg\\min_{\\gamma} \\|g_n(\\gamma)\\|_2^2

where the moment function is:

.. math::

    g_n(\\gamma) = \\sum_{i=1}^{n}
    \\left( \\frac{T_i}{\\pi(\\gamma^T X_{i\\tilde{S}} +
    \\hat{\\beta}_{\\tilde{S}^c}^T X_{i\\tilde{S}^c})} - 1 \\right) X_{i\\tilde{S}}

Note: The paper defines g_n with a 1/n factor, but this implementation uses
the sum (without 1/n) since minimizing ||g||^2 and ||g/n||^2 yield identical
solutions. This calibration step removes bias from the penalized estimators
and achieves the weak covariate balancing property (Equation 9).

References
----------
Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score.
Biometrika, 107(3), 533-554. https://doi.org/10.1093/biomet/asaa020
"""

import numpy as np
from typing import Tuple

from .weight_funcs import ate_wt_func, ate_wt_nl_func, att_wt_func, att_wt_nl_func


def gmm_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    tt: int,
    X_gmm: np.ndarray,
    method: str,
    cov1_coef: np.ndarray,
    cov0_coef: np.ndarray,
    treat: np.ndarray,
    beta_ini: np.ndarray
) -> float:
    """
    Compute the GMM loss function for ATE estimation.

    This function evaluates the covariate balancing objective for average
    treatment effect (ATE) estimation in Step 3 of hdCBPS. It computes
    the squared norm of the moment conditions that enforce balance between
    treatment groups.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates being optimized. Shape depends on method:

        - Linear method: shape ``(len(S),)`` without intercept
        - Nonlinear methods: shape ``(len(S)+1,)`` with intercept

    S : np.ndarray
        Indices of LASSO-selected variables from the outcome model (0-based).
        Corresponds to :math:`\\tilde{S}` in the paper.
    tt : int
        Treatment group indicator:

        - 0: Optimize for control group (estimating :math:`\\mu_0`)
        - 1: Optimize for treated group (estimating :math:`\\mu_1`)

    X_gmm : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    method : str
        Outcome model specification:

        - ``'linear'``: Gaussian outcome model
        - ``'binomial'``: Logistic outcome model
        - ``'poisson'``: Poisson outcome model

    cov1_coef : np.ndarray, shape (p+1,)
        Outcome model coefficients for the treated group.
    cov0_coef : np.ndarray, shape (p+1,)
        Outcome model coefficients for the control group.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO (Step 1).

    Returns
    -------
    loss : float
        GMM loss value: :math:`\\|g_n(\\gamma)\\|_2^2`.

    Notes
    -----
    For the linear outcome model, the moment condition is:

    .. math::

        g_n(\\gamma) = \\sum_{i=1}^{n}
        \\left( \\frac{T_i}{\\pi_i} - 1 \\right) X_{i\\tilde{S}}

    For generalized linear models (binomial/poisson), the weighted covariates
    :math:`f(X) = b''(\\tilde{\\alpha}^T X) X_{\\tilde{S}}` are balanced instead,
    as described in Section 4 of the paper.
    """
    # Convert covariate matrix to numpy array
    x1 = np.asarray(X_gmm)
    n1 = x1.shape[0]
    
    # IMPORTANT: Match R's behavior exactly
    # R code does: X1 = cbind(rep(1,n1), x1) even though x1 already has intercept
    # This adds an extra intercept column, making X1 have shape (n, p+2)
    # The S indices from coef() are 1-based in R and refer to the (p+1) coefficient vector
    # When used with X1[,S], the extra intercept shifts the column alignment
    # We must replicate this behavior for compatibility
    X1 = np.column_stack([np.ones(n1), x1])
    
    # Branch on method type
    if method == "linear":
        # Linear method: simple inverse probability weights
        W = ate_wt_func(beta_curr, S, tt, x1, beta_ini, treat)
        
        # Compute weighted covariate means
        if len(S) > 0:
            # Extract selected columns and compute weighted means
            w_curr_del = X1[:, S].T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            # No selected variables
            w_curr_del = np.array([])
    
    elif method == "poisson":
        # Poisson method: exponential link weights
        W = ate_wt_nl_func(beta_curr, S, tt, x1, beta_ini, treat)
        
        # Select outcome model coefficients by treatment group
        if tt == 1:
            pweight = np.exp(X1 @ cov1_coef)
        else:
            pweight = np.exp(X1 @ cov0_coef)
        
        # Compute weighted covariates
        if len(S) > 0:
            # Stack outcome weights with weighted selected covariates
            weighted_X = np.column_stack([
                pweight,
                pweight[:, None] * X1[:, S]
            ])
            w_curr_del = weighted_X.T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            # Only outcome weight when no covariates selected
            w_curr_del = pweight @ W
            w_curr_del = np.asarray([w_curr_del]).ravel()
    
    elif method == "binomial":
        # Binomial method: logistic link weights
        W = ate_wt_nl_func(beta_curr, S, tt, x1, beta_ini, treat)
        
        # Compute logistic probabilities and derivatives
        if tt == 1:
            # Treated group outcome model
            exp_term = np.exp(X1 @ cov1_coef)
            pweight1 = exp_term / (1.0 + exp_term)
            pweight2 = exp_term / (1.0 + exp_term)**2
        else:
            # Control group outcome model
            exp_term = np.exp(X1 @ cov0_coef)
            pweight1 = exp_term / (1.0 + exp_term)
            pweight2 = exp_term / (1.0 + exp_term)**2
        
        # Compute weighted covariates
        if len(S) > 0:
            # Stack probability with derivative-weighted selected covariates
            weighted_X = np.column_stack([
                pweight1,
                pweight2[:, None] * X1[:, S]
            ])
            w_curr_del = weighted_X.T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            # Only probability weight when no covariates selected
            w_curr_del = pweight1 @ W
            w_curr_del = np.asarray([w_curr_del]).ravel()
    
    else:
        raise ValueError(
            f"method '{method}' not supported. "
            f"Choose from: 'linear', 'binomial', 'poisson'"
        )
    
    # Compute GMM loss as squared norm of moment conditions
    gbar = w_curr_del
    loss = gbar @ gbar
    
    return float(loss)


def att_gmm_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    X_gmm: np.ndarray,
    method: str,
    cov0_coef: np.ndarray,
    treat: np.ndarray,
    beta_ini: np.ndarray
) -> float:
    """
    Compute the GMM loss function for ATT estimation.

    This function evaluates the covariate balancing objective for the average
    treatment effect on the treated (ATT) in Step 3 of hdCBPS. For ATT, only
    the control group propensity score is calibrated to match the treated
    group covariate distribution.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current coefficient estimates being optimized. Shape depends on method:

        - Linear method: shape ``(len(S),)`` without intercept
        - Nonlinear methods: shape ``(len(S)+1,)`` with intercept

    S : np.ndarray
        Indices of LASSO-selected variables from the control outcome model.
    X_gmm : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    method : str
        Outcome model specification: ``'linear'``, ``'binomial'``, or ``'poisson'``.
    cov0_coef : np.ndarray, shape (p+1,)
        Outcome model coefficients for the control group.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO.

    Returns
    -------
    loss : float
        GMM loss value: :math:`\\|g_n(\\gamma)\\|_2^2`.

    Notes
    -----
    Unlike ATE estimation which requires separate optimization for treated
    and control groups, ATT estimation only requires calibrating the control
    group weights to match the treated group distribution. The ATT moment
    condition ensures that the reweighted control group has the same covariate
    means as the treated group.

    See the Supplementary Material of Ning et al. (2020) for theoretical
    details on ATT estimation in high-dimensional settings.
    """
    # Convert covariate matrix to numpy array
    x1 = np.asarray(X_gmm)
    n1 = x1.shape[0]
    
    # IMPORTANT: Match R's behavior exactly
    # R code does: X1 = cbind(rep(1,n1), x1) even though x1 already has intercept
    # This adds an extra intercept column, making X1 have shape (n, p+2)
    # The S indices from coef() are 1-based in R and refer to the (p+1) coefficient vector
    # When used with X1[,S], the extra intercept shifts the column alignment
    # We must replicate this behavior for compatibility
    X1 = np.column_stack([np.ones(n1), x1])
    
    # Branch on method type
    if method == "linear":
        # Linear method: simple inverse probability weights for ATT
        W = att_wt_func(beta_curr, S, x1, beta_ini, treat)
        
        # Compute weighted covariate means
        if len(S) > 0:
            w_curr_del = X1[:, S].T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            w_curr_del = np.array([])
    
    elif method == "poisson":
        # Poisson method: exponential link weights for ATT
        W = att_wt_nl_func(beta_curr, S, x1, beta_ini, treat)
        
        # Compute exponential weights from control outcome model
        # Note: X1 = cbind(1, x1) has shape (n, p+2), cov0_coef has length p+2
        pweight = np.exp(X1 @ cov0_coef)
        
        # Compute weighted covariates
        if len(S) > 0:
            # Stack outcome weights with weighted selected covariates
            weighted_X = np.column_stack([
                pweight,
                pweight[:, None] * X1[:, S]
            ])
            w_curr_del = weighted_X.T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            # Only outcome weight when no covariates selected
            w_curr_del = pweight @ W
            w_curr_del = np.asarray([w_curr_del]).ravel()
    
    elif method == "binomial":
        # Binomial method: logistic link weights for ATT
        W = att_wt_nl_func(beta_curr, S, x1, beta_ini, treat)
        
        # Compute logistic probabilities and derivatives (control group only)
        # Note: X1 = cbind(1, x1) has shape (n, p+2), cov0_coef has length p+2
        exp_term = np.exp(X1 @ cov0_coef)
        pweight1 = exp_term / (1.0 + exp_term)
        pweight2 = exp_term / (1.0 + exp_term)**2
        
        # Compute weighted covariates
        if len(S) > 0:
            # Stack probability with derivative-weighted selected covariates
            weighted_X = np.column_stack([
                pweight1,
                pweight2[:, None] * X1[:, S]
            ])
            w_curr_del = weighted_X.T @ W
            w_curr_del = np.asarray(w_curr_del).ravel()
        else:
            # Only probability weight when no covariates selected
            w_curr_del = pweight1 @ W
            w_curr_del = np.asarray([w_curr_del]).ravel()
    
    else:
        raise ValueError(
            f"method '{method}' not supported. "
            f"Choose from: 'linear', 'binomial', 'poisson'"
        )
    
    # Compute GMM loss as squared norm of moment conditions
    gbar = w_curr_del
    loss = gbar @ gbar
    
    return float(loss)
