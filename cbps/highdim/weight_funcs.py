"""
Weight Functions for High-Dimensional CBPS
===========================================

This module implements inverse probability weight functions for ATE and ATT
estimation in hdCBPS. These weights are used in the GMM objective (Step 3)
to calibrate propensity scores for covariate balance.

The weight functions compute the difference between unit weights and inverse
propensity score weights, which appear in the covariate balancing moment
conditions. For ATE:

.. math::

    W_i = \\frac{T_i}{\\pi_i} - 1 \\quad \\text{(treated)}

    W_i = 1 - \\frac{1-T_i}{1-\\pi_i} \\quad \\text{(control)}

References
----------
Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score.
Biometrika, 107(3), 533-554. https://doi.org/10.1093/biomet/asaa020
"""

import numpy as np


def ate_wt_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    tt: int,
    X_wt: np.ndarray,
    beta_ini: np.ndarray,
    treat: np.ndarray
) -> np.ndarray:
    """
    Compute ATE weights for linear outcome models.

    This function computes inverse probability weights used in the covariate
    balancing moment conditions for ATE estimation with linear (Gaussian)
    outcome models.

    Parameters
    ----------
    beta_curr : np.ndarray, shape (len(S),)
        Current propensity score coefficients at LASSO-selected positions.
    S : np.ndarray
        Indices of selected variables (0-based, excluding intercept).
    tt : int
        Treatment group indicator:

        - 0: Compute control group weights
        - 1: Compute treated group weights

    X_wt : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO (including intercept).
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).

    Returns
    -------
    W : np.ndarray, shape (n,)
        Weight vector for the moment conditions:

        - ``tt=1``: :math:`W_i = T_i / \\pi_i - 1`
        - ``tt=0``: :math:`W_i = 1 - (1-T_i) / (1-\\pi_i)`

    Notes
    -----
    The coefficient vector is constructed by starting from ``beta_ini`` and
    replacing entries at positions ``S`` with ``beta_curr``. This allows
    optimization over the selected subset while holding other coefficients
    fixed at their LASSO estimates.
    """
    # Convert to array and get sample size
    x2 = np.asarray(X_wt)
    n2 = x2.shape[0]
    
    # Add intercept column: X2 = [1, X_wt]
    X2 = np.column_stack([np.ones(n2), x2])
    
    # Update coefficients at selected positions
    beta_all = beta_ini.copy()
    beta_all[S] = beta_curr
    
    # Compute linear predictor and propensity scores
    theta_curr = X2 @ beta_all
    probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta_curr))
    
    # Compute weights based on treatment group indicator
    if tt == 0:
        # Control group weights
        W = 1.0 - (1.0 - treat) / (1.0 - probs_curr)
    else:
        # Treatment group weights
        W = treat / probs_curr - 1.0
    
    return W


def ate_wt_nl_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    tt: int,
    X_wt: np.ndarray,
    beta_ini: np.ndarray,
    treat: np.ndarray
) -> np.ndarray:
    """
    Compute ATE weights for nonlinear outcome models (binomial/poisson).

    This function computes inverse probability weights for ATE estimation
    when the outcome model is a generalized linear model (binomial or poisson).
    It differs from :func:`ate_wt_func` in that the intercept is included
    in the optimization.

    Parameters
    ----------
    beta_curr : np.ndarray, shape (len(S)+1,)
        Current propensity score coefficients at ``[0, S]`` positions,
        where index 0 is the intercept.
    S : np.ndarray
        Indices of selected variables (0-based, excluding intercept).
    tt : int
        Treatment group indicator (0=control, 1=treated).
    X_wt : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).

    Returns
    -------
    W : np.ndarray, shape (n,)
        Weight vector for the moment conditions.

    Notes
    -----
    For nonlinear outcome models, LASSO is fitted with ``intercept=False``,
    so the intercept must be optimized along with the selected covariates.
    The selected indices become ``SS = [0, S]`` (including the intercept).
    """
    # Convert to array and get sample size
    x2 = np.asarray(X_wt)
    n2 = x2.shape[0]
    
    # Add intercept column: X2 = [1, X_wt]
    X2 = np.column_stack([np.ones(n2), x2])
    
    # SS = [0, S] includes intercept (0-based indexing)
    SS = np.r_[0, S]
    
    # Update coefficients at [intercept, S] positions
    beta_all = beta_ini.copy()
    beta_all[SS] = beta_curr
    
    # Compute linear predictor and propensity scores
    theta_curr = X2 @ beta_all
    probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta_curr))
    
    # Compute weights based on treatment group indicator
    if tt == 0:
        # Control group weights
        W = 1.0 - (1.0 - treat) / (1.0 - probs_curr)
    else:
        # Treatment group weights
        W = treat / probs_curr - 1.0
    
    return W


def att_wt_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    X_wt: np.ndarray,
    beta_ini: np.ndarray,
    treat: np.ndarray
) -> np.ndarray:
    """
    Compute ATT weights for linear outcome models.

    This function computes weights for the average treatment effect on the
    treated (ATT) with linear outcome models. The ATT weight reweights
    control units to match the treated group covariate distribution.

    Parameters
    ----------
    beta_curr : np.ndarray, shape (len(S),)
        Current propensity score coefficients at selected positions.
    S : np.ndarray
        Indices of selected variables (0-based).
    X_wt : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).

    Returns
    -------
    W : np.ndarray, shape (n,)
        ATT weight vector:

        .. math::

            W_i = T_i - \\frac{(1-T_i) \\pi_i}{1 - \\pi_i}

    Notes
    -----
    Unlike ATE weights which have separate formulas for treated and control
    groups, ATT uses a single weight formula. For treated units (T=1), W=1.
    For control units (T=0), W equals the odds ratio :math:`-\\pi/(1-\\pi)`.
    """
    # Convert to array and get sample size
    x2 = np.asarray(X_wt)
    n2 = x2.shape[0]
    
    # Add intercept column: X2 = [1, X_wt]
    X2 = np.column_stack([np.ones(n2), x2])
    
    # Update coefficients at selected positions
    beta_all = beta_ini.copy()
    beta_all[S] = beta_curr
    
    # Compute linear predictor and propensity scores
    theta_curr = X2 @ beta_all
    probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta_curr))
    
    # ATT weight formula
    W = treat - ((1.0 - treat) * probs_curr / (1.0 - probs_curr))
    
    return W


def att_wt_nl_func(
    beta_curr: np.ndarray,
    S: np.ndarray,
    X_wt: np.ndarray,
    beta_ini: np.ndarray,
    treat: np.ndarray
) -> np.ndarray:
    """
    Compute ATT weights for nonlinear outcome models (binomial/poisson).

    This function computes ATT weights when the outcome model is a generalized
    linear model. It includes the intercept in the optimization, similar to
    :func:`ate_wt_nl_func`.

    Parameters
    ----------
    beta_curr : np.ndarray, shape (len(S)+1,)
        Current propensity score coefficients at ``[0, S]`` positions.
    S : np.ndarray
        Indices of selected variables (0-based, excluding intercept).
    X_wt : np.ndarray, shape (n, p)
        Covariate matrix without intercept column.
    beta_ini : np.ndarray, shape (p+1,)
        Initial propensity score coefficients from LASSO.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator (0/1).

    Returns
    -------
    W : np.ndarray, shape (n,)
        ATT weight vector.

    Notes
    -----
    The weight formula is identical to :func:`att_wt_func`, but the
    coefficient update includes the intercept at position 0.
    """
    # Convert to array and get sample size
    x2 = np.asarray(X_wt)
    n2 = x2.shape[0]
    
    # Add intercept column: X2 = [1, X_wt]
    X2 = np.column_stack([np.ones(n2), x2])
    
    # SS = [0, S] includes intercept (0-based indexing)
    SS = np.r_[0, S]
    
    # Update coefficients at [intercept, S] positions
    beta_all = beta_ini.copy()
    beta_all[SS] = beta_curr
    
    # Compute linear predictor and propensity scores
    theta_curr = X2 @ beta_all
    probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta_curr))
    
    # ATT weight formula
    W = treat - ((1.0 - treat) * probs_curr / (1.0 - probs_curr))
    
    return W

