"""
Propensity Score Weight Computation

This module provides functions for computing inverse probability weights
(IPW) for different treatment types and target estimands within the
CBPS framework.

Supported weight types:

- **ATE weights**: Average Treatment Effect weights for binary treatments
- **ATT weights**: Average Treatment Effect on Treated weights
- **Continuous treatment**: Weighted treatment variable for balance conditions
- **Standardized weights**: Group-normalized weights (Hajek estimator)

Mathematical Framework
----------------------
For binary treatments with propensity score e(X):

**ATE**: w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))

**ATT**: w_i = (n/n_t) * (T_i - e(X_i))/(1 - e(X_i))

For continuous treatments with generalized propensity score f(T|X):

**Stabilized**: w_i = f(T_i) / f(T_i|X_i)

Functions
---------
compute_ate_weights
    Compute ATE inverse probability weights.
compute_att_weights  
    Compute ATT inverse probability weights.
compute_continuous_weights
    Compute stabilized continuous treatment weights.
standardize_weights
    Normalize weights by treatment group.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1),
156-177.
"""

from typing import Optional

import numpy as np


def compute_ate_weights(
    treat: np.ndarray,
    probs: np.ndarray
) -> np.ndarray:
    """
    Compute ATE inverse probability weights for binary treatments.

    Implements the standard IPW formula:
    
        w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
        Should be clipped to (0, 1) before calling for numerical stability.

    Returns
    -------
    np.ndarray
        Unstandardized ATE weights, shape (n,).
        All weights are guaranteed positive.

    Notes
    -----
    This formula naturally produces positive weights:
    
    - Treated units (T=1): weight = 1/e(X)
    - Control units (T=0): weight = 1/(1-e(X))

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 0, 1, 0])
    >>> probs = np.array([0.6, 0.4, 0.7, 0.3])
    >>> w = compute_ate_weights(treat, probs)
    >>> bool(np.all(w > 0))
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Standard form: ensures positive weights directly
    weights = treat / probs + (1 - treat) / (1 - probs)

    # Note: Alternative algebraically equivalent form
    # weights = np.abs(1 / (probs - 1 + treat))

    return weights


def compute_att_weights(
    treat: np.ndarray,
    probs: np.ndarray,
    sample_weights: np.ndarray
) -> np.ndarray:
    """
    Compute ATT inverse probability weights for binary treatments.

    Implements the ATT weighting formula:
    
        w_i = (n/n_t) * (T_i - e(X_i))/(1 - e(X_i))

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
    sample_weights : np.ndarray
        Sampling weights normalized to sum to n, shape (n,).

    Returns
    -------
    np.ndarray
        Unstandardized ATT weights, shape (n,).
        Control unit weights are negative by construction.

    Notes
    -----
    The formula produces different signs by treatment status:
    
    - Treated (T=1): w = n/n_t (positive)
    - Control (T=0): w = -(n/n_t) * e/(1-e) (negative)

    The calling function typically applies absolute value to control weights
    during standardization.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 1, 0, 0])
    >>> probs = np.array([0.6, 0.7, 0.4, 0.3])
    >>> sw = np.ones(4)
    >>> w = compute_att_weights(treat, probs, sw)
    >>> bool(all(w[treat == 1] > 0))  # Treated positive
    True
    >>> bool(all(w[treat == 0] < 0))  # Control negative
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Calculate effective sample size of treated group
    n_t = np.sum(sample_weights[treat == 1])
    n = len(treat)

    # ATT weight formula (may produce negative values for controls)
    weights = (n / n_t) * (treat - probs) / (1 - probs)

    return weights


def compute_continuous_weights(
    Ttilde: np.ndarray,
    stabilizers: np.ndarray,
    log_density: np.ndarray
) -> np.ndarray:
    """
    Compute weighted standardized treatment for continuous CBPS balance conditions.

    Computes the quantity T* × w where w is the stabilized weight:
    
        T̃_i × w_i = T̃_i × exp(log f(T̃_i) - log f(T̃_i|X_i))

    This is used in the CBPS balance condition E[T* × w × X*] = 0
    (Fong et al., 2018, Eq. 2).

    Parameters
    ----------
    Ttilde : np.ndarray
        Standardized treatment (mean=0, std=1), shape (n,).
    stabilizers : np.ndarray
        Log marginal density log f(T̃), shape (n,).
    log_density : np.ndarray
        Log conditional density log f(T̃|X) (GPS), shape (n,).

    Returns
    -------
    np.ndarray
        Weighted treatment T̃ × w, shape (n,).
        Note: This is NOT the weight itself; the stabilized weight is
        w = f(T̃)/f(T̃|X) = exp(stabilizers - log_density).

    Notes
    -----
    **Numerical stability**: The log-density difference is clipped to
    [-50, 50] before exponentiation to prevent overflow.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> Ttilde = np.array([0.0, 1.0, -1.0])
    >>> stabilizers = norm.logpdf(Ttilde, 0, 1)  # Marginal density
    >>> log_density = norm.logpdf(Ttilde, 0, 1)  # Same as marginal (no confounding)
    >>> Tw = compute_continuous_weights(Ttilde, stabilizers, log_density)
    >>> bool(np.all(np.isfinite(Tw)))
    True

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. The Annals of Applied Statistics, 12(1),
    156-177.
    """
    # Compute weights in log space for numerical stability
    # Equivalent to: Ttilde * exp(stabilizers - log_density)

    # Clip difference to prevent overflow in exp()
    log_diff = stabilizers - log_density
    log_diff_clipped = np.clip(log_diff, -50, 50)

    # Final weight computation
    weights = Ttilde * np.exp(log_diff_clipped)

    return weights


def standardize_weights(
    weights: np.ndarray,
    treat: np.ndarray,
    probs: np.ndarray,
    sample_weights: np.ndarray,
    att: int = 0,
    standardize: bool = True
) -> np.ndarray:
    """
    Normalize IPW weights by treatment group.

    Applies group-wise normalization so that weights within each treatment
    group sum to 1, or returns unnormalized Horvitz-Thompson weights.

    Parameters
    ----------
    weights : np.ndarray
        Raw weights (unused, kept for API compatibility), shape (n,).
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
    sample_weights : np.ndarray
        Sampling weights (normalized to sum to n), shape (n,).
    att : int, default=0
        Target estimand: 0 = ATE, 1 = ATT (T=1 as treated), 2 = ATT (T=0 as treated).
    standardize : bool, default=True
        If True, apply group-wise normalization.
        If False, return Horvitz-Thompson weights.

    Returns
    -------
    np.ndarray
        Final weights multiplied by sample_weights, shape (n,).

    Notes
    -----
    **Normalization behavior**:
    
    - ATE: Weights are normalized so treated and control groups each
      sum to 1 (Hajek-style normalization)
    - ATT: Similar normalization with absolute value for control weights
    
    **Implementation**: Weights are reconstructed from propensity scores
    rather than directly normalizing the input weights.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 0, 1, 0])
    >>> probs = np.array([0.6, 0.4, 0.7, 0.3])
    >>> sw = np.ones(4)
    >>> w = np.ones(4)
    >>> w_std = standardize_weights(w, treat, probs, sw, att=0, standardize=True)
    >>> bool(np.isclose(w_std[treat==1].sum(), 1.0))  # Treated group normalized
    True
    >>> bool(np.isclose(w_std[treat==0].sum(), 1.0))  # Control group normalized
    True
    """
    n = len(treat)
    # Use unweighted count for ATT normalization, matching R's CBPSBinary.R
    # R redefines n.t = sum(treat==1) (unweighted) before weight standardization
    n_t_unweighted = np.sum(treat == 1)
    
    if standardize:
        # Step 1: Compute normalization factors
        if att:  # ATT branch
            norm1 = np.sum(treat * sample_weights * n / n_t_unweighted)
            norm2 = np.sum((1 - treat) * sample_weights * n / n_t_unweighted * 
                          (treat - probs) / (1 - probs))
        else:  # ATE branch
            norm1 = np.sum(treat * sample_weights / probs)
            norm2 = np.sum((1 - treat) * sample_weights / (1 - probs))
    else:
        # Step 2: Horvitz-Thompson weights (no normalization)
        norm1 = 1.0
        norm2 = 1.0
    
    # Step 3: Reconstruct standardized weights
    # Note: weights are reconstructed, not directly standardized from input
    if att:  # ATT branch
        # Treatment group + abs(control group)
        weights_std = (
            (treat == 1) * n / n_t_unweighted / norm1 +
            np.abs((treat == 0) * n / n_t_unweighted * 
                   (treat - probs) / (1 - probs) / norm2)
        )
    else:  # ATE branch
        # Standard form, no abs() needed (1/π and 1/(1-π) always positive)
        weights_std = (
            (treat == 1) / probs / norm1 +
            (treat == 0) / (1 - probs) / norm2
        )
    
    # Step 4: Multiply by sample_weights
    weights_std = weights_std * sample_weights
    
    return weights_std

