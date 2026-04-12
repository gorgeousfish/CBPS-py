"""
Covariate Balancing Propensity Score for Multi-valued Treatments.

This module implements CBPS for categorical treatments with 3 or 4 levels,
using multinomial logistic regression and contrast weights within the GMM
framework.

Algorithm Overview
------------------
1. Multinomial logistic regression for MLE initialization
2. GMM optimization with covariate balance constraints
3. Contrast weight computation for treatment effects

Notes on Implementation
-----------------------
This implementation uses statsmodels.MNLogit for multinomial logistic
initialization. Baseline-category logit models may have minor numerical
variations across different statistical libraries due to optimization
algorithms (±1e-2 to ±1e-1 in MLE estimates).

The CBPS optimization process typically reduces these differences,
with final results usually achieving ±1e-3 accuracy depending on the data.

References
----------
Imai, Kosuke and Marc Ratkovic. 2014. "Covariate Balancing Propensity Score."
Journal of the Royal Statistical Society, Series B (Statistical Methodology).
§4.1 Multi-valued Treatments, Eq.22-24 (p.260)
DOI:10.1111/rssb.12027
http://imai.princeton.edu/research/CBPS.html
"""

import warnings
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize
import statsmodels.api as sm

from .results import CBPSResults
from ..utils.helpers import normalize_sample_weights
from ..utils.numerics import r_ginv_like, pinv_match_r

# Constants
PROBS_MIN = 1e-6  # Minimum probability clipping threshold


from typing import Optional


def _r_ginv(X: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """
    R-compatible pseudoinverse.

    Default matches MASS::ginv cutoff: tol = max(dim) * smax * eps.
    If tol is provided (absolute), apply it via explicit SVD for
    consistent behavior regardless of SciPy version.
    """
    if tol is None:
        # Match MASS::ginv by default (preferred for R parity)
        return pinv_match_r(X)
    # Absolute tol requested: compute via explicit SVD to avoid
    # version-specific SciPy kwargs differences
    return r_ginv_like(X, tol=tol)


def _compute_softmax_probs_3treat(
    theta: np.ndarray,
    probs_min: float = PROBS_MIN
) -> np.ndarray:
    """
    Compute softmax probabilities for 3-level treatments.

    This function implements numerically stable softmax computation
    to avoid exponential overflow. It uses the baseline category
    logit parameterization where the first category serves as reference.

    Parameters
    ----------
    theta : np.ndarray
        Logit parameters for categories 2 and 3, shape (n, 2).
    probs_min : float, default PROBS_MIN
        Minimum probability threshold for clipping.

    Returns
    -------
    np.ndarray
        Probability matrix, shape (n, 3), with each row summing to 1.
    """
    n = theta.shape[0]
    # Numerically stable softmax: subtract row maximum before exponentiation
    theta_with_baseline = np.column_stack([np.zeros(n), theta])  # (n, 3): [0, theta[:,0], theta[:,1]]
    theta_max = theta_with_baseline.max(axis=1, keepdims=True)
    theta_stable = theta_with_baseline - theta_max

    # Compute exp(theta_stable) without overflow
    exp_theta = np.exp(theta_stable)
    probs = exp_theta / exp_theta.sum(axis=1, keepdims=True)

    # Iterative clipping and renormalization for numerical stability
    # Single-pass clipping can yield probabilities below threshold when sum > 1
    # after clipping. Iteration ensures all probabilities meet the minimum bound.
    max_iter = 10
    for iteration in range(max_iter):
        # Lower bound clipping
        probs_clipped = np.maximum(probs_min, probs)

        # Renormalization
        probs_new = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

        # Check convergence (all probabilities >= probs_min * 0.999 for numerical tolerance)
        if np.all(probs_new >= probs_min * 0.999):
            probs = probs_new
            break
            
        probs = probs_new
        
        # If the last iteration still doesn't converge, issue warning
        if iteration == max_iter - 1:
            min_prob = probs.min()
            if min_prob < probs_min * 0.999:
                import warnings
                warnings.warn(
                    f"Iterative clipping did not fully converge: min_prob={min_prob:.2e} < {probs_min:.2e}. "
                    f"This may occur in extremely imbalanced data (probabilities > 99.9999%).",
                    UserWarning
                )

    assert probs.shape == (n, 3) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-10), \
        f"Softmax probability anomaly: shape={probs.shape}, sum range=[{probs.sum(axis=1).min()}, {probs.sum(axis=1).max()}]"

    # Verify minimum probability threshold is maintained (with 0.1% numerical tolerance)
    min_prob_actual = probs.min()
    assert min_prob_actual >= probs_min * 0.999, \
        f"Probability threshold violation: min={min_prob_actual:.2e} < {probs_min:.2e}"
    
    return probs


def _compute_softmax_probs_4treat(theta: np.ndarray, probs_min: float = PROBS_MIN) -> np.ndarray:
    """
    Compute 4-treatment softmax probabilities.

    Uses numerically stable softmax computation to avoid exp overflow.
    """
    n = theta.shape[0]
    # Numerically stable softmax
    theta_with_baseline = np.column_stack([np.zeros(n), theta])  # (n, 4): [0, theta[:,0], theta[:,1], theta[:,2]]
    theta_max = theta_with_baseline.max(axis=1, keepdims=True)
    theta_stable = theta_with_baseline - theta_max

    exp_theta = np.exp(theta_stable)
    probs = exp_theta / exp_theta.sum(axis=1, keepdims=True)

    # Iterative clipping and renormalization for numerical stability (same as 3-treatment)
    max_iter = 10
    for iteration in range(max_iter):
        # Lower bound clipping
        probs_clipped = np.maximum(probs_min, probs)

        # Re-normalization
        probs_new = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)

        # Check convergence
        if np.all(probs_new >= probs_min * 0.999):
            probs = probs_new
            break

        probs = probs_new
        
        # If the last iteration still doesn't converge, issue warning
        if iteration == max_iter - 1:
            min_prob = probs.min()
            if min_prob < probs_min * 0.999:
                import warnings
                warnings.warn(
                    f"Iterative clipping did not fully converge: min_prob={min_prob:.2e} < {probs_min:.2e}. "
                    f"This may occur with extremely imbalanced data (probability >99.9999%).",
                    UserWarning
                )

    assert probs.shape == (n, 4) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-10), \
        f"Softmax probability error: shape={probs.shape}, sum range=[{probs.sum(axis=1).min()}, {probs.sum(axis=1).max()}]"
    
    # Verify probability threshold
    min_prob_actual = probs.min()
    assert min_prob_actual >= probs_min * 0.999, \
        f"Probability threshold violated: min={min_prob_actual:.2e} < {probs_min:.2e}"
    
    return probs


def _compute_contrast_weights_3treat(T1: np.ndarray, T2: np.ndarray, T3: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute contrast weights for 3-level treatment."""
    w_contrast = np.column_stack([
        2*T1/probs[:,0] - T2/probs[:,1] - T3/probs[:,2],
        T2/probs[:,1] - T3/probs[:,2]
    ])
    assert w_contrast.shape == (len(T1), 2)
    return w_contrast


def _compute_contrast_weights_4treat(T1: np.ndarray, T2: np.ndarray, T3: np.ndarray, T4: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute contrast weights for 4-level treatment."""
    w_contrast = np.column_stack([
        T1/probs[:,0] + T2/probs[:,1] - T3/probs[:,2] - T4/probs[:,3],
        T1/probs[:,0] - T2/probs[:,1] - T3/probs[:,2] + T4/probs[:,3],
        -T1/probs[:,0] + T2/probs[:,1] - T3/probs[:,2] + T4/probs[:,3]
    ])
    assert w_contrast.shape == (len(T1), 3)
    return w_contrast


def _compute_V_matrix_3treat(X: np.ndarray, probs: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                             T3: np.ndarray, wtX: np.ndarray, n: int) -> np.ndarray:
    """Compute V matrix (4k x 4k) for 3-level treatment."""
    k = X.shape[1]
    # 10 block matrices with proper broadcasting
    X_1_1 = wtX * (probs[:,1] * (1 - probs[:,1]))[:, None]
    X_2_2 = wtX * (probs[:,2] * (1 - probs[:,2]))[:, None]
    X_3_3 = wtX * (4*probs[:,0]**(-1) + probs[:,1]**(-1) + probs[:,2]**(-1))[:, None]
    X_4_4 = wtX * (probs[:,1]**(-1) + probs[:,2]**(-1))[:, None]
    X_1_2 = wtX * (-probs[:,1] * probs[:,2])[:, None]
    X_1_3 = wtX * (-1)
    X_1_4 = wtX * 1
    X_2_3 = wtX * (-1)
    X_2_4 = wtX * (-1)
    X_3_4 = wtX * (-probs[:,1]**(-1) + probs[:,2]**(-1))[:, None]
    # Assemble 4x4 block matrix
    V = (1.0/n) * np.block([[X_1_1.T @ X, X_1_2.T @ X, X_1_3.T @ X, X_1_4.T @ X],
                            [X_1_2.T @ X, X_2_2.T @ X, X_2_3.T @ X, X_2_4.T @ X],
                            [X_1_3.T @ X, X_2_3.T @ X, X_3_3.T @ X, X_3_4.T @ X],
                            [X_1_4.T @ X, X_2_4.T @ X, X_3_4.T @ X, X_4_4.T @ X]])
    assert V.shape == (4*k, 4*k) and np.allclose(V, V.T, atol=1e-12)
    return V


def _compute_V_matrix_4treat(X: np.ndarray, probs: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                             T3: np.ndarray, T4: np.ndarray, wtX: np.ndarray, n: int) -> np.ndarray:
    """Compute V matrix (6k x 6k) for 4-level treatment."""
    k = X.shape[1]
    # 21 block matrices with proper broadcasting
    X_1_1 = wtX * (probs[:,1] * (1 - probs[:,1]))[:, None]
    X_2_2 = wtX * (probs[:,2] * (1 - probs[:,2]))[:, None]
    X_3_3 = wtX * (probs[:,3] * (1 - probs[:,3]))[:, None]
    X_4_4 = wtX * (probs[:,0]**(-1) + probs[:,1]**(-1) + probs[:,2]**(-1) + probs[:,3]**(-1))[:, None]
    X_5_5 = X_4_4
    X_6_6 = X_4_4
    X_1_2 = wtX * (-probs[:,1] * probs[:,2])[:, None]
    X_1_3 = wtX * (-probs[:,1] * probs[:,3])[:, None]
    X_2_3 = wtX * (-probs[:,2] * probs[:,3])[:, None]
    X_1_4, X_1_6, X_3_5, X_3_6 = wtX, wtX, wtX, wtX
    X_1_5, X_2_4, X_2_5, X_2_6, X_3_4 = wtX * (-1), wtX * (-1), wtX * (-1), wtX * (-1), wtX * (-1)
    X_4_5 = wtX * (probs[:,0]**(-1) - probs[:,1]**(-1) + probs[:,2]**(-1) - probs[:,3]**(-1))[:, None]
    X_4_6 = wtX * (-probs[:,0]**(-1) + probs[:,1]**(-1) + probs[:,2]**(-1) - probs[:,3]**(-1))[:, None]
    X_5_6 = wtX * (-probs[:,0]**(-1) - probs[:,1]**(-1) + probs[:,2]**(-1) + probs[:,3]**(-1))[:, None]
    # Assemble 6x6 block matrix
    V = (1.0/n) * np.block([[X_1_1.T @ X, X_1_2.T @ X, X_1_3.T @ X, X_1_4.T @ X, X_1_5.T @ X, X_1_6.T @ X],
                            [X_1_2.T @ X, X_2_2.T @ X, X_2_3.T @ X, X_2_4.T @ X, X_2_5.T @ X, X_2_6.T @ X],
                            [X_1_3.T @ X, X_2_3.T @ X, X_3_3.T @ X, X_3_4.T @ X, X_3_5.T @ X, X_3_6.T @ X],
                            [X_1_4.T @ X, X_2_4.T @ X, X_3_4.T @ X, X_4_4.T @ X, X_4_5.T @ X, X_4_6.T @ X],
                            [X_1_5.T @ X, X_2_5.T @ X, X_3_5.T @ X, X_4_5.T @ X, X_5_5.T @ X, X_5_6.T @ X],
                            [X_1_6.T @ X, X_2_6.T @ X, X_3_6.T @ X, X_4_6.T @ X, X_5_6.T @ X, X_6_6.T @ X]])
    assert V.shape == (6*k, 6*k) and np.allclose(V, V.T, atol=1e-12)
    return V


def _gmm_func_3treat(beta_curr: np.ndarray, X: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     T3: np.ndarray, sample_weights: np.ndarray, n: int,
                     inv_V: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """GMM objective function for 3-level treatment."""
    k = X.shape[1]
    beta_curr = beta_curr.reshape(k, 2) if beta_curr.ndim == 1 else beta_curr
    theta = X @ beta_curr
    probs = _compute_softmax_probs_3treat(theta, PROBS_MIN)
    w_contrast = _compute_contrast_weights_3treat(T1, T2, T3, probs)
    wtX = sample_weights[:, None] * X
    w_curr_del = (1.0/n) * wtX.T @ w_contrast
    gbar = np.concatenate([(1.0/n) * wtX.T @ (T2 - probs[:,1]),
                          (1.0/n) * wtX.T @ (T3 - probs[:,2]),
                          w_curr_del.ravel(order='F')])
    if inv_V is None:
        V = _compute_V_matrix_3treat(X, probs, T1, T2, T3, wtX, n)
        inv_V = _r_ginv(V)
    loss = float(gbar.T @ inv_V @ gbar)
    return {'loss': loss, 'inv_V': inv_V}


def _gmm_func_4treat(beta_curr: np.ndarray, X: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     T3: np.ndarray, T4: np.ndarray, sample_weights: np.ndarray, n: int,
                     inv_V: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """GMM objective function for 4-level treatment."""
    k = X.shape[1]
    beta_curr = beta_curr.reshape(k, 3) if beta_curr.ndim == 1 else beta_curr
    theta = X @ beta_curr
    probs = _compute_softmax_probs_4treat(theta, PROBS_MIN)
    w_contrast = _compute_contrast_weights_4treat(T1, T2, T3, T4, probs)
    wtX = sample_weights[:, None] * X
    w_curr_del = (1.0/n) * wtX.T @ w_contrast
    gbar = np.concatenate([(1.0/n) * wtX.T @ (T2 - probs[:,1]),
                          (1.0/n) * wtX.T @ (T3 - probs[:,2]),
                          (1.0/n) * wtX.T @ (T4 - probs[:,3]),
                          w_curr_del.ravel(order='F')])
    if inv_V is None:
        V = _compute_V_matrix_4treat(X, probs, T1, T2, T3, T4, wtX, n)
        inv_V = _r_ginv(V)
    loss = float(gbar.T @ inv_V @ gbar)
    return {'loss': loss, 'inv_V': inv_V}


def _bal_loss_3treat(beta_curr: np.ndarray, X: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     T3: np.ndarray, sample_weights: np.ndarray, XprimeX_inv: np.ndarray,
                     k: int, n: int) -> float:
    """Balance loss function for 3-level treatment."""
    beta_mat = beta_curr.reshape(k, 2) if beta_curr.ndim == 1 else beta_curr
    theta = X @ beta_mat
    probs = _compute_softmax_probs_3treat(theta, PROBS_MIN)
    w_contrast = _compute_contrast_weights_3treat(T1, T2, T3, probs) / n  # Divide by n
    wtX = sample_weights[:, None] * X
    wtXprimew = wtX.T @ w_contrast
    loss = np.sum(np.diag(wtXprimew.T @ XprimeX_inv @ wtXprimew))
    return float(loss)


def _bal_loss_4treat(beta_curr: np.ndarray, X: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                     T3: np.ndarray, T4: np.ndarray, sample_weights: np.ndarray,
                     XprimeX_inv: np.ndarray, k: int, n: int) -> float:
    """Balance loss function for 4-level treatment."""
    beta_mat = beta_curr.reshape(k, 3) if beta_curr.ndim == 1 else beta_curr
    theta = X @ beta_mat
    probs = _compute_softmax_probs_4treat(theta, PROBS_MIN)
    w_contrast = _compute_contrast_weights_4treat(T1, T2, T3, T4, probs) / n
    wtX = sample_weights[:, None] * X
    wtXprimew = wtX.T @ w_contrast
    loss = np.sum(np.diag(wtXprimew.T @ XprimeX_inv @ wtXprimew))
    return float(loss)


def _mnlogit_init_3treat(treat: np.ndarray, X: np.ndarray, sample_weights: np.ndarray,
                         treat_levels: np.ndarray, k: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial logit initialization for 3-level treatment."""
    # Encode treat as 0,1,2 according to treat_levels order
    # Handle multiple types: treat may be integer, string, pd.Categorical, etc.
    
    # Convert to numpy array (handles pd.Categorical etc.)
    treat_array = np.asarray(treat)
    
    # Check if already integer encoded (check values, not dtype)
    try:
        treat_as_int = treat_array.astype(int)
        if np.array_equal(treat_as_int, treat_array) and np.all((treat_as_int >= 0) & (treat_as_int < len(treat_levels))):
            # treat is already valid integer encoding
            treat_encoded = treat_as_int
        else:
            raise ValueError("Re-encoding needed")
    except (ValueError, TypeError):
        # treat is not integer or needs re-encoding
        treat_map = {level: i for i, level in enumerate(treat_levels)}
        treat_encoded = np.array([treat_map[t] for t in treat_array])
    
    # Fit MNLogit with sample weights via row replication
    # (statsmodels.MNLogit doesn't support freq_weights)
    weights_unique = np.unique(sample_weights)
    if len(weights_unique) == 1:
        # Uniform weights, fit directly
        mnl_model = sm.MNLogit(treat_encoded, X)
        mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
    else:
        # Non-uniform weights, use row replication method
        # Normalize weights so minimum is 1
        min_weight = sample_weights.min()
        weights_normalized = sample_weights / min_weight

        # Check if can convert to integers (tolerance 1e-6)
        weights_int_candidate = np.round(weights_normalized)
        if np.allclose(weights_normalized, weights_int_candidate, atol=1e-6):
            # Use integer weight replication
            weights_int = weights_int_candidate.astype(int)
            X_expanded = np.repeat(X, weights_int, axis=0)
            treat_expanded = np.repeat(treat_encoded, weights_int)

            mnl_model = sm.MNLogit(treat_expanded, X_expanded)
            mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
        else:
            # Non-integer weights, use approximation
            # Scale weights to be closer to integers
            scale_factor = 100  # Adjustable
            weights_scaled = weights_normalized * scale_factor
            weights_int = np.round(weights_scaled).astype(int)
            weights_int = np.maximum(weights_int, 1)  # Ensure at least 1

            X_expanded = np.repeat(X, weights_int, axis=0)
            treat_expanded = np.repeat(treat_encoded, weights_int)

            mnl_model = sm.MNLogit(treat_expanded, X_expanded)
            mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
    # statsmodels returns params in (k, K-1) format (no transpose needed)
    mcoef = mnl_result.params  # shape (k, 2)
    # Handle NA coefficients
    mcoef[np.isnan(mcoef[:, 0]), 0] = 0
    mcoef[np.isnan(mcoef[:, 1]), 1] = 0
    # Compute MLE probabilities
    theta_mnl = X @ mcoef  # (n, 2)
    probs_mnl = _compute_softmax_probs_3treat(theta_mnl, PROBS_MIN)
    return mcoef, probs_mnl


def _mnlogit_init_4treat(treat: np.ndarray, X: np.ndarray, sample_weights: np.ndarray,
                         treat_levels: np.ndarray, k: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial logit initialization for 4-level treatment."""
    # Encode treat as 0,1,2,3 according to treat_levels order
    # Handle multiple types: pd.Categorical, etc.
    
    # Convert to numpy array
    treat_array = np.asarray(treat)
    
    # Check if already integer encoded
    try:
        treat_as_int = treat_array.astype(int)
        if np.array_equal(treat_as_int, treat_array) and np.all((treat_as_int >= 0) & (treat_as_int < len(treat_levels))):
            treat_encoded = treat_as_int
        else:
            raise ValueError("Re-encoding needed")
    except (ValueError, TypeError):
        treat_map = {level: i for i, level in enumerate(treat_levels)}
        treat_encoded = np.array([treat_map[t] for t in treat_array])
    
    # Fit MNLogit with sample weights via row replication
    weights_unique = np.unique(sample_weights)
    if len(weights_unique) == 1:
        # Uniform weights, fit directly
        mnl_model = sm.MNLogit(treat_encoded, X)
        mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
    else:
        # Non-uniform weights, use row replication
        min_weight = sample_weights.min()
        weights_normalized = sample_weights / min_weight

        weights_int_candidate = np.round(weights_normalized)
        if np.allclose(weights_normalized, weights_int_candidate, atol=1e-6):
            weights_int = weights_int_candidate.astype(int)
            X_expanded = np.repeat(X, weights_int, axis=0)
            treat_expanded = np.repeat(treat_encoded, weights_int)

            mnl_model = sm.MNLogit(treat_expanded, X_expanded)
            mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
        else:
            # Non-integer weights, use approximation
            scale_factor = 100
            weights_scaled = weights_normalized * scale_factor
            weights_int = np.round(weights_scaled).astype(int)
            weights_int = np.maximum(weights_int, 1)

            X_expanded = np.repeat(X, weights_int, axis=0)
            treat_expanded = np.repeat(treat_encoded, weights_int)

            mnl_model = sm.MNLogit(treat_expanded, X_expanded)
            mnl_result = mnl_model.fit(maxiter=100, disp=False, method='bfgs')
    # statsmodels returns params in (k, K-1) format
    mcoef = mnl_result.params  # shape (k, 3)
    mcoef[np.isnan(mcoef[:, 0]), 0] = 0
    mcoef[np.isnan(mcoef[:, 1]), 1] = 0
    mcoef[np.isnan(mcoef[:, 2]), 2] = 0
    # Compute MLE probabilities
    theta_mnl = X @ mcoef  # (n, 3)
    probs_mnl = _compute_softmax_probs_4treat(theta_mnl, PROBS_MIN)
    return mcoef, probs_mnl


def _standardize_weights_3treat(T1: np.ndarray, T2: np.ndarray, T3: np.ndarray,
                                probs_opt: np.ndarray, sample_weights: np.ndarray,
                                standardize: bool) -> np.ndarray:
    """Standardize weights for 3-level treatment."""
    if standardize:
        norm1 = np.sum(T1 * sample_weights / probs_opt[:,0])
        norm2 = np.sum(T2 * sample_weights / probs_opt[:,1])
        norm3 = np.sum(T3 * sample_weights / probs_opt[:,2])
    else:
        norm1 = norm2 = norm3 = 1.0
    w_opt = (T1 / probs_opt[:,0] / norm1 + 
             T2 / probs_opt[:,1] / norm2 + 
             T3 / probs_opt[:,2] / norm3)
    return w_opt


def _standardize_weights_4treat(T1: np.ndarray, T2: np.ndarray, T3: np.ndarray, T4: np.ndarray,
                                probs_opt: np.ndarray, sample_weights: np.ndarray,
                                standardize: bool) -> np.ndarray:
    """Standardize weights for 4-level treatment."""
    if standardize:
        norm1 = np.sum(T1 * sample_weights / probs_opt[:,0])
        norm2 = np.sum(T2 * sample_weights / probs_opt[:,1])
        norm3 = np.sum(T3 * sample_weights / probs_opt[:,2])
        norm4 = np.sum(T4 * sample_weights / probs_opt[:,3])
    else:
        norm1 = norm2 = norm3 = norm4 = 1.0
    w_opt = (T1 / probs_opt[:,0] / norm1 + T2 / probs_opt[:,1] / norm2 + 
             T3 / probs_opt[:,2] / norm3 + T4 / probs_opt[:,3] / norm4)
    return w_opt


def _check_and_fallback_to_mle(J_opt: float, beta_opt: np.ndarray, probs_opt: np.ndarray,
                               mcoef: np.ndarray, probs_mnl: np.ndarray,
                               gmm_loss_func: Any, bal_loss_func: Any) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Check MLE fallback with dual AND condition."""
    mle_J = gmm_loss_func(mcoef.ravel())
    mle_bal = bal_loss_func(mcoef.ravel())
    opt_bal = bal_loss_func(beta_opt.ravel())
    if (J_opt > mle_J) and (opt_bal > mle_bal):
        warnings.warn("Optimization failed.  Results returned are for MLE.")
        return mcoef, probs_mnl, mle_J, True
    return beta_opt, probs_opt, J_opt, False


def _compute_vcov_3treat(beta_opt: np.ndarray, probs_opt: np.ndarray, T1: np.ndarray,
                        T2: np.ndarray, T3: np.ndarray, X: np.ndarray,
                        sample_weights: np.ndarray, gmm_func: Any, n: int, k: int) -> np.ndarray:
    """Compute variance-covariance matrix for 3-level treatment."""
    wtX = sample_weights[:, None] * X
    # Recompute invV
    result = gmm_func(beta_opt.ravel(), inv_V=None)
    W = result['inv_V']
    # 8 XG block matrices with proper broadcasting
    XG_1_1 = (-wtX * (probs_opt[:,1] * (1 - probs_opt[:,1]))[:, None]).T @ X
    XG_1_2 = (wtX * (probs_opt[:,1] * probs_opt[:,2])[:, None]).T @ X
    XG_1_3 = (wtX * (2*T1*probs_opt[:,1]/probs_opt[:,0] + T2*(1-probs_opt[:,1])/probs_opt[:,1] - 
                     T3*probs_opt[:,1]/probs_opt[:,2])[:, None]).T @ X
    XG_1_4 = (wtX * (-T2*(1-probs_opt[:,1])/probs_opt[:,1] - T3*probs_opt[:,1]/probs_opt[:,2])[:, None]).T @ X
    XG_2_1 = (wtX * (probs_opt[:,1] * probs_opt[:,2])[:, None]).T @ X
    XG_2_2 = (-wtX * (probs_opt[:,2] * (1 - probs_opt[:,2]))[:, None]).T @ X
    XG_2_3 = (wtX * (2*T1*probs_opt[:,2]/probs_opt[:,0] - T2*probs_opt[:,2]/probs_opt[:,1] + 
                     T3*(1-probs_opt[:,2])/probs_opt[:,2])[:, None]).T @ X
    XG_2_4 = (wtX * (T2*probs_opt[:,2]/probs_opt[:,1] + T3*(1-probs_opt[:,2])/probs_opt[:,2])[:, None]).T @ X
    # Assemble G matrix (2k x 4k)
    G = (1.0/n) * np.vstack([
        np.hstack([XG_1_1, XG_1_2, XG_1_3, XG_1_4]),
        np.hstack([XG_2_1, XG_2_2, XG_2_3, XG_2_4])
    ])
    # W1 matrix (4k x n)
    XW_1 = X * (T2 - probs_opt[:,1])[:, None] * (sample_weights**0.5)[:, None]
    XW_2 = X * (T3 - probs_opt[:,2])[:, None] * (sample_weights**0.5)[:, None]
    XW_3 = X * (2*T1/probs_opt[:,0] - T2/probs_opt[:,1] - T3/probs_opt[:,2])[:, None] * (sample_weights**0.5)[:, None]
    XW_4 = X * (T2/probs_opt[:,1] - T3/probs_opt[:,2])[:, None] * (sample_weights**0.5)[:, None]
    W1 = np.vstack([XW_1.T, XW_2.T, XW_3.T, XW_4.T])
    # Omega matrix
    Omega = (1.0/n) * (W1 @ W1.T)
    # Sandwich formula
    GWG = G @ W @ G.T
    GWGinv = _r_ginv(GWG)
    GWGinvGW = GWGinv @ G @ W
    vcov = GWGinvGW @ Omega @ GWGinvGW.T
    assert vcov.shape == (2*k, 2*k)
    return vcov


def _compute_vcov_4treat(beta_opt: np.ndarray, probs_opt: np.ndarray, T1: np.ndarray,
                        T2: np.ndarray, T3: np.ndarray, T4: np.ndarray, X: np.ndarray,
                        sample_weights: np.ndarray, gmm_func: Any, n: int, k: int) -> np.ndarray:
    """Compute variance-covariance matrix for 4-level treatment."""
    wtX = sample_weights[:, None] * X
    result = gmm_func(beta_opt.ravel(), inv_V=None)
    W = result['inv_V']
    # 18 XG block matrices with proper broadcasting
    XG_1_1 = (-wtX * (probs_opt[:,1] * (1 - probs_opt[:,1]))[:, None]).T @ X
    XG_1_2 = (wtX * (probs_opt[:,1] * probs_opt[:,2])[:, None]).T @ X
    XG_1_3 = (wtX * (probs_opt[:,1] * probs_opt[:,3])[:, None]).T @ X
    XG_1_4 = (wtX * (probs_opt[:,1] * (T1/probs_opt[:,0] - T2*(1-probs_opt[:,1])/probs_opt[:,1]**2 - 
                                      T3/probs_opt[:,2] - T4/probs_opt[:,3]))[:, None]).T @ X
    XG_1_5 = (wtX * (probs_opt[:,1] * (T1/probs_opt[:,0] + T2*(1-probs_opt[:,1])/probs_opt[:,1]**2 - 
                                      T3/probs_opt[:,2] + T4/probs_opt[:,3]))[:, None]).T @ X
    XG_1_6 = (wtX * (probs_opt[:,1] * (-T1/probs_opt[:,0] - T2*(1-probs_opt[:,1])/probs_opt[:,1]**2 - 
                                       T3/probs_opt[:,2] + T4/probs_opt[:,3]))[:, None]).T @ X
    XG_2_1 = (wtX * (probs_opt[:,1] * probs_opt[:,2])[:, None]).T @ X
    XG_2_2 = (-wtX * (probs_opt[:,2] * (1 - probs_opt[:,2]))[:, None]).T @ X
    XG_2_3 = (wtX * (probs_opt[:,2] * probs_opt[:,3])[:, None]).T @ X
    XG_2_4 = (wtX * (probs_opt[:,2] * (T1/probs_opt[:,0] + T2/probs_opt[:,1] + 
                                      T3*(1-probs_opt[:,2])/probs_opt[:,2]**2 - T4/probs_opt[:,3]))[:, None]).T @ X
    XG_2_5 = (wtX * (probs_opt[:,2] * (T1/probs_opt[:,0] - T2/probs_opt[:,1] + 
                                      T3*(1-probs_opt[:,2])/probs_opt[:,2]**2 + T4/probs_opt[:,3]))[:, None]).T @ X
    XG_2_6 = (wtX * (probs_opt[:,2] * (-T1/probs_opt[:,0] + T2/probs_opt[:,1] + 
                                       T3*(1-probs_opt[:,2])/probs_opt[:,2]**2 + T4/probs_opt[:,3]))[:, None]).T @ X
    XG_3_1 = (wtX * (probs_opt[:,1] * probs_opt[:,3])[:, None]).T @ X
    XG_3_2 = (wtX * (probs_opt[:,2] * probs_opt[:,3])[:, None]).T @ X
    XG_3_3 = (-wtX * (probs_opt[:,3] * (1 - probs_opt[:,3]))[:, None]).T @ X
    XG_3_4 = (wtX * (probs_opt[:,3] * (T1/probs_opt[:,0] + T2/probs_opt[:,1] - 
                                      T3/probs_opt[:,2] + T4*(1-probs_opt[:,3])/probs_opt[:,3]**2))[:, None]).T @ X
    XG_3_5 = (wtX * (probs_opt[:,3] * (T1/probs_opt[:,0] - T2/probs_opt[:,1] - 
                                      T3/probs_opt[:,2] - T4*(1-probs_opt[:,3])/probs_opt[:,3]**2))[:, None]).T @ X
    XG_3_6 = (wtX * (probs_opt[:,3] * (-T1/probs_opt[:,0] + T2/probs_opt[:,1] - 
                                       T3/probs_opt[:,2] - T4*(1-probs_opt[:,3])/probs_opt[:,3]**2))[:, None]).T @ X
    # G matrix (3k x 6k)
    G = (1.0/n) * np.vstack([
        np.hstack([XG_1_1, XG_1_2, XG_1_3, XG_1_4, XG_1_5, XG_1_6]),
        np.hstack([XG_2_1, XG_2_2, XG_2_3, XG_2_4, XG_2_5, XG_2_6]),
        np.hstack([XG_3_1, XG_3_2, XG_3_3, XG_3_4, XG_3_5, XG_3_6])
    ])
    # W1 matrix (6k x n)
    XW_1 = X * (T2 - probs_opt[:,1])[:, None] * (sample_weights**0.5)[:, None]
    XW_2 = X * (T3 - probs_opt[:,2])[:, None] * (sample_weights**0.5)[:, None]
    XW_3 = X * (T4 - probs_opt[:,3])[:, None] * (sample_weights**0.5)[:, None]
    XW_4 = X * (T1/probs_opt[:,0] + T2/probs_opt[:,1] - T3/probs_opt[:,2] - T4/probs_opt[:,3])[:, None] * (sample_weights**0.5)[:, None]
    XW_5 = X * (T1/probs_opt[:,0] - T2/probs_opt[:,1] - T3/probs_opt[:,2] + T4/probs_opt[:,3])[:, None] * (sample_weights**0.5)[:, None]
    XW_6 = X * (-T1/probs_opt[:,0] + T2/probs_opt[:,1] - T3/probs_opt[:,2] + T4/probs_opt[:,3])[:, None] * (sample_weights**0.5)[:, None]
    W1 = np.vstack([XW_1.T, XW_2.T, XW_3.T, XW_4.T, XW_5.T, XW_6.T])
    # Omega matrix
    Omega = (1.0/n) * (W1 @ W1.T)
    # Sandwich formula
    GWG = G @ W @ G.T
    GWGinv = _r_ginv(GWG)
    GWGinvGW = GWGinv @ G @ W
    vcov = GWGinvGW @ Omega @ GWGinvGW.T
    assert vcov.shape == (3*k, 3*k)
    return vcov


def cbps_3treat_fit(
    treat: np.ndarray,
    X: np.ndarray,
    method: str = 'over',
    k: int = None,
    XprimeX_inv: np.ndarray = None,
    bal_only: bool = False,
    iterations: int = 1000,
    standardize: bool = True,
    two_step: bool = True,
    sample_weights: np.ndarray = None,
    treat_levels: np.ndarray = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Fit CBPS for 3-level categorical treatments.

    This function implements the full CBPS algorithm for treatments with
    exactly three levels, using multinomial logistic regression for
    initialization and GMM optimization for covariate balance.

    Parameters
    ----------
    treat : np.ndarray
        Treatment indicator with 3 levels, shape (n,).
    X : np.ndarray, shape (n, k)
        Covariate matrix (SVD-orthogonalized if applicable).
    method : str, default 'over'
        Estimation method: 'over' for overidentified GMM,
        'exact' for exactly identified GMM.
    k : int
        Rank of covariate matrix after SVD.
    XprimeX_inv : np.ndarray, shape (k, k)
        Inverse of X'X matrix for balance loss computation.
    bal_only : bool, default False
        If True, use balance constraints only.
        If False, include score constraints (overidentified).
    iterations : int, default 1000
        Maximum number of optimization iterations.
    standardize : bool, default True
        If True, apply weight standardization.
        If False, use Horvitz-Thompson weights.
    two_step : bool, default True
        If True, use two-step GMM with pre-computed invV.
        If False, use continuous-updating GMM.
    sample_weights : np.ndarray, optional
        Sampling weights. If None, defaults to uniform weights.
    treat_levels : np.ndarray, optional
        Treatment level values for labeling.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing fitted model results including:
        - coefficients: Estimated coefficients
        - fitted_values: Propensity scores
        - weights: CBPS weights
        - Additional diagnostic information
        
        Keys include:
        - coefficients: Coefficients in orthogonal space, shape (k, 2)
        - fitted_values: Probability matrix, shape (n, 3)
        - linear_predictor: Linear predictor values, shape (n, 2)
        - weights: ATE weights, shape (n,)
        - y: Treatment indicator vector
        - x: Orthogonalized covariate matrix
        - J: J-statistic for overidentification test
        - mle_J: MLE J-statistic
        - deviance: Negative twice log-likelihood
        - converged: Convergence status
        - var: Covariance matrix in orthogonal space, shape (2k, 2k)

    Algorithm Flow
    -------------
    1. Initialize constants and treatment indicators
    2. MNLogit initialization
    3. Alpha scaling
    4. Balance optimization
    5. Return if bal_only=True
    6. GMM dual initialization optimization
    7. Compute optimal probabilities
    8. Calculate J-statistic
    9. Check for MLE fallback
    10. Compute deviance and weight standardization
    11. Compute covariance matrix
    12. Construct return object

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # ========== Initialization ==========
    # Step 0: Define n first due to Python scoping requirements
    n = len(treat)
    
    # Step 1: Treatment levels and indicators
    if treat_levels is None:
        treat_levels = np.unique(treat)
    assert len(treat_levels) == 3, "Must be 3-valued treatment"
    
    T1 = (treat == treat_levels[0]).astype(float)
    T2 = (treat == treat_levels[1]).astype(float)
    T3 = (treat == treat_levels[2]).astype(float)
    
    # Step 2: Normalize sample_weights
    sample_weights = normalize_sample_weights(sample_weights, n)
    
    # Step 3: Compute k and XprimeX_inv
    if k is None:
        k = X.shape[1]
    if XprimeX_inv is None:
        wtX_sqrt = (sample_weights**0.5)[:, None] * X
        XprimeX_inv = _r_ginv(wtX_sqrt.T @ wtX_sqrt)
    
    # ========== Define closure functions (using external variables) ==========
    def gmm_loss(beta):
        return _gmm_func_3treat(beta, X, T1, T2, T3, sample_weights, n, None)['loss']
    
    def bal_loss(beta):
        return _bal_loss_3treat(beta, X, T1, T2, T3, sample_weights, XprimeX_inv, k, n)
    
    # ========== MNLogit initialization ==========
    mcoef, probs_mnl = _mnlogit_init_3treat(treat, X, sample_weights, treat_levels, k, n)
    
    # ========== Alpha scaling ==========
    def alpha_func(alpha):
        return gmm_loss(mcoef.ravel() * alpha)
    alpha_result = scipy.optimize.minimize_scalar(alpha_func, bounds=(0.8, 1.1), method='bounded')
    gmm_init = mcoef.ravel() * alpha_result.x
    
    # ========== Pre-compute invV (two-step method) ==========
    this_invV = _gmm_func_3treat(gmm_init, X, T1, T2, T3, sample_weights, n, None)['inv_V']
    
    # ========== Balance optimization ==========
    if verbose >= 1:
        print(f"[CBPS 3-Treat] Starting balance optimization (max_iter={iterations})...")
    
    if two_step:
        opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='BFGS',
                                         options={'maxiter': iterations})
        if verbose >= 1:
            print(f"[CBPS 3-Treat] Balance optimization complete: loss={opt_bal.fun:.6f}, converged={opt_bal.success}")
    else:
        try:
            opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='BFGS',
                                             options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='Nelder-Mead',
                                             options={'maxiter': iterations})
    
    beta_bal = opt_bal.x
    
    # ========== Compute nulldeviance (before all return paths) ==========
    # Null model: each category's probability = its sample proportion
    T1_mean = np.average(T1, weights=sample_weights)
    T2_mean = np.average(T2, weights=sample_weights)
    T3_mean = np.average(T3, weights=sample_weights)
    # Prevent log(0)
    T1_mean = np.clip(T1_mean, 1e-10, 1.0)
    T2_mean = np.clip(T2_mean, 1e-10, 1.0)
    T3_mean = np.clip(T3_mean, 1e-10, 1.0)
    nulldeviance = -2 * np.sum(T1 * np.log(T1_mean) + T2 * np.log(T2_mean) + T3 * np.log(T3_mean))
    
    # ========== bal_only early return ==========
    if bal_only:
        beta_opt = beta_bal.reshape(k, 2)
        theta_opt = X @ beta_opt
        probs_opt = _compute_softmax_probs_3treat(theta_opt, PROBS_MIN)
        w_opt = _standardize_weights_3treat(T1, T2, T3, probs_opt, sample_weights, standardize)
        J_opt = bal_loss(beta_opt.ravel())
        deviance = -2 * np.sum(T1 * np.log(probs_opt[:,0]) + T2 * np.log(probs_opt[:,1]) + T3 * np.log(probs_opt[:,2]))
        vcov = _compute_vcov_3treat(beta_opt, probs_opt, T1, T2, T3, X, sample_weights,
                                   lambda b, inv_V=None: _gmm_func_3treat(b, X, T1, T2, T3, sample_weights, n, inv_V),
                                   n, k)
        mle_J_val = _gmm_func_3treat(mcoef.ravel(), X, T1, T2, T3, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(mcoef.ravel())
        return {'coefficients': beta_opt, 'fitted_values': probs_opt, 'linear_predictor': theta_opt,
                'deviance': deviance, 'nulldeviance': nulldeviance, 'weights': w_opt * sample_weights, 'y': treat, 'x': X,
                'converged': opt_bal.success, 'J': J_opt, 'var': vcov, 'mle_J': mle_J_val}
    
    # ========== GMM dual initialization selection ==========
    def gmm_loss_with_invV(beta):
        return _gmm_func_3treat(beta, X, T1, T2, T3, sample_weights, n, this_invV)['loss']
    
    if two_step:
        gmm_glm_init = scipy.optimize.minimize(gmm_loss_with_invV, gmm_init, method='BFGS',
                                              options={'maxiter': iterations})
        gmm_bal_init = scipy.optimize.minimize(gmm_loss_with_invV, beta_bal, method='BFGS',
                                              options={'maxiter': iterations})
    else:
        try:
            gmm_glm_init = scipy.optimize.minimize(gmm_loss, gmm_init, method='BFGS',
                                                  options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            gmm_glm_init = scipy.optimize.minimize(gmm_loss, gmm_init, method='Nelder-Mead',
                                                  options={'maxiter': iterations})
        try:
            gmm_bal_init = scipy.optimize.minimize(gmm_loss, beta_bal, method='BFGS',
                                                  options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            gmm_bal_init = scipy.optimize.minimize(gmm_loss, beta_bal, method='Nelder-Mead',
                                                  options={'maxiter': iterations})
    
    # Select the optimization result with lower loss
    opt1 = gmm_glm_init if gmm_glm_init.fun < gmm_bal_init.fun else gmm_bal_init
    
    # ========== Optimal probabilities and J-statistic ==========
    beta_opt = opt1.x.reshape(k, 2)
    theta_opt = X @ beta_opt
    probs_opt = _compute_softmax_probs_3treat(theta_opt, PROBS_MIN)
    J_opt = _gmm_func_3treat(beta_opt.ravel(), X, T1, T2, T3, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(beta_opt.ravel())
    
    # ========== MLE fallback check ==========
    beta_opt, probs_opt, J_opt, used_mle = _check_and_fallback_to_mle(
        J_opt, beta_opt, probs_opt, mcoef, probs_mnl, gmm_loss, bal_loss
    )
    
    # ========== Deviance and weights ==========
    deviance = -2 * np.sum(T1 * np.log(probs_opt[:,0]) + T2 * np.log(probs_opt[:,1]) + T3 * np.log(probs_opt[:,2]))
    
    # Null deviance already computed above
    
    w_opt = _standardize_weights_3treat(T1, T2, T3, probs_opt, sample_weights, standardize)
    
    # ========== Vcov computation ==========
    vcov = _compute_vcov_3treat(beta_opt, probs_opt, T1, T2, T3, X, sample_weights,
                               lambda b, inv_V=None: _gmm_func_3treat(b, X, T1, T2, T3, sample_weights, n, inv_V),
                               n, k)
    
    # ========== Return dict ==========
    mle_J_val = _gmm_func_3treat(mcoef.ravel(), X, T1, T2, T3, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(mcoef.ravel())
    
    # Enhanced non-convergence warning
    if not opt1.success:
        warnings.warn(
            f"Multi-valued CBPS (3-treat) optimization did not converge (converged=False). "
            f"Results may be unreliable. Consider:\n"
            f"  1. Increasing iterations (current: {iterations})\n"
            f"  2. Checking for perfect separation or collinearity\n"
            f"  3. Examining the balance diagnostics\n"
            f"  4. J-statistic: {J_opt:.6f}\n"
            f"  5. Trying different starting values or method='exact'",
            UserWarning,
            stacklevel=2
        )
    
    return {
        'coefficients': beta_opt,
        'fitted_values': probs_opt,
        'linear_predictor': theta_opt,
        'deviance': deviance,
        'nulldeviance': nulldeviance,
        'weights': w_opt * sample_weights,
        'y': treat,
        'x': X,
        'converged': opt1.success,
        'J': J_opt,
        'var': vcov,
        'mle_J': mle_J_val
    }


def cbps_4treat_fit(
    treat: np.ndarray,
    X: np.ndarray,
    method: str = 'over',
    k: int = None,
    XprimeX_inv: np.ndarray = None,
    bal_only: bool = False,
    iterations: int = 1000,
    standardize: bool = True,
    two_step: bool = True,
    sample_weights: np.ndarray = None,
    treat_levels: np.ndarray = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    4-valued treatment CBPS fitting function (complete workflow).
    
    Four-valued treatment CBPS estimator using GMM optimization.
    
    Parameters
    ----------
    Same as cbps_3treat_fit, but treatment has 4 levels.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 11 core attributes.
        
        Keys include (same structure as 3-treat):
        - coefficients: (k, 3) orthogonal space coefficients (4-treat needs 3 columns)
        - fitted_values: (n, 4) probability matrix (4 columns)
        - linear_predictor: (n, 3) linear predictor (3 columns)
        - weights: (n,) ATE weights
        - var: (3k, 3k) orthogonal space vcov (larger for 4-treat)
        - Other fields same as 3-treat
    
    Algorithm Flow
    --------------
    Mostly same as cbps_3treat_fit, main differences:
    - K=4 levels → 3 coefficient columns
    - softmax computes 4 probability columns
    - contrast weights 3 columns (3 of 6 pairwise contrasts)
    - V matrix expands to (6k, 6k) (15 blocks)
    - G matrix expands to (3k, 6k)
    
    Notes
    -----
    The 4-treat specific invV selection logic chooses between GMM and balance
    initialization based on which yields lower GMM loss.
    """
    # ========== Initialization ==========
    n = len(treat)
    
    if treat_levels is None:
        treat_levels = np.unique(treat)
    assert len(treat_levels) == 4, "Must be 4-valued treatment"
    
    T1 = (treat == treat_levels[0]).astype(float)
    T2 = (treat == treat_levels[1]).astype(float)
    T3 = (treat == treat_levels[2]).astype(float)
    T4 = (treat == treat_levels[3]).astype(float)
    
    sample_weights = normalize_sample_weights(sample_weights, n)
    
    if k is None:
        k = X.shape[1]
    if XprimeX_inv is None:
        wtX_sqrt = (sample_weights**0.5)[:, None] * X
        XprimeX_inv = _r_ginv(wtX_sqrt.T @ wtX_sqrt)
    
    # ========== Define closure functions ==========
    def gmm_loss(beta):
        return _gmm_func_4treat(beta, X, T1, T2, T3, T4, sample_weights, n, None)['loss']
    
    def bal_loss(beta):
        return _bal_loss_4treat(beta, X, T1, T2, T3, T4, sample_weights, XprimeX_inv, k, n)
    
    # ========== MNLogit initialization ==========
    mcoef, probs_mnl = _mnlogit_init_4treat(treat, X, sample_weights, treat_levels, k, n)
    
    # ========== Alpha scaling ==========
    def alpha_func(alpha):
        return gmm_loss(mcoef.ravel() * alpha)
    alpha_result = scipy.optimize.minimize_scalar(alpha_func, bounds=(0.8, 1.1), method='bounded')
    gmm_init = mcoef.ravel() * alpha_result.x
    
    # ========== Pre-compute invV ==========
    temp_invV = _gmm_func_4treat(gmm_init, X, T1, T2, T3, T4, sample_weights, n, None)['inv_V']
    
    # ========== Balance optimization ==========
    if verbose >= 1:
        print(f"[CBPS 4-Treat] Starting balance optimization (max_iter={iterations})...")
    
    if two_step:
        opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='BFGS',
                                         options={'maxiter': iterations})
        if verbose >= 1:
            print(f"[CBPS 4-Treat] Balance optimization complete: loss={opt_bal.fun:.6f}, converged={opt_bal.success}")
    else:
        try:
            opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='BFGS',
                                             options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            opt_bal = scipy.optimize.minimize(bal_loss, gmm_init, method='Nelder-Mead',
                                             options={'maxiter': iterations})
    
    beta_bal = opt_bal.x
    
    # ========== Compute nulldeviance (before all return paths) ==========
    # Null model: each category's probability = its sample proportion
    T1_mean = np.average(T1, weights=sample_weights)
    T2_mean = np.average(T2, weights=sample_weights)
    T3_mean = np.average(T3, weights=sample_weights)
    T4_mean = np.average(T4, weights=sample_weights)
    T1_mean = np.clip(T1_mean, 1e-10, 1.0)
    T2_mean = np.clip(T2_mean, 1e-10, 1.0)
    T3_mean = np.clip(T3_mean, 1e-10, 1.0)
    T4_mean = np.clip(T4_mean, 1e-10, 1.0)
    nulldeviance = -2 * np.sum(T1 * np.log(T1_mean) + T2 * np.log(T2_mean) +
                               T3 * np.log(T3_mean) + T4 * np.log(T4_mean))
    
    # ========== 4-treat specific: invV selection logic ==========
    if two_step:
        if gmm_loss(gmm_init) < gmm_loss(beta_bal):
            this_invV = _gmm_func_4treat(gmm_init, X, T1, T2, T3, T4, sample_weights, n, None)['inv_V']
        else:
            this_invV = _gmm_func_4treat(beta_bal, X, T1, T2, T3, T4, sample_weights, n, None)['inv_V']
        if bal_only:
            this_invV = _gmm_func_4treat(beta_bal, X, T1, T2, T3, T4, sample_weights, n, None)['inv_V']
    else:
        this_invV = temp_invV
    
    # ========== bal_only early return ==========
    if bal_only:
        beta_opt = beta_bal.reshape(k, 3)
        theta_opt = X @ beta_opt
        probs_opt = _compute_softmax_probs_4treat(theta_opt, PROBS_MIN)
        w_opt = _standardize_weights_4treat(T1, T2, T3, T4, probs_opt, sample_weights, standardize)
        J_opt = bal_loss(beta_opt.ravel())
        deviance = -2 * np.sum(T1 * np.log(probs_opt[:,0]) + T2 * np.log(probs_opt[:,1]) +
                              T3 * np.log(probs_opt[:,2]) + T4 * np.log(probs_opt[:,3]))
        vcov = _compute_vcov_4treat(beta_opt, probs_opt, T1, T2, T3, T4, X, sample_weights,
                                   lambda b, inv_V=None: _gmm_func_4treat(b, X, T1, T2, T3, T4, sample_weights, n, inv_V),
                                   n, k)
        mle_J_val = _gmm_func_4treat(mcoef.ravel(), X, T1, T2, T3, T4, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(mcoef.ravel())
        return {'coefficients': beta_opt, 'fitted_values': probs_opt, 'linear_predictor': theta_opt,
                'deviance': deviance, 'nulldeviance': nulldeviance, 'weights': w_opt * sample_weights, 'y': treat, 'x': X,
                'converged': opt_bal.success, 'J': J_opt, 'var': vcov, 'mle_J': mle_J_val}
    
    # ========== GMM dual initialization selection ==========
    def gmm_loss_with_invV(beta):
        return _gmm_func_4treat(beta, X, T1, T2, T3, T4, sample_weights, n, this_invV)['loss']
    
    if two_step:
        gmm_glm_init = scipy.optimize.minimize(gmm_loss_with_invV, gmm_init, method='BFGS',
                                              options={'maxiter': iterations})
        gmm_bal_init = scipy.optimize.minimize(gmm_loss_with_invV, beta_bal, method='BFGS',
                                              options={'maxiter': iterations})
    else:
        try:
            gmm_glm_init = scipy.optimize.minimize(gmm_loss, gmm_init, method='BFGS',
                                                  options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            gmm_glm_init = scipy.optimize.minimize(gmm_loss, gmm_init, method='Nelder-Mead',
                                                  options={'maxiter': iterations})
        try:
            gmm_bal_init = scipy.optimize.minimize(gmm_loss, beta_bal, method='BFGS',
                                                  options={'maxiter': iterations})
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            gmm_bal_init = scipy.optimize.minimize(gmm_loss, beta_bal, method='Nelder-Mead',
                                                  options={'maxiter': iterations})
    
    opt1 = gmm_glm_init if gmm_glm_init.fun < gmm_bal_init.fun else gmm_bal_init
    
    # ========== Optimal probabilities and J-statistic ==========
    beta_opt = opt1.x.reshape(k, 3)
    theta_opt = X @ beta_opt
    probs_opt = _compute_softmax_probs_4treat(theta_opt, PROBS_MIN)
    J_opt = _gmm_func_4treat(beta_opt.ravel(), X, T1, T2, T3, T4, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(beta_opt.ravel())
    
    # ========== MLE fallback check ==========
    beta_opt, probs_opt, J_opt, used_mle = _check_and_fallback_to_mle(
        J_opt, beta_opt, probs_opt, mcoef, probs_mnl, gmm_loss, bal_loss
    )
    
    # ========== Deviance and weights ==========
    deviance = -2 * np.sum(T1 * np.log(probs_opt[:,0]) + T2 * np.log(probs_opt[:,1]) +
                          T3 * np.log(probs_opt[:,2]) + T4 * np.log(probs_opt[:,3]))
    
    # Null deviance already computed above
    
    w_opt = _standardize_weights_4treat(T1, T2, T3, T4, probs_opt, sample_weights, standardize)
    
    # ========== Vcov computation ==========
    vcov = _compute_vcov_4treat(beta_opt, probs_opt, T1, T2, T3, T4, X, sample_weights,
                               lambda b, inv_V=None: _gmm_func_4treat(b, X, T1, T2, T3, T4, sample_weights, n, inv_V),
                               n, k)
    
    # ========== Return dict ==========
    mle_J_val = _gmm_func_4treat(mcoef.ravel(), X, T1, T2, T3, T4, sample_weights, n, this_invV)['loss'] if two_step else gmm_loss(mcoef.ravel())
    
    # Enhanced non-convergence warning
    if not opt1.success:
        warnings.warn(
            f"Multi-valued CBPS (4-treat) optimization did not converge (converged=False). "
            f"Results may be unreliable. Consider:\n"
            f"  1. Increasing iterations (current: {iterations})\n"
            f"  2. Checking for perfect separation or collinearity\n"
            f"  3. Examining the balance diagnostics\n"
            f"  4. J-statistic: {J_opt:.6f}\n"
            f"  5. Trying different starting values or method='exact'",
            UserWarning,
            stacklevel=2
        )
    
    return {
        'coefficients': beta_opt,
        'fitted_values': probs_opt,
        'linear_predictor': theta_opt,
        'deviance': deviance,
        'nulldeviance': nulldeviance,
        'weights': w_opt * sample_weights,
        'y': treat,
        'x': X,
        'converged': opt1.success,
        'J': J_opt,
        'var': vcov,
        'mle_J': mle_J_val
    }
