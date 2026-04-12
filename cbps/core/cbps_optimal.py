"""
Optimal Covariate Balancing Propensity Score (oCBPS)
====================================================

This module implements optimal CBPS (oCBPS) that extends the standard
CBPS by incorporating dual balancing conditions for improved efficiency
and robustness through the framework of Fan et al. (2022).

The implementation achieves double robustness and semiparametric efficiency
by separating the covariate balancing conditions for baseline outcome models
and treatment effect heterogeneity models.

Key Innovations
---------------
1. **Dual Balancing Conditions** (Fan 2022 Eq. 3.2-3.3):
   - g1_baseline: Balance covariates h1 related to E(Y(0)|X)
   - g2_diff: Balance covariates h2 related to E(Y(1)-Y(0)|X)

2. **Double Robustness** (Theorem 3.1):
   Consistent if either the propensity score model or outcome model is correct.

3. **Semiparametric Efficiency** (Corollary 3.2):
   Achieves Hahn 1998 efficiency bound when both models are correct and m=q.

Implementation Notes
--------------------
- Only supports att=0 (ATE estimation)
- No sample_weights parameter (oCBPS does not support sampling weights)
- Dual initialization optimization for robust convergence

References
----------
.. [1] Fan, Jianqing, Kosuke Imai, Inbeom Lee, Han Liu, Yang Ning,
       and Xiaolin Yang. 2022.
       "Optimal Covariate Balancing Conditions in Propensity Score
       Estimation."
       Journal of Business & Economic Statistics, 41(1), 97-110.
       https://doi.org/10.1080/07350015.2021.2002159
       https://imai.fas.harvard.edu/research/CBPStheory.html

.. [2] Imai, Kosuke and Marc Ratkovic. 2014.
       "Covariate Balancing Propensity Score."
       Journal of the Royal Statistical Society, Series B.
       DOI:10.1111/rssb.12027

Examples
--------
>>> from cbps import CBPS
>>> from cbps.datasets import load_lalonde
>>>
>>> # Load LaLonde data
>>> lalonde = load_lalonde()
>>>
>>> # oCBPS estimation with dual formula specification
>>> # Note: require m1 + m2 + 1 >= k where k is number of parameters
>>> fit = CBPS(
...     formula='treat ~ age + educ + re75 + re74',
...     data=lalonde,
...     baseline_formula='~age + educ + re75 + re74',
...     diff_formula='~I(re75==0)',
...     att=0  # oCBPS only supports ATE
... )
>>>
>>> # View results
>>> print(fit.summary())
>>> print(f"J-statistic: {fit.J:.6f}")
"""

from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize
import statsmodels.api as sm

# Import generalized inverse function from cbps_binary
from .cbps_binary import _r_ginv

# Constants
PROBS_MIN = 1e-6  # Probability clipping threshold for numerical stability


def _gmm_func1(
    beta_curr: np.ndarray,
    X: np.ndarray,
    treat: np.ndarray,
    baseline_X: np.ndarray,
    diff_X: np.ndarray,
    invV: Optional[np.ndarray] = None,
    option: Optional[str] = None
) -> Dict[str, Any]:
    """
    GMM objective function for optimal CBPS with dual balancing conditions.

    Parameters
    ----------
    beta_curr : np.ndarray
        Current propensity score coefficients (k-dimensional vector).
    X : np.ndarray
        Covariate matrix (n x k, including intercept column).
    treat : np.ndarray
        Binary treatment vector (0/1 encoded).
    baseline_X : np.ndarray
        Design matrix from baseline formula (n x m1).
    diff_X : np.ndarray
        Design matrix from diff formula (n x m2).
    invV : np.ndarray, optional
        Precomputed inverse of V matrix (for two-step GMM).
    option : str, optional
        None for dual balancing (oCBPS standard), "CBPS" for single balancing
        (used in pre-optimization).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'loss': GMM loss value (quadratic form)
        - 'invV': Generalized inverse of the V matrix

    Notes
    -----
    The dual balancing conditions (Fan et al. 2022, Eq. 3.2-3.3):

    - g1_baseline: Balance covariates related to E(Y(0)|X)
    - g2_diff: Balance covariates related to E(Y(1)-Y(0)|X)

    When option="CBPS", uses standard single balance condition for pre-optimization.
    """
    # Step 1: Sample size
    n = X.shape[0]

    # Step 2: Compute propensity scores
    theta_curr = X @ beta_curr
    probs_curr = scipy.special.expit(theta_curr)

    # Sequential clipping (upper bound then lower bound)
    probs_curr = np.minimum(1 - PROBS_MIN, probs_curr)
    probs_curr = np.maximum(PROBS_MIN, probs_curr)

    # Step 3: Compute ATE weights
    w_curr = treat / probs_curr - (1 - treat) / (1 - probs_curr)

    # Step 4: Construct moment conditions based on option
    if option is None:
        # Dual balancing conditions (oCBPS standard)
        # Construct X1new: intercept + baseline covariates
        X1new = np.column_stack([X[:, 0], baseline_X])

        # g1_baseline balance condition
        w_curr_del1 = (1/n) * (X1new.T @ w_curr)

        # g2_diff weights
        w_curr3 = treat / probs_curr - 1

        # g2_diff balance condition
        w_curr_del3 = (1/n) * (diff_X.T @ w_curr3)

        # Concatenate dual balance conditions
        gbar = np.concatenate([w_curr_del1, w_curr_del3])

    elif option == "CBPS":
        # Single balance condition (for pre-optimization)
        # Standard ATE balance using full X matrix
        w_curr_del = (1/n) * (X.T @ w_curr)
        gbar = w_curr_del

    else:
        raise ValueError(f"Unknown option: {option}")

    # Step 5: Compute covariance matrix V and its inverse
    if invV is None:
        # Reconstruct X1new (not defined when option="CBPS")
        X1new = np.column_stack([X[:, 0], baseline_X])

        # Block 1: V11
        factor_1 = ((1 - probs_curr) * probs_curr)**(-0.5)
        X_1 = X1new * factor_1[:, None]

        # Block 2: V22
        factor_2 = (1/probs_curr - 1)**0.5
        X_2 = diff_X * factor_2[:, None]

        # Block 3: V12
        X_1_1 = X1new * (probs_curr**(-0.5))[:, None]

        # Block 4: V21
        X_1_2 = diff_X * (probs_curr**(-0.5))[:, None]

        # Assemble V matrix
        V11 = (1/n) * (X_1.T @ X_1)
        V12 = (1/n) * (X_1_1.T @ X_1_2)
        V21 = (1/n) * (X_1_2.T @ X_1_1)
        V22 = (1/n) * (X_2.T @ X_2)

        V = np.block([[V11, V12],
                      [V21, V22]])

        # Generalized inverse
        invV_g = _r_ginv(V)
    else:
        invV_g = invV

    # Step 6: Compute GMM loss (quadratic form)
    loss = float(gbar.T @ invV_g @ gbar)

    return {'loss': loss, 'invV': invV_g}


def _gmm_loss1(beta: np.ndarray, *args, **kwargs) -> float:
    """
    GMM loss function wrapper for scipy.optimize.

    Parameters
    ----------
    beta : np.ndarray
        Propensity score coefficients.
    *args, **kwargs
        Arguments passed to _gmm_func1.

    Returns
    -------
    float
        GMM loss value.
    """
    return _gmm_func1(beta, *args, **kwargs)['loss']


def cbps_optimal_2treat(
    treat: np.ndarray,
    X: np.ndarray,
    baseline_X: np.ndarray,
    diff_X: np.ndarray,
    iterations: int = 1000,
    att: int = 0,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Optimal CBPS for binary treatments with double robustness and efficiency.

    Implements the optimal covariate balancing conditions from Fan et al.
    (2022), achieving double robustness and semiparametric efficiency by
    separating balance conditions for baseline outcome and treatment effect
    heterogeneity models.

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment vector (0/1 encoded, n-dimensional).
    X : np.ndarray
        Covariate matrix including intercept (n x k).
    baseline_X : np.ndarray
        Design matrix from baseline formula (h1 covariates, n x m1).
        Zero-variance columns should be filtered before calling.
    diff_X : np.ndarray
        Design matrix from diff formula (h2 covariates, n x m2).
        Zero-variance columns should be filtered before calling.
    iterations : int, default 1000
        Maximum BFGS iterations.
    att : int, default 0
        Estimand target. Only att=0 (ATE) is supported for oCBPS.
    standardize : bool, default True
        Whether to standardize weights.

    Returns
    -------
    dict
        Dictionary containing:
        - coefficients: Coefficient matrix (k x 1)
        - fitted_values: Propensity scores (n-dimensional)
        - linear_predictor: Linear predictor X @ beta (n-dimensional)
        - deviance: Negative 2 times log-likelihood
        - weights: Optimal weights (n-dimensional)
        - y: Treatment vector (n-dimensional)
        - x: Covariate matrix (n x k)
        - converged: Convergence flag (bool)
        - J: J-statistic (float)
        - var: Variance-covariance matrix (k x k)
        - mle_J: MLE baseline J-statistic (float)

    Raises
    ------
    ValueError
        If baseline and diff model dimensions are incompatible.

    Notes
    -----
    **Key Features:**

    - Only supports att=0 (ATE estimation)
    - No sample_weights parameter (oCBPS does not support sampling weights)
    - Dual initialization optimization for robust convergence

    **Dual Balancing Conditions** (Fan 2022 Eq. 3.2-3.3):

    - g1_baseline: (T/π - (1-T)/(1-π)) h1(X) = 0, balances E(Y(0)|X)
    - g2_diff: (T/π - 1) h2(X) = 0, balances E(Y(1)-Y(0)|X)

    **Double Robustness** (Theorem 3.1):
    Consistent if either the propensity score model or outcome model is correct.

    **Semiparametric Efficiency** (Corollary 3.2):
    Achieves Hahn 1998 efficiency bound when both models are correct and m=q.

    References
    ----------
    .. [1] Fan et al. (2022). Optimal Covariate Balancing Conditions in
           Propensity Score Estimation. Journal of Business & Economic
           Statistics, 41(1), 97-110. https://doi.org/10.1080/07350015.2021.2002159

    Examples
    --------
    >>> # See module-level documentation for complete examples
    """
    # Initialize constants
    n = X.shape[0]

    # Determine identification status
    m1 = baseline_X.shape[1]
    m2 = diff_X.shape[1]
    k = X.shape[1]

    if m1 + m2 + 1 > k:
        bal_only = 3  # Over-identified: m1 + m2 + 1 > q
        xcov = None
    elif m1 + m2 + 1 == k:
        bal_only = 1  # Exactly identified: m1 + m2 + 1 = q
        xcov = np.eye(m1 + m2 + 1)
    else:
        raise ValueError("Invalid baseline and diff models.")

    # Dual initialization: GLM and CBPS pre-optimization paths

    # GLM initial values
    glm_model = sm.GLM(treat, X, family=sm.families.Binomial())
    glm_result = glm_model.fit()
    glm_beta_curr = glm_result.params.copy()
    glm_beta_curr[np.isnan(glm_beta_curr)] = 0

    # CBPS pre-optimization initial values
    # Precompute simplified inverse matrix for pre-optimization
    invV2 = scipy.linalg.pinv(X.T @ X)

    # CBPS pre-optimization (single balance condition)
    def gmm_loss_for_preopt(beta):
        return _gmm_func1(beta, X, treat, baseline_X, diff_X,
                          invV=invV2, option="CBPS")['loss']

    cbps_preopt = scipy.optimize.minimize(
        gmm_loss_for_preopt,
        glm_beta_curr,
        method='BFGS'
    )
    cbps_beta_curr = cbps_preopt.x

    # GMM optimization branching
    gmm_init = glm_beta_curr

    if bal_only == 1:
        # Exactly identified
        opt_bal = scipy.optimize.minimize(
            lambda b: _gmm_func1(b, X, treat, baseline_X, diff_X, invV=xcov)['loss'],
            gmm_init,
            method='BFGS'
        )
        opt1 = opt_bal

    elif bal_only == 3:
        # Over-identified
        # GMM loss function (recompute invV each iteration)
        def gmm_loss_std(beta):
            return _gmm_func1(beta, X, treat, baseline_X, diff_X)['loss']

        # GLM path optimization
        gmm_glm_init = scipy.optimize.minimize(
            gmm_loss_std,
            glm_beta_curr,
            method='BFGS',
            options={'maxiter': iterations}
        )

        # CBPS pre-optimization path
        gmm_cbps_init = scipy.optimize.minimize(
            gmm_loss_std,
            cbps_beta_curr,
            method='BFGS',
            options={'maxiter': iterations}
        )

        # Select best initialization
        if gmm_glm_init.fun < gmm_cbps_init.fun:
            opt1 = gmm_glm_init
        else:
            opt1 = gmm_cbps_init

    # Compute probabilities and weights

    # Optimal coefficients
    beta_opt = opt1.x

    # Optimal propensity scores
    theta_opt = X @ beta_opt
    probs_opt = scipy.special.expit(theta_opt)
    probs_opt = np.minimum(1 - PROBS_MIN, probs_opt)
    probs_opt = np.maximum(PROBS_MIN, probs_opt)

    # ATE weights (simplified form)
    w_opt = np.abs((probs_opt - 1 + treat)**(-1))

    # Weight standardization
    if standardize:
        norm1 = np.sum((treat == 1) / probs_opt)
        norm2 = np.sum((treat == 0) / (1 - probs_opt))
    else:
        norm1 = norm2 = 1.0

    w_opt = ((treat == 1) / probs_opt / norm1 +
             (treat == 0) / (1 - probs_opt) / norm2)

    # Compute variance-covariance matrix

    # Construct X1new (required for vcov computation)
    X1new = np.column_stack([X[:, 0], baseline_X])

    # G matrix construction
    factor_1 = np.sqrt(np.abs(treat - probs_opt) / (probs_opt * (1 - probs_opt)))
    XG_1 = -X * factor_1[:, None]
    XG_12 = -X1new * factor_1[:, None]
    XW_1 = X1new * ((probs_opt - 1 + treat)**(-1))[:, None]

    factor_2 = np.sqrt(treat * (1 - probs_opt) / probs_opt)
    XG_2 = -X * factor_2[:, None]
    XG_22 = -diff_X * factor_2[:, None]
    XW_2 = diff_X * ((treat / probs_opt - 1))[:, None]

    # W1 matrix
    W1 = np.vstack([XW_1.T, XW_2.T])

    # G matrix
    G = np.column_stack([
        (XG_1.T @ XG_12) / n,
        (XG_2.T @ XG_22) / n
    ])

    # Omega outer product
    Omega = (W1 @ W1.T) / n

    # Sandwich variance formula
    gmm_result = _gmm_func1(beta_opt, X, treat, baseline_X, diff_X, invV=None)
    W = gmm_result['invV']

    GWG_inv = _r_ginv(G @ W @ G.T)
    vcov = GWG_inv @ G @ W @ Omega @ W.T @ G.T @ GWG_inv

    # Construct return object

    # J-statistic
    J_opt = _gmm_func1(beta_opt, X, treat, baseline_X, diff_X, invV=None)['loss']

    # Deviance
    deviance = -2 * np.sum(
        treat * np.log(probs_opt) + (1 - treat) * np.log(1 - probs_opt)
    )

    # MLE baseline J-statistic
    glm1_coef = glm_beta_curr
    mle_J = _gmm_func1(glm1_coef, X, treat, baseline_X, diff_X)['loss']

    # Build output dictionary
    output = {
        'coefficients': beta_opt.reshape(-1, 1),  # k x 1 matrix
        'fitted_values': probs_opt,
        'linear_predictor': theta_opt,
        'deviance': deviance,
        'weights': w_opt,
        'y': treat,
        'x': X,
        'converged': opt1.success,
        'J': J_opt,
        'var': vcov,
        'mle_J': mle_J
    }

    return output
