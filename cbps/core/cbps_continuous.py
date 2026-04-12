"""
Covariate Balancing Propensity Score for Continuous Treatments
===============================================================

This module implements the Covariate Balancing Propensity Score (CBPS) methodology
for continuous treatments using generalized propensity scores (GPS). The implementation
extends the binary CBPS framework to handle continuous treatment variables through
covariate whitening and normal density estimation.

Methodology
-----------
The continuous CBPS estimates the generalized propensity score by maximizing the
covariate balance. The method involves:
1. Cholesky whitening of covariates with sample weights.
2. Log-space normal density computation for numerical stability.
3. GMM optimization with multiple starting values.
4. Coefficient inverse transformation from whitened to original space.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity score
for a continuous treatment: Application to the efficacy of political advertisements.
The Annals of Applied Statistics, 12(1), 156-177.
"""

from typing import Dict, Any, Optional
import warnings
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.linalg
import statsmodels.api as sm

from cbps.utils.validation import validate_cbps_input


# ========== Constants ==========
PROBS_MIN = 1e-6
CONST_COL_THRESHOLD = 1e-10
ALPHA_BOUNDS = (0.8, 1.1)
CLIP_RANGE = 50


def cbps_continuous_fit(
    treat: np.ndarray,
    X: np.ndarray,
    method: str = 'over',
    two_step: bool = True,
    iterations: int = 1000,
    standardize: bool = True,
    sample_weights: Optional[np.ndarray] = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Fit the Covariate Balancing Propensity Score model for continuous treatments.

    Parameters
    ----------
    treat : np.ndarray
        Continuous treatment vector, shape (n,).
    X : np.ndarray
        Covariate matrix (including intercept column), shape (n, k).
    method : {'over', 'exact'}, default='over'
        Estimation method:
        - 'over': Over-identified GMM (score + balance + sigma conditions).
        - 'exact': Exactly identified GMM (balance + sigma conditions).
    two_step : bool, default=True
        If True, use two-step GMM with fixed weight matrix.
        If False, use continuously updating GMM.
    iterations : int, default=1000
        Maximum number of optimization iterations.
    standardize : bool, default=True
        If True, standardize weights to sum to the sample size.
    sample_weights : np.ndarray, optional
        Sampling weights. Defaults to uniform weights if None.
        Weights will be normalized to sum to n.
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    dict
        Dictionary containing estimation results:
        - coefficients: Estimated parameters.
        - fitted_values: Estimated propensity scores.
        - weights: Inverse probability weights.
        - deviance: Model deviance.
        - converged: Convergence status.
        - J: GMM loss function value.
        - var: Variance-covariance matrix.
        - sigmasq: Estimated residual variance.
        - Ttilde: Standardized treatment.
        - Xtilde: Whitened covariates.

    Notes
    -----
    The algorithm performs Cholesky whitening on covariates, standardizes the treatment,
    and then optimizes the GMM objective function. It handles potential numerical
    instability in the weight matrix calculation through regularization when necessary.
    """
    # Input validation
    validate_cbps_input(
        treat, X,
        min_observations=2,
        module_name="Continuous CBPS",
        check_treatment_variance=True
    )
    
    # Auto-fallback: if method='over' encounters infinite V matrix,
    # fall back to method='exact' (matching R CBPS behavior)
    if method == 'over':
        try:
            return _cbps_continuous_fit_impl(
                treat, X, method=method, two_step=two_step,
                iterations=iterations, standardize=standardize,
                sample_weights=sample_weights, verbose=verbose
            )
        except ValueError as e:
            if "infinite value in the weighting matrix" in str(e).lower():
                warnings.warn(
                    f"Over-identified GMM failed due to infinite V matrix values. "
                    f'Automatically falling back to method="exact" '
                    f"(just-identified). Original error: {e}",
                    UserWarning
                )
                return _cbps_continuous_fit_impl(
                    treat, X, method='exact', two_step=two_step,
                    iterations=iterations, standardize=standardize,
                    sample_weights=sample_weights, verbose=verbose
                )
            raise
    else:
        return _cbps_continuous_fit_impl(
            treat, X, method=method, two_step=two_step,
            iterations=iterations, standardize=standardize,
            sample_weights=sample_weights, verbose=verbose
        )


def _cbps_continuous_fit_impl(
    treat: np.ndarray,
    X: np.ndarray,
    method: str = 'over',
    two_step: bool = True,
    iterations: int = 1000,
    standardize: bool = True,
    sample_weights: Optional[np.ndarray] = None,
    verbose: int = 0
) -> Dict[str, Any]:
    """Internal implementation of cbps_continuous_fit."""
    
    # Initialization
    n = len(treat)
    k = X.shape[1]
    bal_only = (method == 'exact')
    
    # Normalize sample weights
    if sample_weights is None:
        sample_weights = np.ones(n)
    sample_weights = sample_weights / sample_weights.mean()
    if not np.isclose(sample_weights.sum(), n, atol=1e-10):
        warnings.warn(f"Sample weights normalization check failed: sum={sample_weights.sum():.6f} != n={n}")
    
    # Save original X
    X_orig = X.copy()
    
    # ========== Covariate Whitening Preprocessing ==========
    
    # Detect constant columns
    col_std = np.std(X, axis=0, ddof=1)
    int_ind = np.where(col_std <= CONST_COL_THRESHOLD)[0]
    non_const_ind = np.where(col_std > CONST_COL_THRESHOLD)[0]

    if len(non_const_ind) == 0:
        warnings.warn(
            "All columns are constant (sd <= 1e-10). "
            "Continuous CBPS will degenerate to no-covariate model. "
            "This is a valid edge case where the model only standardizes the treatment distribution.",
            UserWarning
        )
        # Degenerate case: Xtilde is just X
        Xtilde = X.copy()
    else:
        # Perform Cholesky whitening on non-constant columns
        X_non_const = X[:, non_const_ind]
        sw_X_non_const = sample_weights[:, None] * X_non_const
        cov_weighted = np.cov(sw_X_non_const.T, ddof=1)

        assert np.allclose(cov_weighted, cov_weighted.T, atol=1e-12), \
            "Weighted covariance matrix must be symmetric"

        # Cholesky decomposition to get upper triangular U
        U = scipy.linalg.cholesky(cov_weighted, lower=False)

        assert np.allclose(np.tril(U, k=-1), 0, atol=1e-12), "U must be upper triangular"
        assert np.all(np.diag(U) > 0), "Diagonal elements of U must be positive"

        U_inv = np.linalg.inv(U)

        # Whitening transformation
        X_white = sw_X_non_const @ U_inv

        # Centering (no scaling)
        X_white_centered = X_white - X_white.mean(axis=0)

        assert abs(X_white_centered.mean()) < 1e-10, "Whitened data should be centered"

        # Combine constant and whitened columns
        if len(int_ind) > 0:
            X_const = X[:, int_ind]
            Xtilde = np.column_stack([X_const, X_white_centered])
        else:
            Xtilde = X_white_centered
    
    # Verify shape consistency
    if Xtilde.shape != X.shape:
        raise ValueError(f"Xtilde shape {Xtilde.shape} != X shape {X.shape}")
    
    # ========== Auxiliary Matrix Calculation ==========
    
    # Pre-compute weighted Xtilde
    wtXilde = sample_weights[:, None] * Xtilde
    
    # Standardize treatment (zero mean, unit variance)
    sw_treat = sample_weights * treat
    Ttilde = (sw_treat - sw_treat.mean()) / sw_treat.std(ddof=1)
    
    # Internal consistency checks
    assert abs(Ttilde.mean()) < 1e-10
    assert abs(Ttilde.std(ddof=1) - 1) < 1e-10
    
    n_identity_vec = np.ones((n, 1))
    
    # ========== Stabilizers Calculation ==========
    # Calculate log marginal density log f(T*)
    # Ideally constant, but computed per observation for robustness
    
    pdf_vals = scipy.stats.norm.pdf(Ttilde, 0, 1)
    pdf_clipped = np.clip(pdf_vals, PROBS_MIN, 1 - PROBS_MIN)
    stabilizers = np.log(pdf_clipped)
    
    # ========== GMM Objective Function ==========
    
    def gmm_func(params_curr: np.ndarray, invV: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        GMM objective function for over-identified case.
        
        Parameters
        ----------
        params_curr : np.ndarray
            Parameter vector [beta, log(sigma^2)].
        invV : np.ndarray, optional
            Inverse weight matrix V.
        
        Returns
        -------
        dict
            Dictionary containing loss value and inverse V matrix.
        """
        beta_curr = params_curr[:-1]
        sigmasq = np.exp(params_curr[-1])
        
        # Log conditional density
        log_dens = scipy.stats.norm.logpdf(
            Ttilde, 
            loc=Xtilde @ beta_curr, 
            scale=np.sqrt(sigmasq)
        )
        
        # Log-space clipping
        log_dens = np.minimum(np.log(1 - PROBS_MIN), log_dens)
        log_dens = np.maximum(np.log(PROBS_MIN), log_dens)
        
        # Weight calculation in log space
        log_diff = stabilizers - log_dens
        log_diff_clipped = np.clip(log_diff, -CLIP_RANGE, CLIP_RANGE)
        w_curr = Ttilde * np.exp(log_diff_clipped)
        
        if not np.all(np.isfinite(w_curr)):
            raise ValueError("Weights contain non-finite values")
        
        # Construct sample moment conditions gbar
        # Moment 1: Score condition for sigma^2
        gbar_1 = (1/n) * wtXilde.T @ ((Ttilde - Xtilde @ beta_curr) / sigmasq)
        
        # Moment 2: Balance condition
        w_curr_del = (1/n) * wtXilde.T @ w_curr
        gbar_2 = w_curr_del.ravel()
        
        # Moment 3: Score condition for beta
        gbar_3 = (1/n) * sample_weights.T @ (
            (Ttilde - Xtilde @ beta_curr)**2 / sigmasq - 1
        )
        
        gbar = np.concatenate([gbar_1.ravel(), gbar_2, [gbar_3]])
        
        # Compute V matrix or use pre-computed invV
        if invV is None:
            # Construct V matrix blocks
            V11 = (1/sigmasq) * wtXilde.T @ Xtilde
            V12 = wtXilde.T @ Xtilde / sigmasq
            V13 = wtXilde.T @ n_identity_vec * 0
            
            # V22 calculation with scaling vector
            linear_pred = Xtilde @ beta_curr
            linear_pred_sq = linear_pred**2
            term_A = linear_pred_sq / sigmasq
            term_B = np.log(sigmasq + linear_pred_sq)
            
            exponent = term_A + term_B
            if np.any(exponent > 700):
                raise ValueError(
                    f"Potential overflow in V matrix calculation (max exponent={exponent.max():.2f}). "
                    f"Residual variance sigma^2={sigmasq:.6f} might be too small. "
                    f"Consider using method='exact'."
                )
            
            vec_scaling = np.exp(exponent)
            
            if not np.all(np.isfinite(vec_scaling)):
                raise ValueError("V22 scaling vector contains non-finite values.")
            
            Xtilde_swept = vec_scaling[:, None] * Xtilde
            V22 = wtXilde.T @ Xtilde_swept
            
            V23 = (wtXilde.T @ (-Xtilde @ beta_curr) * (-2/sigmasq)).reshape(-1, 1)
            
            V33_scalar = sample_weights.T @ n_identity_vec.ravel() * 2
            V33 = np.array([[V33_scalar]])
            
            # Assemble V
            V = (1/n) * np.block([
                [V11, V12, V13],
                [V12, V22, V23],
                [V13.T, V23.T, V33]
            ])
            
            if not np.allclose(V, V.T, atol=1e-12):
                warnings.warn("V matrix is not symmetric within tolerance")
            
            if np.any(np.isinf(V)):
                raise ValueError(
                    "Encountered an infinite value in the weighting matrix. "
                    'Use the just-identified version of CBPS instead by setting method="exact".'
                )
            
            invV = scipy.linalg.pinv(V)
        
        loss = gbar.T @ invV @ gbar
        
        if loss < -1e-6:
            warnings.warn(
                f"GMM loss is negative ({loss:.2e}). Check numerical stability.",
                UserWarning
            )
        
        return {'loss': float(loss), 'invV': invV}

    def gmm_loss(params_curr: np.ndarray, invV: Optional[np.ndarray] = None) -> float:
        return gmm_func(params_curr, invV)['loss']

    
    # ========== bal_func (exactly-identified, 2 moment conditions) ==========
    
    def bal_func(params_curr: np.ndarray) -> Dict[str, float]:
        """
        Balance objective function for the exactly-identified case.
        
        Parameters
        ----------
        params_curr : np.ndarray
            Parameter vector [beta, log(sigma^2)].
        
        Returns
        -------
        dict
            Dictionary containing the balance loss value.
        """
        beta_curr = params_curr[:-1]
        sigmasq = np.exp(params_curr[-1])
        
        # Log conditional density
        log_dens = scipy.stats.norm.logpdf(
            Ttilde, 
            loc=Xtilde @ beta_curr, 
            scale=np.sqrt(sigmasq)
        )
        
        log_dens = np.minimum(np.log(1 - PROBS_MIN), log_dens)
        log_dens = np.maximum(np.log(PROBS_MIN), log_dens)
        
        # Weight calculation
        log_diff = stabilizers - log_dens
        log_diff_clipped = np.clip(log_diff, -CLIP_RANGE, CLIP_RANGE)
        w_curr = Ttilde * np.exp(log_diff_clipped)
        
        # Construct sample moment conditions
        w_curr_del = (1/n) * wtXilde.T @ w_curr
        
        gbar = np.concatenate([
            w_curr_del.ravel(),  # Balance condition
            [(1/n) * sample_weights.T @ (
                (Ttilde - Xtilde @ beta_curr)**2 / sigmasq - 1
            )]  # Sigma^2 condition
        ])
        
        if gbar.shape != (k + 1,):
            raise ValueError(f"Gradient vector shape mismatch: {gbar.shape}")
        
        # Loss calculation with identity weight matrix
        loss = gbar.T @ np.eye(k + 1) @ gbar
        
        return {'loss': float(loss)}
    
    def bal_loss(params_curr: np.ndarray) -> float:
        """Wrapper for balance loss function."""
        return bal_func(params_curr)['loss']
    
    # ========== GMM Gradient Calculation ==========
    
    def gmm_gradient(params_curr: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        Gradient of the GMM objective function.
        
        Parameters
        ----------
        params_curr : np.ndarray
            Parameter vector.
        invV : np.ndarray
            Inverse weight matrix V.
        
        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        beta_curr = params_curr[:-1]
        sigmasq = np.exp(params_curr[-1])
        
        # Log conditional density
        log_dens = scipy.stats.norm.logpdf(
            Ttilde, 
            loc=Xtilde @ beta_curr, 
            scale=np.sqrt(sigmasq)
        )
        
        log_dens = np.minimum(np.log(1 - PROBS_MIN), log_dens)
        log_dens = np.maximum(np.log(PROBS_MIN), log_dens)
        
        # Weights
        log_diff = stabilizers - log_dens
        log_diff_clipped = np.clip(log_diff, -CLIP_RANGE, CLIP_RANGE)
        w_curr = Ttilde * np.exp(log_diff_clipped)
        
        # Recompute gbar
        gbar_1 = (1/n) * wtXilde.T @ ((Ttilde - Xtilde @ beta_curr) / sigmasq)
        w_curr_del = (1/n) * wtXilde.T @ w_curr
        gbar_2 = w_curr_del.ravel()
        gbar_3 = (1/n) * sample_weights.T @ (
            (Ttilde - Xtilde @ beta_curr)**2 / sigmasq - 1
        )
        gbar = np.concatenate([gbar_1.ravel(), gbar_2, [gbar_3]])
        
        # Calculate dgbar blocks
        # dgbar.1.1 (k x k)
        dgbar_1_1 = (-wtXilde.T @ Xtilde) / sigmasq
        
        # dgbar.1.2 (1 x k)
        dgbar_1_2 = (
            -sample_weights * (Ttilde - Xtilde @ beta_curr) / (sigmasq**2)
        ).reshape(1, -1) @ Xtilde
        
        # dgbar.2.1 (k x k)
        vec_L110 = -(Ttilde - Xtilde @ beta_curr) / sigmasq * w_curr
        dgbar_2_1 = (wtXilde.T * vec_L110) @ Xtilde
        
        # dgbar.2.2 (1 x k)
        dgbar_2_2 = (
            w_curr * (1/(2*sigmasq) - (Ttilde - Xtilde @ beta_curr)**2 / (2*sigmasq**2))
        ).reshape(1, -1) @ Xtilde
        
        # dgbar.3.1 (k x 1)
        dgbar_3_1 = wtXilde.T @ (
            -2 * (Ttilde - Xtilde @ beta_curr) / sigmasq
        ).reshape(-1, 1)
        
        # dgbar.3.2 (scalar)
        dgbar_3_2 = sample_weights.T @ (
            -(Ttilde - Xtilde @ beta_curr)**2 / (sigmasq**2)
        )
        
        # Assemble dgbar
        col1 = np.vstack([dgbar_1_1, dgbar_1_2 * sigmasq])
        col2 = np.vstack([dgbar_2_1, dgbar_2_2 * sigmasq])
        col3 = np.vstack([dgbar_3_1, dgbar_3_2.reshape(1, 1) * sigmasq])
        
        dgbar = (1/n) * np.hstack([col1, col2, col3])
        
        # Gradient calculation: 2 * dgbar @ invV @ gbar
        gradient = 2 * dgbar @ invV @ gbar
        
        return gradient.ravel()
    
    # ========== Balance Gradient Calculation ==========
    
    def bal_gradient(params_curr: np.ndarray) -> np.ndarray:
        """
        Gradient of the balance objective function.
        
        Parameters
        ----------
        params_curr : np.ndarray
            Parameter vector.
        
        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        beta_curr = params_curr[:-1]
        sigmasq = np.exp(params_curr[-1])
        
        log_dens = scipy.stats.norm.logpdf(
            Ttilde, 
            loc=Xtilde @ beta_curr, 
            scale=np.sqrt(sigmasq)
        )
        
        log_dens = np.minimum(np.log(1 - PROBS_MIN), log_dens)
        log_dens = np.maximum(np.log(PROBS_MIN), log_dens)
        
        log_diff = stabilizers - log_dens
        log_diff_clipped = np.clip(log_diff, -CLIP_RANGE, CLIP_RANGE)
        w_curr = Ttilde * np.exp(log_diff_clipped)
        
        w_curr_del = (1/n) * wtXilde.T @ w_curr
        gbar = np.concatenate([
            w_curr_del.ravel(),
            [(1/n) * sample_weights.T @ (
                (Ttilde - Xtilde @ beta_curr)**2 / sigmasq - 1
            )]
        ])
        
        # Calculate dgbar blocks
        vec_L145 = -(Ttilde - Xtilde @ beta_curr) / sigmasq * w_curr
        dgbar_2_1 = (wtXilde.T * vec_L145) @ Xtilde
        
        dgbar_2_2 = (
            w_curr * (1/(2*sigmasq) - (Ttilde - Xtilde @ beta_curr)**2 / (2*sigmasq**2))
        ).reshape(1, -1) @ Xtilde
        
        dgbar_3_1 = wtXilde.T @ (
            -2 * (Ttilde - Xtilde @ beta_curr) / sigmasq
        ).reshape(-1, 1)
        
        dgbar_3_2 = sample_weights.T @ (
            -(Ttilde - Xtilde @ beta_curr)**2 / (sigmasq**2)
        )
        
        col1 = np.vstack([dgbar_2_1, dgbar_2_2 * sigmasq])
        col2 = np.vstack([dgbar_3_1, dgbar_3_2.reshape(1, 1) * sigmasq])
        
        dgbar = (1/n) * np.hstack([col1, col2])
        
        gradient = 2 * dgbar @ np.eye(k + 1) @ gbar
        
        return gradient.ravel()
    
    # ========== Optimization Initialization and Scaling ==========
    
    # Initial Linear Regression estimate
    lm_model = sm.WLS(Ttilde, Xtilde, weights=sample_weights).fit()
    
    mcoef = lm_model.params.copy()
    mcoef[np.isnan(mcoef)] = 0
    
    residuals = Ttilde - Xtilde @ mcoef
    sigmasq_init = np.mean(residuals**2)
    
    assert sigmasq_init > 0, f"Initial residual variance must be positive (got {sigmasq_init})"
    
    # Calculate MLE probabilities
    probs_mle = scipy.stats.norm.logpdf(
        Ttilde, 
        loc=Xtilde @ mcoef, 
        scale=np.sqrt(sigmasq_init)
    )
    probs_mle = np.minimum(np.log(1 - PROBS_MIN), probs_mle)
    probs_mle = np.maximum(np.log(PROBS_MIN), probs_mle)
    
    # Construct initial parameter vector
    params_curr = np.concatenate([mcoef, [np.log(sigmasq_init)]])
    
    # Pre-compute MLE baseline loss for fallback
    mle_J = np.nan
    try:
        mle_J = gmm_loss(params_curr)
    except Exception as e:
        warnings.warn(f"Failed to compute MLE J statistic: {e}")
    
    mle_bal = bal_loss(params_curr)
    
    # Alpha scaling optimization
    # Implementation Note:
    # We use a fixed V matrix (calculated at alpha=1.0) during the alpha scaling phase.
    # While continuous updating of V is theoretically possible, fixed V provides better
    # numerical stability in pathological cases (e.g., extremely poor initial fit)
    # and matches the performance of standard two-step GMM approaches.
    
    glm_invV = None
    try:
        # Pre-compute V inverse at alpha=1.0
        glm_invV = gmm_func(params_curr, invV=None)['invV']
        
        def alpha_func(alpha):
            return gmm_loss(params_curr * alpha, invV=glm_invV)
        
        alpha_result = scipy.optimize.minimize_scalar(
            alpha_func, 
            bounds=ALPHA_BOUNDS, 
            method='bounded'
        )
        
        # Update parameters with optimal alpha scaling
        params_curr = params_curr * alpha_result.x
        
    except Exception as e:
        warnings.warn(f"Alpha scaling failed, using unscaled LM initialization: {e}")
        glm_invV = None
    
    gmm_init = params_curr.copy()
    
    # ========== Balance and GMM Optimization ==========
    
    if verbose >= 1:
        print(f"[CBPS Continuous] Starting balance optimization (max_iter={iterations}, two_step={two_step})...")
    
    if two_step:
        # Two-step estimation using BFGS
        opt_bal = scipy.optimize.minimize(
            bal_loss, gmm_init, 
            method='BFGS', 
            jac=bal_gradient,
            options={
                'maxiter': iterations,
                'gtol': 1e-05
            }
        )
    else:
        # Continuous updating with fallback
        try:
            opt_bal = scipy.optimize.minimize(
                bal_loss, gmm_init, 
                method='BFGS',
                options={'maxiter': iterations}
            )
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
            warnings.warn(f"Balance BFGS failed, falling back to Nelder-Mead: {e}")
            opt_bal = scipy.optimize.minimize(
                bal_loss, gmm_init, 
                method='Nelder-Mead',
                options={'maxiter': iterations}
            )
    
    params_bal = opt_bal.x
    
    if bal_only:
        opt1 = opt_bal
    
    if not bal_only:
        if verbose >= 1:
            print(f"[CBPS Continuous] Starting GMM optimization with dual initialization...")
        
        if two_step:
            # Initialize from GLM and Balance solutions
            gmm_glm_init = scipy.optimize.minimize(
                lambda p: gmm_loss(p, invV=glm_invV), 
                gmm_init,
                method='BFGS', 
                jac=lambda p: gmm_gradient(p, glm_invV),
                options={
                    'maxiter': iterations,
                    'gtol': 1e-05
                }
            )
            gmm_bal_init = scipy.optimize.minimize(
                lambda p: gmm_loss(p, invV=glm_invV), 
                params_bal,
                method='BFGS', 
                jac=lambda p: gmm_gradient(p, glm_invV),
                options={
                    'maxiter': iterations,
                    'gtol': 1e-05
                }
            )
        else:
            # Continuous updating
            try:
                gmm_glm_init = scipy.optimize.minimize(
                    gmm_loss, gmm_init, 
                    method='BFGS',
                    options={
                        'maxiter': iterations,
                        'gtol': 1e-05
                    }
                )
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                warnings.warn(f"GMM-GLM BFGS failed, falling back to Nelder-Mead: {e}")
                gmm_glm_init = scipy.optimize.minimize(
                    gmm_loss, gmm_init, 
                    method='Nelder-Mead',
                    options={'maxiter': iterations}
                )
            
            try:
                gmm_bal_init = scipy.optimize.minimize(
                    gmm_loss, params_bal, 
                    method='BFGS',
                    options={
                        'maxiter': iterations,
                        'gtol': 1e-05
                    }
                )
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                warnings.warn(f"GMM-Balance BFGS failed, falling back to Nelder-Mead: {e}")
                gmm_bal_init = scipy.optimize.minimize(
                    gmm_loss, params_bal, 
                    method='Nelder-Mead',
                    options={'maxiter': iterations}
                )
        
        # Select best solution
        if gmm_glm_init.fun < gmm_bal_init.fun:
            opt1 = gmm_glm_init
            pick_glm = 1
        else:
            opt1 = gmm_bal_init
            pick_glm = 0
        
        if verbose >= 1:
            source = "GLM" if pick_glm == 1 else "Balance"
            print(f"[CBPS Continuous] GMM optimization complete: J={opt1.fun:.6f}, converged={opt1.success}, source={source}")
    
    # ========== Parameter Extraction and MLE Fallback ==========
    
    params_opt = opt1.x
    beta_opt = params_opt[:-1]
    sigmasq = np.exp(params_opt[-1])
    
    # Recalculate probabilities
    probs_opt = scipy.stats.norm.logpdf(
        Ttilde, 
        loc=Xtilde @ beta_opt, 
        scale=np.sqrt(sigmasq)
    )
    probs_opt = np.minimum(np.log(1 - PROBS_MIN), probs_opt)
    probs_opt = np.maximum(np.log(PROBS_MIN), probs_opt)
    
    if not bal_only:
        if two_step:
            J_opt = gmm_func(params_opt, invV=glm_invV)['loss']
        else:
            J_opt = gmm_func(params_opt)['loss']
        
        # MLE Fallback Logic
        # Check 1: Significantly negative J statistic (theoretical violation)
        if J_opt < -1e-6:
            raise ValueError(
                f"Encountered an infinite value in the weighting matrix. "
                f"J statistic is significantly negative (J={J_opt:.6e}), "
                f"indicating numerical instability in the V matrix. "
                f'Use the just-identified version of CBPS instead by setting method="exact".'
            )
        
        # Check 2: Optimization result worse than MLE
        # R code: if ((J.opt > mle.J) & (bal.loss(params.opt) > mle.bal))
        elif (J_opt > mle_J) and (bal_loss(params_opt) > mle_bal):
            warnings.warn(
                f"Optimization produced worse results than MLE (|J_opt|={abs(J_opt):.6e} > "
                f"|J_mle|={abs(mle_J):.6e}). Falling back to MLE.",
                UserWarning
            )
            beta_opt = mcoef
            probs_opt = probs_mle
            J_opt = mle_J
        
        # Check 3: Minor negative J
        elif J_opt < 0:
            warnings.warn(
                f"J statistic is slightly negative (J={J_opt:.6e}). "
                f"This may indicate minor numerical precision issues.",
                UserWarning
            )
    else:
        J_opt = bal_loss(params_opt)
    
    # ========== Final Weight Calculation and Variance Estimation ==========
    
    w_opt = np.exp(stabilizers - probs_opt)
    
    if standardize:
        w_opt = w_opt / np.sum(w_opt * sample_weights)
        if not np.isclose(np.sum(w_opt * sample_weights), 1.0, atol=1e-10):
            warnings.warn("Weight standardization failed to sum to 1")
    
    if not np.all(np.isfinite(w_opt)):
        raise ValueError("Final weights contain non-finite values")
    
    deviance = -2 * np.sum(probs_opt)
    
    # Compute XG matrix blocks (Gradient of moment conditions)
    XG_1_1 = (-wtXilde.T @ Xtilde) / sigmasq
    
    XG_2_1 = (wtXilde.T @ (
        -2 * (Ttilde - Xtilde @ beta_opt) / sigmasq
    )).reshape(-1, 1)
    
    vec_L258 = -(Ttilde - Xtilde @ beta_opt) / sigmasq * Ttilde * w_opt
    XG_3_1 = (wtXilde.T * vec_L258) @ Xtilde
    
    XG_1_2 = ((-wtXilde.T @ (Ttilde - Xtilde @ beta_opt)) / (sigmasq**2)).reshape(-1, 1)
    
    XG_2_2_scalar = sample_weights.T @ (
        -(Ttilde - Xtilde @ beta_opt)**2 / (sigmasq**2)
    )
    XG_2_2 = np.array([[XG_2_2_scalar]])
    
    XG_3_2 = (
        -Ttilde * sample_weights * w_opt * (
            (Ttilde - Xtilde @ beta_opt)**2 / (2*sigmasq**2) - 1/(2*sigmasq)
        )
    ).reshape(1, -1) @ Xtilde
    
    # Compute XW matrix blocks
    XW_1 = Xtilde * (
        (Ttilde - Xtilde @ beta_opt) / sigmasq * sample_weights**0.5
    )[:, None]
    
    XW_2 = (
        (Ttilde - Xtilde @ beta_opt)**2 / sigmasq - 1
    ) * sample_weights**0.5
    
    XW_3 = Xtilde * (Ttilde * w_opt * sample_weights)[:, None]
    
    if bal_only:
        W = np.eye(k + 1)
        G = (1/n) * np.vstack([
            np.hstack([XG_3_1, XG_3_2.T]),
            np.hstack([XG_2_1.T, XG_2_2])
        ])
        W1 = np.vstack([XW_3.T, XW_2.reshape(1, -1)])
    else:
        W = gmm_func(params_opt)['invV']
        G = (1/n) * np.vstack([
            np.hstack([XG_1_1, XG_1_2]),
            np.hstack([XG_3_1, XG_3_2.T]),
            np.hstack([XG_2_1.T, XG_2_2])
        ])
        W1 = np.vstack([XW_1.T, XW_3.T, XW_2.reshape(1, -1)])
    
    Omega = (1/n) * (W1 @ W1.T)
    
    GWG_inv = scipy.linalg.pinv(G.T @ W @ G)
    GWGinvGW = W @ G @ GWG_inv
    
    vcov_tilde = (GWGinvGW.T @ Omega @ GWGinvGW)[0:k, 0:k]
    vcov_tilde = (vcov_tilde + vcov_tilde.T) / 2
    
    # Inverse transformation to original space
    beta_tilde = beta_opt.copy()
    
    XtX_inv = scipy.linalg.pinv(X.T @ X)
    beta_opt = XtX_inv @ X.T @ (
        Xtilde @ beta_tilde * np.std(sw_treat, ddof=1) + np.mean(sw_treat)
    )
    
    sigmasq_tilde = sigmasq
    sigmasq = sigmasq_tilde * np.var(sw_treat, ddof=1)
    
    # Variance-covariance transformation
    sw_treat_var = np.var(sw_treat, ddof=1)
    middle = Xtilde @ vcov_tilde @ Xtilde.T * sw_treat_var
    vcov = XtX_inv @ X.T @ middle @ X @ XtX_inv
    vcov = (vcov + vcov.T) / 2
    
    result = {
        'coefficients': beta_opt.reshape(-1, 1),
        'fitted_values': np.clip(
            scipy.stats.norm.pdf(
                Ttilde, 
                loc=Xtilde @ beta_tilde, 
                scale=np.sqrt(sigmasq_tilde)
            ),
            PROBS_MIN, 
            1 - PROBS_MIN
        ),
        'linear_predictor': Xtilde @ beta_tilde,
        'deviance': deviance,
        'weights': w_opt * sample_weights,
        'y': treat,
        'x': X,
        'converged': opt1.success,
        'J': J_opt,
        'var': vcov,
        'mle_J': mle_J,
        'sigmasq': sigmasq,
        'Ttilde': Ttilde,
        'Xtilde': Xtilde,
        'beta_tilde': beta_tilde,
        'sigmasq_tilde': sigmasq_tilde,
        'stabilizers': stabilizers
    }
    
    return result

