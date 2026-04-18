"""
Asymptotic Variance Estimation for ATE
=======================================

Variance estimation and confidence interval construction for IPTW estimators
of the average treatment effect (ATE) under binary treatment.

Two variance formulas are implemented:

- **oCBPS**: Semiparametric efficiency bound (Fan et al., 2022),
  which attains the Hahn (1998) efficiency bound when both the propensity score
  and outcome models are correctly specified.

- **CBPS**: Full sandwich variance (Fan et al., 2022), accounting
  for estimation uncertainty in propensity score parameters via the GMM
  asymptotic variance formula.

References
----------
Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
    Optimal covariate balancing conditions in propensity score estimation.
    Journal of Business & Economic Statistics, 41(1), 97-110.
    https://doi.org/10.1080/07350015.2021.2002159

Hahn, J. (1998). On the role of the propensity score in efficient
    semiparametric estimation of average treatment effects.
    Econometrica, 66(2), 315-331.
"""

import numpy as np
import scipy.stats
import scipy.linalg
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, Any
import warnings


def asy_var(
    Y: np.ndarray,
    Y_1_hat: Optional[np.ndarray] = None,
    Y_0_hat: Optional[np.ndarray] = None,
    CBPS_obj: Optional[Dict[str, Any]] = None,
    method: str = "CBPS",
    X: Optional[np.ndarray] = None,
    TL: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    CI: float = 0.95,
    use_observed_y: bool = False
) -> Dict[str, Any]:
    """
    Estimate asymptotic variance and confidence intervals for ATE.

    Computes the asymptotic variance of the IPTW estimator for the average
    treatment effect under binary treatment. Two methods are available:
    the semiparametric efficiency bound (oCBPS) and the full sandwich
    formula (CBPS).

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Observed outcome values.
    Y_1_hat : ndarray of shape (n,), optional
        Predicted potential outcomes under treatment, E[Y(1)|X]. If None,
        fitted via OLS on the treatment group.
    Y_0_hat : ndarray of shape (n,), optional
        Predicted potential outcomes under control, E[Y(0)|X]. If None,
        fitted via OLS on the control group.
    CBPS_obj : dict or CBPSResults, optional
        Fitted CBPS object containing 'x', 'y', 'fitted_values', and
        'coefficients' attributes.
    method : {'CBPS', 'oCBPS'}, default='CBPS'
        Variance estimation method:

        - 'CBPS': Full sandwich formula (Fan et al., 2022)
        - 'oCBPS': Semiparametric efficiency bound (Fan et al., 2022)

    X : ndarray of shape (n, p), optional
        Covariate matrix with intercept column. Extracted from CBPS_obj
        if not provided.
    TL : ndarray of shape (n,), optional
        Binary treatment indicator (0/1). Extracted from CBPS_obj if not
        provided.
    pi : ndarray of shape (n,), optional
        Estimated propensity scores in (0, 1). Extracted from CBPS_obj
        if not provided.
    mu : float, optional
        ATE point estimate. If None, computed via AIPW (Augmented IPW).
        Note: The variance formula is the semiparametric efficiency bound,
        which is valid for AIPW but NOT for simple IPTW.
    CI : float, default=0.95
        Confidence level for the interval, in (0, 1).
    use_observed_y : bool, default=False
        Sigma_mu computation method:

        - False (default): Use predicted values Y_1_hat, Y_0_hat.
          This matches R CBPS package behavior and is recommended.
        - True: Use observed Y values. This is an experimental option
          not implemented in the R package.

    Returns
    -------
    dict
        Dictionary with keys:

        - 'mu.hat': ATE point estimate
        - 'asy.var': Asymptotic variance of sqrt(n) * (mu_hat - mu)
        - 'var': Finite-sample variance (asy.var / n)
        - 'std.err': Standard error
        - 'CI.mu.hat': ndarray of shape (2,), confidence interval bounds

    See Also
    --------
    cbps.AsyVar : Public interface accepting CBPSResults objects.
    vcov_outcome : Variance for continuous treatment weighted regressions.

    Notes
    -----
    **Important**: This function uses the AIPW (Augmented Inverse Probability
    Weighting) estimator for the ATE point estimate, NOT simple IPTW. This is
    because the variance formula (semiparametric efficiency bound) is the
    variance of AIPW, not IPTW.
    
    The AIPW estimator is:
    
    .. math::
    
        \\hat{\\mu}_{AIPW} = \\frac{1}{n}\\sum_{i=1}^{n}\\left[
            \\hat{m}_1(X_i) - \\hat{m}_0(X_i) +
            \\frac{T_i(Y_i - \\hat{m}_1(X_i))}{\\pi_i} -
            \\frac{(1-T_i)(Y_i - \\hat{m}_0(X_i))}{1-\\pi_i}
        \\right]
    
    where :math:`\\hat{m}_1(X) = E[Y(1)|X]` and :math:`\\hat{m}_0(X) = E[Y(0)|X]`
    are fitted via OLS on the treatment and control groups respectively.
    
    The oCBPS variance attains the semiparametric efficiency bound
    (Hahn, 1998; Fan et al., 2022, Eq. 2.6):

    .. math::

        V_{\\text{opt}} = E\\left[\\frac{\\text{Var}(Y(1)|X)}{\\pi(X)} +
            \\frac{\\text{Var}(Y(0)|X)}{1-\\pi(X)} + (L(X) - \\mu)^2\\right]

    where L(X) = E[Y(1) - Y(0) | X] is the conditional average treatment
    effect.

    The CBPS variance uses the sandwich formula (Fan et al., 2022, Eq. 2.4):

    .. math::

        V = \\bar{H}^{*T} \\Sigma \\bar{H}^*

    where :math:`\\bar{H}^* = (1, H_y^{*T})^T` and :math:`\\Sigma` is the
    joint variance-covariance matrix (Eq. 2.3) that accounts for
    propensity score estimation uncertainty.
    
    **Historical Note**: The original R CBPS package uses simple IPTW for the
    point estimate. This Python implementation corrects this mismatch to ensure
    the variance formula matches the estimator.

    Examples
    --------
    >>> from cbps import CBPS, AsyVar
    >>> fit = CBPS('treat ~ age + education', data=df, att=0)
    >>> result = AsyVar(Y=df['outcome'], CBPS_obj=fit, method='oCBPS')
    >>> print(f"ATE: {result['mu.hat']:.3f} (SE: {result['std.err']:.3f})")
    """

    # Parameter extraction (order: X -> TL -> pi -> mu)
    def _get_attr(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    # Extract X
    if X is None:
        if CBPS_obj is None:
            raise ValueError("Must specify X or CBPS_obj")
        X = _get_attr(CBPS_obj, 'x')

    # Extract TL
    if TL is None:
        if CBPS_obj is None:
            raise ValueError("Must specify TL or CBPS_obj")
        TL = _get_attr(CBPS_obj, 'y')

    # Extract pi
    if pi is None:
        if CBPS_obj is None:
            raise ValueError("Must specify pi or CBPS_obj")
        pi = _get_attr(CBPS_obj, 'fitted_values')

    # Dimension and range validation

    n = len(Y)
    p = X.shape[1]

    # Dimension checks
    if X.shape[0] != n:
        raise ValueError(f"X row count ({X.shape[0]}) does not match Y length ({n})")
    if len(TL) != n:
        raise ValueError(f"TL length ({len(TL)}) does not match Y length ({n})")
    if len(pi) != n:
        raise ValueError(f"pi length ({len(pi)}) does not match Y length ({n})")

    # Propensity score range validation
    if not np.all((pi >= 0) & (pi <= 1)):
        raise ValueError("Propensity score pi must be in [0, 1] interval")
    if np.any((pi <= 0) | (pi >= 1)):
        warnings.warn(
            "Propensity scores contain boundary values (0 or 1); "
            "variance estimate may be Inf or NaN.",
            UserWarning
        )

    # Treatment indicator validation
    if not np.all((TL == 0) | (TL == 1)):
        unique_tl = np.unique(TL)
        if len(unique_tl) > 4:
            raise ValueError(
                "asy_var supports binary treatment only. "
                f"Detected {len(unique_tl)} unique treatment values. "
                "For continuous treatment, use vcov_outcome()."
            )
        else:
            warnings.warn(
                f"Treatment indicator TL is not standard 0/1 coding "
                f"(values: {unique_tl}). "
                "Results may be invalid for non-binary treatment.",
                UserWarning
            )

    if not (0 < CI < 1):
        raise ValueError(f"Confidence level CI must be in (0, 1), got {CI}")
    if method not in ["CBPS", "oCBPS"]:
        raise ValueError(f"method must be 'CBPS' or 'oCBPS', got '{method}'")

    # Sample size requirement: n_group > p for residual variance estimation
    n_1 = np.sum(TL == 1)
    n_0 = np.sum(TL == 0)
    if n_1 <= p or n_0 <= p:
        raise ValueError(
            f"Group sizes must exceed covariate dimension p. "
            f"Current: n_1={n_1}, n_0={n_0}, p={p}."
        )

    # Extreme propensity score warning
    eps = 1e-6
    if np.any(pi <= eps) or np.any(pi >= 1 - eps):
        warnings.warn(
            f"Extreme propensity scores detected "
            f"(min={pi.min():.6f}, max={pi.max():.6f}); "
            "numerical instability may occur.",
            RuntimeWarning
        )

    # Fit outcome models for potential outcomes FIRST (needed for AIPW)
    Y_1_hat, Y_0_hat = _fit_outcome_models(Y, X, TL, Y_1_hat, Y_0_hat, p)

    # Compute ATE via AIPW if not provided
    # CRITICAL FIX: Use AIPW (Augmented IPW) instead of simple IPTW
    # The efficiency bound variance formula is for AIPW, not IPTW!
    # AIPW formula: (1/n) Σ [m₁(X) - m₀(X) + T(Y - m₁(X))/π - (1-T)(Y - m₀(X))/(1-π)]
    if mu is None:
        if CBPS_obj is None:
            raise ValueError("Must specify mu or CBPS_obj")
        # AIPW estimator (doubly robust, matches efficiency bound variance)
        mu = np.mean(
            Y_1_hat - Y_0_hat +
            TL * (Y - Y_1_hat) / pi -
            (1 - TL) * (Y - Y_0_hat) / (1 - pi)
        )

    # Auxiliary estimates: L(X) = CATE, K(X) = E[Y(0)|X]
    L_hat = Y_1_hat - Y_0_hat
    K_hat = Y_0_hat

    # Dispatch to method-specific variance computation
    if method == "oCBPS":
        result = _compute_asy_var_ocbps(
            Y, Y_1_hat, Y_0_hat, TL, pi, mu, L_hat, n, n_1, n_0, p, CI
        )
    elif method == "CBPS":
        # CBPS method requires propensity score coefficients
        if CBPS_obj is None:
            raise ValueError("CBPS method requires CBPS_obj to obtain coefficients")
        if isinstance(CBPS_obj, dict):
            if 'coefficients' not in CBPS_obj:
                raise ValueError("CBPS_obj must contain 'coefficients' key")
        else:
            if not hasattr(CBPS_obj, 'coefficients'):
                raise ValueError("CBPS_obj must have 'coefficients' attribute")

        beta = _get_attr(CBPS_obj, 'coefficients')
        # Ensure beta is 1D array
        if beta.ndim == 2:
            beta = beta.ravel()
        result = _compute_asy_var_cbps(
            Y, Y_1_hat, Y_0_hat, X, TL, pi, mu, L_hat, K_hat, beta, n, p, CI,
            use_observed_y
        )

    return result


def _fit_outcome_models(
    Y: np.ndarray,
    X: np.ndarray,
    TL: np.ndarray,
    Y_1_hat: Optional[np.ndarray],
    Y_0_hat: Optional[np.ndarray],
    p: int
) -> tuple:
    """
    Fit outcome models E[Y(1)|X] and E[Y(0)|X] via OLS.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Observed outcomes.
    X : ndarray of shape (n, p)
        Covariates with intercept column.
    TL : ndarray of shape (n,)
        Treatment indicator.
    Y_1_hat, Y_0_hat : ndarray or None
        Pre-computed predictions; fitted here if None.
    p : int
        Number of covariates including intercept.

    Returns
    -------
    Y_1_hat, Y_0_hat : ndarray of shape (n,)
        Full-sample predictions of potential outcomes.
    """
    # Fit E[Y(1)|X] on treatment group
    if Y_1_hat is None:
        X_1 = X[TL == 1]
        Y_1 = Y[TL == 1]
        model_1 = LinearRegression(fit_intercept=False).fit(X_1, Y_1)
        Y_1_hat = X @ model_1.coef_

    # Fit E[Y(0)|X] on control group
    if Y_0_hat is None:
        X_0 = X[TL == 0]
        Y_0 = Y[TL == 0]
        model_0 = LinearRegression(fit_intercept=False).fit(X_0, Y_0)
        Y_0_hat = X @ model_0.coef_

    return Y_1_hat, Y_0_hat


def _compute_asy_var_ocbps(
    Y: np.ndarray,
    Y_1_hat: np.ndarray,
    Y_0_hat: np.ndarray,
    TL: np.ndarray,
    pi: np.ndarray,
    mu: float,
    L_hat: np.ndarray,
    n: int,
    n_1: int,
    n_0: int,
    p: int,
    CI: float
) -> Dict[str, Any]:
    """
    Compute variance using the semiparametric efficiency bound.

    Implements the semiparametric efficiency-bound variance (Eq. 2.6) of
    Fan et al. (2022), which attains the Hahn (1998) efficiency bound.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Observed outcomes.
    Y_1_hat, Y_0_hat : ndarray of shape (n,)
        Predicted potential outcomes.
    TL : ndarray of shape (n,)
        Treatment indicator.
    pi : ndarray of shape (n,)
        Propensity scores.
    mu : float
        ATE point estimate.
    L_hat : ndarray of shape (n,)
        Estimated CATE, L(X) = E[Y(1) - Y(0) | X].
    n, n_1, n_0 : int
        Sample sizes (total, treated, control).
    p : int
        Number of covariates including intercept.
    CI : float
        Confidence level.

    Returns
    -------
    dict
        Keys: 'mu.hat', 'asy.var', 'var', 'std.err', 'CI.mu.hat'.
    """
    # Estimate conditional variances via residuals
    residuals_1 = (Y - Y_1_hat) * TL
    sigma_hat_1_squared = np.sum(residuals_1**2) / (n_1 - p)

    residuals_0 = (Y - Y_0_hat) * (1 - TL)
    sigma_hat_0_squared = np.sum(residuals_0**2) / (n_0 - p)

    # Semiparametric efficiency bound (Eq. 2.6)
    asy_var = np.mean(
        sigma_hat_1_squared / pi +
        sigma_hat_0_squared / (1 - pi) +
        (L_hat - mu)**2
    )

    # Finite-sample variance and standard error
    var = asy_var / n
    std_err = np.sqrt(var)

    # Confidence interval
    z_alpha = scipy.stats.norm.ppf(1 - (1 - CI) / 2)
    diff = z_alpha * std_err
    lower = mu - diff
    upper = mu + diff

    return {
        'mu.hat': mu,
        'asy.var': asy_var,
        'var': var,
        'std.err': std_err,
        'CI.mu.hat': np.array([lower, upper])
    }


def _compute_asy_var_cbps(
    Y: np.ndarray,
    Y_1_hat: np.ndarray,
    Y_0_hat: np.ndarray,
    X: np.ndarray,
    TL: np.ndarray,
    pi: np.ndarray,
    mu: float,
    L_hat: np.ndarray,
    K_hat: np.ndarray,
    beta: np.ndarray,
    n: int,
    p: int,
    CI: float,
    use_observed_y: bool = False
) -> Dict[str, Any]:
    """
    Compute variance using the sandwich formula.

    Implements the sandwich variance formula of Fan et al. (2022). The
    asymptotic variance is V = H_bar' Sigma H_bar (Eq. 2.4), where Sigma
    is the joint variance-covariance matrix defined in Eq. 2.3.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        Observed outcomes.
    Y_1_hat, Y_0_hat : ndarray of shape (n,)
        Predicted potential outcomes.
    X : ndarray of shape (n, p)
        Covariate matrix with intercept.
    TL : ndarray of shape (n,)
        Treatment indicator.
    pi : ndarray of shape (n,)
        Propensity scores.
    mu : float
        ATE point estimate.
    L_hat : ndarray of shape (n,)
        Estimated CATE, L(X) = E[Y(1) - Y(0) | X].
    K_hat : ndarray of shape (n,)
        Estimated baseline mean, K(X) = E[Y(0)|X].
    beta : ndarray of shape (p,)
        Propensity score coefficients.
    n : int
        Sample size.
    p : int
        Number of covariates including intercept.
    CI : float
        Confidence level.
    use_observed_y : bool
        Use observed Y (True, experimental) or predicted values (False,
        matches R package) in Sigma_mu.

    Returns
    -------
    dict
        Keys: 'mu.hat', 'asy.var', 'var', 'std.err', 'CI.mu.hat'.
    """
    # Common denominator pi(1-pi)
    denom = pi * (1 - pi)

    # Omega = var(g) = E[XX' / (pi(1-pi))] per Eq. 2.3
    omega_hat = np.mean(
        X[:, :, None] * X[:, None, :] / denom[:, None, None],
        axis=0
    )
    omega_hat = (omega_hat + omega_hat.T) / 2  # Symmetrize

    # Sigma_mu = var(mu_beta) per Eq. 2.3
    # CRITICAL FIX: The formula is E[Y(1)^2/pi + Y(0)^2/(1-pi)] - mu^2
    # where E[Y(1)^2|X] = Var(Y(1)|X) + E[Y(1)|X]^2 = sigma_1^2 + m_1(X)^2
    # The original implementation only used m_1(X)^2, missing the variance term!
    #
    # Correct formula: Sigma_mu = E[sigma_1^2/pi + sigma_0^2/(1-pi)] 
    #                           + E[m_1^2/pi + m_0^2/(1-pi)] - mu^2
    
    # Estimate conditional variances (same as oCBPS method)
    n_1 = np.sum(TL == 1)
    n_0 = np.sum(TL == 0)
    residuals_1 = (Y - Y_1_hat) * TL
    sigma_1_sq = np.sum(residuals_1**2) / max(n_1 - p, 1)
    residuals_0 = (Y - Y_0_hat) * (1 - TL)
    sigma_0_sq = np.sum(residuals_0**2) / max(n_0 - p, 1)
    
    if use_observed_y:
        # Use observed Y values: E[T*Y^2/pi + (1-T)*Y^2/(1-pi)]
        Sigma_mu_hat = np.mean(TL * Y**2 / pi + (1 - TL) * Y**2 / (1 - pi)) - mu**2
    else:
        # Use predicted values with estimated conditional variances
        # E[(sigma_1^2 + m_1^2)/pi + (sigma_0^2 + m_0^2)/(1-pi)] - mu^2
        Sigma_mu_hat = np.mean(
            (sigma_1_sq + Y_1_hat**2) / pi + 
            (sigma_0_sq + Y_0_hat**2) / (1 - pi)
        ) - mu**2

    # cov(mu_beta, g_beta) per Eq. 2.3
    cov_hat = np.mean(
        X * (K_hat + (1 - pi) * L_hat)[:, None] / denom[:, None],
        axis=0
    )

    # H_y^* = d(mu)/d(beta), the Jacobian defined after Eq. 2.2
    eta = X @ beta
    prop_modified = pi / (1.0 + np.exp(eta))
    derivative = prop_modified[:, None] * X
    factor = (K_hat + (1 - pi) * L_hat) / denom
    H_0_hat = -np.mean(derivative * factor[:, None], axis=0)

    # H_f^* = d(g)/d(beta), the Jacobian defined after Eq. 2.2
    H_f_hat = -np.mean(
        X[:, :, None] * (derivative / denom[:, None])[:, None, :], axis=0
    )
    H_f_hat = (H_f_hat + H_f_hat.T) / 2  # Symmetrize

    # Check condition numbers
    cond_Hf = np.linalg.cond(H_f_hat)
    cond_Om = np.linalg.cond(omega_hat)
    if cond_Hf > 1e10 or cond_Om > 1e10:
        warnings.warn(
            f"Ill-conditioned matrices: cond(H_f)={cond_Hf:.2e}, "
            f"cond(Omega)={cond_Om:.2e}.",
            RuntimeWarning
        )

    # Assemble V = H_bar' Sigma H_bar per Fan et al. (2022)
    # Computed as: Sigma_mu + H_y' Sigma_beta H_y + 2 H_y' Sigma_mu_beta
    term1 = Sigma_mu_hat

    # H_y' Sigma_beta H_y where Sigma_beta = (H_f' Omega^{-1} H_f)^{-1}
    try:
        A = scipy.linalg.solve(omega_hat, H_f_hat, assume_a='sym')
    except Exception:
        A = scipy.linalg.pinvh(omega_hat) @ H_f_hat
    G = H_f_hat.T @ A
    G = (G + G.T) / 2

    try:
        sol = scipy.linalg.solve(G, H_0_hat, assume_a='sym', check_finite=False)
        term2 = float(H_0_hat.T @ sol)
    except Exception:
        G_pinv = scipy.linalg.pinvh(G)
        term2 = float(H_0_hat.T @ (G_pinv @ H_0_hat))

    # 2 H_y' Sigma_mu_beta (note: Sigma_mu_beta has negative sign in Eq. 2.3)
    try:
        sol_hf = scipy.linalg.solve(
            H_f_hat, cov_hat, assume_a='sym', check_finite=False
        )
    except Exception:
        Hf_pinv = np.linalg.pinv(H_f_hat)
        sol_hf = Hf_pinv @ cov_hat
    term3 = float(2.0 * (H_0_hat.T @ sol_hf))

    # Asymptotic variance
    asy_var = float(term1 + term2 - term3)

    # Handle negative variance (numerical artifact)
    if asy_var < 0:
        if asy_var > -1e-12:
            warnings.warn(
                f"Asymptotic variance slightly negative ({asy_var:.2e}); clipped to 0.",
                RuntimeWarning,
                stacklevel=2
            )
            asy_var = 0.0
        else:
            warnings.warn(
                f"Asymptotic variance negative ({asy_var:.6f}). "
                "Consider using method='oCBPS' or bootstrap inference.",
                RuntimeWarning,
                stacklevel=2
            )

    # Finite-sample variance and confidence interval
    var = asy_var / n
    std_err = np.sqrt(var)

    z_alpha = scipy.stats.norm.ppf(1 - (1 - CI) / 2)
    diff = z_alpha * std_err
    lower = mu - diff
    upper = mu + diff

    return {
        'mu.hat': mu,
        'asy.var': asy_var,
        'var': var,
        'std.err': std_err,
        'CI.mu.hat': np.array([lower, upper])
    }
