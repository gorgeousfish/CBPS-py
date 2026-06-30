"""Normality diagnostics for continuous treatment CBPS.

Fong, Hazlett, and Imai (2018) assume the treatment variable is
conditionally normal: T | X ~ N(X'beta, sigma^2). This module provides
diagnostic tools to check this assumption.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1):
156-177.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any
import warnings


def test_treatment_normality(
    treat: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test normality of treatment residuals (T - X'beta_hat).

    Under the continuous CBPS model, T | X ~ N(X'beta, sigma^2). This
    function fits OLS to obtain residuals e = T - X @ beta_hat, then
    tests whether these residuals follow a normal distribution.

    For n <= 5000, the Shapiro-Wilk test is used (highest power for
    detecting departures from normality in moderate samples). For n > 5000,
    the D'Agostino-Pearson omnibus test is used (Shapiro-Wilk becomes
    computationally expensive and overly sensitive for very large n).

    Parameters
    ----------
    treat : np.ndarray, shape (n,)
        Continuous treatment variable.
    X : np.ndarray, shape (n, k)
        Covariate matrix (should include intercept if desired).
    alpha : float, default=0.05
        Significance level for the normality test.

    Returns
    -------
    dict with:
        - statistic : float
            Test statistic value.
        - p_value : float
            p-value of the normality test.
        - test_used : str
            Name of the test applied ('shapiro-wilk' or 'dagostino-pearson').
        - reject_normality : bool
            True if normality is rejected at the given alpha level.
        - skewness : float
            Sample skewness of residuals.
        - kurtosis : float
            Sample excess kurtosis of residuals.
        - warning_message : str or None
            If normality is rejected, suggests using npCBPS.

    Notes
    -----
    If normality is rejected, the parametric CBPS estimator may be
    misspecified. Consider using the nonparametric variant (npCBPS)
    which does not require distributional assumptions on T | X.
    """
    treat = np.asarray(treat, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = len(treat)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Input validation: NaN/Inf checks
    if np.any(~np.isfinite(treat)):
        return {
            'statistic': np.nan, 'p_value': np.nan, 'test_used': 'none',
            'reject_normality': False, 'skewness': np.nan, 'kurtosis': np.nan,
            'n_observations': n,
            'warning_message': "Treatment variable contains NaN or Inf values. "
                              "Cannot perform normality test. Clean data first."
        }
    if np.any(~np.isfinite(X)):
        return {
            'statistic': np.nan, 'p_value': np.nan, 'test_used': 'none',
            'reject_normality': False, 'skewness': np.nan, 'kurtosis': np.nan,
            'n_observations': n,
            'warning_message': "Covariate matrix contains NaN or Inf values. "
                              "Cannot perform normality test. Clean data first."
        }

    # Minimum sample size check
    if n < 3:
        return {
            'statistic': np.nan, 'p_value': np.nan, 'test_used': 'none',
            'reject_normality': False, 'skewness': np.nan, 'kurtosis': np.nan,
            'n_observations': n,
            'warning_message': f"Sample size n={n} too small for normality testing "
                              f"(minimum 3 required for Shapiro-Wilk)."
        }

    if X.shape[0] != n:
        raise ValueError(
            f"Dimension mismatch: treat has {n} observations but X has "
            f"{X.shape[0]} rows."
        )

    # Fit OLS: beta_hat = (X'X)^{-1} X'T
    # Use lstsq for numerical stability
    beta_hat, _, _, _ = np.linalg.lstsq(X, treat, rcond=None)
    residuals = treat - X @ beta_hat

    # Near-zero residual variance guard
    resid_std = np.std(residuals, ddof=1)
    if resid_std < 1e-10:
        return {
            'statistic': np.nan, 'p_value': np.nan, 'test_used': 'none',
            'reject_normality': False, 'skewness': 0.0, 'kurtosis': 0.0,
            'n_observations': n,
            'warning_message': "Treatment is a near-perfect linear function of covariates "
                              f"(residual std = {resid_std:.2e}). Normality test not applicable."
        }

    # Compute descriptive statistics
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))  # excess kurtosis

    # Choose test based on sample size
    if n <= 5000:
        test_used = "shapiro-wilk"
        stat, p_value = stats.shapiro(residuals)
    else:
        test_used = "dagostino-pearson"
        stat, p_value = stats.normaltest(residuals)

    stat = float(stat)
    p_value = float(p_value)

    # NaN result guard
    if not np.isfinite(stat) or not np.isfinite(p_value):
        return {
            'statistic': stat, 'p_value': p_value, 'test_used': test_used,
            'reject_normality': False, 'skewness': np.nan, 'kurtosis': np.nan,
            'n_observations': n,
            'warning_message': f"Normality test ({test_used}) returned non-finite result. "
                              f"This may indicate degenerate data or numerical issues."
        }

    reject = p_value < alpha

    # Construct warning message
    warning_message = None
    if reject:
        warning_message = (
            f"Normality of treatment residuals rejected ({test_used} test, "
            f"p={p_value:.4g}, alpha={alpha}). The conditional normality "
            f"assumption T|X ~ N(X'beta, sigma^2) may not hold. Consider "
            f"using npCBPS (nonparametric CBPS) which does not require "
            f"distributional assumptions."
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)

    return {
        "statistic": stat,
        "p_value": p_value,
        "test_used": test_used,
        "reject_normality": reject,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "warning_message": warning_message,
    }
