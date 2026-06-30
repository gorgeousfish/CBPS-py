"""
Continuous Treatment CBPS Diagnostic Tools
==========================================

Diagnostic utilities for assessing the quality of CBGPS (Covariate Balancing
Generalized Propensity Score) estimation results.

Author: CBPS Development Team
Date: 2026-01-28
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional
import statsmodels.api as sm


def compute_f_statistic(treat: np.ndarray, X: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """
    Compute F-statistic from weighted regression of treatment on covariates.
    
    This is the primary balance metric used in Fong et al. (2018).
    Lower F-statistic indicates better covariate balance.
    
    Parameters
    ----------
    treat : np.ndarray
        Treatment variable (n,)
    X : np.ndarray
        Covariate matrix (n, K), should NOT include intercept
    weights : np.ndarray
        CBGPS weights (n,)
        
    Returns
    -------
    f_stat : float
        F-statistic from weighted regression T ~ X
    p_value : float
        P-value for the F-test
    """
    n, K = X.shape
    X_with_const = sm.add_constant(X)
    
    # Normalize weights
    weights_norm = weights * n / weights.sum()
    
    # Weighted least squares
    model = sm.WLS(treat, X_with_const, weights=weights_norm)
    result = model.fit()
    
    return result.fvalue, result.f_pvalue


def compute_weighted_correlations(treat: np.ndarray, X: np.ndarray, 
                                  weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted Pearson correlations between treatment and each covariate.
    
    Parameters
    ----------
    treat : np.ndarray
        Treatment variable (n,)
    X : np.ndarray
        Covariate matrix (n, K)
    weights : np.ndarray
        CBGPS weights (n,)
        
    Returns
    -------
    np.ndarray
        Weighted correlations for each covariate (K,)
    """
    n, K = X.shape
    w_sum = np.sum(weights)
    w_mean_T = np.sum(weights * treat) / w_sum
    
    correlations = []
    for j in range(K):
        X_j = X[:, j]
        w_mean_X = np.sum(weights * X_j) / w_sum
        
        w_cov = np.sum(weights * (treat - w_mean_T) * (X_j - w_mean_X)) / w_sum
        w_var_T = np.sum(weights * (treat - w_mean_T)**2) / w_sum
        w_var_X = np.sum(weights * (X_j - w_mean_X)**2) / w_sum
        
        if w_var_T > 1e-10 and w_var_X > 1e-10:
            corr = w_cov / np.sqrt(w_var_T * w_var_X)
        else:
            corr = np.nan
        
        correlations.append(corr)
    
    return np.array(correlations)


def diagnose_cbgps_quality(cbgps_result: Dict, treat: np.ndarray, 
                           X: np.ndarray) -> Dict[str, any]:
    """
    Diagnose the quality of CBGPS estimation results.
    
    This function assesses covariate balance quality and provides
    recommendations for users.
    
    Parameters
    ----------
    cbgps_result : dict
        Result dictionary from cbps_continuous_fit()
    treat : np.ndarray
        Treatment variable (n,)
    X : np.ndarray
        Covariate matrix (n, K), without intercept
        
    Returns
    -------
    dict
        Diagnostic information including:
        - quality_level: 'excellent', 'acceptable', 'moderate', or 'poor'
        - f_statistic: Overall balance F-statistic
        - max_abs_correlation: Maximum absolute weighted correlation
        - mean_abs_correlation: Mean absolute weighted correlation
        - recommendation: User-facing recommendation string
        - j_statistic: GMM objective value (if available)
        
    Examples
    --------
    >>> from cbps.core import cbps_continuous_fit
    >>> result = cbps_continuous_fit(treat, X, method='exact')
    >>> diag = diagnose_cbgps_quality(result, treat, X)
    >>> print(f"Balance quality: {diag['quality_level']}")
    >>> print(f"Recommendation: {diag['recommendation']}")
    """
    weights = cbgps_result.get('weights')
    if weights is None:
        raise ValueError("cbgps_result must contain 'weights' key")
    
    # Compute balance metrics
    f_stat, f_pval = compute_f_statistic(treat, X, weights)
    correlations = compute_weighted_correlations(treat, X, weights)
    max_abs_corr = np.max(np.abs(correlations))
    mean_abs_corr = np.mean(np.abs(correlations))
    
    # Determine quality level
    if f_stat < 0.1 and max_abs_corr < 0.05:
        quality_level = 'excellent'
        recommendation = (
            "Excellent covariate balance achieved. "
            "Proceed with confidence using these weights."
        )
    elif f_stat < 0.5 and max_abs_corr < 0.15:
        quality_level = 'acceptable'
        recommendation = (
            "Acceptable covariate balance. "
            "Check balance_table() for any concerning imbalances."
        )
    elif f_stat < 2.0:
        quality_level = 'moderate'
        recommendation = (
            "Moderate balance quality. Consider: "
            "(1) using npCBPS for more robust results, "
            "(2) reducing the number of covariates, or "
            "(3) increasing sample size if possible."
        )
    else:
        quality_level = 'poor'
        recommendation = (
            "Poor balance quality detected. Strongly recommend: "
            "(1) switching to npCBPS (method='nonparametric'), "
            "(2) careful variable selection to reduce noise covariates, or "
            "(3) verifying data quality and model specification."
        )
    
    # Collect diagnostic info
    diagnostic = {
        'quality_level': quality_level,
        'f_statistic': float(f_stat),
        'f_pvalue': float(f_pval),
        'max_abs_correlation': float(max_abs_corr),
        'mean_abs_correlation': float(mean_abs_corr),
        'correlations': correlations,
        'recommendation': recommendation,
        'converged': cbgps_result.get('converged', None),
        'j_statistic': cbgps_result.get('J', None),
        'weight_range': (float(np.min(weights)), float(np.max(weights))),
        'weight_sum': float(np.sum(weights)),
    }
    
    return diagnostic


def print_balance_diagnosis(diagnostic: Dict) -> None:
    """
    Print a user-friendly diagnostic report.
    
    Parameters
    ----------
    diagnostic : dict
        Output from diagnose_cbgps_quality()
    """
    print("=" * 70)
    print("CBGPS Balance Quality Diagnostic Report")
    print("=" * 70)
    print()
    
    # Quality badge
    quality = diagnostic['quality_level']
    badges = {
        'excellent': '✓✓✓ EXCELLENT',
        'acceptable': '✓✓ ACCEPTABLE',
        'moderate': '⚠ MODERATE',
        'poor': '✗ POOR'
    }
    print(f"Overall Quality: {badges.get(quality, quality.upper())}")
    print()
    
    # Metrics
    print("Balance Metrics:")
    print(f"  F-statistic: {diagnostic['f_statistic']:.4f} (p={diagnostic['f_pvalue']:.4f})")
    print(f"  Max |correlation|: {diagnostic['max_abs_correlation']:.4f}")
    print(f"  Mean |correlation|: {diagnostic['mean_abs_correlation']:.4f}")
    
    if diagnostic['j_statistic'] is not None:
        print(f"  J-statistic: {diagnostic['j_statistic']:.6f}")
    
    if diagnostic['converged'] is not None:
        conv_str = "Yes" if diagnostic['converged'] else "No"
        print(f"  Converged: {conv_str}")
    
    print()
    
    # Weight diagnostics
    w_min, w_max = diagnostic['weight_range']
    print(f"Weight Diagnostics:")
    print(f"  Sum of weights: {diagnostic['weight_sum']:.4f} (should be ~1.0)")
    print(f"  Weight range: [{w_min:.4f}, {w_max:.4f}]")
    
    if w_max / w_min > 100:
        print("  ⚠ Warning: Extreme weight ratio detected")
    
    print()
    
    # Recommendation
    print("Recommendation:")
    print(f"  {diagnostic['recommendation']}")
    print()
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # This is a demonstration - normally called after cbps_continuous_fit()
    print("Continuous CBPS Diagnostic Tools")
    print()
    print("Usage:")
    print("  from cbps.diagnostics.continuous_diagnostics import diagnose_cbgps_quality")
    print("  from cbps.core import cbps_continuous_fit")
    print()
    print("  result = cbps_continuous_fit(treat, X, method='exact')")
    print("  diag = diagnose_cbgps_quality(result, treat, X)")
    print("  print_balance_diagnosis(diag)")
