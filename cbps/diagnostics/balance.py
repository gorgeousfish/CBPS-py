"""
Covariate Balance Assessment
============================

Functions for computing covariate balance statistics before and after CBPS
weighting. Balance diagnostics are essential for evaluating whether propensity
score methods have successfully removed confounding due to observed covariates.

For binary and multi-valued treatments, balance is measured via standardized
mean differences (SMD). For continuous treatments, balance is assessed via
weighted Pearson correlations between covariates and the treatment variable.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1),
156-177.

Stuart, E.A. (2010). Matching methods for causal inference: A review and a
look forward. Statistical Science, 25(1), 1-21.

Austin, P.C. (2009). Balance diagnostics for comparing the distribution of
baseline covariates between treatment groups in propensity-score matched
samples. Statistics in Medicine, 28(25), 3083-3107.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def balance_cbps(cbps_obj: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute covariate balance statistics for binary or multi-valued treatments.

    Calculates weighted and unweighted covariate means within each treatment
    group, along with their standardized versions (divided by pooled standard
    deviation). These statistics enable assessment of covariate balance before
    and after CBPS weighting.

    Parameters
    ----------
    cbps_obj : dict
        Fitted CBPS result containing:

        - **weights** : ndarray of shape (n,) - CBPS weights
        - **x** : ndarray of shape (n, k) - Covariate matrix with intercept
        - **y** : ndarray of shape (n,) - Treatment indicator

    Returns
    -------
    dict
        Balance statistics with keys:

        - **balanced** : ndarray of shape (n_covars, 2*n_treats)
          Weighted covariate means (first n_treats columns) and
          standardized weighted means (remaining columns).
        - **original** : ndarray of shape (n_covars, 2*n_treats)
          Unweighted covariate means and standardized means.

    Notes
    -----
    The standardized mean difference (SMD) between treatment groups can be
    computed from the output as the difference in standardized means across
    columns. For binary treatments with groups 0 and 1:

    .. math::

        \\text{SMD} = |\\bar{X}_1 - \\bar{X}_0| / s

    where :math:`s` is the full-sample standard deviation (computed across
    all observations regardless of treatment). This choice ensures the
    standardization denominator remains constant before and after weighting.

    Following Austin (2009), SMD < 0.1 indicates acceptable balance.

    Examples
    --------
    >>> import cbps
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>> fit = cbps.CBPS('treat ~ age + educ + re74', data=df, att=1)
    >>> cbps_dict = {'weights': fit.weights, 'x': fit.x, 'y': fit.y}
    >>> from cbps.diagnostics.balance import balance_cbps
    >>> result = balance_cbps(cbps_dict)
    >>> print(result['balanced'].shape)
    (3, 4)
    """
    # Detect npCBPS object (has log_el attribute)
    is_npcbps = 'log_el' in cbps_obj
    
    # Step 1: Extract input data
    treats = pd.Categorical(cbps_obj['y'])
    treat_levels = treats.categories
    n_treats = len(treat_levels)
    
    # Extract X matrix and weights
    X = cbps_obj['x']  # Covariate matrix
    w = cbps_obj['weights']  # CBPS weights
    
    # Step 2: Initialize result matrices
    # npCBPS: X has no intercept, all columns are covariates
    # CBPS: X has intercept in column 0, skip it
    if is_npcbps:
        n_covars = X.shape[1]  # All columns are covariates
        j_start = 0  # Start from column 0
    else:
        n_covars = X.shape[1] - 1  # Exclude intercept column
        j_start = 1  # Skip intercept column X[:, 0]
    
    bal = np.zeros((n_covars, 2 * n_treats), dtype=np.float64)
    baseline = np.zeros((n_covars, 2 * n_treats), dtype=np.float64)
    
    # Step 3: Compute weighted means and standardized values
    for i, level in enumerate(treat_levels):
        # Create treatment group mask
        mask = (treats == level).values if hasattr(treats, 'values') else (treats == level)
        
        for j_col in range(j_start, X.shape[1]):
            idx = j_col - j_start
            
            # Weighted mean
            bal[idx, i] = np.sum(mask * X[:, j_col] * w) / np.sum(w * mask)
            
            # Standardized mean (using pooled std)
            std_pooled = np.std(X[:, j_col], ddof=1)
            bal[idx, i + n_treats] = bal[idx, i] / std_pooled
            
            # Original mean (unweighted baseline)
            baseline[idx, i] = np.mean(X[mask, j_col])
            
            # Standardized original mean
            baseline[idx, i + n_treats] = baseline[idx, i] / std_pooled
    
    # Step 4: Return results
    return {"balanced": bal, "original": baseline}


def _compute_smd_binary(bal: np.ndarray, baseline: np.ndarray, n_treats: int) -> tuple:
    """
    Compute standardized mean differences from balance matrices.

    Parameters
    ----------
    bal : ndarray of shape (n_covars, 2*n_treats)
        Weighted balance matrix from balance_cbps.
    baseline : ndarray of shape (n_covars, 2*n_treats)
        Unweighted balance matrix from balance_cbps.
    n_treats : int
        Number of treatment levels.

    Returns
    -------
    smd_weighted : ndarray of shape (n_covars, n_treats-1)
        Absolute SMD after weighting.
    smd_unweighted : ndarray of shape (n_covars, n_treats-1)
        Absolute SMD before weighting (baseline).

    Notes
    -----
    For binary treatments, SMD is computed as the absolute difference
    between standardized means of the two groups. For multi-valued
    treatments, SMD is computed for each non-reference group versus
    the reference group.
    """
    n_covars = bal.shape[0]
    
    # Extract standardized means (last n_treats columns)
    std_means_weighted = bal[:, n_treats:]  # (n_covars, n_treats)
    std_means_unweighted = baseline[:, n_treats:]
    
    # Compute SMD: compare first group vs other groups
    if n_treats == 2:
        # Binary treatment: group1 - group0
        smd_weighted = np.abs(std_means_weighted[:, 1] - std_means_weighted[:, 0]).reshape(-1, 1)
        smd_unweighted = np.abs(std_means_unweighted[:, 1] - std_means_unweighted[:, 0]).reshape(-1, 1)
    else:
        # Multi-valued treatment: each group vs first group
        smd_weighted = np.abs(std_means_weighted[:, 1:] - std_means_weighted[:, [0]])
        smd_unweighted = np.abs(std_means_unweighted[:, 1:] - std_means_unweighted[:, [0]])
    
    return smd_weighted, smd_unweighted


def balance_cbps_enhanced(
    cbps_obj: Dict[str, Any],
    threshold: float = 0.1,
    covariate_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Extended balance diagnostics with summary statistics and reports.

    Augments the basic balance_cbps output with improvement metrics,
    threshold-based assessments, and a formatted text report suitable
    for publication or diagnostic review.

    Parameters
    ----------
    cbps_obj : dict
        Fitted CBPS result object.
    threshold : float, default=0.1
        SMD threshold for acceptable balance. The conventional threshold
        of 0.1 follows Stuart (2010) and Austin (2009).
    covariate_names : list of str, optional
        Names for covariates. If provided, included in detailed report.

    Returns
    -------
    dict
        Extended diagnostics containing:

        - **balanced** : ndarray - Weighted balance matrix
        - **original** : ndarray - Unweighted balance matrix
        - **smd_weighted** : ndarray - SMD after weighting
        - **smd_unweighted** : ndarray - SMD before weighting
        - **improvement_pct** : ndarray - Percent reduction in SMD
        - **n_imbalanced_before** : int - Count exceeding threshold before
        - **n_imbalanced_after** : int - Count exceeding threshold after
        - **summary** : dict - Aggregate statistics
        - **report** : str - Formatted text report
    """
    # Call the base balance function
    result = balance_cbps(cbps_obj)
    bal = result['balanced']
    baseline = result['original']
    
    # Extract number of treatment levels
    treats = pd.Categorical(cbps_obj['y'])
    n_treats = len(treats.categories)
    
    # Compute SMD
    smd_weighted, smd_unweighted = _compute_smd_binary(bal, baseline, n_treats)
    
    # Compute improvement percentage
    improvement_pct = np.zeros_like(smd_weighted)
    nonzero_mask = smd_unweighted > 1e-10
    improvement_pct[nonzero_mask] = (
        (smd_unweighted[nonzero_mask] - smd_weighted[nonzero_mask]) / 
        smd_unweighted[nonzero_mask] * 100
    )
    
    # Count imbalanced covariates
    n_imbalanced_before = np.sum(smd_unweighted > threshold)
    n_imbalanced_after = np.sum(smd_weighted > threshold)
    
    # Summary statistics
    summary = {
        'mean_smd_before': float(np.mean(smd_unweighted)),
        'mean_smd_after': float(np.mean(smd_weighted)),
        'max_smd_before': float(np.max(smd_unweighted)),
        'max_smd_after': float(np.max(smd_weighted)),
        'n_imbalanced_before': int(n_imbalanced_before),
        'n_imbalanced_after': int(n_imbalanced_after),
        'pct_imbalanced_before': float(n_imbalanced_before / smd_unweighted.size * 100),
        'pct_imbalanced_after': float(n_imbalanced_after / smd_weighted.size * 100),
        'mean_improvement_pct': float(np.mean(improvement_pct[nonzero_mask])) if nonzero_mask.any() else 0.0
    }
    
    # Generate text report
    n_covars = bal.shape[0]
    report = f"""
Covariate Balance Diagnostic Report
{'=' * 60}

Sample Statistics:
  Number of covariates: {n_covars}
  Number of treatment groups: {n_treats}
  SMD threshold: {threshold}

Balance Before Weighting:
  Mean SMD: {summary['mean_smd_before']:.4f}
  Max SMD: {summary['max_smd_before']:.4f}
  Imbalanced covariates: {summary['n_imbalanced_before']} ({summary['pct_imbalanced_before']:.1f}%)

Balance After CBPS Weighting:
  Mean SMD: {summary['mean_smd_after']:.4f}
  Max SMD: {summary['max_smd_after']:.4f}
  Imbalanced covariates: {summary['n_imbalanced_after']} ({summary['pct_imbalanced_after']:.1f}%)

Improvement:
  Mean improvement: {summary['mean_improvement_pct']:.1f}%
  Reduced imbalanced covariates: {summary['n_imbalanced_before'] - summary['n_imbalanced_after']}

Interpretation:
  SMD < 0.1: Excellent balance
  SMD < 0.25: Acceptable balance
  SMD > 0.25: Poor balance (consider model adjustment)
{'=' * 60}
"""
    
    # Detailed covariate report
    if covariate_names is not None and len(covariate_names) == n_covars:
        report += "\nDetailed Covariate Balance:\n"
        report += f"{'Covariate':<20} {'Before':>10} {'After':>10} {'Improve':>10} {'Status':>10}\n"
        report += "-" * 65 + "\n"
        
        for i in range(n_covars):
            name = covariate_names[i][:18]
            before = smd_unweighted[i, 0]
            after = smd_weighted[i, 0]
            improve = improvement_pct[i, 0]
            
            if after < 0.1:
                status = "Excellent"
            elif after < 0.25:
                status = "Good"
            else:
                status = "⚠ Poor"
            
            report += f"{name:<20} {before:>10.4f} {after:>10.4f} {improve:>9.1f}% {status:>10}\n"
    
    # Return enhanced results
    return {
        # Core output
        'balanced': bal,
        'original': baseline,
        # Enhanced output
        'smd_weighted': smd_weighted,
        'smd_unweighted': smd_unweighted,
        'improvement_pct': improvement_pct,
        'n_imbalanced_before': int(n_imbalanced_before),
        'n_imbalanced_after': int(n_imbalanced_after),
        'summary': summary,
        'report': report
    }


def balance_cbps_continuous_enhanced(
    cbps_obj: Dict[str, Any],
    threshold: float = 0.1,
    covariate_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Extended balance diagnostics for continuous treatments.

    Augments balance_cbps_continuous with improvement metrics and a
    formatted text report for continuous treatment settings.

    Parameters
    ----------
    cbps_obj : dict
        Fitted continuous treatment CBPS result object.
    threshold : float, default=0.1
        Absolute correlation threshold for imbalance detection.
    covariate_names : list of str, optional
        Names for covariates. If provided, included in detailed report.

    Returns
    -------
    dict
        Extended diagnostics containing:

        - **balanced** : ndarray - Weighted correlations
        - **unweighted** : ndarray - Unweighted correlations
        - **abs_corr_weighted** : ndarray - Absolute weighted correlations
        - **abs_corr_unweighted** : ndarray - Absolute unweighted correlations
        - **improvement_pct** : ndarray - Percent reduction in |correlation|
        - **n_imbalanced_before** : int - Count exceeding threshold before
        - **n_imbalanced_after** : int - Count exceeding threshold after
        - **summary** : dict - Aggregate statistics
        - **report** : str - Formatted text report
    """
    # Call the base balance function
    result = balance_cbps_continuous(cbps_obj)
    bal = result['balanced']
    baseline = result['unweighted']
    
    # Compute absolute correlations
    abs_corr_weighted = np.abs(bal)
    abs_corr_unweighted = np.abs(baseline)
    
    # Compute improvement
    improvement = abs_corr_unweighted - abs_corr_weighted
    improvement_pct = np.zeros_like(improvement)
    nonzero_mask = abs_corr_unweighted > 1e-10
    improvement_pct[nonzero_mask] = (
        improvement[nonzero_mask] / abs_corr_unweighted[nonzero_mask] * 100
    )
    
    # Count imbalanced covariates
    n_imbalanced_before = np.sum(abs_corr_unweighted > threshold)
    n_imbalanced_after = np.sum(abs_corr_weighted > threshold)
    
    # Summary statistics
    n_covars = bal.shape[0]
    summary = {
        'mean_abs_corr_before': float(np.mean(abs_corr_unweighted)),
        'mean_abs_corr_after': float(np.mean(abs_corr_weighted)),
        'max_abs_corr_before': float(np.max(abs_corr_unweighted)),
        'max_abs_corr_after': float(np.max(abs_corr_weighted)),
        'n_imbalanced_before': int(n_imbalanced_before),
        'n_imbalanced_after': int(n_imbalanced_after),
        'pct_imbalanced_before': float(n_imbalanced_before / n_covars * 100),
        'pct_imbalanced_after': float(n_imbalanced_after / n_covars * 100),
        'mean_improvement_pct': float(np.mean(improvement_pct[nonzero_mask])) if nonzero_mask.any() else 0.0
    }
    
    # Generate text report
    report = f"""
Covariate Balance Diagnostic Report (Continuous Treatment)
{'=' * 60}

Sample Statistics:
  Number of covariates: {n_covars}
  Correlation threshold: {threshold}

Balance Before Weighting:
  Mean |correlation|: {summary['mean_abs_corr_before']:.4f}
  Max |correlation|: {summary['max_abs_corr_before']:.4f}
  Imbalanced covariates: {summary['n_imbalanced_before']} ({summary['pct_imbalanced_before']:.1f}%)

Balance After CBGPS Weighting:
  Mean |correlation|: {summary['mean_abs_corr_after']:.4f}
  Max |correlation|: {summary['max_abs_corr_after']:.4f}
  Imbalanced covariates: {summary['n_imbalanced_after']} ({summary['pct_imbalanced_after']:.1f}%)

Improvement:
  Mean improvement: {summary['mean_improvement_pct']:.1f}%
  Reduced imbalanced covariates: {summary['n_imbalanced_before'] - summary['n_imbalanced_after']}

Interpretation:
  |correlation| ≈ 0: Excellent balance
  |correlation| < 0.1: Good balance
  |correlation| > 0.2: Poor balance (consider model adjustment)
{'=' * 60}
"""
    
    # Detailed covariate report
    if covariate_names is not None and len(covariate_names) == n_covars:
        report += "\nDetailed Covariate Balance:\n"
        report += f"{'Covariate':<20} {'Before':>10} {'After':>10} {'Improve':>10} {'Status':>10}\n"
        report += "-" * 65 + "\n"
        
        for i in range(n_covars):
            name = covariate_names[i][:18]
            before = abs_corr_unweighted[i, 0]
            after = abs_corr_weighted[i, 0]
            improve = improvement_pct[i, 0]
            
            if after < 0.05:
                status = "Excellent"
            elif after < 0.1:
                status = "Good"
            else:
                status = "⚠ Poor"
            
            report += f"{name:<20} {before:>10.4f} {after:>10.4f} {improve:>9.1f}% {status:>10}\n"
    
    return {
        # Standard balance output
        'balanced': bal,
        'unweighted': baseline,
        # Enhanced diagnostics
        'abs_corr_weighted': abs_corr_weighted,
        'abs_corr_unweighted': abs_corr_unweighted,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'n_imbalanced_before': int(n_imbalanced_before),
        'n_imbalanced_after': int(n_imbalanced_after),
        'summary': summary,
        'report': report
    }


def balance_cbps_continuous(cbps_obj: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute weighted Pearson correlations for continuous treatments.

    For continuous treatments, covariate balance is assessed by the correlation
    between each covariate and the treatment variable. Effective weighting
    should reduce these correlations toward zero, indicating that covariates
    are no longer predictive of treatment assignment.

    Parameters
    ----------
    cbps_obj : dict
        Fitted continuous treatment CBPS result containing:

        - **weights** : ndarray of shape (n,) - Stabilized inverse probability weights
        - **x** : ndarray of shape (n, k) - Covariate matrix with intercept
        - **y** : ndarray of shape (n,) - Continuous treatment variable

    Returns
    -------
    dict
        Correlation statistics with keys:

        - **balanced** : ndarray of shape (n_covars, 1)
          Weighted Pearson correlations between treatment and each covariate.
        - **unweighted** : ndarray of shape (n_covars, 1)
          Unweighted Pearson correlations (baseline).

    Notes
    -----
    The weighted correlation is computed using the formula:

    .. math::

        r_w = \\frac{\\sum w_i (X_i - \\bar{X}_w)(T_i - \\bar{T}_w)}
                    {\\sqrt{\\sum w_i (X_i - \\bar{X}_w)^2}
                     \\sqrt{\\sum w_i (T_i - \\bar{T}_w)^2}}

    where :math:`\\bar{X}_w` and :math:`\\bar{T}_w` are weighted means.

    A correlation near zero after weighting indicates that the covariate
    is balanced with respect to the treatment.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. The Annals of Applied Statistics, 12(1),
    156-177.

    Examples
    --------
    >>> import cbps
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     'dose': np.random.uniform(0, 100, n),
    ...     'age': np.random.normal(45, 12, n),
    ...     'income': np.random.lognormal(10, 0.5, n)
    ... })
    >>> fit = cbps.CBPS('dose ~ age + income', data=df, att=0)  # doctest: +SKIP
    >>> cbps_dict = {'weights': fit.weights, 'x': fit.x, 'y': fit.y}  # doctest: +SKIP
    >>> from cbps.diagnostics.balance import balance_cbps_continuous
    >>> result = balance_cbps_continuous(cbps_dict)  # doctest: +SKIP
    >>> print(result['balanced'].shape)  # doctest: +SKIP
    (2, 1)
    """
    # Step 1: Extract input data
    treat = cbps_obj['y']  # Continuous treatment vector
    X = cbps_obj['x']  # Covariate matrix (with or without intercept)
    w = cbps_obj['weights']  # Optimal weights
    n = len(w)  # Sample size
    
    # Step 2: Detect npCBPS vs CBPS
    # npCBPS: X matrix does NOT have intercept column (all columns are covariates)
    # CBPS: X matrix HAS intercept column (first column is all 1s)
    is_npcbps = 'log_el' in cbps_obj
    
    if is_npcbps:
        # npCBPS path: X has no intercept, start from column 0
        j_start = 0
        n_covars = X.shape[1]
    else:
        # CBPS path: X has intercept, skip first column
        j_start = 1
        n_covars = X.shape[1] - 1
    
    # Initialize result vectors
    bal = np.zeros((n_covars, 1), dtype=np.float64)
    baseline = np.zeros((n_covars, 1), dtype=np.float64)
    
    # Step 3: Compute weighted Pearson correlations
    # R code reference (balance.CBPSContinuous):
    # bal[j,1]<-(mean(w*X[,j]*treat) - mean(w*X[,j])*mean(w*treat)*n/sum(w))/
    #           (sqrt(mean(w*X[,j]^2) - mean(w*X[,j])^2*n/sum(w))*
    #            sqrt(mean(w*treat^2) - mean(w*treat)^2*n/sum(w)))
    # baseline[j,1]<-cor(treat, X[,j], method = "pearson")
    
    for j_col in range(j_start, X.shape[1]):
        idx = j_col - j_start  # Row index: 0, 1, 2, ..., n_covars-1
        
        # Weighted Pearson correlation 6-term formula
        # 1. Compute 3 weighted means
        mean_wXT = np.mean(w * X[:, j_col] * treat)  # mean(w*X*T)
        mean_wX = np.mean(w * X[:, j_col])  # mean(w*X)
        mean_wT = np.mean(w * treat)  # mean(w*T)
        sum_w = np.sum(w)  # sum(w)
        
        # 2. Numerator: weighted covariance (with correction n/sum(w))
        numerator = mean_wXT - mean_wX * mean_wT * n / sum_w
        
        # 3. Denominator term 1: weighted variance of X (with correction)
        var_wX = np.mean(w * X[:, j_col]**2) - mean_wX**2 * n / sum_w
        
        # 4. Denominator term 2: weighted variance of T (with correction)
        var_wT = np.mean(w * treat**2) - mean_wT**2 * n / sum_w
        
        # 5. Denominator: product of standard deviations
        denominator = np.sqrt(var_wX) * np.sqrt(var_wT)
        
        # 6. Weighted correlation coefficient
        bal[idx, 0] = numerator / denominator
        
        # Baseline: unweighted standard Pearson correlation
        baseline[idx, 0] = np.corrcoef(treat, X[:, j_col])[0, 1]
    
    # Step 4: Return results
    # Note: key is "unweighted" rather than "original" (differs from binary treatment)
    return {"balanced": bal, "unweighted": baseline}

