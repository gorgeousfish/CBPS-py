"""Condition verification for optimal CBPS (Fan et al. 2022).

The optimal CBPS (oCBPS) of Fan et al. (2022) achieves semiparametric
efficiency under specific regularity conditions. This module provides
observable checks for necessary conditions that can be empirically verified.

NOTE: Some conditions (e.g., correct specification of the propensity score
model, smoothness of the true propensity score function) cannot be directly
tested from data. This module checks only observable necessary conditions.

References
----------
Fan, J., Imai, K., Liu, H., Ning, Y., and Yang, X. (2022). Optimal
Covariate Balancing Conditions in Propensity Score Estimation. Journal of
Business and Economic Statistics, 40(4): 1433-1445.
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings


def verify_ocbps_conditions(
    result: Dict[str, Any],
    X: np.ndarray,
    treat: np.ndarray,
    outcome: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Check observable necessary conditions for optimal CBPS validity.

    Verifies four empirically testable conditions:
    1. **Identification (dimension)**: m1 + m2 + 1 >= k, where m1 and m2 are
       the numbers of propensity score and outcome moment conditions, and k is
       the covariate dimension. This ensures the system is not under-identified.
    2. **Balance achieved**: Weighted correlations between covariates and
       treatment are approximately zero after weighting.
    3. **J-test (overidentification)**: Hansen's J-statistic should not reject
       (p > 0.05), indicating moment conditions are compatible.
    4. **Overlap (positivity)**: Propensity scores are bounded away from 0/1.

    Parameters
    ----------
    result : dict
        Output from a CBPS fit. Expected keys:
        - 'weights' or 'w': np.ndarray of estimated weights
        - 'J' or 'j_stat': float, J-statistic (optional)
        - 'J_pval' or 'j_pval': float, J-test p-value (optional)
        - 'ps' or 'propensity_scores': np.ndarray (optional)
        - 'n_moment_conditions' or 'n_moments': int (optional)
    X : np.ndarray, shape (n, k)
        Covariate matrix.
    treat : np.ndarray, shape (n,)
        Binary treatment indicator.
    outcome : np.ndarray, shape (n,), optional
        Outcome variable (used for enhanced diagnostics if available).

    Returns
    -------
    dict with:
        - identification_ok : bool
            True if dimension condition m1 + m2 + 1 >= k is satisfied.
        - balance_achieved : bool
            True if max |weighted correlation| < 0.1.
        - j_test_result : dict or None
            {'statistic': float, 'p_value': float, 'reject': bool} or None
            if J-test info not available.
        - overlap_ok : bool
            True if propensity scores are in [0.02, 0.98].
        - all_conditions_met : bool
            True if all verifiable conditions pass.
        - warnings : list of str
            Descriptions of any failed conditions.
    """
    X = np.asarray(X, dtype=float)
    treat = np.asarray(treat, dtype=float).ravel()
    n, k = X.shape
    warn_list = []

    # --- Extract weights ---
    weights = _extract_key(result, ['weights', 'w'])
    if weights is None:
        raise ValueError(
            "Result dict must contain 'weights' or 'w' key with "
            "estimated CBPS weights."
        )
    weights = np.asarray(weights, dtype=float).ravel()

    # --- 1. Identification (dimension) condition ---
    # m1 = number of propensity score moment conditions (at least k for score equations)
    # m2 = number of balance conditions
    # For standard CBPS: m1 = k (score), m2 = k (balance), total = 2k >= k always
    # For 'just-identified' CBPS: m1 + m2 = k, so m1+m2+1 = k+1 >= k
    n_moments = _extract_key(result, ['n_moment_conditions', 'n_moments'])
    if n_moments is not None:
        identification_ok = int(n_moments) + 1 >= k
    else:
        # Default: standard over-identified CBPS has 2k moments >= k
        identification_ok = True

    if not identification_ok:
        warn_list.append(
            f"Identification condition violated: number of moment conditions "
            f"({n_moments}) + 1 < k ({k}). The system may be under-identified."
        )

    # --- 2. Balance check (weighted correlation ≈ 0) ---
    # Compute weighted correlation between each covariate and treatment
    w_norm = weights / np.sum(weights) * n
    max_abs_corr = 0.0
    for j in range(k):
        xj = X[:, j]
        # Weighted means
        wx_mean = np.sum(w_norm * xj) / np.sum(w_norm)
        wt_mean = np.sum(w_norm * treat) / np.sum(w_norm)
        # Weighted correlation
        cov_num = np.sum(w_norm * (xj - wx_mean) * (treat - wt_mean))
        var_x = np.sum(w_norm * (xj - wx_mean) ** 2)
        var_t = np.sum(w_norm * (treat - wt_mean) ** 2)
        denom = np.sqrt(var_x * var_t) if (var_x > 0 and var_t > 0) else 1.0
        corr = abs(cov_num / denom) if denom > 0 else 0.0
        max_abs_corr = max(max_abs_corr, corr)

    balance_achieved = bool(max_abs_corr < 0.1)
    if not balance_achieved:
        warn_list.append(
            f"Balance not achieved: max |weighted correlation| = "
            f"{max_abs_corr:.4f} >= 0.1. Consider increasing the number "
            f"of moment conditions or using over-identified CBPS."
        )

    # --- 3. J-test (overidentification) ---
    j_stat = _extract_key(result, ['J', 'j_stat', 'j_statistic'])
    j_pval = _extract_key(result, ['J_pval', 'j_pval', 'j_p_value'])
    if j_stat is not None and j_pval is not None:
        j_test_result = {
            "statistic": float(j_stat),
            "p_value": float(j_pval),
            "reject": float(j_pval) < 0.05,
        }
        if j_test_result["reject"]:
            warn_list.append(
                f"J-test rejects (p={float(j_pval):.4g}): overidentifying "
                f"restrictions may be incompatible. This suggests potential "
                f"model misspecification."
            )
    else:
        j_test_result = None

    # --- 4. Overlap (positivity) ---
    ps = _extract_key(result, ['ps', 'propensity_scores', 'fitted', 'fitted_values'])
    if ps is not None:
        ps = np.asarray(ps, dtype=float).ravel()
        ps_min, ps_max = float(np.min(ps)), float(np.max(ps))
        overlap_ok = ps_min >= 0.02 and ps_max <= 0.98
        if not overlap_ok:
            warn_list.append(
                f"Overlap violation: propensity scores range [{ps_min:.4f}, "
                f"{ps_max:.4f}]. Extreme scores suggest positivity violation. "
                f"Consider trimming observations with extreme scores."
            )
    else:
        # Without propensity scores, check via weights (extreme weights ↔ poor overlap)
        w_cv = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
        overlap_ok = w_cv < 3.0  # Heuristic: CV > 3 indicates extreme weights
        if not overlap_ok:
            warn_list.append(
                f"Potential overlap violation inferred from weight variability "
                f"(CV={w_cv:.2f} > 3.0). Consider checking propensity score "
                f"distributions directly."
            )

    # --- Aggregate ---
    all_ok = bool(identification_ok and balance_achieved and overlap_ok)
    if j_test_result is not None:
        all_ok = bool(all_ok and (not j_test_result["reject"]))

    if warn_list:
        for w in warn_list:
            warnings.warn(w, UserWarning, stacklevel=2)

    return {
        "identification_ok": identification_ok,
        "balance_achieved": balance_achieved,
        "j_test_result": j_test_result,
        "overlap_ok": overlap_ok,
        "all_conditions_met": all_ok,
        "warnings": warn_list,
    }


def _extract_key(d: dict, keys: list):
    """Extract first matching key from a dict."""
    for k in keys:
        if k in d:
            return d[k]
    return None
