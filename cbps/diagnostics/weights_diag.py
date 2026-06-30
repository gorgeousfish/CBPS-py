"""
Weight Quality Diagnostics
===========================

Comprehensive diagnostics for inverse probability weights produced by CBPS
estimation, including effective sample size (ESS), weight distribution
summaries, and extreme value detection.

The Kish (1965) effective sample size is the primary metric for assessing
whether extreme weights are degrading estimation precision.

References
----------
Kish, L. (1965). Survey Sampling. Wiley, New York.

Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
"""

import numpy as np
from typing import Optional


def weight_diagnostics(weights, treat=None):
    """Compute comprehensive weight quality diagnostics.

    Based on Kish (1965) effective sample size and standard
    IPW weight quality indicators.

    Parameters
    ----------
    weights : np.ndarray
        IPW weights from CBPS estimation. Should be non-negative for
        meaningful ESS interpretation. If negative weights are present
        (e.g., from ATT balance conditions), a warning is issued and
        ESS is computed on absolute values.
    treat : np.ndarray, optional
        Treatment indicator for group-specific diagnostics.

    Returns
    -------
    dict with:
        - ess: Kish effective sample size
        - ess_ratio: ESS / n (closer to 1 = better)
        - weight_max: maximum absolute weight
        - weight_min: minimum absolute weight (among nonzero)
        - weight_ratio: max/min ratio (of absolute values)
        - cv: coefficient of variation of weights
        - n_extreme: count of weights with abs(w) > 10*median(abs(w))
        - n_negative: count of negative weights (0 if all non-negative)
        - warning_level: 'ok'/'caution'/'severe'
        - group_diagnostics: dict per treatment group (if treat provided)

    Notes
    -----
    Warning thresholds (based on Kish 1965, Chapter 11):
        - ESS/n < 0.5 → 'caution'
        - ESS/n < 0.2 → 'severe'

    The ESS formula is: ESS = (sum(w))^2 / sum(w^2)
    For uniform weights, ESS = n. For highly variable weights, ESS << n.

    When negative weights are present, the Kish ESS formula does not have
    its standard interpretation. In this case, ESS is computed on abs(w) and
    a warning is included in the result.

    References
    ----------
    Kish, L. (1965). Survey Sampling. Wiley, New York. Chapter 11.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    n = len(weights)

    # Handle degenerate cases
    if n == 0:
        return {
            'ess': 0.0,
            'ess_ratio': 0.0,
            'weight_max': np.nan,
            'weight_min': np.nan,
            'weight_ratio': np.nan,
            'cv': np.nan,
            'n_extreme': 0,
            'n_negative': 0,
            'warning_level': 'severe',
            'group_diagnostics': None,
        }

    # Detect negative weights
    n_negative = int(np.sum(weights < 0))
    has_negative = n_negative > 0

    # For ESS computation: use absolute values when negative weights present
    # Kish ESS is only interpretable for non-negative weights
    if has_negative:
        import warnings
        warnings.warn(
            f"Kish ESS is defined for non-negative weights. "
            f"{n_negative} negative weight(s) detected; "
            f"ESS is computed on |weights| as an approximation. "
            f"Consider using only the final IPW weights (not balance weights) "
            f"for this diagnostic.",
            UserWarning,
            stacklevel=2
        )
        w_for_ess = np.abs(weights)
    else:
        w_for_ess = weights

    sum_w = np.sum(w_for_ess)
    sum_w2 = np.sum(w_for_ess ** 2)

    # ESS computation (Kish 1965)
    if sum_w2 == 0:
        # All weights are zero
        ess = 0.0
        ess_ratio = 0.0
    else:
        ess = (sum_w ** 2) / sum_w2
        ess_ratio = ess / n

    # Weight range based on absolute values (captures extreme negative weights)
    abs_weights = np.abs(weights)
    nonzero_mask = abs_weights > 0
    if np.any(nonzero_mask):
        weight_min = float(np.min(abs_weights[nonzero_mask]))
        weight_max = float(np.max(abs_weights[nonzero_mask]))
    else:
        weight_min = 0.0
        weight_max = 0.0

    # Max/min ratio
    if weight_min > 0:
        weight_ratio = weight_max / weight_min
    else:
        weight_ratio = np.inf if weight_max > 0 else np.nan

    # Coefficient of variation (on absolute values when negative present)
    w_for_cv = w_for_ess
    w_mean = np.mean(w_for_cv)
    if w_mean > 0:
        cv = float(np.std(w_for_cv) / w_mean)
    else:
        cv = np.nan

    # Extreme weight count: abs(w) > 10 * median(abs(w))
    median_abs_w = np.median(abs_weights)
    if median_abs_w > 0:
        n_extreme = int(np.sum(abs_weights > 10 * median_abs_w))
    else:
        # If median is 0, count all nonzero weights as extreme
        n_extreme = int(np.sum(abs_weights > 0))

    # Warning level
    if ess_ratio < 0.2:
        warning_level = 'severe'
    elif ess_ratio < 0.5:
        warning_level = 'caution'
    else:
        warning_level = 'ok'

    result = {
        'ess': float(ess),
        'ess_ratio': float(ess_ratio),
        'weight_max': float(weight_max),
        'weight_min': float(weight_min),
        'weight_ratio': float(weight_ratio) if np.isfinite(weight_ratio) else weight_ratio,
        'cv': float(cv) if np.isfinite(cv) else cv,
        'n_extreme': n_extreme,
        'n_negative': n_negative,
        'warning_level': warning_level,
        'group_diagnostics': None,
    }

    # Group-specific diagnostics
    if treat is not None:
        treat = np.asarray(treat).ravel()
        if len(treat) == n:
            group_diag = {}
            for level in np.unique(treat):
                mask = treat == level
                g_weights = weights[mask]
                g_n = len(g_weights)

                # Use absolute values for ESS when negatives present
                g_abs_w = np.abs(g_weights)
                g_sum_w = np.sum(g_abs_w)
                g_sum_w2 = np.sum(g_abs_w ** 2)

                if g_sum_w2 > 0:
                    g_ess = (g_sum_w ** 2) / g_sum_w2
                else:
                    g_ess = 0.0

                group_diag[level] = {
                    'n': g_n,
                    'ess': float(g_ess),
                    'ess_ratio': float(g_ess / g_n) if g_n > 0 else 0.0,
                    'weight_mean': float(np.mean(g_weights)),
                    'weight_max': float(np.max(g_abs_w)) if g_n > 0 else np.nan,
                    'n_negative': int(np.sum(g_weights < 0)),
                }
            result['group_diagnostics'] = group_diag

    return result
