"""Overlap (positivity) assumption diagnostics.

Implements Crump et al. (2009) approach to detecting and handling
limited overlap in propensity score distributions.

NOTE: This is a general causal inference diagnostic tool (Crump et al. 2009),
not a CBPS-specific requirement. It complements CBPS by verifying that
the common support assumption holds.

References
----------
Crump, R.K., Hotz, V.J., Imbens, G.W., and Mitnik, O.A. (2009).
"Dealing with Limited Overlap in Estimation of Average Treatment Effects."
Biometrika 96(1): 187-199.
"""

import numpy as np
from typing import Optional, List


def check_overlap(propensity_scores, treat, alphas=None):
    """Check common support (overlap) assumption.

    Evaluates whether the propensity score distributions of treated and
    control groups have sufficient overlap to support reliable causal
    inference. Implements the trimming approach of Crump et al. (2009).

    Parameters
    ----------
    propensity_scores : np.ndarray
        Estimated propensity scores, shape (n,).
    treat : np.ndarray
        Binary treatment indicator, shape (n,).
    alphas : list of float, optional
        Trimming thresholds to evaluate. Default: [0.05, 0.10, 0.15, 0.20].
        Units with propensity scores outside [alpha, 1-alpha] are trimmed.

    Returns
    -------
    dict with:
        - ps_range: (min, max) of propensity scores
        - ps_range_treated: (min, max) for treated group
        - ps_range_control: (min, max) for control group
        - overlap_region: (max of mins, min of maxes) — common support bounds
        - n_outside_overlap: count outside common support
        - trimming_analysis: dict of {alpha: {n_retained, pct_retained}}
        - recommended_alpha: suggested trimming threshold (or None)
        - violation_detected: bool (True if serious overlap violation)
        - warning_message: str or None

    Notes
    -----
    This is a general causal inference diagnostic (Crump et al. 2009),
    not a CBPS-specific requirement. It complements CBPS estimation by
    verifying the positivity/overlap assumption that underpins all IPW
    estimators.

    A violation is detected when:
    - The overlap region is empty (max of mins > min of maxes), OR
    - More than 20% of observations fall outside common support.

    References
    ----------
    Crump, R.K., Hotz, V.J., Imbens, G.W., and Mitnik, O.A. (2009).
    "Dealing with Limited Overlap in Estimation of Average Treatment Effects."
    Biometrika 96(1): 187-199.
    """
    propensity_scores = np.asarray(propensity_scores, dtype=float).ravel()
    treat = np.asarray(treat).ravel()

    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20]

    n = len(propensity_scores)

    # Input validation
    if n == 0:
        return {
            'ps_range': (np.nan, np.nan),
            'ps_range_treated': (np.nan, np.nan),
            'ps_range_control': (np.nan, np.nan),
            'overlap_region': (np.nan, np.nan),
            'n_outside_overlap': 0,
            'trimming_analysis': {},
            'recommended_alpha': None,
            'violation_detected': True,
            'warning_message': 'SEVERE: No observations provided.',
        }

    if len(treat) != n:
        raise ValueError(
            f"Length mismatch: propensity_scores has {n} elements, "
            f"treat has {len(treat)} elements."
        )

    # Validate propensity scores are in [0, 1]
    ps_min_val = float(np.min(propensity_scores))
    ps_max_val = float(np.max(propensity_scores))
    if ps_min_val < 0.0 or ps_max_val > 1.0:
        import warnings
        warnings.warn(
            f"Propensity scores should be in [0, 1]. "
            f"Found range [{ps_min_val:.4f}, {ps_max_val:.4f}]. "
            f"Values outside [0, 1] are not valid probabilities and may "
            f"produce misleading overlap diagnostics.",
            UserWarning,
            stacklevel=2
        )

    # Identify groups
    treated_mask = treat == 1
    control_mask = treat == 0

    ps_treated = propensity_scores[treated_mask]
    ps_control = propensity_scores[control_mask]

    # Check for empty groups
    if len(ps_treated) == 0 or len(ps_control) == 0:
        empty_group = 'treated' if len(ps_treated) == 0 else 'control'
        return {
            'ps_range': (ps_min_val, ps_max_val),
            'ps_range_treated': (float(np.min(ps_treated)), float(np.max(ps_treated))) if len(ps_treated) > 0 else (np.nan, np.nan),
            'ps_range_control': (float(np.min(ps_control)), float(np.max(ps_control))) if len(ps_control) > 0 else (np.nan, np.nan),
            'overlap_region': (np.nan, np.nan),
            'n_outside_overlap': n,
            'trimming_analysis': {alpha: {'n_retained': 0, 'pct_retained': 0.0} for alpha in sorted(alphas)},
            'recommended_alpha': None,
            'violation_detected': True,
            'warning_message': (
                f"SEVERE: The {empty_group} group has no observations. "
                f"Overlap assessment requires both treated and control units."
            ),
        }

    # Propensity score ranges
    ps_range = (ps_min_val, ps_max_val)

    ps_range_treated = (float(np.min(ps_treated)), float(np.max(ps_treated)))
    ps_range_control = (float(np.min(ps_control)), float(np.max(ps_control)))

    # Common support region: [max of mins, min of maxes]
    overlap_lower = max(ps_range_treated[0], ps_range_control[0])
    overlap_upper = min(ps_range_treated[1], ps_range_control[1])
    overlap_region = (float(overlap_lower), float(overlap_upper))

    # Count observations outside common support
    outside_mask = (propensity_scores < overlap_lower) | (propensity_scores > overlap_upper)
    n_outside_overlap = int(np.sum(outside_mask))

    # Trimming analysis (Crump et al. 2009 approach)
    trimming_analysis = {}
    for alpha in sorted(alphas):
        retained_mask = (propensity_scores >= alpha) & (propensity_scores <= 1 - alpha)
        n_retained = int(np.sum(retained_mask))
        pct_retained = float(n_retained / n * 100) if n > 0 else 0.0
        trimming_analysis[alpha] = {
            'n_retained': n_retained,
            'pct_retained': pct_retained,
        }

    # Recommended alpha: smallest alpha that retains >= 90% of sample
    recommended_alpha = None
    for alpha in sorted(alphas):
        if trimming_analysis[alpha]['pct_retained'] >= 90.0:
            recommended_alpha = alpha
            break

    # Violation detection
    violation_detected = False
    warning_message = None

    if overlap_lower > overlap_upper:
        # No overlap at all
        violation_detected = True
        warning_message = (
            "SEVERE: No common support detected. The propensity score "
            "distributions of treated and control groups do not overlap. "
            "Causal effect estimation is unreliable."
        )
    elif n > 0 and (n_outside_overlap / n) > 0.20:
        violation_detected = True
        warning_message = (
            f"WARNING: {n_outside_overlap} observations ({n_outside_overlap/n*100:.1f}%) "
            f"fall outside the common support region [{overlap_lower:.3f}, {overlap_upper:.3f}]. "
            f"Consider trimming observations with extreme propensity scores."
        )

    return {
        'ps_range': ps_range,
        'ps_range_treated': ps_range_treated,
        'ps_range_control': ps_range_control,
        'overlap_region': overlap_region,
        'n_outside_overlap': n_outside_overlap,
        'trimming_analysis': trimming_analysis,
        'recommended_alpha': recommended_alpha,
        'violation_detected': violation_detected,
        'warning_message': warning_message,
    }
