"""Rank selection diagnostics for CBMSM covariate matrices.

WARNING: Automatic rank selection methods (energy ratio, information criteria)
go beyond Imai & Ratkovic (2015) specification. These tools are provided for
sensitivity analysis only. The default fixed threshold (1e-4) should be used
for published analyses unless justified.

References
----------
Imai, K. & Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. JASA, 110(511), 1013-1023.
"""

import numpy as np
from typing import Any, Dict, List, Optional


def diagnose_rank_selection(
    X_mat: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compare rank under different SVD thresholds.

    Helps users assess sensitivity of CBMSM results to rank choice.
    This is a diagnostic tool only; it does NOT change the default behavior
    of the CBMSM estimator.

    Parameters
    ----------
    X_mat : np.ndarray, shape (n, k)
        Covariate matrix (mean-centered recommended).
    thresholds : list of float, optional
        SVD thresholds to compare. Default: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2].

    Returns
    -------
    dict with keys:
        - 'singular_values': np.ndarray, all singular values (descending)
        - 'total_columns': int, original number of columns k
        - 'ranks_by_threshold': dict mapping threshold -> retained rank
        - 'energy_by_rank': np.ndarray, cumulative variance explained
          at each rank (cumsum(s**2) / sum(s**2))
        - 'recommended_action': str, guidance for the user

    Examples
    --------
    >>> import numpy as np
    >>> from cbps.msm.rank_diagnostics import diagnose_rank_selection
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 5))
    >>> result = diagnose_rank_selection(X)
    >>> result['total_columns']
    5
    """
    if thresholds is None:
        thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    X = np.asarray(X_mat, dtype=np.float64)
    n, k = X.shape

    # Compute SVD
    if k == 0 or n == 0:
        return {
            "singular_values": np.array([], dtype=float),
            "total_columns": k,
            "ranks_by_threshold": {t: 0 for t in thresholds},
            "energy_by_rank": np.array([], dtype=float),
            "recommended_action": "No covariates provided.",
        }

    _U, s, _Vt = np.linalg.svd(X, full_matrices=False)

    # Ranks by threshold (number of singular values exceeding each threshold)
    ranks_by_threshold = {}
    for t in sorted(thresholds):
        ranks_by_threshold[t] = int(np.sum(s > t))

    # Cumulative energy (variance explained)
    s_sq = s ** 2
    total_energy = s_sq.sum()
    if total_energy > 0:
        energy_by_rank = np.cumsum(s_sq) / total_energy
    else:
        energy_by_rank = np.zeros_like(s_sq)

    # Generate recommendation
    default_rank = int(np.sum(s > 1e-4))
    if default_rank == k:
        recommended_action = (
            "All singular values exceed 1e-4. The matrix appears full rank; "
            "no dimension reduction occurs with the default threshold."
        )
    elif default_rank == 0:
        recommended_action = (
            "No singular values exceed 1e-4. Consider using a smaller threshold "
            "or checking for degenerate covariates."
        )
    else:
        energy_at_default = energy_by_rank[default_rank - 1] if default_rank > 0 else 0.0
        recommended_action = (
            f"Default threshold (1e-4) retains {default_rank}/{k} components "
            f"explaining {energy_at_default:.4f} of total variance. "
            f"Verify that CBMSM estimates are stable across nearby thresholds."
        )

    return {
        "singular_values": s,
        "total_columns": k,
        "ranks_by_threshold": ranks_by_threshold,
        "energy_by_rank": energy_by_rank,
        "recommended_action": recommended_action,
    }
