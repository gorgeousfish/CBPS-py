"""
Covariate Balance for Marginal Structural Models
=================================================

Balance diagnostics for marginal structural models estimated via CBMSM.
These diagnostics assess whether CBMSM weights achieve covariate balance
across treatment history groups in longitudinal settings with time-varying
confounding.

The balance_cbmsm function computes weighted covariate means within each
treatment trajectory pattern (e.g., "0+1+1" for control-treated-treated),
comparing CBMSM weights against standard GLM-based inverse probability weights.

References
----------
Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. Journal of the American Statistical
Association, 110(511), 1013-1023.
"""
import numpy as np
from typing import Dict, Any


def balance_cbmsm(cbmsm_obj: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute covariate balance across treatment history groups.

    Calculates weighted covariate means within each treatment trajectory
    pattern, enabling assessment of balance in marginal structural models.
    Treatment histories are represented as concatenated strings (e.g.,
    "0+1+1" for a unit that was untreated in period 1, then treated in
    periods 2 and 3).

    Parameters
    ----------
    cbmsm_obj : dict
        Fitted CBMSM result containing:

        - **y** : ndarray of shape (n_obs,) - Binary treatment at each observation
        - **x** : ndarray of shape (n_obs, k) - Covariates with intercept
        - **weights** : ndarray - CBMSM weights (unit-level)
        - **id** : ndarray - Unit identifiers
        - **time** : ndarray - Time period indicators
        - **glm_weights** : ndarray - Baseline GLM inverse probability weights

    Returns
    -------
    dict
        Balance statistics with keys:

        - **Balanced** : ndarray of shape (n_covars, 2*n_hist)
          CBMSM-weighted means (first n_hist columns) and standardized
          means (remaining columns) for each treatment history.
        - **Unweighted** : ndarray of shape (n_covars, 2*n_hist)
          GLM-weighted baseline statistics for comparison.
        - **StatBal** : float
          Summary statistic measuring overall imbalance.

    Notes
    -----
    Balance is assessed using first-period covariate values, following
    standard practice for evaluating baseline balance in MSM settings.

    See Also
    --------
    plot_cbmsm : Visualization of CBMSM balance diagnostics.
    """
    # Extract panel data components
    y = cbmsm_obj['y']
    x = cbmsm_obj['x']
    w = cbmsm_obj.get('weights', cbmsm_obj.get('w'))
    glm_w = cbmsm_obj.get('glm_weights', cbmsm_obj.get('glm.w', cbmsm_obj.get('glm_w')))
    ids = cbmsm_obj['id']
    times = cbmsm_obj['time']

    unique_ids = np.sort(np.unique(ids))
    unique_times = np.sort(np.unique(times))
    n_units = len(unique_ids)
    n_periods = len(unique_times)

    # Reconstruct treatment history matrix (n_units x n_periods)
    treat_hist = np.full((n_units, n_periods), np.nan)
    for i, unit_id in enumerate(unique_ids):
        for j, period in enumerate(unique_times):
            mask = (ids == unit_id) & (times == period)
            if mask.any():
                treat_hist[i, j] = y[mask][0]

    # Create treatment history factor (e.g., "0+1+1" for trajectory)
    treat_hist_fac = []
    for i in range(n_units):
        hist_str = '+'.join(
            [str(int(t)) if not np.isnan(t) else 'NA' for t in treat_hist[i, :]]
        )
        treat_hist_fac.append(hist_str)
    treat_hist_fac = np.array(treat_hist_fac)

    unique_hist = np.unique(treat_hist_fac)
    n_hist = len(unique_hist)

    # Initialize balance matrices
    n_covars = x.shape[1] - 1  # Exclude intercept column
    bal = np.zeros((n_covars, 2 * n_hist))
    baseline = np.zeros((n_covars, 2 * n_hist))

    # Use first-period covariates for balance assessment
    first_period = unique_times[0]
    first_period_mask = (times == first_period)
    x_first = x[first_period_mask, :]
    ids_first = ids[first_period_mask]

    # Map unit-level weights to first-period observations
    if len(w) == n_units:
        id_to_w = dict(zip(unique_ids, w))
        id_to_glm_w = dict(zip(unique_ids, glm_w))
        w_first = np.array([id_to_w[uid] for uid in ids_first])
        glm_w_first = np.array([id_to_glm_w[uid] for uid in ids_first])
    else:
        w_first = w[first_period_mask]
        glm_w_first = glm_w[first_period_mask]

    # Compute balance statistics for each treatment history group
    for i, hist in enumerate(unique_hist):
        hist_mask_units = (treat_hist_fac == hist)
        hist_mask_obs = np.array(
            [uid in unique_ids[hist_mask_units] for uid in ids_first]
        )

        # Iterate over covariates (skip intercept at index 0)
        for j in range(1, x.shape[1]):
            idx = j - 1

            if hist_mask_obs.sum() > 0:
                # CBMSM-weighted mean
                bal[idx, i] = (
                    np.sum(hist_mask_obs * x_first[:, j] * w_first) /
                    np.sum(w_first * hist_mask_obs)
                )
                # Standardized mean (divided by weighted std)
                bal[idx, i + n_hist] = bal[idx, i] / np.std(w_first * x_first[:, j])

                # GLM baseline weighted mean
                baseline[idx, i] = (
                    np.sum(hist_mask_obs * x_first[:, j] * glm_w_first) /
                    np.sum(glm_w_first * hist_mask_obs)
                )
                baseline[idx, i + n_hist] = (
                    baseline[idx, i] / np.std(glm_w_first * x_first[:, j])
                )

    # Handle numerical edge cases
    bal[np.isnan(bal)] = 0
    baseline[np.isnan(baseline)] = 0

    # Summary balance statistic
    statbal = np.sum((bal - bal[:, 0:1]) * (bal != 0) ** 2)

    return {
        'Balanced': bal,
        'Unweighted': baseline,
        'StatBal': statbal
    }
