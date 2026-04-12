"""
Covariate Balancing Propensity Score for Marginal Structural Models (CBMSM).

This module implements the Covariate Balancing Propensity Score (CBPS) methodology
for marginal structural models, as described in Imai and Ratkovic (2015). The
approach estimates inverse probability weights that simultaneously maximize
covariate balance and treatment assignment prediction in longitudinal settings.

Key Features
------------
- **Time-invariant coefficients**: Single set of propensity score parameters
  shared across all time periods (default).
- **Time-varying coefficients**: Period-specific parameter estimation when
  treatment effects on covariates vary over time.
- **Stabilized weights**: Variance reduction through numerator modeling as
  P(T)/P(T|X) rather than 1/P(T|X).
- **Low-rank variance approximation**: Computational efficiency for settings
  with many time periods via diagonal covariance assumption.
- **Hadamard matrix decomposition**: Orthogonal representation of covariate
  balancing conditions based on 2^J factorial design framework.

Algorithm Overview
------------------
The estimator solves a Generalized Method of Moments (GMM) problem where
moment conditions are derived from the covariate balancing property of MSM
weights. At each time period j, the weights balance covariates across all
possible current and future treatment sequences, conditional on past treatment
history.

The number of moment conditions grows exponentially with the number of time
periods. For K covariates and J periods, each period j contributes
K × (2^J - 2^{j-1}) conditions, yielding a total that scales as O(K × J × 2^J).
The low-rank approximation addresses this by assuming zero correlation across
balance conditions.

References
----------
Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. Journal of the American Statistical
Association, 110(511), 1013-1023. https://doi.org/10.1080/01621459.2014.956872
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special

from cbps.utils.formula import parse_formula
from cbps.core.cbps_binary import _r_ginv
from cbps.core.cbps_binary import cbps_binary_fit
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
import re
import warnings


PROBS_TRIM = 1e-4  # CBMSM probability clipping threshold


class CBMSMSummary:
    """Summary object for CBMSMResults.

    Returned by :meth:`CBMSMResults.summary`. Provides a structured
    representation of CBMSM estimation results that can be printed
    via ``print()`` or ``str()``.

    Attributes
    ----------
    n_obs : int
        Total number of observations (n_units × n_periods).
    n_periods : int
        Number of time periods.
    n_units : int
        Number of unique units.
    time_vary : bool
        Whether period-specific coefficients were estimated.
    converged : bool
        Whether the GMM optimization converged.
    J : float
        Hansen J-statistic.
    weights : np.ndarray
        Propensity score weights P(T|X).
    fitted_values : np.ndarray
        Stabilized MSM weights P(T)/P(T|X).
    coefficients : np.ndarray
        Estimated propensity score model coefficients.
    """

    def __init__(
        self,
        weights: np.ndarray,
        fitted_values: np.ndarray,
        coefficients: np.ndarray,
        n_periods: int,
        n_units: int,
        time_vary: bool,
        converged: bool,
        J: float,
    ):
        self.weights = weights
        self.fitted_values = fitted_values
        self.coefficients = coefficients
        self.n_periods = n_periods
        self.n_units = n_units
        self.time_vary = time_vary
        self.converged = converged
        self.J = J

    def __str__(self) -> str:
        """Return formatted summary text.

        The output is identical to the legacy ``CBMSMResults.summary()``
        string for backward compatibility.
        """
        output = "\n" + "="*70 + "\n"
        output += "CBMSM (Covariate Balancing Propensity Score for MSM) Results\n"
        output += "="*70 + "\n\n"

        # Basic information
        output += f"Number of observations: {len(self.weights)}\n"
        output += f"Number of time periods: {self.n_periods}\n"
        output += f"Number of units: {self.n_units}\n"
        output += f"Time-varying treatment model: {'Yes' if self.time_vary else 'No'}\n"
        output += f"Convergence: {'Yes' if self.converged else 'No'}\n"
        output += f"J-statistic: {self.J:.6f}\n\n"

        # Propensity score summary P(T|X)
        output += "Propensity Scores P(T|X) Summary:\n"
        output += "-"*70 + "\n"
        weights = np.array(self.weights)
        output += f"  Min:    {np.min(weights):.6f}\n"
        output += f"  1Q:     {np.percentile(weights, 25):.6f}\n"
        output += f"  Median: {np.median(weights):.6f}\n"
        output += f"  Mean:   {np.mean(weights):.6f}\n"
        output += f"  3Q:     {np.percentile(weights, 75):.6f}\n"
        output += f"  Max:    {np.max(weights):.6f}\n\n"

        # MSM weight summary (stabilized weights P(T)/P(T|X))
        if self.fitted_values is not None:
            output += "MSM Weights (Stabilized) Summary:\n"
            output += "-"*70 + "\n"
            msm_weights = np.array(self.fitted_values)
            output += f"  Min:    {np.min(msm_weights):.6f}\n"
            output += f"  1Q:     {np.percentile(msm_weights, 25):.6f}\n"
            output += f"  Median: {np.median(msm_weights):.6f}\n"
            output += f"  Mean:   {np.mean(msm_weights):.6f}\n"
            output += f"  3Q:     {np.percentile(msm_weights, 75):.6f}\n"
            output += f"  Max:    {np.max(msm_weights):.6f}\n\n"

        # Coefficient information
        if self.coefficients is not None:
            output += "Coefficients:\n"
            output += "-"*70 + "\n"
            coefs = np.array(self.coefficients)
            if coefs.ndim == 1:
                # time_vary=False: single column of coefficients
                for i, coef in enumerate(coefs):
                    output += f"  beta[{i}]: {coef:.6f}\n"
            else:
                # time_vary=True: coefficient matrix (k, n_periods)
                output += f"  Shape: {coefs.shape[0]} parameters × {coefs.shape[1]} periods\n"
                for i in range(min(5, coefs.shape[0])):  # Show first 5 parameters
                    output += f"  beta[{i}]: "
                    for j in range(coefs.shape[1]):
                        output += f"{coefs[i,j]:8.4f} "
                    output += "\n"
                if coefs.shape[0] > 5:
                    output += f"  ... ({coefs.shape[0] - 5} more parameters)\n"
            output += "\n"

        # Warning message
        if not self.converged:
            output += "WARNING: Optimization did not converge!\n"
            output += "   Results may be unreliable. Consider:\n"
            output += "   - Increasing the number of iterations\n"
            output += "   - Checking data quality and balance\n"
            output += "   - Trying different starting values\n"
            output += "   - Examining the balance diagnostics\n\n"

        output += "="*70 + "\n"
        return output

    def __repr__(self) -> str:
        return f"CBMSMSummary(n_units={self.n_units}, converged={self.converged})"


@dataclass
class CBMSMResults:
    """Container for CBMSM estimation results.

    Stores inverse probability weights, model diagnostics, and estimation
    metadata from fitting a Covariate Balancing Propensity Score model for
    Marginal Structural Models.

    Attributes
    ----------
    weights : np.ndarray, shape (n_units,)
        Inverse probability weights 1/P(T|X), computed as the product of
        per-period weights across all time periods for each unit.
    fitted_values : np.ndarray, shape (n_units,)
        Stabilized MSM weights P(T)/P(T|X), suitable for use in weighted
        outcome regression models.
    glm_g : np.ndarray
        Sample moment conditions under standard logistic regression.
    msm_g : np.ndarray
        Sample moment conditions under the fitted CBMSM model.
    glm_weights : np.ndarray, shape (n_units,)
        Stabilized weights from standard logistic regression for comparison.
    id : np.ndarray
        Unit identifiers corresponding to each observation.
    treat_hist : np.ndarray, shape (n_units, n_periods)
        Binary treatment history matrix. Entry (i, j) indicates whether
        unit i received treatment at time period j.
    treat_cum : np.ndarray, shape (n_units,)
        Cumulative treatment count for each unit across all periods.
    coefficients : np.ndarray
        Estimated propensity score model coefficients. Shape is (k,) when
        ``time_vary=False`` or (k, n_periods) when ``time_vary=True``.
    n_periods : int
        Number of time periods J in the panel.
    n_units : int
        Number of unique units (individuals) N in the sample.
    time_vary : bool
        Whether period-specific coefficients were estimated.
    converged : bool
        Whether the GMM optimization converged successfully.
    J : float
        Hansen J-statistic (normalized GMM objective value), useful for
        assessing model specification.
    var : np.ndarray
        Inverse of the GMM weighting matrix used in optimization. Note
        that this is NOT the variance-covariance matrix of coefficient
        estimates; it has dimension (n_moments, n_moments).
    call : str
        String representation of the function call for reproducibility.
    formula : str
        Model formula specification (None for matrix interface).
    y : np.ndarray
        Treatment vector (observation-level, sorted by time then id).
    x : np.ndarray
        Original covariate matrix from formula parsing (observation-level).
        Note: The actual GMM estimation uses an SVD-processed version with
        intercept; this attribute stores the pre-processed input.
    time : np.ndarray
        Time period indicators (observation-level, sorted).
    model : Any
        Model frame containing all variables.
    data : Any
        Original input DataFrame.

    See Also
    --------
    CBMSM : Formula interface for CBMSM estimation.
    cbmsm_fit : Matrix interface for CBMSM estimation.

    Notes
    -----
    The ``fitted_values`` attribute contains the recommended weights for
    outcome model estimation. These are stabilized weights that incorporate
    the marginal treatment probability P(T) in the numerator, reducing
    variance compared to unstabilized weights (Robins, Hernan, and Brumback,
    2000).

    The ``var`` attribute stores the inverse weighting matrix from GMM
    optimization, not coefficient standard errors. Computing inference
    for coefficients requires the GMM sandwich variance estimator, which
    is not currently implemented.
    """
    # Core weights (6)
    weights: np.ndarray
    fitted_values: np.ndarray
    glm_g: np.ndarray
    msm_g: np.ndarray
    glm_weights: np.ndarray
    id: np.ndarray

    # Treatment history (2)
    treat_hist: np.ndarray
    treat_cum: np.ndarray

    # Diagnostics (7)
    coefficients: np.ndarray
    n_periods: int
    n_units: int
    time_vary: bool
    converged: bool
    J: float
    var: np.ndarray

    # Metadata (7)
    call: str
    formula: str
    y: np.ndarray
    x: np.ndarray
    time: np.ndarray
    model: Any
    data: Any
    
    def __repr__(self) -> str:
        """Concise repr output for interactive environments."""
        converged_str = "Yes" if self.converged else "No"
        return f"CBMSMResults(n_units={self.n_units}, n_periods={self.n_periods}, converged={converged_str}, J={self.J:.6f})"
    
    def __str__(self) -> str:
        """Complete string output for print calls."""
        output = "\nCall:\n  " + (self.call or "CBMSM()") + "\n\n"
        
        # Basic information
        output += f"Sample size: {self.n_units} units × {self.n_periods} periods = {self.n_units * self.n_periods} observations\n"
        output += f"Time-varying treatment model: {'Yes' if self.time_vary else 'No'}\n"
        output += f"Converged: {'Yes' if self.converged else 'No'}\n"
        
        # Statistics
        output += f"\nModel Statistics:\n"
        output += f"  J-statistic: {self.J:.6f}\n"
        
        # Weight information
        if self.fitted_values is not None:
            output += f"\nMSM Weights (stabilized):\n"
            output += f"  Min: {self.fitted_values.min():.6f}\n"
            output += f"  Max: {self.fitted_values.max():.6f}\n"
            output += f"  Mean: {self.fitted_values.mean():.6f}\n"
        
        if self.weights is not None:
            output += f"\nPropensity Scores P(T|X):\n"
            output += f"  Min: {self.weights.min():.6f}\n"
            output += f"  Max: {self.weights.max():.6f}\n"
            output += f"  Mean: {self.weights.mean():.6f}\n"
        
        # Treatment history
        if self.treat_hist is not None:
            output += f"\nTreatment History: {self.treat_hist.shape[0]} × {self.treat_hist.shape[1]} matrix\n"
        
        # Coefficients
        if self.coefficients is not None:
            output += f"\nCoefficients: {self.coefficients.shape[0]} parameters\n"
        
        return output
    
    def vcov(self) -> np.ndarray:
        """Return the inverse GMM weighting matrix.

        Returns
        -------
        np.ndarray, shape (n_moments, n_moments)
            Inverse of the weighting matrix used in GMM optimization.

        Raises
        ------
        ValueError
            If the weighting matrix was not computed during estimation.

        Warnings
        --------
        This matrix is the inverse of the moment condition covariance, NOT
        the variance-covariance matrix of coefficient estimates. It cannot
        be used directly for computing standard errors or confidence intervals.

        Notes
        -----
        The GMM weighting matrix W is defined in Imai and Ratkovic (2015)
        equation (25). For coefficient inference, the sandwich variance
        estimator would be required, which involves the Jacobian of moment
        conditions with respect to parameters.
        """
        if self.var is None:
            raise ValueError("GMM weighting matrix inverse not computed.")
        return self.var

    def summary(self) -> 'CBMSMSummary':
        """Generate a formatted summary of estimation results.

        Returns
        -------
        CBMSMSummary
            Summary object with ``__str__`` method. Use ``print(result.summary())``
            to display the formatted text.

        Examples
        --------
        >>> result = CBMSM("treat ~ x1 + x2", id="id", time="time", data=df)
        >>> print(result.summary())
        """
        return CBMSMSummary(
            weights=self.weights,
            fitted_values=self.fitted_values,
            coefficients=self.coefficients,
            n_periods=self.n_periods,
            n_units=self.n_units,
            time_vary=self.time_vary,
            converged=self.converged,
            J=self.J,
        )
    
    def balance(self) -> Dict[str, Any]:
        """Compute covariate balance diagnostics across treatment histories.

        Assesses whether the estimated weights successfully balance baseline
        covariates across all distinct treatment history patterns. This is
        the key diagnostic for MSM weight quality.

        Returns
        -------
        dict
            Dictionary containing:

            - ``'Balanced'``: np.ndarray, shape (n_covars, 2*n_patterns)
                Weighted covariate means and standardized means for each
                treatment history pattern.
            - ``'Unweighted'``: np.ndarray, shape (n_covars, 2*n_patterns)
                Corresponding statistics using GLM weights for comparison.
            - ``'StatBal'``: float
                Overall balance statistic (sum of squared deviations).
            - ``'column_names'``: list of str
                Labels in format ``"{pattern}.mean"`` and ``"{pattern}.std.mean"``.
            - ``'row_names'``: list of str
                Covariate names (excluding intercept).

        Notes
        -----
        Balance is computed using first-period covariates only, as these are
        not affected by treatment. Standardized means divide by the weighted
        standard deviation to facilitate comparison across variables with
        different scales.
        """
        # 1. Construct treatment history matrix
        if self.treat_hist is None:
            raise ValueError("CBMSMResults object missing treat_hist matrix")
        
        # 2. Convert treatment history to string factors
        # Each row represents a unit's complete treatment history, joined with "+"
        treat_hist_fac = np.array(['+'.join(map(str, row.astype(int))) for row in self.treat_hist])
        
        # Get unique treatment history patterns
        unique_treat_hist = np.unique(treat_hist_fac)
        n_unique = len(unique_treat_hist)
        
        # Get time periods
        unique_times = np.unique(self.time)
        times = np.sort(unique_times)
        first_time = times[0]
        
        # Get first-period data indices
        first_time_idx = (self.time == first_time)
        X_first = self.x[first_time_idx, :]  # First-period covariate matrix (observation-level)
        
        # 3. Initialize balance matrices
        n_covars = self.x.shape[1] - 1  # Covariates excluding intercept
        bal = np.full((n_covars, n_unique * 2), np.nan)
        baseline = np.full((n_covars, n_unique * 2), np.nan)
        
        # 4. Compute balance statistics for each treatment history and covariate
        # Note: treat_hist_fac is unit-level (N units), needs mapping to observation-level
        unique_ids = np.unique(self.id)
        id_to_treathist = dict(zip(unique_ids, treat_hist_fac))
        
        # For each first-period observation, get its corresponding treatment history
        id_first = self.id[first_time_idx]
        treat_hist_fac_first = np.array([id_to_treathist[uid] for uid in id_first])
        
        for i, th in enumerate(unique_treat_hist):
            # Find first-period observations with this treatment history
            obs_mask = (treat_hist_fac_first == th)
            
            for j in range(1, self.x.shape[1]):  # Skip intercept column (j=0)
                # Get first-period covariate values and weights
                X_col_first = X_first[:, j]
                
                # Expand unit-level weights to observation-level (first period)
                w_unit = self.weights  # unit-level
                id_to_weight = dict(zip(unique_ids, w_unit))
                w_first = np.array([id_to_weight[uid] for uid in id_first])
                
                # GLM weights are also unit-level, need expansion
                glm_w_unit = self.glm_weights  # unit-level
                id_to_glm_weight = dict(zip(unique_ids, glm_w_unit))
                glm_w_first = np.array([id_to_glm_weight[uid] for uid in id_first])
                
                # Compute weighted mean
                numerator = np.sum(obs_mask * X_col_first * w_first)
                denominator = np.sum(w_first * obs_mask)
                bal[j-1, i] = numerator / denominator if denominator > 0 else 0.0
                
                # Standardized mean
                weighted_X_std = np.std(w_first * X_col_first, ddof=1)
                bal[j-1, i + n_unique] = bal[j-1, i] / weighted_X_std if weighted_X_std > 0 else 0.0
                
                # Unweighted (GLM) mean
                numerator_glm = np.sum(obs_mask * X_col_first * glm_w_first)
                denominator_glm = np.sum(glm_w_first * obs_mask)
                baseline[j-1, i] = numerator_glm / denominator_glm if denominator_glm > 0 else 0.0
                
                # Unweighted standardized mean
                glm_weighted_X_std = np.std(glm_w_first * X_col_first, ddof=1)
                baseline[j-1, i + n_unique] = baseline[j-1, i] / glm_weighted_X_std if glm_weighted_X_std > 0 else 0.0
        
        # 5. Set NA values to 0
        bal[np.isnan(bal)] = 0.0
        baseline[np.isnan(baseline)] = 0.0
        
        # 6. Set column names
        cnames = []
        for th in unique_treat_hist:
            cnames.append(f"{th}.mean")
        for th in unique_treat_hist:
            cnames.append(f"{th}.std.mean")
        
        # 7. Set row names (covariate names, excluding intercept)
        if hasattr(self, 'covariate_names') and self.covariate_names is not None:
            rnames = [name for name in self.covariate_names if name != '(Intercept)']
        else:
            rnames = [f"X{i}" for i in range(1, self.x.shape[1])]
        
        # 8. Compute StatBal statistic
        # sum((bal - bal[,1]) * (bal != 0)^2)
        bal_diff = bal - bal[:, 0:1]  # Difference from first column
        statbal = np.sum(bal_diff * (bal != 0)**2)
        
        # 9. Return results
        return {
            'Balanced': bal,
            'Unweighted': baseline,
            'StatBal': statbal,
            'column_names': cnames,
            'row_names': rnames
        }


def _sort_by_time(
    id_arr: np.ndarray,
    time_arr: np.ndarray,
    y: np.ndarray,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort panel data by (time, id) for consistent processing.

    Parameters
    ----------
    id_arr : np.ndarray, shape (n,)
        Unit identifiers.
    time_arr : np.ndarray, shape (n,)
        Time period indicators.
    y : np.ndarray, shape (n,)
        Treatment vector.
    X : np.ndarray, shape (n, k)
        Covariate matrix.

    Returns
    -------
    id_sorted : np.ndarray
        Sorted unit identifiers.
    time_sorted : np.ndarray
        Sorted time indicators.
    y_sorted : np.ndarray
        Sorted treatment vector.
    X_sorted : np.ndarray
        Sorted covariate matrix.
    order : np.ndarray
        Permutation indices for reconstructing original order.
    """
    order = np.lexsort((id_arr, time_arr))
    return id_arr[order], time_arr[order], y[order], X[order, :], order


def _build_treat_hist(
    y_sorted: np.ndarray,
    id_sorted: np.ndarray,
    time_sorted: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the treatment history matrix from sorted panel data.

    Parameters
    ----------
    y_sorted : np.ndarray, shape (n,)
        Treatment vector sorted by (time, id).
    id_sorted : np.ndarray, shape (n,)
        Unit identifiers sorted by (time, id).
    time_sorted : np.ndarray, shape (n,)
        Time indicators sorted by (time, id).

    Returns
    -------
    treat_hist : np.ndarray, shape (N, J)
        Treatment history matrix. Entry (i, j) is the treatment status
        for unit i at time period j.
    name_cands : np.ndarray, shape (N,)
        Sorted unique unit identifiers.
    unique_time : np.ndarray, shape (J,)
        Sorted unique time periods.

    Warns
    -----
    UserWarning
        If the panel is unbalanced (missing id×time combinations detected).
    """
    import warnings
    name_cands = np.array(sorted(pd.unique(id_sorted)))
    unique_time = np.array(sorted(pd.unique(time_sorted)))
    N = len(name_cands)
    J = len(unique_time)
    hist = np.full((N, J), np.nan, dtype=float)  # Initialize with NaN for missing combinations
    # Construct (id, time) -> y lookup table
    key = pd.Series(y_sorted, index=pd.MultiIndex.from_arrays([id_sorted, time_sorted]))
    missing_combinations = []  # Track missing combinations
    for i, name in enumerate(name_cands):
        for j, t in enumerate(unique_time):
            try:
                hist[i, j] = float(key.loc[(name, t)])
            except KeyError:
                # Keep NaN value for missing combinations
                missing_combinations.append((name, t))
                pass  # Keep initial NaN value
    
    # Warn if there are missing combinations (defensive programming)
    if missing_combinations:
        warnings.warn(
            f"Unbalanced panel detected: {len(missing_combinations)} missing id×time combinations. "
            f"This should have been caught by the balanced panel check. "
            f"Setting treat_hist to NaN for missing combinations.\n"
            f"First few missing: {missing_combinations[:5]}",
            UserWarning
        )
    
    return hist, name_cands, unique_time


def _scale_all_columns(X_in: np.ndarray) -> np.ndarray:
    """Standardize matrix columns to zero mean and unit variance.

    Parameters
    ----------
    X_in : np.ndarray, shape (n, k)
        Input matrix.

    Returns
    -------
    np.ndarray, shape (n, k)
        Standardized matrix. Constant columns are set to zero.
    """
    X = X_in.astype(np.float64, copy=True)
    if X.size == 0:
        return X
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X - mean) / sd
    X[~np.isfinite(X)] = 0.0
    return X


def _svd_boolean_trunc_global(X_mat: np.ndarray) -> np.ndarray:
    """Apply SVD-based dimension reduction with singular value thresholding.

    Standardizes covariates and projects onto principal components with
    singular values exceeding 1e-4, discarding near-collinear directions.

    Parameters
    ----------
    X_mat : np.ndarray, shape (n, k)
        Covariate matrix (may include intercept).

    Returns
    -------
    np.ndarray, shape (n, r)
        Reduced-dimension representation where r <= k.
    """
    Z_full = _scale_all_columns(X_mat)
    if Z_full.shape[1] == 0:
        return Z_full[:, :0]
    U, s, _Vt = np.linalg.svd(Z_full, full_matrices=False)
    mask = (s > 1e-4)
    X_svd = U @ np.diag(mask.astype(float))
    col_sd = X_svd.std(axis=0, ddof=1)
    X_svd = X_svd[:, col_sd > 0]
    return X_svd


def _svd_boolean_trunc_timevary(
    X_mat: np.ndarray,
    time_sorted: np.ndarray,
    unique_time: np.ndarray
) -> np.ndarray:
    """Apply period-specific SVD dimension reduction.

    Performs SVD truncation independently for each time period, then
    vertically concatenates results. Used when ``time_vary=True``.

    Parameters
    ----------
    X_mat : np.ndarray, shape (n, k)
        Covariate matrix.
    time_sorted : np.ndarray, shape (n,)
        Time period indicators (sorted).
    unique_time : np.ndarray, shape (J,)
        Unique time periods.

    Returns
    -------
    np.ndarray, shape (n, r)
        Stacked reduced-dimension matrices.

    Raises
    ------
    ValueError
        If periods yield different numbers of retained components.
    """
    pieces = []
    expected_cols: Optional[int] = None
    for t in unique_time:
        rows = (time_sorted == t)
        Z_full = _scale_all_columns(X_mat[rows, :])
        if Z_full.shape[1] == 0:
            X_sub = np.zeros((rows.sum(), 0), dtype=float)
        else:
            U, s, _Vt = np.linalg.svd(Z_full, full_matrices=False)
            mask = (s > 1e-4)
            X_sub = U @ np.diag(mask.astype(float))
            col_sd = X_sub.std(axis=0, ddof=1)
            X_sub = X_sub[:, col_sd > 0]
        if expected_cols is None:
            expected_cols = X_sub.shape[1]
        if X_sub.shape[1] != expected_cols:
            raise ValueError("Inconsistent SVD column counts across time slices; check data & threshold 1e-4.")
        pieces.append(X_sub)
    return np.vstack(pieces) if pieces else np.zeros((0, 0), dtype=float)


def _integer_base_b(x: int, b: int = 2) -> np.ndarray:
    """Convert integer to base-b digit array.

    Parameters
    ----------
    x : int
        Non-negative integer.
    b : int, default=2
        Base for representation.

    Returns
    -------
    np.ndarray
        Digits in base-b, most significant first.
    """
    if x <= 0:
        return np.array([0], dtype=int)
    bits = []
    while x > 0:
        bits.append(x % b)
        x //= b
    return np.array(list(reversed(bits)), dtype=int)


def msm_loss_func(
    betas: np.ndarray,
    X_c: np.ndarray,
    treat_vec: np.ndarray,
    time_vec: np.ndarray,
    id_vec: Optional[np.ndarray] = None,
    name_cands: Optional[np.ndarray] = None,
    bal_only: bool = False,
    time_sub: Optional[int] = None,
    twostep: bool = False,
    Vcov_inv: Optional[np.ndarray] = None,
    full_var: bool = False,
) -> Dict[str, Any]:
    """Evaluate the GMM objective function for CBMSM.

    Computes moment conditions based on covariate balancing constraints
    derived from the 2^J factorial design framework (Imai and Ratkovic, 2015).

    Parameters
    ----------
    betas : np.ndarray
        Propensity score coefficients. Length k for time-invariant model,
        or k*J for time-varying coefficients.
    X_c : np.ndarray, shape (n, k)
        Design matrix including intercept.
    treat_vec : np.ndarray, shape (n,)
        Binary treatment indicators.
    time_vec : np.ndarray, shape (n,)
        Time period indicators.
    id_vec : np.ndarray, optional
        Unit identifiers (reserved for future use).
    name_cands : np.ndarray, optional
        Unique unit identifiers for determining sample size N.
    bal_only : bool, default=False
        If True, exclude propensity score moment conditions.
    time_sub : int, optional
        Number of initial periods to exclude from constraints. Default 0.
    twostep : bool, default=False
        Use pre-computed weighting matrix (two-step GMM).
    Vcov_inv : np.ndarray, optional
        Pre-computed inverse weighting matrix for two-step estimation.
    full_var : bool, default=False
        Use full (non-approximated) covariance matrix.

    Returns
    -------
    dict
        ``'loss'``: Scalar GMM criterion value.
        ``'Var.inv'``: Inverse weighting matrix.
        ``'probs'``: Propensity scores, shape (N, J).
        ``'w_all'``: Cross-period weight products, shape (N,).
        ``'g.all'``: Stacked moment conditions.
    """
    k0 = X_c.shape[1]
    time0 = time_vec - np.min(time_vec) + 1
    uniq_t = np.array(sorted(pd.unique(time0)))
    n_t = len(uniq_t)
    if time_sub is None:
        # Default: time_sub = 0 (all i>0 constraints active, including first period)
        time_sub = 0
    # Process betas: single set or one per period
    if betas.size == k0:
        beta_mat = np.tile(betas.reshape(k0, 1), (1, n_t))
    else:
        beta_mat = np.asarray(betas, dtype=float).reshape((n_t, k0)).T  # (k0, n_t)

    # N = number of individuals (defined by name_cands); otherwise inferred from data
    if name_cands is not None:
        N = len(name_cands)
    else:
        N = (X_c.shape[0] // len(uniq_t))
    n = N

    # Reshape to N×J (maintain original row order within time==i subset)
    # Keep data in original order (already sorted by time)
    treat_mat = []
    X_blocks = []
    
    for i, t in enumerate(uniq_t, start=1):
        idx = (time0 == t)
        X_sub = X_c[idx, :]
        y_sub = treat_vec[idx].astype(float)
        treat_mat.append(y_sub)
        X_blocks.append(X_sub)
    
    treat_mat = np.column_stack(treat_mat)  # (N, n_t)

    # thetas & probs (strict two-step clipping: lower bound first, then upper bound)
    thetas = np.column_stack([X_blocks[j] @ beta_mat[:, j] for j in range(n_t)])
    probs = 1.0 / (1.0 + np.exp(-thetas))
    probs = np.maximum(probs, PROBS_TRIM)
    probs = np.minimum(probs, 1.0 - PROBS_TRIM)
    probs_obs = treat_mat * probs + (1.0 - treat_mat) * (1.0 - probs)

    # Cross-period weight product w_all (unit-level)
    w_each = treat_mat / probs + (1.0 - treat_mat) / (1.0 - probs)
    w_all = np.prod(w_each, axis=1)

    # Construct binary matrix bin_mat (2^J-1 × J)
    # Fills from right to left in Yates order, producing:
    # i=1: [0,0,1], i=2: [0,1,0], i=3: [0,1,1], i=4: [1,0,0], etc.
    # Column j corresponds to time period j (1-indexed in R, 0-indexed here)
    n_bins = 2 ** n_t - 1
    bin_mat = np.zeros((n_bins, n_t), dtype=int)
    for i in range(1, n_bins + 1):
        bits = _integer_base_b(i, 2)
        bin_mat[i - 1, (n_t - len(bits)):] = bits
    
    # NOTE: Do NOT reverse column order - R's bin.mat is already in correct order
    # Column 0 corresponds to time 1, column 1 to time 2, etc.
    bin_mat_time = bin_mat

    # Construct constr_outer and num_valid counts per period (keep fixed n_bins columns; invalid columns set to 0)
    constr_blocks = []
    num_valid_counts = []  # Number of valid columns (one scalar per period)
    num_valid_outer_seq: list[int] = []  # num_valid_outer: append n-length constant vectors by time
    first_valid_per_time: list[int] = []  # First valid column index per period
    for j in range(n_t):  # j: 0..n_t-1, corresponds to time 1..n_t
        constr = np.zeros((n, n_bins), dtype=float)
        valid_count = 0
        first_valid = None
        for b in range(n_bins):
            is_valid = np.sum(bin_mat_time[b, j:]) > 0
            if is_valid:
                if first_valid is None:
                    first_valid = b
                expo = (treat_mat @ bin_mat_time[b, :]).astype(int)
                sign = (-1.0) ** expo
                constr[:, b] = w_all * sign
                valid_count += 1
            else:
                constr[:, b] = 0.0
        num_valid_counts.append(valid_count)
        num_valid_outer_seq.extend([valid_count] * n)
        first_valid_per_time.append(0 if first_valid is None else first_valid)
        constr_blocks.append(constr)
    constr_outer = np.vstack(constr_blocks)  # (n*n_t, n_bins)

    # Retain binary natural order (1 through 2^J-1) for column indexing

    # g and design weight matrices
    g_prop_list = []
    g_wt_list = []
    X_prop_list = []
    X_wt_list = []
    # Deduplicate in order of appearance, then take value per period
    uniq_seq: list[int] = []
    for val in num_valid_outer_seq:
        if val not in uniq_seq:
            uniq_seq.append(val)
    scale_per_time = [uniq_seq[j] if j < len(uniq_seq) else num_valid_counts[j] for j in range(n_t)]

    for j in range(n_t):
        Xj = X_blocks[j]
        gj = (1.0 / n) * (Xj.T @ (treat_mat[:, j] - probs[:, j]))  # (k0,)
        g_prop_list.append(gj)
        rows = slice(j * n, (j + 1) * n)
        weight_mat = constr_outer[rows, :] * float((j + 1) > time_sub)
        g_wt_list.append((1.0 / n) * (Xj.T @ weight_mat))  # (k0, n_bins)
        X_prop_list.append((1.0 / (n ** 0.5)) * (Xj * np.sqrt(probs_obs[:, j] * (1.0 - probs_obs[:, j]))[:, None]).T)
        if bal_only:
            X_wt_list.append((1.0 / (n ** 0.5)) * (Xj * (float(scale_per_time[j]) ** 0.5)) .T)
        else:
            X_wt_list.append((1.0 / (n ** 0.5)) * (Xj * (np.sqrt(w_all) * (float(scale_per_time[j]) ** 0.5))[:, None]).T)

    g_prop = np.concatenate(g_prop_list, axis=0)  # (k0*n_t,)
    g_wt = np.vstack(g_wt_list)                    # (k0*n_t, n_bins)
    X_prop = np.vstack(X_prop_list)                # (k0*n_t, n)
    X_wt = np.vstack(X_wt_list)                    # (k0*n_t, n)

    # Force first column as g.prop alignment column, other columns keep reordered sequence
    if g_wt.shape[1] >= 1:
        g_prop_all = np.zeros_like(g_wt)
        g_prop_all[:, 0] = g_prop
    else:
        g_prop_all = np.zeros_like(g_wt)
    if bal_only:
        g_prop_all[:] = 0.0
    g_all = np.vstack([g_prop_all, g_wt])          # ((2*k0*n_t) × n_bins)
    X_all = np.vstack([X_prop * (1.0 - float(bal_only)), X_wt])

    # Inverse variance matrix
    if twostep:
        if Vcov_inv is None:
            raise ValueError("Vcov_inv must be provided when twostep=True")
        var_X_inv = Vcov_inv
    else:
        if not full_var:
            var_X_inv = _r_ginv(X_all @ X_all.T)
        else:
            # Construct full variance var_big (Kronecker sum), then vectorize g_all
            # X_t_big: stack X_t repeatedly (same for each period)
            X_t = np.hstack(X_blocks)              # (n, k0*n_t)
            X_t_big = np.vstack([X_t for _ in range(n_t)])
            var_big = None
            for i in range(X_t_big.shape[0]):
                mat1 = np.outer(X_t_big[i, :], X_t_big[i, :])
                vrow = constr_outer[i, :]
                mat2 = np.outer(vrow, vrow)
                kron = np.kron(mat2, mat1)
                var_big = kron if var_big is None else (var_big + kron)
            var_X_inv = _r_ginv(var_big / n)

    # full_var=True: keep only weighted balance conditions (vectorized)
    if full_var:
        g_use = g_wt.reshape(-1, order='F')  # Column-major vectorization
        var_mat = var_X_inv
        loss_mat = (g_use.T @ var_mat @ g_use)
        loss_scalar = float(loss_mat) * n
        return {"loss": loss_scalar, "Var.inv": var_X_inv, "probs": probs, "w_all": w_all, "g.all": g_use}

    # Objective: tr(g' V g) * n
    prod_mat = g_all.T @ var_X_inv @ g_all
    loss_scalar = float(np.trace(prod_mat)) * n
    return {"loss": loss_scalar, "Var.inv": var_X_inv, "probs": probs, "w_all": w_all, "g.all": g_all}

def cbmsm_fit(
    treat: np.ndarray,
    X: np.ndarray,
    id: np.ndarray,
    time: np.ndarray,
    type: str = "MSM",
    twostep: bool = True,
    msm_variance: str = "approx",
    time_vary: bool = False,
    init: str = "opt",
    sample_weights: Optional[np.ndarray] = None,
    iterations: Optional[int] = None,
    **kwargs: Any,
) -> CBMSMResults:
    """Fit CBMSM using the matrix interface.

    Low-level function accepting preprocessed numpy arrays. For most users,
    the formula interface ``CBMSM()`` is recommended.

    Parameters
    ----------
    treat : np.ndarray, shape (n,)
        Binary treatment vector for N units over J time periods.
    X : np.ndarray, shape (n, k)
        Design matrix including intercept column.
    id : np.ndarray, shape (n,)
        Unit identifiers.
    time : np.ndarray, shape (n,)
        Time period indicators.
    type : {'MSM', 'MultiBin'}, default='MSM'
        ``'MSM'`` for marginal structural models (longitudinal),
        ``'MultiBin'`` for multiple binary treatments (cross-sectional).
    twostep : bool, default=True
        Use two-step GMM (faster) vs. continuous updating.
    msm_variance : {'approx', 'full'}, default='approx'
        ``'approx'`` uses low-rank approximation (Imai and Ratkovic, 2015,
        equation 27). ``'full'`` uses complete covariance (slower).
    time_vary : bool, default=False
        Estimate separate coefficients for each time period.
    init : {'opt', 'glm', 'CBPS'}, default='opt'
        Initialization method. ``'opt'`` selects the better of GLM and CBPS.
    sample_weights : np.ndarray, optional
        Observation weights (not currently used).
    iterations : int, optional
        Maximum optimization iterations.
    **kwargs
        Additional arguments (reserved).

    Returns
    -------
    CBMSMResults
        Fitted model with weights, coefficients, and diagnostics.

    Notes
    -----
    Data must form a balanced panel: each unit appears exactly once per
    time period. The function automatically sorts by (time, id) internally.

    References
    ----------
    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. Journal of the American Statistical
    Association, 110(511), 1013-1023.

    See Also
    --------
    CBMSM : Formula interface (recommended).

    Examples
    --------
    >>> import numpy as np
    >>> from cbps import cbmsm_fit
    >>> # 10 units, 3 time periods
    >>> n, J = 10, 3
    >>> treat = np.random.binomial(1, 0.5, n * J)
    >>> X = np.column_stack([np.ones(n * J), np.random.randn(n * J, 2)])
    >>> id_vec = np.tile(np.arange(n), J)
    >>> time_vec = np.repeat(np.arange(J), n)
    >>> result = cbmsm_fit(treat, X, id_vec, time_vec)
    >>> print(result.converged)
    """
    # Parameter validation
    valid_types = {'MSM', 'MultiBin'}
    if type not in valid_types:
        raise ValueError(
            f"Invalid type='{type}'. Must be one of {valid_types}. "
            f"Use 'MSM' for marginal structural models or 'MultiBin' for multiple binary treatments."
        )
    
    valid_inits = {'glm', 'opt', 'CBPS'}
    if init not in valid_inits:
        raise ValueError(
            f"Invalid init='{init}'. Must be one of {valid_inits}. "
            f"Use 'glm' for GLM initialization, 'CBPS' for CBPS initialization, "
            f"or 'opt' for choosing the best between CBPS and GLM (recommended)."
        )
    
    valid_variances = {'approx', 'full', None}
    if msm_variance not in valid_variances:
        raise ValueError(
            f"Invalid msm_variance='{msm_variance}'. "
            f"Must be one of {{'approx', 'full', None}}. "
            f"Use 'approx' for low-rank approximation (recommended) or 'full' for full variance matrix."
        )
    
    # Convert to numpy arrays
    treat = np.asarray(treat).ravel()
    X = np.asarray(X)
    id_arr = np.asarray(id).ravel()
    time_arr = np.asarray(time).ravel()
    
    # Validate dimensions
    n_obs = len(treat)
    if X.shape[0] != n_obs:
        raise ValueError(f"X.shape[0]={X.shape[0]} must equal len(treat)={n_obs}")
    if len(id_arr) != n_obs:
        raise ValueError(f"len(id)={len(id_arr)} must equal len(treat)={n_obs}")
    if len(time_arr) != n_obs:
        raise ValueError(f"len(time)={len(time_arr)} must equal len(treat)={n_obs}")
    
    # Zero-variance column filtering
    std = X.std(axis=0, ddof=1)
    X_mat = X[:, std > 0]
    
    # Sort by time in ascending order
    id_sorted, time_sorted, y_sorted, X_sorted, order = _sort_by_time(id_arr, time_arr, treat, X_mat)
    
    # Balanced panel check
    name_cands_check = np.array(sorted(pd.unique(id_sorted)))
    unique_time_check = np.array(sorted(pd.unique(time_sorted)))
    N_expected = len(name_cands_check)
    J_expected = len(unique_time_check)
    N_total_expected = N_expected * J_expected
    N_actual = len(id_sorted)
    
    if N_actual != N_total_expected:
        id_time_counts = pd.DataFrame({
            'id': id_sorted,
            'time': time_sorted
        }).groupby(['id', 'time']).size()
        
        all_combinations = pd.MultiIndex.from_product(
            [name_cands_check, unique_time_check],
            names=['id', 'time']
        )
        missing = all_combinations.difference(id_time_counts.index)
        duplicates = id_time_counts[id_time_counts > 1]
        
        error_msg = (
            f"CBMSM requires a balanced panel: each id must appear exactly once at each time point.\n"
            f"Expected {N_expected} units × {J_expected} time periods = {N_total_expected} observations, "
            f"but found {N_actual} observations.\n"
        )
        
        if len(missing) > 0:
            missing_sample = missing[:5].tolist()
            error_msg += f"\nMissing {len(missing)} id×time combinations (showing first 5): {missing_sample}"
        
        if len(duplicates) > 0:
            dup_sample = duplicates.head(5).to_dict()
            error_msg += f"\nDuplicate id×time combinations (showing first 5): {dup_sample}"
        
        raise ValueError(error_msg)
    
    # Verify at least 2 individuals and 2 time periods
    if N_expected < 2:
        raise ValueError(
            f"CBMSM requires at least 2 individuals (units), but found only {N_expected} unique id."
        )
    
    if J_expected < 2:
        raise ValueError(
            f"CBMSM requires at least 2 time periods, but found only {J_expected} unique time value."
        )
    
    # Build treatment history matrix
    treat_hist, name_cands, unique_time = _build_treat_hist(y_sorted, id_sorted, time_sorted)
    treat_cum = treat_hist.sum(axis=1)
    
    # Design matrix: MSM uses SVD boolean truncation; MultiBin uses zero-variance removal + intercept
    if type == "MultiBin":
        X_nozero = X_sorted[:, 1:] if X_sorted.shape[1] > 1 else X_sorted[:, :0]
        std2 = X_nozero.std(axis=0, ddof=1)
        X_nz = X_nozero[:, std2 > 0]
        X_c = np.column_stack([np.ones(X_sorted.shape[0]), X_nz])
    else:
        if not time_vary:
            X_svd = _svd_boolean_trunc_global(X_sorted)
        else:
            X_svd = _svd_boolean_trunc_timevary(X_sorted, time_sorted, unique_time)
        X_c = np.column_stack([np.ones(X_svd.shape[0]), X_svd]) if X_svd.size else np.ones((X_sorted.shape[0], 1))
    
    # Determine full variance matrix computation mode
    full_var = (msm_variance == "full")
    
    # Starting points: GLM and CBPS
    if not time_vary:
        glm_model = sm.GLM(y_sorted, X_c, family=Binomial())
        glm_fit_res = glm_model.fit(maxiter=50, tol=1e-8)
        glm1 = glm_fit_res.params.astype(float)
        cbps_res = cbps_binary_fit(
            treat=y_sorted,
            X=X_c,
            att=0,
            method='exact',
            two_step=True,
            standardize=True,
            iterations=200,
        )
        glm_cb = cbps_res['coefficients'].ravel()
    else:
        # Time-varying: per-period GLM and CBPS, vectorized
        glm_cols = []
        cbps_cols = []
        for t in unique_time:
            mask = (time_sorted == t)
            X_sub = X_c[mask, :]
            y_sub = y_sorted[mask]
            glm_sub = sm.GLM(y_sub, X_sub, family=Binomial()).fit(maxiter=50, tol=1e-8).params
            cbps_sub = cbps_binary_fit(
                treat=y_sub,
                X=X_sub,
                att=0,
                method='exact',
                two_step=True,
                standardize=True,
                iterations=200,
            )['coefficients'].ravel()
            glm_cols.append(glm_sub)
            cbps_cols.append(cbps_sub)
        glm_mat = np.column_stack(glm_cols)
        cbps_mat = np.column_stack(cbps_cols)
        glm_mat[np.isnan(glm_mat)] = 0.0
        cbps_mat[np.isnan(cbps_mat)] = 0.0
        glm1 = glm_mat.ravel(order='F')
        glm_cb = cbps_mat.ravel(order='F')
    
    # Select best: compare initial loss in twostep=FALSE setting
    glm_fit_dict = msm_loss_func(glm1, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    cb_fit_dict = msm_loss_func(glm_cb, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    loss_glm = glm_fit_dict['loss']
    loss_cb = cb_fit_dict['loss']
    type_fit = "Returning Estimates from Logistic Regression\n"
    if ((loss_cb < loss_glm and init == 'opt') or init == 'CBPS'):
        init_betas = glm_cb
        init_fit = cb_fit_dict
        type_fit = "Returning Estimates from CBPS\n"
    else:
        init_betas = glm1
        init_fit = glm_fit_dict
    
    # R always runs twostep first as warm start, then optionally runs
    # continuous updating. This two-stage strategy is standard GMM practice.
    glm_fit0 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    Vcov_inv = glm_fit0['Var.inv'] if 'Var.inv' in glm_fit0 else None
    if Vcov_inv is None:
        Vcov_inv = _r_ginv(X_c.T @ X_c)

    # Stage 1: Always run twostep optimization (warm start)
    def twostep_loss(b: np.ndarray) -> float:
        return msm_loss_func(b, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)['loss']
    opt1 = scipy.optimize.minimize(
        twostep_loss, init_betas, method='BFGS', options={'maxiter': iterations or 500}
    )
    msm_fit = msm_loss_func(opt1.x, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)
    l3 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)
    if (l3['loss'] < msm_fit['loss']) and (init == 'opt'):
        msm_fit = l3
        warnings.warn("Warning: Optimization did not improve over initial estimates\n")
        print(type_fit, end='')

    # Stage 2: If continuous updating requested, refine from twostep result
    if not twostep:
        def cont_loss(b: np.ndarray) -> float:
            return msm_loss_func(b, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)['loss']
        # Use twostep result as starting point (warm start)
        cont_start = opt1.x
        opt1 = scipy.optimize.minimize(
            cont_loss, cont_start, method='BFGS', options={'maxiter': iterations or 500}
        )
        msm_fit = msm_loss_func(opt1.x, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)
        l3 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)
        if (l3['loss'] < msm_fit['loss']) and (init == 'opt'):
            msm_fit = l3
            print("\nWarning: Optimization did not improve over initial estimates\n", end='')
            print(type_fit, end='')
    
    # Unconditional frequencies and final weights (first period subset)
    n_obs = len(np.unique(id_sorted))
    n_time = len(np.unique(time_sorted))
    treat_hist2, name_cands2, uniq_t2 = _build_treat_hist(y_sorted, id_sorted, time_sorted)
    rows_as_str = ["|".join(map(str, row.tolist())) for row in treat_hist2]
    uniq_rows, counts = np.unique(rows_as_str, return_counts=True)
    freq_map = {u: c / n_obs for u, c in zip(uniq_rows, counts)}
    uncond_probs = np.array([freq_map[s] for s in rows_as_str])
    
    # w_all = ∏_j 1/P(T_j|X_j) = 1/P(T̄|X̄)  (unstabilized IPW)
    # fitted_values = uncond_probs / w_all = P(T̄) × P(T̄|X̄)
    # weights = w_all = 1/P(T̄|X̄)
    # The fitted_values are used as WLS weights for outcome regression
    # (reproduces Imai & Ratkovic 2015, Table 3).
    probs_out = msm_fit['w_all']
    wts_out_all = np.tile(uncond_probs / probs_out, n_time)
    wts_out = wts_out_all[time_sorted == np.min(time_sorted)]
    
    glm_weights_all = np.tile(uncond_probs / glm_fit_dict['w_all'], n_time)
    glm_weights = glm_weights_all[time_sorted == np.min(time_sorted)]
    
    # Compute loss values
    loss_glm_final = glm_fit_dict['loss']
    loss_msm_final = msm_fit['loss']
    
    # Compute diagnostics
    final_coefficients = opt1.x if 'opt1' in locals() else init_betas
    
    if time_vary:
        k = X_c.shape[1]
        n_periods = len(unique_time)
        final_coefficients = final_coefficients.reshape(k, n_periods, order='F')
    
    J_statistic = loss_msm_final / n_obs
    var_matrix = msm_fit.get('Var.inv', Vcov_inv)
    if var_matrix is None:
        var_matrix = np.array([[]])
    
    converged_status = opt1.success if 'opt1' in locals() else True
    
    # Assemble return object
    out = CBMSMResults(
        weights=probs_out,
        fitted_values=wts_out,
        glm_g=glm_fit_dict['g.all'],
        msm_g=msm_fit['g.all'],
        glm_weights=glm_weights,
        id=id_sorted,
        treat_hist=treat_hist,
        treat_cum=treat_cum,
        coefficients=final_coefficients,
        n_periods=n_time,
        n_units=n_obs,
        time_vary=time_vary,
        converged=converged_status,
        J=J_statistic,
        var=var_matrix,
        call=f"cbmsm_fit(type='{type}', twostep={twostep}, msm_variance='{msm_variance}', time_vary={time_vary}, init='{init}')",
        formula=None,  # Matrix interface has no formula
        y=y_sorted,
        x=X_sorted,
        time=time_sorted,
        model=None,  # Matrix interface has no model frame
        data=None,  # Matrix interface has no original data
    )
    
    # Final override with first period subset
    mask_first = (time_sorted == np.min(time_sorted))
    if out.weights.shape[0] == time_sorted.shape[0]:
        out.weights = out.weights[mask_first]
    
    # Warning
    if loss_glm_final < loss_msm_final:
        warnings.warn(
            f"CBMSM fails to improve covariate balance relative to MLE.  \n GLM loss:    {loss_glm_final} \n CBMSM loss:  {loss_msm_final} \n"
        )
    
    return out


def CBMSM(
    formula: str,
    id: str | pd.Series | np.ndarray,
    time: str | pd.Series | np.ndarray,
    data: pd.DataFrame,
    type: str = "MSM",
    twostep: bool = True,
    msm_variance: str = "approx",
    time_vary: bool = False,
    init: str = "opt",
    iterations: Optional[int] = None,
    **kwargs: Any,
) -> CBMSMResults:
    """Covariate Balancing Propensity Score for Marginal Structural Models.

    Estimates inverse probability weights for longitudinal causal inference
    by jointly optimizing treatment prediction and covariate balance across
    all time periods. The method is robust to treatment model misspecification
    because it directly targets the covariate balancing property of IPW.

    Parameters
    ----------
    formula : str
        Model specification in Patsy format, e.g., ``"treat ~ x1 + x2"``.
        The same formula applies to all time periods.
    id : str or array-like
        Unit (individual) identifiers. Either a column name in ``data``
        or an array of length n.
    time : str or array-like
        Time period indicators. Either a column name or an array.
    data : pd.DataFrame
        Panel data containing all model variables.
    type : {'MSM', 'MultiBin'}, default='MSM'
        ``'MSM'`` for longitudinal treatment sequences, ``'MultiBin'``
        for multiple simultaneous binary treatments.
    twostep : bool, default=True
        Use two-step GMM estimator (faster, recommended) or continuous
        updating GMM.
    msm_variance : {'approx', 'full'}, default='approx'
        Covariance approximation method. ``'approx'`` uses the low-rank
        approximation from Imai and Ratkovic (2015, eq. 27), which assumes
        zero correlation across balance conditions. ``'full'`` computes
        the complete covariance but is computationally expensive.
    time_vary : bool, default=False
        If True, estimate separate propensity score coefficients for each
        time period. Default shares coefficients across periods.
    init : {'opt', 'glm', 'CBPS'}, default='opt'
        Starting value selection. ``'opt'`` tries both GLM and CBPS
        initializations and selects the one with better balance.
    iterations : int, optional
        Maximum optimization iterations.
    **kwargs
        Additional arguments (reserved for future use).

    Returns
    -------
    CBMSMResults
        Fitted model object containing:

        - ``weights``: Inverse probability weights 1/P(T|X)
        - ``fitted_values``: Stabilized MSM weights P(T)/P(T|X)
        - ``coefficients``: Estimated propensity score parameters
        - ``treat_hist``: Treatment history matrix (N × J)
        - ``converged``: Optimization convergence indicator
        - ``J``: Hansen J-statistic for model diagnostics

    Notes
    -----
    The estimator solves a GMM problem with moment conditions derived from
    the covariate balancing property of MSM weights. At time period j,
    the weights should balance covariates across all 2^{J-j+1} possible
    current and future treatment sequences, conditional on past treatment.

    The ``fitted_values`` (stabilized weights) are recommended for outcome
    regression to reduce variance. These incorporate the marginal treatment
    probability in the numerator: w* = P(T) / P(T|X).

    Data must form a balanced panel where each unit appears exactly once
    at each time period.

    References
    ----------
    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. Journal of the American Statistical
    Association, 110(511), 1013-1023.

    See Also
    --------
    cbmsm_fit : Matrix interface for advanced users.
    CBPS : Cross-sectional propensity score estimation.

    Examples
    --------
    >>> from cbps import CBMSM
    >>> from cbps.datasets import load_blackwell
    >>> blackwell = load_blackwell()
    >>> fit = CBMSM(
    ...     formula="d.gone.neg ~ d.gone.neg.l1 + camp.length",
    ...     id="demName",
    ...     time="time",
    ...     data=blackwell,
    ... )
    >>> # Use stabilized weights for outcome analysis
    >>> import statsmodels.api as sm
    >>> y = blackwell.groupby("demName")["demprcnt"].last()
    >>> # ... weighted outcome regression with fit.fitted_values
    """
    # Parameter validation
    
    # Validate formula parameter
    if not isinstance(formula, str):
        raise TypeError(
            f"formula must be a string, got {type(formula).__name__}. "
            f"CBMSM uses a single formula applied to all time periods. "
            f"Example: 'treat ~ x1 + x2'. "
            f"Set time_vary=True for period-specific coefficients."
        )
    
    # Validate type parameter
    valid_types = {'MSM', 'MultiBin'}
    if type not in valid_types:
        raise ValueError(
            f"Invalid type='{type}'. Must be one of {valid_types}. "
            f"Note: parameter values are case-sensitive. "
            f"Use 'MSM' for marginal structural models or 'MultiBin' for multiple binary treatments."
        )
    
    # Validate init parameter
    valid_inits = {'glm', 'opt', 'CBPS'}
    if init not in valid_inits:
        raise ValueError(
            f"Invalid init='{init}'. Must be one of {valid_inits}. "
            f"Use 'glm' for GLM initialization, 'CBPS' for CBPS initialization, "
            f"or 'opt' for choosing the best between CBPS and GLM (recommended)."
        )
    
    # Validate msm_variance parameter
    valid_variances = {'approx', 'full', None}
    if msm_variance not in valid_variances:
        raise ValueError(
            f"Invalid msm_variance='{msm_variance}'. "
            f"Must be one of {{'approx', 'full', None}}. "
            f"Use 'approx' for low-rank approximation (recommended) or 'full' for full variance matrix."
        )
    
    # Preprocessing: auto-quote column names with special characters for patsy compatibility
    def _quote_formula_variables(formula_str: str, columns: pd.Index) -> str:
        # Only quote column names containing non-alphanumeric characters (except underscore)
        special_cols = [c for c in columns if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(c))]
        # Sort by length descending to avoid prefix replacement issues
        special_cols.sort(key=lambda s: len(str(s)), reverse=True)
        out = formula_str
        for col in special_cols:
            name = re.escape(str(col))
            pattern = rf"(?<![A-Za-z0-9_\"]){name}(?![A-Za-z0-9_\"])"
            out = re.sub(pattern, f'Q("{str(col)}")', out)
        return out

    safe_formula = _quote_formula_variables(formula, data.columns)
    
    # Save original data
    data_original = data.copy()
    
    # Extract terms object
    from patsy import dmatrices
    _, X_design = dmatrices(safe_formula, data, return_type='dataframe')
    terms_obj = X_design.design_info

    # Formula parsing (generates y, X matrices, parse_formula includes intercept)
    y_raw, X_raw = parse_formula(safe_formula, data)
    # Zero variance filtering (removes intercept column since sd=0)
    std = X_raw.std(axis=0, ddof=1)
    X_mat = X_raw[:, std > 0]  # Don't manually add intercept (intercept column sd=0 is filtered)

    # Input vectorization - supports column name string or array
    if isinstance(id, str):
        id_arr = data[id].values
    else:
        id_arr = np.asarray(id)

    if isinstance(time, str):
        time_arr = data[time].values
    else:
        time_arr = np.asarray(time)

    # Sort by time in ascending order for temporal consistency
    id_sorted, time_sorted, y_sorted, X_sorted, order = _sort_by_time(id_arr, time_arr, y_raw, X_mat)

    # Balanced panel check: verify each id appears exactly once at each time
    # This is a hard constraint for CBMSM algorithm, must be checked before _build_treat_hist
    # Otherwise np.column_stack in msm_loss_func will fail
    name_cands_check = np.array(sorted(pd.unique(id_sorted)))
    unique_time_check = np.array(sorted(pd.unique(time_sorted)))
    N_expected = len(name_cands_check)
    J_expected = len(unique_time_check)
    N_total_expected = N_expected * J_expected
    N_actual = len(id_sorted)

    if N_actual != N_total_expected:
        # Check for duplicate or missing id×time combinations
        id_time_counts = pd.DataFrame({
            'id': id_sorted,
            'time': time_sorted
        }).groupby(['id', 'time']).size()

        # Find missing combinations
        all_combinations = pd.MultiIndex.from_product(
            [name_cands_check, unique_time_check],
            names=['id', 'time']
        )
        missing = all_combinations.difference(id_time_counts.index)
        duplicates = id_time_counts[id_time_counts > 1]

        error_msg = (
            f"CBMSM requires a balanced panel: each id must appear exactly once at each time point.\n"
            f"Expected {N_expected} units × {J_expected} time periods = {N_total_expected} observations, "
            f"but found {N_actual} observations.\n"
        )

        if len(missing) > 0:
            missing_sample = missing[:5].tolist()
            error_msg += f"\nMissing {len(missing)} id×time combinations (showing first 5): {missing_sample}"
            if len(missing) > 5:
                error_msg += f" ... and {len(missing) - 5} more"

        if len(duplicates) > 0:
            dup_sample = duplicates.head(5).to_dict()
            error_msg += f"\nDuplicate id×time combinations (showing first 5): {dup_sample}"
            if len(duplicates) > 5:
                error_msg += f" ... and {len(duplicates) - 5} more"

        error_msg += (
            "\n\nPlease ensure your data is a complete balanced panel before calling CBMSM.\n"
            "You may need to:\n"
            "  1. Filter to only include units with complete time series, or\n"
            "  2. Impute/fill missing time points with appropriate values."
        )

        raise ValueError(error_msg)

    # Verify at least 2 individuals and 2 time periods (MSM minimum requirement)
    if N_expected < 2:
        raise ValueError(
            f"CBMSM requires at least 2 individuals (units), but found only {N_expected} unique id. "
            f"Marginal Structural Models are designed for panel data with multiple units over time.\n\n"
            f"Reason: MSM estimates treatment effects using within-unit variation over time "
            f"and across-unit comparison. With only 1 unit, there is no cross-sectional variation.\n\n"
            f"If you have cross-sectional data (one time point), use CBPS() instead."
        )
    
    if J_expected < 2:
        raise ValueError(
            f"CBMSM requires at least 2 time periods, but found only {J_expected} unique time value. "
            f"Marginal Structural Models are designed for longitudinal data with multiple time points.\n\n"
            f"Reason: MSM estimates treatment effects using temporal variation in treatment assignment. "
            f"With only 1 time period, this reduces to standard cross-sectional CBPS.\n\n"
            f"If you have cross-sectional data, use CBPS() instead."
        )
    
    # Build treatment history matrix (N×J)
    treat_hist, name_cands, unique_time = _build_treat_hist(y_sorted, id_sorted, time_sorted)
    treat_cum = treat_hist.sum(axis=1)

    # Design matrix: MSM uses SVD boolean truncation; MultiBin uses zero-variance removal + intercept
    if type == "MultiBin":
        X_nozero = X_sorted[:, 1:] if X_sorted.shape[1] > 1 else X_sorted[:, :0]
        std2 = X_nozero.std(axis=0, ddof=1)
        X_nz = X_nozero[:, std2 > 0]
        X_c = np.column_stack([np.ones(X_sorted.shape[0]), X_nz])
    else:
        if not time_vary:
            # SVD-based dimension reduction: scale, decompose, and threshold
            X_svd = _svd_boolean_trunc_global(X_sorted)
        else:
            X_svd = _svd_boolean_trunc_timevary(X_sorted, time_sorted, unique_time)
        X_c = np.column_stack([np.ones(X_svd.shape[0]), X_svd]) if X_svd.size else np.ones((X_sorted.shape[0], 1))

    # Determine full variance matrix computation mode
    full_var = (msm_variance == "full")

    # Starting points: GLM and CBPS
    if not time_vary:
        glm_model = sm.GLM(y_sorted, X_c, family=Binomial())
        glm_fit_res = glm_model.fit(maxiter=50, tol=1e-8)
        glm1 = glm_fit_res.params.astype(float)
        cbps_res = cbps_binary_fit(
            treat=y_sorted,
            X=X_c,
            att=0,
            method='exact',
            two_step=True,
            standardize=True,
            iterations=200,
        )
        glm_cb = cbps_res['coefficients'].ravel()
    else:
        # Time-varying: per-period GLM and CBPS, vectorized
        glm_cols = []
        cbps_cols = []
        for t in unique_time:
            mask = (time_sorted == t)
            X_sub = X_c[mask, :]
            y_sub = y_sorted[mask]
            glm_sub = sm.GLM(y_sub, X_sub, family=Binomial()).fit(maxiter=50, tol=1e-8).params
            cbps_sub = cbps_binary_fit(
                treat=y_sub,
                X=X_sub,
                att=0,
                method='exact',
                two_step=True,
                standardize=True,
                iterations=200,
            )['coefficients'].ravel()
            glm_cols.append(glm_sub)
            cbps_cols.append(cbps_sub)
        glm_mat = np.column_stack(glm_cols)
        cbps_mat = np.column_stack(cbps_cols)
        glm_mat[np.isnan(glm_mat)] = 0.0
        cbps_mat[np.isnan(cbps_mat)] = 0.0
        glm1 = glm_mat.ravel(order='F')  # Column-major flattening
        glm_cb = cbps_mat.ravel(order='F')

    # Select best: compare initial loss in twostep=FALSE setting
    glm_fit_dict = msm_loss_func(glm1, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    cb_fit_dict = msm_loss_func(glm_cb, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    loss_glm = glm_fit_dict['loss']
    loss_cb = cb_fit_dict['loss']
    type_fit = "Returning Estimates from Logistic Regression\n"
    if ((loss_cb < loss_glm and init == 'opt') or init == 'CBPS'):
        init_betas = glm_cb
        init_fit = cb_fit_dict
        type_fit = "Returning Estimates from CBPS\n"
    else:
        init_betas = glm1
        init_fit = glm_fit_dict

    # R always runs twostep first as warm start, then optionally runs
    # continuous updating. This two-stage strategy is standard GMM practice.
    glm_fit0 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, twostep=False, full_var=full_var)
    Vcov_inv = glm_fit0['Var.inv'] if 'Var.inv' in glm_fit0 else None
    if Vcov_inv is None:
        Vcov_inv = _r_ginv(X_c.T @ X_c)

    # Stage 1: Always run twostep optimization (warm start)
    def twostep_loss(b: np.ndarray) -> float:
        return msm_loss_func(b, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)['loss']
    opt1 = scipy.optimize.minimize(
        twostep_loss, init_betas, method='BFGS', options={'maxiter': iterations or 500}
    )
    msm_fit = msm_loss_func(opt1.x, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)
    l3 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=True, Vcov_inv=Vcov_inv, full_var=full_var)
    if (l3['loss'] < msm_fit['loss']) and (init == 'opt'):
        msm_fit = l3
        warnings.warn("Warning: Optimization did not improve over initial estimates\n")
        print(type_fit, end='')

    # Stage 2: If continuous updating requested, refine from twostep result
    if not twostep:
        def cont_loss(b: np.ndarray) -> float:
            return msm_loss_func(b, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)['loss']
        # Use twostep result as starting point (warm start)
        cont_start = opt1.x
        opt1 = scipy.optimize.minimize(
            cont_loss, cont_start, method='BFGS', options={'maxiter': iterations or 500}
        )
        msm_fit = msm_loss_func(opt1.x, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)
        l3 = msm_loss_func(init_betas, X_c, y_sorted, time_sorted, id_vec=id_sorted, name_cands=name_cands, bal_only=True, twostep=False, Vcov_inv=None, full_var=full_var)
        if (l3['loss'] < msm_fit['loss']) and (init == 'opt'):
            msm_fit = l3
            print("\nWarning: Optimization did not improve over initial estimates\n", end='')
            print(type_fit, end='')

    # Unconditional frequencies and final weights (first period subset)
    n_obs = len(np.unique(id_sorted))
    n_time = len(np.unique(time_sorted))
    # Build treat_hist and unique row frequencies from sorted id/time
    treat_hist2, name_cands2, uniq_t2 = _build_treat_hist(y_sorted, id_sorted, time_sorted)
    # Compute frequency of each row in unique set
    # Use string key comparison for row equivalence (unique row identification)
    rows_as_str = ["|".join(map(str, row.tolist())) for row in treat_hist2]
    uniq_rows, counts = np.unique(rows_as_str, return_counts=True)
    freq_map = {u: c / n_obs for u, c in zip(uniq_rows, counts)}
    uncond_probs = np.array([freq_map[s] for s in rows_as_str])

    probs_out = msm_fit['w_all']  # Length n (unit-level)
    # w_all = ∏_j 1/P(T_j|X_j) = 1/P(T̄|X̄)  (unstabilized IPW)
    # fitted_values = uncond_probs / w_all = P(T̄) × P(T̄|X̄)
    # weights = w_all = 1/P(T̄|X̄)
    # The fitted_values are used as WLS weights for outcome regression
    # (reproduces Imai & Ratkovic 2015, Table 3).
    wts_out_all = np.tile(uncond_probs / probs_out, n_time)
    wts_out = wts_out_all[time_sorted == np.min(time_sorted)]

    glm_weights_all = np.tile(uncond_probs / glm_fit_dict['w_all'], n_time)
    glm_weights = glm_weights_all[time_sorted == np.min(time_sorted)]

    # Compute loss values (for warnings and J-statistic)
    loss_glm_final = glm_fit_dict['loss']
    loss_msm_final = msm_fit['loss']

    # Compute diagnostics
    # 1. coefficients: final optimized treatment model coefficients
    final_coefficients = opt1.x if 'opt1' in locals() else init_betas
    
    # When time_vary=True, reshape to (k, n_periods) matrix
    # Each column represents coefficients for one time period
    if time_vary:
        k = X_c.shape[1]
        n_periods = len(unique_time)
        final_coefficients = final_coefficients.reshape(k, n_periods, order='F')
    # When time_vary=False, keep (k,) vector format (all periods share same coefficients)

    # 2. J-statistic: GMM loss value (Hansen J-statistic)
    J_statistic = loss_msm_final / n_obs  # Normalized J-statistic

    # 3. Inverse of variance-covariance matrix
    var_matrix = msm_fit.get('Var.inv', Vcov_inv)
    if var_matrix is None:
        var_matrix = np.array([[]])  # Empty matrix as placeholder

    # 4. Convergence status
    converged_status = opt1.success if 'opt1' in locals() else True

    # Assemble return object
    out = CBMSMResults(
        # Core weights (6)
        weights=probs_out,  # Will be overwritten to first-period subset
        fitted_values=wts_out,
        glm_g=glm_fit_dict['g.all'],
        msm_g=msm_fit['g.all'],
        glm_weights=glm_weights,
        id=id_sorted,  # Full length vector (aligned with y/time)
        # Treatment history (2)
        treat_hist=treat_hist,
        treat_cum=treat_cum,
        # Diagnostics (7)
        coefficients=final_coefficients,
        n_periods=n_time,
        n_units=n_obs,
        time_vary=time_vary,
        converged=converged_status,
        J=J_statistic,
        var=var_matrix,
        # Metadata (7)
        call=f"CBMSM(formula='{formula}', type='{type}', twostep={twostep}, msm_variance='{msm_variance}', time_vary={time_vary}, init='{init}')",
        formula=formula,
        y=y_sorted,
        x=X_sorted,
        time=time_sorted,
        # Model frame construction
        model=data_original,  # Use original data as model frame
        data=data_original,  # Use saved original data
    )

    # Final override to first-period subset
    mask_first = (time_sorted == np.min(time_sorted))
    if out.weights.shape[0] == time_sorted.shape[0]:
        out.weights = out.weights[mask_first]
    else:
        # Already unit-level length (N), keep unchanged
        pass

    # Warn if GLM loss is less than MSM loss
    if loss_glm_final < loss_msm_final:
        warnings.warn(
            f"CBMSM fails to improve covariate balance relative to MLE.  \n GLM loss:    {loss_glm_final} \n CBMSM loss:  {loss_msm_final} \n"
        )

    return out


