"""
Nonparametric Covariate Balancing Propensity Score (npCBPS).

This module implements the nonparametric covariate balancing generalized
propensity score (npCBGPS) estimator from Section 3.3 of Fong, Hazlett,
and Imai (2018). Unlike parametric CBPS, this approach does not specify
a functional form for the propensity score. Instead, it directly estimates
inverse probability weights by maximizing the empirical likelihood subject
to covariate balance constraints.

Key Features
------------
- **Model-free**: No parametric assumptions about treatment assignment.
- **Empirical likelihood**: Weights chosen to maximize data likelihood.
- **Penalized imbalance**: Allows controlled finite-sample imbalance
  via the ``corprior`` parameter.

Main API
--------
- :func:`npCBPS`: High-level function accepting formula and DataFrame.
- :class:`NPCBPSResults`: Container for estimated weights and diagnostics.

Algorithm Overview
------------------
1. Whiten covariates: :math:`X^* = S_X^{-1/2}(X - \\bar{X})`.
2. Standardize treatment: :math:`T^* = (T - \\bar{T})/s_T`.
3. Construct constraint matrix: :math:`g_i = (X_i^* T_i^*, X_i^*, T_i^*)^T`.
4. Line search over :math:`\\alpha \\in [0, 1]` to maximize penalized
   likelihood (Equation 10).
5. Recover weights: :math:`w_i = 1/(1 - \\gamma^T(g_i - \\eta))`.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. The Annals of Applied Statistics, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101
"""

from typing import Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
import scipy.optimize

from ..utils.formula import parse_formula
from .cholesky_whitening import cholesky_whitening
from .empirical_likelihood import get_w, log_post


class NPCBPSSummary:
    """Summary object for NPCBPSResults.

    Returned by :meth:`NPCBPSResults.summary`. Provides a structured
    representation of npCBPS estimation results that can be printed
    via ``print()`` or ``str()``.

    Attributes
    ----------
    call : str or None
        String representation of the function call.
    n : int
        Total sample size.
    treatment_type : {"binary", "continuous", "unknown"}
        ``"binary"`` when the treatment vector takes only values in ``{0, 1}``;
        ``"continuous"`` otherwise. npCBPS internally treats any numeric
        treatment (including ``0/1``) as continuous during estimation, but
        the summary still respects the original treatment type when
        describing sample composition.
    n_treat : int or None
        Number of treated units (``y == 1``). Populated only when
        ``treatment_type == "binary"``.
    n_control : int or None
        Number of control units (``y == 0``). Populated only when
        ``treatment_type == "binary"``.
    treatment_range : tuple of (float, float) or None
        ``(min, max)`` of the treatment vector. Populated only when
        ``treatment_type == "continuous"``.
    treatment_mean : float or None
        Arithmetic mean of the treatment vector. Populated only when
        ``treatment_type == "continuous"``.
    converged : bool or None
        Whether the optimization converged.
    iterations : int or None
        Number of iterations used.
    sumw0 : float or None
        Sum of unnormalized weights (should ≈ 1.0).
    log_el : float or None
        Log empirical likelihood at the optimum.
    log_p_eta : float or None
        Log prior density at the optimum.
    par : float or None
        Optimal scaling parameter alpha.
    eta : np.ndarray or None
        Weighted correlations.
    weights : np.ndarray or None
        Final normalized weights.
    """

    def __init__(
        self,
        call: Optional[str],
        y: Optional[np.ndarray],
        converged: Optional[bool],
        iterations: Optional[int],
        sumw0: Optional[float],
        par: Optional[float],
        log_el: Optional[float],
        log_p_eta: Optional[float],
        eta: Optional[np.ndarray],
        weights: Optional[np.ndarray],
    ):
        self.call = call
        self.converged = converged
        self.iterations = iterations
        self.sumw0 = sumw0
        self.par = par
        self.log_el = log_el
        self.log_p_eta = log_p_eta
        self.eta = eta
        self.weights = weights

        # Derived sample info.
        # npCBPS supports both binary and continuous treatments (see npcbps_fit:
        # any numeric treatment is routed through the continuous path). The
        # ``Treatment/Control group`` split is only meaningful when the treatment
        # is 0/1 binary; for continuous treatments we record the range instead.
        self.treatment_type: str = "unknown"
        self.n_treat: Optional[int] = None
        self.n_control: Optional[int] = None
        self.treatment_range: Optional[Tuple[float, float]] = None
        self.treatment_mean: Optional[float] = None
        if y is not None:
            y_arr = np.asarray(y)
            self.n = int(y_arr.shape[0])
            unique_vals = np.unique(y_arr[~np.isnan(y_arr)]) if y_arr.dtype.kind in "fc" else np.unique(y_arr)
            is_binary = (
                self.n > 0
                and unique_vals.size <= 2
                and np.all(np.isin(unique_vals, [0, 1]))
            )
            if is_binary:
                self.treatment_type = "binary"
                self.n_treat = int(y_arr.sum())
                self.n_control = self.n - self.n_treat
            else:
                self.treatment_type = "continuous"
                self.treatment_range = (float(np.nanmin(y_arr)), float(np.nanmax(y_arr)))
                self.treatment_mean = float(np.nanmean(y_arr))
        else:
            self.n = 0

    def __str__(self) -> str:
        """Return formatted summary text.

        The output is identical to the legacy ``NPCBPSResults.summary()``
        string for backward compatibility.
        """
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("npCBPS: Nonparametric Covariate Balancing Propensity Score")
        lines.append("=" * 70)
        lines.append("")

        # Call
        lines.append("Call:")
        lines.append(f"  {self.call or 'npCBPS()'}")
        lines.append("")

        # Sample information
        if self.n > 0:
            lines.append(f"Sample size: {self.n}")
            if self.treatment_type == "binary" and self.n_treat is not None:
                lines.append(
                    f"  Treatment group: {self.n_treat} ({100*self.n_treat/self.n:.1f}%)"
                )
                lines.append(
                    f"  Control group:   {self.n_control} ({100*self.n_control/self.n:.1f}%)"
                )
            elif self.treatment_type == "continuous" and self.treatment_range is not None:
                lo, hi = self.treatment_range
                mean = self.treatment_mean
                lines.append(f"  Treatment: continuous")
                lines.append(
                    f"  Range: [{lo:.4f}, {hi:.4f}]  Mean: {mean:.4f}"
                )
            lines.append("")

        # Convergence diagnostics
        lines.append("Convergence Diagnostics:")
        lines.append("-" * 70)
        if self.converged is not None:
            conv_status = "✓ Yes" if self.converged else "✗ No"
            lines.append(f"  Converged: {conv_status}")

        if self.iterations is not None:
            lines.append(f"  Iterations used: {self.iterations}")

        if self.sumw0 is not None:
            # sumw0 should be approximately 1.0 (key diagnostic from Fong et al. 2018)
            deviation = abs(self.sumw0 - 1.0)
            if deviation < 0.01:
                status = "✓ Excellent (within 1%)"
            elif deviation < 0.05:
                status = "✓ Good (within 5%)"
            elif deviation < 0.10:
                status = "⚠ Acceptable (within 10%)"
            else:
                status = "✗ Warning (>10% deviation)"

            lines.append(f"  Sum of weights (sumw0): {self.sumw0:.6f} {status}")
            lines.append("    Theoretical value: 1.0")
            lines.append(f"    Deviation: {deviation:.4f}")

        lines.append("")

        # Optimization results
        lines.append("Optimization Results:")
        lines.append("-" * 70)

        if self.par is not None:
            lines.append(f"  Optimization parameter (alpha): {self.par:.6f}")

        if self.log_el is not None:
            lines.append(f"  Log Empirical Likelihood: {self.log_el:.6f}")

        if self.log_p_eta is not None:
            lines.append(f"  Log Prior Density p(η): {self.log_p_eta:.6f}")

        if self.log_el is not None and self.log_p_eta is not None:
            total_obj = self.log_el + self.log_p_eta
            lines.append(f"  Total objective: {total_obj:.6f}")

        lines.append("")

        # Weighted correlations (key statistics)
        if self.eta is not None:
            lines.append("Weighted Correlations (η):")
            lines.append("-" * 70)

            eta_array = np.atleast_1d(self.eta)
            if len(eta_array) == 1:
                lines.append(f"  η = {eta_array[0]:.6f}")
            else:
                lines.append(f"  Number of correlations: {len(eta_array)}")
                lines.append(f"  Mean: {eta_array.mean():.6f}")
                lines.append(f"  Range: [{eta_array.min():.6f}, {eta_array.max():.6f}]")
                if len(eta_array) <= 10:
                    lines.append("  Values:")
                    for i, val in enumerate(eta_array):
                        lines.append(f"    η[{i}] = {val:.6f}")

            lines.append("")

        # Weight statistics
        if self.weights is not None:
            lines.append("Weight Distribution:")
            lines.append("-" * 70)
            lines.append(f"  Min:    {self.weights.min():.6f}")
            lines.append(f"  Q1:     {np.percentile(self.weights, 25):.6f}")
            lines.append(f"  Median: {np.median(self.weights):.6f}")
            lines.append(f"  Mean:   {self.weights.mean():.6f}")
            lines.append(f"  Q3:     {np.percentile(self.weights, 75):.6f}")
            lines.append(f"  Max:    {self.weights.max():.6f}")
            lines.append(f"  Sum:    {self.weights.sum():.6f}")

            # Effective sample size
            ess = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
            lines.append(f"  Effective sample size: {ess:.1f}")
            if self.n > 0:
                efficiency = ess / self.n
                lines.append(f"  Efficiency: {100*efficiency:.1f}%")

            lines.append("")

        # Diagnostic recommendations
        lines.append("Diagnostics:")
        lines.append("-" * 70)

        diagnostics = []

        if self.converged is False:
            diagnostics.append("⚠ Optimization did not converge - results may be unreliable")

        if self.sumw0 is not None and abs(self.sumw0 - 1.0) > 0.10:
            diagnostics.append("⚠ sumw0 deviates >10% from 1.0 - check optimization quality")

        if self.weights is not None:
            # Check weight range
            weight_range = self.weights.max() / self.weights.min() if self.weights.min() > 0 else float('inf')
            if weight_range > 100:
                diagnostics.append(f"⚠ Large weight range ({weight_range:.1f}x) - may indicate overlap issues")

            # Check effective sample size
            if self.n > 0:
                ess = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
                efficiency = ess / self.n
                if efficiency < 0.5:
                    eff_pct = 100 * efficiency
                    diagnostics.append(
                        f"⚠ Low weighting efficiency ({eff_pct:.1f}%) - consider different corprior"
                    )

        if diagnostics:
            for diag in diagnostics:
                lines.append(f"  {diag}")
        else:
            lines.append("  ✓ All diagnostics passed")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"NPCBPSSummary(n={self.n}, converged={self.converged})"


class NPCBPSResults:
    """
    Container for nonparametric CBPS estimation results.

    Stores the output from :func:`npCBPS`, including estimated weights,
    optimization diagnostics, and the original data. Unlike parametric
    CBPS, npCBPS does not estimate propensity score model coefficients.

    Attributes
    ----------
    weights : np.ndarray of shape (n,)
        Final normalized weights summing to n. Use these for weighted
        outcome analysis.
    sumw0 : float
        Sum of unnormalized weights before normalization. Should be close
        to 1.0 (within 5%). Values far from 1 indicate potential convergence
        issues.
    eta : np.ndarray of shape (K,)
        Optimal weighted correlation :math:`\\eta = \\alpha \\cdot \\eta_0`,
        where K is the number of covariates. Measures the remaining
        covariate-treatment association after weighting.
    par : float
        Optimal scaling parameter :math:`\\alpha \\in [0, 1]` from line search.
        Values near 0 indicate tight balance; values near 1 indicate relaxed
        balance.
    log_el : float
        Log empirical likelihood at the optimum.
    log_p_eta : float
        Log prior density :math:`\\log f(\\eta)` at the optimum.
    y : np.ndarray of shape (n,)
        Treatment variable.
    x : np.ndarray of shape (n, K)
        Original covariate matrix (before whitening).
    converged : bool
        Whether the optimization converged successfully.
    iterations : int or None
        Number of iterations used in the optimization.
    formula : str
        Model formula used for fitting.
    data : pd.DataFrame
        Original input DataFrame.
    call : str
        String representation of the function call.
    terms : object
        patsy DesignInfo object for formula parsing (used by diagnostics).
    na_action : dict or None
        Information about missing value handling.

    See Also
    --------
    npCBPS : Function that creates this results object.
    CBPSResults : Results container for parametric CBPS.
    """

    def __init__(self):
        # npCBPS-specific fields
        self.par: Optional[float] = None
        self.log_p_eta: Optional[float] = None
        self.log_el: Optional[float] = None
        self.eta: Optional[np.ndarray] = None
        self.sumw0: Optional[float] = None

        # Common fields (shared with CBPS)
        self.weights: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None

        # Convergence diagnostic fields (documented fields)
        self.converged: Optional[bool] = None
        self.iterations: Optional[int] = None

        # Metadata
        self.call: Optional[str] = None
        self.formula: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None

        # Metadata attributes for compatibility with predict() and model diagnostics
        self.terms: Optional[object] = None  # patsy DesignInfo object
        self.na_action: Optional[dict] = None  # Missing value handling information

    def __repr__(self) -> str:
        """Concise repr output (for interactive environment)"""
        n = len(self.y) if self.y is not None else 0
        converged_str = "Yes" if self.converged else "No" if self.converged is not None else "Unknown"
        sumw0_str = f"{self.sumw0:.4f}" if self.sumw0 is not None else "N/A"
        return f"NPCBPSResults(n={n}, converged={converged_str}, sumw0={sumw0_str})"

    def __str__(self) -> str:
        """Complete string output (for print calls)"""
        output = "\nCall:\n  " + (self.call or "npCBPS()") + "\n\n"

        # Sample information
        if self.y is not None:
            output += f"Sample size: {len(self.y)}\n"

        # Convergence information
        if self.converged is not None:
            output += f"Converged: {'Yes' if self.converged else 'No'}\n"
        if self.iterations is not None:
            output += f"Iterations: {self.iterations}\n"

        # Key statistics
        if self.sumw0 is not None:
            sumw0_status = "Good" if abs(self.sumw0 - 1.0) < 0.05 else "Check"
            output += f"Sum of weights (sumw0): {self.sumw0:.6f} ({sumw0_status}: should ≈ 1.0 ± 5%)\n"

        if self.log_el is not None:
            output += f"Log Empirical Likelihood: {self.log_el:.6f}\n"

        if self.log_p_eta is not None:
            output += f"Log Prior Density: {self.log_p_eta:.6f}\n"

        if self.par is not None:
            output += f"Optimization parameter (alpha): {self.par:.6f}\n"

        # Weight information
        if self.weights is not None:
            output += "\nWeights:\n"
            output += f"  Min: {self.weights.min():.6f}\n"
            output += f"  Max: {self.weights.max():.6f}\n"
            output += f"  Mean: {self.weights.mean():.6f}\n"
            output += f"  Sum: {self.weights.sum():.6f}\n"

        return output

    def summary(self) -> 'NPCBPSSummary':
        """
        Display npCBPS fit summary (detailed diagnostic information).

        This method provides comprehensive diagnostic information for npCBPS results.

        Returns
        -------
        NPCBPSSummary
            Summary object with ``__str__`` method for formatted output.
            Use ``print(result.summary())`` to display.

        Notes
        -----
        summary() provides more detailed diagnostic information than __str__(), including:
        - Convergence diagnostics (whether sumw0 ≈ 1)
        - Optimization parameters
        - Empirical likelihood and priors
        - Weighted correlation eta (if multiple covariates)
        - Weight distribution statistics

        This provides a comprehensive view of the nonparametric CBPS estimation results.

        Examples
        --------
        >>> from cbps import npCBPS
        >>> fit = npCBPS('treat ~ x1 + x2', data=df)
        >>> print(fit.summary())

        References
        ----------
        .. [1] Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing
               propensity score for a continuous treatment. The Annals of Applied
               Statistics, 12(1), 156-177. https://doi.org/10.1214/17-AOAS1101
        """
        return NPCBPSSummary(
            call=self.call,
            y=self.y,
            converged=self.converged,
            iterations=self.iterations,
            sumw0=self.sumw0,
            par=self.par,
            log_el=self.log_el,
            log_p_eta=self.log_p_eta,
            eta=self.eta,
            weights=self.weights,
        )

    def balance(self, **kwargs):
        """
        Compute covariate balance statistics (convenience method).

        This method wraps the standalone balance() function to provide
        a convenient object-oriented interface.

        The balance function routes to the appropriate method based on treatment type:
        - For continuous treatments: computes weighted correlations
        - For discrete treatments: computes standardized mean differences

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the balance() function.
            Supported keys: enhanced (bool), threshold (float), covariate_names (list).

        Returns
        -------
        dict
            Dictionary containing balance statistics:
            - balanced: weighted covariate balance statistics
            - original/unweighted: unweighted baseline statistics
            See cbps.balance() documentation for details

        Notes
        -----
        This method maintains API consistency with CBPSResults.balance()
        and CBMSMResults.balance().

        Implementation:
        - Calls the standalone balance(self) function
        - The global balance() function supports NPCBPSResults objects
        - Automatically routes to the appropriate balance function based on treatment type

        Examples
        --------
        >>> fit = npCBPS('treat ~ age + educ', data=df)
        >>>
        >>> # Method 1: Standalone function
        >>> from cbps import balance
        >>> bal = balance(fit)
        >>>
        >>> # Method 2: Object method
        >>> bal = fit.balance()
        >>>
        >>> # Both methods return identical results
        """
        # Import standalone function (avoid circular import)
        from cbps import balance as balance_func

        # Call the global balance() function directly
        return balance_func(self, **kwargs)

    def vcov(self):
        """
        Return the variance-covariance matrix of coefficients.

        Notes
        -----
        npCBPS is a nonparametric method that does not estimate parametric
        model coefficients, and therefore has no variance-covariance matrix.
        This method raises a ValueError to inform users of this limitation.

        Raises
        ------
        ValueError
            Always raised because npCBPS does not estimate coefficients.
        """
        raise ValueError(
            "npCBPS is a nonparametric method and does not estimate "
            "coefficients or their variance-covariance matrix."
        )




def npCBPS(
    formula: str,
    data: pd.DataFrame,
    na_action: Optional[str] = None,
    corprior: Optional[float] = None,
    print_level: int = 0,
    seed: Optional[int] = None,
    **kwargs: Any
) -> NPCBPSResults:
    """
    Estimate nonparametric covariate balancing weights.

    Implements the nonparametric CBGPS estimator from Section 3.3 of Fong,
    Hazlett, and Imai (2018). This method estimates inverse probability
    weights directly via empirical likelihood without specifying a
    parametric propensity score model.

    Parameters
    ----------
    formula : str
        Model formula specifying the treatment and covariates, e.g.,
        ``'treat ~ age + educ + income'``. The left-hand side is the
        treatment variable; the right-hand side lists covariates.
    data : pd.DataFrame
        DataFrame containing all variables referenced in the formula.
    na_action : {'warn', 'fail', 'ignore'}, optional
        How to handle missing values:

        - ``'warn'`` (default): Drop rows with missing values and warn.
        - ``'fail'``: Raise ValueError if missing values are present.
        - ``'ignore'``: Silently drop rows with missing values.

    corprior : float, optional
        Prior standard deviation :math:`\\sigma` for the allowed weighted
        correlation :math:`\\eta \\sim N(0, \\sigma^2 I_K)`. Controls the
        tradeoff between exact balance and stable weights.

        If ``None`` (default), set to ``0.1/n`` following the paper's
        recommendation in Section 3.3.4.

        Interpretation:

        - Smaller values enforce tighter balance but may produce extreme
          weights or fail to converge.
        - Larger values allow more imbalance but ensure convergence.
        - The default ``0.1/n`` generally provides good balance while
          ensuring convergence.

    print_level : int, default=0
        Verbosity level. If > 0, prints optimization diagnostics including
        ``log_post``, ``log_el``, ``log_p_eta``, and ``sumw0``.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Reserved for future extensions.

    Returns
    -------
    NPCBPSResults
        Fitted result object with attributes:

        - **weights**: Final weights normalized to sum to n.
        - **sumw0**: Sum of unnormalized weights (should be close to 1).
        - **eta**: Optimal weighted correlations.
        - **par**: Optimal scaling parameter :math:`\\alpha`.
        - **log_el**: Log empirical likelihood.
        - **log_p_eta**: Log prior density.
        - **converged**: Whether optimization converged.
        - **y**, **x**: Treatment and covariate arrays.
        - **formula**, **data**, **call**: Metadata.

    Raises
    ------
    ValueError
        If ``na_action='fail'`` and missing values are present, or if
        ``corprior`` is outside [0, 10].
    RuntimeError
        If optimization produces NaN weights.

    Notes
    -----
    **Algorithm (Section 3.3 of Fong et al., 2018):**

    1. Parse formula and extract treatment :math:`T` and covariates :math:`X`.
    2. Whiten covariates via Cholesky decomposition (Section 3.1).
    3. Construct constraint matrix :math:`g = (X^* T^*, X^*, T^*)^T`.
    4. Line search over :math:`\\alpha \\in [0, 1]` to maximize the penalized
       likelihood (Equation 10).
    5. Recover weights :math:`w_i = 1/(1 - \\gamma^T(g_i - \\eta))`.
    6. Normalize weights so that :math:`\\sum w_i = n`.

    **Non-convexity (Section 3.3.2):**

    The empirical likelihood objective is not generally convex, so there is
    no guarantee of finding the global optimum. Results may vary slightly
    between runs, which is expected behavior.

    **Convergence diagnostic:**

    The key diagnostic is ``sumw0``, the sum of unnormalized weights.
    Values within 5% of 1.0 indicate successful convergence. Large
    deviations suggest adjusting the ``corprior`` parameter.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing
    propensity score for a continuous treatment: Application to the
    efficacy of political advertisements. The Annals of Applied
    Statistics, 12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

    Examples
    --------
    Basic usage with the LaLonde dataset:

    >>> import pandas as pd
    >>> from cbps import npCBPS
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde()
    >>> fit = npCBPS('treat ~ age + educ + black + hisp + married + nodegr',
    ...              data=df)
    >>> print(f"Sum of weights: {fit.weights.sum():.1f}")
    >>> print(f"sumw0 (should be ~1): {fit.sumw0:.4f}")

    Adjusting the balance-variance tradeoff:

    >>> # Tighter balance (may not converge for all datasets)
    >>> fit_tight = npCBPS('treat ~ age + educ', data=df, corprior=0.001)
    >>> # Looser balance (ensures convergence)
    >>> fit_loose = npCBPS('treat ~ age + educ', data=df, corprior=0.1)
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Handle na_action parameter
    # Set default value
    if na_action is None:
        na_action = 'warn'  # Default: warn and drop missing values

    # Validate na_action parameter value
    valid_na_actions = {'warn', 'fail', 'ignore'}
    if na_action not in valid_na_actions:
        raise ValueError(
            f"Invalid na_action='{na_action}'. "
            f"Valid options are: {', '.join(repr(x) for x in valid_na_actions)}. "
            f"Note: use 'warn' (not 'drop') to remove missing values with a warning."
        )

    # Missing value handling (before formula parsing)
    # Extract columns involved in formula
    treat_col = formula.split('~')[0].strip()
    covar_part = formula.split('~')[1]
    # Simple variable name extraction (handles basic formulas, complex ones handled by patsy)
    import re
    covar_cols = re.findall(r'\b[a-zA-Z_]\w*\b', covar_part)
    relevant_cols = [treat_col] + [col for col in covar_cols if col in data.columns]

    # Check for missing values
    n_missing = data[relevant_cols].isna().any(axis=1).sum()
    na_action_info = None
    if n_missing > 0:
        if na_action == 'fail':
            raise ValueError(
                f"npCBPS: Missing values detected in {n_missing} observations. "
                f"Set na_action='warn' to remove them, or handle missing values before calling npCBPS()."
            )
        elif na_action == 'warn':
            import warnings
            data_clean = data.dropna(subset=relevant_cols)
            n_dropped = len(data) - len(data_clean)
            warnings.warn(
                f"npCBPS: Removed {n_dropped} observations with missing values. "
                f"Remaining sample size: {len(data_clean)}.",
                UserWarning
            )
            data = data_clean
            na_action_info = {'method': 'warn', 'n_dropped': n_dropped}
        elif na_action == 'ignore':
            # Ignore mode: silently drop missing values
            data_clean = data.dropna(subset=relevant_cols)
            n_dropped = len(data) - len(data_clean)
            data = data_clean
            na_action_info = {'method': 'ignore', 'n_dropped': n_dropped}

    # Sample-size adaptive corprior default value
    # Paper recommendation: ρ = 0.1/N (Fong et al. 2018, Section 3.3.4)
    n_obs = len(data)
    if corprior is None:
        corprior = 0.1 / n_obs
        if print_level > 0:
            print(f"npCBPS: Using paper-recommended corprior = 0.1/n = {corprior:.6f}")

    # Input validation
    # Validate corprior range (based on paper recommendation and experience)
    # Allow corprior=0 but issue a warning
    if not (0.0 <= corprior <= 10.0):
        raise ValueError(
            f"corprior={corprior} is outside the valid range [0.0, 10.0]. "
            f"The paper recommends corprior ≈ 0.1/n (for this dataset: {0.1/n_obs:.6f}). "
            f"Values >>1 often lead to NaN weights."
        )

    # Special warning for corprior=0
    if corprior == 0.0:
        import warnings
        warnings.warn(
            "corprior=0 removes all correlation prior penalty, which may lead to "
            "unstable or extreme weights in small samples. "
            "The paper recommends corprior ≈ 0.1/n for most applications. "
            "Use corprior=0 only for specific purposes like sensitivity analysis.",
            UserWarning
        )

    # Validate sample size (avoid numerical instability with small samples)
    if n_obs < 30:
        import warnings
        warnings.warn(
            f"Small sample size (n={n_obs}). npCBPS may be unstable with n<30. "
            f"Consider using standard CBPS for small samples.",
            UserWarning
        )

    # Formula parsing
    # parse_formula returns (y, X), where y is the treatment vector and X is the covariate matrix (with intercept)
    # preserve_categorical=True to maintain factor semantics
    # Also extract terms object for model diagnostics
    from patsy import dmatrices

    # Save original data for metadata
    data_original = data.copy()

    # Use dmatrices to parse and obtain terms information
    _, X_design = dmatrices(formula, data, return_type='dataframe')
    terms_obj = X_design.design_info  # patsy DesignInfo object

    treat, X_mat = parse_formula(formula, data, preserve_categorical=True)

    # Remove zero-variance columns
    non_zero_var_cols = X_mat.std(axis=0) > 0
    X_mat = X_mat[:, non_zero_var_cols]

    # Call core fitting function
    fit_result = npCBPS_fit(
        treat=treat,
        X=X_mat,
        corprior=corprior,
        print_level=print_level
    )

    # Append metadata
    fit_result.call = f"npCBPS(formula={formula}, data=..., corprior={corprior})"
    fit_result.formula = formula
    fit_result.data = data_original  # Save original data

    # Add terms and na_action for model diagnostics
    fit_result.terms = terms_obj
    fit_result.na_action = na_action_info

    return fit_result


def npCBPS_fit(
    treat: Union[np.ndarray, pd.Series],
    X: np.ndarray,
    corprior: float,
    print_level: int
) -> NPCBPSResults:
    """
    Core fitting procedure for nonparametric CBPS.

    This is the internal implementation called by :func:`npCBPS` after
    formula parsing and data preprocessing. It performs the empirical
    likelihood optimization to estimate covariate balancing weights.

    Parameters
    ----------
    treat : np.ndarray or pd.Series of shape (n,)
        Treatment variable. If a pandas Categorical Series, it is treated
        as a factor with J levels. Otherwise, it is treated as continuous.
    X : np.ndarray of shape (n, K)
        Covariate matrix with zero-variance columns removed.
    corprior : float
        Prior standard deviation :math:`\\sigma` for the allowed weighted
        correlation (see Section 3.3.4 of Fong et al., 2018).
    print_level : int
        Verbosity level for diagnostic output.

    Returns
    -------
    NPCBPSResults
        Fitted result object. Note that ``formula``, ``data``, ``call``,
        ``terms``, and ``na_action`` attributes are populated by the caller.

    Notes
    -----
    **Treatment types:**

    - *Continuous*: Constraint matrix has K correlation constraints.
    - *Factor (J levels)*: Constraint matrix has K*(J-1) correlation
      constraints using one-hot encoding.

    **Implementation details:**

    - Covariates are whitened using :func:`cholesky_whitening`.
    - The treatment is standardized to zero mean and unit variance.
    - The line search is bounded to :math:`\\alpha \\in [0, 1]`.

    See Also
    --------
    npCBPS : High-level interface with formula parsing.
    """
    # Initialization
    # Detect if treatment is a Categorical factor
    is_factor_treat = isinstance(treat, pd.Series) and isinstance(treat.dtype, pd.CategoricalDtype)

    if is_factor_treat:
        # Factor treatment: preserve original Series, convert later
        _D_original = treat.copy()  # Reserved for future use
        D = treat.cat.codes.values.astype(np.float64)
    else:
        # Continuous treatment: direct copy
        D = treat.copy() if isinstance(treat, np.ndarray) else treat.values.copy()
        _D_original = None  # noqa: F841

    rescale_orig = True
    orig_X = X.copy()

    # Preprocessing: Cholesky whitening
    X = cholesky_whitening(X, verify=True)

    n = X.shape[0]

    # Compute epsilon (numerical tolerance)
    eps = 1.0 / n

    # Construct constraint matrix z
    # Determine treatment type:
    # - pd.Categorical -> factor treatment
    # - Numeric types (including binary 0/1) -> continuous treatment
    #
    # Note: Numeric treatment variables (including 0/1 binary values) use
    # the continuous treatment path for consistency.

    if not is_factor_treat:
        # Continuous treatment path
        if print_level > 0:
            print("Estimating npCBPS as a continuous treatment.")

        # Redirect X to ensure positive correlation with T
        correlations = np.array([np.corrcoef(X[:, j], D)[0, 1] for j in range(X.shape[1])])
        signs = np.sign(correlations)
        X = X @ np.diag(signs)

        # Standardize treatment
        D = (D - D.mean()) / D.std(ddof=1)

        # Construct constraint matrix: z = cbind(X*D, X, D)
        X_times_D = X * D[:, None]  # Element-wise multiplication, broadcast D
        D_col = D[:, None]  # Convert to column vector
        z = np.column_stack([X_times_D, X, D_col])

        _ncon = z.shape[1]  # Total constraints (reserved for diagnostics)  # noqa: F841
        ncon_cor = X.shape[1]  # K

        # cor_init only used for factor treatment
        cor_init = None

    else:
        # Factor treatment path
        if print_level > 0:
            print("Estimating npCBPS as a factor treatment.")

        # Convert to one-hot encoding
        unique_levels = np.unique(D)
        conds = len(unique_levels)
        Td = np.zeros((n, conds))
        for i, level in enumerate(unique_levels):
            Td[:, i] = (D == level).astype(float)

        dimX = X.shape[1]

        # Normalize each column
        colsums = Td.sum(axis=0)
        Td = Td @ np.diag(1 / colsums)

        # Subtract last column and remove it
        subtract_mat = Td[:, -1:] @ np.ones((1, conds))
        Td = Td - subtract_mat
        Td = Td[:, :-1]

        # Center and scale
        Td = (Td - Td.mean(axis=0)) / Td.std(axis=0, ddof=1)

        # Construct z using Kronecker product
        z_list = []
        for i in range(n):
            kron_prod = np.kron(Td[i, :], X[i, :])
            z_list.append(kron_prod)
        z = np.array(z_list)

        # Compute cor_init for eta initialization
        # For each column of X, compute correlations with all columns of Td
        cor_init_list = []
        for j in range(dimX):
            cors_with_Td = np.array([np.corrcoef(Td[:, i], X[:, j])[0, 1] for i in range(Td.shape[1])])
            cor_init_list.append(cors_with_Td)
        # Transpose and flatten: stack into matrix (dimX, conds-1), transpose to (conds-1, dimX), then flatten
        cor_init_matrix = np.array(cor_init_list)  # shape: (dimX, conds-1)
        cor_init = cor_init_matrix.T.ravel()  # Transpose and flatten, shape: (dimX*(conds-1),)

        # Add mean constraints
        ncon_cor = z.shape[1]  # Record number of correlation constraints
        z = np.column_stack([z, X])
        _ncon = z.shape[1]  # Total constraints (reserved for diagnostics)  # noqa: F841

    # Optimization preparation
    # Prior standard deviation
    # eta_prior_sd = corprior (standard deviation, not variance)
    eta_prior_sd = np.full(ncon_cor, corprior)

    # Initialize eta
    if not is_factor_treat:
        # Continuous treatment: eta_init = cor(X, D)
        # Note: D is already standardized, X is already whitened
        eta_init = np.array([np.corrcoef(X[:, j], D)[0, 1] for j in range(X.shape[1])])
    else:
        # Factor treatment: use cor_init computed above
        eta_init = cor_init

    # Eta scaling vector
    if rescale_orig:
        eta_to_be_scaled = eta_init
    else:
        eta_to_be_scaled = np.ones(ncon_cor)

    # Main optimization: line search over α ∈ [0, 1] (Fong et al. 2018, Equation 10)
    # The paper specifies α ∈ [0, 1] to ensure algorithm stability

    # Define wrapper function for maximization
    def objective_for_maximize(par_scalar):
        return log_post(par_scalar, eta_to_be_scaled, eta_prior_sd, z, eps, 0.001, ncon_cor, n)

    # Maximize log_post using bounded scalar optimization
    result = scipy.optimize.minimize_scalar(
        lambda par: -objective_for_maximize(par),
        bounds=(0, 1),
        method='bounded',
        options={'xatol': 1e-10, 'maxiter': 2000}
    )
    par_opt = result.x

    # Print warning if optimization did not converge
    if not result.success:
        if print_level > 0:
            print(f"Warning: optimization may not have converged: {result.message}")

    # Compute optimal eta
    eta_opt = par_opt * eta_to_be_scaled

    # Compute optimal weights
    el_out_opt = get_w(eta_opt, z, 0.05, eps, ncon_cor, n)
    w_opt = el_out_opt['w']
    sumw0 = el_out_opt['sumw']

    # Weight normalization following theory (Fong et al. 2018, Equation 8)
    # The paper requires: Σw_i = N (constraint in Equation 8)
    # Normalize weights so that sum(w) = n
    w = w_opt * n / sumw0

    # Check for NaN weights and raise meaningful error
    if np.isnan(w).any() or np.isnan(sumw0):
        raise RuntimeError(
            f"npCBPS optimization failed and produced NaN weights. "
            f"This usually indicates:\n"
            f"  1. corprior is too large (current: {corprior}, try < 1.0)\n"
            f"  2. corprior is too small (current: {corprior}, try > 0.0001)\n"
            f"  3. Sample size is too small (current: n={n}, recommend n>=30)\n"
            f"  4. Covariate-treatment correlation is extreme\n"
            f"Suggestion: Try adjusting corprior or using standard CBPS instead."
        )

    # Compute log prior density
    log_p_eta_opt = np.sum(
        -0.5 * np.log(2 * np.pi * eta_prior_sd**2)
        - eta_opt**2 / (2 * eta_prior_sd**2)
    )
    log_el_opt = el_out_opt['log_el']

    # Construct result object
    result_obj = NPCBPSResults()
    result_obj.par = par_opt
    result_obj.log_p_eta = log_p_eta_opt
    result_obj.log_el = log_el_opt
    result_obj.eta = eta_opt
    result_obj.sumw0 = sumw0
    result_obj.weights = w
    result_obj.y = treat  # Original treatment variable
    result_obj.x = orig_X

    # Convergence diagnostic fields
    result_obj.converged = result.success
    result_obj.iterations = result.nit if hasattr(result, 'nit') else None

    # Diagnostic output (optional)
    if print_level > 0:
        print(f"par: {par_opt:.6f}")
        print(f"log_post: {-(log_el_opt + log_p_eta_opt):.6f}")
        print(f"log_el: {log_el_opt:.6f}")
        print(f"log_p_eta: {log_p_eta_opt:.6f}")
        print(f"sumw0: {sumw0:.6f}")
        print(f"converged: {result_obj.converged}")
        if result_obj.iterations is not None:
            print(f"iterations: {result_obj.iterations}")

    return result_obj
