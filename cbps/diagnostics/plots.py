"""
Covariate Balance Visualization
===============================

Visualization functions for assessing covariate balance before and after
propensity score weighting. Requires matplotlib (optional dependency).

For binary and multi-valued treatments, plots display standardized mean
differences (SMD) across treatment contrasts. For continuous treatments,
plots display Pearson correlations between covariates and the treatment.

Functions
---------
plot_cbps
    Balance plots for binary/multi-valued treatments.

plot_cbps_continuous
    Correlation plots for continuous treatments.

plot_cbmsm
    Balance plots for marginal structural models.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1),
156-177.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# matplotlib as optional dependency
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .balance import balance_cbps, balance_cbps_continuous

# Import Results classes from CBMSM and npCBPS modules for type checking
try:
    from cbps.msm.cbmsm import CBMSMResults
except ImportError:
    CBMSMResults = None

try:
    from cbps.nonparametric.npcbps import NPCBPSResults
except ImportError:
    NPCBPSResults = None


def _compute_boxplot_stats_tukey(data):
    """
    Compute boxplot statistics using Tukey's hinges method.

    Parameters
    ----------
    data : array-like
        1-dimensional array of numeric data.

    Returns
    -------
    dict
        Boxplot statistics compatible with matplotlib's bxp() function:
        whislo, q1, med, q3, whishi.

    Notes
    -----
    Uses Tukey's five-number summary where hinges are medians of each
    half of the data, which may differ slightly from standard quantiles.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Compute median and hinges using Tukey's fivenum algorithm
    if n % 2 == 0:
        # Even number of data points
        m = n // 2
        median = (sorted_data[m-1] + sorted_data[m]) / 2
        # Lower half: indices 0 to m-1
        # Upper half: indices m to n-1
        lower_half = sorted_data[:m]
        upper_half = sorted_data[m:]
    else:
        # Odd number of data points
        m = n // 2
        median = sorted_data[m]
        # Include median in both halves per Tukey's method
        lower_half = sorted_data[:m+1]
        upper_half = sorted_data[m:]

    # Hinges are medians of each half
    q1 = np.median(lower_half)
    q3 = np.median(upper_half)

    # Whisker range (default multiplier = 1.5)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # Whisker endpoints: most extreme values within fences
    whislo = np.min(sorted_data[sorted_data >= lower_fence])
    whishi = np.max(sorted_data[sorted_data <= upper_fence])

    return {
        'whislo': whislo,  # Lower whisker endpoint
        'q1': q1,          # Lower hinge (box bottom)
        'med': median,     # Median
        'q3': q3,          # Upper hinge (box top)
        'whishi': whishi   # Upper whisker endpoint
    }


def plot_cbps(cbps_obj: Dict[str, Any],
              covars: Optional[List[int]] = None,
              silent: bool = True,
              boxplot: bool = False,
              **kwargs) -> Optional[pd.DataFrame]:
    """
    Visualize covariate balance for binary or multi-valued treatments.

    Creates a two-panel figure showing absolute standardized mean differences
    (SMD) before and after CBPS weighting. Points closer to zero indicate
    better balance.

    Parameters
    ----------
    cbps_obj : CBPSResults or dict
        Fitted CBPS object containing weights, covariates (x), and treatment (y).
    covars : list of int, optional
        Indices of covariates to plot (0-based, excluding intercept).
        Default plots all covariates.
    silent : bool, default=True
        If False, returns a DataFrame with balance statistics.
    boxplot : bool, default=False
        If True, displays boxplots instead of scatter plots.
    **kwargs
        Additional arguments passed to matplotlib scatter() or bxp().

    Returns
    -------
    pd.DataFrame or None
        If silent=False, returns DataFrame with columns: contrast, covariate,
        balanced (SMD after weighting), original (SMD before weighting).

    Notes
    -----
    The number of contrasts equals C(k,2) for k treatment levels:

    - Binary (k=2): 1 contrast
    - Three-valued (k=3): 3 contrasts
    - Four-valued (k=4): 6 contrasts

    Following Austin (2009), SMD < 0.1 indicates acceptable balance.

    Examples
    --------
    >>> import cbps
    >>> from cbps.datasets import load_lalonde
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>> fit = cbps.CBPS('treat ~ age + educ + re74', data=df, att=1)
    >>> cbps.plot_cbps(fit, silent=True)  # Display plot
    >>> balance_df = cbps.plot_cbps(fit, silent=False)  # Get data
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib>=3.3.0"
        )

    # Detect common parameter misuse and provide friendly error message
    # Users familiar with pandas/seaborn may try kind='boxplot', but this function uses boxplot=True
    if 'kind' in kwargs:
        kind_value = kwargs.pop('kind')  # Remove 'kind' to avoid passing to matplotlib
        if kind_value == 'boxplot' or kind_value == 'box':
            raise TypeError(
                f"plot_cbps() does not accept 'kind' parameter.\n"
                f"To plot boxplot, use: plot_cbps(cbps_obj, boxplot=True)\n"
                f"To plot scatter (default), use: plot_cbps(cbps_obj, boxplot=False)"
            )
        else:
            raise TypeError(
                f"plot_cbps() got unexpected keyword argument 'kind'.\n"
                f"Valid plotting options:\n"
                f"  - boxplot=True: Draw boxplot\n"
                f"  - boxplot=False: Draw scatter plot (default)\n"
                f"Matplotlib scatter/bxp parameters can be passed via **kwargs."
            )

    # Convert CBPSResults or NPCBPSResults to dict if needed
    from cbps.core.results import CBPSResults
    from cbps.nonparametric.npcbps import NPCBPSResults

    if isinstance(cbps_obj, CBPSResults):
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'fitted_values': cbps_obj.fitted_values
        }
    elif isinstance(cbps_obj, NPCBPSResults):
        # npCBPS result object
        # Route to appropriate plot function based on treatment type
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'log_el': cbps_obj.log_el,  # Marker for npCBPS detection
        }
        # Detect continuous treatment based on data type and unique values
        # Continuous: floating type AND many unique values (> 10)
        # Discrete: few unique values (<= 10) regardless of dtype
        n_unique = len(np.unique(cbps_obj.y))
        is_continuous = np.issubdtype(cbps_obj.y.dtype, np.floating) and n_unique > 10
        
        if is_continuous:
            # Route to continuous treatment plot function
            return plot_cbps_continuous(cbps_obj, covars=covars, silent=silent, **kwargs)
        # Otherwise continue with discrete treatment path
    else:
        cbps_dict = cbps_obj

    # Step 1: Compute balance statistics
    bal_x = balance_cbps(cbps_dict)
    
    # Step 2: Process covars parameter
    if covars is None:
        covars = list(range(bal_x["balanced"].shape[0]))
    
    # Step 3: Extract standardized means
    balanced_std_mean = bal_x["balanced"][covars, :]
    original_std_mean = bal_x["original"][covars, :]
    
    # Step 4: Calculate number of treatment levels and contrasts
    no_treats = bal_x["balanced"].shape[1] // 2
    
    # Number of contrasts: C(k,2) = k*(k-1)/2 pairwise comparisons
    if no_treats == 2:
        no_contrasts = 1
    elif no_treats == 3:
        no_contrasts = 3
    else:
        no_contrasts = 6
    
    # Step 5: Initialize contrast matrices
    abs_mean_ori_contrasts = np.zeros((len(covars), no_contrasts), dtype=np.float64)
    abs_mean_bal_contrasts = np.zeros((len(covars), no_contrasts), dtype=np.float64)
    
    # Step 6: Prepare data collection lists
    contrast_names = []
    true_contrast_names = []
    
    # Get treatment levels and covariate names for labels (from y and balance results)
    treats = pd.Categorical(cbps_dict['y'])
    treat_levels = treats.categories

    # Get covariate names from balanced matrix
    X = cbps_dict['x']
    # Detect npCBPS (has log_el key) - no intercept column
    is_npcbps = 'log_el' in cbps_dict
    
    if is_npcbps:
        # npCBPS: X has no intercept, all columns are covariates
        covar_names = [f"X{i+1}" for i in range(X.shape[1])]
    else:
        # CBPS: X has intercept in column 0, skip it
        if X.shape[1] > 1:
            covar_names = [f"X{i}" for i in range(1, X.shape[1])]
        else:
            covar_names = ["X1"]
    
    # Use actual covars index subset
    rownames = [covar_names[i] for i in covars]
    
    # Step 7: Double loop to calculate absolute differences for all pairwise contrasts
    ctr = 0
    for i in range(no_treats - 1):
        for j in range(i + 1, no_treats):
            # Compute absolute difference for original data
            # Standardized mean columns are at indices i+no_treats and j+no_treats
            abs_mean_ori_contrasts[:, ctr] = np.abs(
                original_std_mean[:, i + no_treats] - 
                original_std_mean[:, j + no_treats]
            )
            
            # Compute absolute difference after weighting
            abs_mean_bal_contrasts[:, ctr] = np.abs(
                balanced_std_mean[:, i + no_treats] - 
                balanced_std_mean[:, j + no_treats]
            )
            
            # Record contrast names using 1-based display indexing
            contrast_names.append(f"{i+1}:{j+1}")
            true_contrast_names.append(f"{treat_levels[i]}:{treat_levels[j]}")
            
            ctr += 1
    
    # Step 7.5: Construct long-format data for DataFrame
    contrasts_list = []
    covar_list = []
    for contrast_name in true_contrast_names:
        contrasts_list.extend([contrast_name] * len(covars))
        covar_list.extend(rownames)
    
    # Step 8: Calculate xlim range
    max_abs_contrast = max(
        np.max(abs_mean_ori_contrasts), 
        np.max(abs_mean_bal_contrasts)
    )
    
    # Add margins for visual clarity (4% on each side)
    left_margin = -0.04 * max_abs_contrast
    right_margin = max_abs_contrast * 1.04
    
    # Step 9: Create plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    if not boxplot:
        # Scatter plot mode
        # Upper panel: Before Weighting
        ax1 = axes[0]
        ax1.set_xlim(left_margin, right_margin)  # Use range with margins
        ax1.set_ylim(0.5, no_contrasts + 0.5)
        ax1.set_xlabel("Absolute Difference of Standardized Means")
        ax1.set_ylabel("Contrasts")
        ax1.set_title("Before Weighting", fontweight='bold')
        ax1.set_yticks(range(1, no_contrasts + 1))
        ax1.set_yticklabels(contrast_names)
        
        # Plot points for each covariate at each contrast
        # Collect all point coordinates and draw at once (maintain same color)
        x_all_ori = []
        y_all_ori = []
        for i in range(no_contrasts):
            for j in range(len(covars)):
                x_all_ori.append(abs_mean_ori_contrasts[j, i])
                y_all_ori.append(i + 1)
        
        # Draw all points at once with default styling (hollow black circles)
        # Users can override via kwargs (e.g., c='red', marker='x')
        default_scatter_params = {
            'facecolors': 'none',  # hollow circle
            'edgecolors': 'black',
            's': 20  # default point size
        }
        # kwargs has higher priority, overrides defaults
        scatter_params = {**default_scatter_params, **kwargs}
        ax1.scatter(x_all_ori, y_all_ori, **scatter_params)
        
        # Lower panel: After Weighting
        ax2 = axes[1]
        ax2.set_xlim(left_margin, right_margin)  # Use range with margins
        ax2.set_ylim(0.5, no_contrasts + 0.5)
        ax2.set_xlabel("Absolute Difference of Standardized Means")
        ax2.set_ylabel("Contrasts")
        ax2.set_title("After Weighting", fontweight='bold')
        ax2.set_yticks(range(1, no_contrasts + 1))
        ax2.set_yticklabels(contrast_names)
        
        # Collect After Weighting points
        x_all_bal = []
        y_all_bal = []
        for i in range(no_contrasts):
            for j in range(len(covars)):
                x_all_bal.append(abs_mean_bal_contrasts[j, i])
                y_all_bal.append(i + 1)
        
        # Use same default parameters
        scatter_params = {**default_scatter_params, **kwargs}
        ax2.scatter(x_all_bal, y_all_bal, **scatter_params)
    
    else:
        # Boxplot mode using Tukey's hinges method
        # Python needs to manually compute hinges statistics, then use bxp() to draw
        
        # Upper panel: Before Weighting
        ax1 = axes[0]
        
        # Compute Tukey-style statistics for each contrast
        bxp_stats_ori = []
        for i in range(no_contrasts):
            data = abs_mean_ori_contrasts[:, i]
            stats = _compute_boxplot_stats_tukey(data)
            bxp_stats_ori.append({
                'whislo': stats['whislo'],
                'q1': stats['q1'],
                'med': stats['med'],
                'q3': stats['q3'],
                'whishi': stats['whishi'],
                'fliers': []  # No outliers
            })
        
        # Use bxp() to draw (passing pre-computed statistics)
        # bxp() supports: widths, patch_artist, boxprops, whiskerprops, capprops, medianprops
        # Example: plot_cbps(fit, boxplot=True, widths=0.8, boxprops=dict(facecolor='gray'))
        ax1.bxp(
            bxp_stats_ori,
            positions=range(1, no_contrasts + 1),
            vert=False,  # horizontal
            showmeans=False,
            showfliers=False,
            **kwargs  # Pass boxplot-related parameters
        )
        ax1.set_xlim(left_margin, right_margin)
        ax1.set_ylim(0.5, no_contrasts + 0.5)
        ax1.set_xlabel("Absolute Difference of Standardized Means")
        ax1.set_ylabel("Contrasts")
        ax1.set_title("Before Weighting", fontweight='bold')
        ax1.set_yticks(range(1, no_contrasts + 1))
        ax1.set_yticklabels(contrast_names)
        
        # Lower panel: After Weighting
        ax2 = axes[1]
        
        bxp_stats_bal = []
        for i in range(no_contrasts):
            data = abs_mean_bal_contrasts[:, i]
            stats = _compute_boxplot_stats_tukey(data)
            bxp_stats_bal.append({
                'whislo': stats['whislo'],
                'q1': stats['q1'],
                'med': stats['med'],
                'q3': stats['q3'],
                'whishi': stats['whishi'],
                'fliers': []
            })
        
        ax2.bxp(
            bxp_stats_bal,
            positions=range(1, no_contrasts + 1),
            vert=False,
            showmeans=False,
            showfliers=False,
            **kwargs  # Pass boxplot-related parameters
        )
        ax2.set_xlim(left_margin, right_margin)
        ax2.set_ylim(0.5, no_contrasts + 0.5)
        ax2.set_xlabel("Absolute Difference of Standardized Means")
        ax2.set_ylabel("Contrasts")
        ax2.set_title("After Weighting", fontweight='bold')
        ax2.set_yticks(range(1, no_contrasts + 1))
        ax2.set_yticklabels(contrast_names)
    
    plt.tight_layout()
    # Note: Do not call plt.show(), let caller decide whether to display/save
    
    # Step 10: Return DataFrame if requested
    if not silent:
        return pd.DataFrame({
            "contrast": contrasts_list,
            "covariate": covar_list,
            "balanced": abs_mean_bal_contrasts.ravel(order='F'),  # Column-major flatten
            "original": abs_mean_ori_contrasts.ravel(order='F')
        })
    
    return None


def plot_cbps_continuous(cbps_obj: Dict[str, Any],
                         covars: Optional[List[int]] = None,
                         silent: bool = True,
                         boxplot: bool = False,
                         **kwargs) -> Optional[pd.DataFrame]:
    """
    Visualize covariate balance for continuous treatments.

    Displays absolute Pearson correlations between covariates and the
    treatment variable before and after CBPS weighting. Correlations
    closer to zero indicate better balance.

    Parameters
    ----------
    cbps_obj : CBPSResults or dict
        Fitted continuous treatment CBPS object.
    covars : list of int, optional
        Indices of covariates to plot (0-based, excluding intercept).
        Default plots all covariates.
    silent : bool, default=True
        If False, returns a DataFrame with correlation statistics.
    boxplot : bool, default=False
        If True, displays boxplots instead of scatter plots.
    **kwargs
        Additional arguments passed to matplotlib scatter() or bxp().

    Returns
    -------
    pd.DataFrame or None
        If silent=False, returns DataFrame with columns: covariate,
        balanced (correlation after weighting), original (correlation before).

    Notes
    -----
    For continuous treatments, balance is assessed via weighted Pearson
    correlations. A correlation near zero indicates that the covariate
    is conditionally independent of the treatment given the weights.

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
    >>> cbps.plot_cbps_continuous(fit, silent=True)  # doctest: +SKIP
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib>=3.3.0"
        )

    # Convert CBPSResults or NPCBPSResults to dict if needed
    from cbps.core.results import CBPSResults
    from cbps.nonparametric.npcbps import NPCBPSResults

    if isinstance(cbps_obj, CBPSResults):
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'fitted_values': cbps_obj.fitted_values
        }
    elif isinstance(cbps_obj, NPCBPSResults):
        # npCBPS result object - include log_el to identify as npCBPS
        cbps_dict = {
            'weights': cbps_obj.weights,
            'x': cbps_obj.x,
            'y': cbps_obj.y,
            'log_el': cbps_obj.log_el,  # Marker for npCBPS detection
        }
    else:
        cbps_dict = cbps_obj

    # Step 1: Compute balance statistics
    bal_x = balance_cbps_continuous(cbps_dict)
    
    # Step 2: Process covars parameter
    if covars is None:
        covars = list(range(bal_x["balanced"].shape[0]))
    
    # Step 3: Extract absolute correlations
    balanced_abs_cor = np.abs(bal_x["balanced"][covars].ravel())
    original_abs_cor = np.abs(bal_x["unweighted"][covars].ravel())  # Read "unweighted" key
    
    # Step 4: Calculate xlim range
    max_abs_cor = max(np.max(original_abs_cor), np.max(balanced_abs_cor))
    
    # Step 5: Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    if not boxplot:
        # Scatter plot mode
        # Single figure with 2 rows of points
        ax.set_xlim(0, max_abs_cor)
        ax.set_ylim(1.5, 3.5)
        ax.set_xlabel("Absolute Pearson Correlation")
        ax.set_ylabel("")
        ax.set_yticks([2, 3])
        ax.set_yticklabels(["CBPS Weighted", "Unweighted"])
        
        # Draw points at two y-positions: unweighted (y=3) and weighted (y=2)
        # Use filled circles for continuous treatment
        default_scatter_params_cont = {
            'marker': 'o',  # Circle
            'c': 'black',
            's': 50
        }
        scatter_params = {**default_scatter_params_cont, **kwargs}
        
        # Draw original correlations at y=3 position (Unweighted)
        ax.scatter(
            x=original_abs_cor, 
            y=np.full(len(covars), 3), 
            **scatter_params
        )
        
        # Draw weighted correlations at y=2 position (CBPS Weighted)
        ax.scatter(
            x=balanced_abs_cor, 
            y=np.full(len(covars), 2), 
            **scatter_params
        )
    
    else:
        # Boxplot mode using Tukey's hinges statistics
        stats_balanced = _compute_boxplot_stats_tukey(balanced_abs_cor)
        stats_original = _compute_boxplot_stats_tukey(original_abs_cor)
        
        bxp_stats = [
            {  # position 1: CBPS Weighted
                'whislo': stats_balanced['whislo'],
                'q1': stats_balanced['q1'],
                'med': stats_balanced['med'],
                'q3': stats_balanced['q3'],
                'whishi': stats_balanced['whishi'],
                'fliers': []
            },
            {  # position 2: Unweighted
                'whislo': stats_original['whislo'],
                'q1': stats_original['q1'],
                'med': stats_original['med'],
                'q3': stats_original['q3'],
                'whishi': stats_original['whishi'],
                'fliers': []
            }
        ]
        
        ax.bxp(
            bxp_stats,
            positions=[1, 2],
            vert=False,
            showmeans=False,
            showfliers=False,
            **kwargs  # Pass boxplot parameters
        )
        ax.set_xlabel("Absolute Pearson Correlation")
        ax.set_ylabel("")
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["CBPS Weighted", "Unweighted"])
    
    plt.tight_layout()
    # Note: Do not call plt.show(), let caller decide whether to display/save
    
    # Step 6: Return DataFrame if requested
    if not silent:
        # Get covariate names
        if hasattr(bal_x["balanced"], 'index'):
            rownames = bal_x["balanced"].index[covars].tolist()
        else:
            rownames = [f"X{i+1}" for i in covars]
        
        return pd.DataFrame({
            "covariate": rownames,
            "balanced": balanced_abs_cor,
            "original": original_abs_cor  # Naming convention: unweighted -> original
        })
    
    return None


def plot_npcbps(npcbps_obj,
                covars: Optional[List[int]] = None,
                silent: bool = True,
                **kwargs) -> Optional[pd.DataFrame]:
    """
    Visualize covariate balance for nonparametric CBPS.

    Automatically selects the appropriate plotting method based on
    treatment type: plot_cbps for discrete treatments, plot_cbps_continuous
    for continuous treatments.

    Parameters
    ----------
    npcbps_obj : NPCBPSResults or dict
        Fitted npCBPS result object.
    covars : list of int, optional
        Indices of covariates to plot.
    silent : bool, default=True
        If False, returns a DataFrame with balance statistics.
    **kwargs
        Additional arguments passed to the underlying plot function.

    Returns
    -------
    pd.DataFrame or None
        If silent=False, returns DataFrame with balance statistics.
    """
    # Extract treatment variable
    if isinstance(npcbps_obj, dict):
        y = npcbps_obj.get('y')
    elif hasattr(npcbps_obj, 'y'):
        y = npcbps_obj.y
    else:
        raise ValueError("npcbps_obj must have a 'y' attribute or key")
    
    # Determine treatment type based on data characteristics
    # Continuous treatment: floating type AND many unique values (> 10)
    # Discrete treatment: few unique values (<= 10) regardless of dtype
    n_unique = len(np.unique(y))
    is_continuous = np.issubdtype(y.dtype, np.floating) and n_unique > 10
    
    if is_continuous:
        # Continuous treatment
        return plot_cbps_continuous(npcbps_obj, covars=covars, silent=silent, **kwargs)
    else:
        # Binary/multi-valued treatment
        return plot_cbps(npcbps_obj, covars=covars, silent=silent, **kwargs)


def plot_cbmsm(
    cbmsm_obj,
    covars: Optional[List[int]] = None,
    silent: bool = True,
    boxplot: bool = False,
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Visualize covariate balance for marginal structural models.

    Creates a scatter plot comparing unweighted versus CBMSM-weighted
    standardized mean differences across treatment history contrasts.
    Points below the y=x reference line indicate balance improvement.

    Parameters
    ----------
    cbmsm_obj : CBMSMResults
        Fitted CBMSM result object.
    covars : list of int, optional
        Covariate indices to plot (1-based). Default plots all covariates.
    silent : bool, default=True
        If False, returns a DataFrame with balance statistics.
    boxplot : bool, default=False
        If True, displays boxplots instead of scatter plot.
    **kwargs
        Additional arguments passed to matplotlib.

    Returns
    -------
    pd.DataFrame or None
        If silent=False, returns DataFrame with columns: Covariate,
        Contrast, Unweighted, Balanced.

    Notes
    -----
    The x-axis shows unweighted SMD (baseline), y-axis shows CBMSM-weighted
    SMD. Points below the diagonal indicate improved balance.

    References
    ----------
    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. Journal of the American Statistical
    Association, 110(511), 1013-1023.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plot_cbmsm(). "
            "Install it with: pip install matplotlib"
        )
    
    # Call balance method to get balance statistics
    if CBMSMResults is not None and isinstance(cbmsm_obj, CBMSMResults):
        bal_out = cbmsm_obj.balance()
    else:
        raise TypeError(
            "cbmsm_obj must be a CBMSMResults object. "
            "Ensure you have fitted a CBMSM model first."
        )
    
    bal = bal_out['Balanced']  # (n_covars, 2*n_treat_hist)
    baseline = bal_out['Unweighted']
    
    # Extract treatment history count
    # First half of columns are means, second half are standardized means
    no_treats = bal.shape[1] // 2
    
    # Select covariates to plot
    if covars is None:
        # All covariates (0-based indexing)
        covars = list(range(bal.shape[0]))
    else:
        # Convert 1-based index to Python 0-based
        covars = [c - 1 for c in covars]
        # Validate indices
        if any(c < 0 or c >= bal.shape[0] for c in covars):
            raise ValueError(
                f"covars indices out of range. "
                f"Valid range: 1 to {bal.shape[0]} (1-based)"
            )
    
    # Initialize result lists
    covarlist = []
    contrast = []
    bal_std_diff = []
    baseline_std_diff = []
    
    # Extract treatment history names from column names
    # Column name format: "0+0+1.mean", "0+0+1.std.mean", etc.
    cnames = bal_out.get('column_names', [f"TH{i}" for i in range(bal.shape[1])])
    treat_hist_names = []
    for i in range(no_treats):
        name = cnames[i]
        # Remove ".mean" suffix if present
        if name.endswith('.mean'):
            treat_hist_names.append(name[:-5])
        else:
            treat_hist_names.append(name)
    
    # Get covariate names from balance output
    rnames = bal_out.get('row_names', [f"X{i+1}" for i in range(bal.shape[0])])
    
    # Calculate standardized mean differences for all treatment history contrasts
    for i in covars:
        # For each covariate, calculate pairwise contrasts
        for j in range(no_treats - 1):
            for k in range(j + 1, no_treats):
                covarlist.append(rnames[i])
                contrast.append(f"{treat_hist_names[j]}:{treat_hist_names[k]}")
                
                # Compute absolute difference in standardized means
                bal_std_diff.append(abs(bal[i, no_treats + j] - bal[i, no_treats + k]))
                baseline_std_diff.append(abs(baseline[i, no_treats + j] - baseline[i, no_treats + k]))
    
    # Check for empty covariate list
    if len(bal_std_diff) == 0 or len(baseline_std_diff) == 0:
        import warnings
        warnings.warn(
            "No covariates available for plotting. "
            "The balance matrix is empty, possibly because:\n"
            "  1. All covariates were filtered out due to zero variance\n"
            "  2. The model has no valid covariates after preprocessing\n"
            "  3. CBMSM's x matrix structure issue (missing intercept)\n\n"
            "Skipping plot generation. To diagnose:\n"
            "  - Check cbmsm_fit.x.shape (expected > (n, 0))\n"
            "  - Verify formula includes time-varying covariates\n"
            "  - Ensure covariates have non-zero variance",
            UserWarning
        )
        return None
    
    # Determine plot range
    range_xy = [
        min(min(bal_std_diff), min(baseline_std_diff)),
        max(max(bal_std_diff), max(baseline_std_diff))
    ]
    
    if not boxplot:
        # Scatter plot mode
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (8, 8)))
        
        ax.scatter(baseline_std_diff, bal_std_diff, **kwargs)
        ax.plot(range_xy, range_xy, 'k-', linewidth=1, label='y=x')  # y=x reference line
        
        ax.set_xlabel('Unweighted Regression Imbalance', fontsize=12)
        ax.set_ylabel('CBMSM Imbalance', fontsize=12)
        ax.set_title('Difference in Standardized Means', fontsize=14)
        ax.set_xlim(range_xy)
        ax.set_ylim(range_xy)
        ax.set_aspect('equal')  # equal aspect ratio for comparison
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        # Boxplot mode
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))
        
        # Horizontal boxplot comparing unweighted vs weighted balance
        bp = ax.boxplot(
            [baseline_std_diff, bal_std_diff],
            vert=False,  # horizontal orientation
            labels=['Unweighted', 'CBMSM Weighted'],
            **kwargs
        )
        
        ax.set_xlabel('Difference in Standardized Means', fontsize=12)
        ax.set_title('Covariate Balance Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    # Return data if requested
    if not silent:
        return pd.DataFrame({
            'Covariate': covarlist,
            'Contrast': contrast,
            'Unweighted': baseline_std_diff,
            'Balanced': bal_std_diff
        })
    else:
        return None
