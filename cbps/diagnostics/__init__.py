"""
Diagnostics and Visualization Module
=====================================

This module provides tools for assessing covariate balance before and after
propensity score weighting, a critical step in evaluating causal inference
methodology.

Balance Assessment
------------------
For binary and multi-valued treatments, balance is measured via **standardized
mean differences (SMD)**. Following Austin (2009) and Stuart (2010), an SMD
below 0.1 indicates acceptable balance.

For continuous treatments, balance is assessed via **weighted Pearson
correlations** between covariates and the treatment variable. An absolute
correlation near zero indicates that the treatment is conditionally independent
of covariates given the weights.

Functions
---------
balance_cbps
    Compute standardized mean differences for binary/multi-valued treatments.

balance_cbps_continuous
    Compute weighted correlations for continuous treatments.

plot_cbps
    Visualize SMD before and after weighting (requires matplotlib).

plot_cbps_continuous
    Visualize correlation reduction for continuous treatments.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1),
156-177.

Austin, P.C. (2009). Balance diagnostics for comparing the distribution of
baseline covariates between treatment groups in propensity-score matched
samples. Statistics in Medicine, 28(25), 3083-3107.

Stuart, E.A. (2010). Matching methods for causal inference: A review and a
look forward. Statistical Science, 25(1), 1-21.
"""

from .balance import balance_cbps, balance_cbps_continuous
from .continuous_diagnostics import (
    diagnose_cbgps_quality,
    print_balance_diagnosis,
    compute_f_statistic as compute_f_statistic_continuous,
    compute_weighted_correlations
)

# plots module is imported as optional dependency
try:
    from .plots import plot_cbps, plot_cbps_continuous
    __all__ = [
        'balance_cbps',
        'balance_cbps_continuous',
        'plot_cbps',
        'plot_cbps_continuous',
        'diagnose_cbgps_quality',
        'print_balance_diagnosis',
        'compute_f_statistic_continuous',
        'compute_weighted_correlations'
    ]
except ImportError:
    # When matplotlib is not installed, only export balance functions
    __all__ = [
        'balance_cbps', 
        'balance_cbps_continuous',
        'diagnose_cbgps_quality',
        'print_balance_diagnosis',
        'compute_f_statistic_continuous',
        'compute_weighted_correlations'
    ]
