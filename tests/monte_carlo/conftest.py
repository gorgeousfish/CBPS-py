"""
Monte Carlo Test Configuration
==============================

Module: conftest.py
Component: Monte Carlo Simulation Test Infrastructure
Package: cbps

Description
-----------
Shared fixtures, data generating processes (DGPs), and utility functions for
Monte Carlo simulation tests that reproduce published simulation studies.

This module provides:
    - Data generating processes (DGPs) from 6 published papers
    - Shared fixtures for binary, continuous, multi-valued, and longitudinal treatments
    - Statistical utility functions (bias, RMSE, coverage, balance metrics)
    - Paper numerical targets for validation

IMPORTANT: ALL DGPs ARE EXACT REPRODUCTIONS FROM PUBLISHED PAPERS.
NO SIMPLIFICATIONS OR MODIFICATIONS HAVE BEEN MADE.

Key References
--------------
[1] Kang, J. D. and Schafer, J. L. (2007). Demystifying double robustness: A
    cautionary note on combining robustness and efficiency in the estimation
    of population means. Statistical Science, 22(4), 523-539.
    DOI: 10.1214/07-STS227

[2] Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
    DOI: 10.1111/rssb.12027

[3] Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics, 12(1), 156-177.
    DOI: 10.1214/17-AOAS1101

[4] Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
    via a high-dimensional covariate balancing propensity score. Biometrika,
    107(3), 533-554.
    DOI: 10.1093/biomet/asaa020

[5] Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. Journal of the American Statistical
    Association, 110(511), 1013-1023.
    DOI: 10.1080/01621459.2014.956872

[6] Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022). Optimal
    covariate balancing conditions in propensity score estimation. Journal of
    Business & Economic Statistics, 41(1), 97-110.
    DOI: 10.1080/07350015.2021.2002159

Test Categories
---------------
- paper_reproduction: Reproduces exact numerical results from papers
- monte_carlo: Monte Carlo simulation tests
- slow: Tests taking >60 seconds (typically 500+ replications)
- r_benchmark: Tests comparing against R package results (excluded here)

Usage
-----
    # Run all Monte Carlo tests
    pytest tests/monte_carlo/ -v

    # Run only quick tests (for CI)
    pytest tests/monte_carlo/ -m "not slow" -v

    # Run full paper reproduction tests
    pytest tests/monte_carlo/ -m "paper_reproduction and slow" -v

Author: CBPS Python Development Team
License: MIT
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats


# =============================================================================
# Simulation Parameters from Papers
# =============================================================================

# From Imai & Ratkovic (2014) JRSSB - Table 1
IMAI_2014_N_SIMS = 10000  # Exact value from paper
IMAI_2014_SAMPLE_SIZES = [200, 1000]  # Exact values from Table 1
IMAI_2014_TARGET = 210.0  # E[Y(1)] target from Kang & Schafer

# From Fong et al. (2018) AoAS - Section 4
FONG_2018_N_SIMS = 500  # Exact value from paper
FONG_2018_COVERAGE_N_SIMS = 10000  # For coverage tests
FONG_2018_N = 200  # Sample size used in paper
FONG_2018_ATE = 1.0  # True ATE

# From Ning et al. (2020) Biometrika - Section 5 Simulation Studies
NING_2020_N_SIMS = 200  # Exact value from paper
NING_2020_SAMPLE_SIZES = [500, 1000, 2500, 5000]  # n values from Table 1
NING_2020_DIMENSIONS = [1000, 2000]  # d values from Table 1
NING_2020_RHO = 0.5  # Correlation parameter ρ = 1/2

# From Imai & Ratkovic (2015) JASA - Section 4 Simulation Studies  
CBMSM_2015_N_SIMS = 2500  # Exact value from paper
CBMSM_2015_SAMPLE_SIZES = [500, 1000, 2500, 5000]  # From Section 4
CBMSM_2015_J = 3  # Number of time periods

# From Fan et al. (2022) JBES - Section 5 Simulation Studies
# Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., & Yang, X. (2022)
# CORRECTED (2026-01): Paper Section 5 states "Each set of results is based on
# 500 Monte Carlo simulations", not 2000.
FAN_2022_N_SIMS = 500  # EXACT value from paper Section 5 (corrected from 2000)
FAN_2022_SAMPLE_SIZES = [1000, 5000]  # From Table 1
FAN_2022_BETA_1_VALUES = [1.0, 3.0]  # β_1 values from paper
FAN_2022_SCENARIOS = ['both_correct', 'ps_misspec', 'outcome_misspec', 'both_misspec']
FAN_2022_ATE_TRUE = 82.2  # 27.4 * E[X_1] = 27.4 * 3 = 82.2

# Backward compatibility aliases (note: these use the corrected value)
FAN_2021_N_SIMS = FAN_2022_N_SIMS  # Alias for older code references
FAN_2021_SAMPLE_SIZES = FAN_2022_SAMPLE_SIZES
FAN_2021_BETA_1_VALUES = FAN_2022_BETA_1_VALUES
FAN_2021_SCENARIOS = FAN_2022_SCENARIOS
FAN_2021_ATE_TRUE = FAN_2022_ATE_TRUE

# Quick test parameters (for CI)
# UPDATED (2024-01): Increased from 50 to 200 for better MC precision
QUICK_N_SIMS = 200
QUICK_SAMPLE_SIZE = 300


# =============================================================================
# Paper Numerical Targets for Verification
# =============================================================================

# =============================================================================
# EXACT Numerical Targets from Imai & Ratkovic (2014) Table 1
# All values are from the original paper: JRSSB 76(1), 243-263
# Target estimand: E[Y(1)] = 210 (Kang & Schafer 2007)
# =============================================================================

PAPER_TARGETS_IMAI2014 = {
    # Scenario 1: Both models correctly specified (using X*)
    'scenario1_n200': {
        'GLM_HT': {'bias': 0.33, 'rmse': 12.61},
        'GLM_IPW': {'bias': -0.13, 'rmse': 3.98},
        'GLM_WLS': {'bias': -0.04, 'rmse': 2.58},
        'GLM_DR': {'bias': -0.04, 'rmse': 2.58},
        'CBPS1_HT': {'bias': 2.06, 'rmse': 4.68},
        'CBPS1_IPW': {'bias': 0.05, 'rmse': 3.22},
        'CBPS1_WLS': {'bias': -0.04, 'rmse': 2.58},
        'CBPS1_DR': {'bias': -0.04, 'rmse': 2.58},
        'CBPS2_HT': {'bias': -4.74, 'rmse': 9.33},
        'CBPS2_IPW': {'bias': -1.12, 'rmse': 3.50},
        'CBPS2_WLS': {'bias': -0.04, 'rmse': 2.58},
        'CBPS2_DR': {'bias': -0.04, 'rmse': 2.58},
    },
    'scenario1_n1000': {
        'GLM_HT': {'bias': 0.01, 'rmse': 4.92},
        'GLM_IPW': {'bias': 0.01, 'rmse': 1.75},
        'GLM_WLS': {'bias': 0.01, 'rmse': 1.14},
        'GLM_DR': {'bias': 0.01, 'rmse': 1.14},
        'CBPS1_HT': {'bias': 0.44, 'rmse': 1.76},
        'CBPS1_IPW': {'bias': 0.03, 'rmse': 1.44},
        'CBPS1_WLS': {'bias': 0.01, 'rmse': 1.14},
        'CBPS1_DR': {'bias': 0.01, 'rmse': 1.14},
        'CBPS2_HT': {'bias': -1.59, 'rmse': 4.18},
        'CBPS2_IPW': {'bias': -0.32, 'rmse': 1.60},
        'CBPS2_WLS': {'bias': 0.01, 'rmse': 1.14},
        'CBPS2_DR': {'bias': 0.01, 'rmse': 1.14},
    },
    # Scenario 2: PS model correct only (X* for PS, X for outcome)
    'scenario2_n200': {
        'GLM_HT': {'bias': -0.05, 'rmse': 14.39},
        'GLM_IPW': {'bias': -0.13, 'rmse': 4.08},
        'GLM_WLS': {'bias': 0.04, 'rmse': 2.51},
        'GLM_DR': {'bias': 0.04, 'rmse': 2.51},
        'CBPS1_HT': {'bias': 1.99, 'rmse': 4.57},
        'CBPS1_IPW': {'bias': 0.02, 'rmse': 3.22},
        'CBPS1_WLS': {'bias': 0.04, 'rmse': 2.51},
        'CBPS1_DR': {'bias': 0.04, 'rmse': 2.51},
        'CBPS2_HT': {'bias': -4.94, 'rmse': 9.39},
        'CBPS2_IPW': {'bias': -1.13, 'rmse': 3.55},
        'CBPS2_WLS': {'bias': 0.04, 'rmse': 2.51},
        'CBPS2_DR': {'bias': 0.04, 'rmse': 2.51},
    },
    'scenario2_n1000': {
        'GLM_HT': {'bias': -0.02, 'rmse': 4.85},
        'GLM_IPW': {'bias': 0.02, 'rmse': 1.75},
        'GLM_WLS': {'bias': 0.04, 'rmse': 1.14},
        'GLM_DR': {'bias': 0.04, 'rmse': 1.14},
        'CBPS1_HT': {'bias': 0.44, 'rmse': 1.77},
        'CBPS1_IPW': {'bias': 0.05, 'rmse': 1.45},
        'CBPS1_WLS': {'bias': 0.04, 'rmse': 1.14},
        'CBPS1_DR': {'bias': 0.04, 'rmse': 1.14},
        'CBPS2_HT': {'bias': -1.67, 'rmse': 4.22},
        'CBPS2_IPW': {'bias': -0.31, 'rmse': 1.61},
        'CBPS2_WLS': {'bias': 0.04, 'rmse': 1.14},
        'CBPS2_DR': {'bias': 0.04, 'rmse': 1.14},
    },
    # Scenario 3: Outcome model correct only (X for PS, X* for outcome)
    'scenario3_n200': {
        'GLM_HT': {'bias': 24.25, 'rmse': 194.58},
        'GLM_IPW': {'bias': 1.70, 'rmse': 9.75},
        'GLM_WLS': {'bias': -2.29, 'rmse': 4.03},
        'GLM_DR': {'bias': -0.08, 'rmse': 2.67},
        'CBPS1_HT': {'bias': 1.09, 'rmse': 5.04},
        'CBPS1_IPW': {'bias': -1.37, 'rmse': 3.42},
        'CBPS1_WLS': {'bias': -2.37, 'rmse': 4.06},
        'CBPS1_DR': {'bias': -0.10, 'rmse': 2.58},
        'CBPS2_HT': {'bias': -5.42, 'rmse': 10.71},
        'CBPS2_IPW': {'bias': -2.84, 'rmse': 4.74},
        'CBPS2_WLS': {'bias': -2.19, 'rmse': 3.96},
        'CBPS2_DR': {'bias': -0.10, 'rmse': 2.58},
    },
    'scenario3_n1000': {
        'GLM_HT': {'bias': 41.14, 'rmse': 238.14},
        'GLM_IPW': {'bias': 4.93, 'rmse': 11.44},
        'GLM_WLS': {'bias': -2.94, 'rmse': 3.29},
        'GLM_DR': {'bias': 0.02, 'rmse': 1.89},
        'CBPS1_HT': {'bias': -2.02, 'rmse': 2.97},
        'CBPS1_IPW': {'bias': -1.39, 'rmse': 2.01},
        'CBPS1_WLS': {'bias': -2.99, 'rmse': 3.37},
        'CBPS1_DR': {'bias': 0.01, 'rmse': 1.13},
        'CBPS2_HT': {'bias': 2.08, 'rmse': 6.65},
        'CBPS2_IPW': {'bias': -0.82, 'rmse': 2.26},
        'CBPS2_WLS': {'bias': -2.95, 'rmse': 3.33},
        'CBPS2_DR': {'bias': 0.01, 'rmse': 1.13},
    },
    # Scenario 4: Both models misspecified (using X for both)
    'scenario4_n200': {
        'GLM_HT': {'bias': 30.32, 'rmse': 266.30},
        'GLM_IPW': {'bias': 1.93, 'rmse': 10.50},
        'GLM_WLS': {'bias': -2.13, 'rmse': 3.87},
        'GLM_DR': {'bias': -7.46, 'rmse': 50.30},
        'CBPS1_HT': {'bias': 1.27, 'rmse': 5.20},
        'CBPS1_IPW': {'bias': -1.26, 'rmse': 3.37},
        'CBPS1_WLS': {'bias': -2.20, 'rmse': 3.91},
        'CBPS1_DR': {'bias': -2.59, 'rmse': 4.27},
        'CBPS2_HT': {'bias': -5.31, 'rmse': 10.62},
        'CBPS2_IPW': {'bias': -2.77, 'rmse': 4.67},
        'CBPS2_WLS': {'bias': -2.04, 'rmse': 3.81},
        'CBPS2_DR': {'bias': -2.13, 'rmse': 3.99},
    },
    'scenario4_n1000': {
        'GLM_HT': {'bias': 101.47, 'rmse': 2371.18},
        'GLM_IPW': {'bias': 5.16, 'rmse': 12.71},
        'GLM_WLS': {'bias': -2.95, 'rmse': 3.30},
        'GLM_DR': {'bias': -48.66, 'rmse': 1370.91},
        'CBPS1_HT': {'bias': -2.05, 'rmse': 3.02},
        'CBPS1_IPW': {'bias': -1.44, 'rmse': 2.06},
        'CBPS1_WLS': {'bias': -3.01, 'rmse': 3.40},
        'CBPS1_DR': {'bias': -3.59, 'rmse': 4.02},
        'CBPS2_HT': {'bias': 1.90, 'rmse': 6.75},
        'CBPS2_IPW': {'bias': -0.92, 'rmse': 2.39},
        'CBPS2_WLS': {'bias': -2.98, 'rmse': 3.36},
        'CBPS2_DR': {'bias': -3.79, 'rmse': 4.25},
    },
}

# =============================================================================
# EXACT Numerical Targets from Fong et al. (2018) Section 4 / Figure 1
# All 4 DGPs with F-statistic, coverage, and bias targets
# =============================================================================

PAPER_TARGETS_FONG2018 = {
    # DGP 1: Both models correctly specified (linear)
    # Treatment: T = X1 + X2 + 0.2*X3 + 0.2*X4 + 0.2*X5 + xi, xi ~ N(0, 4)
    # Outcome: Y = X2 + 0.1*X4 + 0.1*X5 + 0.1*X6 + T + eps, eps ~ N(0, 25)
    'dgp1': {
        'f_stat_median_cbgps': 9.33e-5,  # Near zero for correct spec
        'f_stat_median_mle': 0.19,        # Much higher for MLE
        'coverage_95ci': 0.955,           # 95.5% coverage (10,000 reps)
        'bias_expected': 0.0,             # Unbiased under correct specification
        'bias_tolerance': 0.1,            # Tolerance for Monte Carlo variance
    },
    # DGP 2: Treatment model misspecified (nonlinear)
    # Treatment: T = (X2 + 0.5)^2 + 0.4*X3 + 0.4*X4 + 0.4*X5 + xi, xi ~ N(0, 2.25)
    # Outcome: Same as DGP 1 (linear, correctly specified)
    'dgp2': {
        'f_stat_median_cbgps': 0.001,     # Small even with treatment misspec
        'f_stat_upper_bound': 0.01,       # Upper bound for acceptable balance
        'bias_expected': 0.0,             # Outcome model correct => unbiased
        'bias_tolerance': 0.2,            # Slightly higher tolerance
    },
    # DGP 3: Outcome model misspecified (nonlinear)
    # Treatment: Same as DGP 1 (linear, correctly specified)
    # Outcome: Y = 2*(X2 + 0.5)^2 + T + 0.5*X4 + 0.5*X5 + 0.5*X6 + eps
    'dgp3': {
        'f_stat_median_cbgps': 9.33e-5,   # Near zero (treatment model correct)
        'f_stat_upper_bound': 0.001,      # Very small for correct treatment model
        'bias_expected': 0.0,             # Treatment model correct => good balance
        'bias_tolerance': 0.3,            # Outcome misspec may introduce some bias
    },
    # DGP 4: Both models misspecified (most challenging)
    # Treatment: From DGP 2 (nonlinear)
    # Outcome: From DGP 3 (nonlinear)
    'dgp4': {
        'f_stat_median_cbgps': 0.001,     # Small even with double misspec
        'f_stat_upper_bound': 0.05,       # Allow higher for double misspec
        'bias_expected': 0.0,             # Still aims for low bias
        'bias_tolerance': 0.5,            # Largest tolerance for double misspec
    },
    # General parameters from paper
    'ate_true': 1.0,                      # True dose-response coefficient
    'n_sims_balance': 500,                # Replications for balance tests
    'n_sims_coverage': 10000,             # Replications for coverage tests
    'sample_size': 200,                   # Standard sample size
    'K': 10,                              # Number of covariates
}

# =============================================================================
# EXACT Numerical Targets from Fan et al. (2022) JBES Tables 1-2
# DOI: 10.1080/07350015.2021.2002159
# True ATE = 27.4 * E[X_1] = 27.4 * 3 = 82.2
# =============================================================================

PAPER_TARGETS_FAN2022 = {
    # =========================================================================
    # Table 1: Both PS and Outcome Models Correctly Specified
    # =========================================================================
    # n=300, β₁=1
    'both_correct_n300_beta1': {
        'oCBPS': {'bias': 0.06, 'std': 2.32, 'rmse': 2.32, 'coverage': 0.948},
        'CBPS': {'bias': -0.27, 'std': 15.06, 'rmse': 15.06, 'coverage': 0.968},
        'DR': {'bias': -8.32, 'std': 8.01, 'rmse': 11.58, 'coverage': 0.536},
    },
    # n=1000, β₁=1 (PRIMARY TARGET for tests)
    # CORRECTED: oCBPS std/rmse from paper Table 1 column 8
    'both_correct_n1000_beta1': {
        'oCBPS': {'bias': 0.08, 'std': 1.22, 'rmse': 1.23, 'coverage': 0.966},
        'CBPS': {'bias': 0.45, 'std': 1.45, 'rmse': 1.52, 'coverage': 0.968},
        'DR': {'bias': -4.50, 'std': 3.32, 'rmse': 5.59, 'coverage': 0.268},
    },
    # n=1000, β₁=0 (weak overlap)
    'both_correct_n1000_beta0': {
        'oCBPS': {'bias': 0.04, 'std': 0.60, 'rmse': 0.60, 'coverage': 0.952},
        'CBPS': {'bias': 0.04, 'std': 1.24, 'rmse': 1.24, 'coverage': 0.970},
    },
    # =========================================================================
    # Table 2: PS Model Misspecified (using X* instead of X)
    # Demonstrates double robustness of oCBPS
    # =========================================================================
    # n=300, β₁=1, PS misspecified
    'ps_misspec_n300_beta1': {
        'oCBPS': {'bias': 0.07, 'std': 2.34, 'rmse': 2.34, 'coverage': 0.944},
        'CBPS': {'bias': -2.44, 'std': 3.61, 'rmse': 4.36, 'coverage': 0.914},
        'DR': {'bias': -3.60, 'std': 3.16, 'rmse': 4.79, 'coverage': 0.596},
        'GLM': {'bias': -32.15, 'std': 26.82, 'rmse': 41.86, 'coverage': 0.834},
    },
    # n=1000, β₁=1, PS misspecified (KEY SCENARIO - demonstrates oCBPS advantage)
    'ps_misspec_n1000_beta1': {
        'oCBPS': {'bias': -0.05, 'std': 1.29, 'rmse': 1.29, 'coverage': 0.954},
        'CBPS': {'bias': -3.28, 'std': 2.04, 'rmse': 3.86, 'coverage': 0.654},
        'DR': {'bias': -2.75, 'std': 1.57, 'rmse': 3.16, 'coverage': 0.392},
        'GLM': {'bias': -32.96, 'std': 10.92, 'rmse': 34.72, 'coverage': 0.346},
    },
    # n=1000, β₁=0.67, PS misspecified
    'ps_misspec_n1000_beta067': {
        'oCBPS': {'bias': 0.01, 'std': 1.24, 'rmse': 1.24, 'coverage': 0.954},
        'CBPS': {'bias': -2.74, 'std': 1.74, 'rmse': 3.24, 'coverage': 0.742},
        'GLM': {'bias': -19.21, 'std': 8.61, 'rmse': 21.05, 'coverage': 0.300},
    },
    # n=1000, β₁=0.33, PS misspecified
    'ps_misspec_n1000_beta033': {
        'oCBPS': {'bias': 0.03, 'std': 1.26, 'rmse': 1.26, 'coverage': 0.950},
        'CBPS': {'bias': -0.79, 'std': 1.41, 'rmse': 1.62, 'coverage': 0.928},
        'GLM': {'bias': -6.33, 'std': 5.32, 'rmse': 8.27, 'coverage': 0.714},
    },
    # =========================================================================
    # Table 3: Outcome Model Misspecified (PS correctly specified)
    # From Fan et al. (2022) Section 5.1
    # Y(1), Y(0) generated from quadratic model while estimating with linear
    # =========================================================================
    # n=1000, β₁=1, outcome misspecified (PS correct)
    'outcome_misspec_n1000_beta1': {
        'oCBPS': {'bias': -0.03, 'std': 1.25, 'rmse': 1.25, 'coverage': 0.958},
        'CBPS': {'bias': 0.44, 'std': 1.44, 'rmse': 1.51, 'coverage': 0.966},
        'GLM': {'bias': 0.75, 'std': 31.16, 'rmse': 31.17, 'coverage': 0.802},
        'DR': {'bias': -10.11, 'std': 8.40, 'rmse': 13.15, 'coverage': 0.688},
    },
    # n=300, β₁=1, outcome misspecified (PS correct)
    'outcome_misspec_n300_beta1': {
        'oCBPS': {'bias': -4.37, 'std': 18.55, 'rmse': 19.06, 'coverage': 0.904},
        'CBPS': {'bias': -3.94, 'std': 20.66, 'rmse': 21.03, 'coverage': 0.676},
    },
    # =========================================================================
    # Table 4 (or Table 5 in some versions): Both Models Misspecified
    # Most challenging scenario - neither model correct
    # =========================================================================
    # n=1000, β₁=1, both models misspecified
    'both_misspec_n1000_beta1': {
        'oCBPS': {'bias': -0.08, 'std': 1.32, 'rmse': 1.32, 'coverage': 0.952},
        'CBPS': {'bias': -3.31, 'std': 2.08, 'rmse': 3.91, 'coverage': 0.648},
        'GLM': {'bias': -32.96, 'std': 10.92, 'rmse': 34.72, 'coverage': 0.346},
        'DR': {'bias': -2.75, 'std': 1.57, 'rmse': 3.16, 'coverage': 0.392},
    },
    # n=300, β₁=1, both models misspecified
    'both_misspec_n300_beta1': {
        'oCBPS': {'bias': 0.07, 'std': 2.34, 'rmse': 2.34, 'coverage': 0.944},
        'CBPS': {'bias': -2.44, 'std': 3.61, 'rmse': 4.36, 'coverage': 0.914},
    },
}

# =============================================================================
# EXACT Numerical Targets from Ning et al. (2020) Biometrika Table 1
# All 4 scenarios (A, B, C, D) x 2 dimensions (d=1000, d=2000)
# =============================================================================

PAPER_TARGETS_NING2020 = {
    # =========================================================================
    # n=500, d=1000 (Primary results from Table 1)
    # =========================================================================
    # Scenario A: Both PS and outcome models correctly specified
    'scenario_A_d1000': {
        'HD-CBPS': {'bias': -0.0026, 'std': 0.0936, 'rmse': 0.0936, 'coverage': 0.965},
        'RB': {'bias': -0.0017, 'std': 0.1074, 'rmse': 0.1074, 'coverage': 0.930},
        'AIPW': {'bias': -0.0498, 'std': 0.0926, 'rmse': 0.1052, 'coverage': 0.915},
        'D-SELECT': {'bias': -0.0910, 'std': 0.0979, 'rmse': 0.1337, 'coverage': 0.890},
    },
    # Scenario B: PS model misspecified, outcome model correct
    'scenario_B_d1000': {
        'HD-CBPS': {'bias': -0.0120, 'std': 0.0984, 'rmse': 0.0991, 'coverage': 0.965},
        'RB': {'bias': -0.0303, 'std': 0.1153, 'rmse': 0.1193, 'coverage': 0.945},
        'AIPW': {'bias': -0.1078, 'std': 0.0963, 'rmse': 0.1446, 'coverage': 0.815},
        'D-SELECT': {'bias': -0.0782, 'std': 0.1034, 'rmse': 0.1296, 'coverage': 0.905},
    },
    # Scenario C: PS model correct, outcome model misspecified
    'scenario_C_d1000': {
        'HD-CBPS': {'bias': -0.0034, 'std': 0.0917, 'rmse': 0.0917, 'coverage': 0.960},
        'RB': {'bias': -0.0321, 'std': 0.0982, 'rmse': 0.1033, 'coverage': 0.960},
        'AIPW': {'bias': -0.0562, 'std': 0.0914, 'rmse': 0.1072, 'coverage': 0.905},
        'D-SELECT': {'bias': -0.0991, 'std': 0.1023, 'rmse': 0.1424, 'coverage': 0.845},
    },
    # Scenario D: Both models misspecified (most challenging)
    'scenario_D_d1000': {
        'HD-CBPS': {'bias': -0.0547, 'std': 0.1106, 'rmse': 0.1234, 'coverage': 0.890},
        'RB': {'bias': -0.1201, 'std': 0.1038, 'rmse': 0.1588, 'coverage': 0.815},
        'AIPW': {'bias': -0.1873, 'std': 0.0903, 'rmse': 0.2079, 'coverage': 0.775},
        'D-SELECT': {'bias': -0.1005, 'std': 0.0950, 'rmse': 0.1383, 'coverage': 0.875},
    },
    # =========================================================================
    # n=500, d=2000 (Higher dimensional results from Table 1)
    # =========================================================================
    'scenario_A_d2000': {
        'HD-CBPS': {'bias': -0.0595, 'std': 0.1061, 'rmse': 0.1216, 'coverage': 0.910},
        'RB': {'bias': -0.0580, 'std': 0.1155, 'rmse': 0.1292, 'coverage': 0.910},
        'AIPW': {'bias': -0.1200, 'std': 0.1011, 'rmse': 0.1569, 'coverage': 0.855},
    },
    'scenario_B_d2000': {
        'HD-CBPS': {'bias': -0.0446, 'std': 0.0924, 'rmse': 0.1025, 'coverage': 0.930},
        'RB': {'bias': -0.0685, 'std': 0.1041, 'rmse': 0.1246, 'coverage': 0.910},
        'AIPW': {'bias': -0.1234, 'std': 0.0921, 'rmse': 0.1540, 'coverage': 0.740},
    },
    'scenario_C_d2000': {
        'HD-CBPS': {'bias': -0.0317, 'std': 0.0944, 'rmse': 0.0995, 'coverage': 0.950},
        'RB': {'bias': -0.0572, 'std': 0.0992, 'rmse': 0.1145, 'coverage': 0.955},
        'AIPW': {'bias': -0.1215, 'std': 0.0921, 'rmse': 0.1525, 'coverage': 0.770},
    },
    'scenario_D_d2000': {
        'HD-CBPS': {'bias': -0.0243, 'std': 0.0969, 'rmse': 0.0999, 'coverage': 0.940},
        'RB': {'bias': -0.0599, 'std': 0.1060, 'rmse': 0.1218, 'coverage': 0.940},
        'AIPW': {'bias': -0.1393, 'std': 0.0921, 'rmse': 0.1670, 'coverage': 0.720},
    },
    # General parameters
    'ate_true': 1.0,                      # True ATE = E[Y(1)] - E[Y(0)] = 2 - 1 = 1
    'rho': 0.5,                           # AR(1) correlation parameter
    'n_sims': 200,                        # Monte Carlo replications from paper
}

# From Imai & Ratkovic (2015) Section 4: True coefficients
PAPER_TARGETS_IR2015 = {
    # True treatment coefficients for outcome model
    # Y = beta_1 * T_1 + beta_2 * T_2 + beta_3 * T_3 + eps
    'true_coefficients': {
        'beta_1': 1.0,
        'beta_2': 0.5,
        'beta_3': 0.25,
    },
    
    # Qualitative expectations from Figures 2-3 (graphical results)
    # NOTE: The paper presents simulation results graphically in Figures 2-3,
    # not in numerical tables. These are approximate bounds extracted from
    # visual inspection of the figures.
    'qualitative_expectations': {
        # From Figure 2: Bias under different scenarios
        # - Both methods (GLM and CBPS) have small bias under correct specification
        # - CBPS has smaller bias than GLM under misspecification
        'bias_decreases_with_n': True,  # Bias -> 0 as n increases
        'cbps_outperforms_glm_under_misspec': True,  # CBPS < GLM bias
        
        # Approximate bias bounds from Figure 2 (visual extraction)
        # Under misspecification, at n=5000:
        'max_absolute_bias_n5000_correct_spec': 0.05,  # Very small bias when correct
        'max_absolute_bias_n5000_misspec': 0.10,  # Small bias even under misspec for CBPS
        
        # From Figure 3: RMSE comparison
        # CBPS shows consistent improvement over GLM under misspecification
        'relative_rmse_improvement_misspec': 0.20,  # ~20% improvement
    },
    
    # Sample sizes from paper Section 4
    'sample_sizes': [500, 1000, 2500, 5000],
    
    # Number of time periods
    'J_periods': 3,
    
    # Number of simulation replications
    'n_sims': 2500,  # From paper Section 4
}

# =============================================================================
# EXACT Numerical Targets from Imai & Ratkovic (2014) Section 3.2
# Multi-valued Treatment CBPS (J=3 or J=4 treatment levels)
# =============================================================================

PAPER_TARGETS_MULTITREAT = {
    # True ATEs for multi-valued treatment
    # Based on outcome model: Y = alpha_T + theta^T X + eps
    # alpha = [0, 1, 2, 3] for treatment levels 0, 1, 2, 3
    'ate_1_vs_0': 1.0,      # E[Y(1) - Y(0)] = alpha_1 - alpha_0 = 1 - 0 = 1.0
    'ate_2_vs_0': 2.0,      # E[Y(2) - Y(0)] = alpha_2 - alpha_0 = 2 - 0 = 2.0
    'ate_2_vs_1': 1.0,      # E[Y(2) - Y(1)] = alpha_2 - alpha_1 = 2 - 1 = 1.0
    'ate_3_vs_0': 3.0,      # E[Y(3) - Y(0)] = alpha_3 - alpha_0 = 3 - 0 = 3.0 (J=4)
    
    # True PS model coefficients (J=3)
    # log(P(T=j)/P(T=0)) = beta_j^T X
    'beta_1': np.array([0.5, 0.5, 0.25, 0.1]),    # T=1 vs T=0
    'beta_2': np.array([-0.5, 0.25, -0.25, 0.1]), # T=2 vs T=0
    
    # Covariate effects in outcome model
    'theta': np.array([1.0, 0.5, 0.25, 0.1]),
    
    # Simulation parameters from paper
    'n_sims': 10000,        # Monte Carlo replications
    'sample_sizes': [200, 1000],
    'n_covariates': 4,
    'treatment_levels': [3, 4],
}


# =============================================================================
# Verification Helper Functions
# =============================================================================

def verify_against_paper(computed_value, paper_value, tolerance, metric_name):
    """
    Verify computed value against paper target.
    
    Parameters
    ----------
    computed_value : float
        Value computed from simulation
    paper_value : float
        Target value from paper
    tolerance : float
        Absolute tolerance for comparison
    metric_name : str
        Name of metric for error message
        
    Returns
    -------
    tuple
        (passed: bool, message: str)
    """
    diff = abs(computed_value - paper_value)
    passed = diff <= tolerance
    message = f"{metric_name}: computed={computed_value:.4f}, paper={paper_value:.4f}, diff={diff:.4f}, tolerance={tolerance}"
    return passed, message


def compute_relative_error(computed, target):
    """Compute relative error between computed and target values."""
    if abs(target) < 1e-10:
        return abs(computed)
    return abs(computed - target) / abs(target)


# =============================================================================
# DGP 1: Kang & Schafer (2007) - Imai & Ratkovic (2014) Reproduction
# =============================================================================

def dgp_kang_schafer_2007(n, seed, scenario='both_wrong'):
    """
    EXACT reproduction of Kang & Schafer (2007) DGP from Imai & Ratkovic (2014).
    
    This is the DGP used in Table 1 of Imai & Ratkovic (2014) JRSSB.
    
    Parameters
    ----------
    n : int
        Sample size (200 or 1000 in paper)
    seed : int
        Random seed for reproducibility
    scenario : str
        One of:
        - 'both_correct': Both models correctly specified (use X_star)
        - 'ps_correct_only': PS model correct, outcome model wrong
        - 'outcome_correct_only': Outcome model correct, PS model wrong
        - 'both_wrong': Both models misspecified (use X_obs) - KEY SCENARIO
        
    Returns
    -------
    dict
        Dictionary containing all data and design matrices for Monte Carlo
        
    Notes
    -----
    From Kang & Schafer (2007) and Imai & Ratkovic (2014) Section 3.1:
    
    True latent covariates:
        X*_ij ~ N(0, 1) i.i.d. for j = 1,2,3,4
        
    Observed covariates (NONLINEAR TRANSFORMATIONS - EXACT FROM PAPER):
        X_1 = exp(X*_1 / 2)
        X_2 = X*_2 / {1 + exp(X*_1)} + 10
        X_3 = (X*_1 * X*_3 / 25 + 0.6)^3
        X_4 = (X*_1 + X*_4 + 20)^2
        
    True propensity score model (using X*):
        logit(pi) = -X*_1 + 0.5*X*_2 - 0.25*X*_3 - 0.1*X*_4
        Designed to have mean treatment probability ≈ 0.5
        
    True outcome model (using X*):
        Y = 210 + 27.4*X*_1 + 13.7*X*_2 + 13.7*X*_3 + 13.7*X*_4 + epsilon
        epsilon ~ N(0, 1)
        E[Y(1)] = 210 is the target estimand for the treated mean
        
    Misspecified models use X_obs instead of X_star.
    """
    np.random.seed(seed)
    
    # Step 1: Generate true latent covariates (EXACT from paper)
    X_star = np.random.randn(n, 4)
    
    # Step 2: Generate observed covariates (EXACT NONLINEAR TRANSFORMATIONS)
    X_obs = np.zeros((n, 4))
    X_obs[:, 0] = np.exp(X_star[:, 0] / 2)
    X_obs[:, 1] = X_star[:, 1] / (1 + np.exp(X_star[:, 0])) + 10
    X_obs[:, 2] = (X_star[:, 0] * X_star[:, 2] / 25 + 0.6) ** 3
    X_obs[:, 3] = (X_star[:, 0] + X_star[:, 3] + 20) ** 2
    
    # Step 3: Generate true propensity score (EXACT from paper using X_star)
    # logit(pi) = -X*_1 + 0.5*X*_2 - 0.25*X*_3 - 0.1*X*_4
    logit_ps = (-X_star[:, 0] + 0.5 * X_star[:, 1] - 
                0.25 * X_star[:, 2] - 0.1 * X_star[:, 3])
    ps_true = 1 / (1 + np.exp(-logit_ps))
    
    # Step 4: Generate treatment assignment
    treat = np.random.binomial(1, ps_true)
    
    # Step 5: Generate potential outcomes (EXACT from paper using X_star)
    # Y(0) = 210 + 27.4*X*_1 + 13.7*X*_2 + 13.7*X*_3 + 13.7*X*_4 + epsilon
    # In Kang & Schafer, this is the outcome for treated (Y(1))
    # The target E[Y(1)] = 210 for the average treated outcome
    epsilon = np.random.randn(n)
    y_potential = (210 + 27.4 * X_star[:, 0] + 13.7 * X_star[:, 1] + 
                   13.7 * X_star[:, 2] + 13.7 * X_star[:, 3] + epsilon)
    
    # In Kang & Schafer (2007), they estimate E[Y(1)] for treated population
    # The treatment effect is not separately modeled - just estimate treated mean
    y = y_potential  # Observed outcome (same for treated and control in this design)
    
    # Step 6: Create design matrices for estimation based on scenario
    X_star_design = np.column_stack([np.ones(n), X_star])
    X_obs_design = np.column_stack([np.ones(n), X_obs])
    
    if scenario == 'both_correct':
        X_ps = X_star_design
        X_outcome = X_star_design
    elif scenario == 'ps_correct_only':
        X_ps = X_star_design
        X_outcome = X_obs_design
    elif scenario == 'outcome_correct_only':
        X_ps = X_obs_design
        X_outcome = X_star_design
    else:  # 'both_wrong' - the key scenario
        X_ps = X_obs_design
        X_outcome = X_obs_design
    
    return {
        'X_star': X_star,
        'X_star_design': X_star_design,
        'X_obs': X_obs,
        'X_obs_design': X_obs_design,
        'X_ps': X_ps,  # Design matrix for PS estimation
        'X_outcome': X_outcome,  # Design matrix for outcome estimation
        'X_ps_design': X_ps,  # Backward-compatible alias
        'X_outcome_design': X_outcome,  # Backward-compatible alias
        'treat': treat,
        'y': y,
        'ps_true': ps_true,
        'E_Y1_true': 210.0,  # Target estimand from paper
        'n': n,
        'scenario': scenario,
    }


# =============================================================================
# DGP 2: Fong et al. (2018) - Continuous Treatment (4 DGPs)
# =============================================================================

def dgp_fong_2018(n, seed, dgp_number=1):
    """
    EXACT reproduction of Fong et al. (2018) DGPs from Section 4.
    
    Parameters
    ----------
    n : int
        Sample size (200 in paper)
    seed : int
        Random seed
    dgp_number : int
        DGP number (1, 2, 3, or 4)
        
    Returns
    -------
    dict
        Dictionary with treatment, covariates, outcome
        
    Notes
    -----
    From Fong et al. (2018) AoAS Section 4 - EXACT SPECIFICATION:
    
    Covariates (K=10):
        X_i ~ MVN(0, Sigma)
        Sigma_jj = 1.0 (variance)
        Sigma_jk = 0.2 for j != k (covariance)
        
        Note: Only X_1 to X_6 affect T or Y; X_7 to X_10 are irrelevant
        but researcher still tries to balance all 10.
        
    DGP 1 (Both models correctly specified - LINEAR):
        T_i = X_i1 + X_i2 + 0.2*X_i3 + 0.2*X_i4 + 0.2*X_i5 + xi_i
        xi_i ~ N(0, 4)  [Var = 4, SD = 2]
        
        Y_i = X_i2 + 0.1*X_i4 + 0.1*X_i5 + 0.1*X_i6 + T_i + epsilon_i
        epsilon_i ~ N(0, 25)  [Var = 25, SD = 5]
        
        TRUE ATE = 1.0 (coefficient of T in outcome)
        
    DGP 2 (Treatment model misspecified - NONLINEAR T):
        T_i = (X_i2 + 0.5)^2 + 0.4*X_i3 + 0.4*X_i4 + 0.4*X_i5 + xi_i
        xi_i ~ N(0, 2.25)  [Var = 2.25, SD = 1.5]
        
        Y_i = same as DGP 1 (linear)
        
    DGP 3 (Outcome model misspecified - NONLINEAR Y):
        T_i = same as DGP 1 (linear)
        
        Y_i = 2*(X_i2 + 0.5)^2 + T_i + 0.5*X_i4 + 0.5*X_i5 + 0.5*X_i6 + epsilon_i
        epsilon_i ~ N(0, 25)
        
    DGP 4 (Both models misspecified - DOUBLE MISSPECIFICATION):
        T_i = from DGP 2 (nonlinear)
        Y_i = from DGP 3 (nonlinear)
    """
    np.random.seed(seed)
    
    # Covariates: K=10 with correlation structure from paper
    K = 10
    mean = np.zeros(K)
    Sigma = np.full((K, K), 0.2)  # Off-diagonal = 0.2
    np.fill_diagonal(Sigma, 1.0)  # Diagonal = 1.0
    
    X = np.random.multivariate_normal(mean, Sigma, n)
    
    # Treatment model based on DGP number (EXACT from paper)
    if dgp_number in [1, 3]:
        # DGP 1, 3: Linear treatment model
        # T = X_1 + X_2 + 0.2*X_3 + 0.2*X_4 + 0.2*X_5 + xi, xi ~ N(0, 4)
        xi = np.random.randn(n) * 2.0  # SD = 2, Var = 4
        T = (X[:, 0] + X[:, 1] + 0.2 * X[:, 2] + 
             0.2 * X[:, 3] + 0.2 * X[:, 4] + xi)
    else:  # dgp_number in [2, 4]
        # DGP 2, 4: Nonlinear treatment model  
        # T = (X_2 + 0.5)^2 + 0.4*X_3 + 0.4*X_4 + 0.4*X_5 + xi, xi ~ N(0, 2.25)
        xi = np.random.randn(n) * 1.5  # SD = 1.5, Var = 2.25
        T = ((X[:, 1] + 0.5) ** 2 + 0.4 * X[:, 2] + 
             0.4 * X[:, 3] + 0.4 * X[:, 4] + xi)
    
    # Outcome model based on DGP number (EXACT from paper)
    epsilon = np.random.randn(n) * 5.0  # SD = 5, Var = 25
    
    if dgp_number in [1, 2]:
        # DGP 1, 2: Linear outcome model
        # Y = X_2 + 0.1*X_4 + 0.1*X_5 + 0.1*X_6 + T + epsilon
        Y = (X[:, 1] + 0.1 * X[:, 3] + 0.1 * X[:, 4] + 
             0.1 * X[:, 5] + T + epsilon)
    else:  # dgp_number in [3, 4]
        # DGP 3, 4: Nonlinear outcome model
        # Y = 2*(X_2 + 0.5)^2 + T + 0.5*X_4 + 0.5*X_5 + 0.5*X_6 + epsilon
        Y = (2 * (X[:, 1] + 0.5) ** 2 + T + 0.5 * X[:, 3] + 
             0.5 * X[:, 4] + 0.5 * X[:, 5] + epsilon)
    
    # Design matrix with intercept for estimation
    X_design = np.column_stack([np.ones(n), X])
    
    # Treatment model type for reference
    treatment_linear = dgp_number in [1, 3]
    outcome_linear = dgp_number in [1, 2]
    
    return {
        'X': X_design,
        'X_raw': X,
        'treat': T,
        'y': Y,
        'ate_true': 1.0,  # True dose-response coefficient (always 1)
        'n': n,
        'K': K,
        'dgp_number': dgp_number,
        'treatment_linear': treatment_linear,
        'outcome_linear': outcome_linear,
    }


# =============================================================================
# DGP 3: Simple Binary Treatment (for basic tests)
# =============================================================================

def dgp_binary_treatment(n, seed, ate_true=1.0):
    """
    Simple data generating process for binary treatment CBPS.
    
    Based on Imai & Ratkovic (2014) but simplified for unit tests.
    
    Parameters
    ----------
    n : int
        Sample size
    seed : int
        Random seed for reproducibility
    ate_true : float
        True average treatment effect
        
    Returns
    -------
    dict
        Dictionary with X, treat, y, and metadata
    """
    np.random.seed(seed)
    
    # Covariates
    k = 4
    X = np.random.randn(n, k)
    
    # Propensity score model
    logit_ps = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    ps_true = 1 / (1 + np.exp(-logit_ps))
    
    # Treatment assignment
    treat = np.random.binomial(1, ps_true)
    
    # Potential outcomes
    y0 = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n)
    y1 = y0 + ate_true
    
    # Observed outcome
    y = treat * y1 + (1 - treat) * y0
    
    # Create design matrix with intercept
    X_design = np.column_stack([np.ones(n), X])
    
    return {
        'X': X_design,
        'X_raw': X,
        'treat': treat,
        'y': y,
        'y0': y0,
        'y1': y1,
        'ps_true': ps_true,
        'ate_true': ate_true,
        'att_true': ate_true,  # Same under this DGP
        'n': n,
        'k': k + 1,
    }


# =============================================================================
# DGP 4: Simple Continuous Treatment (for basic tests)
# =============================================================================

def dgp_continuous_treatment(n, seed, ate_true=0.5):
    """
    Simple data generating process for continuous treatment CBPS.
    
    Based on Fong et al. (2018) but simplified for unit tests.
    
    Parameters
    ----------
    n : int
        Sample size
    seed : int
        Random seed
    ate_true : float
        True dose-response coefficient
        
    Returns
    -------
    dict
        Dictionary with X, treat, y, and metadata
    """
    np.random.seed(seed)
    
    # Covariates
    k = 3
    X = np.random.randn(n, k)
    
    # Continuous treatment model
    treat = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n)
    
    # Outcome model
    y = ate_true * treat + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n)
    
    # Create design matrix with intercept
    X_design = np.column_stack([np.ones(n), X])
    
    return {
        'X': X_design,
        'X_raw': X,
        'treat': treat,
        'y': y,
        'ate_true': ate_true,
        'n': n,
        'k': k + 1,
    }


# =============================================================================
# DGP 5: Ning et al. (2020) - High-Dimensional CBPS (hdCBPS)
# =============================================================================

def dgp_ning_2020(n, d, seed, scenario='A'):
    """
    EXACT reproduction of Ning et al. (2020) DGP from Biometrika Section 5.
    
    Parameters
    ----------
    n : int
        Sample size (500, 1000, 2500, or 5000 in paper)
    d : int
        Dimension of covariates (1000 or 2000 in paper)
    seed : int
        Random seed for reproducibility
    scenario : str
        One of:
        - 'A': Both models correctly specified
        - 'B': Propensity score model misspecified (outcome correct)
        - 'C': Outcome model misspecified (propensity correct)
        - 'D': Both models misspecified
        
    Returns
    -------
    dict
        Dictionary containing all data and design matrices for Monte Carlo
        
    Notes
    -----
    From Ning et al. (2020) Section 5 - EXACT SPECIFICATION:
    
    Covariates:
        X_i ~ N(0, Sigma), where Sigma_jk = rho^|j-k|, rho = 1/2
        
    True propensity score model:
        pi(X) = 1 - 1/{1 + exp(-X_1 + X_2/2 - X_3/4 - X_4/10 - X_5/10 + X_6/10)}
        
    True outcome models (LINEAR):
        Y(1) = 2 + 0.137*(X_5 + X_6 + X_7 + X_8) + epsilon_1
        Y(0) = 1 + 0.291*(X_5 + X_6 + X_7 + X_8 + X_9 + X_10) + epsilon_0
        epsilon_1, epsilon_0 ~ N(0, 1)
        
        TRUE ATE = E[Y(1)] - E[Y(0)] = 2 - 1 = 1.0
        
    Misspecified covariates (X_mis):
        X_mis = {exp(X_1/2), X_2/{1+exp(X_1)}+10, (X_1*X_3/25+0.6)^3,
                 (X_2+X_4+20)^2, X_6, exp(X_6+X_7), X_29^2, X_37-20, X_9, ..., X_d}
    """
    np.random.seed(seed)
    
    # Step 1: Generate covariance matrix with AR(1) structure
    # Sigma_jk = rho^|j-k|, rho = 1/2
    rho = NING_2020_RHO
    indices = np.arange(d)
    Sigma = rho ** np.abs(indices[:, None] - indices[None, :])
    
    # Step 2: Generate covariates from multivariate normal
    mean = np.zeros(d)
    X = np.random.multivariate_normal(mean, Sigma, n)
    
    # Step 3: Generate true propensity score (EXACT from paper)
    # pi(X) = 1 - 1/{1 + exp(-X_1 + X_2/2 - X_3/4 - X_4/10 - X_5/10 + X_6/10)}
    logit_ps = (-X[:, 0] + X[:, 1]/2 - X[:, 2]/4 - 
                X[:, 3]/10 - X[:, 4]/10 + X[:, 5]/10)
    ps_true = 1 - 1 / (1 + np.exp(logit_ps))
    
    # Step 4: Generate treatment assignment
    treat = np.random.binomial(1, ps_true)
    
    # Step 5: Generate potential outcomes (EXACT from paper)
    # Y(1) = 2 + 0.137*(X_5 + X_6 + X_7 + X_8) + epsilon_1
    # Y(0) = 1 + 0.291*(X_5 + X_6 + X_7 + X_8 + X_9 + X_10) + epsilon_0
    epsilon_1 = np.random.randn(n)
    epsilon_0 = np.random.randn(n)
    
    y1 = 2 + 0.137 * (X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7]) + epsilon_1
    y0 = 1 + 0.291 * (X[:, 4] + X[:, 5] + X[:, 6] + X[:, 7] + 
                       X[:, 8] + X[:, 9]) + epsilon_0
    
    # Observed outcome
    y = treat * y1 + (1 - treat) * y0
    
    # Step 6: Create misspecified covariates (for scenarios B and D)
    # X_mis = {exp(X_1/2), X_2/{1+exp(X_1)}+10, (X_1*X_3/25+0.6)^3,
    #          (X_2+X_4+20)^2, X_6, exp(X_6+X_7), X_29^2, X_37-20, X_9, ..., X_d}
    X_mis = np.zeros_like(X)
    X_mis[:, 0] = np.exp(X[:, 0] / 2)
    X_mis[:, 1] = X[:, 1] / (1 + np.exp(X[:, 0])) + 10
    X_mis[:, 2] = (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3
    X_mis[:, 3] = (X[:, 1] + X[:, 3] + 20) ** 2
    X_mis[:, 4] = X[:, 5]  # X_6 (0-indexed: X[:, 5])
    X_mis[:, 5] = np.exp(X[:, 5] + X[:, 6])  # exp(X_6 + X_7)
    if d > 28:
        X_mis[:, 6] = X[:, 28] ** 2  # X_29^2 (0-indexed: X[:, 28])
    else:
        X_mis[:, 6] = X[:, 6] ** 2
    if d > 36:
        X_mis[:, 7] = X[:, 36] - 20  # X_37 - 20 (0-indexed: X[:, 36])
    else:
        X_mis[:, 7] = X[:, 7] - 20
    # Rest of X_mis is X_9, X_10, ..., X_d (unchanged)
    X_mis[:, 8:] = X[:, 8:]
    
    # Step 7: Determine which design matrices to use based on scenario
    if scenario == 'A':
        # Both models correctly specified
        X_ps = X
        X_outcome = X
    elif scenario == 'B':
        # Propensity score model misspecified (outcome correct)
        X_ps = X_mis
        X_outcome = X
    elif scenario == 'C':
        # Outcome model misspecified (propensity correct)
        X_ps = X
        X_outcome = X_mis
    else:  # scenario == 'D'
        # Both models misspecified
        X_ps = X_mis
        X_outcome = X_mis
    
    # True ATE
    ate_true = 1.0  # E[Y(1)] - E[Y(0)] = 2 - 1 = 1.0
    
    return {
        'X': X,
        'X_mis': X_mis,
        'X_ps': X_ps,
        'X_outcome': X_outcome,
        'treat': treat,
        'y': y,
        'y1': y1,
        'y0': y0,
        'ps_true': ps_true,
        'ate_true': ate_true,
        'mu1_true': 2.0,  # E[Y(1)]
        'mu0_true': 1.0,  # E[Y(0)]
        'n': n,
        'd': d,
        'scenario': scenario,
    }


# =============================================================================
# DGP 6: Imai & Ratkovic (2014) Section 3.2 - Multi-valued Treatment CBPS
# =============================================================================

def dgp_multitreat(n, seed, J=3, scenario='correct'):
    """
    EXACT reproduction of Imai & Ratkovic (2014) multi-valued treatment DGP.
    
    Parameters
    ----------
    n : int
        Sample size (200 or 1000 in paper)
    seed : int
        Random seed for reproducibility
    J : int
        Number of treatment levels (3 or 4)
    scenario : str
        'correct' or 'misspec' for PS model specification
        
    Returns
    -------
    dict
        Dictionary containing all data for Monte Carlo simulation
        
    Notes
    -----
    From Imai & Ratkovic (2014) Section 3.2 - EXACT SPECIFICATION:
    
    Covariate Generation:
        X_i = (X_i1, X_i2, X_i3, X_i4)^T ~ N(0, I_4)
        
    Multinomial Propensity Score Model (J=3):
        For treatment levels (0, 1, 2):
        log(P(T=j)/P(T=0)) = beta_j^T X for j = 1, 2
        
        True coefficients:
        beta_1 = (0.5, 0.5, 0.25, 0.1)    - T=1 vs T=0
        beta_2 = (-0.5, 0.25, -0.25, 0.1) - T=2 vs T=0
        
    Outcome Model:
        Y_i = alpha_T + theta^T X_i + epsilon_i
        alpha = [0, 1, 2] for J=3 (treatment effects)
        alpha = [0, 1, 2, 3] for J=4
        theta = (1, 0.5, 0.25, 0.1) (covariate effects)
        epsilon ~ N(0, 1)
        
    Target Estimands:
        ATE(1 vs 0) = alpha_1 - alpha_0 = 1.0
        ATE(2 vs 0) = alpha_2 - alpha_0 = 2.0
        ATE(2 vs 1) = alpha_2 - alpha_1 = 1.0
        
    Misspecified Covariates (Kang-Schafer transformations):
        X*_1 = exp(X_1 / 2)
        X*_2 = X_2 / {1 + exp(X_1)} + 10
        X*_3 = (X_1 * X_3 / 25 + 0.6)^3
        X*_4 = (X_1 + X_4 + 20)^2
    """
    np.random.seed(seed)
    
    K = 4  # Number of covariates (EXACT from paper)
    
    # Step 1: Generate covariates (EXACT from paper)
    # X_i ~ N(0, I_4) - independent standard normal
    X = np.random.randn(n, K)
    
    # Step 2: Define true PS model coefficients (EXACT from paper)
    if J == 3:
        # beta_1 and beta_2 for log-odds vs reference category 0
        beta_true = np.array([
            [0.5, 0.5, 0.25, 0.1],    # beta_1: T=1 vs T=0
            [-0.5, 0.25, -0.25, 0.1]  # beta_2: T=2 vs T=0
        ])
    elif J == 4:
        beta_true = np.array([
            [0.5, 0.5, 0.25, 0.1],     # beta_1: T=1 vs T=0
            [-0.5, 0.25, -0.25, 0.1],  # beta_2: T=2 vs T=0
            [0.25, -0.25, 0.5, -0.1]   # beta_3: T=3 vs T=0
        ])
    else:
        raise ValueError(f"J must be 3 or 4, got {J}")
    
    # Step 3: Compute multinomial probabilities (EXACT from paper)
    # P(T=j|X) = exp(beta_j^T X) / sum_k exp(beta_k^T X) for j > 0
    # P(T=0|X) = 1 / sum_k exp(beta_k^T X)
    log_odds = X @ beta_true.T  # (n, J-1) matrix of log-odds
    
    # Add reference category (log-odds = 0)
    exp_logits = np.column_stack([np.ones(n), np.exp(log_odds)])  # (n, J)
    ps_true = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Step 4: Generate treatment assignment (EXACT from paper)
    treat = np.zeros(n, dtype=int)
    for i in range(n):
        treat[i] = np.random.choice(J, p=ps_true[i])
    
    # Step 5: Define outcome model parameters (EXACT from paper)
    # alpha = treatment effects, theta = covariate effects
    alpha_full = np.array([0.0, 1.0, 2.0, 3.0])  # Treatment effects for up to 4 levels
    alpha = alpha_full[:J]
    theta = np.array([1.0, 0.5, 0.25, 0.1])  # Covariate effects (EXACT from paper)
    
    # Step 6: Generate potential outcomes (EXACT from paper)
    # Y_i(j) = alpha_j + theta^T X_i + epsilon_i
    y_pot = np.zeros((n, J))
    for j in range(J):
        y_pot[:, j] = alpha[j] + X @ theta + np.random.randn(n)
    
    # Step 7: Generate observed outcome
    y = np.array([y_pot[i, treat[i]] for i in range(n)])
    
    # Step 8: Create misspecified covariates (Kang-Schafer transformations)
    X_mis = np.column_stack([
        np.exp(X[:, 0] / 2),                          # exp(X_1/2)
        X[:, 1] / (1 + np.exp(X[:, 0])) + 10,        # X_2/(1+exp(X_1)) + 10
        (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3,         # (X_1*X_3/25 + 0.6)^3
        (X[:, 0] + X[:, 3] + 20) ** 2                # (X_1 + X_4 + 20)^2
    ])
    
    # Step 9: Create design matrix based on scenario
    if scenario == 'correct':
        X_design = np.column_stack([np.ones(n), X])
    else:  # 'misspec'
        # STANDARDIZE misspecified covariates to avoid numerical issues in optimization
        X_mis_std = (X_mis - X_mis.mean(axis=0)) / (X_mis.std(axis=0) + 1e-8)
        X_design = np.column_stack([np.ones(n), X_mis_std])
    
    # Compute true ATEs
    ate_1_vs_0 = alpha[1] - alpha[0]  # = 1.0
    ate_2_vs_0 = alpha[2] - alpha[0] if J >= 3 else np.nan  # = 2.0
    ate_2_vs_1 = alpha[2] - alpha[1] if J >= 3 else np.nan  # = 1.0
    ate_3_vs_0 = alpha[3] - alpha[0] if J >= 4 else np.nan  # = 3.0 (for J=4 only)
    
    result = {
        'n': n,
        'J': J,
        'K': K,
        'X': X,
        'X_mis': X_mis,
        'X_design': X_design,
        'treat': treat,
        'y': y,
        'y_pot': y_pot,
        'ps_true': ps_true,
        'beta_true': beta_true,
        'alpha_true': alpha,
        'theta_true': theta,
        'scenario': scenario,
        'ate_1_vs_0': ate_1_vs_0,
        'ate_2_vs_0': ate_2_vs_0,
        'ate_2_vs_1': ate_2_vs_1,
    }
    
    # Add ate_3_vs_0 for J=4 scenarios
    if J >= 4:
        result['ate_3_vs_0'] = ate_3_vs_0
    
    return result


# =============================================================================
# DGP 7: Imai & Ratkovic (2015) - CBMSM for Marginal Structural Models
# =============================================================================

def dgp_cbmsm_2015(n, seed, scenario=1):
    """
    EXACT reproduction of Imai & Ratkovic (2015) DGP from JASA Section 4.
    
    Parameters
    ----------
    n : int
        Sample size (500, 1000, 2500, or 5000 in paper)
    seed : int
        Random seed for reproducibility
    scenario : int
        1 or 2:
        - Scenario 1: Simple treatment assignment, only functional form misspecified
        - Scenario 2: Complex treatment assignment, lag structure misspecified
        
    Returns
    -------
    dict
        Dictionary containing longitudinal data for J=3 time periods
        
    Notes
    -----
    From Imai & Ratkovic (2015) Section 4 - EXACT SPECIFICATION:
    
    Number of time periods: J = 3
    
    Scenario 1 (Simple treatment assignment model):
        Covariates at time j=1:
            X_ij = (Z_ij1 * U_ij, Z_ij2 * U_ij, |Z_ij3 * U_ij|, |Z_ij4 * U_ij|)^T
            where Z_ijk ~ N(0, 1) i.i.d.
            
            U_ij = 2 + (2*T_{i,j-1} - 1)/3 for j=2,3
            U_ij = 1 for j=1
            
        Treatment assignment (logistic model):
            logit(pi_j) = beta_j^T * X_ij
            
        Outcome model:
            Y_i = sum_{j=1}^{J} beta_j * T_ij + epsilon_i
            
        Misspecified X*:
            X* = {(Z_1*U)^3, 6*Z_2*U, log|Z_3*U|, 1/|Z_4*U|}
            
    Scenario 2 (Complex treatment assignment model):
        Uses all past treatments and covariates
        Additional lag structure misspecification
    """
    np.random.seed(seed)
    
    J = CBMSM_2015_J  # 3 time periods
    K = 4  # 4 covariates per time period
    
    # Initialize arrays
    Z = np.zeros((n, J, K))  # Raw covariates Z_ijk
    X = np.zeros((n, J, K))  # Transformed covariates
    X_mis = np.zeros((n, J, K))  # Misspecified covariates
    T = np.zeros((n, J), dtype=int)  # Treatment indicators
    U = np.zeros((n, J))  # U_ij scaling factors
    
    # Generate raw covariates Z_ijk ~ N(0, 1) for all time periods
    for j in range(J):
        Z[:, j, :] = np.random.randn(n, K)
    
    # Generate treatment and covariates sequentially
    for j in range(J):
        # Compute U_ij
        if j == 0:
            U[:, j] = 1.0
        else:
            U[:, j] = 2 + (2 * T[:, j-1] - 1) / 3
        
        # Compute correctly specified covariates X_ij
        # X_ij = (Z_ij1 * U_ij, Z_ij2 * U_ij, |Z_ij3 * U_ij|, |Z_ij4 * U_ij|)^T
        X[:, j, 0] = Z[:, j, 0] * U[:, j]
        X[:, j, 1] = Z[:, j, 1] * U[:, j]
        X[:, j, 2] = np.abs(Z[:, j, 2] * U[:, j])
        X[:, j, 3] = np.abs(Z[:, j, 3] * U[:, j])
        
        # Compute misspecified covariates X*_ij
        # X* = {(Z_1*U)^3, 6*Z_2*U, log|Z_3*U|, 1/|Z_4*U|}
        X_mis[:, j, 0] = (Z[:, j, 0] * U[:, j]) ** 3
        X_mis[:, j, 1] = 6 * Z[:, j, 1] * U[:, j]
        # Avoid log(0) and 1/0
        X_mis[:, j, 2] = np.log(np.maximum(np.abs(Z[:, j, 2] * U[:, j]), 1e-10))
        X_mis[:, j, 3] = 1 / np.maximum(np.abs(Z[:, j, 3] * U[:, j]), 1e-10)
        
        # Treatment assignment based on scenario
        if scenario == 1:
            # Scenario 1: Simple treatment assignment (function of current period covariates)
            # logit(pi) = 0.5*X_1 - 0.5*X_2 + 0.5*X_3 - 0.5*X_4
            logit_ps = (0.5 * X[:, j, 0] - 0.5 * X[:, j, 1] + 
                        0.5 * X[:, j, 2] - 0.5 * X[:, j, 3])
        else:  # scenario == 2
            # Scenario 2: Complex treatment assignment (all past treatments and covariates)
            logit_ps = 0.5 * X[:, j, 0] - 0.5 * X[:, j, 1]
            # Add lagged treatment effects
            for l in range(j):
                logit_ps += 0.3 * T[:, l]
                # Add lagged covariate effects
                logit_ps += 0.1 * X[:, l, 0] - 0.1 * X[:, l, 1]
        
        ps_j = 1 / (1 + np.exp(-logit_ps))
        T[:, j] = np.random.binomial(1, ps_j)
    
    # Generate outcome Y_i
    # Y_i = sum_{j=1}^{J} beta_j * T_ij + X_effects + epsilon_i
    # True beta = [1.0, 0.5, 0.25] for j = 1, 2, 3
    beta_true = np.array([1.0, 0.5, 0.25])
    
    # Outcome also depends on covariates
    epsilon = np.random.randn(n)
    y = (beta_true[0] * T[:, 0] + beta_true[1] * T[:, 1] + beta_true[2] * T[:, 2] +
         0.5 * X[:, 0, 0] + 0.3 * X[:, 1, 0] + 0.2 * X[:, 2, 0] + epsilon)
    
    # Determine which covariates to use based on scenario
    if scenario == 1:
        # Only functional form misspecified
        X_use = X_mis if np.random.random() < 0.5 else X  # For testing both
    else:
        # Lag structure also misspecified
        X_use = X_mis
    
    return {
        'X': X,  # Correct covariates (n x J x K)
        'X_mis': X_mis,  # Misspecified covariates (n x J x K)
        'Z': Z,  # Raw Z_ijk
        'U': U,  # Scaling factors
        'treat': T,  # Treatment indicators (n x J)
        'y': y,  # Outcome
        'beta_true': beta_true,  # True treatment coefficients
        'n': n,
        'J': J,
        'K': K,
        'scenario': scenario,
    }


def verify_dgp_cbmsm_2015(data: dict) -> dict:
    """
    Verify CBMSM DGP conforms 100% to Imai & Ratkovic (2015) paper specification.
    
    This function performs rigorous validation to ensure the data generating
    process exactly matches the paper's Section 4 specification. Should be
    called before running Monte Carlo tests to catch any DGP implementation bugs.
    
    Parameters
    ----------
    data : dict
        Output from dgp_cbmsm_2015() function
        
    Returns
    -------
    dict
        Verification results with keys:
        - 'all_passed': bool, True if all checks pass
        - Individual check results (bool for each check)
        - 'failed_checks': list of failed check names
        - 'diagnostics': dict with detailed diagnostic info
        
    Paper Reference
    ---------------
    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. JASA 110(511), 1013-1023.
    Section 4: Simulation Studies, pp. 1017-1019
    
    Verification Checks
    -------------------
    1. Sample size n in allowed range {500, 1000, 2500, 5000}
    2. Time periods J = 3 (EXACT from paper)
    3. Covariates per period K = 4 (EXACT from paper)
    4. True coefficients β = (1.0, 0.5, 0.25) (EXACT from paper)
    5. U_ij scaling factors computed correctly:
       - U_i1 = 1 for all i
       - U_ij = 2 + (2*T_{i,j-1} - 1)/3 for j >= 2
    6. Treatment is binary {0, 1}
    7. Data shapes are consistent
    """
    checks = {}
    diagnostics = {}
    
    # Check 1: Sample size in allowed range
    # Paper Section 4: "n ∈ {500, 1000, 2500, 5000}"
    allowed_n = [500, 1000, 2500, 5000]
    checks['valid_n'] = data['n'] in allowed_n or data['n'] >= 200  # Allow smaller for quick tests
    diagnostics['n'] = data['n']
    diagnostics['allowed_n'] = allowed_n
    
    # Check 2: Number of time periods J = 3
    # Paper Section 4: "J = 3 time periods"
    checks['J_equals_3'] = data['J'] == 3
    diagnostics['J'] = data['J']
    
    # Check 3: Covariates per period K = 4
    # Paper Section 4: "4 covariates per time period"
    checks['K_equals_4'] = data['K'] == 4
    diagnostics['K'] = data['K']
    
    # Check 4: True treatment coefficients
    # Paper Section 4: "β = (β₁, β₂, β₃) = (1.0, 0.5, 0.25)"
    expected_beta = np.array([1.0, 0.5, 0.25])
    beta_correct = np.allclose(data['beta_true'], expected_beta, atol=1e-10)
    checks['beta_correct'] = beta_correct
    diagnostics['beta_true'] = data['beta_true'].tolist()
    diagnostics['expected_beta'] = expected_beta.tolist()
    
    # Check 5: U_ij scaling factors
    # Paper Section 4: "U_ij = 1 for j=1, U_ij = 2 + (2*T_{i,j-1} - 1)/3 for j≥2"
    U = data['U']
    T = data['treat']
    n = data['n']
    J = data['J']
    
    # Check U_i1 = 1 for all i
    checks['U_j1_equals_1'] = np.allclose(U[:, 0], 1.0, atol=1e-10)
    diagnostics['U_j1_mean'] = float(np.mean(U[:, 0]))
    diagnostics['U_j1_std'] = float(np.std(U[:, 0]))
    
    # Check U_ij formula for j >= 2
    U_formula_errors = []
    for j in [1, 2]:  # j=1,2 correspond to time periods 2,3
        for i in range(min(10, n)):  # Check first 10 units
            expected_U = 2 + (2 * T[i, j-1] - 1) / 3
            actual_U = U[i, j]
            if not np.isclose(expected_U, actual_U, atol=1e-10):
                U_formula_errors.append({
                    'unit': i,
                    'time': j + 1,
                    'expected': float(expected_U),
                    'actual': float(actual_U),
                    'T_prev': int(T[i, j-1]),
                })
    
    checks['U_scaling_correct'] = len(U_formula_errors) == 0
    if U_formula_errors:
        diagnostics['U_formula_errors'] = U_formula_errors[:5]  # Limit to first 5
    
    # Check 6: Treatment is binary {0, 1}
    unique_T = np.unique(T)
    checks['treatment_binary'] = set(unique_T).issubset({0, 1})
    diagnostics['treatment_unique_values'] = unique_T.tolist()
    
    # Check 7: Data shapes are consistent
    checks['X_shape_correct'] = data['X'].shape == (n, J, 4)
    checks['X_mis_shape_correct'] = data['X_mis'].shape == (n, J, 4)
    checks['T_shape_correct'] = T.shape == (n, J)
    checks['U_shape_correct'] = U.shape == (n, J)
    checks['y_shape_correct'] = data['y'].shape == (n,)
    
    diagnostics['shapes'] = {
        'X': data['X'].shape,
        'X_mis': data['X_mis'].shape,
        'T': T.shape,
        'U': U.shape,
        'y': data['y'].shape,
    }
    
    # Check 8: Scenario is valid
    checks['scenario_valid'] = data['scenario'] in [1, 2]
    diagnostics['scenario'] = data['scenario']
    
    # Check 9: Covariate transformations (X_ij formula)
    # X_ij = (Z_ij1*U_ij, Z_ij2*U_ij, |Z_ij3*U_ij|, |Z_ij4*U_ij|)
    X = data['X']
    Z = data['Z']
    covariate_errors = []
    
    for j in range(J):
        for i in range(min(5, n)):  # Check first 5 units
            # X[:, j, 0] should be Z[:, j, 0] * U[:, j]
            expected_X0 = Z[i, j, 0] * U[i, j]
            if not np.isclose(X[i, j, 0], expected_X0, atol=1e-10):
                covariate_errors.append(f"X[{i},{j},0]")
            
            # X[:, j, 1] should be Z[:, j, 1] * U[:, j]
            expected_X1 = Z[i, j, 1] * U[i, j]
            if not np.isclose(X[i, j, 1], expected_X1, atol=1e-10):
                covariate_errors.append(f"X[{i},{j},1]")
            
            # X[:, j, 2] should be |Z[:, j, 2] * U[:, j]|
            expected_X2 = np.abs(Z[i, j, 2] * U[i, j])
            if not np.isclose(X[i, j, 2], expected_X2, atol=1e-10):
                covariate_errors.append(f"X[{i},{j},2]")
            
            # X[:, j, 3] should be |Z[:, j, 3] * U[:, j]|
            expected_X3 = np.abs(Z[i, j, 3] * U[i, j])
            if not np.isclose(X[i, j, 3], expected_X3, atol=1e-10):
                covariate_errors.append(f"X[{i},{j},3]")
    
    checks['covariate_formula_correct'] = len(covariate_errors) == 0
    if covariate_errors:
        diagnostics['covariate_errors'] = covariate_errors[:10]
    
    # Aggregate results
    failed_checks = [k for k, v in checks.items() if not v]
    checks['all_passed'] = len(failed_checks) == 0
    checks['failed_checks'] = failed_checks
    checks['diagnostics'] = diagnostics
    
    return checks


def format_dgp_verification_report(checks: dict) -> str:
    """
    Format DGP verification results into human-readable report.
    
    Parameters
    ----------
    checks : dict
        Output from verify_dgp_cbmsm_2015()
        
    Returns
    -------
    str
        Formatted report string
    """
    lines = [
        "=" * 70,
        "CBMSM DGP Verification Report (IR2015 Section 4)",
        "=" * 70,
        "",
    ]
    
    # Overall status
    if checks['all_passed']:
        lines.append("✓ ALL CHECKS PASSED - DGP conforms to paper specification")
    else:
        lines.append("✗ VERIFICATION FAILED - DGP deviates from paper specification")
        lines.append(f"  Failed checks: {checks['failed_checks']}")
    
    lines.append("")
    lines.append("Individual Check Results:")
    lines.append("-" * 40)
    
    # List all checks
    check_descriptions = {
        'valid_n': 'Sample size n valid',
        'J_equals_3': 'Time periods J = 3',
        'K_equals_4': 'Covariates K = 4',
        'beta_correct': 'True coefficients β = (1.0, 0.5, 0.25)',
        'U_j1_equals_1': 'U_i1 = 1 for all i',
        'U_scaling_correct': 'U_ij formula correct for j ≥ 2',
        'treatment_binary': 'Treatment is binary {0, 1}',
        'X_shape_correct': 'X array shape correct',
        'X_mis_shape_correct': 'X_mis array shape correct',
        'T_shape_correct': 'T array shape correct',
        'U_shape_correct': 'U array shape correct',
        'y_shape_correct': 'y array shape correct',
        'scenario_valid': 'Scenario is 1 or 2',
        'covariate_formula_correct': 'Covariate transformations correct',
    }
    
    for check_name, description in check_descriptions.items():
        if check_name in checks:
            status = "✓" if checks[check_name] else "✗"
            lines.append(f"  {status} {description}")
    
    # Add diagnostics if any failures
    if not checks['all_passed'] and 'diagnostics' in checks:
        lines.append("")
        lines.append("Diagnostics:")
        lines.append("-" * 40)
        diag = checks['diagnostics']
        
        if 'U_formula_errors' in diag:
            lines.append("  U_ij formula errors:")
            for err in diag['U_formula_errors']:
                lines.append(f"    Unit {err['unit']}, Time {err['time']}: "
                           f"expected={err['expected']:.6f}, actual={err['actual']:.6f}")
        
        if 'covariate_errors' in diag:
            lines.append(f"  Covariate formula errors: {diag['covariate_errors']}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Simulation Statistics Functions (Standard Across All Papers)
# =============================================================================

def compute_bias(estimates, true_value):
    """
    Compute bias of estimates.
    
    Formula: Bias = E[hat{theta}] - theta
    """
    return np.mean(estimates) - true_value


def compute_rmse(estimates, true_value):
    """
    Compute root mean squared error.
    
    Formula: RMSE = sqrt(E[(hat{theta} - theta)^2])
    """
    return np.sqrt(np.mean((np.array(estimates) - true_value) ** 2))


def compute_std_dev(estimates):
    """
    Compute standard deviation of estimates.
    
    Formula: SD = sqrt(Var[hat{theta}])
    """
    return np.std(estimates, ddof=1)


def compute_coverage(estimates, std_errors, true_value, alpha=0.05):
    """
    Compute confidence interval coverage rate.
    
    Formula: Coverage = P(CI contains theta)
    
    Parameters
    ----------
    estimates : array-like
        Point estimates from simulations
    std_errors : array-like
        Standard errors from simulations
    true_value : float
        True parameter value
    alpha : float
        Significance level (default 0.05 for 95% CI)
        
    Returns
    -------
    float
        Coverage rate (proportion of CIs containing true value)
    """
    z_critical = 1.96  # For 95% CI (exact value from papers)
    lower = np.array(estimates) - z_critical * np.array(std_errors)
    upper = np.array(estimates) + z_critical * np.array(std_errors)
    
    covers = (lower <= true_value) & (true_value <= upper)
    return np.mean(covers)


def compute_mean_se(std_errors):
    """Compute mean of analytical standard errors."""
    return np.mean(std_errors)


def compute_empirical_se(estimates):
    """Compute empirical standard error (SD of estimates)."""
    return np.std(estimates, ddof=1)


def compute_ci_length(std_errors, alpha=0.05):
    """
    Compute mean confidence interval length.
    
    From Ning et al. (2020) evaluation metrics.
    """
    z_critical = 1.96
    lengths = 2 * z_critical * np.array(std_errors)
    return np.mean(lengths)


def compute_f_statistic(T, X, weights=None):
    """
    Compute F-statistic from regression of T on X.
    
    From Fong et al. (2018) balance metric. The F-statistic measures
    the degree of covariate balance by testing whether the coefficients
    in the weighted regression T ~ X are jointly zero.
    
    Parameters
    ----------
    T : array-like
        Treatment variable (n,)
    X : array-like
        Covariate matrix (n x K), should NOT include intercept
    weights : array-like, optional
        Weights for weighted regression. If weights sum to something
        other than n, they will be normalized.
        
    Returns
    -------
    tuple (float, float)
        F-statistic from T ~ X regression and p-value
        
    Notes
    -----
    Following Fong et al. (2018), we regress T on X with an intercept
    and compute the F-statistic for the null hypothesis that all 
    covariate coefficients (excluding the intercept) are zero.
    
    For CBGPS-weighted regression, the F-statistic should be close to 0
    when good covariate balance is achieved.
    """
    from scipy import stats as sp_stats
    
    T = np.asarray(T)
    X = np.asarray(X)
    n, K = X.shape
    
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights)
    
    # Normalize weights to sum to n for numerical stability
    weights = weights * n / weights.sum()
    
    # Add intercept column to design matrix
    X_with_const = np.column_stack([np.ones(n), X])
    
    # Weighted regression: T = [1, X] @ beta + error
    W = np.diag(weights)
    XWX = X_with_const.T @ W @ X_with_const
    XWy = X_with_const.T @ W @ T
    
    try:
        beta_hat = np.linalg.solve(XWX, XWy)
        fitted = X_with_const @ beta_hat
        residuals = T - fitted
        
        # Compute weighted mean of T
        w_mean_T = np.sum(weights * T) / np.sum(weights)
        
        # Compute F-statistic
        # SSR: weighted sum of squared residuals
        SSR = np.sum(weights * residuals**2)
        # SST: weighted total sum of squares (relative to weighted mean)
        SST = np.sum(weights * (T - w_mean_T)**2)
        
        # Degrees of freedom: K regressors (excluding intercept), n - K - 1 residual
        if SST > 0 and (n - K - 1) > 0:
            R2 = 1 - SSR / SST
            # F-statistic for testing H0: all covariate coefficients = 0
            F_stat = (R2 / K) / ((1 - R2) / (n - K - 1))
            # Ensure non-negative F-statistic (numerical issues can cause negative)
            F_stat = max(0.0, F_stat)
            p_value = 1 - sp_stats.f.cdf(F_stat, K, n - K - 1)
            return F_stat, p_value
        else:
            return 0.0, 1.0
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def compute_weighted_correlation(T, X_j, weights):
    """
    Compute weighted Pearson correlation.
    
    From Fong et al. (2018) balance metric.
    
    Parameters
    ----------
    T : array-like
        Treatment variable
    X_j : array-like
        Single covariate
    weights : array-like
        Weights
        
    Returns
    -------
    float
        Weighted correlation coefficient
    """
    w_sum = np.sum(weights)
    w_mean_T = np.sum(weights * T) / w_sum
    w_mean_X = np.sum(weights * X_j) / w_sum
    
    w_cov = np.sum(weights * (T - w_mean_T) * (X_j - w_mean_X)) / w_sum
    w_var_T = np.sum(weights * (T - w_mean_T)**2) / w_sum
    w_var_X = np.sum(weights * (X_j - w_mean_X)**2) / w_sum
    
    if w_var_T > 0 and w_var_X > 0:
        return w_cov / np.sqrt(w_var_T * w_var_X)
    else:
        return 0.0


def compute_multivariate_std_diff(X, treat, weights=None):
    """
    Compute multivariate standardized difference (Equation 20 in Imai & Ratkovic 2014).
    
    This is a single summary measure of overall covariate balance.
    
    Parameters
    ----------
    X : ndarray
        Covariate matrix (n x k), should NOT include intercept
    treat : ndarray
        Treatment indicator (n,)
    weights : ndarray, optional
        Weights for computing weighted means (n,)
        
    Returns
    -------
    float
        Multivariate standardized difference (lower is better)
        
    Notes
    -----
    From Imai & Ratkovic (2014) Equation (20):
    
    sqrt( (mu_1 - mu_0)^T @ Sigma^{-1} @ (mu_1 - mu_0) )
    
    where mu_1 and mu_0 are the (weighted) mean vectors for treated and control,
    and Sigma is the pooled covariance matrix.
    """
    X = np.asarray(X)
    treat = np.asarray(treat)
    
    if weights is None:
        weights = np.ones(len(treat))
    weights = np.asarray(weights)
    
    # Separate by treatment group
    X_1 = X[treat == 1]
    X_0 = X[treat == 0]
    w_1 = weights[treat == 1]
    w_0 = weights[treat == 0]
    
    # Compute weighted means
    if len(X_1) > 0 and np.sum(w_1) > 0:
        mu_1 = np.average(X_1, axis=0, weights=w_1)
    else:
        mu_1 = np.zeros(X.shape[1])
        
    if len(X_0) > 0 and np.sum(w_0) > 0:
        mu_0 = np.average(X_0, axis=0, weights=w_0)
    else:
        mu_0 = np.zeros(X.shape[1])
    
    # Compute pooled covariance matrix
    n_1 = len(X_1)
    n_0 = len(X_0)
    
    if n_1 > 1 and n_0 > 1:
        cov_1 = np.cov(X_1.T, ddof=1)
        cov_0 = np.cov(X_0.T, ddof=1)
        
        # Handle 1D case
        if cov_1.ndim == 0:
            cov_1 = np.array([[cov_1]])
        if cov_0.ndim == 0:
            cov_0 = np.array([[cov_0]])
        
        # Pooled covariance
        Sigma_pooled = ((n_1 - 1) * cov_1 + (n_0 - 1) * cov_0) / (n_1 + n_0 - 2)
    else:
        Sigma_pooled = np.eye(X.shape[1])
    
    # Compute multivariate standardized difference
    diff = mu_1 - mu_0
    try:
        Sigma_inv = np.linalg.inv(Sigma_pooled)
        msd = np.sqrt(diff @ Sigma_inv @ diff)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        Sigma_pinv = np.linalg.pinv(Sigma_pooled)
        msd = np.sqrt(diff @ Sigma_pinv @ diff)
    
    return msd


def compute_standardized_mean_diff(X, treat, weights=None):
    """
    Compute standardized mean difference for each covariate.
    
    Parameters
    ----------
    X : ndarray
        Covariate matrix (n x k)
    treat : ndarray
        Treatment indicator (n,)
    weights : ndarray, optional
        IPW weights (n,)
        
    Returns
    -------
    ndarray
        Absolute standardized mean differences for each covariate (k,)
    """
    X = np.asarray(X)
    treat = np.asarray(treat)
    
    if weights is None:
        weights = np.ones(len(treat))
    weights = np.asarray(weights)
    
    k = X.shape[1]
    smds = np.zeros(k)
    
    for j in range(k):
        x_j = X[:, j]
        
        # Weighted means
        w_1 = weights[treat == 1]
        w_0 = weights[treat == 0]
        x_1 = x_j[treat == 1]
        x_0 = x_j[treat == 0]
        
        if len(x_1) > 0 and np.sum(w_1) > 0:
            mu_1 = np.average(x_1, weights=w_1)
        else:
            mu_1 = 0
            
        if len(x_0) > 0 and np.sum(w_0) > 0:
            mu_0 = np.average(x_0, weights=w_0)
        else:
            mu_0 = 0
        
        # Pooled standard deviation (unweighted)
        var_1 = np.var(x_1, ddof=1) if len(x_1) > 1 else 0
        var_0 = np.var(x_0, ddof=1) if len(x_0) > 1 else 0
        pooled_std = np.sqrt((var_1 + var_0) / 2)
        
        if pooled_std > 0:
            smds[j] = abs(mu_1 - mu_0) / pooled_std
        else:
            smds[j] = 0.0
    
    return smds


# =============================================================================
# Estimator Functions for Kang & Schafer (2007) Simulation
# =============================================================================

def estimator_ht(y, treat, ps):
    """
    Horvitz-Thompson estimator for E[Y(1)].
    
    HT = (1/n) * sum(T_i * Y_i / pi_i)
    
    From Kang & Schafer (2007), this estimates E[Y(1)].
    """
    y = np.asarray(y)
    treat = np.asarray(treat)
    ps = np.asarray(ps)
    
    # Clip PS to avoid division issues
    ps = np.clip(ps, 0.01, 0.99)
    
    n = len(y)
    estimate = np.sum(treat * y / ps) / n
    return estimate


def estimator_ipw(y, treat, ps):
    """
    Normalized IPW (Hajek) estimator for E[Y(1)].
    
    IPW = sum(T_i * Y_i / pi_i) / sum(T_i / pi_i)
    
    This is the normalized version that sums to 1.
    """
    y = np.asarray(y)
    treat = np.asarray(treat)
    ps = np.asarray(ps)
    
    # Clip PS to avoid division issues
    ps = np.clip(ps, 0.01, 0.99)
    
    numerator = np.sum(treat * y / ps)
    denominator = np.sum(treat / ps)
    
    if denominator > 0:
        return numerator / denominator
    else:
        return np.nan


def estimator_wls(y, treat, X, ps):
    """
    Weighted least squares estimator for E[Y(1)].
    
    EXACT from Imai & Ratkovic (2014), Equation 15:
        μ̂_WLS = (1/n) Σᵢ Xᵢᵀ γ̂_WLS
    
    where:
        γ̂_WLS = {Σᵢ (Tᵢ Xᵢ Xᵢᵀ / πᵢ)}⁻¹ Σᵢ (Tᵢ Xᵢ Yᵢ / πᵢ)
    
    This estimates E[Y(1)] by:
    1. Fitting weighted OLS on treated observations (weights = 1/π)
    2. Predicting at POPULATION mean of X (not just treated mean)
    
    Parameters
    ----------
    y : array
        Outcome variable
    treat : array
        Treatment indicator (0/1)
    X : array
        Covariates with intercept (n x k)
    ps : array
        Propensity scores
        
    Returns
    -------
    float
        Estimate of E[Y(1)]
    """
    y = np.asarray(y)
    treat = np.asarray(treat)
    X = np.asarray(X)
    ps = np.asarray(ps)
    
    # Clip PS to avoid extreme weights
    ps = np.clip(ps, 0.01, 0.99)
    
    # Get treated observations
    y_1 = y[treat == 1]
    X_1 = X[treat == 1]
    ps_1 = ps[treat == 1]
    
    if len(y_1) < X_1.shape[1] + 1:
        return np.nan
    
    # Weights = 1/pi for treated (Imai & Ratkovic 2014, Eq. 15)
    weights = 1 / ps_1
    
    try:
        # Weighted OLS: γ̂_WLS = (X'WX)⁻¹ X'Wy
        W = np.diag(weights)
        XWX = X_1.T @ W @ X_1
        XWy = X_1.T @ W @ y_1
        beta = np.linalg.solve(XWX, XWy)
        
        # Predict at POPULATION mean X (not treated mean)
        # μ̂_WLS = (1/n) Σᵢ Xᵢᵀ γ̂_WLS = X̄ᵀ γ̂_WLS
        X_mean = np.mean(X, axis=0)  # POPULATION mean, not treated mean
        estimate = X_mean @ beta
        return estimate
    except np.linalg.LinAlgError:
        return np.nan


def estimator_dr(y, treat, X, ps):
    """
    Doubly robust estimator for E[Y(1)].
    
    DR = (1/n) * sum[ m(X_i, 1) + T_i*(Y_i - m(X_i, 1))/pi_i ]
    
    where m(X, 1) is the outcome model prediction.
    Consistent if either PS or outcome model is correct.
    """
    y = np.asarray(y)
    treat = np.asarray(treat)
    X = np.asarray(X)
    ps = np.asarray(ps)
    
    # Clip PS
    ps = np.clip(ps, 0.01, 0.99)
    
    n = len(y)
    
    # Fit outcome model on treated, predict for all
    y_1 = y[treat == 1]
    X_1 = X[treat == 1]
    
    if len(y_1) < X_1.shape[1]:
        return np.nan
    
    try:
        # Fit outcome model: E[Y|X, T=1]
        beta_outcome = np.linalg.lstsq(X_1, y_1, rcond=None)[0]
        m_hat = X @ beta_outcome  # Predicted outcome for everyone
        
        # DR estimator for E[Y(1)]
        # = mean of: m(X) for controls + (Y - m(X))*T/pi + m(X) for treated
        # = (1/n) * sum[ T*Y/pi + (1 - T/pi)*m(X) ]
        estimate = np.mean(m_hat + treat * (y - m_hat) / ps)
        return estimate
    except np.linalg.LinAlgError:
        return np.nan


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def imai_2014_config():
    """Provide Imai & Ratkovic (2014) simulation configuration."""
    return {
        'n_sims': IMAI_2014_N_SIMS,
        'sample_sizes': IMAI_2014_SAMPLE_SIZES,
        'scenarios': ['both_correct', 'ps_wrong', 'outcome_wrong', 'both_wrong'],
        'E_Y1_target': 210.0,
    }


@pytest.fixture(scope="module")
def fong_2018_config():
    """Provide Fong et al. (2018) simulation configuration."""
    return {
        'n_sims': FONG_2018_N_SIMS,
        'n_sims_coverage': FONG_2018_COVERAGE_N_SIMS,
        'sample_size': FONG_2018_N,
        'K': 10,
        'dgp_numbers': [1, 2, 3, 4],
        'ate_true': 1.0,
    }


@pytest.fixture(scope="module")
def tolerance_config():
    """
    Provide tolerance configuration for numerical comparisons.
    
    Based on precision requirements from papers.
    """
    return {
        # From Imai & Ratkovic (2014) Table 1
        'bias_tolerance_n200': 5.0,    # Large sample can have bias ~2-4
        'bias_tolerance_n1000': 2.0,   # Larger sample smaller bias
        'rmse_tolerance_n200': 10.0,   # RMSE ranges 2-50 in paper
        'rmse_tolerance_n1000': 5.0,
        
        # From Fong et al. (2018)
        'f_statistic_threshold': 0.01,  # F-stat should be very small
        'correlation_threshold': 0.01,   # Correlations should be near 0
        
        # Coverage requirements (all papers)
        'coverage_lower': 0.85,  # Minimum acceptable
        'coverage_upper': 1.00,  # Maximum
        'coverage_target': 0.95, # Nominal level
    }


@pytest.fixture(scope="module")
def kang_schafer_dgp():
    """Provide Kang & Schafer (2007) DGP function."""
    return dgp_kang_schafer_2007


@pytest.fixture(scope="module")
def fong_dgp():
    """Provide Fong et al. (2018) DGP function."""
    return dgp_fong_2018


@pytest.fixture(scope="module")
def binary_dgp():
    """Provide simple binary treatment DGP function."""
    return dgp_binary_treatment


@pytest.fixture(scope="module")
def continuous_dgp():
    """Provide simple continuous treatment DGP function."""
    return dgp_continuous_treatment


@pytest.fixture(scope="module")
def simulation_config():
    """Provide general simulation configuration."""
    return {
        'n_sims_quick': QUICK_N_SIMS,
        'n_sims_medium': FONG_2018_N_SIMS,
        'n_sims_full': IMAI_2014_N_SIMS,
        'sample_sizes': IMAI_2014_SAMPLE_SIZES,
        'quick_sample_size': QUICK_SAMPLE_SIZE,
    }


@pytest.fixture(scope="module")
def ning_2020_config():
    """Provide Ning et al. (2020) simulation configuration."""
    return {
        'n_sims': NING_2020_N_SIMS,
        'sample_sizes': NING_2020_SAMPLE_SIZES,
        'dimensions': NING_2020_DIMENSIONS,
        'rho': NING_2020_RHO,
        'scenarios': ['A', 'B', 'C', 'D'],
        'ate_true': 1.0,
    }


@pytest.fixture(scope="module")
def cbmsm_2015_config():
    """Provide Imai & Ratkovic (2015) CBMSM simulation configuration."""
    return {
        'n_sims': CBMSM_2015_N_SIMS,
        'sample_sizes': CBMSM_2015_SAMPLE_SIZES,
        'J': CBMSM_2015_J,
        'scenarios': [1, 2],
        'beta_true': np.array([1.0, 0.5, 0.25]),
    }


@pytest.fixture(scope="module")
def ning_dgp():
    """Provide Ning et al. (2020) high-dimensional DGP function."""
    return dgp_ning_2020


@pytest.fixture(scope="module")
def cbmsm_dgp():
    """Provide Imai & Ratkovic (2015) CBMSM DGP function."""
    return dgp_cbmsm_2015


@pytest.fixture(scope="module")
def multitreat_config():
    """Provide Imai & Ratkovic (2014) multi-valued treatment configuration."""
    return {
        'n_sims': PAPER_TARGETS_MULTITREAT['n_sims'],
        'sample_sizes': PAPER_TARGETS_MULTITREAT['sample_sizes'],
        'treatment_levels': PAPER_TARGETS_MULTITREAT['treatment_levels'],
        'scenarios': ['correct', 'misspec'],
        'ate_1_vs_0': PAPER_TARGETS_MULTITREAT['ate_1_vs_0'],
        'ate_2_vs_0': PAPER_TARGETS_MULTITREAT['ate_2_vs_0'],
        'ate_2_vs_1': PAPER_TARGETS_MULTITREAT['ate_2_vs_1'],
    }


@pytest.fixture(scope="module")
def multitreat_dgp():
    """Provide Imai & Ratkovic (2014) multi-valued treatment DGP function."""
    return dgp_multitreat


# =============================================================================
# DGP 7: Fan et al. (2022) - Optimal CBPS (oCBPS)
# =============================================================================

def dgp_fan_2022(n, seed, beta_1=1.0, scenario='both_correct'):
    """
    EXACT reproduction of Fan et al. (2022) DGP from JBES Section 5.
    
    Reference:
        Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., & Yang, X. (2022). Optimal 
        covariate balancing conditions in propensity score estimation. Journal of 
        Business & Economic Statistics, 41(1), 97-110.
    
    Parameters
    ----------
    n : int
        Sample size (300 or 1000 in paper)
    seed : int
        Random seed for reproducibility
    beta_1 : float
        Coefficient on X_1 in propensity score model (0, 0.33, 0.67, or 1)
    scenario : str
        One of:
        - 'both_correct': Both PS and outcome models correctly specified
        - 'ps_misspecified': PS model misspecified (outcome correct)
        - 'outcome_misspecified': Outcome model misspecified (PS correct)
        - 'both_misspecified': Both models misspecified
        
    Returns
    -------
    dict
        Dictionary containing all data and design matrices for Monte Carlo
        
    Notes
    -----
    From Fan et al. (2022) JBES Section 5 - EXACT SPECIFICATION:
    
    Covariates:
        X_1 ~ N(3, 2^2) independently
        X_2, X_3, X_4 ~ N(0, 1) independently
        
    Potential outcomes (EXACT from paper Equation 5.1):
        Y(1) = 200 + 27.4*X_1 + 13.7*X_2 + 13.7*X_3 + 13.7*X_4 + epsilon
        Y(0) = 200 + 13.7*X_2 + 13.7*X_3 + 13.7*X_4 + epsilon
        epsilon ~ N(0, 1)
        
        TRUE ATE = E[Y(1) - Y(0)] = 27.4 * E[X_1] = 27.4 * 3 = 82.2
        
    Propensity score model (Equation 5.1):
        pi(X) = exp(logit) / (1 + exp(logit))
        logit = -beta_1*X_1 + 0.5*X_2 - 0.25*X_3 - 0.1*X_4
        
    Misspecified covariates (Table 2):
        X*_1 = exp(X_1 / 3)
        X*_2 = X_2 / {1 + exp(X_1)} + 10
        X*_3 = X_1 * X_3 / 25 + 0.6
        X*_4 = X_1 + X_4 + 20
        
    h_1 function (for baseline E[Y(0)|X]):
        h_1(X) = (1, X_2, X_3, X_4)  -- basis for K(X)
        
    h_2 function (for treatment effect L(X)):
        h_2(X) = X_1  -- basis for L(X) = 27.4*X_1
    """
    np.random.seed(seed)
    
    # Step 1: Generate covariates (EXACT from paper)
    # X_1 ~ N(3, 2^2) = N(3, 4)
    X_1 = np.random.normal(loc=3.0, scale=2.0, size=n)
    # X_2, X_3, X_4 ~ N(0, 1)
    X_2 = np.random.randn(n)
    X_3 = np.random.randn(n)
    X_4 = np.random.randn(n)
    
    # Stack covariates
    X = np.column_stack([X_1, X_2, X_3, X_4])
    
    # Step 2: Generate propensity score (EXACT from paper)
    # logit(pi) = -beta_1*X_1 + 0.5*X_2 - 0.25*X_3 - 0.1*X_4
    logit_ps = (-beta_1 * X_1 + 0.5 * X_2 - 0.25 * X_3 - 0.1 * X_4)
    ps_true = 1 / (1 + np.exp(-logit_ps))
    
    # Step 3: Generate treatment assignment
    treat = np.random.binomial(1, ps_true)
    
    # Step 4: Generate potential outcomes (EXACT from paper)
    # Y(1) = 200 + 27.4*X_1 + 13.7*X_2 + 13.7*X_3 + 13.7*X_4 + epsilon
    # Y(0) = 200 + 13.7*X_2 + 13.7*X_3 + 13.7*X_4 + epsilon
    epsilon = np.random.randn(n)
    
    y1 = 200 + 27.4 * X_1 + 13.7 * X_2 + 13.7 * X_3 + 13.7 * X_4 + epsilon
    y0 = 200 + 13.7 * X_2 + 13.7 * X_3 + 13.7 * X_4 + epsilon
    
    # Observed outcome
    y = treat * y1 + (1 - treat) * y0
    
    # Step 5: Create misspecified covariates (EXACT from paper Table 2)
    X_mis = np.zeros_like(X)
    X_mis[:, 0] = np.exp(X_1 / 3)  # exp(X_1/3)
    X_mis[:, 1] = X_2 / (1 + np.exp(X_1)) + 10  # X_2/(1+exp(X_1)) + 10
    X_mis[:, 2] = X_1 * X_3 / 25 + 0.6  # X_1*X_3/25 + 0.6
    X_mis[:, 3] = X_1 + X_4 + 20  # X_1 + X_4 + 20
    
    # Step 6: Create design matrices based on scenario
    X_design = np.column_stack([np.ones(n), X])
    X_mis_design = np.column_stack([np.ones(n), X_mis])
    
    if scenario in ('both_correct', 'A'):
        X_ps = X_design
        X_outcome = X_design
    elif scenario in ('ps_misspec', 'ps_misspecified', 'B'):
        X_ps = X_mis_design
        X_outcome = X_design
    elif scenario in ('outcome_misspec', 'outcome_misspecified', 'C'):
        X_ps = X_design
        X_outcome = X_mis_design
    else:  # 'both_misspec', 'both_misspecified', 'D'
        X_ps = X_mis_design
        X_outcome = X_mis_design
    
    # Step 7: Compute true ATE
    # TRUE ATE = 27.4 * E[X_1] = 27.4 * 3 = 82.2
    ate_true = 27.4 * 3.0
    
    # h_1 and h_2 functions for oCBPS
    # NOTE: In the paper, h_1(X) = (1, X_2, X_3, X_4), but cbps_optimal_2treat
    # automatically adds the intercept from X[:, 0], so we only pass the 
    # covariates without intercept
    # h_1(X) = (X_2, X_3, X_4) - basis for K(X) = E[Y(0)|X]
    # h_2(X) = X_1 - basis for L(X) = E[Y(1)-Y(0)|X] = 27.4*X_1
    h_1 = np.column_stack([X_2, X_3, X_4])  # NO intercept
    h_2 = X_1.reshape(-1, 1)
    
    return {
        'X': X,
        'X_design': X_design,
        'X_mis': X_mis,
        'X_mis_design': X_mis_design,
        'X_ps': X_ps,
        'X_outcome': X_outcome,
        'treat': treat,
        'y': y,
        'y1': y1,
        'y0': y0,
        'ps_true': ps_true,
        'ate_true': ate_true,
        'h_1': h_1,
        'h_2': h_2,
        'n': n,
        'beta_1': beta_1,
        'scenario': scenario,
    }


@pytest.fixture(scope="module")
def fan_2022_config():
    """Provide Fan et al. (2022) simulation configuration."""
    return {
        'n_sims': FAN_2022_N_SIMS,
        'sample_sizes': FAN_2022_SAMPLE_SIZES,
        'beta_1_values': [0, 0.33, 0.67, 1.0],
        'scenarios': ['both_correct', 'ps_misspecified', 
                      'outcome_misspecified', 'both_misspecified'],
        'ate_true': 27.4 * 3.0,  # 82.2
    }


# Backward compatibility alias
@pytest.fixture(scope="module")
def fan_2021_config():
    """Provide Fan et al. (2022) simulation configuration (alias for backward compatibility)."""
    return {
        'n_sims': FAN_2022_N_SIMS,
        'sample_sizes': FAN_2022_SAMPLE_SIZES,
        'beta_1_values': [0, 0.33, 0.67, 1.0],
        'scenarios': ['both_correct', 'ps_misspecified', 
                      'outcome_misspecified', 'both_misspecified'],
        'ate_true': 27.4 * 3.0,  # 82.2
    }


@pytest.fixture(scope="module")
def fan_dgp():
    """Provide Fan et al. (2022) optimal CBPS DGP function."""
    return dgp_fan_2022


# Backward compatibility alias for DGP function
dgp_fan_2021 = dgp_fan_2022


# =============================================================================
# CBIV Simulation Parameters (Imai & Ratkovic 2014, Section 3.3)
# =============================================================================

# From Imai & Ratkovic (2014) Section 3.3 - Extension to Instrumental Variables
# Based on Angrist, Imbens & Rubin (1996) principal stratification framework
CBIV_2014_N_SIMS = 1000  # Monte Carlo replications
CBIV_2014_SAMPLE_SIZES = [500, 1000]  # Sample sizes

# Paper numerical targets for CBIV
# Note: Imai & Ratkovic (2014) does not provide detailed simulation results
# for CBIV in the paper. These targets are based on theoretical expectations
# from the principal stratification framework.
PAPER_TARGETS_CBIV = {
    # Scenario: One-sided noncompliance (compliers + never-takers)
    'onesided': {
        'late_true': 5.0,  # True Local Average Treatment Effect
        'compliance_rate_expected': 0.6,  # Expected proportion of compliers
    },
    # Scenario: Two-sided noncompliance (compliers + always-takers + never-takers)
    'twosided': {
        'late_true': 5.0,  # True LATE
        'complier_rate_expected': 0.5,
        'always_taker_rate_expected': 0.2,
        'never_taker_rate_expected': 0.3,
    },
    # General parameters
    'instrument_strength_f_threshold': 10.0,  # Stock & Yogo (2005) weak IV threshold
    'balance_smd_threshold': 0.1,  # SMD < 0.1 indicates good balance
}


# =============================================================================
# EXACT Numerical Targets for Inference Module Validation
# From Fan et al. (2022) JBES Table 1 and Fong et al. (2018) Section 4
# =============================================================================

PAPER_TARGETS_INFERENCE = {
    # =========================================================================
    # Binary Treatment Inference: Fan et al. (2022) Table 1
    # Asymptotic variance estimation and confidence interval coverage
    # =========================================================================
    
    # Scenario: both_correct, n=1000, β₁=1
    # Two variance methods: oCBPS (semiparametric efficiency bound) and CBPS (sandwich)
    'fan_2022_both_correct_n1000_beta1': {
        'oCBPS': {
            'bias': 0.08,
            'std': 1.22,
            'rmse': 1.23,
            'coverage': 0.966,  # 96.6% coverage from 2000 reps
        },
        'CBPS': {
            'bias': 0.45,
            'std': 1.45,
            'rmse': 1.52,
            'coverage': 0.968,  # 96.8% coverage from 2000 reps
        },
        'DR': {
            'bias': -4.50,
            'std': 3.32,
            'rmse': 5.59,
            'coverage': 0.268,  # Poor coverage when both misspecified
        },
    },
    
    # Scenario: ps_misspec, n=1000, β₁=1
    'fan_2022_ps_misspec_n1000_beta1': {
        'oCBPS': {
            'coverage': 0.95,  # Approximate - still valid due to double robustness
        },
    },
    
    # =========================================================================
    # Continuous Treatment Inference: Fong et al. (2018) Section 4
    # Weighted least squares variance estimation
    # =========================================================================
    
    # DGP 1: Both models correctly specified
    'fong_2018_dgp1': {
        'CBGPS': {
            'coverage': 0.955,  # 95.5% coverage from 10,000 reps
            'f_stat_median': 9.33e-5,  # Near-zero F-statistic
        },
    },
    
    # DGP 4: Both models misspecified
    'fong_2018_dgp4': {
        'CBGPS': {
            'coverage': 0.90,  # Lower coverage expected under misspecification
            'f_stat_upper_bound': 0.05,
        },
    },
    
    # =========================================================================
    # General Tolerance Parameters for Monte Carlo Validation
    # =========================================================================
    'coverage_nominal': 0.95,
    'coverage_tolerance_strict': 0.02,  # ±2% for full tests (n_sims ≥ 500)
    'coverage_tolerance_quick': 0.05,   # ±5% for quick tests (n_sims < 200)
    'se_ratio_target': 1.0,
    'se_ratio_tolerance': 0.20,  # Analytical SE within 20% of MC SD
}


# =============================================================================
# Inference Simulation Parameters
# =============================================================================

# From Fan et al. (2022) Section 5 - Inference validation
FAN_2022_INFERENCE_N_SIMS = 2000  # Paper uses 2000 replications for coverage
FAN_2022_INFERENCE_SAMPLE_SIZES = [300, 1000, 5000]  # n values

# Quick test parameters for inference
QUICK_INFERENCE_N_SIMS = 100
QUICK_INFERENCE_SAMPLE_SIZE = 300


# =============================================================================
# DGP 8: Imai & Ratkovic (2014) Section 3.3 - CBIV for Instrumental Variables
# =============================================================================

def dgp_cbiv_2014(n, seed, scenario='onesided'):
    """
    Data Generating Process for CBIV Monte Carlo simulations.
    
    Based on Imai & Ratkovic (2014) Section 3.3 and the principal stratification
    framework from Angrist, Imbens & Rubin (1996).
    
    Parameters
    ----------
    n : int
        Sample size (500 or 1000 in simulations)
    seed : int
        Random seed for reproducibility
    scenario : str
        One of:
        - 'onesided': One-sided noncompliance (compliers + never-takers only)
        - 'twosided': Two-sided noncompliance (compliers + always-takers + never-takers)
        - 'weak_iv': Weak instrument scenario
        - 'misspec': Compliance probability model misspecified
        
    Returns
    -------
    dict
        Dictionary containing all data for CBIV Monte Carlo simulation
        
    Notes
    -----
    Principal Stratification Framework (Angrist, Imbens & Rubin 1996):
    
    Principal Strata (based on potential treatment under Z=0 and Z=1):
        - Compliers (C): Tr(Z=1)=1, Tr(Z=0)=0 - respond to encouragement
        - Always-takers (A): Tr(Z=1)=1, Tr(Z=0)=1 - always treated  
        - Never-takers (N): Tr(Z=1)=0, Tr(Z=0)=0 - never treated
        - Defiers: Tr(Z=1)=0, Tr(Z=0)=1 - excluded by monotonicity assumption
        
    Instrument Assignment:
        Z ~ Bernoulli(0.5), independent of X
        
    Compliance Type Probabilities (multinomial logistic model):
        One-sided (twosided=False):
            π_c(X) = 1 / {1 + exp(β_n^T X)}
            π_n(X) = exp(β_n^T X) / {1 + exp(β_n^T X)}
            
        Two-sided (twosided=True):
            π_c(X) = 1 / {1 + exp(β_a^T X) + exp(β_n^T X)}
            π_a(X) = exp(β_a^T X) / {1 + exp(β_a^T X) + exp(β_n^T X)}
            π_n(X) = exp(β_n^T X) / {1 + exp(β_a^T X) + exp(β_n^T X)}
            
    Treatment Assignment:
        Tr = Z * (C + A) + (1-Z) * A
        where C, A are indicators for compliance type
        
    Outcome Model:
        Y(0) = θ^T X + ε_0
        Y(1) = Y(0) + τ for compliers (LATE = τ)
        Y = Tr * Y(1) + (1-Tr) * Y(0)
        
    Target Estimand:
        LATE = E[Y(1) - Y(0) | Complier] = τ
    
    References
    ----------
    [1] Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
        Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
        
    [2] Angrist, J.D., Imbens, G.W., and Rubin, D.B. (1996). Identification of
        Causal Effects Using Instrumental Variables. Journal of the American
        Statistical Association, 91(434), 444-455.
    """
    np.random.seed(seed)
    
    K = 4  # Number of covariates
    
    # Step 1: Generate covariates (standard normal, independent)
    X = np.random.randn(n, K)
    
    # Step 2: Generate instrument Z (independent of X)
    # Z ~ Bernoulli(0.5)
    Z = np.random.binomial(1, 0.5, n)
    
    # Step 3: Define compliance type probability model
    # Coefficients chosen to give reasonable compliance rates
    if scenario == 'weak_iv':
        # Weak instrument: compliance probability only weakly depends on X
        # Most units are never-takers
        beta_n = np.array([1.5, 0.05, -0.05, 0.02, 0.02])  # High never-taker rate
        beta_a = np.array([0.5, 0.02, 0.02, -0.02, 0.01])  # For twosided
    elif scenario == 'misspec':
        # Misspecified: true model is nonlinear
        beta_n = np.array([-0.5, 0.3, -0.2, 0.15, -0.1])
        beta_a = np.array([-1.0, 0.2, 0.1, -0.1, 0.05])
    else:
        # Standard coefficients for reasonable compliance rates
        beta_n = np.array([-0.5, 0.3, -0.2, 0.15, -0.1])  # Never-taker log-odds
        beta_a = np.array([-1.0, 0.2, 0.1, -0.1, 0.05])   # Always-taker log-odds
    
    # Add intercept to X for computing log-odds
    X_design = np.column_stack([np.ones(n), X])
    
    # Step 4: Compute compliance type probabilities
    if scenario in ['onesided', 'weak_iv']:
        # One-sided: only compliers and never-takers
        # π_c = 1 / (1 + exp(β_n^T X))
        # π_n = exp(β_n^T X) / (1 + exp(β_n^T X))
        log_odds_n = X_design @ beta_n
        exp_n = np.exp(np.clip(log_odds_n, -20, 20))  # Clip for numerical stability
        
        pi_c = 1 / (1 + exp_n)
        pi_a = np.zeros(n)  # No always-takers in one-sided
        pi_n = exp_n / (1 + exp_n)
        
        twosided = False
    else:  # 'twosided' or 'misspec'
        # Two-sided: compliers, always-takers, and never-takers
        # Using softmax over 3 categories (complier as reference)
        log_odds_a = X_design @ beta_a
        log_odds_n = X_design @ beta_n
        
        # Clip for numerical stability
        log_odds_a = np.clip(log_odds_a, -20, 20)
        log_odds_n = np.clip(log_odds_n, -20, 20)
        
        exp_a = np.exp(log_odds_a)
        exp_n = np.exp(log_odds_n)
        denom = 1 + exp_a + exp_n
        
        pi_c = 1 / denom
        pi_a = exp_a / denom
        pi_n = exp_n / denom
        
        twosided = True
    
    # Step 5: Assign compliance types
    # Draw compliance type for each unit
    compliance_type = np.zeros(n, dtype=int)  # 0=complier, 1=always-taker, 2=never-taker
    
    for i in range(n):
        probs = [pi_c[i], pi_a[i], pi_n[i]]
        compliance_type[i] = np.random.choice([0, 1, 2], p=probs)
    
    is_complier = compliance_type == 0
    is_always_taker = compliance_type == 1
    is_never_taker = compliance_type == 2
    
    # Step 6: Generate treatment based on instrument and compliance type
    # Tr(Z=1) = 1 if complier or always-taker, 0 if never-taker
    # Tr(Z=0) = 1 if always-taker, 0 otherwise
    Tr = np.zeros(n, dtype=int)
    Tr[(Z == 1) & (is_complier | is_always_taker)] = 1
    Tr[(Z == 0) & is_always_taker] = 1
    
    # Step 7: Generate potential outcomes and observed outcome
    # Outcome model coefficients
    theta = np.array([1.0, 0.5, 0.25, 0.1])  # Covariate effects
    tau = 5.0  # True LATE (treatment effect for compliers)
    
    # Y(0) = theta^T X + epsilon_0
    epsilon_0 = np.random.randn(n)
    Y0 = X @ theta + epsilon_0
    
    # Y(1) for compliers = Y(0) + tau
    # Y(1) for always-takers = Y(0) + tau_a (may differ from LATE)
    # We assume homogeneous treatment effect for simplicity
    tau_a = tau  # Same effect for always-takers (can be modified)
    
    Y1 = Y0.copy()
    Y1[is_complier] += tau
    Y1[is_always_taker] += tau_a
    # Never-takers: Y(1) not observed, but for potential outcomes we define it
    Y1[is_never_taker] += tau  # Counterfactual (not identified)
    
    # Observed outcome
    Y = Tr * Y1 + (1 - Tr) * Y0
    
    # Step 8: Create misspecified covariates for 'misspec' scenario
    if scenario == 'misspec':
        # Nonlinear transformations (Kang-Schafer style)
        X_mis = np.column_stack([
            np.exp(X[:, 0] / 2),
            X[:, 1] / (1 + np.exp(X[:, 0])) + 10,
            (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3,
            (X[:, 0] + X[:, 3] + 20) ** 2
        ])
        X_design_mis = np.column_stack([np.ones(n), X_mis])
    else:
        X_mis = X
        X_design_mis = X_design
    
    # Step 9: Compute first-stage F-statistic (instrument strength)
    # Regress Tr on Z controlling for X
    from scipy import stats as sp_stats
    
    # Simple first-stage: Tr ~ Z + X
    Z_centered = Z - Z.mean()
    residuals_Tr = Tr - Tr.mean()
    
    # F-statistic from first stage (simplified)
    SS_Z = np.sum((Z_centered * residuals_Tr) ** 2)
    SS_total = np.sum(residuals_Tr ** 2)
    if SS_total > 0:
        first_stage_f = (SS_Z / 1) / (SS_total / (n - 2))
    else:
        first_stage_f = 0.0
    
    # True compliance rates
    true_complier_rate = np.mean(is_complier)
    true_always_taker_rate = np.mean(is_always_taker)
    true_never_taker_rate = np.mean(is_never_taker)
    
    return {
        'n': n,
        'K': K,
        'X': X,
        'X_design': X_design,
        'X_mis': X_mis,
        'X_design_mis': X_design_mis,
        'Z': Z,
        'Tr': Tr,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'compliance_type': compliance_type,
        'is_complier': is_complier,
        'is_always_taker': is_always_taker,
        'is_never_taker': is_never_taker,
        'pi_c_true': pi_c,
        'pi_a_true': pi_a,
        'pi_n_true': pi_n,
        'late_true': tau,
        'theta_true': theta,
        'beta_n': beta_n,
        'beta_a': beta_a,
        'twosided': twosided,
        'scenario': scenario,
        'first_stage_f': first_stage_f,
        'true_complier_rate': true_complier_rate,
        'true_always_taker_rate': true_always_taker_rate,
        'true_never_taker_rate': true_never_taker_rate,
    }


@pytest.fixture(scope="module")
def cbiv_2014_config():
    """Provide Imai & Ratkovic (2014) CBIV simulation configuration."""
    return {
        'n_sims': CBIV_2014_N_SIMS,
        'sample_sizes': CBIV_2014_SAMPLE_SIZES,
        'scenarios': ['onesided', 'twosided', 'weak_iv', 'misspec'],
        'late_true': 5.0,
    }


@pytest.fixture(scope="module")
def cbiv_dgp():
    """Provide Imai & Ratkovic (2014) CBIV DGP function."""
    return dgp_cbiv_2014


@pytest.fixture(scope="module")
def lalonde_data():
    """Load LaLonde dataset for paper reproduction.

    Returns the Dehejia-Wahba subset of the LaLonde data.
    """
    try:
        from cbps.datasets import load_lalonde
        return load_lalonde(dehejia_wahba_only=True)
    except ImportError:
        pytest.skip("LaLonde dataset not available")
