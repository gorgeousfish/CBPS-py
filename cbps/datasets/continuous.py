"""
Continuous Treatment Dataset Loaders
=====================================

This module provides datasets for continuous treatment CBPS methods, including
simulation data and real-world application data from Fong, Hazlett, and Imai (2018).

The simulation data enables systematic evaluation of CBPS estimators under
various model misspecification scenarios. The political ads dataset provides
a real-world application for continuous treatment effect estimation.

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.
DOI: 10.1214/17-AOAS1101

Urban, C. and Niebler, S. (2014). Dollars on the sidewalk: Should U.S.
Presidential candidates advertise in uncontested states? *American Journal
of Political Science*, 58(2), 322-336.
"""
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def load_continuous_simulation(
    dgp: int = 1,
    seed: int = 12345
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load continuous treatment simulation data (Fong et al. 2018, Section 4).

    Parameters
    ----------
    dgp : {1, 2, 3, 4}, default=1
        Data generating process type:

        - **DGP1**: Both treatment and outcome correctly specified (linear)
        - **DGP2**: Treatment model misspecified (contains (X2+0.5)² nonlinear term)
        - **DGP3**: Outcome model misspecified (contains 2(X2+0.5)² nonlinear term)
        - **DGP4**: Doubly misspecified (both models incorrect)

    seed : int, default=12345
        Random seed (current CSV files use seed=12345).

    Returns
    -------
    data : DataFrame
        Simulation data (200 observations × 12 columns: T, Y, X1-X10).
    metadata : dict
        Metadata dictionary containing:

        - ``'dgp'``: DGP type (1-4)
        - ``'n'``: Number of observations (200)
        - ``'k'``: Number of covariates (10)
        - ``'true_ate'``: True ATE (1.0)
        - ``'cov_structure'``: Covariance structure (equicorrelated MVN, ρ=0.2)

    Notes
    -----
    Based on Fong, Hazlett, and Imai (2018) Section 4.1 simulation design.

    **Column Ordering**: T (treatment), Y (outcome), X1-X10 (covariates).

    **Paper Benchmarks** (500 Monte Carlo replications):

    - DGP1: CBGPS balance F-statistic < 0.01 (near-perfect balance)
    - DGP2: CBGPS F < 0.1 (robust despite treatment misspecification)
    - DGP3: CBGPS F < 0.1 (robust despite outcome misspecification)
    - DGP4: All methods biased (double misspecification, expected)

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment: Application to the efficacy of political
    advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.

    Examples
    --------
    >>> data, meta = load_continuous_simulation(dgp=1, seed=12345)
    >>> data.shape
    (200, 12)
    >>> meta['true_ate']
    1.0
    >>> list(data.columns)
    ['T', 'Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    """
    if dgp not in [1, 2, 3, 4]:
        raise ValueError(f"dgp must be one of [1, 2, 3, 4], got: {dgp}")
    
    # Get data directory (supports both installed and development modes)
    package_data_dir = Path(__file__).parent.parent / "data"
    if package_data_dir.exists():
        data_dir = package_data_dir
    else:
        # Fall back to project root data directory (development mode)
        current_file = Path(__file__)
        cbps_package = current_file.parent.parent
        cbps_python = cbps_package.parent
        project_root = cbps_python.parent
        data_dir = project_root / "data"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Cannot find data directory. Tried:\n"
                f"  1. Package data: {package_data_dir}\n"
                f"  2. Project data: {data_dir}\n"
                f"Please ensure the package is properly installed with data files."
            )

    # Construct filename
    filename = f"simulation_dgp{dgp}_seed{seed}.csv"
    filepath = data_dir / filename
    
    # Read CSV file
    df = pd.read_csv(filepath, dtype=np.float64)
    
    # Metadata
    metadata = {
        'dgp': dgp,
        'n': 200,
        'k': 10,
        'true_ate': 1.0,
        'cov_structure': 'Equicorrelated MVN with rho=0.2',
        'sigma_xi_sq': 4.0 if dgp == 1 else 2.25,
        'sigma_epsilon_sq': 25.0,
        'misspecified_treatment': dgp in [2, 4],
        'misspecified_outcome': dgp in [3, 4]
    }
    
    return df, metadata


def load_political_ads() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load political advertising data (Urban and Niebler 2014).

    Returns
    -------
    data : DataFrame
        Political advertising data (16265 observations × 155 columns).
    metadata : dict
        Metadata containing:

        - ``'n'``: 16265 (number of observations)
        - ``'p'``: 155 (number of columns)
        - ``'treatment_col'``: 'TotAds' (total advertising count)
        - ``'boxcox_lambda'``: -0.16 (transformation parameter)
        - ``'paper_f_stat_cbgps'``: 9.33e-5 (paper benchmark)

    Notes
    -----
    **Treatment Variable**: TotAds (advertising count, range: 0-22,380).

    The treatment requires Box-Cox transformation (λ=-0.16) for approximate
    normality before applying continuous treatment CBPS.

    **Paper Benchmarks** (Fong et al. 2018, Section 5):

    - Table 1: 15 covariates with Pearson correlation < 0.001 (CBGPS)
    - Figure 3: Full model F-statistic = 9.33e-5 (CBGPS) vs 215.3 (MLE)

    References
    ----------
    Urban, C. and Niebler, S. (2014). Dollars on the sidewalk: Should U.S.
    Presidential candidates advertise in uncontested states? *American Journal
    of Political Science*, 58(2), 322-336.

    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. *The Annals of Applied Statistics*,
    12(1), 156-177.

    Examples
    --------
    >>> data, meta = load_political_ads()
    >>> data.shape
    (16265, 155)
    >>> 'TotAds' in data.columns
    True
    >>> meta['boxcox_lambda']
    -0.16
    """
    # Get data directory (supports both installed and development modes)
    package_data_dir = Path(__file__).parent.parent / "data"
    if package_data_dir.exists():
        data_dir = package_data_dir
    else:
        # Fall back to project root data directory (development mode)
        current_file = Path(__file__)
        cbps_package = current_file.parent.parent
        cbps_python = cbps_package.parent
        project_root = cbps_python.parent
        data_dir = project_root / "data"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Cannot find data directory. Tried:\n"
                f"  1. Package data: {package_data_dir}\n"
                f"  2. Project data: {data_dir}\n"
                f"Please ensure the package is properly installed with data files."
            )

    filepath = data_dir / "political_ads_urban_niebler.csv"
    
    # Read CSV file
    df = pd.read_csv(filepath, low_memory=False)
    
    metadata = {
        'n': 16265,
        'p': 155,
        'treatment_col': 'TotAds',
        'treatment_range': (0, 22380),
        'boxcox_lambda': -0.16,
        'paper_f_stat_cbgps': 9.33e-5,
        'paper_f_stat_mle': 215.3,
        'covariates_count': 15,
        'squared_terms_count': 8
    }
    
    return df, metadata


