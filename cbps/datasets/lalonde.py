"""
LaLonde Dataset Loaders
========================

This module provides functions to load the National Supported Work (NSW)
demonstration dataset, widely used as a benchmark in propensity score and
causal inference research.

The NSW demonstration was a labor training program conducted in the mid-1970s.
LaLonde (1986) used this dataset to evaluate econometric methods for estimating
treatment effects from observational data by comparing results to experimental
benchmarks.

References
----------
LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
programs with experimental data. *American Economic Review*, 76(4), 604-620.

Dehejia, R. H. and Wahba, S. (1999). Causal effects in nonexperimental
studies: Reevaluating the evaluation of training programs. *Journal of the
American Statistical Association*, 94(448), 1053-1062.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
import numpy as np


def _get_data_dir() -> Path:
    """
    Get the absolute path to the data directory.

    Supports two modes:
    1. Installed mode: Load from package data directory (site-packages/cbps/data/)
    2. Development mode: Fall back to project root data directory

    Priority is given to the package data directory; if it does not exist,
    falls back to the project root data directory.

    Returns
    -------
    Path
        Absolute path to the data directory.
    """
    # Option 1: Try package data directory (installed mode)
    package_data_dir = Path(__file__).parent.parent / "data"
    if package_data_dir.exists():
        return package_data_dir

    # Option 2: Fall back to project root data directory (development mode)
    current_file = Path(__file__)
    cbps_package = current_file.parent.parent
    cbps_python = cbps_package.parent
    project_root = cbps_python.parent
    project_data_dir = project_root / "data"

    if project_data_dir.exists():
        return project_data_dir

    # Raise clear error if neither exists
    raise FileNotFoundError(
        f"Cannot find data directory. Tried:\n"
        f"  1. Package data: {package_data_dir}\n"
        f"  2. Project data: {project_data_dir}\n"
        f"Please ensure the package is properly installed with data files."
    )


def load_lalonde(
    dehejia_wahba_only: bool = True,
    return_X_y: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Load the LaLonde dataset (NSW job training program evaluation).

    Parameters
    ----------
    dehejia_wahba_only : bool, default=True
        If True, load the Dehejia-Wahba subsample (445 observations).
        If False, load the full LaLonde dataset (3212 observations).
    return_X_y : bool, default=False
        If True, return (X, y) tuple; otherwise return DataFrame.

    Returns
    -------
    DataFrame or (DataFrame, Series)
        If return_X_y=False: Complete DataFrame with standardized column names.
        If return_X_y=True: (covariates, treatment) tuple.

    Notes
    -----
    The dataset contains data from the National Supported Work (NSW)
    demonstration, a labor training program conducted in the mid-1970s.

    **Variables**

    - ``treat``: Treatment indicator (1=received training, 0=control)
    - ``age``: Age in years
    - ``educ``: Years of education
    - ``black``: African American indicator
    - ``hisp``: Hispanic indicator
    - ``married``: Marital status indicator
    - ``nodegr``: No high school degree indicator
    - ``re74``, ``re75``: Real earnings in 1974, 1975 (pre-treatment)
    - ``re78``: Real earnings in 1978 (outcome)

    **Data Dimensions**

    - Dehejia-Wahba subsample: 445 observations, 11 columns
    - Full dataset: 3212 observations, 12 columns

    References
    ----------
    LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
    programs with experimental data. *American Economic Review*, 76(4), 604-620.

    Dehejia, R. H. and Wahba, S. (1999). Causal effects in nonexperimental
    studies: Reevaluating the evaluation of training programs. *Journal of the
    American Statistical Association*, 94(448), 1053-1062.

    Examples
    --------
    >>> df = load_lalonde(dehejia_wahba_only=True)
    >>> df.shape
    (445, 11)
    >>> 'educ' in df.columns
    True
    """
    data_dir = _get_data_dir()
    
    if dehejia_wahba_only:
        # Load Dehejia-Wahba subsample
        filepath = data_dir / "nsw_dw.csv"
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'education': 'educ',
            'hispanic': 'hisp', 
            'nodegree': 'nodegr'
        }
        df = df.rename(columns=column_mapping)
        
        # Set appropriate dtypes
        df = df.astype({
            'treat': np.int64, 'age': np.int64, 'educ': np.int64,
            'black': np.int64, 'hisp': np.int64, 'married': np.int64,
            'nodegr': np.int64, 're74': np.float64, 're75': np.float64,
            're78': np.float64
        })
    else:
        # Load full LaLonde dataset
        filepath = data_dir / "LaLonde.csv"
        df = pd.read_csv(filepath)
        # Column names already standardized in this file
    
    if return_X_y:
        y = df['treat']
        X = df.drop('treat', axis=1)
        return X, y
    else:
        return df


def load_lalonde_psid_combined(
    psid_version: str = 'main',
    return_combined: bool = True
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Load NSW experimental and PSID control group data for evaluation bias tests.

    Parameters
    ----------
    psid_version : {'main', 'controls2', 'controls3'}, default='main'
        PSID control group version:

        - ``'main'``: Full PSID controls (2490 observations)
        - ``'controls2'``: Restricted PSID controls (253 observations)
        - ``'controls3'``: Further restricted PSID controls (128 observations)

    return_combined : bool, default=True
        If True (default), return combined DataFrame.
        If False, return (nsw_exp, psid_ctrl) tuple.

    Returns
    -------
    combined : DataFrame (default)
        Combined DataFrame with standardized column names.

        - main version: (3212, 11)
        - controls2 version: (975, 11)
        - controls3 version: (850, 11)

    (nsw_exp, psid_ctrl) : tuple of DataFrame (when return_combined=False)
        Tuple of NSW experimental sample and PSID control sample.

        - nsw_exp: (722, 10) - NSW experimental sample (missing re74)
        - psid_ctrl: (2490, 11) or (253, 11) or (128, 11) - PSID controls

    Notes
    -----
    Used for reproducing evaluation bias tests from Smith and Todd (2005) and
    the original CBPS paper. The NSW sample contains experimental participants,
    while the PSID samples serve as observational comparison groups.

    **Data Dimensions**

    - main: nsw=(722, 10), psid=(2490, 11), combined=(3212, 11)
    - controls2: nsw=(722, 10), psid=(253, 11), combined=(975, 11)
    - controls3: nsw=(722, 10), psid=(128, 11), combined=(850, 11)

    References
    ----------
    Smith, J. A. and Todd, P. E. (2005). Does matching overcome LaLonde's
    critique of nonexperimental estimators? *Journal of Econometrics*,
    125(1-2), 305-353.

    Examples
    --------
    >>> df = load_lalonde_psid_combined()
    >>> df.shape
    (3212, 11)
    >>> int((df['treat'] == 0).sum()), int((df['treat'] == 1).sum())
    (2915, 297)
    >>> 'educ' in df.columns
    True

    >>> # Return separate DataFrames
    >>> nsw_exp, psid_ctrl = load_lalonde_psid_combined(return_combined=False)
    >>> nsw_exp.shape, psid_ctrl.shape
    ((722, 10), (2490, 11))
    """
    data_dir = _get_data_dir()
    
    # Load NSW experimental sample (722 observations, 10 columns - missing re74)
    nsw = pd.read_csv(data_dir / "nsw.csv")
    
    # Standardize column names to match package convention
    column_mapping = {
        'education': 'educ',
        'hispanic': 'hisp',
        'nodegree': 'nodegr'
    }
    nsw = nsw.rename(columns=column_mapping)

    # Load corresponding PSID control group
    psid_files = {
        'main': 'psid_controls.csv',
        'controls2': 'psid_controls2.csv',
        'controls3': 'psid_controls3.csv'
    }
    
    if psid_version not in psid_files:
        raise ValueError(
            f"psid_version must be one of {list(psid_files.keys())}, "
            f"got: {psid_version}"
        )
    
    psid = pd.read_csv(data_dir / psid_files[psid_version])
    # Standardize column names for PSID control sample
    psid = psid.rename(columns=column_mapping)

    if return_combined:
        # Return combined DataFrame with consistent column ordering
        # NSW sample is missing re74, which will be filled with NaN
        nsw_with_re74 = nsw.copy()
        if 're74' not in nsw_with_re74.columns:
            nsw_with_re74['re74'] = np.nan
        combined = pd.concat([nsw_with_re74, psid], ignore_index=True)
        return combined
    else:
        # Return separate DataFrames: (nsw_exp, psid_ctrl)
        return nsw, psid
