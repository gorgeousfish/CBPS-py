"""
Blackwell Campaign Advertising Dataset
=======================================

This module provides the Blackwell (2013) longitudinal dataset for analyzing
the causal effects of negative campaign advertising on electoral outcomes.
The data consists of U.S. Senate and gubernatorial candidates observed over
multiple weeks leading up to elections.

The dataset is designed for marginal structural model (MSM) estimation with
time-varying treatments and confounders, as described in Imai and Ratkovic (2015).

References
----------
Blackwell, M. (2013). A framework for dynamic causal inference in political
science. *American Journal of Political Science*, 57(2), 504-519.

Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. *Journal of the American Statistical
Association*, 110(511), 1013-1023.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_blackwell() -> pd.DataFrame:
    """Load the Blackwell negative campaign advertising dataset.

    This dataset contains longitudinal observations of U.S. Senate and
    gubernatorial candidates during the five weeks leading up to elections.
    It is designed for marginal structural model estimation with time-varying
    treatments.

    Returns
    -------
    pd.DataFrame
        Panel data with 570 observations (114 candidates × 5 time periods)
        and 15 columns.

    Notes
    -----
    **Data Structure**

    - Units: 114 candidates (identified by ``demName``)
    - Time periods: 5 weeks (``time`` = 1, 2, 3, 4, 5)
    - Treatment: ``d.gone.neg`` (binary indicator for negative advertising)
    - Outcome: ``demprcnt`` (Democratic vote share percentage)

    **Key Variables**

    - ``demName``: Candidate identifier (string)
    - ``d.gone.neg``: Whether candidate ran negative ads this period (0/1)
    - ``d.gone.neg.l1``: Lagged treatment (previous period)
    - ``d.gone.neg.l2``: Twice-lagged treatment
    - ``camp.length``: Campaign duration
    - ``demprcnt``: Democratic vote share (outcome)
    - ``time``: Time period indicator (1-5)

    References
    ----------
    Blackwell, M. (2013). A framework for dynamic causal inference in
    political science. *American Journal of Political Science*, 57(2), 504-520.

    Examples
    --------
    >>> from cbps.datasets import load_blackwell
    >>> df = load_blackwell()
    >>> df.shape
    (570, 15)
    >>> df['demName'].nunique()  # Number of candidates
    114
    >>> df['time'].unique().tolist()  # Time periods
    [1, 2, 3, 4, 5]
    """
    # Get data directory path (supports both installed and development modes)
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

    filepath = data_dir / "Blackwell.csv"
    
    df = pd.read_csv(
        filepath,
        dtype={
            'd.gone.neg': np.int64,
            'd.gone.neg.l1': np.int64,
            'd.gone.neg.l2': np.int64,
            'deminc': np.int64,
            'time': np.int64,
            'demName': str,
        }
    )
    
    return df

