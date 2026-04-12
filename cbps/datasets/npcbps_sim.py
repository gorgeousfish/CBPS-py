"""
npCBPS Simulation Data Loaders
===============================

This module provides functions to load continuous treatment simulation data
for validating nonparametric CBPS (npCBPS) implementations.

The npCBPS methodology estimates stabilizing inverse propensity score weights
using empirical likelihood without requiring a parametric model for the
generalized propensity score, as described in Fong, Hazlett, and Imai (2018).

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.
"""

from pathlib import Path
import pandas as pd


def load_npcbps_continuous_sim() -> pd.DataFrame:
    """Load npCBPS continuous treatment simulation data.

    Returns
    -------
    pd.DataFrame
        Simulation data (500 rows × 7 columns).

    Notes
    -----
    **Variables**

    - ``Y``: Outcome variable (continuous)
    - ``T``: Treatment variable (continuous)
    - ``X1``-``X5``: Covariates (multivariate normal, correlation=0.5)

    **Data Characteristics**

    - Observations: 500
    - Covariate structure: Multivariate normal with pairwise correlation 0.5

    Raises
    ------
    FileNotFoundError
        If simulation data file does not exist.

    Examples
    --------
    >>> from cbps.datasets import load_npcbps_continuous_sim
    >>> df = load_npcbps_continuous_sim()
    >>> df.shape
    (500, 7)
    >>> list(df.columns)
    ['Y', 'T', 'X1', 'X2', 'X3', 'X4', 'X5']
    """
    # Get data directory (supports both installed and development modes)
    package_data_dir = Path(__file__).parent.parent / "data"
    if package_data_dir.exists():
        data_dir = package_data_dir
    else:
        # Fall back to project root data directory (development mode)
        data_dir = Path(__file__).parent.parent.parent.parent / "data"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Cannot find data directory. Tried:\n"
                f"  1. Package data: {package_data_dir}\n"
                f"  2. Project data: {data_dir}\n"
                f"Please ensure the package is properly installed with data files."
            )

    filepath = data_dir / "npcbps_continuous_sim.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Simulation data not found: {filepath}\n"
            "Please ensure the package is properly installed with data files."
        )
    
    df = pd.read_csv(filepath)
    
    # Validate data dimensions
    expected_shape = (500, 7)
    if df.shape != expected_shape:
        raise ValueError(
            f"Unexpected data shape: {df.shape}\n"
            f"Expected: {expected_shape}"
        )
    
    # Validate column names
    expected_columns = ['Y', 'T', 'X1', 'X2', 'X3', 'X4', 'X5']
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns: {list(df.columns)}\n"
            f"Expected: {expected_columns}"
        )
    
    return df

