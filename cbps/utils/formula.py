"""
Formula Parsing Utilities

This module provides formula parsing functionality using patsy, with
extensions tailored for CBPS treatment models.

Supported formula interfaces:

- **Standard formulas**: Parse ``treatment ~ covariates`` specifications
  into treatment vectors and design matrices
- **Dual formulas**: Parse separate baseline and difference formulas
  for optimal CBPS estimation
- **Array interface**: Direct matrix input for programmatic use

The formula parser supports standard patsy syntax including:

- Categorical variables via ``C(variable)`` or ``factor(variable)``
- Interactions via ``:`` operator
- Polynomial terms via ``I()`` for as-is expressions
- Automatic intercept handling

Functions
---------
parse_formula
    Parse treatment ~ covariates formula to arrays.
parse_dual_formulas
    Parse baseline and difference formulas for optimal CBPS.
parse_arrays
    Construct design matrix from array inputs.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

from typing import Optional, Tuple, Union
import re

import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix


def _convert_r_formula_to_patsy(formula: str) -> str:
    """
    Convert alternative formula syntax to patsy-compatible format.
    
    Transforms ``factor(var)`` notation to patsy's ``C(var)`` notation
    for categorical variable specification.
    
    Parameters
    ----------
    formula : str
        Formula string potentially containing factor() syntax.
    
    Returns
    -------
    str
        Patsy-compatible formula string with C() notation.
    
    Examples
    --------
    >>> _convert_r_formula_to_patsy('treat ~ x1 + factor(country)')
    'treat ~ x1 + C(country)'
    >>> _convert_r_formula_to_patsy('treat ~ factor(year) + x1')
    'treat ~ C(year) + x1'
    """
    # Replace factor(var) with C(var)
    # Use regex to match factor(...) and replace with C(...)
    converted = re.sub(r'\bfactor\s*\(', 'C(', formula)
    return converted


def parse_formula(
    formula: str,
    data: pd.DataFrame,
    return_type: str = 'dataframe',
    preserve_categorical: bool = False
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Parse a Wilkinson-Rogers formula into treatment vector and design matrix.

    Parameters
    ----------
    formula : str
        Formula specification in the form ``"treatment ~ covariates"``.
        Supports patsy syntax including:
        
        - Main effects: ``age + educ``
        - Interactions: ``age:educ``
        - Categorical: ``C(region)`` or ``factor(region)``
        - As-is expressions: ``I(re75==0)``
        - Remove intercept: ``-1``
        
    data : pd.DataFrame
        DataFrame containing all variables referenced in the formula.
    return_type : str, default='dataframe'
        Currently unused; arrays are always returned as numpy arrays.
    preserve_categorical : bool, default=False
        If True and treatment is pd.Categorical, preserve the original
        Categorical dtype. Used internally for multi-level treatment models.

    Returns
    -------
    y : np.ndarray or pd.Series
        Treatment vector, shape (n,).
        Returns float64 array unless preserve_categorical=True and
        the original treatment is Categorical.
    X : np.ndarray
        Design matrix, shape (n, k), dtype=float64.
        Includes intercept column by default (first column).

    Notes
    -----
    **Design matrix structure**:
    
    - Patsy adds an intercept column by default (suppress with ``-1``)
    - Categorical variables are dummy-coded with K-1 columns
    - Column order follows patsy conventions: intercept first,
      then main effects, interactions, and I() terms
    
    **Post-processing**: The caller is typically responsible for
    zero-variance column filtering if needed::
    
        std = X.std(axis=0, ddof=1)
        X = X[:, std > 0]

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'treat': [0, 1, 0, 1],
    ...     'age': [25, 30, 35, 40],
    ...     'educ': [12, 16, 14, 18]
    ... })
    >>> y, X = parse_formula("treat ~ age + educ", df)
    >>> y.shape
    (4,)
    >>> X.shape  # (n, k) with intercept
    (4, 3)

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Step 0a: Convert factor() syntax to patsy C() notation
    formula = _convert_r_formula_to_patsy(formula)
    
    # Step 0b: Detect if treatment variable is Categorical
    # Extract treatment variable name from formula
    if '~' in formula:
        lhs = formula.split('~')[0].strip()
        treat_var_name = lhs

        # Check if variable in original data is Categorical
        is_categorical_treat = (
            preserve_categorical and
            treat_var_name in data.columns and
            isinstance(data[treat_var_name].dtype, pd.CategoricalDtype)
        )

        if is_categorical_treat:
            # Save original Categorical Series
            original_treat = data[treat_var_name].copy()
            # Temporarily convert to numeric to avoid patsy one-hot encoding
            data_temp = data.copy()
            data_temp[treat_var_name] = data[treat_var_name].cat.codes.astype(np.float64)
        else:
            data_temp = data
            original_treat = None
    else:
        data_temp = data
        original_treat = None
        is_categorical_treat = False

    # Step 1: Parse formula using patsy
    y, X_df = dmatrices(formula, data_temp, return_type='dataframe')
    
    # Step 2: Reorder columns to match formula order
    # Patsy orders alphabetically, we want formula order
    # Strategy: Intercept first, then non-I() terms, then I() terms
    col_names = X_df.columns.tolist()
    
    # Extract variable order from formula
    if '~' in formula:
        rhs = formula.split('~')[1].strip()
        formula_terms = [t.strip() for t in rhs.split('+')]
        
        # Separate simple variables and I() variables
        simple_vars = []
        i_vars = []
        for term in formula_terms:
            if term.startswith('I('):
                i_vars.append(term)
            else:
                simple_vars.append(term)
        
        # Build new order
        new_order = []
        
        # 1. Intercept first
        for i, col in enumerate(col_names):
            if col == 'Intercept':
                new_order.append(i)
                break
        
        # 2. Add simple variables in formula order (exact match)
        for var in simple_vars:
            for i, col in enumerate(col_names):
                # Exact match: column name equals variable name
                if i not in new_order and col == var and '[T.' not in col:
                    new_order.append(i)
                    break
        
        # 3. Add I() variables in formula order
        import re
        for i_var in i_vars:
            # I(re75==0) needs to match "I(re75 == 0)[T.True]"
            for i, col in enumerate(col_names):
                if i not in new_order and col.startswith('I('):
                    # Check if variable in I() matches
                    formula_match = re.search(r'I\((\w+)', i_var)
                    col_match = re.search(r'I\((\w+)', col)
                    if formula_match and col_match:
                        if formula_match.group(1) == col_match.group(1):
                            new_order.append(i)
                            break
        
        # 4. Add any remaining columns
        for i in range(len(col_names)):
            if i not in new_order:
                new_order.append(i)
        
        # Reorder columns
        X_reordered = X_df.iloc[:, new_order]
    else:
        X_reordered = X_df
    
    # Step 3: Convert to numpy array
    X = X_reordered.values

    # Step 4: Process treatment variable
    if is_categorical_treat:
        # Return original Categorical Series (preserve factor semantics)
        y = original_treat
    else:
        # Convert to float64 array
        y = y.values.ravel()
        y = y.astype(np.float64)

    # Step 5: Convert X to float64 (enforce double precision)
    X = X.astype(np.float64)

    # Step 6: Return reordered design matrix
    return y, X


def parse_dual_formulas(
    baseline_formula: Optional[str],
    diff_formula: Optional[str],
    data: pd.DataFrame
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse baseline and difference formulas for optimal CBPS.
    
    Optimal CBPS uses separate covariate specifications for the baseline
    outcome model E[Y(0)|X] and the treatment effect heterogeneity
    E[Y(1)-Y(0)|X]. This function parses both formulas and returns
    their design matrices.
    
    Parameters
    ----------
    baseline_formula : str or None
        Right-hand-side formula for baseline covariates, e.g., ``"~ age + educ"``.
    diff_formula : str or None
        Right-hand-side formula for treatment effect covariates, e.g., ``"~ age"``.
    data : pd.DataFrame
        DataFrame containing all referenced variables.
    
    Returns
    -------
    baselineX : np.ndarray or None
        Design matrix for baseline formula, shape (n, k1).
        Zero-variance columns are automatically removed.
    diffX : np.ndarray or None
        Design matrix for difference formula, shape (n, k2).
        Zero-variance columns are automatically removed.
    
    Raises
    ------
    ValueError
        If exactly one formula is None (both must be specified together
        or both must be None).
    
    Notes
    -----
    Unlike ``parse_formula()``, this function:
    
    - Takes right-hand-side only formulas (no treatment variable)
    - Automatically filters zero-variance columns
    - Returns None for both outputs if both inputs are None
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'educ': [12, 16, 14, 18]
    ... })
    >>> baselineX, diffX = parse_dual_formulas("~ age + educ", "~ age", df)
    >>> baselineX.shape[1] >= diffX.shape[1]
    True

    References
    ----------
    Fan, J., Imai, K., Liu, H., Ning, Y., and Yang, X. (2021). Improving
    covariate balancing propensity score: A doubly robust and efficient
    approach. Working paper.
    """
    # Step 1: XOR check - both must be specified or both None
    if (baseline_formula is None) != (diff_formula is None):
        raise ValueError(
            "Either baseline_formula or diff_formula not specified. "
            "Both must be specified to use CBPSOptimal. Otherwise, leave both None."
        )
    
    # Step 2: Return None if both are None
    if baseline_formula is None and diff_formula is None:
        return None, None
    
    # Step 3: Parse baseline formula
    baselineX = dmatrix(baseline_formula, data, return_type='matrix')
    baselineX = np.asarray(baselineX, dtype=np.float64)
    
    # Filter zero-variance columns (threshold > 0)
    std_baseline = baselineX.std(axis=0, ddof=1)
    baselineX = baselineX[:, std_baseline > 0]
    
    # Step 4: Parse diff formula
    diffX = dmatrix(diff_formula, data, return_type='matrix')
    diffX = np.asarray(diffX, dtype=np.float64)
    
    # Filter zero-variance columns
    std_diff = diffX.std(axis=0, ddof=1)
    diffX = diffX[:, std_diff > 0]
    
    return baselineX, diffX


def parse_arrays(
    treatment: Union[np.ndarray, pd.Series],
    covariates: Union[np.ndarray, pd.DataFrame],
    add_intercept: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct treatment vector and design matrix from array inputs.
    
    Provides a programmatic interface for CBPS when data is already
    available as arrays rather than in a DataFrame with formula specification.
    
    Parameters
    ----------
    treatment : array-like
        Treatment variable, shape (n,) or (n, 1).
    covariates : array-like
        Covariate matrix, shape (n, k) or (n,) for single covariate.
    add_intercept : bool, default=True
        If True, prepend a column of ones to the covariate matrix.
    
    Returns
    -------
    y : np.ndarray
        Treatment vector, shape (n,), dtype=float64.
    X : np.ndarray
        Design matrix, shape (n, k) or (n, k+1) with intercept.
        dtype=float64.
    
    Notes
    -----
    This function produces output compatible with ``parse_formula()``,
    enabling consistent downstream processing regardless of input method.
    
    Examples
    --------
    >>> import numpy as np
    >>> treatment = np.array([0, 1, 0, 1])
    >>> covariates = np.array([[25, 12], [30, 16], [35, 14], [40, 18]])
    >>> y, X = parse_arrays(treatment, covariates, add_intercept=True)
    >>> X.shape
    (4, 3)
    >>> np.allclose(X[:, 0], 1.0)  # First column is intercept
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Convert to numpy arrays
    y = np.asarray(treatment, dtype=np.float64).ravel()
    X = np.asarray(covariates, dtype=np.float64)
    
    # Ensure X is 2-dimensional
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Add intercept column if requested
    if add_intercept:
        intercept = np.ones((len(y), 1), dtype=np.float64)
        X = np.column_stack([intercept, X])
    
    return y, X

