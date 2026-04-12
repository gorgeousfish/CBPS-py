"""
CBPS Utility Functions

This module provides shared components for formula parsing, weight computation,
and data preprocessing used across the CBPS package.

Submodules
----------
formula
    Wilkinson-Rogers formula parsing using patsy with extensions for
    treatment models and dual formula specifications.

weights
    Inverse probability weight computation for ATE, ATT, and continuous
    treatment estimands with group-wise standardization.

helpers
    Data validation, missing value handling, and treatment encoding utilities.

numerics
    Numerical linear algebra utilities including pseudoinverse computation.

validation
    Centralized input validation with informative error messages.

Exported Functions
------------------
**Formula Parsing**:

- ``parse_formula`` - Parse treatment ~ covariates formulas
- ``parse_dual_formulas`` - Parse baseline and difference formulas
- ``parse_arrays`` - Construct design matrix from arrays

**Weight Computation**:

- ``compute_ate_weights`` - ATE inverse probability weights
- ``compute_att_weights`` - ATT inverse probability weights
- ``compute_continuous_weights`` - Continuous treatment weights
- ``standardize_weights`` - Group-normalized weights

**Data Utilities**:

- ``validate_arrays`` - Validate array dimensions and types
- ``handle_missing`` - Remove observations with missing values
- ``encode_treatment_factor`` - Convert categorical treatment to numeric
- ``normalize_sample_weights`` - Normalize weights to sum to n
"""

from cbps.utils.formula import parse_arrays, parse_dual_formulas, parse_formula
from cbps.utils.helpers import (
    encode_treatment_factor,
    handle_missing,
    normalize_sample_weights,
    validate_arrays,
)
from cbps.utils.weights import (
    compute_ate_weights,
    compute_att_weights,
    compute_continuous_weights,
    standardize_weights,
)

__all__ = [
    # Formula parsing functions
    "parse_formula",
    "parse_dual_formulas",
    "parse_arrays",
    # Weight computation functions
    "compute_ate_weights",
    "compute_att_weights",
    "compute_continuous_weights",
    "standardize_weights",
    # Utility functions
    "normalize_sample_weights",
    "validate_arrays",
    "handle_missing",
    "encode_treatment_factor",
]
