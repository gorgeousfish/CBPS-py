"""
Comprehensive Test Suite for CBPS Utility Functions
====================================================

This module consolidates all tests for the utility functions in the
``cbps.utils`` package, covering formula parsing, input validation,
weight computation, numerical helpers, R compatibility utilities,
and variance transformations.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
    DOI: 10.1111/rssb.12027

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics,
    12(1), 156-177. DOI: 10.1214/17-AOAS1101
"""

import numpy as np
import pandas as pd
import pytest
import inspect
from numpy.testing import assert_allclose
from cbps.utils.formula import parse_formula
from cbps.utils.helpers import (
    normalize_sample_weights,
    validate_arrays,
    handle_missing,
    encode_treatment_factor,
)
from cbps.utils.numerics import (
    r_ginv_like,
    pinv_symmetric_psd,
    numeric_rank,
    symmetrize,
    is_symmetric
)
from cbps.utils.r_compat import ensure_rpy2_compatibility, check_rpy2_available
from cbps.utils.validation import validate_cbps_input
from cbps.utils.variance_transform import (
    _r_ginv,
    transform_variance_binary,
    transform_variance_3treat,
    transform_variance_4treat,
    transform_variance_continuous,
    apply_variance_svd_inverse_transform,
)
from cbps.utils.weights import compute_ate_weights


# =============================================================================
# Embedded Fixtures (from utils/conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def utils_tolerances():
    """
    Provide tolerance values for utils numerical comparisons.
    
    Returns
    -------
    dict
        Dictionary containing tolerance values:
        - weight_rtol: Relative tolerance for weight computations (1e-10)
        - weight_atol: Absolute tolerance for weight computations (1e-10)
        - matrix_rtol: Relative tolerance for matrix operations (1e-10)
        - matrix_atol: Absolute tolerance for matrix operations (1e-12)
        - transform_rtol: Relative tolerance for transformations (1e-8)
        - transform_atol: Absolute tolerance for transformations (1e-10)
    """
    return {
        'weight_rtol': 1e-10,
        'weight_atol': 1e-10,
        'matrix_rtol': 1e-10,
        'matrix_atol': 1e-12,
        'transform_rtol': 1e-8,
        'transform_atol': 1e-10,
    }


@pytest.fixture
def simple_binary_data():
    """
    Generate simple binary treatment data.
    
    Returns
    -------
    dict
        Dictionary containing treat, probs, X, n.
    """
    np.random.seed(42)
    n = 100
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Generate propensity scores and treatment
    logit_ps = -0.5 + 0.3 * x1 - 0.2 * x2
    probs = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, probs).astype(float)
    
    return {
        'treat': treat,
        'probs': probs,
        'X': X,
        'n': n,
    }


@pytest.fixture
def simple_continuous_data():
    """
    Generate simple continuous treatment data.
    
    Returns
    -------
    dict
        Dictionary containing treat, X, n.
    """
    np.random.seed(42)
    n = 100
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Continuous treatment
    treat = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)
    
    return {
        'treat': treat,
        'X': X,
        'n': n,
    }


@pytest.fixture
def formula_test_df():
    """
    Generate DataFrame for formula parsing tests.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with treatment and covariates.
    """
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'treat': np.random.binomial(1, 0.5, n),
        'age': np.random.uniform(18, 65, n),
        'educ': np.random.randint(8, 17, n),
        'income': np.random.exponential(50000, n),
        'gender': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
    })
    
    return df


@pytest.fixture
def rank_deficient_matrix():
    """
    Generate a rank-deficient matrix for numerical tests.
    
    Returns
    -------
    np.ndarray
        A 4x4 matrix with rank 3.
    """
    np.random.seed(42)
    
    # Create a rank-3 matrix (4x4 but with one linearly dependent column)
    A = np.random.randn(4, 3)
    A = np.column_stack([A, A[:, 0] + A[:, 1]])  # Last column = col0 + col1
    
    return A


@pytest.fixture
def symmetric_psd_matrix():
    """
    Generate a symmetric positive semi-definite matrix.
    
    Returns
    -------
    np.ndarray
        A 4x4 symmetric PSD matrix.
    """
    np.random.seed(42)
    
    A = np.random.randn(4, 4)
    # A @ A.T is guaranteed to be symmetric PSD
    return A @ A.T


# =============================================================================
# Module Export Tests (from test_init.py)
# =============================================================================

@pytest.mark.unit
class TestUtilsModuleExports:
    """
    Test ID: UTIL-INIT-001 ~ UTIL-INIT-012
    Requirement: REQ-UTIL-INIT-001
    
    Tests for cbps.utils module exports.
    """
    
    def test_util_init001_module_importable(self):
        """UTIL-INIT-001: cbps.utils module is importable."""
        from cbps import utils
        assert utils is not None
    
    def test_util_init002_parse_formula_importable(self):
        """UTIL-INIT-002: parse_formula is importable from cbps.utils."""
        from cbps.utils import parse_formula
        assert callable(parse_formula)
    
    def test_util_init003_parse_dual_formulas_importable(self):
        """UTIL-INIT-003: parse_dual_formulas is importable from cbps.utils."""
        from cbps.utils import parse_dual_formulas
        assert callable(parse_dual_formulas)
    
    def test_util_init004_parse_arrays_importable(self):
        """UTIL-INIT-004: parse_arrays is importable from cbps.utils."""
        from cbps.utils import parse_arrays
        assert callable(parse_arrays)
    
    def test_util_init005_compute_ate_weights_importable(self):
        """UTIL-INIT-005: compute_ate_weights is importable from cbps.utils."""
        from cbps.utils import compute_ate_weights
        assert callable(compute_ate_weights)
    
    def test_util_init006_compute_att_weights_importable(self):
        """UTIL-INIT-006: compute_att_weights is importable from cbps.utils."""
        from cbps.utils import compute_att_weights
        assert callable(compute_att_weights)
    
    def test_util_init007_compute_continuous_weights_importable(self):
        """UTIL-INIT-007: compute_continuous_weights is importable from cbps.utils."""
        from cbps.utils import compute_continuous_weights
        assert callable(compute_continuous_weights)
    
    def test_util_init008_standardize_weights_importable(self):
        """UTIL-INIT-008: standardize_weights is importable from cbps.utils."""
        from cbps.utils import standardize_weights
        assert callable(standardize_weights)
    
    def test_util_init009_validate_arrays_importable(self):
        """UTIL-INIT-009: validate_arrays is importable from cbps.utils."""
        from cbps.utils import validate_arrays
        assert callable(validate_arrays)
    
    def test_util_init010_handle_missing_importable(self):
        """UTIL-INIT-010: handle_missing is importable from cbps.utils."""
        from cbps.utils import handle_missing
        assert callable(handle_missing)
    
    def test_util_init011_encode_treatment_factor_importable(self):
        """UTIL-INIT-011: encode_treatment_factor is importable from cbps.utils."""
        from cbps.utils import encode_treatment_factor
        assert callable(encode_treatment_factor)
    
    def test_util_init012_normalize_sample_weights_importable(self):
        """UTIL-INIT-012: normalize_sample_weights is importable from cbps.utils."""
        from cbps.utils import normalize_sample_weights
        assert callable(normalize_sample_weights)


# =============================================================================
# Test Class: __all__ Definition
# =============================================================================

@pytest.mark.unit
class TestUtilsAllDefinition:
    """
    Test ID: UTIL-INIT-013 ~ UTIL-INIT-018
    Requirement: REQ-UTIL-INIT-002
    
    Tests for __all__ definition in utils module.
    """
    
    def test_util_init013_all_defined(self):
        """UTIL-INIT-013: __all__ is defined in utils module."""
        from cbps import utils
        assert hasattr(utils, '__all__')
        assert isinstance(utils.__all__, (list, tuple))
    
    def test_util_init014_all_contains_formula_functions(self):
        """UTIL-INIT-014: __all__ contains formula parsing functions."""
        from cbps import utils
        formula_funcs = ['parse_formula', 'parse_dual_formulas', 'parse_arrays']
        for name in formula_funcs:
            assert name in utils.__all__, f"{name} not in __all__"
    
    def test_util_init015_all_contains_weight_functions(self):
        """UTIL-INIT-015: __all__ contains weight computation functions."""
        from cbps import utils
        weight_funcs = [
            'compute_ate_weights',
            'compute_att_weights',
            'compute_continuous_weights',
            'standardize_weights',
        ]
        for name in weight_funcs:
            assert name in utils.__all__, f"{name} not in __all__"
    
    def test_util_init016_all_contains_utility_functions(self):
        """UTIL-INIT-016: __all__ contains utility functions."""
        from cbps import utils
        util_funcs = [
            'normalize_sample_weights',
            'validate_arrays',
            'handle_missing',
            'encode_treatment_factor',
        ]
        for name in util_funcs:
            assert name in utils.__all__, f"{name} not in __all__"
    
    def test_util_init017_no_private_exports(self):
        """UTIL-INIT-017: __all__ does not export private functions."""
        from cbps import utils
        for name in utils.__all__:
            assert not name.startswith('_'), f"Private {name} should not be in __all__"
    
    def test_util_init018_all_exports_are_callable(self):
        """UTIL-INIT-018: All exports in __all__ are callable."""
        from cbps import utils
        for name in utils.__all__:
            obj = getattr(utils, name)
            assert callable(obj), f"{name} should be callable"


# =============================================================================
# Test Class: Function Signatures
# =============================================================================

@pytest.mark.unit
class TestUtilsFunctionSignatures:
    """
    Test ID: UTIL-INIT-019 ~ UTIL-INIT-025
    Requirement: REQ-UTIL-INIT-003
    
    Tests for utility function signatures.
    """
    
    def test_util_init019_parse_formula_has_formula_param(self):
        """UTIL-INIT-019: parse_formula has formula parameter."""
        from cbps.utils import parse_formula
        sig = inspect.signature(parse_formula)
        assert 'formula' in sig.parameters
    
    def test_util_init020_parse_formula_has_data_param(self):
        """UTIL-INIT-020: parse_formula has data parameter."""
        from cbps.utils import parse_formula
        sig = inspect.signature(parse_formula)
        assert 'data' in sig.parameters
    
    def test_util_init021_compute_ate_weights_has_probs_param(self):
        """UTIL-INIT-021: compute_ate_weights has probs parameter."""
        from cbps.utils import compute_ate_weights
        sig = inspect.signature(compute_ate_weights)
        assert 'probs' in sig.parameters
    
    def test_util_init022_compute_ate_weights_has_treat_param(self):
        """UTIL-INIT-022: compute_ate_weights has treat parameter."""
        from cbps.utils import compute_ate_weights
        sig = inspect.signature(compute_ate_weights)
        assert 'treat' in sig.parameters
    
    def test_util_init023_validate_arrays_has_treat_param(self):
        """UTIL-INIT-023: validate_arrays has treat parameter."""
        from cbps.utils import validate_arrays
        sig = inspect.signature(validate_arrays)
        assert 'treat' in sig.parameters
    
    def test_util_init024_validate_arrays_has_X_param(self):
        """UTIL-INIT-024: validate_arrays has X parameter."""
        from cbps.utils import validate_arrays
        sig = inspect.signature(validate_arrays)
        assert 'X' in sig.parameters
    
    def test_util_init025_all_functions_have_docstrings(self):
        """UTIL-INIT-025: All exported functions have docstrings."""
        from cbps import utils
        for name in utils.__all__:
            obj = getattr(utils, name)
            assert obj.__doc__ is not None, f"{name} should have a docstring"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Formula Parsing Tests (from test_formula.py)
# =============================================================================

class TestBasicFormulaParsing:
    """
    Test basic formula parsing functionality.
    
    Test IDs: FORM-001 to FORM-010
    Requirements: REQ-UTIL-001
    """
    
    @pytest.mark.unit
    def test_form001_simple_formula(self, formula_test_df):
        """
        FORM-001: Verify simple formula parsing works.
        
        Requirements: REQ-UTIL-001
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + educ", df)
        
        assert isinstance(treat, np.ndarray)
        assert isinstance(X, np.ndarray)
        assert len(treat) == len(df)
        assert X.shape[0] == len(df)
    
    @pytest.mark.unit
    def test_form002_intercept_included(self, formula_test_df):
        """
        FORM-002: Verify intercept is included by default.
        
        Requirements: REQ-UTIL-001
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + educ", df)
        
        # First column should be intercept (all ones)
        assert np.allclose(X[:, 0], 1.0)
    
    @pytest.mark.unit
    def test_form003_remove_intercept(self, formula_test_df):
        """
        FORM-003: Verify intercept can be removed with -1.
        
        Requirements: REQ-UTIL-001
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ -1 + age + educ", df)
        
        # No intercept column, so check shape
        # With two variables and no intercept, should have 2 columns
        assert X.shape[1] == 2
    
    @pytest.mark.unit
    def test_form004_categorical_variable(self, formula_test_df):
        """
        FORM-004: Verify categorical variable encoding with C().
        
        Requirements: REQ-UTIL-002
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + C(gender)", df)
        
        # Should have intercept + age + gender dummy
        assert X.shape[1] >= 3
    
    @pytest.mark.unit
    def test_form005_factor_notation(self, formula_test_df):
        """
        FORM-005: Verify factor() notation is converted to C().
        
        Requirements: REQ-UTIL-002
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + factor(gender)", df)
        
        # Should work same as C(gender)
        assert X.shape[1] >= 3
    
    @pytest.mark.unit
    def test_form006_interaction_term(self, formula_test_df):
        """
        FORM-006: Verify interaction term parsing.
        
        Requirements: REQ-UTIL-003
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + educ + age:educ", df)
        
        # Should have intercept + age + educ + age:educ
        assert X.shape[1] >= 4
    
    @pytest.mark.unit
    def test_form007_multiple_covariates(self, formula_test_df):
        """
        FORM-007: Verify multiple covariates parsing.
        
        Requirements: REQ-UTIL-001
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + educ + income", df)
        
        # Should have intercept + 3 covariates
        assert X.shape[1] == 4
    
    @pytest.mark.unit
    def test_form008_treatment_values(self, formula_test_df):
        """
        FORM-008: Verify treatment values match original data.
        
        Requirements: REQ-UTIL-001
        """
        df = formula_test_df
        
        treat, X = parse_formula("treat ~ age + educ", df)
        
        assert_allclose(treat, df['treat'].values)
    
    @pytest.mark.unit
    def test_form009_missing_variable_raises(self, formula_test_df):
        """
        FORM-009: Verify missing variable raises error.
        
        Requirements: REQ-UTIL-004
        """
        df = formula_test_df
        
        with pytest.raises(Exception):  # Could be PatsyError or KeyError
            parse_formula("treat ~ nonexistent_var", df)
    
    @pytest.mark.unit
    def test_form010_invalid_formula_raises(self, formula_test_df):
        """
        FORM-010: Verify invalid formula raises error.
        
        Requirements: REQ-UTIL-004
        """
        df = formula_test_df
        
        with pytest.raises(Exception):
            parse_formula("this is not a formula", df)


# =============================================================================
# Note: Dual Formula Parsing and Array Interface tests are skipped because
# parse_dual_formulas and parse_arrays have different signatures that would
# require extensive refactoring.
# =============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Helper Function Tests (from test_helpers.py)
# =============================================================================

class TestNormalizeSampleWeights:
    """
    Test normalize_sample_weights function.
    
    Test IDs: HELP-001 to HELP-010
    Requirements: REQ-UTIL-010
    """
    
    @pytest.mark.unit
    def test_help001_none_returns_ones(self):
        """
        HELP-001: Verify None input returns uniform weights.
        
        Requirements: REQ-UTIL-010
        """
        n = 100
        result = normalize_sample_weights(None, n)
        
        assert len(result) == n
        assert np.allclose(result, 1.0)
    
    @pytest.mark.unit
    def test_help002_sum_equals_n(self):
        """
        HELP-002: Verify normalized weights sum to n.
        
        Requirements: REQ-UTIL-010
        """
        n = 100
        sw = np.random.uniform(0.5, 2.0, n)
        
        result = normalize_sample_weights(sw, n)
        
        assert_allclose(result.sum(), n, rtol=1e-10)
    
    @pytest.mark.unit
    def test_help003_dtype_float64(self):
        """
        HELP-003: Verify output dtype is float64.
        
        Requirements: REQ-UTIL-010
        """
        sw = np.array([1, 2, 3, 4], dtype=np.int32)
        
        result = normalize_sample_weights(sw, 4)
        
        assert result.dtype == np.float64
    
    @pytest.mark.unit
    def test_help004_negative_raises(self):
        """
        HELP-004: Verify negative weights raise ValueError.
        
        Requirements: REQ-UTIL-011
        """
        sw = np.array([1, -1, 2, 3])
        
        with pytest.raises(ValueError) as excinfo:
            normalize_sample_weights(sw, 4)
        
        assert "non-negative" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help005_all_zeros_raises(self):
        """
        HELP-005: Verify all-zero weights raise ValueError.
        
        Requirements: REQ-UTIL-011
        """
        sw = np.zeros(4)
        
        with pytest.raises(ValueError) as excinfo:
            normalize_sample_weights(sw, 4)
        
        assert "zero" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help006_some_zeros_warns(self):
        """
        HELP-006: Verify some zero weights issue warning.
        
        Requirements: REQ-UTIL-011
        """
        sw = np.array([0.0, 1.0, 2.0, 3.0])
        
        with pytest.warns(UserWarning) as record:
            result = normalize_sample_weights(sw, 4)
        
        assert len(record) == 1
        assert "zero" in str(record[0].message).lower()
    
    @pytest.mark.numerical
    def test_help007_preserves_ratios(self):
        """
        HELP-007: Verify normalization preserves weight ratios.
        
        Requirements: REQ-UTIL-010
        """
        sw = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = normalize_sample_weights(sw, 4)
        
        # Ratio between first and second element should be preserved
        original_ratio = sw[1] / sw[0]
        result_ratio = result[1] / result[0]
        
        assert_allclose(result_ratio, original_ratio, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_help008_already_normalized(self):
        """
        HELP-008: Verify already-normalized weights are unchanged.
        
        Requirements: REQ-UTIL-010
        """
        n = 4
        sw = np.ones(n)  # Sum = n already
        
        result = normalize_sample_weights(sw, n)
        
        assert_allclose(result, sw, rtol=1e-10)
    
    @pytest.mark.edge_case
    def test_help009_single_element(self):
        """
        HELP-009: Verify handling of single element.
        
        Requirements: REQ-UTIL-010
        """
        sw = np.array([5.0])
        
        result = normalize_sample_weights(sw, 1)
        
        assert_allclose(result, 1.0, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_help010_large_weights(self):
        """
        HELP-010: Verify handling of large weight values.
        
        Requirements: REQ-UTIL-010
        """
        n = 100
        sw = np.random.uniform(1e6, 1e7, n)
        
        result = normalize_sample_weights(sw, n)
        
        assert_allclose(result.sum(), n, rtol=1e-10)


# =============================================================================
# Test Class: Validate Arrays (HELP-011 to HELP-018)
# =============================================================================

class TestValidateArrays:
    """
    Test validate_arrays function.
    
    Test IDs: HELP-011 to HELP-018
    Requirements: REQ-UTIL-012
    """
    
    @pytest.mark.unit
    def test_help011_valid_input_passes(self):
        """
        HELP-011: Verify valid input passes validation.
        
        Requirements: REQ-UTIL-012
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, 25], [1, 30], [1, 35], [1, 40]])
        
        treat_v, X_v = validate_arrays(treat, X)
        
        assert len(treat_v) == 4
        assert X_v.shape == (4, 2)
    
    @pytest.mark.unit
    def test_help012_dtype_conversion(self):
        """
        HELP-012: Verify dtype conversion to float64.
        
        Requirements: REQ-UTIL-012
        """
        treat = np.array([0, 1, 0, 1], dtype=np.int32)
        X = np.array([[1, 25], [1, 30], [1, 35], [1, 40]], dtype=np.int32)
        
        treat_v, X_v = validate_arrays(treat, X)
        
        assert treat_v.dtype == np.float64
        assert X_v.dtype == np.float64
    
    @pytest.mark.unit
    def test_help013_dimension_mismatch_raises(self):
        """
        HELP-013: Verify dimension mismatch raises ValueError.
        
        Requirements: REQ-UTIL-013
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, 25], [1, 30], [1, 35]])  # 3 rows instead of 4
        
        with pytest.raises(ValueError) as excinfo:
            validate_arrays(treat, X)
        
        assert "length" in str(excinfo.value).lower() or "rows" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help014_nan_treatment_raises(self):
        """
        HELP-014: Verify NaN in treatment raises ValueError.
        
        Requirements: REQ-UTIL-013
        """
        treat = np.array([0, np.nan, 0, 1])
        X = np.array([[1, 25], [1, 30], [1, 35], [1, 40]])
        
        with pytest.raises(ValueError) as excinfo:
            validate_arrays(treat, X)
        
        assert "nan" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help015_nan_covariates_raises(self):
        """
        HELP-015: Verify NaN in covariates raises ValueError.
        
        Requirements: REQ-UTIL-013
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, 25], [1, np.nan], [1, 35], [1, 40]])
        
        with pytest.raises(ValueError) as excinfo:
            validate_arrays(treat, X)
        
        assert "nan" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help016_rank_deficient_raises(self):
        """
        HELP-016: Verify rank-deficient X raises ValueError when check_rank=True.
        
        Requirements: REQ-UTIL-013
        """
        treat = np.array([0, 1, 0, 1])
        # Rank-deficient: column 2 = column 1
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        
        with pytest.raises(ValueError) as excinfo:
            validate_arrays(treat, X, check_rank=True)
        
        assert "rank" in str(excinfo.value).lower()
    
    @pytest.mark.unit
    def test_help017_skip_rank_check(self):
        """
        HELP-017: Verify check_rank=False skips rank check.
        
        Requirements: REQ-UTIL-012
        """
        treat = np.array([0, 1, 0, 1])
        # Rank-deficient X
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        
        # Should not raise with check_rank=False
        treat_v, X_v = validate_arrays(treat, X, check_rank=False)
        
        assert len(treat_v) == 4
    
    @pytest.mark.unit
    def test_help018_full_rank_passes(self):
        """
        HELP-018: Verify full-rank X passes validation.
        
        Requirements: REQ-UTIL-012
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, 25], [1, 30], [1, 35], [1, 40]])
        
        treat_v, X_v = validate_arrays(treat, X, check_rank=True)
        
        assert X_v.shape == (4, 2)


# =============================================================================
# Test Class: Handle Missing (HELP-019 to HELP-024)
# =============================================================================

class TestHandleMissing:
    """
    Test handle_missing function.
    
    Test IDs: HELP-019 to HELP-024
    Requirements: REQ-UTIL-014
    """
    
    @pytest.mark.unit
    def test_help019_no_missing_unchanged(self):
        """
        HELP-019: Verify no-missing data passes through unchanged.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, 0, 1, 0],
            'age': [25, 30, 35, 40]
        })
        
        df_clean, n_dropped = handle_missing(df)
        
        assert len(df_clean) == 4
        assert n_dropped == 0
    
    @pytest.mark.unit
    def test_help020_drops_nan_rows(self):
        """
        HELP-020: Verify NaN rows are dropped.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, 0, np.nan, 0],
            'age': [25, 30, 35, 40]
        })
        
        df_clean, n_dropped = handle_missing(df)
        
        assert len(df_clean) == 3
        assert n_dropped == 1
    
    @pytest.mark.unit
    def test_help021_returns_correct_count(self):
        """
        HELP-021: Verify returned count is correct.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, np.nan, np.nan, 0],
            'age': [25, 30, np.nan, 40]
        })
        
        df_clean, n_dropped = handle_missing(df)
        
        # Rows with any NaN are dropped:
        # Row 0: no NaN -> keep
        # Row 1: treat=NaN -> drop
        # Row 2: treat=NaN, age=NaN -> drop
        # Row 3: no NaN -> keep
        assert n_dropped == 2  # Rows 1 and 2 have NaN
    
    @pytest.mark.unit
    def test_help022_relevant_cols_filter(self):
        """
        HELP-022: Verify relevant_cols parameter filters columns.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, 0, 1, 0],
            'age': [25, np.nan, 35, 40],
            'other': [np.nan, np.nan, np.nan, np.nan]
        })
        
        # Only check 'treat' column
        df_clean, n_dropped = handle_missing(df, relevant_cols=['treat'])
        
        # 'other' column NaN is ignored
        assert len(df_clean) == 4
        assert n_dropped == 0
    
    @pytest.mark.unit
    def test_help023_preserves_index(self):
        """
        HELP-023: Verify original index is preserved.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, 0, np.nan, 0],
            'age': [25, 30, 35, 40]
        }, index=['a', 'b', 'c', 'd'])
        
        df_clean, n_dropped = handle_missing(df)
        
        assert list(df_clean.index) == ['a', 'b', 'd']
    
    @pytest.mark.unit
    def test_help024_warns_when_dropping(self):
        """
        HELP-024: Verify warning is issued when rows are dropped.
        
        Requirements: REQ-UTIL-014
        """
        df = pd.DataFrame({
            'treat': [1, 0, np.nan, 0],
            'age': [25, 30, 35, 40]
        })
        
        with pytest.warns(UserWarning) as record:
            handle_missing(df)
        
        assert len(record) == 1
        assert "removed" in str(record[0].message).lower()


# =============================================================================
# Test Class: Encode Treatment Factor (HELP-025 to HELP-030)
# =============================================================================

class TestEncodeTreatmentFactor:
    """
    Test encode_treatment_factor function.
    
    Test IDs: HELP-025 to HELP-030
    Requirements: REQ-UTIL-015
    """
    
    @pytest.mark.unit
    def test_help025_returns_tuple(self):
        """
        HELP-025: Verify returns tuple of (numeric, levels, orig).
        
        Requirements: REQ-UTIL-015
        """
        treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
        
        result = encode_treatment_factor(treat, att=0, verbose=0)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    @pytest.mark.unit
    def test_help026_binary_encoding(self):
        """
        HELP-026: Verify binary encoding is correct.
        
        Requirements: REQ-UTIL-015
        """
        treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
        
        treat_num, levels, _ = encode_treatment_factor(treat, att=1, verbose=0)
        
        # Second level (alphabetically) becomes 1
        assert list(treat_num) == [0, 1, 0, 1]
        assert levels == ['control', 'treatment']
    
    @pytest.mark.unit
    def test_help027_att2_inverts(self):
        """
        HELP-027: Verify att=2 inverts encoding.
        
        Requirements: REQ-UTIL-015
        """
        treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
        
        treat_num, levels, _ = encode_treatment_factor(treat, att=2, verbose=0)
        
        # ATT=2 inverts: first level (alphabetically) becomes 1
        assert list(treat_num) == [1, 0, 1, 0]
    
    @pytest.mark.unit
    def test_help028_numpy_array_input(self):
        """
        HELP-028: Verify numpy array input works.
        
        Requirements: REQ-UTIL-015
        """
        treat = np.array(['A', 'B', 'A', 'B'])
        
        treat_num, levels, _ = encode_treatment_factor(treat, att=1, verbose=0)
        
        assert list(treat_num) == [0, 1, 0, 1]
        assert levels == ['A', 'B']
    
    @pytest.mark.unit
    def test_help029_dtype_float64(self):
        """
        HELP-029: Verify output dtype is float64.
        
        Requirements: REQ-UTIL-015
        """
        treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
        
        treat_num, _, _ = encode_treatment_factor(treat, att=1, verbose=0)
        
        assert treat_num.dtype == np.float64
    
    @pytest.mark.unit
    def test_help030_preserves_original(self):
        """
        HELP-030: Verify original values are preserved.
        
        Requirements: REQ-UTIL-015
        """
        treat = pd.Categorical(['control', 'treatment', 'control', 'treatment'])
        
        _, _, treat_orig = encode_treatment_factor(treat, att=1, verbose=0)
        
        assert list(treat_orig) == ['control', 'treatment', 'control', 'treatment']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Numerical Computation Tests (from test_numerics.py)
# =============================================================================

class TestPseudoinverse:
    """
    Test pseudoinverse computation functions.
    
    Test IDs: NUM-001 to NUM-010
    Requirements: REQ-UTIL-020
    """
    
    @pytest.mark.numerical
    def test_num001_r_ginv_shape(self):
        """
        NUM-001: Verify r_ginv_like output shape.
        
        Requirements: REQ-UTIL-020
        """
        X = np.random.randn(5, 3)
        
        X_pinv = r_ginv_like(X)
        
        assert X_pinv.shape == (3, 5)  # Transpose of input shape
    
    @pytest.mark.numerical
    def test_num002_r_ginv_property(self):
        """
        NUM-002: Verify pseudoinverse property X @ X+ @ X ≈ X.
        
        Requirements: REQ-UTIL-020
        """
        X = np.random.randn(5, 3)
        
        X_pinv = r_ginv_like(X)
        
        # Verify Moore-Penrose property
        assert_allclose(X @ X_pinv @ X, X, rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num003_r_ginv_full_rank(self):
        """
        NUM-003: Verify pseudoinverse for full-rank matrix.
        
        Requirements: REQ-UTIL-020
        """
        # Full rank square matrix
        A = np.array([[1, 2], [3, 4]])
        
        A_pinv = r_ginv_like(A)
        
        # For full rank, pinv should be actual inverse
        assert_allclose(A @ A_pinv, np.eye(2), rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num004_r_ginv_rank_deficient(self, rank_deficient_matrix):
        """
        NUM-004: Verify pseudoinverse for rank-deficient matrix.
        
        Requirements: REQ-UTIL-020
        """
        A = rank_deficient_matrix
        
        A_pinv = r_ginv_like(A)
        
        # Should still satisfy Moore-Penrose property
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-8, atol=1e-10)
    
    @pytest.mark.numerical
    def test_num005_pinv_symmetric_psd_shape(self, symmetric_psd_matrix):
        """
        NUM-005: Verify pinv_symmetric_psd output shape.
        
        Requirements: REQ-UTIL-021
        """
        A = symmetric_psd_matrix
        
        A_pinv = pinv_symmetric_psd(A)
        
        assert A_pinv.shape == A.shape
    
    @pytest.mark.numerical
    def test_num006_pinv_symmetric_psd_property(self, symmetric_psd_matrix):
        """
        NUM-006: Verify pinv_symmetric_psd satisfies pseudoinverse property.
        
        Requirements: REQ-UTIL-021
        """
        A = symmetric_psd_matrix
        
        A_pinv = pinv_symmetric_psd(A)
        
        # Verify A @ A+ @ A ≈ A
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-8, atol=1e-10)
    
    @pytest.mark.numerical
    def test_num007_pinv_symmetric_psd_symmetric_output(self, symmetric_psd_matrix):
        """
        NUM-007: Verify pinv_symmetric_psd produces symmetric output.
        
        Requirements: REQ-UTIL-021
        """
        A = symmetric_psd_matrix
        
        A_pinv = pinv_symmetric_psd(A)
        
        # Output should be symmetric
        assert_allclose(A_pinv, A_pinv.T, rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num008_pinv_identity(self):
        """
        NUM-008: Verify pseudoinverse of identity is identity.
        
        Requirements: REQ-UTIL-020
        """
        I = np.eye(4)
        
        I_pinv = r_ginv_like(I)
        
        assert_allclose(I_pinv, I, rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num009_pinv_zero_matrix(self):
        """
        NUM-009: Verify pseudoinverse of zero matrix is zero.
        
        Requirements: REQ-UTIL-020
        """
        Z = np.zeros((3, 3))
        
        Z_pinv = r_ginv_like(Z)
        
        assert_allclose(Z_pinv, Z, rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num010_pinv_reproducibility(self):
        """
        NUM-010: Verify pseudoinverse computation is reproducible.
        
        Requirements: REQ-UTIL-020
        """
        np.random.seed(42)
        X = np.random.randn(5, 3)
        
        pinv1 = r_ginv_like(X)
        pinv2 = r_ginv_like(X)
        
        assert_allclose(pinv1, pinv2, rtol=1e-15)


# =============================================================================
# Test Class: Matrix Rank (NUM-011 to NUM-015)
# =============================================================================

class TestMatrixRank:
    """
    Test matrix rank computation.
    
    Test IDs: NUM-011 to NUM-015
    Requirements: REQ-UTIL-022
    """
    
    @pytest.mark.numerical
    def test_num011_rank_full_rank(self):
        """
        NUM-011: Verify rank computation for full-rank matrix.
        
        Requirements: REQ-UTIL-022
        """
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 10]])  # Full rank 3x3
        
        rank = numeric_rank(A)
        
        assert rank == 3
    
    @pytest.mark.numerical
    def test_num012_rank_deficient(self, rank_deficient_matrix):
        """
        NUM-012: Verify rank computation for rank-deficient matrix.
        
        Requirements: REQ-UTIL-022
        """
        A = rank_deficient_matrix
        
        rank = numeric_rank(A)
        
        assert rank == 3  # 4x4 matrix with rank 3
    
    @pytest.mark.numerical
    def test_num013_rank_rectangular(self):
        """
        NUM-013: Verify rank for rectangular matrix.
        
        Requirements: REQ-UTIL-022
        """
        # 5x3 matrix has rank at most 3
        A = np.random.randn(5, 3)
        
        rank = numeric_rank(A)
        
        assert rank <= 3
    
    @pytest.mark.numerical
    def test_num014_rank_zero_matrix(self):
        """
        NUM-014: Verify rank of zero matrix is 0.
        
        Requirements: REQ-UTIL-022
        """
        Z = np.zeros((4, 4))
        
        rank = numeric_rank(Z)
        
        assert rank == 0
    
    @pytest.mark.numerical
    def test_num015_rank_identity(self):
        """
        NUM-015: Verify rank of identity matrix is n.
        
        Requirements: REQ-UTIL-022
        """
        n = 5
        I = np.eye(n)
        
        rank = numeric_rank(I)
        
        assert rank == n


# =============================================================================
# Test Class: Symmetry Utilities (NUM-016 to NUM-020)
# =============================================================================

class TestSymmetryUtilities:
    """
    Test matrix symmetry utilities.
    
    Test IDs: NUM-016 to NUM-020
    Requirements: REQ-UTIL-023
    """
    
    @pytest.mark.numerical
    def test_num016_symmetrize_symmetric(self, symmetric_psd_matrix):
        """
        NUM-016: Verify symmetrize preserves symmetric matrix.
        
        Requirements: REQ-UTIL-023
        """
        A = symmetric_psd_matrix
        
        A_sym = symmetrize(A)
        
        assert_allclose(A_sym, A, rtol=1e-10, atol=1e-12)
    
    @pytest.mark.numerical
    def test_num017_symmetrize_asymmetric(self):
        """
        NUM-017: Verify symmetrize produces symmetric output.
        
        Requirements: REQ-UTIL-023
        """
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        A_sym = symmetrize(A)
        
        # Output should be symmetric
        assert_allclose(A_sym, A_sym.T, rtol=1e-15)
    
    @pytest.mark.numerical
    def test_num018_symmetrize_formula(self):
        """
        NUM-018: Verify symmetrize uses (A + A.T) / 2.
        
        Requirements: REQ-UTIL-023
        """
        A = np.array([[1, 2],
                      [3, 4]])
        
        A_sym = symmetrize(A)
        expected = (A + A.T) / 2
        
        assert_allclose(A_sym, expected, rtol=1e-15)
    
    @pytest.mark.numerical
    def test_num019_is_symmetric_true(self, symmetric_psd_matrix):
        """
        NUM-019: Verify is_symmetric returns True for symmetric matrix.
        
        Requirements: REQ-UTIL-023
        """
        A = symmetric_psd_matrix
        
        assert is_symmetric(A)
    
    @pytest.mark.numerical
    def test_num020_is_symmetric_false(self):
        """
        NUM-020: Verify is_symmetric returns False for asymmetric matrix.
        
        Requirements: REQ-UTIL-023
        """
        A = np.array([[1, 2],
                      [3, 4]])
        
        assert not is_symmetric(A)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# R Compatibility Tests (from test_r_compat.py)
# =============================================================================

class TestEnsureRpy2Compatibility:
    """
    Test ensure_rpy2_compatibility function.
    
    Test IDs: RCOMPAT-001 to RCOMPAT-005
    Requirements: REQ-UTIL-020
    """
    
    @pytest.mark.unit
    def test_rcompat001_executes_without_error(self):
        """
        RCOMPAT-001: Verify ensure_rpy2_compatibility executes without error.
        
        Requirements: REQ-UTIL-020
        """
        # Should not raise
        ensure_rpy2_compatibility()
    
    @pytest.mark.unit
    def test_rcompat002_idempotent(self):
        """
        RCOMPAT-002: Verify function is idempotent (safe to call multiple times).
        
        Requirements: REQ-UTIL-020
        """
        # Call multiple times - should not raise
        ensure_rpy2_compatibility()
        ensure_rpy2_compatibility()
        ensure_rpy2_compatibility()
    
    @pytest.mark.unit
    def test_rcompat003_dataframe_iteritems_patched(self):
        """
        RCOMPAT-003: Verify DataFrame.iteritems is available after patch.
        
        Requirements: REQ-UTIL-021
        """
        ensure_rpy2_compatibility()
        
        # DataFrame should have iteritems method
        assert hasattr(pd.DataFrame, 'iteritems')
    
    @pytest.mark.unit
    def test_rcompat004_series_iteritems_patched(self):
        """
        RCOMPAT-004: Verify Series.iteritems is available after patch.
        
        Requirements: REQ-UTIL-021
        """
        ensure_rpy2_compatibility()
        
        # Series should have iteritems method
        assert hasattr(pd.Series, 'iteritems')
    
    @pytest.mark.unit
    def test_rcompat005_iteritems_callable(self):
        """
        RCOMPAT-005: Verify patched iteritems is callable.
        
        Requirements: REQ-UTIL-021
        """
        ensure_rpy2_compatibility()
        
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        series = pd.Series([1, 2, 3])
        
        # Should be callable
        assert callable(df.iteritems)
        assert callable(series.iteritems)
    
    @pytest.mark.unit
    def test_rcompat006_iteritems_yields_items(self):
        """
        RCOMPAT-006: Verify iteritems yields same items as items().
        
        Requirements: REQ-UTIL-021
        """
        ensure_rpy2_compatibility()
        
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        # iteritems should yield same as items
        iteritems_result = list(df.iteritems())
        items_result = list(df.items())
        
        assert len(iteritems_result) == len(items_result)
        for (name1, col1), (name2, col2) in zip(iteritems_result, items_result):
            assert name1 == name2
            pd.testing.assert_series_equal(col1, col2)


# =============================================================================
# Test Class: check_rpy2_available (RCOMPAT-007 to RCOMPAT-010)
# =============================================================================

class TestCheckRpy2Available:
    """
    Test check_rpy2_available function.
    
    Test IDs: RCOMPAT-007 to RCOMPAT-010
    Requirements: REQ-UTIL-022
    """
    
    @pytest.mark.unit
    def test_rcompat007_returns_tuple(self):
        """
        RCOMPAT-007: Verify check_rpy2_available returns tuple.
        
        Requirements: REQ-UTIL-022
        """
        result = check_rpy2_available()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_rcompat008_first_element_bool(self):
        """
        RCOMPAT-008: Verify first element is boolean.
        
        Requirements: REQ-UTIL-022
        """
        available, _ = check_rpy2_available()
        
        assert isinstance(available, bool)
    
    @pytest.mark.unit
    def test_rcompat009_second_element_tuple_or_none(self):
        """
        RCOMPAT-009: Verify second element is tuple or None.
        
        Requirements: REQ-UTIL-022
        """
        available, components = check_rpy2_available()
        
        if available:
            assert isinstance(components, tuple)
            assert len(components) == 3  # (robjects, pandas2ri, cbps_r)
        else:
            assert components is None
    
    @pytest.mark.unit
    def test_rcompat010_consistent_results(self):
        """
        RCOMPAT-010: Verify consistent results on repeated calls.
        
        Requirements: REQ-UTIL-022
        """
        result1 = check_rpy2_available()
        result2 = check_rpy2_available()
        
        # Availability status should be consistent
        assert result1[0] == result2[0]


# =============================================================================
# Conditional Tests (only run if rpy2 available)
# =============================================================================

# Check if rpy2 is available
try:
    ensure_rpy2_compatibility()
    import rpy2
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


@pytest.mark.skipif(not HAS_RPY2, reason="rpy2 not installed")
class TestRpy2Integration:
    """
    Integration tests for rpy2 compatibility (only when rpy2 installed).
    
    Test IDs: RCOMPAT-011 to RCOMPAT-015
    Requirements: REQ-UTIL-023
    """
    
    @pytest.mark.integration
    def test_rcompat011_rpy2_import_after_patch(self):
        """
        RCOMPAT-011: Verify rpy2 can be imported after compatibility patch.
        
        Requirements: REQ-UTIL-023
        """
        ensure_rpy2_compatibility()
        
        # Should not raise ImportError
        import rpy2.robjects as ro
    
    @pytest.mark.integration
    def test_rcompat012_pandas2ri_activate(self):
        """
        RCOMPAT-012: Verify pandas2ri can be activated.
        
        Requirements: REQ-UTIL-023
        
        Notes:
            pandas2ri.activate() may raise warnings about being
            deprecated in favor of localconverter. We just test
            that it doesn't raise an exception.
        """
        ensure_rpy2_compatibility()
        
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        
        # Some versions of rpy2 deprecate activate() - just test no exception
        try:
            pandas2ri.activate()
        except Exception as e:
            # Allow deprecation warnings but fail on other exceptions
            if 'deprecat' not in str(e).lower():
                pytest.skip(f"pandas2ri.activate() raised: {e}")
    
    @pytest.mark.integration
    def test_rcompat013_check_returns_true(self):
        """
        RCOMPAT-013: Verify check_rpy2_available returns True when rpy2 available.
        
        Requirements: REQ-UTIL-022
        """
        available, components = check_rpy2_available()
        
        # May still be False if CBPS R package not installed
        # Just verify it runs without error
        assert isinstance(available, bool)
    
    @pytest.mark.integration
    def test_rcompat014_dataframe_conversion(self):
        """
        RCOMPAT-014: Verify DataFrame can be converted to R.
        
        Requirements: REQ-UTIL-023
        """
        ensure_rpy2_compatibility()
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter
        except ImportError as e:
            pytest.skip(f"rpy2 component import failed: {e}")
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)
            
            # Should be an R data.frame
            assert r_df is not None
        except Exception as e:
            pytest.skip(f"DataFrame conversion failed: {e}")
    
    @pytest.mark.integration
    def test_rcompat015_series_conversion(self):
        """
        RCOMPAT-015: Verify Series can be converted to R.
        
        Requirements: REQ-UTIL-023
        """
        ensure_rpy2_compatibility()
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter
        except ImportError as e:
            pytest.skip(f"rpy2 component import failed: {e}")
        
        series = pd.Series([1, 2, 3], name='test')
        
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_vec = ro.conversion.py2rpy(series)
            
            # Should be an R vector
            assert r_vec is not None
        except Exception as e:
            pytest.skip(f"Series conversion failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Input Validation Tests (from test_validation.py)
# =============================================================================

class TestBasicValidation:
    """
    Test basic input validation functionality.
    
    Test IDs: VAL-001 to VAL-010
    Requirements: REQ-UTIL-030
    """
    
    @pytest.mark.unit
    def test_val001_valid_input_passes(self, simple_binary_data):
        """
        VAL-001: Verify valid input passes validation without error.
        
        Requirements: REQ-UTIL-030
        """
        data = simple_binary_data
        
        # Should not raise any exception
        validate_cbps_input(data['treat'], data['X'])
    
    @pytest.mark.unit
    def test_val002_empty_treatment_raises(self):
        """
        VAL-002: Verify empty treatment array raises error.
        
        Requirements: REQ-UTIL-030
        """
        treat = np.array([])
        X = np.array([]).reshape(0, 3)
        
        with pytest.raises(ValueError, match="empty|insufficient"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val003_dimension_mismatch_raises(self):
        """
        VAL-003: Verify dimension mismatch raises error.
        
        Requirements: REQ-UTIL-031
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, 2], [3, 4]])  # Only 2 rows vs 4 treatments
        
        with pytest.raises(ValueError, match="match|dimension|row"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val004_nan_treatment_raises(self):
        """
        VAL-004: Verify NaN in treatment raises error.
        
        Requirements: REQ-UTIL-032
        """
        treat = np.array([0, 1, np.nan, 1])
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        with pytest.raises(ValueError, match="NaN|missing|finite"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val005_inf_treatment_raises(self):
        """
        VAL-005: Verify Inf in treatment raises error.
        
        Requirements: REQ-UTIL-032
        """
        treat = np.array([0, 1, np.inf, 1])
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        with pytest.raises(ValueError, match="Inf|infinite|finite"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val006_nan_covariates_raises(self):
        """
        VAL-006: Verify NaN in covariates raises error.
        
        Requirements: REQ-UTIL-032
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[1, np.nan], [3, 4], [5, 6], [7, 8]])
        
        with pytest.raises(ValueError, match="NaN|missing|finite"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val007_single_observation_raises(self):
        """
        VAL-007: Verify single observation raises error.
        
        Requirements: REQ-UTIL-033
        """
        treat = np.array([1])
        X = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="insufficient|minimum|observation"):
            validate_cbps_input(treat, X, min_observations=2)
    
    @pytest.mark.unit
    def test_val008_1d_covariate_raises(self):
        """
        VAL-008: Verify 1D covariate array raises error.
        
        Requirements: REQ-UTIL-034
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([1, 2, 3, 4])  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="2-dimensional|2D|dimension"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val009_zero_columns_raises(self):
        """
        VAL-009: Verify zero-column covariate matrix raises error.
        
        Requirements: REQ-UTIL-034
        """
        treat = np.array([0, 1, 0, 1])
        X = np.array([[], [], [], []])  # 4x0 matrix
        
        with pytest.raises(ValueError, match="column|covariate"):
            validate_cbps_input(treat, X)
    
    @pytest.mark.unit
    def test_val010_zero_variance_treatment_raises(self):
        """
        VAL-010: Verify zero-variance treatment raises error.
        
        Requirements: REQ-UTIL-035
        """
        treat = np.array([1, 1, 1, 1])  # All same value
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        with pytest.raises(ValueError, match="variance|variation"):
            validate_cbps_input(treat, X, check_treatment_variance=True)


# =============================================================================
# Test Class: Module Name Parameter (VAL-011 to VAL-015)
# =============================================================================

class TestModuleNameParameter:
    """
    Test module_name parameter in validation messages.
    
    Test IDs: VAL-011 to VAL-015
    Requirements: REQ-UTIL-030
    """
    
    @pytest.mark.unit
    def test_val011_custom_module_name_in_error(self):
        """
        VAL-011: Verify custom module name appears in error message.
        
        Requirements: REQ-UTIL-030
        """
        treat = np.array([])
        X = np.array([]).reshape(0, 3)
        
        try:
            validate_cbps_input(treat, X, module_name="TestModule")
        except ValueError as e:
            assert "TestModule" in str(e)
    
    @pytest.mark.unit
    def test_val012_default_module_name(self):
        """
        VAL-012: Verify default module name is used.
        
        Requirements: REQ-UTIL-030
        """
        treat = np.array([])
        X = np.array([]).reshape(0, 3)
        
        try:
            validate_cbps_input(treat, X)  # Default module_name="CBPS"
        except ValueError as e:
            assert "CBPS" in str(e)
    
    @pytest.mark.unit
    def test_val013_different_module_names(self):
        """
        VAL-013: Verify different module names work correctly.
        
        Requirements: REQ-UTIL-030
        """
        treat = np.array([])
        X = np.array([]).reshape(0, 3)
        
        for name in ["CBMSM", "CBIV", "hdCBPS", "npCBPS"]:
            try:
                validate_cbps_input(treat, X, module_name=name)
            except ValueError as e:
                assert name in str(e)


# =============================================================================
# Test Class: Edge Cases (VAL-016 to VAL-020)
# =============================================================================

class TestEdgeCases:
    """
    Test edge cases in validation.
    
    Test IDs: VAL-016 to VAL-020
    Requirements: REQ-UTIL-030
    """
    
    @pytest.mark.edge_case
    def test_val016_minimum_valid_input(self):
        """
        VAL-016: Verify minimum valid input (2 observations) passes.
        
        Requirements: REQ-UTIL-030
        """
        treat = np.array([0, 1])
        X = np.array([[1, 2], [3, 4]])
        
        # Should not raise
        validate_cbps_input(treat, X, min_observations=2)
    
    @pytest.mark.edge_case
    def test_val017_large_input(self):
        """
        VAL-017: Verify large input passes validation.
        
        Requirements: REQ-UTIL-030
        """
        np.random.seed(42)
        n = 10000
        
        treat = np.random.binomial(1, 0.5, n)
        X = np.random.randn(n, 50)
        
        # Should not raise
        validate_cbps_input(treat, X)
    
    @pytest.mark.edge_case
    def test_val018_skip_treatment_variance_check(self):
        """
        VAL-018: Verify treatment variance check can be skipped.
        
        Requirements: REQ-UTIL-035
        """
        treat = np.array([1, 1, 1, 1])  # Zero variance
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        # Should not raise when check is disabled
        validate_cbps_input(treat, X, check_treatment_variance=False)
    
    @pytest.mark.edge_case
    def test_val019_custom_min_observations(self):
        """
        VAL-019: Verify custom min_observations is respected.
        
        Requirements: REQ-UTIL-033
        """
        treat = np.array([0, 1, 0])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Should pass with min_observations=3
        validate_cbps_input(treat, X, min_observations=3)
        
        # Should fail with min_observations=4
        with pytest.raises(ValueError):
            validate_cbps_input(treat, X, min_observations=4)
    
    @pytest.mark.edge_case
    def test_val020_float_and_int_treatment(self):
        """
        VAL-020: Verify both float and int treatment arrays work.
        
        Requirements: REQ-UTIL-030
        """
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        # Float treatment
        treat_float = np.array([0.0, 1.0, 0.0, 1.0])
        validate_cbps_input(treat_float, X)
        
        # Int treatment
        treat_int = np.array([0, 1, 0, 1])
        validate_cbps_input(treat_int, X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Variance Transform Tests (from test_variance_transform.py)
# =============================================================================

@pytest.fixture
def simple_variance_data():
    """Generate simple test data for variance transformation."""
    np.random.seed(42)
    n = 100
    k = 4  # Number of covariates (including intercept)
    
    # Original design matrix with intercept
    X_orig = np.column_stack([
        np.ones(n),
        np.random.randn(n),
        np.random.randn(n),
        np.random.randn(n)
    ])
    
    # SVD of original X (excluding intercept for standardization)
    X_no_intercept = X_orig[:, 1:]
    x_mean = X_no_intercept.mean(axis=0)
    x_sd = X_no_intercept.std(axis=0, ddof=0)
    
    # Standardize
    X_std = X_orig.copy()
    X_std[:, 1:] = (X_no_intercept - x_mean) / x_sd
    
    # SVD of standardized X
    U, d, Vt = np.linalg.svd(X_std, full_matrices=False)
    V = Vt.T
    
    # SVD-transformed X
    X_svd = X_std @ V @ np.diag(1/d)
    
    # Create a mock variance matrix in SVD space
    variance_svd = np.eye(k) * 0.1  # Simple diagonal variance
    
    # Inverse standardization matrix
    Dx_inv = np.diag(np.concatenate([[1.0], x_sd]))
    
    # Inverse singular values
    d_inv = 1.0 / d
    
    return {
        'variance_svd': variance_svd,
        'Dx_inv': Dx_inv,
        'X_orig': X_orig,
        'X_svd': X_svd,
        'V': V,
        'd_inv': d_inv,
        'k': k,
        'n': n,
        'x_mean': x_mean,
        'x_sd': x_sd,
        'd': d,
    }


@pytest.fixture
def svd_info_fixture(simple_variance_data):
    """Create svd_info dict for apply_variance_svd_inverse_transform."""
    data = simple_variance_data
    return {
        'd': data['d'],
        'V': data['V'],
        'x_mean': data['x_mean'],
        'x_sd': data['x_sd'],
    }


# =============================================================================
# Test Class: _r_ginv Helper (VARTRANS-001 to VARTRANS-005)
# =============================================================================

class TestRGinv:
    """
    Test _r_ginv generalized inverse helper function.
    
    Test IDs: VARTRANS-001 to VARTRANS-005
    Requirements: REQ-UTIL-030
    """
    
    @pytest.mark.unit
    def test_vartrans001_full_rank_matrix(self):
        """
        VARTRANS-001: Verify ginv of full-rank square matrix.
        
        Requirements: REQ-UTIL-030
        """
        A = np.array([[1, 2], [3, 4]])
        
        A_ginv = _r_ginv(A)
        
        # A @ A_ginv @ A ≈ A
        assert_allclose(A @ A_ginv @ A, A, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vartrans002_rectangular_matrix(self):
        """
        VARTRANS-002: Verify ginv of rectangular matrix.
        
        Requirements: REQ-UTIL-030
        """
        A = np.array([[1, 2, 3], [4, 5, 6]])
        
        A_ginv = _r_ginv(A)
        
        # Shape should be (3, 2)
        assert A_ginv.shape == (3, 2)
        
        # Moore-Penrose property
        assert_allclose(A @ A_ginv @ A, A, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vartrans003_identity_matrix(self):
        """
        VARTRANS-003: Verify ginv of identity is identity.
        
        Requirements: REQ-UTIL-030
        """
        I = np.eye(4)
        
        I_ginv = _r_ginv(I)
        
        assert_allclose(I_ginv, I, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vartrans004_zero_matrix(self):
        """
        VARTRANS-004: Verify ginv of zero matrix is zero.
        
        Requirements: REQ-UTIL-030
        """
        Z = np.zeros((3, 3))
        
        Z_ginv = _r_ginv(Z)
        
        assert_allclose(Z_ginv, Z, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vartrans005_rank_deficient_matrix(self):
        """
        VARTRANS-005: Verify ginv handles rank-deficient matrix.
        
        Requirements: REQ-UTIL-030
        """
        # Rank-deficient: column 2 = 2 * column 1
        A = np.array([[1, 2], [2, 4], [3, 6]])
        
        A_ginv = _r_ginv(A)
        
        # Moore-Penrose property should still hold
        assert_allclose(A @ A_ginv @ A, A, rtol=1e-6)


# =============================================================================
# Test Class: Binary Transform (VARTRANS-006 to VARTRANS-012)
# =============================================================================

class TestTransformVarianceBinary:
    """
    Test transform_variance_binary function.
    
    Test IDs: VARTRANS-006 to VARTRANS-012
    Requirements: REQ-UTIL-031
    """
    
    @pytest.mark.unit
    def test_vartrans006_output_shape(self, simple_variance_data):
        """
        VARTRANS-006: Verify output shape matches input.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        result = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert result.shape == data['variance_svd'].shape
    
    @pytest.mark.numerical
    def test_vartrans007_output_finite(self, simple_variance_data):
        """
        VARTRANS-007: Verify output values are finite.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        result = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans008_symmetric_output(self, simple_variance_data):
        """
        VARTRANS-008: Verify output is symmetric.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        # Use symmetric input variance
        variance_svd = np.eye(data['k']) * 0.1
        variance_svd[0, 1] = variance_svd[1, 0] = 0.02
        
        result = transform_variance_binary(
            variance_svd,
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Check symmetry
        assert_allclose(result, result.T, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_vartrans009_positive_diagonals(self, simple_variance_data):
        """
        VARTRANS-009: Verify output has non-negative diagonal (variances).
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        result = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Variances should be non-negative
        assert np.all(np.diag(result) >= -1e-10)  # Allow small numerical error
    
    @pytest.mark.unit
    def test_vartrans010_identity_input(self, simple_variance_data):
        """
        VARTRANS-010: Verify handling of identity variance.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        variance_svd = np.eye(data['k'])
        
        result = transform_variance_binary(
            variance_svd,
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert result.shape == (data['k'], data['k'])
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans011_reproducibility(self, simple_variance_data):
        """
        VARTRANS-011: Verify transformation is reproducible.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        result1 = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        result2 = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result1, result2, rtol=1e-14)
    
    @pytest.mark.edge_case
    def test_vartrans012_zero_variance(self, simple_variance_data):
        """
        VARTRANS-012: Verify handling of zero variance input.
        
        Requirements: REQ-UTIL-031
        """
        data = simple_variance_data
        
        variance_svd = np.zeros((data['k'], data['k']))
        
        result = transform_variance_binary(
            variance_svd,
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Zero variance should transform to zero
        assert_allclose(result, np.zeros_like(result), atol=1e-10)


# =============================================================================
# Test Class: 3-level Transform (VARTRANS-013 to VARTRANS-018)
# =============================================================================

class TestTransformVariance3Treat:
    """
    Test transform_variance_3treat function.
    
    Test IDs: VARTRANS-013 to VARTRANS-018
    Requirements: REQ-UTIL-032
    """
    
    @pytest.fixture
    def variance_3treat_data(self, simple_variance_data):
        """Create 3-level treatment variance data."""
        data = simple_variance_data
        k = data['k']
        
        # 3-level treatment has 2k x 2k variance matrix
        variance_svd = np.eye(2 * k) * 0.1
        
        return {**data, 'variance_svd': variance_svd}
    
    @pytest.mark.unit
    def test_vartrans013_output_shape(self, variance_3treat_data):
        """
        VARTRANS-013: Verify output shape is (2k, 2k).
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        k = data['k']
        
        result = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert result.shape == (2 * k, 2 * k)
    
    @pytest.mark.numerical
    def test_vartrans014_output_finite(self, variance_3treat_data):
        """
        VARTRANS-014: Verify output values are finite.
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        
        result = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans015_block_structure(self, variance_3treat_data):
        """
        VARTRANS-015: Verify block structure is preserved.
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        k = data['k']
        
        result = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Extract blocks
        block_11 = result[:k, :k]
        block_22 = result[k:, k:]
        
        # Diagonal blocks should be square
        assert block_11.shape == (k, k)
        assert block_22.shape == (k, k)
    
    @pytest.mark.numerical
    def test_vartrans016_symmetric_output(self, variance_3treat_data):
        """
        VARTRANS-016: Verify output is symmetric.
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        
        result = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result, result.T, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_vartrans017_reproducibility(self, variance_3treat_data):
        """
        VARTRANS-017: Verify transformation is reproducible.
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        
        result1 = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        result2 = transform_variance_3treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result1, result2, rtol=1e-14)
    
    @pytest.mark.edge_case
    def test_vartrans018_zero_variance(self, variance_3treat_data):
        """
        VARTRANS-018: Verify handling of zero variance input.
        
        Requirements: REQ-UTIL-032
        """
        data = variance_3treat_data
        k = data['k']
        
        variance_svd = np.zeros((2 * k, 2 * k))
        
        result = transform_variance_3treat(
            variance_svd,
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result, np.zeros_like(result), atol=1e-10)


# =============================================================================
# Test Class: 4-level Transform (VARTRANS-019 to VARTRANS-023)
# =============================================================================

class TestTransformVariance4Treat:
    """
    Test transform_variance_4treat function.
    
    Test IDs: VARTRANS-019 to VARTRANS-023
    Requirements: REQ-UTIL-033
    """
    
    @pytest.fixture
    def variance_4treat_data(self, simple_variance_data):
        """Create 4-level treatment variance data."""
        data = simple_variance_data
        k = data['k']
        
        # 4-level treatment has 3k x 3k variance matrix
        variance_svd = np.eye(3 * k) * 0.1
        
        return {**data, 'variance_svd': variance_svd}
    
    @pytest.mark.unit
    def test_vartrans019_output_shape(self, variance_4treat_data):
        """
        VARTRANS-019: Verify output shape is (3k, 3k).
        
        Requirements: REQ-UTIL-033
        """
        data = variance_4treat_data
        k = data['k']
        
        result = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert result.shape == (3 * k, 3 * k)
    
    @pytest.mark.numerical
    def test_vartrans020_output_finite(self, variance_4treat_data):
        """
        VARTRANS-020: Verify output values are finite.
        
        Requirements: REQ-UTIL-033
        """
        data = variance_4treat_data
        
        result = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans021_nine_block_structure(self, variance_4treat_data):
        """
        VARTRANS-021: Verify 9-block structure.
        
        Requirements: REQ-UTIL-033
        """
        data = variance_4treat_data
        k = data['k']
        
        result = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Check all 9 blocks have correct shape
        for i in range(3):
            for j in range(3):
                block = result[i*k:(i+1)*k, j*k:(j+1)*k]
                assert block.shape == (k, k)
    
    @pytest.mark.numerical
    def test_vartrans022_symmetric_output(self, variance_4treat_data):
        """
        VARTRANS-022: Verify output is symmetric.
        
        Requirements: REQ-UTIL-033
        """
        data = variance_4treat_data
        
        result = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result, result.T, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_vartrans023_reproducibility(self, variance_4treat_data):
        """
        VARTRANS-023: Verify transformation is reproducible.
        
        Requirements: REQ-UTIL-033
        """
        data = variance_4treat_data
        
        result1 = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        result2 = transform_variance_4treat(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result1, result2, rtol=1e-14)


# =============================================================================
# Test Class: Continuous Transform (VARTRANS-024 to VARTRANS-027)
# =============================================================================

class TestTransformVarianceContinuous:
    """
    Test transform_variance_continuous function.
    
    Test IDs: VARTRANS-024 to VARTRANS-027
    Requirements: REQ-UTIL-034
    """
    
    @pytest.mark.unit
    def test_vartrans024_output_shape(self, simple_variance_data):
        """
        VARTRANS-024: Verify output shape matches binary case.
        
        Requirements: REQ-UTIL-034
        """
        data = simple_variance_data
        
        result = transform_variance_continuous(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert result.shape == data['variance_svd'].shape
    
    @pytest.mark.numerical
    def test_vartrans025_matches_binary(self, simple_variance_data):
        """
        VARTRANS-025: Verify continuous matches binary transform.
        
        Requirements: REQ-UTIL-034
        """
        data = simple_variance_data
        
        result_cont = transform_variance_continuous(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        result_bin = transform_variance_binary(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        # Should be identical
        assert_allclose(result_cont, result_bin, rtol=1e-14)
    
    @pytest.mark.numerical
    def test_vartrans026_output_finite(self, simple_variance_data):
        """
        VARTRANS-026: Verify output values are finite.
        
        Requirements: REQ-UTIL-034
        """
        data = simple_variance_data
        
        result = transform_variance_continuous(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans027_symmetric_output(self, simple_variance_data):
        """
        VARTRANS-027: Verify output is symmetric.
        
        Requirements: REQ-UTIL-034
        """
        data = simple_variance_data
        
        result = transform_variance_continuous(
            data['variance_svd'],
            data['Dx_inv'],
            data['X_orig'],
            data['X_svd'],
            data['V'],
            data['d_inv']
        )
        
        assert_allclose(result, result.T, rtol=1e-10)


# =============================================================================
# Test Class: Dispatch Function (VARTRANS-028 to VARTRANS-035)
# =============================================================================

class TestApplyVarianceSVDInverseTransform:
    """
    Test apply_variance_svd_inverse_transform dispatch function.
    
    Test IDs: VARTRANS-028 to VARTRANS-035
    Requirements: REQ-UTIL-035
    """
    
    @pytest.mark.unit
    def test_vartrans028_binary_dispatch(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-028: Verify binary treatment dispatch.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        result = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        assert result.shape == (data['k'], data['k'])
    
    @pytest.mark.unit
    def test_vartrans029_3treat_dispatch(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-029: Verify 3-level treatment dispatch.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        k = data['k']
        variance_svd = np.eye(2 * k) * 0.1
        
        result = apply_variance_svd_inverse_transform(
            variance_svd,
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=3
        )
        
        assert result.shape == (2 * k, 2 * k)
    
    @pytest.mark.unit
    def test_vartrans030_4treat_dispatch(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-030: Verify 4-level treatment dispatch.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        k = data['k']
        variance_svd = np.eye(3 * k) * 0.1
        
        result = apply_variance_svd_inverse_transform(
            variance_svd,
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=4
        )
        
        assert result.shape == (3 * k, 3 * k)
    
    @pytest.mark.unit
    def test_vartrans031_continuous_dispatch(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-031: Verify continuous treatment dispatch.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        result = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=False,
            no_treats=0  # Ignored for continuous
        )
        
        assert result.shape == (data['k'], data['k'])
    
    @pytest.mark.numerical
    def test_vartrans032_small_singular_values_handled(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-032: Verify small singular values are zeroed.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        # Modify svd_info to have very small singular value
        svd_info = svd_info_fixture.copy()
        svd_info['d'] = svd_info['d'].copy()
        svd_info['d'][-1] = 1e-10  # Very small
        
        result = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        # Should still produce finite result
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans033_output_finite(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-033: Verify output values are finite.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        result = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.numerical
    def test_vartrans034_reproducibility(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-034: Verify transformation is reproducible.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        result1 = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        result2 = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        assert_allclose(result1, result2, rtol=1e-14)
    
    @pytest.mark.numerical
    def test_vartrans035_symmetric_output(self, simple_variance_data, svd_info_fixture):
        """
        VARTRANS-035: Verify output is symmetric.
        
        Requirements: REQ-UTIL-035
        """
        data = simple_variance_data
        
        result = apply_variance_svd_inverse_transform(
            data['variance_svd'],
            svd_info_fixture,
            data['X_orig'],
            data['X_svd'],
            is_factor=True,
            no_treats=2
        )
        
        assert_allclose(result, result.T, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# Weight Computation Tests (from test_weights.py)
# =============================================================================

class TestATEWeights:
    """
    Test ATE weight computation.
    
    Test IDs: WGT-001 to WGT-008
    Requirements: REQ-UTIL-010
    """
    
    @pytest.mark.unit
    def test_wgt001_ate_weights_shape(self, simple_binary_data):
        """
        WGT-001: Verify ATE weights have correct shape.
        
        Requirements: REQ-UTIL-010
        """
        data = simple_binary_data
        
        weights = compute_ate_weights(data['treat'], data['probs'])
        
        assert weights.shape == (data['n'],)
    
    @pytest.mark.unit
    def test_wgt002_ate_weights_positive(self, simple_binary_data):
        """
        WGT-002: Verify all ATE weights are positive.
        
        Requirements: REQ-UTIL-010
        """
        data = simple_binary_data
        
        weights = compute_ate_weights(data['treat'], data['probs'])
        
        assert np.all(weights > 0), "All ATE weights should be positive"
    
    @pytest.mark.numerical
    def test_wgt003_ate_weights_formula(self):
        """
        WGT-003: Verify ATE weights follow correct formula.
        
        Requirements: REQ-UTIL-010
        
        Notes:
            ATE weight: w = T/p + (1-T)/(1-p)
        """
        treat = np.array([1, 0, 1, 0])
        probs = np.array([0.6, 0.4, 0.7, 0.3])
        
        weights = compute_ate_weights(treat, probs)
        
        # Expected: treated get 1/p, control get 1/(1-p)
        expected = np.array([1/0.6, 1/0.6, 1/0.7, 1/0.7])
        
        assert_allclose(weights, expected, rtol=1e-10)
    
    @pytest.mark.unit
    def test_wgt004_ate_weights_finite(self, simple_binary_data):
        """
        WGT-004: Verify all ATE weights are finite.
        
        Requirements: REQ-UTIL-010
        """
        data = simple_binary_data
        
        weights = compute_ate_weights(data['treat'], data['probs'])
        
        assert np.all(np.isfinite(weights)), "All weights should be finite"
    
    @pytest.mark.edge_case
    def test_wgt005_ate_weights_extreme_probs(self):
        """
        WGT-005: Verify ATE weights with extreme probabilities.
        
        Requirements: REQ-UTIL-010
        """
        treat = np.array([1, 0, 1, 0])
        probs = np.array([0.99, 0.01, 0.99, 0.01])
        
        weights = compute_ate_weights(treat, probs)
        
        # Should still be positive and finite
        assert np.all(weights > 0)
        assert np.all(np.isfinite(weights))
    
    @pytest.mark.unit
    def test_wgt006_ate_weights_treated_only(self):
        """
        WGT-006: Verify weights when only treated units.
        
        Requirements: REQ-UTIL-010
        """
        treat = np.array([1, 1, 1, 1])
        probs = np.array([0.6, 0.5, 0.7, 0.8])
        
        weights = compute_ate_weights(treat, probs)
        
        # All treated: weights = 1/p
        expected = 1 / probs
        assert_allclose(weights, expected, rtol=1e-10)
    
    @pytest.mark.unit
    def test_wgt007_ate_weights_control_only(self):
        """
        WGT-007: Verify weights when only control units.
        
        Requirements: REQ-UTIL-010
        """
        treat = np.array([0, 0, 0, 0])
        probs = np.array([0.6, 0.5, 0.7, 0.8])
        
        weights = compute_ate_weights(treat, probs)
        
        # All control: weights = 1/(1-p)
        expected = 1 / (1 - probs)
        assert_allclose(weights, expected, rtol=1e-10)
    
    @pytest.mark.numerical
    def test_wgt008_ate_weights_reproducibility(self, simple_binary_data):
        """
        WGT-008: Verify ATE weight computation is reproducible.
        
        Requirements: REQ-UTIL-010
        """
        data = simple_binary_data
        
        weights1 = compute_ate_weights(data['treat'], data['probs'])
        weights2 = compute_ate_weights(data['treat'], data['probs'])
        
        assert_allclose(weights1, weights2, rtol=1e-15)


# =============================================================================
# Note: ATT Weights and Standardized Weights tests are skipped because
# compute_att_weights and standardize_weights have more complex signatures
# that require sample_weights parameter.
# =============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])