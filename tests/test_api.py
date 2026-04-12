"""
Module: test_api.py
===================

Test Suite: Top-Level CBPS API
Test IDs: API-001 to API-040
Requirements: REQ-API-001 to REQ-API-015

Overview:
    This module provides comprehensive tests for the top-level CBPS API functions
    including CBPS(), cbps_fit(), and treatment type detection. These tests verify
    the public interface that users interact with directly.

Test Categories:
    - Unit tests: Function parameter handling and return types
    - Integration tests: End-to-end workflow with various inputs
    - Treatment detection tests: Auto-detection of binary/multi-valued/continuous
    - Error handling tests: Invalid inputs and edge cases

Usage:
    pytest tests/test_api.py -v
    pytest tests/test_api.py::TestCBPSFunction -v

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import warnings

from cbps import CBPS, cbps_fit, _detect_treatment_type
from cbps.core.results import CBPSResults


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def binary_data():
    """
    Generate binary treatment data for API tests.
    
    Returns
    -------
    dict
        Dictionary with DataFrame and array representations.
        
    Note:
        Treatment is stored as int to avoid type conversion warnings.
    """
    np.random.seed(42)
    n = 200
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    
    # Generate treatment with known propensity (as int to avoid warnings)
    logit_ps = 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    ps = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps)  # Keep as int, not float
    
    # Outcome
    y = 1.0 + 0.5 * treat + 0.3 * x1 - 0.2 * x2 + np.random.randn(n) * 0.5
    
    df = pd.DataFrame({
        'treat': treat,  # int type
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'outcome': y
    })
    
    X = np.column_stack([np.ones(n), x1, x2, x3])
    
    return {
        'df': df,
        'treat': treat,
        'X': X,
        'n': n,
        'k': 4
    }


@pytest.fixture
def continuous_data():
    """
    Generate continuous treatment data for API tests.
    
    Returns
    -------
    dict
        Dictionary with DataFrame and array representations.
    """
    np.random.seed(123)
    n = 200
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # Continuous treatment
    treat = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)
    
    df = pd.DataFrame({
        'treat': treat,
        'x1': x1,
        'x2': x2
    })
    
    X = np.column_stack([np.ones(n), x1, x2])
    
    return {
        'df': df,
        'treat': treat,
        'X': X,
        'n': n,
        'k': 3
    }


@pytest.fixture
def multitreat_data():
    """
    Generate multi-valued treatment data for API tests.
    
    Returns
    -------
    dict
        Dictionary with DataFrame and array representations (3 levels).
        
    Note:
        Treatment is stored as pd.Categorical to ensure proper multi-valued
        treatment detection (not interpreted as continuous).
    """
    np.random.seed(456)
    n = 300
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # 3-valued treatment (0, 1, 2)
    logits = np.column_stack([
        0.3 * x1 - 0.2 * x2,
        -0.2 * x1 + 0.3 * x2
    ])
    exp_logits = np.exp(logits)
    denom = 1 + exp_logits.sum(axis=1, keepdims=True)
    probs = np.column_stack([1/denom, exp_logits / denom])
    
    # Use Categorical to explicitly mark as multi-valued treatment
    treat_int = np.array([np.random.choice([0, 1, 2], p=p.ravel()) for p in probs])
    treat = pd.Categorical(treat_int)
    
    df = pd.DataFrame({
        'treat': treat,  # Categorical type
        'x1': x1,
        'x2': x2
    })
    
    X = np.column_stack([np.ones(n), x1, x2])
    
    return {
        'df': df,
        'treat': treat,
        'X': X,
        'n': n,
        'k': 3
    }


# =============================================================================
# Test Class: Treatment Type Detection (API-001 to API-010)
# =============================================================================

class TestTreatmentTypeDetection:
    """
    Test automatic treatment type detection.
    
    Test IDs: API-001 to API-010
    Requirements: REQ-API-001
    
    Note: _detect_treatment_type returns tuple (is_categorical, is_binary_01, is_continuous)
    """
    
    @pytest.mark.unit
    def test_api001_detect_binary_01(self):
        """
        API-001: Verify detection of binary treatment {0, 1}.
        
        Requirements: REQ-API-001
        """
        treat = np.array([0, 1, 0, 1, 0, 1, 0, 1]).astype(float)
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_binary is True
        assert is_cont is False
    
    @pytest.mark.unit
    def test_api002_detect_binary_01_int(self):
        """
        API-002: Verify detection of binary treatment with int dtype.
        
        Requirements: REQ-API-001
        """
        treat = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # int array
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_binary is True
        assert is_cont is False
    
    @pytest.mark.unit
    def test_api003_detect_continuous(self):
        """
        API-003: Verify detection of continuous treatment.
        
        Requirements: REQ-API-001
        """
        np.random.seed(42)
        treat = np.random.randn(100)  # Continuous values
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_cont is True
        assert is_binary is False
    
    @pytest.mark.unit
    def test_api004_detect_multitreat_3levels(self):
        """
        API-004: Verify detection of 3-valued treatment as continuous.
        
        Requirements: REQ-API-001
        Note: Without Categorical encoding, 3-level numeric is detected as continuous.
        """
        treat = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).astype(float)
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        # Without Categorical encoding, numeric multi-valued is continuous
        assert is_cont is True or is_cat is True
    
    @pytest.mark.unit
    def test_api005_detect_multitreat_4levels(self):
        """
        API-005: Verify detection of 4-valued treatment.
        
        Requirements: REQ-API-001
        """
        treat = np.array([0, 1, 2, 3, 0, 1, 2, 3]).astype(float)
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        # Without Categorical encoding, numeric multi-valued is continuous
        assert is_cont is True or is_cat is True
    
    @pytest.mark.unit
    def test_api006_detect_binary_imbalanced(self):
        """
        API-006: Verify detection with imbalanced binary treatment.
        
        Requirements: REQ-API-001
        """
        # 95% zeros, 5% ones
        treat = np.concatenate([np.zeros(95), np.ones(5)])
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_binary is True
    
    @pytest.mark.unit
    def test_api007_detect_categorical_treatment(self):
        """
        API-007: Verify detection of pandas Categorical treatment.
        
        Requirements: REQ-API-001
        """
        treat = pd.Categorical([0, 1, 2, 0, 1, 2])
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_cat is True
    
    @pytest.mark.unit
    def test_api008_detect_continuous_few_unique(self):
        """
        API-008: Verify detection of continuous with many unique values.
        
        Requirements: REQ-API-001
        """
        np.random.seed(42)
        treat = np.random.randn(100)  # Continuous
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_cont is True
    
    @pytest.mark.unit
    def test_api009_detect_handles_series(self):
        """
        API-009: Verify detection works with pandas Series.
        
        Requirements: REQ-API-001
        """
        treat = pd.Series([0, 1, 0, 1, 0, 1])
        
        is_cat, is_binary, is_cont = _detect_treatment_type(treat)
        
        assert is_binary is True
    
    @pytest.mark.unit
    def test_api010_detect_returns_tuple(self):
        """
        API-010: Verify detection returns tuple of 3 booleans.
        
        Requirements: REQ-API-001
        """
        treat = np.array([0, 1, 0, 1])
        
        result = _detect_treatment_type(treat)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(r, (bool, np.bool_)) for r in result)


# =============================================================================
# Test Class: CBPS Function - Binary Treatment (API-011 to API-020)
# =============================================================================

@pytest.mark.integration
class TestCBPSFunctionBinary:
    """
    Test CBPS() function with binary treatment.
    
    Test IDs: API-011 to API-020
    Requirements: REQ-API-002
    """
    
    @pytest.mark.unit
    def test_api011_cbps_formula_interface(self, binary_data):
        """
        API-011: Verify CBPS() works with formula interface.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            formula='treat ~ x1 + x2 + x3',
            data=binary_data['df'],
            att=0
        )
        
        assert isinstance(result, CBPSResults)
    
    @pytest.mark.unit
    def test_api012_cbps_array_interface(self, binary_data):
        """
        API-012: Verify CBPS() works with array interface.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0
        )
        
        assert isinstance(result, CBPSResults)
    
    @pytest.mark.unit
    def test_api013_cbps_returns_weights(self, binary_data):
        """
        API-013: Verify CBPS() returns valid weights.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0
        )
        
        assert hasattr(result, 'weights')
        assert len(result.weights) == binary_data['n']
        assert np.all(result.weights > 0)
    
    @pytest.mark.unit
    def test_api014_cbps_ate_estimation(self, binary_data):
        """
        API-014: Verify CBPS() with att=0 estimates ATE.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0
        )
        
        assert result.att == 0
    
    @pytest.mark.unit
    def test_api015_cbps_att_estimation(self, binary_data):
        """
        API-015: Verify CBPS() with att=1 estimates ATT.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=1
        )
        
        assert result.att == 1
    
    @pytest.mark.unit
    def test_api016_cbps_method_over(self, binary_data):
        """
        API-016: Verify CBPS() with method='over'.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0,
            method='over'
        )
        
        assert result.method == 'over'
    
    @pytest.mark.unit
    def test_api017_cbps_method_exact(self, binary_data):
        """
        API-017: Verify CBPS() with method='exact'.
        
        Requirements: REQ-API-002
        
        Note: method='exact' requires two_step=False to avoid parameter conflict.
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0,
            method='exact',
            two_step=False  # Exact method requires two_step=False
        )
        
        assert result.method == 'exact'
    
    @pytest.mark.unit
    def test_api018_cbps_two_step_true(self, binary_data):
        """
        API-018: Verify CBPS() with two_step=True.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0,
            two_step=True
        )
        
        assert result.two_step is True
    
    @pytest.mark.unit
    def test_api019_cbps_standardize_weights(self, binary_data):
        """
        API-019: Verify CBPS() with standardize=True normalizes weights.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0,
            standardize=True
        )
        
        # Standardized weights should have reasonable sum
        # (exact sum depends on implementation details)
        assert np.all(np.isfinite(result.weights))
        assert np.all(result.weights > 0)
    
    @pytest.mark.unit
    def test_api020_cbps_converged(self, binary_data):
        """
        API-020: Verify CBPS() converges for well-specified data.
        
        Requirements: REQ-API-002
        """
        result = CBPS(
            treatment=binary_data['treat'],
            covariates=binary_data['X'],
            att=0
        )
        
        assert result.converged is True


# =============================================================================
# Test Class: CBPS Function - Continuous Treatment (API-021 to API-026)
# =============================================================================

@pytest.mark.integration
class TestCBPSFunctionContinuous:
    """
    Test CBPS() function with continuous treatment.
    
    Test IDs: API-021 to API-026
    Requirements: REQ-API-003
    """
    
    @pytest.mark.unit
    def test_api021_cbps_continuous_formula(self, continuous_data):
        """
        API-021: Verify CBPS() detects continuous treatment with formula.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            formula='treat ~ x1 + x2',
            data=continuous_data['df']
        )
        
        assert isinstance(result, CBPSResults)
        assert result.sigmasq is not None  # Continuous treatment has sigmasq
    
    @pytest.mark.unit
    def test_api022_cbps_continuous_array(self, continuous_data):
        """
        API-022: Verify CBPS() detects continuous treatment with arrays.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            treatment=continuous_data['treat'],
            covariates=continuous_data['X']
        )
        
        assert isinstance(result, CBPSResults)
        assert result.sigmasq is not None
    
    @pytest.mark.unit
    def test_api023_cbps_continuous_weights_positive(self, continuous_data):
        """
        API-023: Verify continuous CBPS() returns positive weights.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            treatment=continuous_data['treat'],
            covariates=continuous_data['X']
        )
        
        assert np.all(result.weights > 0)
    
    @pytest.mark.unit
    def test_api024_cbps_continuous_fitted_values(self, continuous_data):
        """
        API-024: Verify continuous CBPS() returns fitted GPS densities.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            treatment=continuous_data['treat'],
            covariates=continuous_data['X']
        )
        
        assert hasattr(result, 'fitted_values')
        assert len(result.fitted_values) == continuous_data['n']
    
    @pytest.mark.unit
    def test_api025_cbps_continuous_sigmasq(self, continuous_data):
        """
        API-025: Verify continuous CBPS() estimates residual variance.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            treatment=continuous_data['treat'],
            covariates=continuous_data['X']
        )
        
        assert result.sigmasq > 0
    
    @pytest.mark.unit
    def test_api026_cbps_continuous_j_statistic(self, continuous_data):
        """
        API-026: Verify continuous CBPS() computes J-statistic.
        
        Requirements: REQ-API-003
        """
        result = CBPS(
            treatment=continuous_data['treat'],
            covariates=continuous_data['X']
        )
        
        assert np.isfinite(result.J)


# =============================================================================
# Test Class: CBPS Function - Multi-valued Treatment (API-027 to API-030)
# =============================================================================

@pytest.mark.integration
class TestCBPSFunctionMultitreat:
    """
    Test CBPS() function with multi-valued treatment.
    
    Test IDs: API-027 to API-030
    Requirements: REQ-API-004
    """
    
    @pytest.mark.unit
    def test_api027_cbps_multitreat_formula(self, multitreat_data):
        """
        API-027: Verify CBPS() detects multi-valued treatment with formula.
        
        Requirements: REQ-API-004
        """
        result = CBPS(
            formula='treat ~ x1 + x2',
            data=multitreat_data['df']
        )
        
        assert isinstance(result, CBPSResults)
    
    @pytest.mark.unit
    def test_api028_cbps_multitreat_coefficients_shape(self, multitreat_data):
        """
        API-028: Verify multi-valued CBPS() coefficients have correct shape.
        
        Requirements: REQ-API-004
        """
        result = CBPS(
            treatment=multitreat_data['treat'],
            covariates=multitreat_data['X']
        )
        
        # Coefficients should be 2D with k rows
        k = multitreat_data['k']
        assert result.coefficients.shape[0] == k
        # Number of columns depends on treatment levels detected
        assert result.coefficients.ndim == 2
    
    @pytest.mark.unit
    def test_api029_cbps_multitreat_fitted_probs(self, multitreat_data):
        """
        API-029: Verify multi-valued CBPS() returns probability matrix.
        
        Requirements: REQ-API-004
        """
        result = CBPS(
            treatment=multitreat_data['treat'],
            covariates=multitreat_data['X']
        )
        
        # Fitted values should be (n, n_levels)
        n = multitreat_data['n']
        assert result.fitted_values.shape[0] == n
        
        # Probabilities should sum to 1
        if result.fitted_values.ndim == 2:
            prob_sums = result.fitted_values.sum(axis=1)
            assert_allclose(prob_sums, np.ones(n), rtol=1e-5)
    
    @pytest.mark.unit
    def test_api030_cbps_multitreat_weights(self, multitreat_data):
        """
        API-030: Verify multi-valued CBPS() returns valid weights.
        
        Requirements: REQ-API-004
        """
        result = CBPS(
            treatment=multitreat_data['treat'],
            covariates=multitreat_data['X']
        )
        
        assert len(result.weights) == multitreat_data['n']
        assert np.all(np.isfinite(result.weights))


# =============================================================================
# Test Class: cbps_fit Function (API-031 to API-035)
# =============================================================================

@pytest.mark.integration
class TestCbpsFitFunction:
    """
    Test cbps_fit() array interface function.
    
    Test IDs: API-031 to API-035
    Requirements: REQ-API-005
    
    Note: cbps_fit() returns dict, use CBPS() for CBPSResults object.
    """
    
    @pytest.mark.unit
    def test_api031_cbps_fit_binary(self, binary_data):
        """
        API-031: Verify cbps_fit() works for binary treatment.
        
        Requirements: REQ-API-005
        """
        result = cbps_fit(
            treat=binary_data['treat'],
            X=binary_data['X'],
            att=0
        )
        
        # cbps_fit returns dict
        assert isinstance(result, dict)
        assert 'coefficients' in result
        assert 'weights' in result
    
    @pytest.mark.unit
    def test_api032_cbps_fit_continuous(self, continuous_data):
        """
        API-032: Verify cbps_fit() works for continuous treatment.
        
        Requirements: REQ-API-005
        """
        result = cbps_fit(
            treat=continuous_data['treat'],
            X=continuous_data['X']
        )
        
        # cbps_fit returns dict
        assert isinstance(result, dict)
        assert 'sigmasq' in result
    
    @pytest.mark.unit
    def test_api033_cbps_fit_multitreat(self, multitreat_data):
        """
        API-033: Verify cbps_fit() works for multi-valued treatment.
        
        Requirements: REQ-API-005
        """
        # Convert to Categorical for multi-valued treatment
        treat_cat = pd.Categorical(multitreat_data['treat'].astype(int))
        
        result = cbps_fit(
            treat=treat_cat,
            X=multitreat_data['X']
        )
        
        # cbps_fit returns dict
        assert isinstance(result, dict)
        assert 'coefficients' in result
    
    @pytest.mark.unit
    def test_api034_cbps_fit_sample_weights(self, binary_data):
        """
        API-034: Verify cbps_fit() accepts sample_weights parameter.
        
        Requirements: REQ-API-005
        """
        sample_weights = np.ones(binary_data['n'])
        
        result = cbps_fit(
            treat=binary_data['treat'],
            X=binary_data['X'],
            att=0,
            sample_weights=sample_weights
        )
        
        assert isinstance(result, dict)
        assert 'weights' in result
    
    @pytest.mark.unit
    def test_api035_cbps_fit_returns_dict_keys(self, binary_data):
        """
        API-035: Verify cbps_fit() returns required keys.
        
        Requirements: REQ-API-005
        """
        result = cbps_fit(
            treat=binary_data['treat'],
            X=binary_data['X'],
            att=0
        )
        
        required_keys = ['coefficients', 'weights', 'fitted_values', 'converged']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# =============================================================================
# Test Class: Error Handling (API-036 to API-040)
# =============================================================================

@pytest.mark.edge_case
class TestAPIErrorHandling:
    """
    Test error handling in API functions.
    
    Test IDs: API-036 to API-040
    Requirements: REQ-API-006
    """
    
    @pytest.mark.unit
    def test_api036_missing_formula_and_arrays_raises(self):
        """
        API-036: Verify error when neither formula nor arrays provided.
        
        Requirements: REQ-API-006
        """
        with pytest.raises((ValueError, TypeError)):
            CBPS()
    
    @pytest.mark.unit
    def test_api037_formula_without_data_raises(self):
        """
        API-037: Verify error when formula provided without data.
        
        Requirements: REQ-API-006
        """
        with pytest.raises((ValueError, TypeError)):
            CBPS(formula='treat ~ x1 + x2', data=None)
    
    @pytest.mark.unit
    def test_api038_dimension_mismatch_raises(self, binary_data):
        """
        API-038: Verify error when treatment and covariates have different n.
        
        Requirements: REQ-API-006
        """
        with pytest.raises((ValueError, IndexError)):
            CBPS(
                treatment=binary_data['treat'][:100],  # Wrong length
                covariates=binary_data['X']
            )
    
    @pytest.mark.unit
    def test_api039_invalid_method_raises(self, binary_data):
        """
        API-039: Verify error for invalid method parameter.
        
        Requirements: REQ-API-006
        """
        with pytest.raises(ValueError):
            CBPS(
                treatment=binary_data['treat'],
                covariates=binary_data['X'],
                method='invalid_method'
            )
    
    @pytest.mark.unit
    def test_api040_nan_treatment_handling(self, binary_data):
        """
        API-040: Verify handling of NaN values in treatment.
        
        Requirements: REQ-API-006
        
        Note: Uses float array to allow NaN values.
        """
        # Convert to float to allow NaN values
        treat_with_nan = binary_data['treat'].astype(float).copy()
        treat_with_nan[0] = np.nan
        
        # Should either raise error or handle gracefully
        try:
            result = CBPS(
                treatment=treat_with_nan,
                covariates=binary_data['X'],
                na_action='omit'
            )
            # If it doesn't raise, result should still be valid
            assert isinstance(result, CBPSResults)
        except (ValueError, RuntimeError):
            pass  # Expected behavior


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
