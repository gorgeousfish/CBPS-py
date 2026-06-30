"""
Module: test_cbmsm.py
=====================

Test Suite: CBMSM (Covariate Balancing Propensity Score for Marginal Structural Models)
Test IDs: MSM-001 to MSM-030
Requirements: REQ-MSM-001 to REQ-MSM-012

Overview:
    This module provides comprehensive tests for the CBMSM implementation,
    which estimates inverse probability weights for longitudinal causal
    inference by jointly optimizing treatment prediction and covariate
    balance across all time periods.

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Integration tests: End-to-end workflow with panel data
    - Numerical tests: Weight computation accuracy
    - Edge cases: Boundary conditions and error handling

Usage:
    pytest tests/msm/ -v
    pytest tests/msm/ -m "not slow"
    pytest tests/msm/test_cbmsm.py::TestBasicFunctionality -v
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cbps.msm import CBMSM, cbmsm_fit, CBMSMResults


# =============================================================================
# Test Fixtures (embedded from conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def msm_tolerances():
    """Tolerance values for CBMSM numerical comparisons."""
    return {
        'coefficient_rtol': 0.1,
        'coefficient_atol': 0.1,
        'weight_rtol': 1e-4,
        'weight_atol': 1e-4,
        'j_stat_rtol': 0.1,
    }


@pytest.fixture
def balanced_panel_data():
    """
    Generate balanced panel data for CBMSM tests.

    Returns
    -------
    pd.DataFrame
        Balanced panel with N=50 units, J=3 time periods.
    """
    np.random.seed(42)
    N = 50  # Number of units
    J = 3   # Number of time periods

    # Create balanced panel structure
    id_vec = np.tile(np.arange(N), J)
    time_vec = np.repeat(np.arange(J), N)

    # Generate covariates (time-invariant for simplicity)
    x1_base = np.random.randn(N)
    x2_base = np.random.randn(N)

    x1 = np.tile(x1_base, J)
    x2 = np.tile(x2_base, J)

    # Generate treatment dependent on covariates and time
    treat_probs = 1 / (1 + np.exp(-0.3 * x1 - 0.2 * x2 - 0.1 * time_vec))
    treat = np.random.binomial(1, treat_probs)

    df = pd.DataFrame({
        'id': id_vec,
        'time': time_vec,
        'treat': treat,
        'x1': x1,
        'x2': x2
    })

    return df


@pytest.fixture
def small_panel_data():
    """
    Generate small balanced panel data for quick tests.

    Returns
    -------
    pd.DataFrame
        Small balanced panel with N=20 units, J=2 time periods.
    """
    np.random.seed(123)
    N = 20  # Small number of units
    J = 2   # Two time periods

    id_vec = np.tile(np.arange(N), J)
    time_vec = np.repeat(np.arange(J), N)

    x1 = np.random.randn(N * J)

    treat_probs = 1 / (1 + np.exp(-0.5 * x1))
    treat = np.random.binomial(1, treat_probs)

    df = pd.DataFrame({
        'id': id_vec,
        'time': time_vec,
        'treat': treat,
        'x1': x1
    })

    return df


@pytest.fixture
def blackwell_style_data():
    """
    Generate Blackwell-style panel data for integration tests.

    Mimics the structure of the Blackwell (2013) negative advertising
    data used in the original CBPS R package examples.

    Returns
    -------
    pd.DataFrame
        Panel data similar to Blackwell dataset structure.
    """
    np.random.seed(456)
    N = 100  # Number of units
    J = 4    # Number of time periods

    id_vec = np.tile(np.arange(N), J)
    time_vec = np.repeat(np.arange(1, J + 1), N)

    # Time-invariant baseline covariates
    baseline_x1 = np.random.randn(N)
    baseline_x2 = np.random.binomial(1, 0.5, N)

    x1 = np.tile(baseline_x1, J)
    x2 = np.tile(baseline_x2, J).astype(float)

    # Lagged treatment (for MSM structure)
    treat_lag = np.zeros(N * J)

    # Generate treatment with temporal dependence
    treat = np.zeros(N * J, dtype=int)
    for j in range(J):
        idx = (time_vec == j + 1)
        if j == 0:
            probs = 1 / (1 + np.exp(-0.3 * x1[idx]))
        else:
            # Treatment depends on lagged treatment
            probs = 1 / (1 + np.exp(-0.3 * x1[idx] - 0.2 * treat_lag[idx]))
        treat[idx] = np.random.binomial(1, probs)

        # Update lagged treatment for next period
        if j < J - 1:
            next_idx = (time_vec == j + 2)
            treat_lag[next_idx] = treat[idx]

    df = pd.DataFrame({
        'id': id_vec,
        'time': time_vec,
        'treat': treat,
        'treat_lag': treat_lag,
        'x1': x1,
        'x2': x2
    })

    return df


@pytest.fixture(scope="session")
def cbmsm_available():
    """Check if CBMSM module is available."""
    try:
        from cbps.msm import CBMSM, cbmsm_fit
        return True
    except ImportError:
        return False


# =============================================================================
# Test Class: Basic Functionality (MSM-001 to MSM-010)
# =============================================================================

class TestBasicFunctionality:
    """
    Test basic functionality of CBMSM.
    
    Test IDs: MSM-001 to MSM-010
    Requirements: REQ-MSM-001
    """
    
    @pytest.mark.unit
    def test_msm001_formula_interface(self, balanced_panel_data):
        """
        MSM-001: Verify CBMSM formula interface works correctly.
        
        Requirements: REQ-MSM-001
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert isinstance(result, CBMSMResults), \
            "Result should be CBMSMResults instance"
    
    @pytest.mark.unit
    def test_msm002_matrix_interface(self, balanced_panel_data):
        """
        MSM-002: Verify cbmsm_fit matrix interface works correctly.
        
        Requirements: REQ-MSM-001
        """
        df = balanced_panel_data
        
        # Prepare matrix inputs
        treat = df['treat'].values
        X = np.column_stack([
            np.ones(len(df)),
            df['x1'].values,
            df['x2'].values
        ])
        id_arr = df['id'].values
        time_arr = df['time'].values
        
        result = cbmsm_fit(
            treat=treat,
            X=X,
            id=id_arr,
            time=time_arr
        )
        
        assert isinstance(result, CBMSMResults), \
            "Result should be CBMSMResults instance"
    
    @pytest.mark.unit
    def test_msm003_required_attributes(self, balanced_panel_data):
        """
        MSM-003: Verify all required attributes are present in result.
        
        Requirements: REQ-MSM-002
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        required_attrs = [
            'weights',
            'fitted_values',
            'coefficients',
            'treat_hist',
            'converged',
            'J',
            'n_units',
            'n_periods'
        ]
        
        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing required attribute: {attr}"
    
    @pytest.mark.unit
    def test_msm004_weights_shape(self, balanced_panel_data):
        """
        MSM-004: Verify weights have correct shape (N units).
        
        Requirements: REQ-MSM-003
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        n_units = len(balanced_panel_data['id'].unique())
        
        # Weights should be unit-level (one per unit)
        assert len(result.weights) == n_units, \
            f"Expected {n_units} weights, got {len(result.weights)}"
    
    @pytest.mark.unit
    def test_msm005_weights_positive(self, balanced_panel_data):
        """
        MSM-005: Verify all weights are positive.
        
        Requirements: REQ-MSM-003
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert np.all(result.weights > 0), "All weights should be positive"
    
    @pytest.mark.unit
    def test_msm006_treat_hist_shape(self, balanced_panel_data):
        """
        MSM-006: Verify treatment history matrix has correct shape (N x J).
        
        Requirements: REQ-MSM-004
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        n_units = len(balanced_panel_data['id'].unique())
        n_periods = len(balanced_panel_data['time'].unique())
        
        assert result.treat_hist.shape == (n_units, n_periods), \
            f"Expected shape ({n_units}, {n_periods}), got {result.treat_hist.shape}"
    
    @pytest.mark.unit
    def test_msm007_j_statistic_computed(self, balanced_panel_data):
        """
        MSM-007: Verify J-statistic is computed and finite.
        
        Requirements: REQ-MSM-005
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert np.isfinite(result.J), "J-statistic should be finite"
    
    @pytest.mark.unit
    def test_msm008_n_units_n_periods(self, balanced_panel_data):
        """
        MSM-008: Verify n_units and n_periods are correctly computed.
        
        Requirements: REQ-MSM-006
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        expected_units = len(balanced_panel_data['id'].unique())
        expected_periods = len(balanced_panel_data['time'].unique())
        
        assert result.n_units == expected_units, \
            f"Expected n_units={expected_units}, got {result.n_units}"
        assert result.n_periods == expected_periods, \
            f"Expected n_periods={expected_periods}, got {result.n_periods}"
    
    @pytest.mark.unit
    def test_msm009_summary_method(self, balanced_panel_data):
        """
        MSM-009: Verify summary() method produces output.
        
        Requirements: REQ-MSM-007
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 0, "Summary should not be empty"
    
    @pytest.mark.unit
    def test_msm010_str_repr(self, balanced_panel_data):
        """
        MSM-010: Verify __str__ and __repr__ methods work.
        
        Requirements: REQ-MSM-007
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        str_output = str(result)
        repr_output = repr(result)
        
        assert isinstance(str_output, str)
        assert isinstance(repr_output, str)


# =============================================================================
# Test Class: Parameter Options (MSM-011 to MSM-018)
# =============================================================================

class TestParameterOptions:
    """
    Test different parameter options for CBMSM.
    
    Test IDs: MSM-011 to MSM-018
    Requirements: REQ-MSM-008
    """
    
    @pytest.mark.unit
    def test_msm011_twostep_true(self, small_panel_data):
        """
        MSM-011: Verify twostep=True option works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1",
            id="id",
            time="time",
            data=small_panel_data,
            twostep=True
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_msm012_twostep_false(self, small_panel_data):
        """
        MSM-012: Verify twostep=False (continuous updating) option works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1",
            id="id",
            time="time",
            data=small_panel_data,
            twostep=False
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_msm013_time_vary_false(self, balanced_panel_data):
        """
        MSM-013: Verify time_vary=False (shared coefficients) works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data,
            time_vary=False
        )
        
        assert result.time_vary == False
        # Coefficients should be 1D array
        assert result.coefficients.ndim == 1 or \
               (result.coefficients.ndim == 2 and result.coefficients.shape[1] == 1)
    
    @pytest.mark.unit
    def test_msm014_time_vary_true(self, balanced_panel_data):
        """
        MSM-014: Verify time_vary=True (period-specific coefficients) works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data,
            time_vary=True
        )
        
        assert result.time_vary == True
        # Coefficients should be (k, n_periods) matrix
        assert result.coefficients.ndim == 2
    
    @pytest.mark.unit
    def test_msm015_msm_variance_approx(self, small_panel_data):
        """
        MSM-015: Verify msm_variance='approx' option works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1",
            id="id",
            time="time",
            data=small_panel_data,
            msm_variance='approx'
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_msm016_msm_variance_full(self, small_panel_data):
        """
        MSM-016: Verify msm_variance='full' option works.
        
        Requirements: REQ-MSM-008
        Note: This is slow due to full covariance computation.
        """
        result = CBMSM(
            formula="treat ~ x1",
            id="id",
            time="time",
            data=small_panel_data,
            msm_variance='full'
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_msm017_init_opt(self, balanced_panel_data):
        """
        MSM-017: Verify init='opt' (automatic selection) works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data,
            init='opt'
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_msm018_init_glm(self, balanced_panel_data):
        """
        MSM-018: Verify init='glm' option works.
        
        Requirements: REQ-MSM-008
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data,
            init='glm'
        )
        
        assert result.weights is not None


# =============================================================================
# Test Class: Input Validation (MSM-019 to MSM-024)
# =============================================================================

class TestInputValidation:
    """
    Test input validation for CBMSM.
    
    Test IDs: MSM-019 to MSM-024
    Requirements: REQ-MSM-009
    """
    
    @pytest.mark.unit
    def test_msm019_invalid_type_raises(self, balanced_panel_data):
        """
        MSM-019: Verify invalid type parameter raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        with pytest.raises(ValueError):
            CBMSM(
                formula="treat ~ x1 + x2",
                id="id",
                time="time",
                data=balanced_panel_data,
                type="invalid_type"
            )
    
    @pytest.mark.unit
    def test_msm020_invalid_init_raises(self, balanced_panel_data):
        """
        MSM-020: Verify invalid init parameter raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        with pytest.raises(ValueError):
            CBMSM(
                formula="treat ~ x1 + x2",
                id="id",
                time="time",
                data=balanced_panel_data,
                init="invalid_init"
            )
    
    @pytest.mark.unit
    def test_msm021_unbalanced_panel_raises(self):
        """
        MSM-021: Verify unbalanced panel raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        # Create unbalanced panel (missing observations)
        df = pd.DataFrame({
            'id': [0, 0, 1, 1, 1],  # id=0 missing one period
            'time': [0, 1, 0, 1, 2],
            'treat': [1, 0, 1, 1, 0],
            'x1': np.random.randn(5)
        })
        
        with pytest.raises(ValueError, match="balanced panel"):
            CBMSM(
                formula="treat ~ x1",
                id="id",
                time="time",
                data=df
            )
    
    @pytest.mark.unit
    def test_msm022_single_unit_raises(self):
        """
        MSM-022: Verify single unit raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        df = pd.DataFrame({
            'id': [0, 0],
            'time': [0, 1],
            'treat': [1, 0],
            'x1': np.random.randn(2)
        })
        
        with pytest.raises(ValueError, match="at least 2"):
            CBMSM(
                formula="treat ~ x1",
                id="id",
                time="time",
                data=df
            )
    
    @pytest.mark.unit
    def test_msm023_single_period_raises(self):
        """
        MSM-023: Verify single time period raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        df = pd.DataFrame({
            'id': [0, 1, 2],
            'time': [0, 0, 0],  # Single period
            'treat': [1, 0, 1],
            'x1': np.random.randn(3)
        })
        
        with pytest.raises(ValueError, match="at least 2"):
            CBMSM(
                formula="treat ~ x1",
                id="id",
                time="time",
                data=df
            )
    
    @pytest.mark.unit
    def test_msm024_invalid_msm_variance_raises(self, balanced_panel_data):
        """
        MSM-024: Verify invalid msm_variance parameter raises ValueError.
        
        Requirements: REQ-MSM-009
        """
        with pytest.raises(ValueError):
            CBMSM(
                formula="treat ~ x1 + x2",
                id="id",
                time="time",
                data=balanced_panel_data,
                msm_variance="invalid"
            )


# =============================================================================
# Test Class: Numerical Properties (MSM-025 to MSM-030)
# =============================================================================

class TestNumericalProperties:
    """
    Test numerical properties of CBMSM.
    
    Test IDs: MSM-025 to MSM-030
    Requirements: REQ-MSM-010
    """
    
    @pytest.mark.numerical
    def test_msm025_weights_finite(self, balanced_panel_data):
        """
        MSM-025: Verify all weights are finite (no NaN or Inf).
        
        Requirements: REQ-MSM-010
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert np.all(np.isfinite(result.weights)), \
            "All weights should be finite"
    
    @pytest.mark.numerical
    def test_msm026_coefficients_finite(self, balanced_panel_data):
        """
        MSM-026: Verify all coefficients are finite.
        
        Requirements: REQ-MSM-010
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert np.all(np.isfinite(result.coefficients)), \
            "All coefficients should be finite"
    
    @pytest.mark.numerical
    def test_msm027_fitted_values_positive(self, balanced_panel_data):
        """
        MSM-027: Verify fitted values (stabilized weights) are positive.
        
        Requirements: REQ-MSM-010
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert np.all(result.fitted_values > 0), \
            "All fitted values should be positive"
    
    @pytest.mark.numerical
    def test_msm028_treat_hist_binary(self, balanced_panel_data):
        """
        MSM-028: Verify treatment history contains only 0 and 1.
        
        Requirements: REQ-MSM-010
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        unique_values = np.unique(result.treat_hist)
        assert set(unique_values).issubset({0, 1, 0.0, 1.0}), \
            "Treatment history should contain only 0 and 1"
    
    @pytest.mark.numerical
    def test_msm029_reproducibility(self, balanced_panel_data):
        """
        MSM-029: Verify results are reproducible with same data.
        
        Requirements: REQ-MSM-011
        """
        result1 = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        result2 = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        assert_allclose(
            result1.weights,
            result2.weights,
            rtol=1e-6,
            err_msg="Results should be reproducible"
        )
    
    @pytest.mark.numerical
    def test_msm030_balance_method(self, balanced_panel_data):
        """
        MSM-030: Verify balance() method produces valid output.
        
        Requirements: REQ-MSM-012
        """
        result = CBMSM(
            formula="treat ~ x1 + x2",
            id="id",
            time="time",
            data=balanced_panel_data
        )
        
        balance = result.balance()
        
        assert isinstance(balance, dict), "Balance should return a dictionary"
        assert 'Balanced' in balance, "Balance should contain 'Balanced' key"
        assert 'Unweighted' in balance, "Balance should contain 'Unweighted' key"
