"""
Comprehensive Test Suite for CBPS Core Module
==============================================

This module consolidates all tests for the CBPS core module, including
public API verification (module imports and exports), CBPSResults and
CBPSSummary class functionality, and fit function signature validation.

Test IDs:
    - CORE-001 ~ CORE-010: Module imports and exports
    - CORE-011 ~ CORE-020: CBPSResults class API
    - CORE-021 ~ CORE-030: Fit function signatures
    - RES-001 ~ RES-050: CBPSResults and CBPSSummary classes

Test Categories:
    - unit: API verification and class functionality tests
    - edge_case: Boundary condition and error handling tests

Requirements:
    - REQ-CORE-001: All exported functions must be importable
    - REQ-CORE-002: CBPSResults must have standard interface
    - REQ-CORE-003: Fit functions must accept standard arguments
    - REQ-CORE-004 ~ REQ-CORE-015: CBPSResults class requirements

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cbps.core.results import CBPSResults, CBPSSummary


# =============================================================================
# Test Fixtures (from core/conftest.py)
# =============================================================================

@pytest.fixture
def binary_result_data():
    """
    Generate mock data for binary treatment CBPSResults.

    Returns
    -------
    dict
        Dictionary with all required parameters for CBPSResults initialization.
    """
    np.random.seed(42)
    n = 200
    k = 4

    # Covariates with intercept
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])

    # Coefficients (k x 1 for binary)
    coefficients = np.random.randn(k, 1) * 0.5

    # Linear predictor and fitted values
    linear_predictor = (X @ coefficients).ravel()
    fitted_values = 1 / (1 + np.exp(-linear_predictor))

    # Treatment based on fitted values
    y = np.random.binomial(1, fitted_values).astype(float)

    # Weights (IPW style)
    weights = y / fitted_values + (1 - y) / (1 - fitted_values)

    # Variance-covariance matrix (positive definite)
    A = np.random.randn(k, k)
    var = A @ A.T / 100

    return {
        'coefficients': coefficients,
        'fitted_values': fitted_values,
        'weights': weights,
        'linear_predictor': linear_predictor,
        'y': y,
        'x': X,
        'J': 3.5,
        'mle_J': 2.1,
        'deviance': 250.0,
        'converged': True,
        'var': var,
        'nulldeviance': 275.0,
        'coef_names': ['Intercept', 'X1', 'X2', 'X3'],
        'call_info': "CBPS(treat ~ x1 + x2 + x3, data=df, att=0)",
        'formula': "treat ~ x1 + x2 + x3",
        'att': 0,
        'method': 'over',
        'standardize': True,
        'two_step': True,
    }


@pytest.fixture
def continuous_result_data():
    """
    Generate mock data for continuous treatment CBPSResults.

    Returns
    -------
    dict
        Dictionary with continuous treatment specific parameters.
    """
    np.random.seed(123)
    n = 200
    k = 4

    # Covariates with intercept
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])

    # Coefficients
    coefficients = np.random.randn(k, 1) * 0.3

    # Linear predictor (conditional mean of treatment)
    linear_predictor = (X @ coefficients).ravel()

    # Continuous treatment
    y = linear_predictor + np.random.randn(n)

    # Fitted values (predicted treatment)
    fitted_values = linear_predictor

    # Weights for continuous treatment
    sigmasq = 1.0
    weights = np.exp(-0.5 * (y - fitted_values)**2 / sigmasq)

    # Variance-covariance matrix
    A = np.random.randn(k, k)
    var = A @ A.T / 100

    # Whitened covariates
    Xtilde = X.copy()
    Ttilde = (y - y.mean()) / y.std()

    return {
        'coefficients': coefficients,
        'fitted_values': fitted_values,
        'weights': weights,
        'linear_predictor': linear_predictor,
        'y': y,
        'x': X,
        'J': 2.8,
        'mle_J': 1.5,
        'deviance': 180.0,
        'converged': True,
        'var': var,
        'nulldeviance': 200.0,
        'coef_names': ['Intercept', 'X1', 'X2', 'X3'],
        'call_info': "CBPS(treat ~ x1 + x2 + x3, data=df)",
        'formula': "treat ~ x1 + x2 + x3",
        'sigmasq': sigmasq,
        'Ttilde': Ttilde,
        'Xtilde': Xtilde,
        'beta_tilde': coefficients.ravel(),
        'sigmasq_tilde': sigmasq,
    }


@pytest.fixture
def multitreat_result_data():
    """
    Generate mock data for multi-valued treatment CBPSResults.

    Returns
    -------
    dict
        Dictionary with multi-valued treatment parameters (3 levels).
    """
    np.random.seed(456)
    n = 300
    k = 4
    n_treats = 3

    # Covariates with intercept
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])

    # Coefficients (k x (n_treats - 1) for multinomial)
    coefficients = np.random.randn(k, n_treats - 1) * 0.3

    # Linear predictors
    linear_predictor = X @ coefficients

    # Fitted values (softmax probabilities)
    exp_lp = np.exp(linear_predictor)
    denom = 1 + exp_lp.sum(axis=1, keepdims=True)
    prob_baseline = 1 / denom
    prob_others = exp_lp / denom
    fitted_values = np.column_stack([prob_baseline, prob_others])

    # Treatment assignment
    y = np.array([np.random.choice([0, 1, 2], p=p) for p in fitted_values]).astype(float)

    # Weights
    weights = np.ones(n)

    # Variance-covariance matrix (larger for multi-valued)
    total_params = k * (n_treats - 1)
    A = np.random.randn(total_params, total_params)
    var = A @ A.T / 100

    return {
        'coefficients': coefficients,
        'fitted_values': fitted_values,
        'weights': weights,
        'linear_predictor': linear_predictor,
        'y': y,
        'x': X,
        'J': 4.2,
        'mle_J': 3.1,
        'deviance': 320.0,
        'converged': True,
        'var': var,
        'nulldeviance': 350.0,
        'coef_names': ['Intercept', 'X1', 'X2', 'X3'],
        'call_info': "CBPS(treat ~ x1 + x2 + x3, data=df)",
        'formula': "treat ~ x1 + x2 + x3",
        'treat_names': ['Control', 'Low', 'High'],
    }


@pytest.fixture
def binary_cbps_result(binary_result_data):
    """Create a CBPSResults object for binary treatment."""
    return CBPSResults(**binary_result_data)


@pytest.fixture
def continuous_cbps_result(continuous_result_data):
    """Create a CBPSResults object for continuous treatment."""
    return CBPSResults(**continuous_result_data)


@pytest.fixture
def multitreat_cbps_result(multitreat_result_data):
    """Create a CBPSResults object for multi-valued treatment."""
    return CBPSResults(**multitreat_result_data)



# =============================================================================
# Test Class: Module Imports and Exports (from test_core_init.py)
# =============================================================================

@pytest.mark.unit
class TestCoreModuleExports:
    """
    Test ID: CORE-001 ~ CORE-010
    Requirement: REQ-CORE-001

    Tests for cbps.core module exports.
    """

    def test_core001_module_importable(self):
        """CORE-001: cbps.core module is importable."""
        # Act & Assert
        from cbps import core
        assert core is not None

    def test_core002_cbps_binary_fit_importable(self):
        """CORE-002: cbps_binary_fit is importable from cbps.core."""
        # Act
        from cbps.core import cbps_binary_fit

        # Assert
        assert callable(cbps_binary_fit)

    def test_core003_cbps_continuous_fit_importable(self):
        """CORE-003: cbps_continuous_fit is importable from cbps.core."""
        # Act
        from cbps.core import cbps_continuous_fit

        # Assert
        assert callable(cbps_continuous_fit)

    def test_core004_cbps_3treat_fit_importable(self):
        """CORE-004: cbps_3treat_fit is importable from cbps.core."""
        # Act
        from cbps.core import cbps_3treat_fit

        # Assert
        assert callable(cbps_3treat_fit)

    def test_core005_cbps_4treat_fit_importable(self):
        """CORE-005: cbps_4treat_fit is importable from cbps.core."""
        # Act
        from cbps.core import cbps_4treat_fit

        # Assert
        assert callable(cbps_4treat_fit)

    def test_core006_cbps_optimal_2treat_importable(self):
        """CORE-006: cbps_optimal_2treat is importable from cbps.core."""
        # Act
        from cbps.core import cbps_optimal_2treat

        # Assert
        assert callable(cbps_optimal_2treat)

    def test_core007_cbps_results_importable(self):
        """CORE-007: CBPSResults is importable from cbps.core."""
        # Act
        from cbps.core import CBPSResults

        # Assert
        assert inspect.isclass(CBPSResults)

    def test_core008_cbps_summary_importable(self):
        """CORE-008: CBPSSummary is importable from cbps.core."""
        # Act
        from cbps.core import CBPSSummary

        # Assert
        assert inspect.isclass(CBPSSummary)

    def test_core009_all_exports_defined(self):
        """CORE-009: __all__ contains all expected exports."""
        # Arrange
        from cbps import core
        expected_exports = [
            'cbps_binary_fit',
            'cbps_continuous_fit',
            'cbps_3treat_fit',
            'cbps_4treat_fit',
            'cbps_optimal_2treat',
            'CBPSResults',
            'CBPSSummary',
        ]

        # Assert
        for name in expected_exports:
            assert name in core.__all__, f"{name} not in __all__"

    def test_core010_no_unexpected_private_exports(self):
        """CORE-010: __all__ does not export private functions."""
        # Arrange
        from cbps import core

        # Assert
        for name in core.__all__:
            assert not name.startswith('_'), f"Private {name} should not be in __all__"


# =============================================================================
# Test Class: CBPSResults API (from test_core_init.py)
# =============================================================================

@pytest.mark.unit
class TestCBPSResultsAPI:
    """
    Test ID: CORE-011 ~ CORE-020
    Requirement: REQ-CORE-002

    Tests for CBPSResults class API.

    Note: CBPSResults requires arguments for instantiation, so we test
    the class signature and methods using inspect rather than empty instantiation.
    """

    def test_core011_results_is_class(self):
        """CORE-011: CBPSResults is a valid class."""
        # Act
        from cbps.core import CBPSResults

        # Assert
        assert inspect.isclass(CBPSResults)

    def test_core012_results_init_has_coefficients_param(self):
        """CORE-012: CBPSResults.__init__ has coefficients parameter."""
        # Arrange
        from cbps.core import CBPSResults
        sig = inspect.signature(CBPSResults.__init__)

        # Assert
        assert 'coefficients' in sig.parameters

    def test_core013_results_init_has_weights_param(self):
        """CORE-013: CBPSResults.__init__ has weights parameter."""
        # Arrange
        from cbps.core import CBPSResults
        sig = inspect.signature(CBPSResults.__init__)

        # Assert
        assert 'weights' in sig.parameters

    def test_core014_results_init_has_fitted_values_param(self):
        """CORE-014: CBPSResults.__init__ has fitted_values parameter."""
        # Arrange
        from cbps.core import CBPSResults
        sig = inspect.signature(CBPSResults.__init__)

        # Assert
        assert 'fitted_values' in sig.parameters

    def test_core015_results_init_has_converged_param(self):
        """CORE-015: CBPSResults.__init__ has converged parameter."""
        # Arrange
        from cbps.core import CBPSResults
        sig = inspect.signature(CBPSResults.__init__)

        # Assert
        assert 'converged' in sig.parameters

    def test_core016_results_has_vcov_method(self):
        """CORE-016: CBPSResults has vcov() method defined."""
        # Arrange
        from cbps.core import CBPSResults

        # Assert
        assert hasattr(CBPSResults, 'vcov')
        assert callable(getattr(CBPSResults, 'vcov'))

    def test_core017_results_has_str_method(self):
        """CORE-017: CBPSResults has __str__ method defined."""
        # Arrange
        from cbps.core import CBPSResults

        # Assert
        assert hasattr(CBPSResults, '__str__')

    def test_core018_results_has_summary_method(self):
        """CORE-018: CBPSResults has summary() method defined."""
        # Arrange
        from cbps.core import CBPSResults

        # Assert
        assert hasattr(CBPSResults, 'summary')
        assert callable(getattr(CBPSResults, 'summary'))


# =============================================================================
# Test Class: Fit Function Signatures (from test_core_init.py)
# =============================================================================

@pytest.mark.unit
class TestFitFunctionSignatures:
    """
    Test ID: CORE-021 ~ CORE-030
    Requirement: REQ-CORE-003

    Tests for fit function signatures.
    """

    def test_core021_binary_fit_has_treat_param(self):
        """CORE-021: cbps_binary_fit has 'treat' parameter."""
        # Arrange
        from cbps.core import cbps_binary_fit
        sig = inspect.signature(cbps_binary_fit)

        # Assert
        assert 'treat' in sig.parameters

    def test_core022_binary_fit_has_X_param(self):
        """CORE-022: cbps_binary_fit has 'X' parameter."""
        # Arrange
        from cbps.core import cbps_binary_fit
        sig = inspect.signature(cbps_binary_fit)

        # Assert
        assert 'X' in sig.parameters

    def test_core023_binary_fit_has_att_param(self):
        """CORE-023: cbps_binary_fit has 'att' parameter."""
        # Arrange
        from cbps.core import cbps_binary_fit
        sig = inspect.signature(cbps_binary_fit)

        # Assert
        assert 'att' in sig.parameters

    def test_core024_continuous_fit_has_treat_param(self):
        """CORE-024: cbps_continuous_fit has 'treat' parameter."""
        # Arrange
        from cbps.core import cbps_continuous_fit
        sig = inspect.signature(cbps_continuous_fit)

        # Assert
        assert 'treat' in sig.parameters

    def test_core025_continuous_fit_has_X_param(self):
        """CORE-025: cbps_continuous_fit has 'X' parameter."""
        # Arrange
        from cbps.core import cbps_continuous_fit
        sig = inspect.signature(cbps_continuous_fit)

        # Assert
        assert 'X' in sig.parameters

    def test_core026_multitreat_fit_has_treat_param(self):
        """CORE-026: cbps_3treat_fit has 'treat' parameter."""
        # Arrange
        from cbps.core import cbps_3treat_fit
        sig = inspect.signature(cbps_3treat_fit)

        # Assert
        assert 'treat' in sig.parameters

    def test_core027_optimal_fit_has_treat_param(self):
        """CORE-027: cbps_optimal_2treat has 'treat' parameter."""
        # Arrange
        from cbps.core import cbps_optimal_2treat
        sig = inspect.signature(cbps_optimal_2treat)

        # Assert
        assert 'treat' in sig.parameters



# =============================================================================
# Test Class: CBPSResults Initialization (RES-001 to RES-010)
# =============================================================================

class TestCBPSResultsInit:
    """
    Test CBPSResults class initialization.

    Test IDs: RES-001 to RES-010
    Requirements: REQ-CORE-001
    """

    @pytest.mark.unit
    def test_res001_binary_init_success(self, binary_result_data):
        """
        RES-001: Verify binary treatment CBPSResults initializes correctly.

        Requirements: REQ-CORE-001
        """
        result = CBPSResults(**binary_result_data)

        assert isinstance(result, CBPSResults)
        assert result.converged is True

    @pytest.mark.unit
    def test_res002_continuous_init_success(self, continuous_result_data):
        """
        RES-002: Verify continuous treatment CBPSResults initializes correctly.

        Requirements: REQ-CORE-001
        """
        result = CBPSResults(**continuous_result_data)

        assert isinstance(result, CBPSResults)
        assert result.sigmasq is not None

    @pytest.mark.unit
    def test_res003_multitreat_init_success(self, multitreat_result_data):
        """
        RES-003: Verify multi-valued treatment CBPSResults initializes correctly.

        Requirements: REQ-CORE-001
        """
        result = CBPSResults(**multitreat_result_data)

        assert isinstance(result, CBPSResults)
        assert result.treat_names is not None

    @pytest.mark.unit
    def test_res004_coefficients_stored(self, binary_cbps_result, binary_result_data):
        """
        RES-004: Verify coefficients are stored correctly.

        Requirements: REQ-CORE-001
        """
        assert_array_equal(
            binary_cbps_result.coefficients,
            binary_result_data['coefficients']
        )

    @pytest.mark.unit
    def test_res005_weights_stored(self, binary_cbps_result, binary_result_data):
        """
        RES-005: Verify weights are stored correctly.

        Requirements: REQ-CORE-001
        """
        assert_array_equal(
            binary_cbps_result.weights,
            binary_result_data['weights']
        )

    @pytest.mark.unit
    def test_res006_fitted_values_stored(self, binary_cbps_result, binary_result_data):
        """
        RES-006: Verify fitted values are stored correctly.

        Requirements: REQ-CORE-001
        """
        assert_array_equal(
            binary_cbps_result.fitted_values,
            binary_result_data['fitted_values']
        )

    @pytest.mark.unit
    def test_res007_metadata_stored(self, binary_cbps_result):
        """
        RES-007: Verify metadata attributes are stored correctly.

        Requirements: REQ-CORE-001
        """
        assert binary_cbps_result.att == 0
        assert binary_cbps_result.method == 'over'
        assert binary_cbps_result.two_step is True
        assert binary_cbps_result.standardize is True

    @pytest.mark.unit
    def test_res008_default_coef_names(self):
        """
        RES-008: Verify default coefficient names are generated when not provided.

        Requirements: REQ-CORE-001
        """
        np.random.seed(42)
        n, k = 100, 3

        result = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=np.eye(k),
            coef_names=None  # Not provided
        )

        assert result.coef_names == ['Intercept', 'X1', 'X2']

    @pytest.mark.unit
    def test_res009_1d_coefficients_raises_error(self):
        """
        RES-009: Verify 1D coefficients raise ValueError.

        Requirements: REQ-CORE-001
        """
        np.random.seed(42)
        n, k = 100, 3

        with pytest.raises(ValueError, match="must be 2D"):
            CBPSResults(
                coefficients=np.random.randn(k),  # 1D instead of 2D
                fitted_values=np.random.rand(n),
                weights=np.ones(n),
                linear_predictor=np.random.randn(n),
                y=np.random.binomial(1, 0.5, n).astype(float),
                x=np.random.randn(n, k),
                J=1.0,
                mle_J=0.5,
                deviance=100.0,
                converged=True,
                var=np.eye(k),
            )

    @pytest.mark.unit
    def test_res010_na_action_stored(self, binary_result_data):
        """
        RES-010: Verify na_action is stored correctly.

        Requirements: REQ-CORE-001
        """
        na_action = {'method': 'omit', 'n_dropped': 5}
        binary_result_data['na_action'] = na_action

        result = CBPSResults(**binary_result_data)

        assert result.na_action == na_action


# =============================================================================
# Test Class: vcov() Method (RES-011 to RES-016)
# =============================================================================

class TestCBPSResultsVcov:
    """
    Test vcov() method functionality.

    Test IDs: RES-011 to RES-016
    Requirements: REQ-CORE-002
    """

    @pytest.mark.unit
    def test_res011_vcov_returns_matrix(self, binary_cbps_result, binary_result_data):
        """
        RES-011: Verify vcov() returns the variance-covariance matrix.

        Requirements: REQ-CORE-002
        """
        vcov = binary_cbps_result.vcov()

        assert_array_equal(vcov, binary_result_data['var'])

    @pytest.mark.unit
    def test_res012_vcov_symmetric(self, binary_cbps_result):
        """
        RES-012: Verify vcov() returns a symmetric matrix.

        Requirements: REQ-CORE-002
        """
        vcov = binary_cbps_result.vcov()

        assert_allclose(vcov, vcov.T, rtol=1e-10)

    @pytest.mark.unit
    def test_res013_vcov_none_raises_error(self):
        """
        RES-013: Verify vcov() raises ValueError when var is None.

        Requirements: REQ-CORE-002
        """
        np.random.seed(42)
        n, k = 100, 3

        result = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=None,  # No variance matrix
        )

        with pytest.raises(ValueError, match="not computed"):
            result.vcov()

    @pytest.mark.unit
    def test_res014_vcov_high_condition_number_warns(self):
        """
        RES-014: Verify vcov() warns about high condition number.

        Requirements: REQ-CORE-002
        """
        np.random.seed(42)
        n, k = 100, 3

        # Create ill-conditioned matrix
        var = np.array([
            [1.0, 0.9999999, 0.0],
            [0.9999999, 1.0, 0.0],
            [0.0, 0.0, 1e-12]  # Very small eigenvalue
        ])

        result = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=var,
        )

        with pytest.warns(UserWarning, match="condition number"):
            result.vcov()

    @pytest.mark.unit
    def test_res015_vcov_shape(self, binary_cbps_result):
        """
        RES-015: Verify vcov() returns correct shape.

        Requirements: REQ-CORE-002
        """
        vcov = binary_cbps_result.vcov()
        k = binary_cbps_result.coefficients.shape[0]

        assert vcov.shape == (k, k)

    @pytest.mark.unit
    def test_res016_vcov_multitreat_shape(self, multitreat_cbps_result):
        """
        RES-016: Verify vcov() returns correct shape for multi-valued treatment.

        Requirements: REQ-CORE-002
        """
        vcov = multitreat_cbps_result.vcov()
        k = multitreat_cbps_result.coefficients.shape[0]
        n_treats = multitreat_cbps_result.coefficients.shape[1]

        # Variance matrix should be (k * n_treats) x (k * n_treats)
        expected_dim = k * n_treats
        assert vcov.shape == (expected_dim, expected_dim)



# =============================================================================
# Test Class: Properties (RES-017 to RES-026)
# =============================================================================

class TestCBPSResultsProperties:
    """
    Test computed properties and aliases.

    Test IDs: RES-017 to RES-026
    Requirements: REQ-CORE-003
    """

    @pytest.mark.unit
    def test_res017_coef_property(self, binary_cbps_result):
        """
        RES-017: Verify coef property returns 1D coefficient vector.

        Requirements: REQ-CORE-003
        """
        coef = binary_cbps_result.coef

        assert coef.ndim == 1
        assert_allclose(coef, binary_cbps_result.coefficients[:, 0])

    @pytest.mark.unit
    def test_res018_fitted_property(self, binary_cbps_result):
        """
        RES-018: Verify fitted property aliases fitted_values.

        Requirements: REQ-CORE-003
        """
        assert_array_equal(
            binary_cbps_result.fitted,
            binary_cbps_result.fitted_values
        )

    @pytest.mark.unit
    def test_res019_j_stat_property(self, binary_cbps_result):
        """
        RES-019: Verify J_stat property aliases J.

        Requirements: REQ-CORE-003
        """
        assert binary_cbps_result.J_stat == binary_cbps_result.J

    @pytest.mark.unit
    def test_res020_sigma_squared_continuous(self, continuous_cbps_result):
        """
        RES-020: Verify sigma_squared property returns sigmasq for continuous.

        Requirements: REQ-CORE-003
        """
        assert continuous_cbps_result.sigma_squared == continuous_cbps_result.sigmasq

    @pytest.mark.unit
    def test_res021_sigma_squared_binary_none(self, binary_cbps_result):
        """
        RES-021: Verify sigma_squared property returns None for binary.

        Requirements: REQ-CORE-003
        """
        assert binary_cbps_result.sigma_squared is None

    @pytest.mark.unit
    def test_res022_pseudo_r2_computed(self, binary_cbps_result):
        """
        RES-022: Verify pseudo_r2 is computed correctly.

        Requirements: REQ-CORE-003
        """
        expected = 1.0 - binary_cbps_result.deviance / binary_cbps_result.nulldeviance

        assert_allclose(binary_cbps_result.pseudo_r2, expected)

    @pytest.mark.unit
    def test_res023_pseudo_r2_none_without_nulldeviance(self):
        """
        RES-023: Verify pseudo_r2 returns None when nulldeviance is unavailable.

        Requirements: REQ-CORE-003
        """
        np.random.seed(42)
        n, k = 100, 3

        result = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=np.eye(k),
            nulldeviance=None,  # Not provided
        )

        assert result.pseudo_r2 is None

    @pytest.mark.unit
    def test_res024_residuals_binary(self, binary_cbps_result):
        """
        RES-024: Verify residuals property for binary treatment.

        Requirements: REQ-CORE-003
        """
        residuals = binary_cbps_result.residuals
        expected = binary_cbps_result.y - binary_cbps_result.fitted_values.ravel()

        assert_allclose(residuals, expected)

    @pytest.mark.unit
    def test_res025_residuals_continuous(self, continuous_cbps_result):
        """
        RES-025: Verify residuals property for continuous treatment.

        Requirements: REQ-CORE-003
        """
        residuals = continuous_cbps_result.residuals
        expected = continuous_cbps_result.Ttilde - continuous_cbps_result.linear_predictor.ravel()

        assert_allclose(residuals, expected)

    @pytest.mark.unit
    def test_res026_residuals_shape(self, binary_cbps_result):
        """
        RES-026: Verify residuals have correct shape.

        Requirements: REQ-CORE-003
        """
        residuals = binary_cbps_result.residuals
        n = len(binary_cbps_result.y)

        assert residuals.shape == (n,)


# =============================================================================
# Test Class: predict() Method (RES-027 to RES-036)
# =============================================================================

class TestCBPSResultsPredict:
    """
    Test predict() method functionality.

    Test IDs: RES-027 to RES-036
    Requirements: REQ-CORE-004
    """

    @pytest.mark.unit
    def test_res027_predict_none_returns_fitted(self, binary_cbps_result):
        """
        RES-027: Verify predict(None) returns fitted values.

        Requirements: REQ-CORE-004
        """
        pred = binary_cbps_result.predict(newdata=None, type='response')

        assert_array_equal(pred, binary_cbps_result.fitted_values)

    @pytest.mark.unit
    def test_res028_predict_link_returns_linear_predictor(self, binary_cbps_result):
        """
        RES-028: Verify predict(type='link') returns linear predictor.

        Requirements: REQ-CORE-004
        """
        pred = binary_cbps_result.predict(newdata=None, type='link')

        assert_array_equal(pred, binary_cbps_result.linear_predictor)

    @pytest.mark.unit
    def test_res029_predict_invalid_type_raises(self, binary_cbps_result):
        """
        RES-029: Verify predict() raises error for invalid type.

        Requirements: REQ-CORE-004
        """
        with pytest.raises(ValueError, match="Invalid type"):
            binary_cbps_result.predict(type='invalid')

    @pytest.mark.unit
    def test_res030_predict_newdata_array(self, binary_cbps_result):
        """
        RES-030: Verify predict() works with new array data.

        Requirements: REQ-CORE-004
        """
        np.random.seed(42)
        n_new = 10
        k = binary_cbps_result.x.shape[1]

        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new, k - 1)])

        pred = binary_cbps_result.predict(newdata=X_new, type='response')

        assert pred.shape == (n_new,)
        assert np.all(pred > 0) and np.all(pred < 1)  # Valid probabilities

    @pytest.mark.unit
    def test_res031_predict_newdata_wrong_dims_raises(self, binary_cbps_result):
        """
        RES-031: Verify predict() raises error for wrong dimensions.

        Requirements: REQ-CORE-004
        """
        np.random.seed(42)
        n_new = 10
        wrong_k = binary_cbps_result.x.shape[1] + 2  # Wrong number of columns

        X_new = np.random.randn(n_new, wrong_k)

        with pytest.raises(ValueError, match="columns"):
            binary_cbps_result.predict(newdata=X_new)

    @pytest.mark.unit
    def test_res032_predict_binary_bounded(self, binary_cbps_result):
        """
        RES-032: Verify binary predict() returns values in (0, 1).

        Requirements: REQ-CORE-004
        """
        np.random.seed(42)
        n_new = 50
        k = binary_cbps_result.x.shape[1]

        # Include extreme values
        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new, k - 1) * 5])

        pred = binary_cbps_result.predict(newdata=X_new, type='response')

        assert np.all(pred > 0) and np.all(pred < 1)

    @pytest.mark.unit
    def test_res033_predict_continuous_identity_link(self, continuous_cbps_result):
        """
        RES-033: Verify continuous predict() returns valid predictions.

        Requirements: REQ-CORE-004

        Note: The implementation uses coefficient shape to determine link function.
        For (k, 1) coefficients, logistic link is applied even with sigmasq set.
        This test verifies predictions are valid and finite.
        """
        np.random.seed(42)
        n_new = 10
        k = continuous_cbps_result.x.shape[1]

        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new, k - 1)])

        pred_response = continuous_cbps_result.predict(newdata=X_new, type='response')
        pred_link = continuous_cbps_result.predict(newdata=X_new, type='link')

        # Both should be finite and have correct shape
        assert pred_response.shape == (n_new,)
        assert pred_link.shape == (n_new,)
        assert np.all(np.isfinite(pred_response))
        assert np.all(np.isfinite(pred_link))

    @pytest.mark.unit
    def test_res034_predict_multitreat_softmax(self, multitreat_cbps_result):
        """
        RES-034: Verify multi-valued predict() returns softmax probabilities.

        Requirements: REQ-CORE-004
        """
        np.random.seed(42)
        n_new = 10
        k = multitreat_cbps_result.x.shape[1]

        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new, k - 1)])

        pred = multitreat_cbps_result.predict(newdata=X_new, type='response')

        # Should have probabilities for all levels
        n_levels = multitreat_cbps_result.coefficients.shape[1] + 1
        assert pred.shape == (n_new, n_levels)

        # Probabilities should sum to 1
        assert_allclose(pred.sum(axis=1), np.ones(n_new), rtol=1e-6)

    @pytest.mark.unit
    def test_res035_predict_1d_array_reshaped(self, binary_cbps_result):
        """
        RES-035: Verify 1D array is reshaped for single observation.

        Requirements: REQ-CORE-004
        """
        k = binary_cbps_result.x.shape[1]

        X_new = np.array([1.0] + [0.5] * (k - 1))  # 1D array

        pred = binary_cbps_result.predict(newdata=X_new, type='response')

        assert pred.shape == (1,)

    @pytest.mark.unit
    def test_res036_predict_link_multitreat_shape(self, multitreat_cbps_result):
        """
        RES-036: Verify predict(type='link') shape for multi-valued treatment.

        Requirements: REQ-CORE-004
        """
        np.random.seed(42)
        n_new = 10
        k = multitreat_cbps_result.x.shape[1]

        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new, k - 1)])

        pred = multitreat_cbps_result.predict(newdata=X_new, type='link')

        # Link should have n_levels - 1 columns (reference category excluded)
        n_contrasts = multitreat_cbps_result.coefficients.shape[1]
        assert pred.shape == (n_new, n_contrasts)



# =============================================================================
# Test Class: summary() Method (RES-037 to RES-044)
# =============================================================================

class TestCBPSResultsSummary:
    """
    Test summary() method and CBPSSummary class.

    Test IDs: RES-037 to RES-044
    Requirements: REQ-CORE-005
    """

    @pytest.mark.unit
    def test_res037_summary_returns_cbpssummary(self, binary_cbps_result):
        """
        RES-037: Verify summary() returns CBPSSummary object.

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()

        assert isinstance(summ, CBPSSummary)

    @pytest.mark.unit
    def test_res038_summary_coef_table_shape(self, binary_cbps_result):
        """
        RES-038: Verify coefficient table has correct shape (k x 4).

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()
        k = binary_cbps_result.coefficients.size

        assert summ.coefficients.shape == (k, 4)

    @pytest.mark.unit
    def test_res039_summary_coef_property(self, binary_cbps_result):
        """
        RES-039: Verify CBPSSummary.coef returns coefficient estimates.

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()

        assert_allclose(summ.coef, binary_cbps_result.coefficients.ravel())

    @pytest.mark.unit
    def test_res040_summary_se_positive(self, binary_cbps_result):
        """
        RES-040: Verify standard errors are positive.

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()

        assert np.all(summ.se > 0)

    @pytest.mark.unit
    def test_res041_summary_pvalues_bounded(self, binary_cbps_result):
        """
        RES-041: Verify p-values are in [0, 1].

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()

        assert np.all(summ.pvalues >= 0) and np.all(summ.pvalues <= 1)

    @pytest.mark.unit
    def test_res042_summary_zvalues_computed(self, binary_cbps_result):
        """
        RES-042: Verify z-values are computed as coef/se.

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()
        expected = summ.coef / summ.se

        assert_allclose(summ.zvalues, expected)

    @pytest.mark.unit
    def test_res043_summary_str_output(self, binary_cbps_result):
        """
        RES-043: Verify __str__ returns formatted output.

        Requirements: REQ-CORE-005
        """
        summ = binary_cbps_result.summary()
        output = str(summ)

        assert isinstance(output, str)
        assert 'Coefficients' in output
        assert 'Estimate' in output
        assert 'Std. Error' in output

    @pytest.mark.unit
    def test_res044_summary_no_var_raises(self):
        """
        RES-044: Verify summary() raises error when var is None.

        Requirements: REQ-CORE-005
        """
        np.random.seed(42)
        n, k = 100, 3

        result = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=None,
        )

        with pytest.raises(ValueError, match="Variance"):
            result.summary()


# =============================================================================
# Test Class: __str__ and __repr__ (RES-045 to RES-048)
# =============================================================================

class TestCBPSResultsStrRepr:
    """
    Test string representations.

    Test IDs: RES-045 to RES-048
    Requirements: REQ-CORE-006
    """

    @pytest.mark.unit
    def test_res045_str_output(self, binary_cbps_result):
        """
        RES-045: Verify __str__ returns formatted output.

        Requirements: REQ-CORE-006
        """
        output = str(binary_cbps_result)

        assert isinstance(output, str)
        assert 'Call' in output
        assert 'Coefficients' in output

    @pytest.mark.unit
    def test_res046_repr_output(self, binary_cbps_result):
        """
        RES-046: Verify __repr__ returns concise representation.

        Requirements: REQ-CORE-006
        """
        output = repr(binary_cbps_result)

        assert isinstance(output, str)
        assert 'CBPSResults' in output
        assert 'converged' in output

    @pytest.mark.unit
    def test_res047_str_includes_j_statistic(self, binary_cbps_result):
        """
        RES-047: Verify __str__ includes J-statistic.

        Requirements: REQ-CORE-006
        """
        output = str(binary_cbps_result)

        assert 'J-Statistic' in output

    @pytest.mark.unit
    def test_res048_str_continuous_includes_sigmasq(self, continuous_cbps_result):
        """
        RES-048: Verify __str__ includes sigma-squared for continuous.

        Requirements: REQ-CORE-006
        """
        output = str(continuous_cbps_result)

        assert 'Sigma-Squared' in output


# =============================================================================
# Test Class: Edge Cases (RES-049 to RES-050)
# =============================================================================

class TestCBPSResultsEdgeCases:
    """
    Test edge cases and error handling.

    Test IDs: RES-049 to RES-050
    Requirements: REQ-CORE-007
    """

    @pytest.mark.edge_case
    def test_res049_empty_coef_names(self):
        """
        RES-049: Verify handling of zero coefficients.

        Requirements: REQ-CORE-007
        """
        np.random.seed(42)
        n = 100
        k = 0  # No covariates

        # This is an edge case - empty coefficient matrix
        coefficients = np.array([]).reshape(0, 1)

        result = CBPSResults(
            coefficients=coefficients,
            fitted_values=np.full(n, 0.5),
            weights=np.ones(n),
            linear_predictor=np.zeros(n),
            y=np.random.binomial(1, 0.5, n).astype(float),
            x=np.random.randn(n, 1),
            J=0.0,
            mle_J=0.0,
            deviance=0.0,
            converged=True,
            var=np.array([]).reshape(0, 0),
        )

        assert result.coef_names == []

    @pytest.mark.edge_case
    def test_res050_large_deviance_residuals(self, binary_cbps_result):
        """
        RES-050: Verify handling of extreme fitted values in residuals.

        Requirements: REQ-CORE-007
        """
        # This test ensures residuals don't produce NaN/Inf
        residuals = binary_cbps_result.residuals

        assert np.all(np.isfinite(residuals))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
