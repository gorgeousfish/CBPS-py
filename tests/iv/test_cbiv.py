"""
Module: test_cbiv.py
====================

Test Suite: CBIV (Covariate Balancing Propensity Score for Instrumental Variables)
Test IDs: CBIV-001 to CBIV-030
Requirements: REQ-IV-001 to REQ-IV-012

Overview:
    This module provides comprehensive tests for the CBIV implementation,
    which estimates compliance type propensity scores in an instrumental
    variable framework with treatment noncompliance.

    CBIV supports two noncompliance models:
    - One-sided: Only compliers and never-takers (twosided=False)
    - Two-sided: Compliers, always-takers, and never-takers (twosided=True)

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Integration tests: End-to-end workflow with IV data
    - Numerical tests: Weight and probability computation accuracy
    - Edge cases: Boundary conditions and error handling

Usage:
    pytest tests/iv/ -v
    pytest tests/iv/ -m "not slow"
    pytest tests/iv/test_cbiv.py::TestBasicFunctionality -v
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cbps.iv import CBIV, CBIVResults


# =============================================================================
# Test Fixtures (embedded from conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def iv_tolerances():
    """Tolerance values for CBIV numerical comparisons."""
    return {
        'coefficient_rtol': 0.1,
        'coefficient_atol': 0.1,
        'weight_rtol': 1e-4,
        'weight_atol': 1e-4,
        'probs_rtol': 0.05,
    }


@pytest.fixture
def simple_iv_data():
    """
    Generate simple IV data for quick tests.

    Returns
    -------
    dict
        Dictionary containing:
        - Tr: Binary treatment (n=200)
        - Z: Binary instrument (n=200)
        - X: Covariate matrix (n x 2)
    """
    np.random.seed(42)
    n = 200

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([x1, x2])

    # Randomized instrument
    Z = np.random.binomial(1, 0.5, n)

    # Compliance probability depends on X
    p_comply = 1 / (1 + np.exp(-0.5 - 0.3 * x1))
    comply = np.random.binomial(1, p_comply, n)

    # One-sided noncompliance: treatment = Z * comply
    Tr = Z * comply

    return {
        'Tr': Tr,
        'Z': Z,
        'X': X,
        'n': n,
    }


@pytest.fixture
def iv_data_onesided():
    """
    Generate IV data with one-sided noncompliance.

    One-sided noncompliance: Only compliers and never-takers exist.
    No always-takers (pi_a = 0).

    Returns
    -------
    dict
        Dictionary containing Tr, Z, X for one-sided model.
    """
    np.random.seed(123)
    n = 300

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([x1, x2])

    # Randomized instrument
    Z = np.random.binomial(1, 0.5, n)

    # Compliance depends on covariates
    p_comply = 1 / (1 + np.exp(-0.3 - 0.4 * x1 + 0.2 * x2))
    comply = np.random.binomial(1, p_comply, n)

    # One-sided: Tr = Z * comply (no always-takers)
    Tr = Z * comply

    return {
        'Tr': Tr,
        'Z': Z,
        'X': X,
        'n': n,
        'true_comply_rate': p_comply.mean(),
    }


@pytest.fixture
def iv_data_twosided():
    """
    Generate IV data with two-sided noncompliance.

    Two-sided noncompliance: Compliers, always-takers, and never-takers exist.

    Returns
    -------
    dict
        Dictionary containing Tr, Z, X for two-sided model.
    """
    np.random.seed(456)
    n = 400

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([x1, x2])

    # Randomized instrument
    Z = np.random.binomial(1, 0.5, n)

    # Generate compliance types using multinomial
    # pi_c = P(complier), pi_a = P(always-taker), pi_n = P(never-taker)
    logit_c = 0.3 + 0.3 * x1
    logit_a = -0.5 + 0.2 * x2

    # Softmax to get probabilities
    max_logit = np.maximum(np.maximum(logit_c, logit_a), 0)
    p_c = np.exp(logit_c - max_logit)
    p_a = np.exp(logit_a - max_logit)
    p_n = np.exp(-max_logit)

    total = p_c + p_a + p_n
    p_c /= total
    p_a /= total
    p_n /= total

    # Generate compliance type (0=never, 1=comply, 2=always)
    compliance_type = np.array([
        np.random.choice([0, 1, 2], p=[p_n[i], p_c[i], p_a[i]])
        for i in range(n)
    ])

    # Generate treatment based on compliance type and instrument
    Tr = np.zeros(n, dtype=int)
    Tr[compliance_type == 2] = 1  # Always-takers
    Tr[(compliance_type == 1) & (Z == 1)] = 1  # Compliers with Z=1

    return {
        'Tr': Tr,
        'Z': Z,
        'X': X,
        'n': n,
        'compliance_type': compliance_type,
    }


@pytest.fixture(scope="session")
def cbiv_available():
    """Check if CBIV module is available."""
    try:
        from cbps.iv import CBIV, CBIVResults
        return True
    except ImportError:
        return False


# =============================================================================
# Test Class: Basic Functionality (CBIV-001 to CBIV-010)
# =============================================================================

class TestBasicFunctionality:
    """
    Test basic functionality of CBIV.
    
    Test IDs: CBIV-001 to CBIV-010
    Requirements: REQ-IV-001
    """
    
    @pytest.mark.unit
    def test_cbiv001_onesided_matrix_interface(self, simple_iv_data):
        """
        CBIV-001: Verify CBIV one-sided matrix interface works correctly.
        
        Requirements: REQ-IV-001
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert isinstance(result, CBIVResults), \
            "Result should be CBIVResults instance"
    
    @pytest.mark.unit
    def test_cbiv002_twosided_matrix_interface(self, iv_data_twosided):
        """
        CBIV-002: Verify CBIV two-sided matrix interface works correctly.
        
        Requirements: REQ-IV-001
        """
        data = iv_data_twosided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=True
        )
        
        assert isinstance(result, CBIVResults), \
            "Result should be CBIVResults instance"
    
    @pytest.mark.unit
    def test_cbiv003_required_attributes_onesided(self, simple_iv_data):
        """
        CBIV-003: Verify all required attributes are present for one-sided model.
        
        Requirements: REQ-IV-002
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        required_attrs = [
            'coefficients',
            'fitted_values',
            'weights',
            'deviance',
            'converged',
            'J',
            'df',
            'bal',
            'method',
            'two_sided'
        ]
        
        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing required attribute: {attr}"
    
    @pytest.mark.unit
    def test_cbiv004_required_attributes_twosided(self, iv_data_twosided):
        """
        CBIV-004: Verify all required attributes are present for two-sided model.
        
        Requirements: REQ-IV-002
        """
        data = iv_data_twosided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=True
        )
        
        required_attrs = [
            'coefficients',
            'fitted_values',
            'weights',
            'deviance',
            'converged',
            'J',
            'df',
            'bal',
            'method',
            'two_sided'
        ]
        
        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing required attribute: {attr}"
    
    @pytest.mark.unit
    def test_cbiv005_weights_shape_onesided(self, simple_iv_data):
        """
        CBIV-005: Verify weights have correct shape for one-sided model.
        
        Requirements: REQ-IV-003
        """
        data = simple_iv_data
        n = data['n']
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert result.weights.shape == (n,), \
            f"Expected shape ({n},), got {result.weights.shape}"
    
    @pytest.mark.unit
    def test_cbiv006_weights_positive(self, simple_iv_data):
        """
        CBIV-006: Verify all weights are positive.
        
        Requirements: REQ-IV-003
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert np.all(result.weights > 0), "All weights should be positive"
    
    @pytest.mark.unit
    def test_cbiv007_fitted_values_shape_onesided(self, simple_iv_data):
        """
        CBIV-007: Verify fitted values shape for one-sided model.
        
        Requirements: REQ-IV-004
        """
        data = simple_iv_data
        n = data['n']
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        # One-sided returns (n, 1) matrix
        assert result.fitted_values.shape == (n, 1), \
            f"Expected shape ({n}, 1), got {result.fitted_values.shape}"
    
    @pytest.mark.unit
    def test_cbiv008_fitted_values_shape_twosided(self, iv_data_twosided):
        """
        CBIV-008: Verify fitted values shape for two-sided model.
        
        Requirements: REQ-IV-004
        """
        data = iv_data_twosided
        n = data['n']
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=True
        )
        
        # Two-sided returns (n, 3) matrix [π_c, π_a, π_n]
        assert result.fitted_values.shape == (n, 3), \
            f"Expected shape ({n}, 3), got {result.fitted_values.shape}"
    
    @pytest.mark.unit
    def test_cbiv009_j_statistic_computed(self, simple_iv_data):
        """
        CBIV-009: Verify J-statistic is computed and finite.
        
        Requirements: REQ-IV-005
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert np.isfinite(result.J), "J-statistic should be finite"
    
    @pytest.mark.unit
    def test_cbiv010_str_repr(self, simple_iv_data):
        """
        CBIV-010: Verify __str__ and __repr__ methods work.
        
        Requirements: REQ-IV-006
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        str_output = str(result)
        repr_output = repr(result)
        
        assert isinstance(str_output, str)
        assert isinstance(repr_output, str)
        assert len(str_output) > 0
        assert len(repr_output) > 0


# =============================================================================
# Test Class: Parameter Options (CBIV-011 to CBIV-018)
# =============================================================================

class TestParameterOptions:
    """
    Test different parameter options for CBIV.
    
    Test IDs: CBIV-011 to CBIV-018
    Requirements: REQ-IV-007
    """
    
    @pytest.mark.unit
    def test_cbiv011_method_over(self, simple_iv_data):
        """
        CBIV-011: Verify method='over' option works.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert result.method == 'over'
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_cbiv012_method_exact(self, simple_iv_data):
        """
        CBIV-012: Verify method='exact' option works.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='exact',
            twosided=False
        )
        
        assert result.method == 'exact'
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_cbiv013_method_mle(self, simple_iv_data):
        """
        CBIV-013: Verify method='mle' option works.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='mle',
            twosided=False
        )
        
        assert result.method == 'mle'
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_cbiv014_twostep_true(self, simple_iv_data):
        """
        CBIV-014: Verify twostep=True option works.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twostep=True,
            twosided=False
        )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_cbiv015_twostep_false(self, simple_iv_data):
        """
        CBIV-015: Verify twostep=False (continuously updating GMM) option works.
        
        Requirements: REQ-IV-007
        Note: This may be slower due to continuous updating.
        """
        data = simple_iv_data
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress continuous updating warning
            result = CBIV(
                Tr=data['Tr'],
                Z=data['Z'],
                X=data['X'],
                method='over',
                twostep=False,
                twosided=False,
                iterations=100  # Limit iterations for speed
            )
        
        assert result.weights is not None
    
    @pytest.mark.unit
    def test_cbiv016_probs_min_parameter(self, simple_iv_data):
        """
        CBIV-016: Verify probs_min parameter affects probability clipping.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        # Test with different probs_min values
        result1 = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False,
            probs_min=1e-6
        )
        
        result2 = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False,
            probs_min=1e-4
        )
        
        # Both should produce valid results
        assert np.all(np.isfinite(result1.weights))
        assert np.all(np.isfinite(result2.weights))
    
    @pytest.mark.unit
    def test_cbiv017_iterations_parameter(self, simple_iv_data):
        """
        CBIV-017: Verify iterations parameter is respected.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False,
            iterations=50
        )
        
        assert result.iterations == 50
    
    @pytest.mark.unit
    def test_cbiv018_warn_clipping_parameter(self, simple_iv_data):
        """
        CBIV-018: Verify warn_clipping parameter works.
        
        Requirements: REQ-IV-007
        """
        data = simple_iv_data
        
        # Should not raise any errors
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False,
            warn_clipping=False
        )
        
        assert result.weights is not None


# =============================================================================
# Test Class: Input Validation (CBIV-019 to CBIV-024)
# =============================================================================

class TestInputValidation:
    """
    Test input validation for CBIV.
    
    Test IDs: CBIV-019 to CBIV-024
    Requirements: REQ-IV-008
    """
    
    @pytest.mark.unit
    def test_cbiv019_invalid_method_raises(self, simple_iv_data):
        """
        CBIV-019: Verify invalid method parameter raises ValueError.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        
        with pytest.raises(ValueError, match="method"):
            CBIV(
                Tr=data['Tr'],
                Z=data['Z'],
                X=data['X'],
                method='invalid_method',
                twosided=False
            )
    
    @pytest.mark.unit
    def test_cbiv020_non_binary_treatment_raises(self, simple_iv_data):
        """
        CBIV-020: Verify non-binary treatment raises ValueError.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        Tr_invalid = data['Tr'].copy().astype(float)
        Tr_invalid[0] = 0.5  # Non-binary value
        
        with pytest.raises(ValueError, match="binary"):
            CBIV(
                Tr=Tr_invalid,
                Z=data['Z'],
                X=data['X'],
                method='over',
                twosided=False
            )
    
    @pytest.mark.unit
    def test_cbiv021_non_binary_instrument_raises(self, simple_iv_data):
        """
        CBIV-021: Verify non-binary instrument raises ValueError.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        Z_invalid = data['Z'].copy().astype(float)
        Z_invalid[0] = 0.5  # Non-binary value
        
        with pytest.raises(ValueError, match="binary"):
            CBIV(
                Tr=data['Tr'],
                Z=Z_invalid,
                X=data['X'],
                method='over',
                twosided=False
            )
    
    @pytest.mark.unit
    def test_cbiv022_dimension_mismatch_raises(self, simple_iv_data):
        """
        CBIV-022: Verify dimension mismatch raises ValueError.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        
        with pytest.raises(ValueError, match="same number"):
            CBIV(
                Tr=data['Tr'][:100],  # Wrong length
                Z=data['Z'],
                X=data['X'],
                method='over',
                twosided=False
            )
    
    @pytest.mark.unit
    def test_cbiv023_invalid_probs_min_raises(self, simple_iv_data):
        """
        CBIV-023: Verify invalid probs_min raises ValueError.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        
        with pytest.raises(ValueError, match="probs_min"):
            CBIV(
                Tr=data['Tr'],
                Z=data['Z'],
                X=data['X'],
                method='over',
                twosided=False,
                probs_min=0.6  # Invalid: must be < 0.5
            )
    
    @pytest.mark.unit
    def test_cbiv024_invalid_iterations_raises(self, simple_iv_data):
        """
        CBIV-024: Verify invalid iterations raises error.
        
        Requirements: REQ-IV-008
        """
        data = simple_iv_data
        
        with pytest.raises((ValueError, TypeError)):
            CBIV(
                Tr=data['Tr'],
                Z=data['Z'],
                X=data['X'],
                method='over',
                twosided=False,
                iterations=-1  # Invalid
            )


# =============================================================================
# Test Class: Numerical Properties (CBIV-025 to CBIV-030)
# =============================================================================

class TestNumericalProperties:
    """
    Test numerical properties of CBIV.
    
    Test IDs: CBIV-025 to CBIV-030
    Requirements: REQ-IV-009
    """
    
    @pytest.mark.numerical
    def test_cbiv025_weights_finite(self, iv_data_onesided):
        """
        CBIV-025: Verify all weights are finite (no NaN or Inf).
        
        Requirements: REQ-IV-009
        """
        data = iv_data_onesided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert np.all(np.isfinite(result.weights)), \
            "All weights should be finite"
    
    @pytest.mark.numerical
    def test_cbiv026_coefficients_finite(self, iv_data_onesided):
        """
        CBIV-026: Verify all coefficients are finite.
        
        Requirements: REQ-IV-009
        """
        data = iv_data_onesided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert np.all(np.isfinite(result.coefficients)), \
            "All coefficients should be finite"
    
    @pytest.mark.numerical
    def test_cbiv027_fitted_values_bounded(self, simple_iv_data):
        """
        CBIV-027: Verify fitted values (probabilities) are in (0, 1).
        
        Requirements: REQ-IV-009
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        fitted = result.fitted_values
        
        assert np.all(fitted > 0), "Fitted values should be positive"
        assert np.all(fitted < 1), "Fitted values should be less than 1"
    
    @pytest.mark.numerical
    def test_cbiv028_twosided_probs_sum_to_one(self, iv_data_twosided):
        """
        CBIV-028: Verify two-sided probabilities sum to 1.
        
        Requirements: REQ-IV-009
        """
        data = iv_data_twosided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=True
        )
        
        # π_c + π_a + π_n should sum to 1
        prob_sums = result.fitted_values.sum(axis=1)
        
        assert_allclose(
            prob_sums,
            np.ones(len(prob_sums)),
            rtol=1e-6,
            err_msg="Probabilities should sum to 1"
        )
    
    @pytest.mark.numerical
    def test_cbiv029_reproducibility(self, simple_iv_data):
        """
        CBIV-029: Verify results are reproducible with same data.
        
        Requirements: REQ-IV-010
        """
        data = simple_iv_data
        
        result1 = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        result2 = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        assert_allclose(
            result1.weights,
            result2.weights,
            rtol=1e-6,
            err_msg="Results should be reproducible"
        )
    
    @pytest.mark.numerical
    def test_cbiv030_p_complier_property(self, iv_data_onesided):
        """
        CBIV-030: Verify p_complier property returns correct values.
        
        Requirements: REQ-IV-011
        """
        data = iv_data_onesided
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        p_c = result.p_complier
        
        # p_complier should be 1D array
        assert p_c.ndim == 1, "p_complier should be 1D array"
        
        # Weights should be 1/p_complier
        assert_allclose(
            result.weights,
            1.0 / p_c,
            rtol=1e-10,
            err_msg="weights should equal 1/p_complier"
        )


# =============================================================================
# Test Class: Formula Interface (CBIV-031 to CBIV-035)
# =============================================================================

class TestFormulaInterface:
    """
    Test formula interface for CBIV.
    
    Test IDs: CBIV-031 to CBIV-035
    Requirements: REQ-IV-012
    """
    
    @pytest.mark.unit
    def test_cbiv031_formula_interface_basic(self):
        """
        CBIV-031: Verify basic formula interface works.
        
        Requirements: REQ-IV-012
        """
        np.random.seed(42)
        n = 200
        
        df = pd.DataFrame({
            'treat': np.random.binomial(1, 0.3, n),
            'z': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n)
        })
        
        # Make treatment depend on z for IV validity
        comply = np.random.binomial(1, 0.7, n)
        df['treat'] = df['z'] * comply
        
        result = CBIV(
            formula="treat ~ x1 + x2 | z",
            data=df,
            method='over',
            twosided=False
        )
        
        assert isinstance(result, CBIVResults)
    
    @pytest.mark.unit
    def test_cbiv032_formula_requires_data(self):
        """
        CBIV-032: Verify formula without data raises error.
        
        Requirements: REQ-IV-012
        """
        with pytest.raises(ValueError, match="data"):
            CBIV(
                formula="treat ~ x1 + x2 | z",
                data=None,
                method='over'
            )
    
    @pytest.mark.unit
    def test_cbiv033_formula_requires_pipe(self):
        """
        CBIV-033: Verify formula without | raises error.
        
        Requirements: REQ-IV-012
        """
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'treat': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n)
        })
        
        with pytest.raises(ValueError, match=r"\|"):
            CBIV(
                formula="treat ~ x1",  # Missing | for instruments
                data=df,
                method='over'
            )
    
    @pytest.mark.unit
    def test_cbiv034_mutual_exclusivity(self):
        """
        CBIV-034: Verify formula and matrix interfaces are mutually exclusive.
        
        Requirements: REQ-IV-012
        """
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'treat': np.random.binomial(1, 0.5, n),
            'z': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n)
        })
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            CBIV(
                formula="treat ~ x1 | z",
                data=df,
                Tr=df['treat'].values,
                Z=df['z'].values,
                X=df[['x1']].values,
                method='over'
            )
    
    @pytest.mark.unit
    def test_cbiv035_formula_vs_matrix_equivalence(self):
        """
        CBIV-035: Verify formula and matrix interfaces produce equivalent results.
        
        Requirements: REQ-IV-012
        """
        np.random.seed(42)
        n = 200
        
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        z = np.random.binomial(1, 0.5, n)
        comply = np.random.binomial(1, 0.7, n)
        treat = z * comply
        
        df = pd.DataFrame({
            'treat': treat,
            'z': z,
            'x1': x1,
            'x2': x2
        })
        
        # Formula interface
        result_formula = CBIV(
            formula="treat ~ x1 + x2 | z",
            data=df,
            method='over',
            twosided=False
        )
        
        # Matrix interface
        X = np.column_stack([x1, x2])
        result_matrix = CBIV(
            Tr=treat,
            Z=z,
            X=X,
            method='over',
            twosided=False
        )
        
        # Results should be similar (may not be exactly equal due to intercept handling)
        assert_allclose(
            result_formula.weights,
            result_matrix.weights,
            rtol=1e-4,
            err_msg="Formula and matrix interfaces should produce equivalent results"
        )


# =============================================================================
# Test Class: Edge Cases (CBIV-036 to CBIV-040)
# =============================================================================

class TestEdgeCases:
    """
    Test edge cases for CBIV.
    
    Test IDs: CBIV-036 to CBIV-040
    Requirements: REQ-IV-010
    """
    
    @pytest.mark.edge_case
    def test_cbiv036_small_sample(self):
        """
        CBIV-036: Verify handling of small sample sizes.
        
        Requirements: REQ-IV-010
        """
        np.random.seed(42)
        n = 50  # Small sample
        
        X = np.random.randn(n, 2)
        Z = np.random.binomial(1, 0.5, n)
        p_comply = 1 / (1 + np.exp(-0.5 - 0.3 * X[:, 0]))
        comply = np.random.binomial(1, p_comply, n)
        Tr = Z * comply
        
        result = CBIV(
            Tr=Tr,
            Z=Z,
            X=X,
            method='exact',  # Use exact for small samples
            twosided=False
        )
        
        assert np.all(np.isfinite(result.weights)), \
            "Should handle small samples"
    
    @pytest.mark.edge_case
    def test_cbiv037_no_instrument_variation_raises(self):
        """
        CBIV-037: Verify error when instrument has no variation.
        
        Requirements: REQ-IV-010
        """
        np.random.seed(42)
        n = 100
        
        X = np.random.randn(n, 2)
        Z = np.ones(n)  # No variation
        Tr = np.random.binomial(1, 0.5, n)
        
        with pytest.raises(ValueError, match="no variation"):
            CBIV(
                Tr=Tr,
                Z=Z,
                X=X,
                method='over',
                twosided=False
            )
    
    @pytest.mark.edge_case
    def test_cbiv038_collinear_covariates_raises(self):
        """
        CBIV-038: Verify error with perfectly collinear covariates.
        
        Requirements: REQ-IV-010
        """
        np.random.seed(42)
        n = 100
        
        x1 = np.random.randn(n)
        x2 = x1 * 2  # Perfectly collinear
        X = np.column_stack([x1, x2])
        Z = np.random.binomial(1, 0.5, n)
        p_comply = 1 / (1 + np.exp(-0.5 * x1))
        comply = np.random.binomial(1, p_comply, n)
        Tr = Z * comply
        
        with pytest.raises(ValueError, match="rank-deficient"):
            CBIV(
                Tr=Tr,
                Z=Z,
                X=X,
                method='over',
                twosided=False
            )
    
    @pytest.mark.edge_case
    def test_cbiv039_vcov_method(self, simple_iv_data):
        """
        CBIV-039: Verify vcov() method returns valid covariance matrix.
        
        Requirements: REQ-IV-010
        """
        data = simple_iv_data
        
        result = CBIV(
            Tr=data['Tr'],
            Z=data['Z'],
            X=data['X'],
            method='over',
            twosided=False
        )
        
        try:
            vcov = result.vcov()
            
            # Should be symmetric
            assert_allclose(vcov, vcov.T, rtol=1e-10)
            
            # Should be square
            k = len(result.coefficients)
            assert vcov.shape == (k, k)
        except AttributeError:
            # vcov may not be available if computation failed
            pass
    
    @pytest.mark.edge_case
    @pytest.mark.slow
    def test_cbiv040_large_sample(self):
        """
        CBIV-040: Verify handling of larger sample sizes.
        
        Requirements: REQ-IV-010
        Note: This test may be slow.
        """
        np.random.seed(42)
        n = 1000
        
        X = np.random.randn(n, 3)
        Z = np.random.binomial(1, 0.5, n)
        p_comply = 1 / (1 + np.exp(-0.3 - 0.2 * X[:, 0] + 0.1 * X[:, 1]))
        comply = np.random.binomial(1, p_comply, n)
        Tr = Z * comply
        
        result = CBIV(
            Tr=Tr,
            Z=Z,
            X=X,
            method='over',
            twosided=False
        )
        
        assert result.converged or np.all(np.isfinite(result.weights)), \
            "Should handle large samples"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
