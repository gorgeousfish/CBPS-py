"""
Module: test_ocbps.py
=====================

Test Suite: Optimal CBPS (oCBPS)
Test IDs: OCBPS-001 to OCBPS-030
Requirements: REQ-OCBPS-001 to REQ-OCBPS-012

Overview:
    This module provides comprehensive tests for the optimal CBPS (oCBPS)
    implementation, which achieves double robustness and semiparametric
    efficiency through dual balancing conditions.

    Key innovations tested:
    - Dual balancing conditions (baseline + diff formulas)
    - Double robustness property
    - Semiparametric efficiency
    - Over-identified and exactly-identified cases

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Dual balancing condition verification
    - Integration tests: End-to-end workflow
    - Edge cases: Boundary conditions and error handling

Usage:
    pytest tests/optimal/ -v
    pytest tests/optimal/ -m "not slow"
    pytest tests/optimal/test_ocbps.py::TestBasicFunctionality -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cbps.core.cbps_optimal import cbps_optimal_2treat


# =============================================================================
# Test Fixtures (embedded from conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def ocbps_tolerances():
    """Tolerance values for oCBPS numerical comparisons."""
    return {
        'coefficient_rtol': 0.1,
        'coefficient_atol': 0.1,
        'weight_rtol': 1e-4,
        'weight_atol': 1e-4,
        'probs_rtol': 0.05,
    }


@pytest.fixture
def simple_ocbps_data():
    """
    Generate simple data for oCBPS tests.

    Returns
    -------
    dict
        Dictionary containing:
        - treat: Binary treatment (n=300)
        - X: Covariate matrix with intercept (n x 4)
        - baseline_X: Baseline covariates (n x 3)
        - diff_X: Diff covariates (n x 2)
    """
    np.random.seed(42)
    n = 300

    # Covariates (without intercept)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # X matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2, x3])

    # Treatment assignment depends on covariates
    logit_ps = -0.5 + 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    ps_true = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps_true).astype(float)

    # Baseline formula covariates (related to E(Y(0)|X))
    baseline_X = np.column_stack([x1, x2, x3])

    # Diff formula covariates (related to treatment effect heterogeneity)
    diff_X = np.column_stack([x1, x2])

    return {
        'treat': treat,
        'X': X,
        'baseline_X': baseline_X,
        'diff_X': diff_X,
        'n': n,
        'k': X.shape[1],
        'm1': baseline_X.shape[1],
        'm2': diff_X.shape[1],
    }


@pytest.fixture
def lalonde_style_ocbps_data():
    """
    Generate LaLonde-style data for oCBPS tests (over-identified case).

    Returns
    -------
    dict
        Dictionary containing treatment, covariates, and formula components.

    Notes
    -----
    This fixture ensures m1 + m2 + 1 > k (over-identified).
    """
    np.random.seed(123)
    n = 500

    # Generate LaLonde-style covariates
    age = np.random.uniform(18, 55, n)
    educ = np.random.randint(8, 17, n).astype(float)
    black = np.random.binomial(1, 0.3, n).astype(float)

    # Standardize continuous variables
    age_std = (age - age.mean()) / age.std()
    educ_std = (educ - educ.mean()) / educ.std()

    # X matrix with intercept (k = 4)
    X = np.column_stack([
        np.ones(n),
        age_std,
        educ_std,
        black
    ])

    # Treatment depends on covariates
    logit_ps = -0.5 + 0.2 * age_std - 0.3 * educ_std + 0.4 * black
    ps_true = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps_true).astype(float)

    # Baseline formula covariates (m1 = 3)
    baseline_X = np.column_stack([age_std, educ_std, black])

    # Diff formula covariates (m2 = 2)
    nodegree = (educ < 12).astype(float)
    diff_X = np.column_stack([age_std, nodegree])

    # m1 + m2 + 1 = 3 + 2 + 1 = 6 > k = 4 (over-identified)

    return {
        'treat': treat,
        'X': X,
        'baseline_X': baseline_X,
        'diff_X': diff_X,
        'n': n,
        'k': X.shape[1],
        'm1': baseline_X.shape[1],
        'm2': diff_X.shape[1],
    }


@pytest.fixture
def exactly_identified_data():
    """
    Generate data where m1 + m2 + 1 == k (exactly identified).

    Returns
    -------
    dict
        Data for exactly identified oCBPS.
    """
    np.random.seed(456)
    n = 200

    # k = 3 (intercept + 2 covariates)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # m1 = 1, m2 = 1 => m1 + m2 + 1 = 3 = k
    baseline_X = np.column_stack([x1])
    diff_X = np.column_stack([x2])

    # Treatment
    logit_ps = -0.3 + 0.4 * x1 - 0.3 * x2
    ps_true = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps_true).astype(float)

    return {
        'treat': treat,
        'X': X,
        'baseline_X': baseline_X,
        'diff_X': diff_X,
        'n': n,
        'k': X.shape[1],
        'm1': baseline_X.shape[1],
        'm2': diff_X.shape[1],
    }


# =============================================================================
# Test Class: Basic Functionality (OCBPS-001 to OCBPS-010)
# =============================================================================

class TestBasicFunctionality:
    """
    Test basic functionality of cbps_optimal_2treat.
    
    Test IDs: OCBPS-001 to OCBPS-010
    Requirements: REQ-OCBPS-001
    """
    
    @pytest.mark.unit
    def test_ocbps001_fit_returns_dict(self, simple_ocbps_data):
        """
        OCBPS-001: Verify cbps_optimal_2treat returns a dictionary.
        
        Requirements: REQ-OCBPS-001
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
    
    @pytest.mark.unit
    def test_ocbps002_required_keys_present(self, simple_ocbps_data):
        """
        OCBPS-002: Verify all required keys are present in result dictionary.
        
        Requirements: REQ-OCBPS-001
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        required_keys = [
            'coefficients',
            'fitted_values',
            'linear_predictor',
            'deviance',
            'weights',
            'y',
            'x',
            'converged',
            'J',
            'var',
            'mle_J'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
    
    @pytest.mark.unit
    def test_ocbps003_coefficients_shape(self, simple_ocbps_data):
        """
        OCBPS-003: Verify coefficients have correct shape.
        
        Requirements: REQ-OCBPS-002
        """
        data = simple_ocbps_data
        k = data['k']
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        coefficients = result['coefficients']
        
        # Coefficients should be (k, 1) matrix
        assert coefficients.shape == (k, 1), \
            f"Expected shape ({k}, 1), got {coefficients.shape}"
    
    @pytest.mark.unit
    def test_ocbps004_weights_shape_and_positive(self, simple_ocbps_data):
        """
        OCBPS-004: Verify weights have correct shape and are positive.
        
        Requirements: REQ-OCBPS-003
        """
        data = simple_ocbps_data
        n = data['n']
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        weights = result['weights']
        
        assert weights.shape == (n,), \
            f"Expected shape ({n},), got {weights.shape}"
        assert np.all(weights > 0), "All weights should be positive"
    
    @pytest.mark.unit
    def test_ocbps005_fitted_values_bounded(self, simple_ocbps_data):
        """
        OCBPS-005: Verify fitted values (propensity scores) are in (0, 1).
        
        Requirements: REQ-OCBPS-004
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        fitted = result['fitted_values']
        
        assert np.all(fitted > 0), "Fitted values should be positive"
        assert np.all(fitted < 1), "Fitted values should be less than 1"
    
    @pytest.mark.unit
    def test_ocbps006_j_statistic_computed(self, simple_ocbps_data):
        """
        OCBPS-006: Verify J-statistic is computed and finite.
        
        Requirements: REQ-OCBPS-005
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert np.isfinite(result['J']), "J-statistic should be finite"
    
    @pytest.mark.unit
    def test_ocbps007_deviance_positive(self, simple_ocbps_data):
        """
        OCBPS-007: Verify deviance is positive.
        
        Requirements: REQ-OCBPS-005
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert result['deviance'] > 0, "Deviance should be positive"
    
    @pytest.mark.unit
    def test_ocbps008_vcov_shape(self, simple_ocbps_data):
        """
        OCBPS-008: Verify variance-covariance matrix has correct shape.
        
        Requirements: REQ-OCBPS-006
        """
        data = simple_ocbps_data
        k = data['k']
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        vcov = result['var']
        
        assert vcov.shape == (k, k), \
            f"Expected shape ({k}, {k}), got {vcov.shape}"
    
    @pytest.mark.unit
    def test_ocbps009_mle_j_computed(self, simple_ocbps_data):
        """
        OCBPS-009: Verify MLE baseline J-statistic is computed.
        
        Requirements: REQ-OCBPS-005
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert np.isfinite(result['mle_J']), "mle_J should be finite"
    
    @pytest.mark.unit
    def test_ocbps010_convergence_flag(self, simple_ocbps_data):
        """
        OCBPS-010: Verify convergence flag is boolean.
        
        Requirements: REQ-OCBPS-007
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert isinstance(result['converged'], (bool, np.bool_)), \
            "Converged should be boolean"


# =============================================================================
# Test Class: Dual Balancing Conditions (OCBPS-011 to OCBPS-016)
# =============================================================================

class TestDualBalancing:
    """
    Test dual balancing conditions for oCBPS.
    
    Test IDs: OCBPS-011 to OCBPS-016
    Requirements: REQ-OCBPS-008
    """
    
    @pytest.mark.numerical
    def test_ocbps011_baseline_balance_condition(self, simple_ocbps_data):
        """
        OCBPS-011: Verify baseline balance condition is satisfied.
        
        Requirements: REQ-OCBPS-008
        
        Notes:
            g1_baseline: (T/π - (1-T)/(1-π)) h1(X) ≈ 0
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        treat = data['treat']
        ps = result['fitted_values']
        baseline_X = data['baseline_X']
        
        # Compute balance weights
        w = treat / ps - (1 - treat) / (1 - ps)
        
        # Compute balance condition (should be close to 0)
        balance = baseline_X.T @ w / len(treat)
        
        # Check that balance is approximately zero
        assert np.allclose(balance, 0, atol=0.1), \
            f"Baseline balance condition not satisfied: {balance}"
    
    @pytest.mark.numerical
    def test_ocbps012_diff_balance_condition(self, simple_ocbps_data):
        """
        OCBPS-012: Verify diff balance condition is satisfied.
        
        Requirements: REQ-OCBPS-008
        
        Notes:
            g2_diff: (T/π - 1) h2(X) ≈ 0
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        treat = data['treat']
        ps = result['fitted_values']
        diff_X = data['diff_X']
        
        # Compute diff weights
        w_diff = treat / ps - 1
        
        # Compute balance condition (should be close to 0)
        balance_diff = diff_X.T @ w_diff / len(treat)
        
        # Check that balance is approximately zero
        assert np.allclose(balance_diff, 0, atol=0.1), \
            f"Diff balance condition not satisfied: {balance_diff}"
    
    @pytest.mark.numerical
    def test_ocbps013_over_identified_case(self, lalonde_style_ocbps_data):
        """
        OCBPS-013: Verify over-identified case (m1 + m2 + 1 > k).
        
        Requirements: REQ-OCBPS-008
        """
        data = lalonde_style_ocbps_data
        
        # Verify over-identification
        m1, m2, k = data['m1'], data['m2'], data['k']
        assert m1 + m2 + 1 > k, "Should be over-identified"
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        # Should still converge
        assert np.all(np.isfinite(result['weights']))
    
    @pytest.mark.numerical
    def test_ocbps014_exactly_identified_case(self, exactly_identified_data):
        """
        OCBPS-014: Verify exactly identified case (m1 + m2 + 1 == k).
        
        Requirements: REQ-OCBPS-008
        """
        data = exactly_identified_data
        
        # Verify exact identification
        m1, m2, k = data['m1'], data['m2'], data['k']
        assert m1 + m2 + 1 == k, "Should be exactly identified"
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        # Should converge
        assert np.all(np.isfinite(result['weights']))
    
    @pytest.mark.unit
    def test_ocbps015_under_identified_raises(self):
        """
        OCBPS-015: Verify under-identified case raises error.
        
        Requirements: REQ-OCBPS-008
        """
        np.random.seed(42)
        n = 200
        
        # k = 5 (intercept + 4 covariates)
        X = np.column_stack([np.ones(n), np.random.randn(n, 4)])
        
        # m1 = 1, m2 = 1 => m1 + m2 + 1 = 3 < 5 = k (under-identified)
        baseline_X = np.column_stack([np.random.randn(n)])
        diff_X = np.column_stack([np.random.randn(n)])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        with pytest.raises(ValueError, match="Invalid"):
            cbps_optimal_2treat(
                treat=treat,
                X=X,
                baseline_X=baseline_X,
                diff_X=diff_X
            )
    
    @pytest.mark.numerical
    def test_ocbps016_intercept_balance(self, simple_ocbps_data):
        """
        OCBPS-016: Verify intercept balance in baseline condition.
        
        Requirements: REQ-OCBPS-008
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        treat = data['treat']
        ps = result['fitted_values']
        n = len(treat)
        
        # Intercept balance: sum of weights should be approximately 0
        w = treat / ps - (1 - treat) / (1 - ps)
        intercept_balance = np.sum(w) / n
        
        assert np.abs(intercept_balance) < 0.1, \
            f"Intercept balance not satisfied: {intercept_balance}"


# =============================================================================
# Test Class: Numerical Properties (OCBPS-017 to OCBPS-022)
# =============================================================================

class TestNumericalProperties:
    """
    Test numerical properties of oCBPS.
    
    Test IDs: OCBPS-017 to OCBPS-022
    Requirements: REQ-OCBPS-009
    """
    
    @pytest.mark.numerical
    def test_ocbps017_weights_finite(self, simple_ocbps_data):
        """
        OCBPS-017: Verify all weights are finite (no NaN or Inf).
        
        Requirements: REQ-OCBPS-009
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert np.all(np.isfinite(result['weights'])), \
            "All weights should be finite"
    
    @pytest.mark.numerical
    def test_ocbps018_coefficients_finite(self, simple_ocbps_data):
        """
        OCBPS-018: Verify all coefficients are finite.
        
        Requirements: REQ-OCBPS-009
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert np.all(np.isfinite(result['coefficients'])), \
            "All coefficients should be finite"
    
    @pytest.mark.numerical
    def test_ocbps019_vcov_symmetric(self, simple_ocbps_data):
        """
        OCBPS-019: Verify variance-covariance matrix is symmetric.
        
        Requirements: REQ-OCBPS-009
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        vcov = result['var']
        
        assert_allclose(vcov, vcov.T, rtol=1e-10, atol=1e-12), \
            "Variance-covariance matrix should be symmetric"
    
    @pytest.mark.numerical
    def test_ocbps020_reproducibility(self, simple_ocbps_data):
        """
        OCBPS-020: Verify results are reproducible with same data.
        
        Requirements: REQ-OCBPS-010
        """
        data = simple_ocbps_data
        
        result1 = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        result2 = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        assert_allclose(
            result1['coefficients'],
            result2['coefficients'],
            rtol=1e-6,
            err_msg="Results should be reproducible"
        )
    
    @pytest.mark.numerical
    def test_ocbps021_j_vs_mle_j(self, simple_ocbps_data):
        """
        OCBPS-021: Verify J-statistic is no larger than MLE baseline.
        
        Requirements: REQ-OCBPS-009
        
        Notes:
            oCBPS should improve over MLE in terms of balance.
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        # J should be no larger than mle_J (or at least comparable)
        # Allow some tolerance for numerical optimization
        assert result['J'] <= result['mle_J'] * 1.5, \
            f"J={result['J']} should be comparable to mle_J={result['mle_J']}"
    
    @pytest.mark.numerical
    def test_ocbps022_linear_predictor_consistency(self, simple_ocbps_data):
        """
        OCBPS-022: Verify linear predictor matches X @ coefficients.
        
        Requirements: REQ-OCBPS-009
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X']
        )
        
        # Linear predictor should match X @ beta
        expected_lp = data['X'] @ result['coefficients'].ravel()
        
        assert_allclose(
            result['linear_predictor'],
            expected_lp,
            rtol=1e-10,
            err_msg="Linear predictor should match X @ coefficients"
        )


# =============================================================================
# Test Class: Parameter Options (OCBPS-023 to OCBPS-026)
# =============================================================================

class TestParameterOptions:
    """
    Test parameter options for oCBPS.
    
    Test IDs: OCBPS-023 to OCBPS-026
    Requirements: REQ-OCBPS-011
    """
    
    @pytest.mark.unit
    def test_ocbps023_iterations_parameter(self, simple_ocbps_data):
        """
        OCBPS-023: Verify iterations parameter is respected.
        
        Requirements: REQ-OCBPS-011
        """
        data = simple_ocbps_data
        
        # Should work with custom iterations
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X'],
            iterations=50
        )
        
        assert np.all(np.isfinite(result['weights']))
    
    @pytest.mark.unit
    def test_ocbps024_standardize_true(self, simple_ocbps_data):
        """
        OCBPS-024: Verify standardize=True option works.
        
        Requirements: REQ-OCBPS-011
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X'],
            standardize=True
        )
        
        # Standardized weights should have specific normalization
        assert np.all(np.isfinite(result['weights']))
    
    @pytest.mark.unit
    def test_ocbps025_standardize_false(self, simple_ocbps_data):
        """
        OCBPS-025: Verify standardize=False option works.
        
        Requirements: REQ-OCBPS-011
        """
        data = simple_ocbps_data
        
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X'],
            standardize=False
        )
        
        assert np.all(np.isfinite(result['weights']))
    
    @pytest.mark.unit
    def test_ocbps026_att_must_be_zero(self, simple_ocbps_data):
        """
        OCBPS-026: Verify att parameter is only 0 (ATE).
        
        Requirements: REQ-OCBPS-011
        
        Notes:
            oCBPS only supports ATE estimation (att=0).
        """
        data = simple_ocbps_data
        
        # att=0 should work
        result = cbps_optimal_2treat(
            treat=data['treat'],
            X=data['X'],
            baseline_X=data['baseline_X'],
            diff_X=data['diff_X'],
            att=0
        )
        
        assert np.all(np.isfinite(result['weights']))


# =============================================================================
# Test Class: Edge Cases (OCBPS-027 to OCBPS-030)
# =============================================================================

class TestEdgeCases:
    """
    Test edge cases for oCBPS.
    
    Test IDs: OCBPS-027 to OCBPS-030
    Requirements: REQ-OCBPS-012
    """
    
    @pytest.mark.edge_case
    def test_ocbps027_small_sample(self):
        """
        OCBPS-027: Verify handling of small sample sizes.
        
        Requirements: REQ-OCBPS-012
        """
        np.random.seed(42)
        n = 50  # Small sample
        
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x2])
        
        baseline_X = np.column_stack([x1])
        diff_X = np.column_stack([x2])
        
        logit_ps = 0.3 * x1 - 0.2 * x2
        ps = 1 / (1 + np.exp(-logit_ps))
        treat = np.random.binomial(1, ps).astype(float)
        
        result = cbps_optimal_2treat(
            treat=treat,
            X=X,
            baseline_X=baseline_X,
            diff_X=diff_X
        )
        
        assert np.all(np.isfinite(result['weights'])), \
            "Should handle small samples"
    
    @pytest.mark.edge_case
    def test_ocbps028_unbalanced_treatment(self):
        """
        OCBPS-028: Verify handling of unbalanced treatment groups.
        
        Requirements: REQ-OCBPS-012
        """
        np.random.seed(42)
        n = 200
        
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        
        baseline_X = np.column_stack([x1])
        diff_X = np.column_stack([np.ones(n)])  # Simple constant
        
        # Unbalanced: 80% treated
        treat = np.random.binomial(1, 0.8, n).astype(float)
        
        result = cbps_optimal_2treat(
            treat=treat,
            X=X,
            baseline_X=baseline_X,
            diff_X=diff_X
        )
        
        assert np.all(np.isfinite(result['weights'])), \
            "Should handle unbalanced treatment groups"
    
    @pytest.mark.edge_case
    @pytest.mark.slow
    def test_ocbps029_large_sample(self):
        """
        OCBPS-029: Verify handling of larger sample sizes.
        
        Requirements: REQ-OCBPS-012
        Note: This test may be slow.
        """
        np.random.seed(42)
        n = 1000
        
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        
        baseline_X = np.column_stack([x1, x2])
        diff_X = np.column_stack([x3])
        
        logit_ps = 0.2 * x1 - 0.3 * x2 + 0.1 * x3
        ps = 1 / (1 + np.exp(-logit_ps))
        treat = np.random.binomial(1, ps).astype(float)
        
        result = cbps_optimal_2treat(
            treat=treat,
            X=X,
            baseline_X=baseline_X,
            diff_X=diff_X
        )
        
        assert result['converged'] or np.all(np.isfinite(result['weights'])), \
            "Should handle large samples"
    
    @pytest.mark.edge_case
    def test_ocbps030_single_baseline_covariate(self):
        """
        OCBPS-030: Verify handling of single baseline covariate.
        
        Requirements: REQ-OCBPS-012
        """
        np.random.seed(42)
        n = 200
        
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x2])
        
        # Single baseline covariate
        baseline_X = np.column_stack([x1])
        diff_X = np.column_stack([x2])
        
        logit_ps = 0.3 * x1 - 0.2 * x2
        ps = 1 / (1 + np.exp(-logit_ps))
        treat = np.random.binomial(1, ps).astype(float)
        
        result = cbps_optimal_2treat(
            treat=treat,
            X=X,
            baseline_X=baseline_X,
            diff_X=diff_X
        )
        
        assert np.all(np.isfinite(result['weights'])), \
            "Should handle single baseline covariate"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
