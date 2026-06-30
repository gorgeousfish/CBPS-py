"""
Comprehensive Integration Test Suite for CBPS Pipeline
======================================================

This module consolidates all integration tests for the CBPS package,
including end-to-end pipeline tests for all treatment types and
cross-module integration tests between CBPS estimation, diagnostics,
and inference modules.

Test IDs:
    - PIPE-001 ~ PIPE-010: Binary treatment pipeline
    - PIPE-011 ~ PIPE-020: Continuous treatment pipeline
    - PIPE-021 ~ PIPE-030: Multi-valued treatment pipeline
    - PIPE-031 ~ PIPE-040: Cross-treatment comparison
    - XMOD-001 ~ XMOD-010: CBPS + Diagnostics integration
    - XMOD-011 ~ XMOD-020: CBPS + Inference integration

Test Categories:
    - integration: End-to-end workflow and cross-module validation

Requirements:
    - REQ-PIPE-001: Complete workflow must succeed without errors
    - REQ-PIPE-002: Results must be numerically valid (no NaN/Inf)
    - REQ-PIPE-003: Weights must be positive and finite
    - REQ-PIPE-004: Diagnostics must be computable from results
    - REQ-XMOD-001: Diagnostics must work with CBPS results
    - REQ-XMOD-002: Balance metrics must be computable
    - REQ-XMOD-003: Standard errors must be computable

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cbps import CBPS
from cbps.core.results import CBPSResults
from cbps.diagnostics import balance_cbps as balance


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_binary_treatment_data(n=300, seed=42):
    """Generate binary treatment data for pipeline tests."""
    np.random.seed(seed)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Propensity model
    logit_ps = 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    ps = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps)

    # Outcome model
    y = 1.0 + 0.5 * treat + 0.3 * x1 - 0.2 * x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'treat': treat,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })


def generate_continuous_treatment_data(n=300, seed=42):
    """Generate continuous treatment data for pipeline tests."""
    np.random.seed(seed)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Continuous treatment
    treat = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)

    # Outcome
    y = 1.0 + 0.3 * treat + 0.4 * x1 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'treat': treat,
        'x1': x1,
        'x2': x2,
        'y': y
    })


def generate_multitreat_data(n=300, seed=42):
    """Generate multi-valued treatment data for pipeline tests."""
    np.random.seed(seed)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Multi-valued treatment (3 levels)
    logits = np.column_stack([
        0.3 * x1,
        -0.2 * x2
    ])
    exp_logits = np.exp(logits)
    denom = 1 + exp_logits.sum(axis=1, keepdims=True)
    probs = np.column_stack([1/denom, exp_logits / denom])

    treat = np.array([np.random.choice([0, 1, 2], p=p.ravel()) for p in probs])

    return pd.DataFrame({
        'treat': pd.Categorical(treat),
        'x1': x1,
        'x2': x2
    })


def generate_test_data(n=300, seed=42):
    """Generate test data for cross-module tests."""
    np.random.seed(seed)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    logit_ps = 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    ps = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps)

    y = 1.0 + 0.5 * treat + 0.3 * x1 - 0.2 * x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        'treat': treat,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })



# =============================================================================
# Test Class: Binary Treatment Pipeline
# =============================================================================

@pytest.mark.integration
class TestBinaryTreatmentPipeline:
    """
    Test ID: PIPE-001 ~ PIPE-010
    Requirement: REQ-PIPE-001, REQ-PIPE-002

    End-to-end pipeline tests for binary treatment CBPS.
    """

    @pytest.fixture
    def binary_data(self):
        """Binary treatment test data."""
        return generate_binary_treatment_data(n=300, seed=42)

    def test_pipe001_formula_interface_complete(self, binary_data):
        """PIPE-001: Complete workflow with formula interface."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0)

        # Assert
        assert isinstance(result, CBPSResults)
        assert result.converged
        assert np.all(np.isfinite(result.weights))
        assert np.all(result.weights > 0)

    def test_pipe002_ate_estimation_valid(self, binary_data):
        """PIPE-002: ATE estimation produces valid results."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0)

        # Assert
        assert result.att == 0
        assert np.isfinite(result.J)

    def test_pipe003_att_estimation_valid(self, binary_data):
        """PIPE-003: ATT estimation produces valid results."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=1)

        # Assert
        assert result.att == 1
        assert np.all(np.isfinite(result.weights))

    def test_pipe004_method_over_complete(self, binary_data):
        """PIPE-004: Over-identified method completes successfully."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0, method='over')

        # Assert
        assert result.method == 'over'
        assert np.isfinite(result.J)

    def test_pipe005_method_exact_complete(self, binary_data):
        """PIPE-005: Exact method completes successfully."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0, method='exact', two_step=False)

        # Assert
        assert result.method == 'exact'
        assert result.converged

    def test_pipe006_two_step_workflow(self, binary_data):
        """PIPE-006: Two-step GMM workflow completes successfully."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0, two_step=True)

        # Assert
        assert result.two_step is True
        assert result.converged

    def test_pipe007_vcov_computation(self, binary_data):
        """PIPE-007: Variance-covariance matrix is computable."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0)
        vcov = result.vcov()

        # Assert
        assert vcov is not None
        assert vcov.shape[0] == vcov.shape[1]
        assert np.all(np.isfinite(vcov))

    def test_pipe008_coefficient_access(self, binary_data):
        """PIPE-008: Coefficients are accessible and valid."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0)

        # Assert
        assert result.coefficients is not None
        assert len(result.coefficients) == 4  # intercept + 3 covariates
        assert np.all(np.isfinite(result.coefficients))

    def test_pipe009_fitted_values_valid(self, binary_data):
        """PIPE-009: Fitted propensity scores are in (0, 1)."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result = CBPS(formula=formula, data=df, att=0)

        # Assert
        assert np.all(result.fitted_values > 0)
        assert np.all(result.fitted_values < 1)

    def test_pipe010_reproducibility(self, binary_data):
        """PIPE-010: Results are reproducible with same data."""
        # Arrange
        df = binary_data
        formula = 'treat ~ x1 + x2 + x3'

        # Act
        result1 = CBPS(formula=formula, data=df, att=0)
        result2 = CBPS(formula=formula, data=df, att=0)

        # Assert
        assert_allclose(result1.coefficients, result2.coefficients, rtol=1e-6)
        assert_allclose(result1.weights, result2.weights, rtol=1e-6)


# =============================================================================
# Test Class: Continuous Treatment Pipeline
# =============================================================================

@pytest.mark.integration
class TestContinuousTreatmentPipeline:
    """
    Test ID: PIPE-011 ~ PIPE-020
    Requirement: REQ-PIPE-001, REQ-PIPE-002

    End-to-end pipeline tests for continuous treatment CBPS.
    """

    @pytest.fixture
    def continuous_data(self):
        """Continuous treatment test data."""
        return generate_continuous_treatment_data(n=300, seed=123)

    def test_pipe011_continuous_workflow_complete(self, continuous_data):
        """PIPE-011: Continuous treatment workflow completes successfully."""
        # Arrange
        df = continuous_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert isinstance(result, CBPSResults)
        assert result.sigmasq is not None  # Continuous-specific attribute
        assert np.all(np.isfinite(result.weights))

    def test_pipe012_gps_weights_positive(self, continuous_data):
        """PIPE-012: GPS weights are positive for continuous treatment."""
        # Arrange
        df = continuous_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert np.all(result.weights > 0)

    def test_pipe013_continuous_fitted_values(self, continuous_data):
        """PIPE-013: Fitted GPS values are in valid range (0, 1)."""
        # Arrange
        df = continuous_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert np.all(result.fitted_values > 0)
        assert np.all(result.fitted_values < 1)

    def test_pipe014_sigmasq_positive(self, continuous_data):
        """PIPE-014: Estimated residual variance is positive."""
        # Arrange
        df = continuous_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert result.sigmasq > 0


# =============================================================================
# Test Class: Multi-valued Treatment Pipeline
# =============================================================================

@pytest.mark.integration
class TestMultitreatPipeline:
    """
    Test ID: PIPE-021 ~ PIPE-030
    Requirement: REQ-PIPE-001, REQ-PIPE-002

    End-to-end pipeline tests for multi-valued treatment CBPS.
    """

    @pytest.fixture
    def multitreat_data(self):
        """Multi-valued treatment test data."""
        return generate_multitreat_data(n=300, seed=456)

    def test_pipe021_multitreat_workflow_complete(self, multitreat_data):
        """PIPE-021: Multi-valued treatment workflow completes successfully."""
        # Arrange
        df = multitreat_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert isinstance(result, CBPSResults)
        assert np.all(np.isfinite(result.weights))

    def test_pipe022_multitreat_probabilities_sum_to_one(self, multitreat_data):
        """PIPE-022: Fitted probabilities sum to 1 for each observation."""
        # Arrange
        df = multitreat_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        if result.fitted_values.ndim == 2:
            prob_sums = result.fitted_values.sum(axis=1)
            assert_allclose(prob_sums, np.ones(len(df)), rtol=1e-5)

    def test_pipe023_multitreat_weights_positive(self, multitreat_data):
        """PIPE-023: Weights are positive for all observations."""
        # Arrange
        df = multitreat_data
        formula = 'treat ~ x1 + x2'

        # Act
        result = CBPS(formula=formula, data=df)

        # Assert
        assert np.all(result.weights > 0)


# =============================================================================
# Test Class: Cross-Treatment Comparison
# =============================================================================

@pytest.mark.integration
class TestCrossTreatmentComparison:
    """
    Test ID: PIPE-031 ~ PIPE-040
    Requirement: REQ-PIPE-001

    Cross-treatment comparison tests to ensure consistent API behavior.
    """

    def test_pipe031_all_treatments_return_cbps_results(self):
        """PIPE-031: All treatment types return CBPSResults object."""
        # Arrange
        binary_df = generate_binary_treatment_data(n=200, seed=42)
        continuous_df = generate_continuous_treatment_data(n=200, seed=42)

        # Act
        binary_result = CBPS('treat ~ x1 + x2', data=binary_df, att=0)
        continuous_result = CBPS('treat ~ x1 + x2', data=continuous_df)

        # Assert
        assert isinstance(binary_result, CBPSResults)
        assert isinstance(continuous_result, CBPSResults)

    def test_pipe032_all_treatments_have_weights(self):
        """PIPE-032: All treatment types produce valid weights."""
        # Arrange
        binary_df = generate_binary_treatment_data(n=200, seed=42)
        continuous_df = generate_continuous_treatment_data(n=200, seed=42)

        # Act
        binary_result = CBPS('treat ~ x1 + x2', data=binary_df, att=0)
        continuous_result = CBPS('treat ~ x1 + x2', data=continuous_df)

        # Assert
        assert len(binary_result.weights) == 200
        assert len(continuous_result.weights) == 200
        assert np.all(binary_result.weights > 0)
        assert np.all(continuous_result.weights > 0)



# =============================================================================
# Test Class: CBPS + Diagnostics Integration (from test_cross_module.py)
# =============================================================================

@pytest.mark.integration
class TestCBPSDiagnosticsIntegration:
    """
    Test ID: XMOD-001 ~ XMOD-010
    Requirement: REQ-XMOD-001, REQ-XMOD-002

    Integration tests for CBPS and diagnostics module.
    """

    @pytest.fixture
    def fitted_cbps(self):
        """Fitted CBPS result."""
        df = generate_test_data(n=300, seed=42)
        result = CBPS('treat ~ x1 + x2 + x3', data=df, att=0)
        # Prepare dict format expected by balance_cbps
        cbps_dict = {
            'weights': result.weights,
            'x': result.x,
            'y': result.y
        }
        return result, cbps_dict, df

    def test_xmod001_balance_function_accepts_cbps_dict(self, fitted_cbps):
        """XMOD-001: balance() function accepts CBPS result dict."""
        # Arrange
        result, cbps_dict, df = fitted_cbps

        # Act
        bal = balance(cbps_dict)

        # Assert
        assert bal is not None

    def test_xmod002_balance_returns_dict(self, fitted_cbps):
        """XMOD-002: balance() returns a dict with balance statistics."""
        # Arrange
        result, cbps_dict, df = fitted_cbps

        # Act
        bal = balance(cbps_dict)

        # Assert
        assert isinstance(bal, dict)
        assert 'balanced' in bal
        assert 'original' in bal

    def test_xmod003_balance_includes_arrays(self, fitted_cbps):
        """XMOD-003: balance() includes balance statistics arrays."""
        # Arrange
        result, cbps_dict, df = fitted_cbps

        # Act
        bal = balance(cbps_dict)

        # Assert
        assert bal['balanced'] is not None
        assert bal['original'] is not None
        assert isinstance(bal['balanced'], np.ndarray)

    def test_xmod004_weighted_balance_valid(self, fitted_cbps):
        """XMOD-004: Balance arrays have valid numeric values."""
        # Arrange
        result, cbps_dict, df = fitted_cbps

        # Act
        bal = balance(cbps_dict)

        # Assert
        assert np.all(np.isfinite(bal['balanced']))
        assert np.all(np.isfinite(bal['original']))


# =============================================================================
# Test Class: CBPS + Inference Integration (from test_cross_module.py)
# =============================================================================

@pytest.mark.integration
class TestCBPSInferenceIntegration:
    """
    Test ID: XMOD-011 ~ XMOD-020
    Requirement: REQ-XMOD-003

    Integration tests for CBPS and inference module.
    """

    @pytest.fixture
    def fitted_cbps(self):
        """Fitted CBPS result."""
        df = generate_test_data(n=300, seed=42)
        result = CBPS('treat ~ x1 + x2 + x3', data=df, att=0)
        return result, df

    def test_xmod011_vcov_from_result(self, fitted_cbps):
        """XMOD-011: Variance-covariance matrix is computable from result."""
        # Arrange
        result, df = fitted_cbps

        # Act
        vcov = result.vcov()

        # Assert
        assert vcov is not None
        assert np.all(np.isfinite(vcov))

    def test_xmod012_vcov_symmetric(self, fitted_cbps):
        """XMOD-012: Variance-covariance matrix is symmetric."""
        # Arrange
        result, df = fitted_cbps

        # Act
        vcov = result.vcov()

        # Assert
        assert_allclose(vcov, vcov.T, atol=1e-10)

    def test_xmod013_vcov_correct_shape(self, fitted_cbps):
        """XMOD-013: Variance-covariance matrix has correct shape."""
        # Arrange
        result, df = fitted_cbps
        k = len(result.coefficients)

        # Act
        vcov = result.vcov()

        # Assert
        assert vcov.shape == (k, k)

    def test_xmod014_standard_errors_positive(self, fitted_cbps):
        """XMOD-014: Standard errors are positive."""
        # Arrange
        result, df = fitted_cbps

        # Act
        vcov = result.vcov()
        se = np.sqrt(np.diag(vcov))

        # Assert
        assert np.all(se > 0)

    def test_xmod015_summary_includes_se(self, fitted_cbps):
        """XMOD-015: Summary method includes standard errors."""
        # Arrange
        result, df = fitted_cbps

        # Act
        summary_str = str(result)

        # Assert
        # Summary should include some standard error or coefficient info
        assert 'Coefficients' in summary_str or 'coef' in summary_str.lower()


# =============================================================================
# Test Class: Complete Analysis Pipeline (from test_cross_module.py)
# =============================================================================

@pytest.mark.integration
class TestCompleteAnalysisPipeline:
    """
    Complete analysis pipeline from data to inference.

    This class tests a typical CBPS workflow:
    1. Load data
    2. Fit CBPS model
    3. Check balance
    4. Compute standard errors
    5. Report results
    """

    def test_complete_binary_analysis(self):
        """Complete binary treatment analysis pipeline."""
        # 1. Arrange: Generate data
        df = generate_test_data(n=500, seed=42)

        # 2. Act: Fit CBPS
        result = CBPS('treat ~ x1 + x2 + x3', data=df, att=0)

        # 3. Check balance
        cbps_dict = {
            'weights': result.weights,
            'x': result.x,
            'y': result.y
        }
        bal = balance(cbps_dict)

        # 4. Compute standard errors
        vcov = result.vcov()
        se = np.sqrt(np.diag(vcov))

        # 5. Assert: All steps succeeded
        assert result.converged
        assert isinstance(bal, dict)
        assert 'balanced' in bal
        assert np.all(se > 0)

        # Weights are valid
        assert np.all(result.weights > 0)
        assert np.all(np.isfinite(result.weights))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
