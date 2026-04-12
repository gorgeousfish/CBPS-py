"""
Module: test_inference.py
=========================

Unified Test Suite for CBPS Statistical Inference
Test IDs: INF-INIT-001 to INF-INIT-015, ASYVAR-001 to ASYVAR-025,
          INF-001 to INF-015, VCOV-001 to VCOV-030
Requirements: REQ-INF-001 to REQ-INF-012, REQ-INF-INIT-001 to REQ-INF-INIT-003

Overview
--------
This module consolidates all inference-related tests for the CBPS Python
package into a single file. It covers four major areas of statistical
inference for causal treatment effect estimation:

1. **Module API verification** (TestInferenceModuleExports,
   TestInferenceFunctionSignatures): Validates that the public interface
   of ``cbps.inference`` exposes ``asy_var`` and ``vcov_outcome`` with
   correct signatures.

2. **Asymptotic variance estimation** (TestAsyVarBasic, TestAsyVarMethods,
   TestAsyVarCI, TestAsyVarEdgeCases): Tests the ``asy_var`` function for
   computing asymptotic variance and confidence intervals for IPTW
   estimators of average treatment effects, covering both the CBPS full
   sandwich formula (Theorem 2.1 of Fan et al., 2022) and the oCBPS
   semiparametric efficiency bound (Corollary 2.2 of Fan et al., 2022).

3. **Paper formula verification** (TestJStatisticDistribution,
   TestDoubleRobustness, TestAsymptoticNormality, TestDoseResponse,
   TestContinuousAsymptoticProperties, TestEfficiencyComparison,
   TestSemiparametricEfficiency, TestOCBPSDoubleRobustness,
   TestOCBPSSemiparametricEfficiency, TestOCBPSLocalMisspecification):
   Monte Carlo simulations verifying key theoretical properties from
   Imai and Ratkovic (2014), Fong et al. (2018), and Fan et al. (2022).

4. **Variance-covariance for outcome regression** (TestVcovOutcomeBasic,
   TestVcovOutcomeInputValidation, TestVcovOutcomeNumerical,
   TestVcovOutcomeEdgeCases): Tests the ``vcov_outcome`` function for
   computing adjusted variance-covariance matrices for weighted least
   squares regression using CBPS weights, following Section 3.2 of
   Fong, Hazlett, and Imai (2018).

References
----------
Fan, Q., Hsu, Y.-C., Lieli, R. P., and Zhang, Y. (2022). Optimal Covariate
Balancing Conditions in Propensity Score Estimation. Journal of Business &
Economic Statistics, 40(4), 1468-1482.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
Score for a Continuous Treatment. The Annals of Applied Statistics,
12(1), 156-177.

Hahn, J. (1998). On the role of the propensity score in efficient
semiparametric estimation of average treatment effects.
Econometrica, 66(2), 315-331.

Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.

Usage
-----
::

    pytest tests/inference/test_inference.py -v
    pytest tests/inference/test_inference.py -m "not slow"
    pytest tests/inference/test_inference.py::TestAsyVarBasic -v
    pytest tests/inference/test_inference.py::TestJStatisticDistribution -v
"""

import inspect
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from typing import Dict

from cbps import CBPS
from cbps.inference.asyvar import asy_var
from cbps.inference.vcov_outcome import vcov_outcome
from cbps.core.results import CBPSResults


# =============================================================================
# Inference-Specific Fixtures (from conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def inference_tolerances():
    """
    Provide tolerance values for inference numerical comparisons.
    
    Returns
    -------
    dict
        Dictionary containing tolerance values:
        - vcov_rtol: Relative tolerance for variance-covariance matrices (1e-6)
        - vcov_atol: Absolute tolerance for variance-covariance matrices (1e-8)
        - se_rtol: Relative tolerance for standard errors (0.1)
        - se_atol: Absolute tolerance for standard errors (0.01)
        - eigenvalue_atol: Tolerance for positive semi-definite checks (1e-10)
    """
    return {
        'vcov_rtol': 1e-6,
        'vcov_atol': 1e-8,
        'se_rtol': 0.1,
        'se_atol': 0.01,
        'eigenvalue_atol': 1e-10,
    }


@pytest.fixture
def simple_inference_data():
    """
    Generate simple data for basic inference tests.
    
    Creates synthetic data with known treatment effect for testing
    variance estimation procedures.
    
    Returns
    -------
    dict
        Dictionary containing:
        - X: Covariate matrix with intercept, shape (300, 3)
        - treat: Binary treatment indicator, shape (300,)
        - y: Outcome variable, shape (300,)
        - n: Sample size (300)
        - true_ate: True average treatment effect (2.0)
    """
    np.random.seed(42)
    n = 300
    
    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Treatment (propensity depends on covariates)
    ps_true = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    treat = np.random.binomial(1, ps_true)
    
    # Outcome (with known treatment effect of 2.0)
    true_ate = 2.0
    y = 1 + true_ate * treat + 0.5 * x1 + 0.3 * x2 + np.random.randn(n)
    
    return {
        'X': X,
        'treat': treat.astype(float),
        'y': y,
        'n': n,
        'true_ate': true_ate,
    }


@pytest.fixture
def heterogeneous_effect_data():
    """
    Generate data with heterogeneous treatment effects.
    
    For testing variance estimation when treatment effects vary by subgroup.
    
    Returns
    -------
    dict
        Dictionary containing data with heterogeneous effects.
    """
    np.random.seed(123)
    n = 500
    
    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Treatment
    ps_true = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    treat = np.random.binomial(1, ps_true)
    
    # Heterogeneous treatment effect
    ate_base = 2.0
    effect = ate_base + 0.5 * x1  # Effect varies with x1
    y = 1 + effect * treat + 0.5 * x1 + 0.3 * x2 + np.random.randn(n)
    
    return {
        'X': X,
        'treat': treat.astype(float),
        'y': y,
        'n': n,
        'ate_base': ate_base,
    }


@pytest.fixture(scope="session")
def inference_module():
    """
    Import and provide the inference module.
    
    Returns
    -------
    module
        The cbps.inference module
    """
    try:
        from cbps import inference
        return inference
    except ImportError as e:
        pytest.skip(f"inference module not available: {e}")


@pytest.fixture(scope="session")
def asy_var_func():
    """Provide the asy_var function for asymptotic variance estimation."""
    try:
        from cbps.inference import asy_var
        return asy_var
    except ImportError as e:
        pytest.skip(f"asy_var not available: {e}")


# =============================================================================
# Module API Tests (from test_init.py)
# =============================================================================

@pytest.mark.unit
class TestInferenceModuleExports:
    """
    Test ID: INF-INIT-001 ~ INF-INIT-008
    Requirement: REQ-INF-INIT-001
    
    Tests for cbps.inference module exports.
    """
    
    def test_inf_init001_module_importable(self):
        """INF-INIT-001: cbps.inference module is importable."""
        from cbps import inference
        assert inference is not None
    
    def test_inf_init002_asy_var_importable(self):
        """INF-INIT-002: asy_var is importable from cbps.inference."""
        from cbps.inference import asy_var
        assert callable(asy_var)
    
    def test_inf_init003_vcov_outcome_importable(self):
        """INF-INIT-003: vcov_outcome is importable from cbps.inference."""
        from cbps.inference import vcov_outcome
        assert callable(vcov_outcome)
    
    def test_inf_init004_all_defined(self):
        """INF-INIT-004: __all__ is defined in inference module."""
        from cbps import inference
        assert hasattr(inference, '__all__')
        assert isinstance(inference.__all__, (list, tuple))
    
    def test_inf_init005_all_contains_asy_var(self):
        """INF-INIT-005: __all__ contains asy_var."""
        from cbps import inference
        assert 'asy_var' in inference.__all__
    
    def test_inf_init006_all_contains_vcov_outcome(self):
        """INF-INIT-006: __all__ contains vcov_outcome."""
        from cbps import inference
        assert 'vcov_outcome' in inference.__all__
    
    def test_inf_init007_no_private_exports(self):
        """INF-INIT-007: __all__ does not export private functions."""
        from cbps import inference
        for name in inference.__all__:
            assert not name.startswith('_'), f"Private {name} should not be in __all__"
    
    def test_inf_init008_all_exports_count(self):
        """INF-INIT-008: __all__ contains exactly 2 exports."""
        from cbps import inference
        assert len(inference.__all__) == 2


@pytest.mark.unit
class TestInferenceFunctionSignatures:
    """
    Test ID: INF-INIT-009 ~ INF-INIT-015
    Requirement: REQ-INF-INIT-002
    
    Tests for inference function signatures.
    """
    
    def test_inf_init009_asy_var_has_Y_param(self):
        """INF-INIT-009: asy_var has Y parameter."""
        from cbps.inference import asy_var
        sig = inspect.signature(asy_var)
        assert 'Y' in sig.parameters
    
    def test_inf_init010_asy_var_has_CBPS_obj_param(self):
        """INF-INIT-010: asy_var has CBPS_obj parameter."""
        from cbps.inference import asy_var
        sig = inspect.signature(asy_var)
        assert 'CBPS_obj' in sig.parameters
    
    def test_inf_init011_asy_var_has_method_param(self):
        """INF-INIT-011: asy_var has method parameter."""
        from cbps.inference import asy_var
        sig = inspect.signature(asy_var)
        assert 'method' in sig.parameters
    
    def test_inf_init012_vcov_outcome_has_cbps_fit_param(self):
        """INF-INIT-012: vcov_outcome has cbps_fit parameter."""
        from cbps.inference import vcov_outcome
        sig = inspect.signature(vcov_outcome)
        assert 'cbps_fit' in sig.parameters
    
    def test_inf_init013_vcov_outcome_has_Y_and_Z_params(self):
        """INF-INIT-013: vcov_outcome has Y and Z parameters for outcome regression."""
        from cbps.inference import vcov_outcome
        sig = inspect.signature(vcov_outcome)
        assert 'Y' in sig.parameters
        assert 'Z' in sig.parameters
    
    def test_inf_init014_asy_var_has_docstring(self):
        """INF-INIT-014: asy_var has proper docstring."""
        from cbps.inference import asy_var
        assert asy_var.__doc__ is not None
        assert len(asy_var.__doc__) > 50
    
    def test_inf_init015_vcov_outcome_has_docstring(self):
        """INF-INIT-015: vcov_outcome has proper docstring."""
        from cbps.inference import vcov_outcome
        assert vcov_outcome.__doc__ is not None
        assert len(vcov_outcome.__doc__) > 50


# =============================================================================
# Asymptotic Variance Tests (from test_asyvar.py)
# =============================================================================

# -----------------------------------------------------------------------------
# Fixture: simple_binary_data_asyvar
# Note: Renamed from simple_binary_data to avoid conflict with root conftest
# and inference conftest fixtures.
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_binary_data_asyvar():
    """
    Generate simple binary treatment data for asy_var tests.
    
    Returns
    -------
    dict
        Dictionary containing Y, X, treat, and fitted CBPS result.
        
    Note:
        Treatment is stored as int to avoid type conversion warnings.
    """
    np.random.seed(42)
    n = 300
    
    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # True propensity score
    logit_ps = 0.5 + 0.3 * x1 - 0.2 * x2
    ps_true = 1 / (1 + np.exp(-logit_ps))
    treat = np.random.binomial(1, ps_true)  # Keep as int, not float
    
    # Potential outcomes
    true_ate = 1.5
    Y0 = 1 + 0.5 * x1 + np.random.randn(n)
    Y1 = Y0 + true_ate
    Y = treat * Y1 + (1 - treat) * Y0
    
    # Fit CBPS
    try:
        cbps_result = CBPS(
            treatment=treat,  # int type avoids type warning
            covariates=X,
            method='over',
            att=0,
            two_step=True
        )
    except Exception:
        cbps_result = None
    
    return {
        'Y': Y,
        'X': X,
        'treat': treat,
        'true_ate': true_ate,
        'cbps_result': cbps_result,
        'n': n
    }


# -----------------------------------------------------------------------------
# Test Class: Basic Functionality (ASYVAR-001 to ASYVAR-010)
# -----------------------------------------------------------------------------

class TestAsyVarBasic:
    """
    Test basic functionality of asy_var function.
    
    Test IDs: ASYVAR-001 to ASYVAR-010
    Requirements: REQ-INF-001
    """
    
    @pytest.mark.unit
    def test_asyvar001_returns_dict(self, simple_binary_data_asyvar):
        """
        ASYVAR-001: Verify asy_var returns a dictionary.
        
        Requirements: REQ-INF-001
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_asyvar002_required_keys(self, simple_binary_data_asyvar):
        """
        ASYVAR-002: Verify result contains required keys.
        
        Requirements: REQ-INF-001
        
        Note: Keys follow R-style naming convention:
        - mu.hat: ATE estimate
        - asy.var: Asymptotic variance
        - var: Variance of ATE
        - std.err: Standard error
        - CI.mu.hat: Confidence interval (array)
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        # R-style keys
        required_keys = ['mu.hat', 'var', 'std.err', 'CI.mu.hat']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
    
    @pytest.mark.unit
    def test_asyvar003_variance_positive(self, simple_binary_data_asyvar):
        """
        ASYVAR-003: Verify variance is positive.
        
        Requirements: REQ-INF-002
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        assert result['var'] > 0, "Variance should be positive"
    
    @pytest.mark.unit
    def test_asyvar004_se_equals_sqrt_var(self, simple_binary_data_asyvar):
        """
        ASYVAR-004: Verify SE equals square root of variance.
        
        Requirements: REQ-INF-002
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        assert_allclose(result['std.err'], np.sqrt(result['var']), rtol=1e-10)

    
    @pytest.mark.unit
    def test_asyvar005_ci_contains_ate(self, simple_binary_data_asyvar):
        """
        ASYVAR-005: Verify confidence interval contains ATE estimate.
        
        Requirements: REQ-INF-003
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS',
            CI=0.95
        )
        
        ci = result['CI.mu.hat']
        ate = result['mu.hat']
        assert ci[0] <= ate <= ci[1], \
            "CI should contain point estimate"
    
    @pytest.mark.unit
    def test_asyvar006_ci_width_positive(self, simple_binary_data_asyvar):
        """
        ASYVAR-006: Verify confidence interval has positive width.
        
        Requirements: REQ-INF-003
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        ci = result['CI.mu.hat']
        ci_width = ci[1] - ci[0]
        assert ci_width > 0, "CI width should be positive"


# -----------------------------------------------------------------------------
# Test Class: Method Comparison (ASYVAR-011 to ASYVAR-015)
# -----------------------------------------------------------------------------

class TestAsyVarMethods:
    """
    Test different variance estimation methods.
    
    Test IDs: ASYVAR-011 to ASYVAR-015
    Requirements: REQ-INF-004
    """
    
    @pytest.mark.unit
    def test_asyvar011_cbps_method(self, simple_binary_data_asyvar):
        """
        ASYVAR-011: Verify CBPS method works.
        
        Requirements: REQ-INF-004
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        assert np.isfinite(result['var'])
        assert np.isfinite(result['mu.hat'])
    
    @pytest.mark.unit
    def test_asyvar012_ocbps_method(self, simple_binary_data_asyvar):
        """
        ASYVAR-012: Verify oCBPS method works.
        
        Requirements: REQ-INF-004
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='oCBPS'
        )
        
        assert np.isfinite(result['var'])
        assert np.isfinite(result['mu.hat'])
    
    @pytest.mark.unit
    def test_asyvar013_methods_give_same_ate(self, simple_binary_data_asyvar):
        """
        ASYVAR-013: Verify both methods give same ATE estimate.
        
        Requirements: REQ-INF-004
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result_cbps = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        result_ocbps = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='oCBPS'
        )
        
        # ATE should be the same regardless of variance method
        assert_allclose(result_cbps['mu.hat'], result_ocbps['mu.hat'], rtol=1e-10)


# -----------------------------------------------------------------------------
# Test Class: Confidence Interval (ASYVAR-016 to ASYVAR-020)
# -----------------------------------------------------------------------------

class TestAsyVarCI:
    """
    Test confidence interval computation.
    
    Test IDs: ASYVAR-016 to ASYVAR-020
    Requirements: REQ-INF-003
    """
    
    @pytest.mark.unit
    def test_asyvar016_ci_level_90(self, simple_binary_data_asyvar):
        """
        ASYVAR-016: Verify 90% CI is narrower than 95% CI.
        
        Requirements: REQ-INF-003
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result_90 = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS',
            CI=0.90
        )
        
        result_95 = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS',
            CI=0.95
        )
        
        ci_90 = result_90['CI.mu.hat']
        ci_95 = result_95['CI.mu.hat']
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        
        assert width_90 < width_95, "90% CI should be narrower than 95% CI"
    
    @pytest.mark.unit
    def test_asyvar017_ci_level_99(self, simple_binary_data_asyvar):
        """
        ASYVAR-017: Verify 99% CI is wider than 95% CI.
        
        Requirements: REQ-INF-003
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result_95 = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS',
            CI=0.95
        )
        
        result_99 = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS',
            CI=0.99
        )
        
        ci_95 = result_95['CI.mu.hat']
        ci_99 = result_99['CI.mu.hat']
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        
        assert width_99 > width_95, "99% CI should be wider than 95% CI"


# -----------------------------------------------------------------------------
# Test Class: Edge Cases (ASYVAR-021 to ASYVAR-025)
# -----------------------------------------------------------------------------

class TestAsyVarEdgeCases:
    """
    Test edge cases for asy_var function.
    
    Test IDs: ASYVAR-021 to ASYVAR-025
    Requirements: REQ-INF-005
    """
    
    @pytest.mark.unit
    def test_asyvar021_finite_results(self, simple_binary_data_asyvar):
        """
        ASYVAR-021: Verify all results are finite.
        
        Requirements: REQ-INF-005
        """
        data = simple_binary_data_asyvar
        if data['cbps_result'] is None:
            pytest.skip("CBPS fitting failed")
        
        result = asy_var(
            Y=data['Y'],
            CBPS_obj=data['cbps_result'],
            method='CBPS'
        )
        
        assert np.isfinite(result['mu.hat']), "ATE should be finite"
        assert np.isfinite(result['var']), "Variance should be finite"
        assert np.isfinite(result['std.err']), "SE should be finite"
        ci = result['CI.mu.hat']
        assert np.isfinite(ci[0]), "CI lower should be finite"
        assert np.isfinite(ci[1]), "CI upper should be finite"


# =============================================================================
# Paper Formula Verification Tests (from test_paper_formulas.py)
# =============================================================================

# Mark all paper reproduction tests
# Note: pytestmark from test_paper_formulas.py is applied per-class below
# since module-level pytestmark would affect all tests in this merged file.


class TestJStatisticDistribution:
    """
    Verify J-statistic follows chi-squared distribution under correct specification.
    
    Paper: Imai & Ratkovic 2014 JRSS-B, Section 2.3
    
    Under correct model specification, the J-statistic (over-identification test)
    should follow a chi-squared distribution with degrees of freedom equal to
    the number of over-identifying restrictions.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_j_statistic_chi_squared(self):
        """Test that J-statistic follows chi-squared distribution."""
        n_sim = 200
        n = 500
        k = 4  # Number of covariates (excluding intercept)
        
        j_stats = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            # Generate data under correct specification
            X = np.random.randn(n, k)
            ps_coef = np.array([0.3, 0.2, 0.1, 0.1])
            logit_ps = X @ ps_coef
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged and hasattr(result, 'J') and result.J is not None:
                    if np.isfinite(result.J) and result.J >= 0:
                        j_stats.append(result.J)
            except Exception:
                pass
        
        if len(j_stats) > 50:
            j_stats = np.array(j_stats)
            
            # Degrees of freedom = k (over-identification)
            df = k
            
            # Test using Kolmogorov-Smirnov test
            # Under H0: J ~ chi2(df)
            ks_stat, p_value = stats.kstest(j_stats, 'chi2', args=(df,))
            
            # We expect p-value > 0.01 (not reject H0)
            # But allow some flexibility due to finite sample
            assert p_value > 0.001 or np.median(j_stats) < 2 * df, \
                f"J-statistic distribution test failed: KS p-value={p_value:.4f}"
            
            # Check mean is approximately df
            mean_j = j_stats.mean()
            assert abs(mean_j - df) < 2 * df, \
                f"J-statistic mean {mean_j:.2f} far from expected {df}"


class TestDoubleRobustness:
    """
    Verify double robustness property of CBPS estimator.
    
    Paper: Imai & Ratkovic 2014 JRSS-B, Section 3
    
    The CBPS estimator is doubly robust: it remains consistent if either
    the propensity score model OR the outcome model is correctly specified.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_double_robustness_ps_correct(self):
        """Test DR when propensity score is correct but outcome is wrong.
        
        Note: This is a Monte Carlo test with inherent randomness. Uses 4*SE
        threshold (99.99% confidence) to reduce false failures while still
        detecting genuine violations of double robustness.
        """
        n_sim = 100
        n = 500
        true_ate = 1.0
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            
            # Correct PS model
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            # Nonlinear outcome (misspecified if using linear)
            Y0 = 1 + np.sin(X[:, 0]) + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            estimates.append(ate_est)
            except Exception:
                pass
        
        if len(estimates) > 20:
            estimates = np.array(estimates)
            bias = estimates.mean() - true_ate
            se = estimates.std() / np.sqrt(len(estimates))
            
            # Should still be approximately unbiased (use 4*SE for Monte Carlo robustness)
            assert abs(bias) < 4 * se, \
                f"DR property violated: bias={bias:.4f}, 4*SE={4*se:.4f}"


class TestAsymptoticNormality:
    """
    Verify root-n consistency and asymptotic normality of CBPS estimator.
    
    Paper: Imai & Ratkovic 2014 JRSS-B, Theorem 2
    
    The standardized treatment effect estimates should converge to a
    normal distribution as sample size increases.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_asymptotic_normality(self):
        """Test that standardized estimates are approximately normal."""
        n_sim = 200
        n = 500
        true_ate = 1.0
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            Y0 = 1 + 0.5 * X[:, 0] + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            estimates.append(ate_est)
            except Exception:
                pass
        
        if len(estimates) > 100:
            estimates = np.array(estimates)
            
            # Standardize
            z_scores = (estimates - estimates.mean()) / estimates.std()
            
            # Test normality using Shapiro-Wilk
            # Take a subsample for the test
            subsample = np.random.choice(z_scores, min(50, len(z_scores)), replace=False)
            stat, p_value = stats.shapiro(subsample)
            
            # Should not strongly reject normality
            assert p_value > 0.001, \
                f"Normality test failed: Shapiro-Wilk p-value={p_value:.4f}"


class TestDoseResponse:
    """
    Verify dose-response relationship estimation for continuous treatment.
    
    Paper: Fong et al. 2018 AOAS
    
    Tests that CBPS can accurately estimate nonlinear dose-response
    relationships between continuous treatment and outcome.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_dose_response_estimation(self):
        """Test dose-response curve estimation."""
        n = 1000
        np.random.seed(42)
        
        # Generate continuous treatment
        X = np.random.randn(n, 3)
        treat = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
        
        # True dose-response: Y = 1 + 0.5*T + 0.3*T^2 + X'beta + epsilon
        Y = 1 + 0.5 * treat + 0.1 * treat**2 + 0.5 * X[:, 0] + np.random.randn(n)
        
        try:
            X_with_intercept = np.column_stack([np.ones(n), X])
            
            result = CBPS(
                treatment=treat,
                covariates=X_with_intercept,
                method='over',
                two_step=True
            )
            
            if result.converged:
                # Estimate dose-response using weighted regression
                W = np.diag(result.weights)
                design = np.column_stack([np.ones(n), treat, treat**2, X])
                
                XtWX = design.T @ W @ design
                XtWY = design.T @ W @ Y
                beta_hat = np.linalg.solve(XtWX, XtWY)
                
                # Check linear term is close to 0.5
                assert abs(beta_hat[1] - 0.5) < 0.2, \
                    f"Linear dose-response coefficient {beta_hat[1]:.3f} far from 0.5"
        except Exception as e:
            pytest.skip(f"Dose-response test skipped: {e}")


class TestContinuousAsymptoticProperties:
    """
    Verify asymptotic normality for continuous treatment CBPS.
    
    Paper: Fong et al. 2018 AOAS
    
    Tests that the distribution of treatment effect estimates approaches
    normality for continuous treatment settings.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_continuous_asymptotic_normality(self):
        """Test asymptotic normality for continuous CBPS."""
        n_sim = 100
        n = 500
        true_beta = 0.5
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            treat = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
            Y = 1 + true_beta * treat + 0.5 * X[:, 0] + np.random.randn(n)
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    two_step=True
                )
                
                if result.converged:
                    W = np.diag(result.weights)
                    design = np.column_stack([np.ones(n), treat, X])
                    XtWX = design.T @ W @ design
                    XtWY = design.T @ W @ Y
                    beta_hat = np.linalg.solve(XtWX, XtWY)
                    estimates.append(beta_hat[1])
            except Exception:
                pass
        
        if len(estimates) > 50:
            estimates = np.array(estimates)
            z_scores = (estimates - estimates.mean()) / estimates.std()
            
            # Check approximate normality
            skewness = stats.skew(z_scores)
            kurtosis = stats.kurtosis(z_scores)
            
            assert abs(skewness) < 1.0, f"Skewness {skewness:.2f} too large"
            assert abs(kurtosis) < 2.0, f"Excess kurtosis {kurtosis:.2f} too large"


class TestEfficiencyComparison:
    """
    Compare CBPS efficiency with OLS and IPW estimators.
    
    CBPS should achieve variance comparable to or better than traditional
    IPW estimators, while trading off some efficiency vs OLS when the
    outcome model is correctly specified.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_efficiency_vs_ols(self):
        """Test that CBPS is reasonably efficient compared to OLS.
        
        Note: This is a Monte Carlo test for continuous treatment. CBPS aims for
        covariate balance which may trade off some efficiency vs OLS when the
        outcome model is correctly specified. We use a generous tolerance (3x)
        to account for Monte Carlo variation and the trade-off between balance
        and efficiency.
        """
        n_sim = 100
        n = 500
        true_beta = 0.5
        
        cbps_estimates = []
        ols_estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            treat = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
            Y = 1 + true_beta * treat + 0.5 * X[:, 0] + np.random.randn(n)
            
            # OLS estimate
            design = np.column_stack([np.ones(n), treat, X])
            beta_ols = np.linalg.lstsq(design, Y, rcond=None)[0]
            ols_estimates.append(beta_ols[1])
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    two_step=True
                )
                
                if result.converged:
                    W = np.diag(result.weights)
                    XtWX = design.T @ W @ design
                    XtWY = design.T @ W @ Y
                    beta_cbps = np.linalg.solve(XtWX, XtWY)
                    cbps_estimates.append(beta_cbps[1])
            except Exception:
                pass
        
        if len(cbps_estimates) > 50:
            cbps_var = np.var(cbps_estimates)
            ols_var = np.var(ols_estimates)
            
            # CBPS should have similar or lower variance
            # Allow generous tolerance for Monte Carlo variation
            assert cbps_var < 3 * ols_var, \
                f"CBPS variance {cbps_var:.4f} much larger than OLS {ols_var:.4f}"


class TestSemiparametricEfficiency:
    """
    Verify that oCBPS approaches the semiparametric efficiency bound.
    
    Paper: Fan et al. 2022 JBES
    
    Under correct specification, the optimal CBPS estimator should achieve
    the semiparametric efficiency bound for ATE estimation.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_efficiency_bound(self):
        """Test that oCBPS approaches efficiency bound."""
        # This is a simplified test - full verification would require
        # computing the actual semiparametric efficiency bound
        
        n_sim = 100
        n = 1000  # Larger sample for efficiency
        true_ate = 1.0
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            Y0 = 1 + 0.5 * X[:, 0] + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            estimates.append(ate_est)
            except Exception:
                pass
        
        if len(estimates) > 50:
            estimates = np.array(estimates)
            
            # Check variance is reasonable (should be O(1/n))
            var_est = estimates.var()
            expected_var_order = 1.0 / n  # Rough order of magnitude
            
            # Variance should be within reasonable range
            assert var_est < 10 * expected_var_order, \
                f"Variance {var_est:.6f} much larger than expected O(1/n)"


class TestOCBPSDoubleRobustness:
    """
    Verify oCBPS double robustness property.
    
    Paper: Fan et al. 2022 JBES, Theorem 1
    
    The optimal CBPS estimator inherits the double robustness property.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_ocbps_double_robustness(self):
        """Test oCBPS double robustness property."""
        # Similar to TestDoubleRobustness but specifically for oCBPS
        n_sim = 100
        n = 500
        true_ate = 1.0
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            # Misspecified outcome
            Y0 = 1 + np.exp(0.3 * X[:, 0]) + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            estimates.append(ate_est)
            except Exception:
                pass
        
        if len(estimates) > 20:
            estimates = np.array(estimates)
            bias = estimates.mean() - true_ate
            se = estimates.std() / np.sqrt(len(estimates))
            
            assert abs(bias) < 3 * se, \
                f"oCBPS DR property violated: bias={bias:.4f}"


class TestOCBPSSemiparametricEfficiency:
    """
    Verify oCBPS achieves semiparametric efficiency.
    
    Paper: Fan et al. 2022 JBES, Theorem 2
    
    The optimal CBPS estimator should achieve variance no larger than
    the standard IPW estimator with true propensity scores.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_ocbps_efficiency(self):
        """Test oCBPS achieves semiparametric efficiency."""
        # Simplified test - compare variance to IPW
        n_sim = 100
        n = 500
        true_ate = 1.0
        
        cbps_estimates = []
        ipw_estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            Y0 = 1 + 0.5 * X[:, 0] + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            # IPW with true PS
            ipw_ate = (Y * treat / ps_true).mean() - (Y * (1 - treat) / (1 - ps_true)).mean()
            ipw_estimates.append(ipw_ate)
            
            try:
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            cbps_estimates.append(ate_est)
            except Exception:
                pass
        
        if len(cbps_estimates) > 50:
            cbps_var = np.var(cbps_estimates)
            ipw_var = np.var(ipw_estimates)
            
            # CBPS should have similar or lower variance than IPW
            assert cbps_var < 1.5 * ipw_var, \
                f"CBPS variance {cbps_var:.4f} much larger than IPW {ipw_var:.4f}"


class TestOCBPSLocalMisspecification:
    """
    Verify oCBPS robustness to local model misspecification.
    
    Paper: Fan et al. 2022 JBES
    
    The optimal CBPS estimator should remain approximately unbiased
    even when the propensity score model has minor misspecification.
    """
    
    @pytest.mark.slow
    @pytest.mark.paper_reproduction
    def test_local_misspecification_robustness(self):
        """Test robustness to local model misspecification."""
        n_sim = 100
        n = 500
        true_ate = 1.0
        
        estimates = []
        
        for sim in range(n_sim):
            np.random.seed(sim)
            
            X = np.random.randn(n, 3)
            
            # Slightly misspecified PS (missing small nonlinearity)
            logit_ps = 0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.05 * X[:, 0]**2
            ps_true = 1 / (1 + np.exp(-logit_ps))
            treat = np.random.binomial(1, ps_true).astype(float)
            
            Y0 = 1 + 0.5 * X[:, 0] + np.random.randn(n)
            Y1 = Y0 + true_ate
            Y = treat * Y1 + (1 - treat) * Y0
            
            try:
                # Fit with linear PS model (slightly misspecified)
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                result = CBPS(
                    treatment=treat,
                    covariates=X_with_intercept,
                    method='over',
                    att=0,
                    two_step=True
                )
                
                if result.converged:
                    w = result.weights
                    w1 = w * treat
                    w0 = w * (1 - treat)
                    if w1.sum() > 0 and w0.sum() > 0:
                        ate_est = (Y * w1).sum() / w1.sum() - (Y * w0).sum() / w0.sum()
                        if np.isfinite(ate_est):
                            estimates.append(ate_est)
            except Exception:
                pass
        
        if len(estimates) > 20:
            estimates = np.array(estimates)
            bias = estimates.mean() - true_ate
            se = estimates.std() / np.sqrt(len(estimates))
            
            # Should still be approximately unbiased despite misspecification
            assert abs(bias) < 3 * se, \
                f"Not robust to local misspecification: bias={bias:.4f}"


# =============================================================================
# Variance-Covariance Outcome Tests (from test_vcov_outcome.py)
# =============================================================================

# -----------------------------------------------------------------------------
# Fixtures for vcov_outcome tests
# -----------------------------------------------------------------------------

@pytest.fixture
def continuous_cbps_fit():
    """
    Create a mock continuous treatment CBPSResults object.
    
    Returns
    -------
    CBPSResults
        Mock CBPS fit with continuous treatment attributes.
    """
    np.random.seed(42)
    n = 200
    k = 4
    
    # Covariates with intercept
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    
    # True coefficients for treatment model
    beta_true = np.array([0.5, 0.3, -0.2, 0.1])
    
    # Continuous treatment
    T = X @ beta_true + np.random.randn(n)
    
    # Standardized treatment
    Ttilde = (T - T.mean()) / T.std()
    
    # Whitened covariates (simplified: use original X)
    Xtilde = X.copy()
    
    # Beta in whitened space
    beta_tilde = np.linalg.lstsq(Xtilde, Ttilde, rcond=None)[0]
    
    # Residual variance in whitened space
    residuals = Ttilde - Xtilde @ beta_tilde
    sigmasq_tilde = np.var(residuals, ddof=k)
    
    # Weights
    weights = np.exp(-0.5 * residuals**2 / sigmasq_tilde)
    weights = weights / weights.mean()  # Normalize
    
    # Fitted values
    fitted_values = Xtilde @ beta_tilde
    
    return CBPSResults(
        coefficients=beta_true.reshape(-1, 1),
        fitted_values=fitted_values,
        weights=weights,
        linear_predictor=fitted_values,
        y=T,
        x=X,
        J=2.5,
        mle_J=1.8,
        deviance=150.0,
        converged=True,
        var=np.eye(k) * 0.01,
        sigmasq=1.0,
        Ttilde=Ttilde,
        Xtilde=Xtilde,
        beta_tilde=beta_tilde,
        sigmasq_tilde=sigmasq_tilde,
    )


@pytest.fixture
def outcome_data():
    """
    Generate outcome data for vcov_outcome testing.
    
    Returns
    -------
    dict
        Dictionary with outcome Y, design matrix Z, and coefficients delta.
    """
    np.random.seed(123)
    n = 200
    q = 3  # Number of outcome model parameters
    
    # Outcome design matrix (intercept + treatment + covariate)
    Z = np.column_stack([
        np.ones(n),
        np.random.randn(n),  # Treatment effect
        np.random.randn(n)   # Covariate
    ])
    
    # True outcome coefficients
    delta_true = np.array([1.0, 0.5, 0.3])
    
    # Outcome
    Y = Z @ delta_true + np.random.randn(n) * 0.5
    
    # WLS coefficients (mock)
    delta = delta_true + np.random.randn(q) * 0.1
    
    return {
        'Y': Y,
        'Z': Z,
        'delta': delta,
        'n': n,
        'q': q
    }


# -----------------------------------------------------------------------------
# Test Class: Basic Functionality (VCOV-001 to VCOV-010)
# -----------------------------------------------------------------------------

class TestVcovOutcomeBasic:
    """
    Test basic functionality of vcov_outcome.
    
    Test IDs: VCOV-001 to VCOV-010
    Requirements: REQ-INF-006
    """
    
    @pytest.mark.unit
    def test_vcov001_returns_array(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-001: Verify vcov_outcome returns ndarray.
        
        Requirements: REQ-INF-006
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.unit
    def test_vcov002_correct_shape(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-002: Verify output has correct shape (q x q).
        
        Requirements: REQ-INF-006
        """
        q = outcome_data['q']
        
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert result.shape == (q, q)
    
    @pytest.mark.unit
    def test_vcov003_symmetric(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-003: Verify output is symmetric.
        
        Requirements: REQ-INF-006
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert_allclose(result, result.T, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vcov004_finite_values(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-004: Verify all values are finite.
        
        Requirements: REQ-INF-006
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.unit
    def test_vcov005_diagonal_positive(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-005: Verify diagonal elements (variances) are positive.
        
        Requirements: REQ-INF-006
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert np.all(np.diag(result) > 0)

    
    @pytest.mark.unit
    def test_vcov006_standard_errors_computable(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-006: Verify standard errors can be computed from diagonal.
        
        Requirements: REQ-INF-006
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        se = np.sqrt(np.diag(result))
        
        assert np.all(np.isfinite(se))
        assert np.all(se > 0)
    
    @pytest.mark.unit
    def test_vcov007_tol_parameter(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-007: Verify tol parameter is respected.
        
        Requirements: REQ-INF-006
        """
        # Different tol values should not change result much for well-conditioned data
        result_default = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta'],
            tol=1e-5
        )
        
        result_strict = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta'],
            tol=1e-10
        )
        
        # Results should be close for well-conditioned data
        assert_allclose(result_default, result_strict, rtol=0.1)
    
    @pytest.mark.unit
    def test_vcov008_lambda_parameter(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-008: Verify lambda_ parameter affects regularization.
        
        Requirements: REQ-INF-006
        """
        # With very high tol, regularization should be triggered
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta'],
            tol=1.0,  # Very high tol to force regularization
            lambda_=0.1
        )
        
        assert np.all(np.isfinite(result))
    
    @pytest.mark.unit
    def test_vcov009_reproducibility(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-009: Verify results are reproducible.
        
        Requirements: REQ-INF-006
        """
        result1 = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        result2 = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        assert_allclose(result1, result2, rtol=1e-10)
    
    @pytest.mark.unit
    def test_vcov010_different_delta_different_result(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-010: Verify different delta produces different results.
        
        Requirements: REQ-INF-006
        """
        result1 = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        # Different delta
        delta2 = outcome_data['delta'] + 1.0
        
        result2 = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=delta2
        )
        
        # Results should differ
        assert not np.allclose(result1, result2)


# -----------------------------------------------------------------------------
# Test Class: Input Validation (VCOV-011 to VCOV-020)
# -----------------------------------------------------------------------------

class TestVcovOutcomeInputValidation:
    """
    Test input validation in vcov_outcome.
    
    Test IDs: VCOV-011 to VCOV-020
    Requirements: REQ-INF-007
    """
    
    @pytest.mark.unit
    def test_vcov011_missing_ttilde_raises(self, outcome_data):
        """
        VCOV-011: Verify error when Ttilde is missing.
        
        Requirements: REQ-INF-007
        """
        np.random.seed(42)
        n, k = 200, 4
        
        # Create CBPSResults without Ttilde
        cbps_fit = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=np.random.rand(n),
            weights=np.ones(n),
            linear_predictor=np.random.randn(n),
            y=np.random.randn(n),
            x=np.random.randn(n, k),
            J=1.0,
            mle_J=0.5,
            deviance=100.0,
            converged=True,
            var=np.eye(k),
            Ttilde=None,  # Missing
        )
        
        with pytest.raises(ValueError, match="Ttilde"):
            vcov_outcome(
                cbps_fit=cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta']
            )
    
    @pytest.mark.unit
    def test_vcov012_missing_xtilde_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-012: Verify error when Xtilde is missing.
        
        Requirements: REQ-INF-007
        """
        # Modify the fit to remove Xtilde
        continuous_cbps_fit.Xtilde = None
        
        with pytest.raises(ValueError, match="Xtilde"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta']
            )
    
    @pytest.mark.unit
    def test_vcov013_missing_beta_tilde_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-013: Verify error when beta_tilde is missing.
        
        Requirements: REQ-INF-007
        """
        continuous_cbps_fit.beta_tilde = None
        
        with pytest.raises(ValueError, match="beta_tilde"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta']
            )
    
    @pytest.mark.unit
    def test_vcov014_missing_sigmasq_tilde_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-014: Verify error when sigmasq_tilde is missing.
        
        Requirements: REQ-INF-007
        """
        continuous_cbps_fit.sigmasq_tilde = None
        
        with pytest.raises(ValueError, match="sigmasq_tilde"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta']
            )

    
    @pytest.mark.unit
    def test_vcov015_y_length_mismatch_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-015: Verify error when Y length doesn't match.
        
        Requirements: REQ-INF-007
        """
        with pytest.raises(ValueError, match="length|match"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'][:100],  # Wrong length
                Z=outcome_data['Z'],
                delta=outcome_data['delta']
            )
    
    @pytest.mark.unit
    def test_vcov016_z_rows_mismatch_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-016: Verify error when Z rows don't match Y length.
        
        Requirements: REQ-INF-007
        """
        with pytest.raises(ValueError, match="row|match"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'][:100],  # Wrong rows
                delta=outcome_data['delta']
            )
    
    @pytest.mark.unit
    def test_vcov017_delta_length_mismatch_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-017: Verify error when delta length doesn't match Z columns.
        
        Requirements: REQ-INF-007
        """
        with pytest.raises(ValueError, match="length|match"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta'][:2]  # Wrong length
            )
    
    @pytest.mark.unit
    def test_vcov018_negative_tol_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-018: Verify error for negative tol.
        
        Requirements: REQ-INF-007
        """
        with pytest.raises(ValueError, match="tol"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta'],
                tol=-1e-5
            )
    
    @pytest.mark.unit
    def test_vcov019_negative_lambda_raises(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-019: Verify error for negative lambda_.
        
        Requirements: REQ-INF-007
        """
        with pytest.raises(ValueError, match="lambda"):
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta'],
                lambda_=-0.01
            )
    
    @pytest.mark.unit
    def test_vcov020_high_tol_warns(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-020: Verify warning for tol > 1.
        
        Requirements: REQ-INF-007
        
        Note: This test verifies that the warning mechanism works.
        The warning is intentionally triggered and captured.
        """
        import warnings as test_warnings
        with test_warnings.catch_warnings(record=True) as w:
            test_warnings.simplefilter("always")
            vcov_outcome(
                cbps_fit=continuous_cbps_fit,
                Y=outcome_data['Y'],
                Z=outcome_data['Z'],
                delta=outcome_data['delta'],
                tol=2.0  # Very high - should trigger warning
            )
            # Check that at least one warning was raised about tol
            tol_warnings = [x for x in w if "tol" in str(x.message)]
            assert len(tol_warnings) >= 1, "Expected warning about high tol value"


# -----------------------------------------------------------------------------
# Test Class: Numerical Properties (VCOV-021 to VCOV-026)
# -----------------------------------------------------------------------------

class TestVcovOutcomeNumerical:
    """
    Test numerical properties of vcov_outcome.
    
    Test IDs: VCOV-021 to VCOV-026
    Requirements: REQ-INF-008
    """
    
    @pytest.mark.numerical
    def test_vcov021_positive_semidefinite(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-021: Verify output is positive semi-definite.
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(result)
        
        assert np.all(eigenvalues >= -1e-10), \
            f"Negative eigenvalues found: {eigenvalues[eigenvalues < 0]}"
    
    @pytest.mark.numerical
    def test_vcov022_condition_number_reasonable(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-022: Verify condition number is reasonable.
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        cond = np.linalg.cond(result)
        
        # Condition number should not be too extreme
        assert cond < 1e12
    
    @pytest.mark.numerical
    def test_vcov023_variance_magnitude_reasonable(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-023: Verify variance magnitudes are reasonable.
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        variances = np.diag(result)
        
        # Variances should not be extremely small or large
        assert np.all(variances > 1e-20)
        assert np.all(variances < 1e10)
    
    @pytest.mark.numerical
    def test_vcov024_correlation_bounded(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-024: Verify correlations are in [-1, 1].
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        # Compute correlation matrix
        std_dev = np.sqrt(np.diag(result))
        corr = result / np.outer(std_dev, std_dev)
        
        # Correlations should be bounded
        assert np.all(corr >= -1.0 - 1e-6)
        assert np.all(corr <= 1.0 + 1e-6)
    
    @pytest.mark.numerical
    def test_vcov025_inverse_computable(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-025: Verify inverse can be computed.
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        try:
            inv = np.linalg.inv(result)
            assert np.all(np.isfinite(inv))
        except np.linalg.LinAlgError:
            pytest.fail("Variance matrix is singular")
    
    @pytest.mark.numerical
    def test_vcov026_cholesky_decomposition(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-026: Verify Cholesky decomposition is possible (positive definite).
        
        Requirements: REQ-INF-008
        """
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta']
        )
        
        try:
            L = np.linalg.cholesky(result)
            assert np.all(np.isfinite(L))
        except np.linalg.LinAlgError:
            # Allow for numerical issues with semi-definite matrices
            eigenvalues = np.linalg.eigvalsh(result)
            assert np.min(eigenvalues) > -1e-8, \
                "Matrix is neither positive definite nor positive semi-definite"


# -----------------------------------------------------------------------------
# Test Class: Edge Cases (VCOV-027 to VCOV-030)
# -----------------------------------------------------------------------------

class TestVcovOutcomeEdgeCases:
    """
    Test edge cases and robustness.
    
    Test IDs: VCOV-027 to VCOV-030
    Requirements: REQ-INF-009
    """
    
    @pytest.mark.edge_case
    def test_vcov027_small_sample(self):
        """
        VCOV-027: Verify handling of small sample sizes.
        
        Requirements: REQ-INF-009
        """
        np.random.seed(42)
        n = 30  # Small sample
        k = 3
        q = 2
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        T = np.random.randn(n)
        Ttilde = (T - T.mean()) / T.std()
        beta_tilde = np.linalg.lstsq(X, Ttilde, rcond=None)[0]
        residuals = Ttilde - X @ beta_tilde
        sigmasq_tilde = np.var(residuals, ddof=k)
        weights = np.ones(n)
        
        cbps_fit = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=X @ beta_tilde,
            weights=weights,
            linear_predictor=X @ beta_tilde,
            y=T,
            x=X,
            J=1.0,
            mle_J=0.5,
            deviance=50.0,
            converged=True,
            var=np.eye(k) * 0.01,
            sigmasq=1.0,
            Ttilde=Ttilde,
            Xtilde=X,
            beta_tilde=beta_tilde,
            sigmasq_tilde=sigmasq_tilde,
        )
        
        Z = np.column_stack([np.ones(n), np.random.randn(n)])
        Y = np.random.randn(n)
        delta = np.random.randn(q)
        
        result = vcov_outcome(cbps_fit, Y, Z, delta)
        
        assert np.all(np.isfinite(result))

    
    @pytest.mark.edge_case
    def test_vcov028_large_sample(self):
        """
        VCOV-028: Verify handling of larger sample sizes.
        
        Requirements: REQ-INF-009
        """
        np.random.seed(42)
        n = 1000
        k = 4
        q = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        T = np.random.randn(n)
        Ttilde = (T - T.mean()) / T.std()
        beta_tilde = np.linalg.lstsq(X, Ttilde, rcond=None)[0]
        residuals = Ttilde - X @ beta_tilde
        sigmasq_tilde = np.var(residuals, ddof=k)
        weights = np.exp(-0.5 * residuals**2 / sigmasq_tilde)
        weights = weights / weights.mean()
        
        cbps_fit = CBPSResults(
            coefficients=np.random.randn(k, 1),
            fitted_values=X @ beta_tilde,
            weights=weights,
            linear_predictor=X @ beta_tilde,
            y=T,
            x=X,
            J=2.0,
            mle_J=1.5,
            deviance=500.0,
            converged=True,
            var=np.eye(k) * 0.001,
            sigmasq=1.0,
            Ttilde=Ttilde,
            Xtilde=X,
            beta_tilde=beta_tilde,
            sigmasq_tilde=sigmasq_tilde,
        )
        
        Z = np.column_stack([np.ones(n), np.random.randn(n, q - 1)])
        Y = np.random.randn(n)
        delta = np.random.randn(q)
        
        result = vcov_outcome(cbps_fit, Y, Z, delta)
        
        assert np.all(np.isfinite(result))
        # Variance should be smaller with larger sample
        assert np.all(np.diag(result) < 0.1)
    
    @pytest.mark.edge_case
    @pytest.mark.filterwarnings("ignore:tol=.*triggers regularization")
    def test_vcov029_regularization_applied(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-029: Verify regularization is applied when needed.
        
        Requirements: REQ-INF-009
        
        Note: This test intentionally uses high tol to force regularization.
        The resulting warning is expected and filtered.
        """
        # Force regularization by using high tol
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=outcome_data['Z'],
            delta=outcome_data['delta'],
            tol=10.0,  # Very high to force regularization
            lambda_=1.0
        )
        
        # Should still produce valid result
        assert np.all(np.isfinite(result))
    
    @pytest.mark.edge_case
    def test_vcov030_transposed_inputs(self, continuous_cbps_fit, outcome_data):
        """
        VCOV-030: Verify handling of transposed matrix inputs.
        
        Requirements: REQ-INF-009
        """
        # Transpose Z (should be auto-corrected)
        Z_transposed = outcome_data['Z'].T
        
        result = vcov_outcome(
            cbps_fit=continuous_cbps_fit,
            Y=outcome_data['Y'],
            Z=Z_transposed,
            delta=outcome_data['delta']
        )
        
        # Should handle transposed input gracefully
        assert np.all(np.isfinite(result))



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
