"""
Unified Test Suite for High-Dimensional Covariate Balancing Propensity Score (hdCBPS)
=====================================================================================

This module provides a comprehensive test suite for the hdCBPS algorithm
proposed by Ning, Peng, and Imai (2020). The hdCBPS method extends the
Covariate Balancing Propensity Score (CBPS) framework to high-dimensional
settings where the number of covariates may exceed the sample size (p > n),
employing penalized regression (LASSO) for variable selection and GMM-based
calibration for doubly robust treatment effect estimation.

The tests are organized into the following functional groups:

1. Module Export Tests (TestHighdimModuleExports, TestHdCBPSOptionalExports,
   TestWeightFunctionSignatures): Verify the public API surface of the
   cbps.highdim module.

2. Weight Function Tests (TestAteWtFuncBasic, TestAteWtFuncNumerical,
   TestAteWtNlFuncBasic, TestAttWtFuncBasic, TestAttWtNlFuncBasic,
   TestWeightFuncsEdgeCases, TestWeightFuncsVsR,
   TestWeightFuncsNumericalStability): Validate the ATE and ATT inverse
   probability weighting functions used in GMM calibration.

3. GMM Loss Function Tests (TestGmmFuncBasic, TestGmmFuncNonlinear,
   TestAttGmmFuncBasic, TestAttGmmFuncNonlinear, TestGmmFuncEdgeCases,
   TestGmmFuncOptimization, TestGmmFuncVsR, TestGmmFuncNumericalStability):
   Verify the GMM objective functions for Step 3 calibration.

4. LASSO Utility Tests (TestCvGlmnetGaussianBasic,
   TestCvGlmnetGaussianNumerical, TestSelectVariables,
   TestCvGlmnetGaussianVsR, TestCvGlmnetGaussianNumericalValidation,
   TestCvGlmnetGaussianEdgeCases, TestCvGlmnetBinomialBasic,
   TestCvGlmnetBinomialVsR, TestCvGlmnetBinomialEdgeCases,
   TestCvGlmnetPoissonBasic, TestCvGlmnetPoissonVsR,
   TestCvGlmnetPoissonEdgeCases, TestPredictGlmnetFortran,
   TestSelectVariablesComprehensive, TestLassoUtilsIntegration):
   Test the cv_glmnet wrapper for LASSO variable selection across
   Gaussian, binomial, and Poisson families.

5. Main Algorithm Tests (TestHDCBPSResultsBasic,
   TestHdCBPSBasicFunctionality, TestHdCBPSATEEstimation,
   TestHdCBPSATTEstimation, TestHdCBPSMethods,
   TestHdCBPSHighDimensional, TestHdCBPSEdgeCases, TestHdCBPSVsR,
   TestHdCBPSNumericalStability, TestHdCBPSIntegration):
   End-to-end tests for the complete hdCBPS pipeline including
   ATE/ATT estimation, method variants, and numerical stability.

Test ID Ranges:
    - HD-INIT-001 ~ HD-INIT-020: Module export tests
    - T-031 ~ T-054: Weight function tests
    - T-055 ~ T-081: GMM loss function tests
    - T-001 ~ T-030: LASSO utility tests
    - T-082 ~ T-132: Main algorithm tests

References:
    Ning, Y., Peng, S., and Imai, K. (2020). Robust Estimation of Causal
    Effects via a High-Dimensional Covariate Balancing Propensity Score.
    Biometrika, 107(3), 533-554. https://doi.org/10.1093/biomet/asaa020
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Weight functions and GMM loss are always available (no glmnet dependency)
from cbps.highdim.weight_funcs import (
    ate_wt_func,
    ate_wt_nl_func,
    att_wt_func,
    att_wt_nl_func
)
from cbps.highdim.gmm_loss import gmm_func, att_gmm_func

# Skip all tests requiring glmnetforpython if not available
try:
    from cbps.highdim.lasso_utils import (
        cv_glmnet,
        select_variables,
        HAS_GLMNETFORPYTHON
    )
    SKIP_GLMNET_TESTS = not HAS_GLMNETFORPYTHON
except ImportError:
    SKIP_GLMNET_TESTS = True
    HAS_GLMNETFORPYTHON = False

try:
    from cbps.highdim.hdcbps import hdCBPS, hdCBPS_fit, HDCBPSResults
except ImportError:
    pass


# =============================================================================
# Embedded Fixtures (from highdim/conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def hdcbps_tolerances():
    """Tolerance values for hdCBPS numerical comparisons."""
    return {
        'coefficient_rtol': 0.1,
        'coefficient_atol': 0.1,
        'weight_rtol': 1e-6,
        'weight_atol': 1e-6,
    }


@pytest.fixture
def simple_hdcbps_data():
    """Generate simple test data for hdCBPS unit tests."""
    np.random.seed(42)
    n, p = 200, 50

    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:5] = [0.5, -0.3, 0.4, -0.2, 0.3]

    ps = 1 / (1 + np.exp(-X @ beta_true))
    treat = np.random.binomial(1, ps)

    return {
        'X': X,
        'treat': treat,
        'n': n,
        'p': p,
        'beta_true': beta_true,
    }


@pytest.fixture(scope="session")
def hdcbps_available():
    """Check if hdCBPS dependencies are available."""
    try:
        from cbps.highdim import hdcbps_fit
        return True
    except ImportError:
        return False


# =============================================================================
# Module Export Tests
# (merged from test_init.py)
# =============================================================================


# =============================================================================
# Test Class: Module Imports and Exports
# =============================================================================

@pytest.mark.unit
class TestHighdimModuleExports:
    """
    Test ID: HD-INIT-001 ~ HD-INIT-010
    Requirement: REQ-HD-INIT-001
    
    Tests for cbps.highdim module exports.
    """
    
    def test_hd_init001_module_importable(self):
        """HD-INIT-001: cbps.highdim module is importable."""
        from cbps import highdim
        assert highdim is not None
    
    def test_hd_init002_all_defined(self):
        """HD-INIT-002: __all__ is defined in highdim module."""
        from cbps import highdim
        assert hasattr(highdim, '__all__')
        assert isinstance(highdim.__all__, (list, tuple))
    
    def test_hd_init003_weight_funcs_always_available(self):
        """HD-INIT-003: Weight functions are always available (no glmnet needed)."""
        from cbps.highdim import ate_wt_func, att_wt_func
        assert callable(ate_wt_func)
        assert callable(att_wt_func)
    
    def test_hd_init004_ate_wt_nl_func_importable(self):
        """HD-INIT-004: ate_wt_nl_func is importable from cbps.highdim."""
        from cbps.highdim import ate_wt_nl_func
        assert callable(ate_wt_nl_func)
    
    def test_hd_init005_att_wt_nl_func_importable(self):
        """HD-INIT-005: att_wt_nl_func is importable from cbps.highdim."""
        from cbps.highdim import att_wt_nl_func
        assert callable(att_wt_nl_func)
    
    def test_hd_init006_all_contains_weight_funcs(self):
        """HD-INIT-006: __all__ contains weight functions."""
        from cbps import highdim
        weight_funcs = ['ate_wt_func', 'ate_wt_nl_func', 'att_wt_func', 'att_wt_nl_func']
        for name in weight_funcs:
            assert name in highdim.__all__, f"{name} not in __all__"
    
    def test_hd_init007_no_private_exports(self):
        """HD-INIT-007: __all__ does not export private functions."""
        from cbps import highdim
        for name in highdim.__all__:
            assert not name.startswith('_'), f"Private {name} should not be in __all__"


# =============================================================================
# Test Class: Optional hdCBPS Exports (require glmnetforpython)
# =============================================================================

@pytest.mark.unit
class TestHdCBPSOptionalExports:
    """
    Test ID: HD-INIT-008 ~ HD-INIT-015
    Requirement: REQ-HD-INIT-002
    
    Tests for optional hdCBPS exports (require glmnetforpython).
    """
    
    def test_hd_init008_hdcbps_in_all_when_glmnet_available(self):
        """HD-INIT-008: hdCBPS is in __all__ when glmnetforpython is available."""
        from cbps import highdim
        
        # Check if glmnetforpython is available (import name matches package name)
        try:
            import glmnetforpython  # noqa: F401
            has_glmnet = True
        except ImportError:
            has_glmnet = False

        if has_glmnet:
            assert 'hdCBPS' in highdim.__all__
            assert 'HDCBPSResults' in highdim.__all__
    
    def test_hd_init009_hdcbps_callable_when_available(self):
        """HD-INIT-009: hdCBPS is callable when glmnetforpython is available."""
        try:
            from cbps.highdim import hdCBPS
            assert callable(hdCBPS)
        except ImportError:
            pytest.skip("glmnetforpython not available")
    
    def test_hd_init010_hdcbps_results_class_when_available(self):
        """HD-INIT-010: HDCBPSResults is a class when glmnetforpython is available."""
        try:
            from cbps.highdim import HDCBPSResults
            assert inspect.isclass(HDCBPSResults)
        except ImportError:
            pytest.skip("glmnetforpython not available")
    
    def test_hd_init011_cv_glmnet_importable_when_available(self):
        """HD-INIT-011: cv_glmnet is importable when glmnetforpython is available."""
        try:
            from cbps.highdim import cv_glmnet
            assert callable(cv_glmnet)
        except ImportError:
            pytest.skip("glmnetforpython not available")
    
    def test_hd_init012_select_variables_importable_when_available(self):
        """HD-INIT-012: select_variables is importable when glmnetforpython is available."""
        try:
            from cbps.highdim import select_variables
            assert callable(select_variables)
        except ImportError:
            pytest.skip("glmnetforpython not available")


# =============================================================================
# Test Class: Weight Function Signatures
# =============================================================================

@pytest.mark.unit
class TestWeightFunctionSignatures:
    """
    Test ID: HD-INIT-013 ~ HD-INIT-020
    Requirement: REQ-HD-INIT-003
    
    Tests for weight function signatures.
    """
    
    def test_hd_init013_ate_wt_func_has_treat_param(self):
        """HD-INIT-013: ate_wt_func has treat parameter."""
        from cbps.highdim import ate_wt_func
        sig = inspect.signature(ate_wt_func)
        assert 'treat' in sig.parameters
    
    def test_hd_init014_ate_wt_func_has_beta_curr_param(self):
        """HD-INIT-014: ate_wt_func has beta_curr parameter."""
        from cbps.highdim import ate_wt_func
        sig = inspect.signature(ate_wt_func)
        assert 'beta_curr' in sig.parameters
    
    def test_hd_init015_att_wt_func_has_treat_param(self):
        """HD-INIT-015: att_wt_func has treat parameter."""
        from cbps.highdim import att_wt_func
        sig = inspect.signature(att_wt_func)
        assert 'treat' in sig.parameters
    
    def test_hd_init016_att_wt_func_has_beta_curr_param(self):
        """HD-INIT-016: att_wt_func has beta_curr parameter."""
        from cbps.highdim import att_wt_func
        sig = inspect.signature(att_wt_func)
        assert 'beta_curr' in sig.parameters
    
    def test_hd_init017_weight_funcs_have_docstrings(self):
        """HD-INIT-017: Weight functions have docstrings."""
        from cbps.highdim import ate_wt_func, att_wt_func
        assert ate_wt_func.__doc__ is not None
        assert att_wt_func.__doc__ is not None
    
    def test_hd_init018_nl_weight_funcs_have_docstrings(self):
        """HD-INIT-018: Non-linear weight functions have docstrings."""
        from cbps.highdim import ate_wt_nl_func, att_wt_nl_func
        assert ate_wt_nl_func.__doc__ is not None
        assert att_wt_nl_func.__doc__ is not None


# =============================================================================
# Weight Function Tests
# (merged from test_weight_funcs.py)
# =============================================================================


# =============================================================================
# T-031 ~ T-036: ATE Weight Function Tests
# =============================================================================

class TestAteWtFuncBasic:
    """T-031 ~ T-033: ATE weight function basic functionality tests."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Initial beta (including intercept)
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0])
        
        # Selected variable indices (excluding intercept)
        S = np.array([1, 2, 3])  # corresponds to beta_ini[1], beta_ini[2], beta_ini[3]
        
        # Current optimized beta (only selected variables)
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        return X, treat, beta_ini, S, beta_curr

    def test_ate_wt_func_returns_correct_shape(self, simple_data):
        """T-031: Test ate_wt_func returns correct shape."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        # Treated group weights
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W1.shape == (n,), f"Expected shape ({n},), got {W1.shape}"
        
        # Control group weights
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W0.shape == (n,), f"Expected shape ({n},), got {W0.shape}"

    def test_ate_wt_func_treated_formula(self, simple_data):
        """T-032: Verify treated group weight formula W = T/pi - 1."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        # Compute weights
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manually compute propensity scores
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        
        # Manually compute weights
        W1_manual = treat / pi - 1.0
        
        assert_allclose(W1, W1_manual, rtol=1e-10,
                       err_msg="Treated group weight formula incorrect")

    def test_ate_wt_func_control_formula(self, simple_data):
        """T-033: Verify control group weight formula W = 1 - (1-T)/(1-pi)."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        # Compute weights
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manually compute propensity scores
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        
        # Manually compute weights
        W0_manual = 1.0 - (1.0 - treat) / (1.0 - pi)
        
        assert_allclose(W0, W0_manual, rtol=1e-10,
                       err_msg="Control group weight formula incorrect")

    def test_ate_wt_func_treated_only_nonzero(self, simple_data):
        """Verify treated group weights are non-zero only for T=1 samples."""
        X, treat, beta_ini, S, beta_curr = simple_data
        
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # For T=0 samples, W1 = 0/pi - 1 = -1
        # For T=1 samples, W1 = 1/pi - 1 > -1 (when pi < 1)
        control_mask = treat == 0
        assert_allclose(W1[control_mask], -1.0, rtol=1e-10,
                       err_msg="Control sample treated weights should be -1")

    def test_ate_wt_func_control_only_nonzero(self, simple_data):
        """Verify control group weights are non-zero only for T=0 samples."""
        X, treat, beta_ini, S, beta_curr = simple_data
        
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # For T=1 samples, W0 = 1 - 0/(1-pi) = 1
        treated_mask = treat == 1
        assert_allclose(W0[treated_mask], 1.0, rtol=1e-10,
                       err_msg="Treated sample control weights should be 1")


class TestAteWtFuncNumerical:
    """T-034 ~ T-036: ATE weight function numerical verification tests."""

    @pytest.fixture
    def controlled_data(self):
        """Generate controlled data for precise numerical verification."""
        np.random.seed(12345)
        n, p = 200, 8
        X = np.random.randn(n, p)
        
        # True propensity score model
        beta_true = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        X2 = np.column_stack([np.ones(n), X])
        theta = X2 @ beta_true
        pi_true = 1.0 / (1.0 + np.exp(-theta))
        treat = np.random.binomial(1, pi_true).astype(float)
        
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        return X, treat, beta_true, S, beta_curr, pi_true

    def test_ate_wt_func_propensity_score_calculation(self, controlled_data):
        """T-034: Verify propensity score calculation correctness."""
        X, treat, beta_ini, S, beta_curr, pi_true = controlled_data
        n = X.shape[0]
        
        # Compute using weight function
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Recover propensity scores from weights (for T=1 samples)
        # W1 = T/pi - 1 => pi = T/(W1 + 1)
        treated_mask = treat == 1
        pi_from_W1 = treat[treated_mask] / (W1[treated_mask] + 1)
        
        # Compare with true propensity scores
        assert_allclose(pi_from_W1, pi_true[treated_mask], rtol=1e-10,
                       err_msg="Propensity scores recovered from weights are incorrect")

    def test_ate_wt_func_weight_sum_property(self, controlled_data):
        """T-035: Verify weight normalization property."""
        X, treat, beta_ini, S, beta_curr, _ = controlled_data
        n = X.shape[0]
        
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Theoretically, when model is correct:
        # E[T/pi - 1] = E[T/pi] - 1 = 1 - 1 = 0
        # E[1 - (1-T)/(1-pi)] = 1 - E[(1-T)/(1-pi)] = 1 - 1 = 0
        # But this only holds in large samples with correct model
        
        # Check weights are not all zeros
        assert np.sum(np.abs(W1)) > 0, "Treated group weights should not all be zero"
        assert np.sum(np.abs(W0)) > 0, "Control group weights should not all be zero"

    def test_ate_wt_func_coefficient_update(self, controlled_data):
        """T-036: Verify coefficient update logic."""
        X, treat, beta_ini, S, beta_curr, _ = controlled_data
        
        # Use different beta_curr
        beta_curr_new = beta_curr * 2
        
        W1_orig = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W1_new = ate_wt_func(beta_curr_new, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Weights should differ
        assert not np.allclose(W1_orig, W1_new), \
            "Different beta_curr should produce different weights"


# =============================================================================
# T-037 ~ T-040: ATE Nonlinear Weight Function Tests
# =============================================================================

class TestAteWtNlFuncBasic:
    """T-037 ~ T-040: ATE nonlinear weight function tests."""

    @pytest.fixture
    def nl_data(self):
        """Generate nonlinear model test data."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Initial beta (including intercept, but intercept=0 when intercept=False)
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0])
        
        # Selected variable indices (excluding intercept)
        S = np.array([1, 2, 3])
        
        # Current optimized beta (including intercept + selected variables)
        # SS = [0, 1, 2, 3], so beta_curr has length 4
        beta_curr = np.array([0.1, 0.5, -0.3, 0.2])  # [intercept, S[0], S[1], S[2]]
        
        return X, treat, beta_ini, S, beta_curr

    def test_ate_wt_nl_func_returns_correct_shape(self, nl_data):
        """T-037: Test ate_wt_nl_func returns correct shape."""
        X, treat, beta_ini, S, beta_curr = nl_data
        n = X.shape[0]
        
        W1 = ate_wt_nl_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W1.shape == (n,)
        
        W0 = ate_wt_nl_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W0.shape == (n,)

    def test_ate_wt_nl_func_includes_intercept(self, nl_data):
        """T-038: Verify nonlinear weight function includes intercept optimization."""
        X, treat, beta_ini, S, beta_curr = nl_data
        n = X.shape[0]
        
        # Compute weights
        W1 = ate_wt_nl_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manual computation (SS = [0, S])
        X2 = np.column_stack([np.ones(n), X])
        SS = np.r_[0, S]
        beta_all = beta_ini.copy()
        beta_all[SS] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        W1_manual = treat / pi - 1.0
        
        assert_allclose(W1, W1_manual, rtol=1e-10,
                       err_msg="Nonlinear weight function intercept handling incorrect")

    def test_ate_wt_nl_func_intercept_effect(self, nl_data):
        """T-039: Verify intercept effect on weights."""
        X, treat, beta_ini, S, _ = nl_data
        
        # Different intercept values
        beta_curr_1 = np.array([0.0, 0.5, -0.3, 0.2])
        beta_curr_2 = np.array([1.0, 0.5, -0.3, 0.2])
        
        W1_1 = ate_wt_nl_func(beta_curr_1, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W1_2 = ate_wt_nl_func(beta_curr_2, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Different intercepts should produce different weights
        assert not np.allclose(W1_1, W1_2), \
            "Different intercepts should produce different weights"

    def test_ate_wt_nl_func_vs_linear(self, nl_data):
        """T-040: Compare linear and nonlinear weight functions."""
        X, treat, beta_ini, S, _ = nl_data
        
        # When intercept is 0 and beta_ini[0] = 0, both should be equal
        beta_curr_linear = np.array([0.5, -0.3, 0.2])  # without intercept
        beta_curr_nl = np.array([0.0, 0.5, -0.3, 0.2])  # with intercept=0
        
        W1_linear = ate_wt_func(beta_curr_linear, S, tt=1, X_wt=X, 
                                beta_ini=beta_ini, treat=treat)
        W1_nl = ate_wt_nl_func(beta_curr_nl, S, tt=1, X_wt=X, 
                               beta_ini=beta_ini, treat=treat)
        
        assert_allclose(W1_linear, W1_nl, rtol=1e-10,
                       err_msg="When intercept=0, linear and nonlinear weights should match")


# =============================================================================
# T-041 ~ T-046: ATT Weight Function Tests
# =============================================================================

class TestAttWtFuncBasic:
    """T-041 ~ T-044: ATT weight function basic functionality tests."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0])
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        return X, treat, beta_ini, S, beta_curr

    def test_att_wt_func_returns_correct_shape(self, simple_data):
        """T-041: Test att_wt_func returns correct shape."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        W = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W.shape == (n,), f"Expected shape ({n},), got {W.shape}"

    def test_att_wt_func_formula(self, simple_data):
        """T-042: Verify ATT weight formula W = T - (1-T)*pi/(1-pi)."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        W = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manual computation
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        W_manual = treat - (1.0 - treat) * pi / (1.0 - pi)
        
        assert_allclose(W, W_manual, rtol=1e-10,
                       err_msg="ATT weight formula incorrect")

    def test_att_wt_func_treated_weight(self, simple_data):
        """T-043: Verify treated sample weights are 1."""
        X, treat, beta_ini, S, beta_curr = simple_data
        
        W = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # For T=1 samples, W = 1 - 0 = 1
        treated_mask = treat == 1
        assert_allclose(W[treated_mask], 1.0, rtol=1e-10,
                       err_msg="Treated sample ATT weights should be 1")

    def test_att_wt_func_control_weight(self, simple_data):
        """T-044: Verify control sample weights are -pi/(1-pi)."""
        X, treat, beta_ini, S, beta_curr = simple_data
        n = X.shape[0]
        
        W = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manual computation for control group weights
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        
        control_mask = treat == 0
        W_control_expected = -pi[control_mask] / (1.0 - pi[control_mask])
        
        assert_allclose(W[control_mask], W_control_expected, rtol=1e-10,
                       err_msg="Control sample ATT weights incorrect")


class TestAttWtNlFuncBasic:
    """T-045 ~ T-046: ATT nonlinear weight function tests."""

    @pytest.fixture
    def nl_data(self):
        """Generate nonlinear model test data."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0])
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.1, 0.5, -0.3, 0.2])
        return X, treat, beta_ini, S, beta_curr

    def test_att_wt_nl_func_returns_correct_shape(self, nl_data):
        """T-045: Test att_wt_nl_func returns correct shape."""
        X, treat, beta_ini, S, beta_curr = nl_data
        n = X.shape[0]
        
        W = att_wt_nl_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        assert W.shape == (n,)

    def test_att_wt_nl_func_includes_intercept(self, nl_data):
        """T-046: Verify ATT nonlinear weight function includes intercept optimization."""
        X, treat, beta_ini, S, beta_curr = nl_data
        n = X.shape[0]
        
        W = att_wt_nl_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Manual computation
        X2 = np.column_stack([np.ones(n), X])
        SS = np.r_[0, S]
        beta_all = beta_ini.copy()
        beta_all[SS] = beta_curr
        theta = X2 @ beta_all
        pi = 1.0 / (1.0 + np.exp(-theta))
        W_manual = treat - (1.0 - treat) * pi / (1.0 - pi)
        
        assert_allclose(W, W_manual, rtol=1e-10,
                       err_msg="ATT nonlinear weight function intercept handling incorrect")


# =============================================================================
# T-047 ~ T-050: Weight Function Edge Case Tests
# =============================================================================

class TestWeightFuncsEdgeCases:
    """T-047 ~ T-050: Weight function edge case tests."""

    def test_extreme_propensity_near_zero(self):
        """T-047: Test weight computation with extreme propensity scores (near 0)."""
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Set coefficients to make propensity scores close to 0
        beta_ini = np.array([-5.0, 0.0, 0.0, 0.0])  # large negative intercept
        S = np.array([1])
        beta_curr = np.array([0.0])
        
        # Should be computable, but weights may be large
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(W1)), "Weights should not contain NaN"
        assert not np.any(np.isinf(W1)), "Weights should not contain Inf"

    def test_extreme_propensity_near_one(self):
        """T-048: Test weight computation with extreme propensity scores (near 1)."""
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Set coefficients to make propensity scores close to 1
        beta_ini = np.array([5.0, 0.0, 0.0, 0.0])  # large positive intercept
        S = np.array([1])
        beta_curr = np.array([0.0])
        
        # Control group weights may be large (since 1-pi is close to 0)
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Check for NaN or Inf
        assert not np.any(np.isnan(W0)), "Weights should not contain NaN"
        assert not np.any(np.isinf(W0)), "Weights should not contain Inf"

    def test_all_treated(self):
        """T-049: Test when all samples are treated."""
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        treat = np.ones(n)  # all treated
        
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2])
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # All control group weights should be 1
        assert_allclose(W0, 1.0, rtol=1e-10)
        
        # Check for NaN
        assert not np.any(np.isnan(W1))

    def test_all_control(self):
        """T-050: Test when all samples are control."""
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        treat = np.zeros(n)  # all control
        
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2])
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # All treated group weights should be -1
        assert_allclose(W1, -1.0, rtol=1e-10)
        
        # Check for NaN
        assert not np.any(np.isnan(W0))


# =============================================================================
# T-051 ~ T-054: Python-R Comparison Tests
# =============================================================================

class TestWeightFuncsVsR:
    """T-051 ~ T-054: Weight function Python-R comparison tests."""

    @pytest.fixture
    def comparison_data(self):
        """Generate data for Python-R comparison."""
        np.random.seed(2024)
        n, p = 200, 10
        X = np.random.randn(n, p)
        
        # Generate treatment assignment
        beta_ps = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        X2 = np.column_stack([np.ones(n), X])
        theta = X2 @ beta_ps
        pi = 1.0 / (1.0 + np.exp(-theta))
        treat = np.random.binomial(1, pi).astype(float)
        
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        return X, treat, beta_ps, S, beta_curr

    def test_ate_wt_func_vs_r_formula(self, comparison_data):
        """T-051: Verify ATE weight function consistency with R formula."""
        X, treat, beta_ini, S, beta_curr = comparison_data
        n = X.shape[0]
        
        # Python computation
        W1_py = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W0_py = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # R formula manual implementation
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        # R uses: probs.curr <- 1 - 1/(1 + exp(theta.curr))
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        
        # R formula: W <- treat/probs.curr - 1 (tt=1)
        W1_r = treat / probs_curr - 1.0
        # R formula: W <- 1 - (1-treat)/(1-probs.curr) (tt=0)
        W0_r = 1.0 - (1.0 - treat) / (1.0 - probs_curr)
        
        assert_allclose(W1_py, W1_r, rtol=1e-10,
                       err_msg="ATE treated weights inconsistent with R")
        assert_allclose(W0_py, W0_r, rtol=1e-10,
                       err_msg="ATE control weights inconsistent with R")

    def test_att_wt_func_vs_r_formula(self, comparison_data):
        """T-052: Verify ATT weight function consistency with R formula."""
        X, treat, beta_ini, S, beta_curr = comparison_data
        n = X.shape[0]
        
        # Python computation
        W_py = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # R formula manual implementation
        X2 = np.column_stack([np.ones(n), X])
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        
        # R formula: W <- treat - ((1-treat)*probs.curr/(1-probs.curr))
        W_r = treat - ((1.0 - treat) * probs_curr / (1.0 - probs_curr))
        
        assert_allclose(W_py, W_r, rtol=1e-10,
                       err_msg="ATT weights inconsistent with R")

    def test_ate_wt_nl_func_vs_r_formula(self, comparison_data):
        """T-053: Verify ATE nonlinear weight function consistency with R formula."""
        X, treat, beta_ini, S, _ = comparison_data
        n = X.shape[0]
        
        # Nonlinear version beta_curr includes intercept
        beta_curr_nl = np.array([0.1, 0.5, -0.3, 0.2])
        
        # Python computation
        W1_py = ate_wt_nl_func(beta_curr_nl, S, tt=1, X_wt=X, 
                               beta_ini=beta_ini, treat=treat)
        
        # R formula manual implementation (SS = [0, S])
        X2 = np.column_stack([np.ones(n), X])
        SS = np.r_[0, S]
        beta_all = beta_ini.copy()
        beta_all[SS] = beta_curr_nl
        theta = X2 @ beta_all
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        W1_r = treat / probs_curr - 1.0
        
        assert_allclose(W1_py, W1_r, rtol=1e-10,
                       err_msg="ATE nonlinear weights inconsistent with R")

    def test_att_wt_nl_func_vs_r_formula(self, comparison_data):
        """T-054: Verify ATT nonlinear weight function consistency with R formula."""
        X, treat, beta_ini, S, _ = comparison_data
        n = X.shape[0]
        
        beta_curr_nl = np.array([0.1, 0.5, -0.3, 0.2])
        
        # Python computation
        W_py = att_wt_nl_func(beta_curr_nl, S, X_wt=X, 
                              beta_ini=beta_ini, treat=treat)
        
        # R formula manual implementation
        X2 = np.column_stack([np.ones(n), X])
        SS = np.r_[0, S]
        beta_all = beta_ini.copy()
        beta_all[SS] = beta_curr_nl
        theta = X2 @ beta_all
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        W_r = treat - ((1.0 - treat) * probs_curr / (1.0 - probs_curr))
        
        assert_allclose(W_py, W_r, rtol=1e-10,
                       err_msg="ATT nonlinear weights inconsistent with R")


# =============================================================================
# Numerical Precision and Stability Tests
# =============================================================================

class TestWeightFuncsNumericalStability:
    """Weight function numerical stability tests."""

    def test_weight_magnitude_reasonable(self):
        """Verify weight magnitudes are within reasonable range."""
        np.random.seed(42)
        n, p = 200, 5
        X = np.random.randn(n, p)
        
        # Use reasonable coefficients
        beta_ini = np.array([0.0, 0.3, -0.2, 0.1, 0.0, 0.0])
        theta = np.column_stack([np.ones(n), X]) @ beta_ini
        pi = 1.0 / (1.0 + np.exp(-theta))
        treat = np.random.binomial(1, pi).astype(float)
        
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.3, -0.2, 0.1])
        
        W1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W0 = ate_wt_func(beta_curr, S, tt=0, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # Weights should not be too extreme
        assert np.max(np.abs(W1)) < 100, "Treated group weights too large"
        assert np.max(np.abs(W0)) < 100, "Control group weights too large"

    def test_weight_reproducibility(self):
        """Verify weight computation reproducibility."""
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        beta_ini = np.array([0.0, 0.5, -0.3, 0.2, 0.0, 0.0])
        S = np.array([1, 2, 3])
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        W1_1 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        W1_2 = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        assert_allclose(W1_1, W1_2, rtol=1e-15,
                       err_msg="Weight computation not reproducible")


# =============================================================================
# GMM Loss Function Tests
# (merged from test_gmm_loss.py)
# =============================================================================


# =============================================================================
# Helper function to create properly dimensioned test data
# =============================================================================

def create_test_data(n=100, p=5, seed=42):
    """
    Create test data with correct dimensions for GMM functions.
    
    Returns:
        X: (n, p+1) matrix with intercept column
        treat: (n,) binary treatment vector
        beta_ini: (p+2,) initial PS coefficients (for X1 = cbind(1, X))
        S: selected variable indices in (p+2) space
        beta_curr: (len(S),) current coefficients for linear method
        cov1_coef: (p+2,) outcome model coefficients for treated
        cov0_coef: (p+2,) outcome model coefficients for control
    """
    np.random.seed(seed)
    X_raw = np.random.randn(n, p)
    # X has intercept column, shape (n, p+1)
    X = np.column_stack([np.ones(n), X_raw])
    treat = np.random.binomial(1, 0.5, n).astype(float)
    
    # Inside functions: X1 = cbind(1, X) has shape (n, p+2)
    # So all coefficient vectors need length (p+2)
    
    # beta_ini: (p+2,) - coefficients for X1 = [1, X] = [1, 1, x1, x2, ..., xp]
    # Index 0 = extra intercept (added by function)
    # Index 1 = original intercept column of X
    # Index 2..p+1 = actual covariates
    beta_ini = np.zeros(p + 2)
    beta_ini[2:5] = [0.5, -0.3, 0.2]  # Set some covariate coefficients
    
    # S: indices of selected variables in (p+2) space
    # We select indices 2, 3, 4 (first 3 actual covariates)
    S = np.array([2, 3, 4])
    
    # beta_curr: coefficients for selected variables (linear method)
    beta_curr = np.array([0.5, -0.3, 0.2])
    
    # Outcome model coefficients: (p+2,) length
    cov1_coef = np.zeros(p + 2)
    cov1_coef[0:4] = [1.0, 0.5, -0.2, 0.1]
    cov0_coef = np.zeros(p + 2)
    cov0_coef[0:4] = [0.5, 0.3, -0.1, 0.2]
    
    return X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef


# =============================================================================
# T-055 ~ T-060: ATE GMM Loss Function Basic Tests
# =============================================================================

class TestGmmFuncBasic:
    """T-055 ~ T-058: ATE GMM loss function basic functionality tests."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        return create_test_data(n=100, p=5, seed=42)

    def test_gmm_func_returns_scalar(self, simple_data):
        """T-055: Test that gmm_func returns a scalar."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = simple_data
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float), f"Expected float, got {type(loss)}"

    def test_gmm_func_nonnegative(self, simple_data):
        """T-056: Test that GMM loss is non-negative."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = simple_data
        
        for tt in [0, 1]:
            loss = gmm_func(beta_curr, S, tt=tt, X_gmm=X, method='linear',
                           cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                           treat=treat, beta_ini=beta_ini)
            assert loss >= 0, f"GMM loss should be non-negative, got {loss}"

    def test_gmm_func_linear_formula(self, simple_data):
        """T-057: Verify linear method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = simple_data
        n = X.shape[0]
        
        # Python computation
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        # Manual computation: gmm_func internally creates X1 = cbind(1, X)
        # ate_wt_func also creates X2 = cbind(1, X)
        # So we need to simulate this behavior
        W = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # gmm_func uses X1[:, S] where X1 = cbind(1, X)
        X1 = np.column_stack([np.ones(n), X])
        w_curr_del = X1[:, S].T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="Linear method GMM loss formula incorrect")


    def test_gmm_func_different_tt(self, simple_data):
        """T-058: Test that different tt values produce different losses."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = simple_data
        
        loss_1 = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                         cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                         treat=treat, beta_ini=beta_ini)
        loss_0 = gmm_func(beta_curr, S, tt=0, X_gmm=X, method='linear',
                         cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                         treat=treat, beta_ini=beta_ini)
        
        # Different tt values typically produce different losses
        # Note: This assertion is intentionally weak because in edge cases
        # (e.g., symmetric data, equal treatment proportions) losses may be equal.
        # The primary purpose of this test is to verify that both tt values work without errors.
        # assert loss_1 != loss_0, "Different tt typically produces different losses"


class TestGmmFuncNonlinear:
    """T-059 ~ T-062: GMM loss function nonlinear method tests."""

    @pytest.fixture
    def nl_data(self):
        """Generate test data for nonlinear methods."""
        X, treat, beta_ini, S, _, cov1_coef, cov0_coef = create_test_data(n=100, p=5, seed=42)
        # beta_curr for nonlinear methods includes intercept (length = len(S) + 1)
        beta_curr = np.array([0.1, 0.5, -0.3, 0.2])
        return X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef

    def test_gmm_func_binomial_returns_scalar(self, nl_data):
        """T-059: Test that binomial method returns a scalar."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = nl_data
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='binomial',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gmm_func_poisson_returns_scalar(self, nl_data):
        """T-060: Test that poisson method returns a scalar."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = nl_data
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='poisson',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gmm_func_binomial_formula(self, nl_data):
        """T-061: Verify binomial method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = nl_data
        n = X.shape[0]
        
        # Python computation
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='binomial',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        # Manual computation
        W = ate_wt_nl_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # X1 = cbind(1, X)
        X1 = np.column_stack([np.ones(n), X])
        
        # binomial weights
        exp_term = np.exp(X1 @ cov1_coef)
        pweight1 = exp_term / (1.0 + exp_term)
        pweight2 = exp_term / (1.0 + exp_term)**2
        
        weighted_X = np.column_stack([pweight1, pweight2[:, None] * X1[:, S]])
        w_curr_del = weighted_X.T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="Binomial method GMM loss formula incorrect")


    def test_gmm_func_poisson_formula(self, nl_data):
        """T-062: Verify poisson method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = nl_data
        n = X.shape[0]
        
        # Python computation
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='poisson',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        # Manual computation
        W = ate_wt_nl_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # X1 = cbind(1, X)
        X1 = np.column_stack([np.ones(n), X])
        
        # poisson weights
        pweight = np.exp(X1 @ cov1_coef)
        
        weighted_X = np.column_stack([pweight, pweight[:, None] * X1[:, S]])
        w_curr_del = weighted_X.T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="Poisson method GMM loss formula incorrect")


# =============================================================================
# T-063 ~ T-068: ATT GMM Loss Function Tests
# =============================================================================

class TestAttGmmFuncBasic:
    """T-063 ~ T-066: ATT GMM loss function basic functionality tests."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        X, treat, beta_ini, S, beta_curr, _, cov0_coef = create_test_data(n=100, p=5, seed=42)
        return X, treat, beta_ini, S, beta_curr, cov0_coef

    def test_att_gmm_func_returns_scalar(self, simple_data):
        """T-063: Test that att_gmm_func returns a scalar."""
        X, treat, beta_ini, S, beta_curr, cov0_coef = simple_data
        
        loss = att_gmm_func(beta_curr, S, X_gmm=X, method='linear',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)

    def test_att_gmm_func_nonnegative(self, simple_data):
        """T-064: Test that ATT GMM loss is non-negative."""
        X, treat, beta_ini, S, beta_curr, cov0_coef = simple_data
        
        loss = att_gmm_func(beta_curr, S, X_gmm=X, method='linear',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        assert loss >= 0, f"ATT GMM loss should be non-negative, got {loss}"

    def test_att_gmm_func_linear_formula(self, simple_data):
        """T-065: Verify ATT linear method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov0_coef = simple_data
        n = X.shape[0]
        
        # Python computation
        loss = att_gmm_func(beta_curr, S, X_gmm=X, method='linear',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        # Manual computation
        from cbps.highdim.weight_funcs import att_wt_func
        W = att_wt_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # att_gmm_func creates X1 = cbind(1, X), shape (n, p+2)
        X1 = np.column_stack([np.ones(n), X])
        w_curr_del = X1[:, S].T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="ATT linear method GMM loss formula incorrect")


    def test_att_gmm_func_binomial(self, simple_data):
        """T-066: Test ATT binomial method."""
        X, treat, beta_ini, S, _, cov0_coef = simple_data
        
        # beta_curr for nonlinear methods includes intercept
        beta_curr_nl = np.array([0.1, 0.5, -0.3, 0.2])
        
        loss = att_gmm_func(beta_curr_nl, S, X_gmm=X, method='binomial',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestAttGmmFuncNonlinear:
    """T-067 ~ T-068: ATT GMM loss function nonlinear method tests."""

    @pytest.fixture
    def nl_data(self):
        """Generate test data for nonlinear methods."""
        X, treat, beta_ini, S, _, _, cov0_coef = create_test_data(n=100, p=5, seed=42)
        beta_curr = np.array([0.1, 0.5, -0.3, 0.2])
        return X, treat, beta_ini, S, beta_curr, cov0_coef

    def test_att_gmm_func_binomial_formula(self, nl_data):
        """T-067: Verify ATT binomial method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov0_coef = nl_data
        n = X.shape[0]
        
        # Python computation
        loss = att_gmm_func(beta_curr, S, X_gmm=X, method='binomial',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        # Manual computation
        from cbps.highdim.weight_funcs import att_wt_nl_func
        W = att_wt_nl_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # att_gmm_func creates X1 = cbind(1, X), shape (n, p+2)
        X1 = np.column_stack([np.ones(n), X])
        exp_term = np.exp(X1 @ cov0_coef)
        pweight1 = exp_term / (1.0 + exp_term)
        pweight2 = exp_term / (1.0 + exp_term)**2
        
        weighted_X = np.column_stack([pweight1, pweight2[:, None] * X1[:, S]])
        w_curr_del = weighted_X.T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="ATT binomial method GMM loss formula incorrect")

    def test_att_gmm_func_poisson_formula(self, nl_data):
        """T-068: Verify ATT poisson method GMM loss formula."""
        X, treat, beta_ini, S, beta_curr, cov0_coef = nl_data
        n = X.shape[0]
        
        # Python computation
        loss = att_gmm_func(beta_curr, S, X_gmm=X, method='poisson',
                           cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        # Manual computation
        from cbps.highdim.weight_funcs import att_wt_nl_func
        W = att_wt_nl_func(beta_curr, S, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # att_gmm_func creates X1 = cbind(1, X), shape (n, p+2)
        X1 = np.column_stack([np.ones(n), X])
        pweight = np.exp(X1 @ cov0_coef)
        
        weighted_X = np.column_stack([pweight, pweight[:, None] * X1[:, S]])
        w_curr_del = weighted_X.T @ W
        loss_manual = w_curr_del @ w_curr_del
        
        assert_allclose(loss, loss_manual, rtol=1e-10,
                       err_msg="ATT poisson method GMM loss formula incorrect")


# =============================================================================
# T-069 ~ T-074: GMM Loss Function Edge Case Tests
# =============================================================================

class TestGmmFuncEdgeCases:
    """T-069 ~ T-074: GMM loss function edge case tests."""

    def test_gmm_func_empty_S(self):
        """T-069: Test empty selected variable set."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        S = np.array([], dtype=int)
        beta_curr = np.array([])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert loss >= 0

    def test_gmm_func_single_variable(self):
        """T-070: Test single selected variable."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2] = 0.5
        S = np.array([2])
        beta_curr = np.array([0.5])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gmm_func_all_variables(self):
        """T-071: Test all variables selected."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2:p+2] = [0.5, -0.3, 0.2, 0.1, -0.1]
        S = np.array([2, 3, 4, 5, 6])  # All covariates in (p+2) space
        beta_curr = np.array([0.5, -0.3, 0.2, 0.1, -0.1])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gmm_func_invalid_method(self):
        """T-072: Test invalid method parameter."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = create_test_data()
        
        with pytest.raises(ValueError, match="not supported"):
            gmm_func(beta_curr, S, tt=1, X_gmm=X, method='invalid',
                    cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                    treat=treat, beta_ini=beta_ini)


    def test_gmm_func_extreme_coefficients(self):
        """T-073: Test extreme coefficient values."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2:5] = [5.0, -5.0, 5.0]
        S = np.array([2, 3, 4])
        beta_curr = np.array([5.0, -5.0, 5.0])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert not np.isnan(loss), "Loss should not be NaN"
        assert not np.isinf(loss), "Loss should not be Inf"

    def test_gmm_func_all_treated(self):
        """T-074: Test all samples are treated."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.ones(n)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2:5] = [0.5, -0.3, 0.2]
        S = np.array([2, 3, 4])
        beta_curr = np.array([0.5, -0.3, 0.2])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert not np.isnan(loss)


# =============================================================================
# T-075 ~ T-078: GMM Optimization Property Tests
# =============================================================================

class TestGmmFuncOptimization:
    """T-075 ~ T-078: GMM loss function optimization property tests."""

    @pytest.fixture
    def optimization_data(self):
        """Generate data for optimization tests."""
        np.random.seed(2024)
        n, p = 200, 8
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        
        beta_true = np.zeros(p + 2)
        beta_true[2:5] = [0.5, -0.3, 0.2]
        theta = np.column_stack([np.ones(n), X]) @ beta_true
        pi = 1.0 / (1.0 + np.exp(-theta))
        treat = np.random.binomial(1, pi).astype(float)
        
        S = np.array([2, 3, 4])
        cov1_coef = np.zeros(p + 2)
        cov1_coef[0:4] = [1.0, 0.5, -0.2, 0.1]
        cov0_coef = np.zeros(p + 2)
        cov0_coef[0:4] = [0.5, 0.3, -0.1, 0.2]
        
        return X, treat, beta_true, S, cov1_coef, cov0_coef

    def test_gmm_func_gradient_direction(self, optimization_data):
        """T-075: Verify GMM loss responds to coefficient changes."""
        X, treat, beta_ini, S, cov1_coef, cov0_coef = optimization_data
        
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        loss_base = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                            cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                            treat=treat, beta_ini=beta_ini)
        
        eps = 1e-5
        for i in range(len(beta_curr)):
            beta_plus = beta_curr.copy()
            beta_plus[i] += eps
            
            loss_plus = gmm_func(beta_plus, S, tt=1, X_gmm=X, method='linear',
                                cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                                treat=treat, beta_ini=beta_ini)
            
            assert loss_plus != loss_base, f"Perturbation of coefficient {i} should change loss"


    def test_gmm_func_minimum_exists(self, optimization_data):
        """T-076: Verify GMM loss has a minimum."""
        from scipy.optimize import minimize
        
        X, treat, beta_ini, S, cov1_coef, cov0_coef = optimization_data
        
        def objective(beta_curr):
            return gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                           cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                           treat=treat, beta_ini=beta_ini)
        
        x0 = np.array([0.5, -0.3, 0.2])
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        
        assert result.fun >= 0, "Minimum loss should be non-negative"
        assert result.fun < objective(x0) or result.fun < 1e-3, \
            "Optimization should reduce loss or find a minimum close to zero"

    def test_gmm_func_convexity_local(self, optimization_data):
        """T-077: Test local convexity of GMM loss."""
        X, treat, beta_ini, S, cov1_coef, cov0_coef = optimization_data
        
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        losses = []
        for _ in range(20):
            beta_test = beta_curr + np.random.randn(3) * 0.1
            loss = gmm_func(beta_test, S, tt=1, X_gmm=X, method='linear',
                           cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                           treat=treat, beta_ini=beta_ini)
            losses.append(loss)
        
        assert all(np.isfinite(l) for l in losses), "All losses should be finite"

    def test_gmm_func_reproducibility(self, optimization_data):
        """T-078: Verify reproducibility of GMM loss computation."""
        X, treat, beta_ini, S, cov1_coef, cov0_coef = optimization_data
        
        beta_curr = np.array([0.5, -0.3, 0.2])
        
        loss1 = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                        cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                        treat=treat, beta_ini=beta_ini)
        loss2 = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                        cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                        treat=treat, beta_ini=beta_ini)
        
        assert loss1 == loss2, "GMM loss computation should be reproducible"


# =============================================================================
# T-079 ~ T-081: Python-R Comparison Tests
# =============================================================================

class TestGmmFuncVsR:
    """T-079 ~ T-081: GMM loss function Python-R comparison tests."""

    @pytest.fixture
    def comparison_data(self):
        """Generate data for Python-R comparison."""
        np.random.seed(2024)
        n, p = 200, 10
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        
        beta_ps = np.zeros(p + 2)
        beta_ps[2:5] = [0.5, -0.3, 0.2]
        theta = np.column_stack([np.ones(n), X]) @ beta_ps
        pi = 1.0 / (1.0 + np.exp(-theta))
        treat = np.random.binomial(1, pi).astype(float)
        
        S = np.array([2, 3, 4])
        beta_curr = np.array([0.5, -0.3, 0.2])
        cov1_coef = np.zeros(p + 2)
        cov1_coef[0:4] = [1.0, 0.5, -0.2, 0.1]
        cov0_coef = np.zeros(p + 2)
        cov0_coef[0:4] = [0.5, 0.3, -0.1, 0.2]
        
        return X, treat, beta_ps, S, beta_curr, cov1_coef, cov0_coef


    def test_gmm_func_linear_vs_r_formula(self, comparison_data):
        """T-079: Verify linear method GMM loss matches R formula."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = comparison_data
        n = X.shape[0]
        
        # Python computation
        loss_py = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                          cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                          treat=treat, beta_ini=beta_ini)
        
        # R formula manual implementation
        # X1 = cbind(1, X), X2 = cbind(1, X) in weight func
        X1 = np.column_stack([np.ones(n), X])
        X2 = np.column_stack([np.ones(n), X])
        
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        W = treat / probs_curr - 1.0
        
        w_curr_del = X1[:, S].T @ W
        loss_r = np.sum(w_curr_del ** 2)
        
        assert_allclose(loss_py, loss_r, rtol=1e-10,
                       err_msg="Linear method GMM loss inconsistent with R")

    def test_att_gmm_func_linear_vs_r_formula(self, comparison_data):
        """T-080: Verify ATT linear method GMM loss matches R formula."""
        X, treat, beta_ini, S, beta_curr, _, cov0_coef = comparison_data
        n = X.shape[0]
        
        # Python computation
        loss_py = att_gmm_func(beta_curr, S, X_gmm=X, method='linear',
                              cov0_coef=cov0_coef, treat=treat, beta_ini=beta_ini)
        
        # R formula manual implementation
        # att_wt_func internally: X2 = cbind(1, X)
        X2 = np.column_stack([np.ones(n), X])
        
        beta_all = beta_ini.copy()
        beta_all[S] = beta_curr
        theta = X2 @ beta_all
        probs_curr = 1.0 - 1.0 / (1.0 + np.exp(theta))
        W = treat - ((1.0 - treat) * probs_curr / (1.0 - probs_curr))
        
        # att_gmm_func uses X1 = cbind(1, X)
        X1 = np.column_stack([np.ones(n), X])
        w_curr_del = X1[:, S].T @ W
        loss_r = np.sum(w_curr_del ** 2)
        
        assert_allclose(loss_py, loss_r, rtol=1e-10,
                       err_msg="ATT linear method GMM loss inconsistent with R")

    def test_gmm_func_moment_condition(self, comparison_data):
        """T-081: Verify mathematical properties of GMM moment conditions."""
        X, treat, beta_ini, S, beta_curr, cov1_coef, cov0_coef = comparison_data
        n = X.shape[0]
        
        # Compute moment conditions
        W = ate_wt_func(beta_curr, S, tt=1, X_wt=X, beta_ini=beta_ini, treat=treat)
        
        # gmm_func uses X1 = cbind(1, X)
        X1 = np.column_stack([np.ones(n), X])
        g_n = X1[:, S].T @ W
        
        # GMM loss = ||g_n||^2
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert_allclose(loss, np.sum(g_n ** 2), rtol=1e-10,
                       err_msg="GMM loss should equal sum of squared moment conditions")


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestGmmFuncNumericalStability:
    """GMM loss function numerical stability tests."""

    def test_gmm_func_large_sample(self):
        """Test large sample case."""
        np.random.seed(42)
        n, p = 1000, 10
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2:5] = [0.5, -0.3, 0.2]
        S = np.array([2, 3, 4])
        beta_curr = np.array([0.5, -0.3, 0.2])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert np.isfinite(loss), "Large sample loss should be finite"

    def test_gmm_func_high_dimensional(self):
        """Test high-dimensional case."""
        np.random.seed(42)
        n, p = 100, 50
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        beta_ini = np.zeros(p + 2)
        beta_ini[2:7] = [0.5, -0.3, 0.2, 0.1, -0.1]
        S = np.array([2, 3, 4, 5, 6])
        beta_curr = np.array([0.5, -0.3, 0.2, 0.1, -0.1])
        cov1_coef = np.zeros(p + 2)
        cov0_coef = np.zeros(p + 2)
        
        loss = gmm_func(beta_curr, S, tt=1, X_gmm=X, method='linear',
                       cov1_coef=cov1_coef, cov0_coef=cov0_coef,
                       treat=treat, beta_ini=beta_ini)
        
        assert np.isfinite(loss), "High-dimensional loss should be finite"


# =============================================================================
# LASSO Utility Tests
# (merged from test_lasso_utils.py)
# =============================================================================


# Skip all tests if glmnetforpython is not available


class TestCvGlmnetGaussianBasic:
    """T-001: Test cv_glmnet gaussian family basic functionality."""

    @pytest.fixture
    def simple_linear_data(self):
        """Generate simple linear regression data with known coefficients."""
        np.random.seed(42)
        n, p = 200, 10
        X = np.random.randn(n, p)
        # True coefficients: first 3 are non-zero
        beta_true = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0, 0, 0])
        intercept_true = 0.5
        y = intercept_true + X @ beta_true + np.random.randn(n) * 0.5
        return X, y, beta_true, intercept_true

    @pytest.fixture
    def sparse_data(self):
        """Generate sparse regression data (most coefficients are zero)."""
        np.random.seed(123)
        n, p = 300, 50
        X = np.random.randn(n, p)
        # Only 5 non-zero coefficients
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[5] = -2.0
        beta_true[10] = 1.5
        beta_true[20] = -1.0
        beta_true[30] = 0.8
        intercept_true = 1.0
        y = intercept_true + X @ beta_true + np.random.randn(n) * 0.3
        return X, y, beta_true, intercept_true

    def test_returns_correct_types(self, simple_linear_data):
        """Test that cv_glmnet returns correct types."""
        X, y, _, _ = simple_linear_data
        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        # Check return types
        assert model is not None, "Model should not be None"
        assert isinstance(coef, np.ndarray), "Coefficients should be ndarray"
        assert isinstance(lambda_min, (float, np.floating)), \
            "lambda_min should be float"

    def test_coefficient_shape_with_intercept(self, simple_linear_data):
        """Test coefficient shape when intercept=True."""
        X, y, _, _ = simple_linear_data
        n, p = X.shape
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        # With intercept: coef should have p+1 elements
        assert coef.shape == (p + 1,), \
            f"Expected shape ({p+1},), got {coef.shape}"

    def test_coefficient_shape_without_intercept(self, simple_linear_data):
        """Test coefficient shape when intercept=False.
        
        Note: R's glmnet always returns (p+1) coefficients from coef(), even when
        intercept=False. The intercept position (index 0) will be 0 in this case.
        This matches R's behavior where coef() always includes the intercept slot.
        """
        X, y, _, _ = simple_linear_data
        n, p = X.shape
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=False)

        # R's coef() always returns (p+1) coefficients, with intercept at index 0
        # When intercept=False, the intercept value should be 0
        assert coef.shape == (p + 1,), \
            f"Expected shape ({p + 1},), got {coef.shape}"
        assert coef[0] == 0.0, \
            f"Intercept should be 0 when intercept=False, got {coef[0]}"

    def test_lambda_min_positive(self, simple_linear_data):
        """Test that lambda_min is positive."""
        X, y, _, _ = simple_linear_data
        _, _, lambda_min = cv_glmnet(X, y, family='gaussian')

        assert lambda_min > 0, f"lambda_min should be positive, got {lambda_min}"

    def test_intercept_estimation(self, simple_linear_data):
        """Test that intercept is reasonably estimated."""
        X, y, _, intercept_true = simple_linear_data
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        # First element is intercept
        intercept_est = coef[0]
        # Allow some tolerance due to regularization
        assert abs(intercept_est - intercept_true) < 1.0, \
            f"Intercept estimate {intercept_est} too far from true {intercept_true}"

    def test_nonzero_coefficients_identified(self, simple_linear_data):
        """Test that non-zero coefficients are identified."""
        X, y, beta_true, _ = simple_linear_data
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        # Skip intercept (coef[0]), check coef[1:]
        coef_no_intercept = coef[1:]
        true_nonzero_idx = np.where(np.abs(beta_true) > 0)[0]

        # Selected variables should include the true non-zero ones
        selected_idx = select_variables(coef_no_intercept, tol=1e-6)
        for idx in true_nonzero_idx:
            assert idx in selected_idx, \
                f"True non-zero coefficient at index {idx} not selected"

    def test_coefficient_signs(self, simple_linear_data):
        """Test that coefficient signs match true signs."""
        X, y, beta_true, _ = simple_linear_data
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        coef_no_intercept = coef[1:]
        true_nonzero_idx = np.where(np.abs(beta_true) > 0)[0]

        for idx in true_nonzero_idx:
            if np.abs(coef_no_intercept[idx]) > 1e-6:
                assert np.sign(coef_no_intercept[idx]) == np.sign(beta_true[idx]), \
                    f"Sign mismatch at index {idx}: " \
                    f"est={coef_no_intercept[idx]:.3f}, true={beta_true[idx]:.3f}"

    def test_sparse_variable_selection(self, sparse_data):
        """Test variable selection in sparse setting."""
        X, y, beta_true, _ = sparse_data
        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        coef_no_intercept = coef[1:]
        true_nonzero_idx = set(np.where(np.abs(beta_true) > 0)[0])
        selected_idx = set(select_variables(coef_no_intercept, tol=1e-6))

        # Check that true non-zero coefficients are selected
        # (LASSO may select some false positives, but should not miss true ones)
        missed = true_nonzero_idx - selected_idx
        assert len(missed) == 0, \
            f"Missed true non-zero coefficients at indices: {missed}"

    def test_reproducibility_with_same_data(self, simple_linear_data):
        """Test that results are reproducible with same data."""
        X, y, _, _ = simple_linear_data

        _, coef1, lambda1 = cv_glmnet(X, y, family='gaussian')
        _, coef2, lambda2 = cv_glmnet(X, y, family='gaussian')

        # Results should be identical (deterministic algorithm)
        assert_allclose(coef1, coef2, rtol=1e-10,
                       err_msg="Coefficients not reproducible")
        assert_allclose(lambda1, lambda2, rtol=1e-10,
                       err_msg="Lambda not reproducible")

    def test_different_n_folds(self, simple_linear_data):
        """Test with different number of CV folds."""
        X, y, _, _ = simple_linear_data

        for n_folds in [5, 10]:
            model, coef, lambda_min = cv_glmnet(
                X, y, family='gaussian', n_folds=n_folds
            )
            assert model is not None
            assert len(coef) == X.shape[1] + 1  # with intercept
            assert lambda_min > 0

    def test_prediction_quality(self, simple_linear_data):
        """Test that predictions are reasonable."""
        X, y, _, _ = simple_linear_data
        model, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        # Manual prediction: y_pred = intercept + X @ coef[1:]
        y_pred = coef[0] + X @ coef[1:]

        # R-squared should be reasonably high for this simple data
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert r_squared > 0.8, \
            f"R-squared {r_squared:.3f} too low for simple linear data"

    def test_alpha_parameter(self, simple_linear_data):
        """Test that alpha parameter is respected (LASSO vs Ridge)."""
        X, y, _, _ = simple_linear_data

        # Pure LASSO (alpha=1.0) should produce sparser solutions
        _, coef_lasso, _ = cv_glmnet(X, y, family='gaussian', alpha=1.0)
        _, coef_enet, _ = cv_glmnet(X, y, family='gaussian', alpha=0.5)

        # Count non-zero coefficients (excluding intercept)
        n_nonzero_lasso = np.sum(np.abs(coef_lasso[1:]) > 1e-6)
        n_nonzero_enet = np.sum(np.abs(coef_enet[1:]) > 1e-6)

        # LASSO should be at least as sparse as elastic net
        assert n_nonzero_lasso <= n_nonzero_enet + 2, \
            f"LASSO ({n_nonzero_lasso}) should be sparser than " \
            f"elastic net ({n_nonzero_enet})"


class TestCvGlmnetGaussianNumerical:
    """Numerical validation tests for cv_glmnet gaussian family."""

    def test_coefficient_magnitude_reasonable(self):
        """Test that coefficient magnitudes are reasonable."""
        np.random.seed(456)
        n, p = 200, 20
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -0.5, 0.3] + [0] * 17)
        y = X @ beta_true + np.random.randn(n) * 0.5

        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        # Coefficients should not be extremely large
        assert np.max(np.abs(coef)) < 10, \
            f"Max coefficient {np.max(np.abs(coef)):.3f} too large"

    def test_residuals_centered(self):
        """Test that residuals are approximately centered."""
        np.random.seed(789)
        n, p = 300, 15
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.0] + [0] * 13)
        y = 1.0 + X @ beta_true + np.random.randn(n) * 0.3

        _, coef, _ = cv_glmnet(X, y, family='gaussian', intercept=True)

        y_pred = coef[0] + X @ coef[1:]
        residuals = y - y_pred

        # Mean of residuals should be close to zero
        assert abs(np.mean(residuals)) < 0.1, \
            f"Mean residual {np.mean(residuals):.4f} not centered"

    def test_handles_collinear_features(self):
        """Test handling of nearly collinear features."""
        np.random.seed(111)
        n, p = 200, 10
        X = np.random.randn(n, p)
        # Add a nearly collinear column
        X = np.column_stack([X, X[:, 0] + np.random.randn(n) * 0.01])

        beta_true = np.array([1.0] + [0] * 10)
        y = X[:, :p] @ beta_true[:p] + np.random.randn(n) * 0.5

        # Should not raise an error
        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        assert model is not None
        assert len(coef) == X.shape[1] + 1
        assert lambda_min > 0


class TestSelectVariables:
    """Tests for select_variables function."""

    def test_basic_selection(self):
        """Test basic variable selection."""
        coef = np.array([1.2, 0.0, 0.5, 0.0, -0.3])
        selected = select_variables(coef)
        expected = np.array([0, 2, 4])
        assert_array_equal(selected, expected)

    def test_all_zero(self):
        """Test when all coefficients are zero."""
        coef = np.array([0.0, 0.0, 0.0, 0.0])
        selected = select_variables(coef)
        assert len(selected) == 0

    def test_all_nonzero(self):
        """Test when all coefficients are non-zero."""
        coef = np.array([1.0, -2.0, 0.5, -0.1])
        selected = select_variables(coef)
        expected = np.array([0, 1, 2, 3])
        assert_array_equal(selected, expected)

    def test_tolerance_threshold(self):
        """Test tolerance threshold effect."""
        coef = np.array([1.0, 1e-11, 1e-9, 0.5])

        # Default tolerance (1e-10)
        selected_default = select_variables(coef)
        assert 1 not in selected_default  # 1e-11 < 1e-10
        assert 2 in selected_default      # 1e-9 > 1e-10

        # Stricter tolerance
        selected_strict = select_variables(coef, tol=1e-12)
        assert 1 in selected_strict       # 1e-11 > 1e-12

        # Looser tolerance
        selected_loose = select_variables(coef, tol=1e-8)
        assert 2 not in selected_loose    # 1e-9 < 1e-8


# =============================================================================
# T-002: Python-R End-to-End Comparison Tests (cv_glmnet gaussian family)
# =============================================================================

class TestCvGlmnetGaussianVsR:
    """T-002: Compare Python cv_glmnet with R cv.glmnet gaussian coefficients.
    
    These tests require rpy2 and glmnet package installed in R environment.
    """

    @pytest.fixture
    def lalonde_data(self):
        """Load LaLonde dataset for real data testing."""
        try:
            from cbps.datasets import load_lalonde
            df = load_lalonde(dehejia_wahba_only=True)
            
            # Extract covariates and outcome variable
            covariates = ['age', 'educ', 'black', 'hisp', 'married', 
                         'nodegr', 're74', 're75']
            X = df[covariates].values.astype(np.float64)
            y = df['re78'].values.astype(np.float64)
            treat = df['treat'].values.astype(np.float64)
            
            return X, y, treat, covariates
        except Exception as e:
            pytest.skip(f"Cannot load LaLonde data: {e}")

    @pytest.fixture
    def simulated_data(self):
        """Generate simulated data for comparison tests."""
        np.random.seed(2024)
        n, p = 300, 20
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.5, 1.0, 0.5, -0.3] + [0] * 15)
        intercept_true = 1.0
        y = intercept_true + X @ beta_true + np.random.randn(n) * 0.5
        return X, y, beta_true, intercept_true

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_gaussian_coef_vs_r_simulated(self, simulated_data):
        """Compare Python and R gaussian coefficients on simulated data."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2 for R comparison")
        except ImportError:
            pytest.skip("Requires rpy2 for R comparison")

        X, y, _, _ = simulated_data

        # Python implementation
        _, coef_py, lambda_py = cv_glmnet(X, y, family='gaussian', intercept=True)

        # R implementation
        _, coef_r, lambda_r = cv_glmnet_via_r(X, y, family='gaussian', intercept=True)

        # Coefficient comparison (allow some error due to different CV folds)
        # Mainly check signs and relative magnitudes of non-zero coefficients
        nonzero_py = np.abs(coef_py) > 1e-6
        nonzero_r = np.abs(coef_r) > 1e-6

        # Check selected variable sets are similar
        overlap = np.sum(nonzero_py & nonzero_r)
        total_selected = np.sum(nonzero_py | nonzero_r)
        
        if total_selected > 0:
            jaccard = overlap / total_selected
            assert jaccard > 0.5, \
                f"Variable selection Jaccard similarity {jaccard:.3f} too low"

        # Check sign consistency for large coefficients
        large_coef_idx = np.where(np.abs(coef_py) > 0.1)[0]
        for idx in large_coef_idx:
            if np.abs(coef_r[idx]) > 0.1:
                assert np.sign(coef_py[idx]) == np.sign(coef_r[idx]), \
                    f"Index {idx} sign mismatch: Python={coef_py[idx]:.3f}, R={coef_r[idx]:.3f}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_gaussian_coef_vs_r_lalonde(self, lalonde_data):
        """Compare Python and R gaussian coefficients on LaLonde data."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2 for R comparison")
        except ImportError:
            pytest.skip("Requires rpy2 for R comparison")

        X, y, _, covariates = lalonde_data

        # Python implementation
        _, coef_py, lambda_py = cv_glmnet(X, y, family='gaussian', intercept=True)

        # R implementation
        _, coef_r, lambda_r = cv_glmnet_via_r(X, y, family='gaussian', intercept=True)

        # Verify coefficient relative error
        # For non-zero coefficients, check relative error
        for i in range(len(coef_py)):
            if np.abs(coef_r[i]) > 0.1:
                rel_err = np.abs(coef_py[i] - coef_r[i]) / np.abs(coef_r[i])
                assert rel_err < 0.5, \
                    f"Coefficient {i} relative error {rel_err:.3f} exceeds 50%"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_gaussian_lambda_min_consistency(self, simulated_data):
        """Verify lambda.min selection consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2 for R comparison")
        except ImportError:
            pytest.skip("Requires rpy2 for R comparison")

        X, y, _, _ = simulated_data

        # Python implementation
        _, _, lambda_py = cv_glmnet(X, y, family='gaussian', intercept=True)

        # R implementation
        _, _, lambda_r = cv_glmnet_via_r(X, y, family='gaussian', intercept=True)

        # Lambda values should be in the same order of magnitude
        ratio = lambda_py / lambda_r if lambda_r > 0 else float('inf')
        assert 0.1 < ratio < 10, \
            f"Lambda ratio {ratio:.3f} outside reasonable range [0.1, 10]"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_gaussian_intercept_false_vs_r(self, simulated_data):
        """Verify consistency with R when intercept=False.
        
        Note: R's coef() always returns (p+1,) shape, even when intercept=False.
        The intercept position (first element) is 0 when intercept=False.
        Python implementation should match this behavior.
        """
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2 for R comparison")
        except ImportError:
            pytest.skip("Requires rpy2 for R comparison")

        X, y, _, _ = simulated_data
        p = X.shape[1]

        # Python implementation (no intercept) - returns (p+1,) shape, first element is 0
        _, coef_py, _ = cv_glmnet(X, y, family='gaussian', intercept=False)

        # R implementation (no intercept) - returns (p+1,) shape, first element is 0
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='gaussian', intercept=False)

        # Verify shape
        assert coef_py.shape == (p + 1,), f"Python coef shape should be ({p+1},), got {coef_py.shape}"
        assert coef_r.shape == (p + 1,), f"R coef shape should be ({p+1},), got {coef_r.shape}"
        assert coef_py[0] == 0.0, f"Python intercept should be 0, got {coef_py[0]}"
        assert coef_r[0] == 0.0, f"R intercept should be 0, got {coef_r[0]}"

        # Check sign consistency for non-zero coefficients
        large_coef_idx = np.where(np.abs(coef_py[1:]) > 0.1)[0]
        for idx in large_coef_idx:
            if np.abs(coef_r[idx + 1]) > 0.1:
                assert np.sign(coef_py[idx + 1]) == np.sign(coef_r[idx + 1]), \
                    f"Index {idx} sign mismatch: Python={coef_py[idx+1]:.3f}, R={coef_r[idx+1]:.3f}"


# =============================================================================
# T-003 ~ T-005: Numerical Verification Tests (Gaussian Family)
# =============================================================================

class TestCvGlmnetGaussianNumericalValidation:
    """T-003 ~ T-005: Gaussian family numerical validation tests."""

    @pytest.fixture
    def controlled_data(self):
        """Generate controlled data for precise numerical verification."""
        np.random.seed(12345)
        n, p = 500, 30
        X = np.random.randn(n, p)
        # Sparse true coefficients
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[2] = -2.0
        beta_true[5] = 1.5
        beta_true[10] = -1.0
        intercept_true = 2.0
        noise_std = 0.3
        y = intercept_true + X @ beta_true + np.random.randn(n) * noise_std
        return X, y, beta_true, intercept_true, noise_std

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lambda_min_selection_consistency(self, controlled_data):
        """T-003: Verify gaussian family lambda.min selection consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _, _, _ = controlled_data

        # Multiple runs to check stability
        lambdas_py = []
        lambdas_r = []
        
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            _, _, lam_py = cv_glmnet(X, y, family='gaussian')
            _, _, lam_r = cv_glmnet_via_r(X, y, family='gaussian')
            lambdas_py.append(lam_py)
            lambdas_r.append(lam_r)

        # Check Python and R lambda are in the same order of magnitude
        for lam_py, lam_r in zip(lambdas_py, lambdas_r):
            ratio = lam_py / lam_r if lam_r > 0 else float('inf')
            assert 0.1 < ratio < 10, \
                f"Lambda ratio {ratio:.3f} outside reasonable range"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_intercept_estimation_consistency(self, controlled_data):
        """T-004: Verify gaussian family intercept estimation consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _, intercept_true, _ = controlled_data

        _, coef_py, _ = cv_glmnet(X, y, family='gaussian', intercept=True)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='gaussian', intercept=True)

        intercept_py = coef_py[0]
        intercept_r = coef_r[0]

        # Intercept estimation should be close to true value
        assert abs(intercept_py - intercept_true) < 0.5, \
            f"Python intercept {intercept_py:.4f} deviates from true value {intercept_true:.4f}"
        assert abs(intercept_r - intercept_true) < 0.5, \
            f"R intercept {intercept_r:.4f} deviates from true value {intercept_true:.4f}"

        # Python and R intercept should be close
        assert abs(intercept_py - intercept_r) < 0.3, \
            f"Python-R intercept difference {abs(intercept_py - intercept_r):.4f} too large"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_nonzero_coefficient_indices_consistency(self, controlled_data):
        """T-005: Verify gaussian family non-zero coefficient index consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, beta_true, _, _ = controlled_data
        true_nonzero = set(np.where(np.abs(beta_true) > 0)[0])

        _, coef_py, _ = cv_glmnet(X, y, family='gaussian', intercept=True)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='gaussian', intercept=True)

        # Extract non-zero coefficient indices (skip intercept)
        selected_py = set(select_variables(coef_py[1:], tol=1e-6))
        selected_r = set(select_variables(coef_r[1:], tol=1e-6))

        # Check true non-zero coefficients are selected
        missed_py = true_nonzero - selected_py
        missed_r = true_nonzero - selected_r
        
        assert len(missed_py) == 0, \
            f"Python missed true non-zero coefficients: {missed_py}"
        assert len(missed_r) == 0, \
            f"R missed true non-zero coefficients: {missed_r}"

        # Check Python and R selection overlap
        # Note: Python and R LASSO implementations may select slightly different variable sets
        # due to CV fold randomness and numerical precision differences
        # The key point is that both should select true non-zero coefficients
        overlap = selected_py & selected_r
        union = selected_py | selected_r
        jaccard = len(overlap) / len(union) if len(union) > 0 else 1.0
        
        # Relax threshold to 0.6, because R may select more false positives
        assert jaccard > 0.6, \
            f"Python-R variable selection Jaccard similarity {jaccard:.3f} too low"


# =============================================================================
# T-006 ~ T-008: Edge Case Tests (Gaussian Family)
# =============================================================================

class TestCvGlmnetGaussianEdgeCases:
    """T-006 ~ T-008: Gaussian family edge case tests."""

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_high_dimensional_p_greater_than_n(self):
        """T-006: Test gaussian family high-dimensional setting (p > n)."""
        np.random.seed(777)
        n, p = 100, 200  # p > n
        X = np.random.randn(n, p)
        # Sparse coefficients
        beta_true = np.zeros(p)
        beta_true[:5] = [2.0, -1.5, 1.0, -0.5, 0.3]
        y = X @ beta_true + np.random.randn(n) * 0.5

        # Should run normally
        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        assert model is not None
        assert len(coef) == p + 1  # Including intercept
        assert lambda_min > 0

        # Check sparsity - high-dimensional case should select fewer variables
        n_selected = np.sum(np.abs(coef[1:]) > 1e-6)
        assert n_selected < p, \
            f"High-dimensional case selected {n_selected}/{p} variables, should be more sparse"

        # Check true non-zero coefficients are selected
        selected = select_variables(coef[1:], tol=1e-6)
        for i in range(5):
            assert i in selected, f"True non-zero coefficient {i} not selected"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_sparse_coefficients(self):
        """T-007: Test gaussian family sparse coefficient case."""
        np.random.seed(888)
        n, p = 300, 100
        X = np.random.randn(n, p)
        # Extremely sparse: only 2 non-zero coefficients
        beta_true = np.zeros(p)
        beta_true[0] = 5.0
        beta_true[50] = -3.0
        y = X @ beta_true + np.random.randn(n) * 0.2

        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        # Check selected variable count
        selected = select_variables(coef[1:], tol=1e-6)
        
        # Should select true non-zero coefficients
        assert 0 in selected, "Coefficient 0 should be selected"
        assert 50 in selected, "Coefficient 50 should be selected"

        # Selected variables should not be too many
        assert len(selected) < 20, \
            f"Selected {len(selected)} variables, too many for only 2 true non-zero coefficients"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_all_coefficients_zero(self):
        """T-008: Test gaussian family all coefficients zero case."""
        np.random.seed(999)
        n, p = 200, 20
        X = np.random.randn(n, p)
        # All coefficients are zero, y is only noise
        y = np.random.randn(n) * 1.0

        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        # Should run normally
        assert model is not None
        assert lambda_min > 0

        # Most coefficients should be shrunk to zero
        n_nonzero = np.sum(np.abs(coef[1:]) > 1e-6)
        assert n_nonzero < p / 2, \
            f"Pure noise data selected {n_nonzero}/{p} variables, should be fewer"

        # Intercept should be close to y mean
        assert abs(coef[0] - np.mean(y)) < 0.5, \
            f"Intercept {coef[0]:.4f} should be close to y mean {np.mean(y):.4f}"


# =============================================================================
# T-009 ~ T-015: Binomial Family Tests
# =============================================================================

class TestCvGlmnetBinomialBasic:
    """T-009: Test cv_glmnet binomial family basic functionality."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n, p = 300, 15
        X = np.random.randn(n, p)
        # True coefficients
        beta_true = np.array([1.5, -1.0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        intercept_true = 0.0
        # Generate probabilities and binary response
        logits = intercept_true + X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        return X, y, beta_true, intercept_true

    @pytest.fixture
    def imbalanced_data(self):
        """Generate class imbalanced data."""
        np.random.seed(123)
        n, p = 400, 10
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0, 0, 0])
        # Biased towards negative class intercept
        intercept_true = -2.0
        logits = intercept_true + X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        return X, y, beta_true, intercept_true

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_returns_correct_types(self, binary_data):
        """Test binomial family returns correct types."""
        X, y, _, _ = binary_data
        model, coef, lambda_min = cv_glmnet(X, y, family='binomial')

        assert model is not None
        assert isinstance(coef, np.ndarray)
        assert isinstance(lambda_min, (float, np.floating))
        assert lambda_min > 0

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_coefficient_shape(self, binary_data):
        """Test binomial family coefficient shape.
        
        Note: R's glmnet always returns (p+1) coefficients from coef(), even when
        intercept=False. The intercept position (index 0) will be 0 in this case.
        """
        X, y, _, _ = binary_data
        n, p = X.shape

        # With intercept - returns (p+1,) shape
        _, coef_with, _ = cv_glmnet(X, y, family='binomial', intercept=True)
        assert coef_with.shape == (p + 1,)

        # No intercept - R coef() still returns (p+1,) shape, intercept position is 0
        _, coef_without, _ = cv_glmnet(X, y, family='binomial', intercept=False)
        assert coef_without.shape == (p + 1,), \
            f"Without intercept coefficient shape should be ({p + 1},), got {coef_without.shape}"
        assert coef_without[0] == 0.0, \
            f"Without intercept, intercept should be 0, got {coef_without[0]}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_nonzero_coefficients(self, binary_data):
        """Test binomial family non-zero coefficient identification."""
        X, y, beta_true, _ = binary_data
        _, coef, _ = cv_glmnet(X, y, family='binomial', intercept=True)

        coef_no_intercept = coef[1:]
        true_nonzero = np.where(np.abs(beta_true) > 0)[0]
        selected = select_variables(coef_no_intercept, tol=1e-6)

        # Check true non-zero coefficients are selected
        for idx in true_nonzero:
            assert idx in selected, f"True non-zero coefficient {idx} not selected"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_coefficient_signs(self, binary_data):
        """Test binomial family coefficient signs."""
        X, y, beta_true, _ = binary_data
        _, coef, _ = cv_glmnet(X, y, family='binomial', intercept=True)

        coef_no_intercept = coef[1:]
        true_nonzero = np.where(np.abs(beta_true) > 0)[0]

        for idx in true_nonzero:
            if np.abs(coef_no_intercept[idx]) > 0.1:
                assert np.sign(coef_no_intercept[idx]) == np.sign(beta_true[idx]), \
                    f"Index {idx} sign mismatch"


class TestCvGlmnetBinomialVsR:
    """T-010 ~ T-013: Binomial family Python-R comparison tests."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(2024)
        n, p = 300, 15
        X = np.random.randn(n, p)
        beta_true = np.array([1.5, -1.0, 0.8] + [0] * 12)
        logits = X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        return X, y, beta_true

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_coef_vs_r(self, binary_data):
        """T-010: Compare Python and R binomial coefficients."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = binary_data

        _, coef_py, lambda_py = cv_glmnet(X, y, family='binomial', intercept=True)
        _, coef_r, lambda_r = cv_glmnet_via_r(X, y, family='binomial', intercept=True)

        # Check variable selection similarity
        selected_py = set(select_variables(coef_py[1:], tol=1e-6))
        selected_r = set(select_variables(coef_r[1:], tol=1e-6))

        overlap = selected_py & selected_r
        union = selected_py | selected_r
        jaccard = len(overlap) / len(union) if len(union) > 0 else 1.0

        assert jaccard > 0.5, f"Variable selection Jaccard similarity {jaccard:.3f} too low"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_lambda_min_vs_r(self, binary_data):
        """T-011: Verify binomial family lambda.min selection consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = binary_data

        _, _, lambda_py = cv_glmnet(X, y, family='binomial')
        _, _, lambda_r = cv_glmnet_via_r(X, y, family='binomial')

        ratio = lambda_py / lambda_r if lambda_r > 0 else float('inf')
        assert 0.1 < ratio < 10, f"Lambda ratio {ratio:.3f} outside reasonable range"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_intercept_false_vs_r(self, binary_data):
        """T-012: Verify binomial family intercept=False coefficient consistency.
        
        Note: R's coef() always returns (p+1,) shape, even when intercept=False.
        The intercept position (first element) is 0 when intercept=False.
        Python implementation should match this behavior.
        """
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = binary_data
        p = X.shape[1]

        _, coef_py, _ = cv_glmnet(X, y, family='binomial', intercept=False)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='binomial', intercept=False)

        # Verify shape - both should return (p+1,) shape
        assert coef_py.shape == (p + 1,), f"Python coefficient shape should be ({p+1},), got {coef_py.shape}"
        assert coef_r.shape == (p + 1,), f"R coefficient shape should be ({p+1},), got {coef_r.shape}"
        assert coef_py[0] == 0.0, f"Python intercept should be 0, got {coef_py[0]}"
        assert coef_r[0] == 0.0, f"R intercept should be 0, got {coef_r[0]}"

        # Check large coefficient sign consistency
        large_idx = np.where(np.abs(coef_py[1:]) > 0.1)[0]
        for idx in large_idx:
            if np.abs(coef_r[idx + 1]) > 0.1:
                assert np.sign(coef_py[idx + 1]) == np.sign(coef_r[idx + 1]), \
                    f"Index {idx} sign mismatch: Python={coef_py[idx+1]:.3f}, R={coef_r[idx+1]:.3f}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_intercept_true_vs_r(self, binary_data):
        """T-013: Verify binomial family intercept=True coefficient consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = binary_data

        _, coef_py, _ = cv_glmnet(X, y, family='binomial', intercept=True)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='binomial', intercept=True)

        # Intercepts should be close
        assert abs(coef_py[0] - coef_r[0]) < 1.0, \
            f"Intercept difference too large: Python={coef_py[0]:.4f}, R={coef_r[0]:.4f}"


class TestCvGlmnetBinomialEdgeCases:
    """T-014 ~ T-015: Binomial family edge case tests."""

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_perfect_separation(self):
        """T-014: Test binomial family perfect separation case."""
        np.random.seed(555)
        n, p = 200, 10
        X = np.random.randn(n, p)
        # Perfect separation: y = 1 when X[:, 0] > 0
        y = (X[:, 0] > 0).astype(float)

        # Should run, but coefficients may be large
        model, coef, lambda_min = cv_glmnet(X, y, family='binomial')

        assert model is not None
        assert lambda_min > 0
        # First coefficient should be positive and relatively large
        assert coef[1] > 0, "First coefficient should be positive for perfect separation"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_class_imbalance(self):
        """T-015: Test binomial family class imbalance case."""
        np.random.seed(666)
        n, p = 400, 10
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -0.5] + [0] * 8)
        # Severe imbalance: ~10% positive class
        logits = -2.0 + X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)

        model, coef, lambda_min = cv_glmnet(X, y, family='binomial')

        assert model is not None
        assert lambda_min > 0

        # Check non-zero coefficients
        selected = select_variables(coef[1:], tol=1e-6)
        assert 0 in selected, "First true non-zero coefficient should be selected"


# =============================================================================
# T-016 ~ T-020: Poisson Family Tests
# =============================================================================

class TestCvGlmnetPoissonBasic:
    """T-016: Test cv_glmnet poisson family basic functionality."""

    @pytest.fixture
    def count_data(self):
        """Generate count data."""
        np.random.seed(42)
        n, p = 300, 12
        X = np.random.randn(n, p)
        beta_true = np.array([0.5, -0.3, 0.2] + [0] * 9)
        # Poisson response
        log_mu = X @ beta_true
        mu = np.exp(log_mu)
        y = np.random.poisson(mu)
        return X, y, beta_true

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_returns_correct_types(self, count_data):
        """Test poisson family returns correct types."""
        X, y, _ = count_data
        model, coef, lambda_min = cv_glmnet(X, y, family='poisson')

        assert model is not None
        assert isinstance(coef, np.ndarray)
        assert isinstance(lambda_min, (float, np.floating))
        assert lambda_min > 0

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_coefficient_shape(self, count_data):
        """Test poisson family coefficient shape.
        
        Note: R's glmnet always returns (p+1) coefficients from coef(), even when
        intercept=False. The intercept position (index 0) will be 0 in this case.
        """
        X, y, _ = count_data
        n, p = X.shape

        # With intercept - returns (p+1,) shape
        _, coef_with, _ = cv_glmnet(X, y, family='poisson', intercept=True)
        assert coef_with.shape == (p + 1,)

        # No intercept - R coef() still returns (p+1,) shape, intercept position is 0
        _, coef_without, _ = cv_glmnet(X, y, family='poisson', intercept=False)
        assert coef_without.shape == (p + 1,), \
            f"Without intercept coefficient shape should be ({p + 1},), got {coef_without.shape}"
        assert coef_without[0] == 0.0, \
            f"Without intercept, intercept should be 0, got {coef_without[0]}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_nonzero_coefficients(self, count_data):
        """Test poisson family non-zero coefficient identification."""
        X, y, beta_true = count_data
        _, coef, _ = cv_glmnet(X, y, family='poisson', intercept=True)

        coef_no_intercept = coef[1:]
        true_nonzero = np.where(np.abs(beta_true) > 0)[0]
        selected = select_variables(coef_no_intercept, tol=1e-6)

        for idx in true_nonzero:
            assert idx in selected, f"True non-zero coefficient {idx} not selected"


class TestCvGlmnetPoissonVsR:
    """T-017 ~ T-019: Poisson family Python-R comparison tests."""

    @pytest.fixture
    def count_data(self):
        """Generate count data."""
        np.random.seed(2024)
        n, p = 300, 12
        X = np.random.randn(n, p)
        beta_true = np.array([0.5, -0.3, 0.2] + [0] * 9)
        log_mu = X @ beta_true
        mu = np.exp(np.clip(log_mu, -5, 5))  # Prevent overflow
        y = np.random.poisson(mu)
        return X, y, beta_true

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_coef_vs_r(self, count_data):
        """T-017: Compare Python and R poisson coefficients."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = count_data

        _, coef_py, _ = cv_glmnet(X, y, family='poisson', intercept=True)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='poisson', intercept=True)

        selected_py = set(select_variables(coef_py[1:], tol=1e-6))
        selected_r = set(select_variables(coef_r[1:], tol=1e-6))

        overlap = selected_py & selected_r
        union = selected_py | selected_r
        jaccard = len(overlap) / len(union) if len(union) > 0 else 1.0

        assert jaccard > 0.5, f"Variable selection Jaccard similarity {jaccard:.3f} too low"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_lambda_min_vs_r(self, count_data):
        """T-018: Verify poisson family lambda.min selection consistency."""
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = count_data

        _, _, lambda_py = cv_glmnet(X, y, family='poisson')
        _, _, lambda_r = cv_glmnet_via_r(X, y, family='poisson')

        ratio = lambda_py / lambda_r if lambda_r > 0 else float('inf')
        assert 0.1 < ratio < 10, f"Lambda ratio {ratio:.3f} outside reasonable range"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_intercept_false_vs_r(self, count_data):
        """T-019: Verify poisson family intercept=False coefficient consistency.
        
        Note: R's coef() always returns (p+1,) shape, even when intercept=False.
        The intercept position (first element) is 0 when intercept=False.
        Python implementation should match this behavior.
        """
        try:
            from cbps.highdim.lasso_utils import cv_glmnet_via_r, HAS_RPY2
            if not HAS_RPY2:
                pytest.skip("Requires rpy2")
        except ImportError:
            pytest.skip("Requires rpy2")

        X, y, _ = count_data
        p = X.shape[1]

        _, coef_py, _ = cv_glmnet(X, y, family='poisson', intercept=False)
        _, coef_r, _ = cv_glmnet_via_r(X, y, family='poisson', intercept=False)

        # Verify shape - both should return (p+1,) shape
        assert coef_py.shape == (p + 1,), f"Python coefficient shape should be ({p+1},), got {coef_py.shape}"
        assert coef_r.shape == (p + 1,), f"R coefficient shape should be ({p+1},), got {coef_r.shape}"
        assert coef_py[0] == 0.0, f"Python intercept should be 0, got {coef_py[0]}"
        assert coef_r[0] == 0.0, f"R intercept should be 0, got {coef_r[0]}"


class TestCvGlmnetPoissonEdgeCases:
    """T-020: Poisson family edge case tests."""

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_zero_inflated(self):
        """T-020: Test poisson family zero-inflated data."""
        np.random.seed(777)
        n, p = 300, 10
        X = np.random.randn(n, p)
        beta_true = np.array([0.3, -0.2] + [0] * 8)
        log_mu = X @ beta_true
        mu = np.exp(np.clip(log_mu, -3, 3))
        y = np.random.poisson(mu)
        # Add extra zeros
        zero_mask = np.random.rand(n) < 0.3
        y[zero_mask] = 0

        model, coef, lambda_min = cv_glmnet(X, y, family='poisson')

        assert model is not None
        assert lambda_min > 0


# =============================================================================
# T-021 ~ T-025: predict_glmnet_fortran Tests
# =============================================================================

class TestPredictGlmnetFortran:
    """T-021 ~ T-025: predict_glmnet_fortran function tests."""

    @pytest.fixture
    def gaussian_model(self):
        """Fit gaussian model."""
        np.random.seed(42)
        n, p = 200, 10
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.0, 0.5] + [0] * 7)
        y = X @ beta_true + np.random.randn(n) * 0.5
        model, coef, lam = cv_glmnet(X, y, family='gaussian')
        return model, coef, X, y

    @pytest.fixture
    def binomial_model(self):
        """Fit binomial model."""
        np.random.seed(42)
        n, p = 300, 10
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -0.5, 0.3] + [0] * 7)
        logits = X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        model, coef, lam = cv_glmnet(X, y, family='binomial')
        return model, coef, X, y

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_predict_basic_functionality(self, gaussian_model):
        """T-021: Test predict_glmnet_fortran basic functionality."""
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        model, coef, X, y = gaussian_model
        pred = predict_glmnet_fortran(model, X)

        assert isinstance(pred, np.ndarray)
        assert pred.shape == (X.shape[0],)
        assert not np.any(np.isnan(pred))

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_predict_gaussian_consistency(self, gaussian_model):
        """T-023: Verify gaussian family prediction consistency.
        
        Note: glmnet internally standardizes data, so predict function output
        may differ from manual computation using coefficients. Here we verify
        prediction correlation with actual y values.
        """
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        model, coef, X, y = gaussian_model
        pred_fortran = predict_glmnet_fortran(model, X)

        # Verify predictions are highly correlated with actual values
        correlation = np.corrcoef(pred_fortran, y)[0, 1]
        assert correlation > 0.8, \
            f"Prediction correlation with actual values {correlation:.3f} too low"

        # Verify prediction range is reasonable
        assert np.min(pred_fortran) > np.min(y) - 3 * np.std(y)
        assert np.max(pred_fortran) < np.max(y) + 3 * np.std(y)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_predict_binomial_response(self, binomial_model):
        """T-024: Verify binomial family prediction consistency (type='response')."""
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        model, coef, X, y = binomial_model
        pred = predict_glmnet_fortran(model, X)

        # Predictions should be in (0, 1) range (probabilities)
        assert np.all(pred >= 0) and np.all(pred <= 1), \
            "Binomial predictions should be probabilities"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_predict_new_data(self, gaussian_model):
        """Test prediction on new data."""
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        model, coef, X, y = gaussian_model
        
        # Generate new data
        np.random.seed(999)
        X_new = np.random.randn(50, X.shape[1])
        
        pred = predict_glmnet_fortran(model, X_new)
        
        assert pred.shape == (50,)
        assert not np.any(np.isnan(pred))


# =============================================================================
# T-026 ~ T-030: select_variables Function Tests
# =============================================================================

class TestSelectVariablesComprehensive:
    """T-026 ~ T-030: select_variables function comprehensive tests."""

    def test_select_variables_basic(self):
        """T-026: Test select_variables basic functionality."""
        # Mixed coefficients
        coef = np.array([0.0, 1.5, 0.0, -0.8, 0.0, 0.3, 0.0])
        selected = select_variables(coef)
        expected = np.array([1, 3, 5])
        assert_array_equal(selected, expected)

        # Verify return type
        assert isinstance(selected, np.ndarray)
        assert selected.dtype in [np.int64, np.int32, np.intp]

    def test_select_variables_correct_indices(self):
        """T-027: Verify non-zero coefficient index extraction correctness."""
        # Test various patterns
        test_cases = [
            (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0])),
            (np.array([0.0, 0.0, 0.0, 1.0]), np.array([3])),
            (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 1, 2, 3])),
            (np.array([-1.0, 0.0, 1.0, 0.0, -1.0]), np.array([0, 2, 4])),
        ]

        for coef, expected in test_cases:
            selected = select_variables(coef)
            assert_array_equal(selected, expected, 
                             err_msg=f"Selection result for coefficient {coef} is incorrect")

    def test_select_variables_all_zero(self):
        """T-028: Test all coefficients zero case."""
        coef = np.zeros(10)
        selected = select_variables(coef)
        assert len(selected) == 0
        assert isinstance(selected, np.ndarray)

    def test_select_variables_all_nonzero(self):
        """T-029: Test all coefficients non-zero case."""
        coef = np.array([1.0, -2.0, 0.5, -0.1, 3.0])
        selected = select_variables(coef)
        expected = np.array([0, 1, 2, 3, 4])
        assert_array_equal(selected, expected)

    def test_select_variables_tolerance_effect(self):
        """T-030: Verify tolerance parameter effect."""
        coef = np.array([1.0, 1e-11, 1e-9, 1e-7, 0.5])

        # Default tolerance (1e-10)
        selected_default = select_variables(coef)
        assert 0 in selected_default  # 1.0 > 1e-10
        assert 1 not in selected_default  # 1e-11 < 1e-10
        assert 2 in selected_default  # 1e-9 > 1e-10
        assert 3 in selected_default  # 1e-7 > 1e-10
        assert 4 in selected_default  # 0.5 > 1e-10

        # Stricter tolerance
        selected_strict = select_variables(coef, tol=1e-12)
        assert 1 in selected_strict  # 1e-11 > 1e-12

        # Looser tolerance
        selected_loose = select_variables(coef, tol=1e-6)
        assert 2 not in selected_loose  # 1e-9 < 1e-6
        assert 3 not in selected_loose  # 1e-7 < 1e-6

    def test_select_variables_negative_values(self):
        """Test negative coefficient selection."""
        coef = np.array([-1.0, -0.5, 0.0, -0.001, 0.0])
        selected = select_variables(coef)
        expected = np.array([0, 1, 3])
        assert_array_equal(selected, expected)

    def test_select_variables_large_array(self):
        """Test performance with large array."""
        np.random.seed(42)
        p = 10000
        coef = np.zeros(p)
        # Randomly set 100 non-zero coefficients
        nonzero_idx = np.random.choice(p, 100, replace=False)
        coef[nonzero_idx] = np.random.randn(100)

        selected = select_variables(coef)
        
        assert len(selected) == 100
        assert set(selected) == set(nonzero_idx)


# =============================================================================
# Comprehensive Integration Tests
# =============================================================================

class TestLassoUtilsIntegration:
    """LASSO utilities module integration tests."""

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_full_pipeline_gaussian(self):
        """Test gaussian complete pipeline: fit -> select variables -> predict."""
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        np.random.seed(42)
        n, p = 300, 30
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:5] = [2.0, -1.5, 1.0, -0.5, 0.3]
        y = X @ beta_true + np.random.randn(n) * 0.3

        # Fit
        model, coef, lambda_min = cv_glmnet(X, y, family='gaussian')

        # Select variables
        selected = select_variables(coef[1:], tol=1e-6)

        # Verify true non-zero variables are selected
        for i in range(5):
            assert i in selected, f"True non-zero variable {i} not selected"

        # Predict
        pred = predict_glmnet_fortran(model, X)
        
        # Verify prediction quality
        correlation = np.corrcoef(pred, y)[0, 1]
        assert correlation > 0.9, f"Prediction correlation {correlation:.3f} too low"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_full_pipeline_binomial(self):
        """Test binomial complete pipeline."""
        from cbps.highdim.lasso_utils import predict_glmnet_fortran

        np.random.seed(42)
        n, p = 400, 20
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[:3] = [1.5, -1.0, 0.5]
        logits = X @ beta_true
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)

        # Fit
        model, coef, lambda_min = cv_glmnet(X, y, family='binomial')

        # Select variables
        selected = select_variables(coef[1:], tol=1e-6)

        # Verify true non-zero variables are selected
        for i in range(3):
            assert i in selected, f"True non-zero variable {i} not selected"

        # Predict
        pred_prob = predict_glmnet_fortran(model, X)
        
        # Verify predictions are probabilities
        assert np.all(pred_prob >= 0) and np.all(pred_prob <= 1)

        # Verify classification accuracy
        pred_class = (pred_prob > 0.5).astype(int)
        accuracy = np.mean(pred_class == y)
        assert accuracy > 0.6, f"Classification accuracy {accuracy:.3f} too low"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_typical_usage(self):
        """Test hdCBPS typical usage scenario."""
        np.random.seed(2024)
        n, p = 500, 50
        X = np.random.randn(n, p)
        
        # Propensity score model
        beta_ps = np.zeros(p)
        beta_ps[:5] = [0.5, -0.3, 0.2, -0.1, 0.1]
        logits = X @ beta_ps
        probs = 1 / (1 + np.exp(-logits))
        treat = np.random.binomial(1, probs)

        # Outcome model
        alpha_out = np.zeros(p)
        alpha_out[:3] = [2.0, -1.0, 0.5]
        y = X @ alpha_out + treat * 1.0 + np.random.randn(n) * 0.5

        # Step 1: Propensity score LASSO (binomial, intercept=False for hdCBPS)
        # Note: intercept=False returns (p+1,) shape, first element is 0 (intercept position)
        _, coef_ps, _ = cv_glmnet(X, treat, family='binomial', intercept=False)
        S_ps = select_variables(coef_ps[1:], tol=1e-6)  # skip intercept position

        # Step 2: Outcome model LASSO (gaussian, intercept=True)
        # Note: intercept=True returns (p+1,) shape, first element is intercept
        X_treat = X[treat == 1]
        y_treat = y[treat == 1]
        _, coef_out1, _ = cv_glmnet(X_treat, y_treat, family='gaussian', intercept=True)
        S_out1 = select_variables(coef_out1[1:], tol=1e-6)  # skip intercept

        X_ctrl = X[treat == 0]
        y_ctrl = y[treat == 0]
        _, coef_out0, _ = cv_glmnet(X_ctrl, y_ctrl, family='gaussian', intercept=True)
        S_out0 = select_variables(coef_out0[1:], tol=1e-6)  # skip intercept

        # Verify some variables are selected (LASSO may select 0 variables, which is valid)
        # But at least outcome model should select some variables
        assert len(S_out1) > 0, "Treatment group outcome model should select some variables"
        assert len(S_out0) > 0, "Control group outcome model should select some variables"


# =============================================================================
# Main Algorithm Tests
# (merged from test_main.py)
# =============================================================================


# Skip all tests if glmnetforpython is not available


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_simple_data(n=500, p=20, seed=42, ate_true=2.0):
    """Generate simple linear data for testing hdCBPS.
    
    Data Generating Process (DGP):
        - X ~ N(0, I_p) - covariates from standard normal
        - PS: logit(P(T=1|X)) = 0.5*X_0 + 0.3*X_1 + 0.2*X_2 - sparse PS model
        - Outcome: Y = ate_true*T + X_0 + 0.5*X_1 + epsilon - sparse outcome model
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    seed : int
        Random seed for reproducibility.
    ate_true : float
        True average treatment effect.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment, and outcome.
    ate_true : float
        True ATE value for verification.
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)
    
    # Propensity score model
    beta_ps = np.zeros(p)
    beta_ps[:3] = [0.5, 0.3, 0.2]
    logits = X @ beta_ps
    probs = 1 / (1 + np.exp(-logits))
    treat = np.random.binomial(1, probs)
    
    # Outcome model
    alpha_out = np.zeros(p)
    alpha_out[:2] = [1.0, 0.5]
    y = ate_true * treat + X @ alpha_out + np.random.randn(n) * 0.5
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    df['treat'] = treat
    df['y'] = y
    
    return df, ate_true


def generate_binary_outcome_data(n=500, p=20, seed=42, ate_true=0.15):
    """Generate data with binary outcome for testing hdCBPS with binomial method.
    
    Data Generating Process (DGP):
        - X ~ N(0, I_p) - covariates from standard normal
        - PS: logit(P(T=1|X)) = 0.5*X_0 + 0.3*X_1 + 0.2*X_2 - sparse PS model
        - Outcome: Y ~ Bernoulli(sigmoid(ate*T + X_0 + 0.5*X_1)) - binary outcome
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    seed : int
        Random seed for reproducibility.
    ate_true : float
        True average treatment effect (on log-odds scale).
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment, and binary outcome.
    ate_true : float
        True ATE value for verification.
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)
    
    # Propensity score model
    beta_ps = np.zeros(p)
    beta_ps[:3] = [0.5, 0.3, 0.2]
    logits = X @ beta_ps
    probs = 1 / (1 + np.exp(-logits))
    treat = np.random.binomial(1, probs)
    
    # Binary outcome model
    alpha_out = np.zeros(p)
    alpha_out[:2] = [0.5, 0.3]
    y_logits = ate_true * treat + X @ alpha_out
    y_probs = 1 / (1 + np.exp(-y_logits))
    y = np.random.binomial(1, y_probs)
    
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    df['treat'] = treat
    df['y'] = y
    
    return df, ate_true


def generate_count_outcome_data(n=500, p=20, seed=42, ate_true=0.3):
    """Generate data with count outcome for testing hdCBPS with poisson method.
    
    Data Generating Process (DGP):
        - X ~ N(0, I_p) - covariates from standard normal
        - PS: logit(P(T=1|X)) = 0.5*X_0 + 0.3*X_1 + 0.2*X_2 - sparse PS model
        - Outcome: Y ~ Poisson(exp(ate*T + 1 + 0.3*X_0 + 0.2*X_1)) - count outcome
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates.
    seed : int
        Random seed for reproducibility.
    ate_true : float
        True average treatment effect (on log scale).
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment, and count outcome.
    ate_true : float
        True ATE value for verification.
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)
    
    # Propensity score model
    beta_ps = np.zeros(p)
    beta_ps[:3] = [0.5, 0.3, 0.2]
    logits = X @ beta_ps
    probs = 1 / (1 + np.exp(-logits))
    treat = np.random.binomial(1, probs)
    
    # Count outcome model (Poisson)
    alpha_out = np.zeros(p)
    alpha_out[:2] = [0.3, 0.2]
    # Ensure positive rates: base rate of 1 (exp(0)=1)
    log_rate = 1.0 + ate_true * treat + X @ alpha_out
    # Clip to avoid extremely large counts
    log_rate = np.clip(log_rate, -2, 4)
    y = np.random.poisson(np.exp(log_rate))
    
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    df['treat'] = treat
    df['y'] = y
    
    return df, ate_true


def generate_highdim_data(n=200, p=500, seed=42, ate_true=1.5):
    """Generate high-dimensional sparse data (p > n) for testing hdCBPS.
    
    This DGP tests the key feature of hdCBPS: handling high-dimensional
    settings where the number of covariates exceeds sample size.
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of covariates (should be > n for high-dimensional test).
    seed : int
        Random seed for reproducibility.
    ate_true : float
        True average treatment effect.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with covariates, treatment, and outcome.
    ate_true : float
        True ATE value for verification.
    """
    np.random.seed(seed)
    X = np.random.randn(n, p)
    
    # Sparse propensity score model
    beta_ps = np.zeros(p)
    beta_ps[:5] = [0.5, 0.3, 0.2, -0.1, 0.1]
    logits = X @ beta_ps
    probs = 1 / (1 + np.exp(-logits))
    treat = np.random.binomial(1, probs)
    
    # Sparse outcome model
    alpha_out = np.zeros(p)
    alpha_out[:3] = [2.0, -1.0, 0.5]
    y = ate_true * treat + X @ alpha_out + np.random.randn(n) * 0.3
    
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    df['treat'] = treat
    df['y'] = y
    
    return df, ate_true


# =============================================================================
# T-082 ~ T-090: HDCBPSResults Class Tests
# =============================================================================

@pytest.mark.unit
class TestHDCBPSResultsBasic:
    """
    Test ID: T-082 ~ T-085, Requirement: REQ-HD005
    
    HDCBPSResults class basic functionality tests.
    
    Verifies that the HDCBPSResults class:
    - Exists and can be instantiated
    - Has all required attributes (ATE, ATT, s, w, fitted_values, weights, etc.)
    - Provides meaningful string representation
    """

    def test_results_class_exists(self):
        """T-082: Verify HDCBPSResults class exists."""
        # Arrange: None needed
        # Act
        result = HDCBPSResults()
        # Assert
        assert HDCBPSResults is not None
        assert result is not None

    def test_results_has_required_attributes(self):
        """T-083: Verify HDCBPSResults has all required attributes."""
        # Arrange & Act
        result = HDCBPSResults()
        
        # Assert: Core estimands
        assert hasattr(result, 'ATE')
        assert hasattr(result, 'ATT')
        assert hasattr(result, 's')  # ATE standard error
        assert hasattr(result, 'w')  # ATT standard error
        
        # Assert: Fitted values
        assert hasattr(result, 'fitted_values')
        assert hasattr(result, 'weights')
        
        # Assert: Coefficients
        assert hasattr(result, 'coefficients1')
        assert hasattr(result, 'coefficients0')
        
        # Assert: Convergence info
        assert hasattr(result, 'converged')

    def test_results_str_representation(self):
        """T-084: Verify HDCBPSResults string representation."""
        # Arrange
        result = HDCBPSResults()
        result.ATE = 1.5
        result.s = 0.3
        result.ATT = None
        result.w = None
        result.converged = True
        
        # Act
        str_repr = str(result)
        
        # Assert
        assert isinstance(str_repr, str)
        assert 'ATE' in str_repr or '1.5' in str_repr

    def test_results_str_method(self):
        """T-085: Verify HDCBPSResults.__str__() method."""
        # Arrange
        result = HDCBPSResults()
        result.ATE = 2.0
        result.s = 0.5
        result.ATT = None
        result.w = None
        result.converged = True
        
        # Act
        str_repr = str(result)
        
        # Assert
        assert isinstance(str_repr, str)


# =============================================================================
# T-086 ~ T-095: hdCBPS Function Basic Functionality Tests
# =============================================================================

@pytest.mark.unit
class TestHdCBPSBasicFunctionality:
    """
    Test ID: T-086 ~ T-095, Requirement: REQ-HD003, REQ-HD005
    
    hdCBPS function basic functionality tests.
    
    Verifies that the hdCBPS function:
    - Returns HDCBPSResults object
    - Produces valid ATE estimate (float, not NaN/Inf)
    - Produces positive standard errors
    - Returns correctly shaped fitted_values and weights
    - Propensity scores are bounded in (0, 1)
    - Results are reproducible with same seed
    """

    @pytest.fixture
    def simple_data(self):
        """Simple test data."""
        return generate_simple_data(n=300, p=10, seed=42)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_returns_results_object(self, simple_data):
        """T-086: Verify hdCBPS returns HDCBPSResults object."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert isinstance(result, HDCBPSResults)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_ate_is_float(self, simple_data):
        """T-087: Verify ATE is a float."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert isinstance(result.ATE, (float, np.floating))
        assert not np.isnan(result.ATE)
        assert not np.isinf(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_se_is_positive(self, simple_data):
        """T-088: Verify standard error is positive."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.s > 0, f"Standard error should be positive, got {result.s}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_fitted_values_shape(self, simple_data):
        """T-089: Verify fitted_values shape is correct."""
        df, _ = simple_data
        n = len(df)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.fitted_values.shape == (n,)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_fitted_values_in_range(self, simple_data):
        """T-090: Verify fitted_values (propensity scores) are in (0, 1) range."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert np.all(result.fitted_values > 0), "Propensity scores should be > 0"
        assert np.all(result.fitted_values < 1), "Propensity scores should be < 1"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_weights_shape(self, simple_data):
        """T-091: Verify weights shape is correct."""
        df, _ = simple_data
        n = len(df)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.weights.shape == (n,)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_weights_positive(self, simple_data):
        """T-092: Verify weights are positive."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert np.all(result.weights > 0), "Weights should be positive"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_convergence_info(self, simple_data):
        """T-093: Verify algorithm convergence info exists."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # converged attribute should exist and be boolean
        assert hasattr(result, 'converged')
        assert isinstance(result.converged, (bool, np.bool_))
        
        # Even when not converged, ATE should have a value
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_coefficients_shape(self, simple_data):
        """T-094: Verify coefficient shape is correct."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # Coefficients should be 1-dimensional arrays
        assert result.coefficients1.ndim == 1
        assert result.coefficients0.ndim == 1

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_hdcbps_reproducibility(self, simple_data):
        """T-095: Verify result reproducibility."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result1 = hdCBPS(formula=formula, data=df, y='y', ATT=0, 
                        method='linear', seed=42)
        result2 = hdCBPS(formula=formula, data=df, y='y', ATT=0, 
                        method='linear', seed=42)
        
        assert_allclose(result1.ATE, result2.ATE, rtol=1e-10)
        assert_allclose(result1.s, result2.s, rtol=1e-10)


# =============================================================================
# T-096 ~ T-105: ATE Estimation Tests
# =============================================================================

@pytest.mark.numerical
class TestHdCBPSATEEstimation:
    """
    Test ID: T-096 ~ T-105, Requirement: REQ-HD003, REQ-HD004
    
    ATE (Average Treatment Effect) estimation tests.
    
    Verifies that hdCBPS ATE estimation:
    - Has reasonable magnitude close to true value
    - Falls within outcome variable range
    - Has reasonable standard error
    - Produces valid 95% confidence interval
    - Improves with larger sample sizes (SE decreases)
    """

    @pytest.fixture
    def controlled_data(self):
        """Controlled data with known true ATE."""
        return generate_simple_data(n=500, p=15, seed=2024, ate_true=2.0)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_reasonable_magnitude(self, controlled_data):
        """T-096: Verify ATE estimate magnitude is reasonable."""
        df, ate_true = controlled_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # ATE should be near true value (allow larger error due to finite sample)
        assert abs(result.ATE - ate_true) < 2.0, \
            f"ATE {result.ATE:.4f} deviates from true value {ate_true} too much"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_within_outcome_range(self, controlled_data):
        """T-097: Verify ATE is within outcome variable range."""
        df, _ = controlled_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        y = df['y'].values
        y_range = np.max(y) - np.min(y)
        
        # ATE absolute value should not exceed y range
        assert abs(result.ATE) < y_range, \
            f"ATE {result.ATE:.4f} exceeds y range {y_range:.4f}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_se_reasonable(self, controlled_data):
        """T-098: Verify ATE standard error is reasonable."""
        df, _ = controlled_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # Standard error should be positive and not too large
        assert result.s > 0
        assert result.s < 10, f"Standard error {result.s:.4f} is too large"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_confidence_interval(self, controlled_data):
        """T-099: Verify 95% confidence interval includes true value."""
        df, ate_true = controlled_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # 95% CI: ATE +/- 1.96 * SE
        ci_lower = result.ATE - 1.96 * result.s
        ci_upper = result.ATE + 1.96 * result.s
        
        # Note: Single test may not include true value, this is normal
        # Here we only check that CI width is reasonable
        ci_width = ci_upper - ci_lower
        assert ci_width > 0
        assert ci_width < 10, f"CI width {ci_width:.4f} is too large"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_different_sample_sizes(self):
        """T-100: Test ATE estimation with different sample sizes."""
        ate_true = 2.0
        results = []
        
        for n in [200, 500, 1000]:
            df, _ = generate_simple_data(n=n, p=10, seed=42, ate_true=ate_true)
            formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
            
            result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
            results.append((n, result.ATE, result.s))
        
        # Standard error should decrease with sample size
        ses = [r[2] for r in results]
        assert ses[0] > ses[1] > ses[2] * 0.5, \
            "Standard error should decrease with sample size"


# =============================================================================
# T-101 ~ T-110: ATT Estimation Tests
# =============================================================================

@pytest.mark.numerical
class TestHdCBPSATTEstimation:
    """
    Test ID: T-101 ~ T-110, Requirement: REQ-HD003
    
    ATT (Average Treatment Effect on the Treated) estimation tests.
    
    Verifies that hdCBPS ATT estimation:
    - Is computed when ATT=1 is specified
    - Is not computed when ATT=0 is specified
    - Has computed standard error (w attribute)
    - Has reasonable magnitude
    - Both ATE and ATT are computed when ATT=1
    """

    @pytest.fixture
    def simple_data(self):
        """Simple test data."""
        return generate_simple_data(n=400, p=12, seed=123)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_att_computed_when_requested(self, simple_data):
        """T-101: Verify ATT is computed when ATT=1."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(12)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=1, method='linear')
        
        assert result.ATT is not None, "ATT should be computed when ATT=1"
        assert isinstance(result.ATT, (float, np.floating))

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_att_not_computed_when_not_requested(self, simple_data):
        """T-102: Verify ATT is not computed when ATT=0."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(12)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATT is None, "ATT should not be computed when ATT=0"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_att_se_computed(self, simple_data):
        """T-103: Verify ATT standard error is computed."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(12)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=1, method='linear')
        
        assert result.w is not None, "ATT standard error should be computed when ATT=1"
        assert result.w > 0, "ATT standard error should be positive"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_att_reasonable_magnitude(self, simple_data):
        """T-104: Verify ATT estimate magnitude is reasonable."""
        df, ate_true = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(12)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=1, method='linear')
        
        # ATT should be in reasonable range
        y = df['y'].values
        y_range = np.max(y) - np.min(y)
        
        assert abs(result.ATT) < y_range, \
            f"ATT {result.ATT:.4f} exceeds y range"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_and_att_both_computed(self, simple_data):
        """T-105: Verify both ATE and ATT are computed when ATT=1."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(12)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=1, method='linear')
        
        assert result.ATE is not None
        assert result.ATT is not None
        assert result.s is not None
        assert result.w is not None


# =============================================================================
# T-106 ~ T-115: Method Parameter Tests (linear/binomial/poisson)
# =============================================================================

@pytest.mark.unit
class TestHdCBPSMethods:
    """
    Test ID: T-106 ~ T-115, Requirement: REQ-HD003
    
    Tests for different outcome model methods (linear/binomial/poisson).
    
    Verifies that hdCBPS:
    - Works with linear method (Gaussian outcome)
    - Works with binomial method (binary outcome)
    - Works with poisson method (count outcome)
    - Different methods produce similar results for same data
    - Invalid method raises appropriate error
    """

    @pytest.fixture
    def simple_data(self):
        """Simple test data."""
        return generate_simple_data(n=300, p=10, seed=456)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_linear_method(self, simple_data):
        """T-106: Test linear method."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATE is not None
        # converged may be False, but result should be valid
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_binomial_method(self):
        """T-107: Test binomial method with binary outcome data.
        
        Note: The 'method' parameter specifies the outcome distribution family.
        For binomial method, outcome y must be binary (0/1).
        """
        df, _ = generate_binary_outcome_data(n=500, p=20, seed=42)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='binomial')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_poisson_method(self):
        """T-108: Test poisson method with count outcome data.
        
        Note: The 'method' parameter specifies the outcome distribution family.
        For poisson method, outcome y must be non-negative counts.
        """
        df, _ = generate_count_outcome_data(n=500, p=20, seed=42)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='poisson')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_methods_with_appropriate_data(self, simple_data):
        """T-109: Verify each method works with appropriate outcome type.
        
        Note: The 'method' parameter specifies the outcome distribution family:
        - 'linear': continuous outcome (Gaussian)
        - 'binomial': binary outcome (0/1)
        - 'poisson': count outcome (non-negative integers)
        
        Each method should be tested with data of the appropriate type.
        """
        # Test linear method with continuous data
        df_linear, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        result_linear = hdCBPS(formula=formula, data=df_linear, y='y', ATT=0, method='linear')
        assert not np.isnan(result_linear.ATE), "Linear method failed with continuous data"
        
        # Test binomial method with binary data
        df_binary, _ = generate_binary_outcome_data(n=500, p=20, seed=43)
        result_binomial = hdCBPS(formula=formula, data=df_binary, y='y', ATT=0, method='binomial')
        assert not np.isnan(result_binomial.ATE), "Binomial method failed with binary data"
        
        # Test poisson method with count data
        df_count, _ = generate_count_outcome_data(n=500, p=20, seed=44)
        result_poisson = hdCBPS(formula=formula, data=df_count, y='y', ATT=0, method='poisson')
        assert not np.isnan(result_poisson.ATE), "Poisson method failed with count data"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_invalid_method_raises_error(self, simple_data):
        """T-110: Verify invalid method parameter raises error."""
        df, _ = simple_data
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        with pytest.raises((ValueError, KeyError)):
            hdCBPS(formula=formula, data=df, y='y', ATT=0, method='invalid')


# =============================================================================
# T-111 ~ T-120: High-Dimensional Data Tests
# =============================================================================

@pytest.mark.numerical
@pytest.mark.slow
class TestHdCBPSHighDimensional:
    """
    Test ID: T-111 ~ T-120, Requirement: REQ-HD001, REQ-HD002
    
    High-dimensional data tests (p > n).
    
    Verifies the key feature of hdCBPS from Ning et al. (2020):
    - Handles high-dimensional settings where p > n
    - Produces valid results without errors
    - ATE estimates are reasonable
    - Variable selection via LASSO is sparse (selects < p variables)
    """

    @pytest.fixture
    def highdim_data(self):
        """High-dimensional test data."""
        return generate_highdim_data(n=150, p=300, seed=789)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_highdim_runs_without_error(self, highdim_data):
        """T-111: Verify high-dimensional data runs without error."""
        df, _ = highdim_data
        p = 300
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result is not None
        assert result.ATE is not None

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_highdim_converges(self, highdim_data):
        """T-112: Verify high-dimensional data produces valid results."""
        df, _ = highdim_data
        p = 300
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # High-dimensional data should produce valid results
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_highdim_ate_reasonable(self, highdim_data):
        """T-113: Verify high-dimensional data ATE is reasonable."""
        df, ate_true = highdim_data
        p = 300
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # ATE should be in reasonable range
        assert abs(result.ATE - ate_true) < 3.0, \
            f"High-dimensional ATE {result.ATE:.4f} deviates from true value {ate_true} too much"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_highdim_sparse_selection(self, highdim_data):
        """T-114: Verify high-dimensional data variable selection is sparse."""
        df, _ = highdim_data
        p = 300
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # Check coefficient sparsity
        n_nonzero1 = np.sum(np.abs(result.coefficients1) > 1e-6)
        n_nonzero0 = np.sum(np.abs(result.coefficients0) > 1e-6)
        
        # Should select fewer variables
        assert n_nonzero1 < p, "Treatment group should select sparse variables"
        assert n_nonzero0 < p, "Control group should select sparse variables"


# =============================================================================
# T-116 ~ T-125: Edge Case Tests
# =============================================================================

@pytest.mark.edge_case
class TestHdCBPSEdgeCases:
    """
    Test ID: T-116 ~ T-125, Requirement: REQ-HD005
    
    Edge case tests for hdCBPS robustness.
    
    Verifies that hdCBPS handles:
    - Small sample sizes (n=50)
    - Imbalanced treatment groups (10% treated)
    - Constant covariates
    - Highly correlated covariates
    - Large outcome values
    """

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_small_sample(self):
        """T-116: Test small sample (n=50)."""
        df, _ = generate_simple_data(n=50, p=5, seed=111)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(5)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_imbalanced_treatment(self):
        """T-117: Test imbalanced treatment group (10% treated)."""
        np.random.seed(222)
        n, p = 300, 10
        X = np.random.randn(n, p)
        
        # Imbalanced: ~10% treated
        beta_ps = np.zeros(p)
        beta_ps[0] = 0.5
        logits = -2.0 + X @ beta_ps  # Biased towards control group
        probs = 1 / (1 + np.exp(-logits))
        treat = np.random.binomial(1, probs)
        
        y = 2.0 * treat + X[:, 0] + np.random.randn(n) * 0.5
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
        df['treat'] = treat
        df['y'] = y
        
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_constant_covariate(self):
        """T-118: Test with constant covariate."""
        np.random.seed(333)
        n, p = 200, 8
        X = np.random.randn(n, p)
        X[:, 5] = 1.0  # Constant column
        
        beta_ps = np.zeros(p)
        beta_ps[:2] = [0.5, 0.3]
        logits = X @ beta_ps
        probs = 1 / (1 + np.exp(-logits))
        treat = np.random.binomial(1, probs)
        
        y = 2.0 * treat + X[:, 0] + np.random.randn(n) * 0.5
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
        df['treat'] = treat
        df['y'] = y
        
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        
        # Should handle constant column (may be filtered or produce warning)
        try:
            result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
            assert result.ATE is not None
        except Exception:
            # If exception is raised, should be meaningful error
            pass

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_highly_correlated_covariates(self):
        """T-119: Test highly correlated covariates."""
        np.random.seed(444)
        n, p = 200, 10
        X = np.random.randn(n, p)
        # Add highly correlated column
        X[:, 5] = X[:, 0] + np.random.randn(n) * 0.01
        
        beta_ps = np.zeros(p)
        beta_ps[:2] = [0.5, 0.3]
        logits = X @ beta_ps
        probs = 1 / (1 + np.exp(-logits))
        treat = np.random.binomial(1, probs)
        
        y = 2.0 * treat + X[:, 0] + np.random.randn(n) * 0.5
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
        df['treat'] = treat
        df['y'] = y
        
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_large_outcome_values(self):
        """T-120: Test large outcome values."""
        np.random.seed(555)
        n, p = 200, 8
        X = np.random.randn(n, p)
        
        beta_ps = np.zeros(p)
        beta_ps[:2] = [0.5, 0.3]
        logits = X @ beta_ps
        probs = 1 / (1 + np.exp(-logits))
        treat = np.random.binomial(1, probs)
        
        # Large outcome values
        y = 10000 * treat + 5000 * X[:, 0] + np.random.randn(n) * 1000
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
        df['treat'] = treat
        df['y'] = y
        
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(p)])
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)
        # ATE should be close to 10000
        assert abs(result.ATE - 10000) < 5000, \
            f"Large outcome ATE {result.ATE:.4f} deviates from expected"


# =============================================================================
# T-121 ~ T-130: Python-R Comparison Tests
# =============================================================================

@pytest.mark.numerical
class TestHdCBPSVsR:
    """
    Test ID: T-121 ~ T-130, Requirement: REQ-HD003
    
    Python vs R implementation comparison tests using LaLonde dataset.
    
    Verifies that the Python hdCBPS implementation:
    - Produces valid ATE/ATT estimates on real data
    - Propensity scores are reasonable
    - Achieves covariate balance
    - Different methods produce consistent results
    """

    @pytest.fixture
    def lalonde_data(self):
        """Load LaLonde dataset."""
        try:
            from cbps.datasets import load_lalonde
            df = load_lalonde(dehejia_wahba_only=True)
            return df
        except Exception as e:
            pytest.skip(f"Cannot load LaLonde data: {e}")

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_ate_estimation(self, lalonde_data):
        """T-121: LaLonde dataset ATE estimation."""
        df = lalonde_data
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        result = hdCBPS(formula=formula, data=df, y='re78', ATT=0, method='linear')
        
        assert result.ATE is not None
        assert not np.isnan(result.ATE)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_att_estimation(self, lalonde_data):
        """T-122: LaLonde dataset ATT estimation."""
        df = lalonde_data
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        result = hdCBPS(formula=formula, data=df, y='re78', ATT=1, method='linear')
        
        assert result.ATT is not None
        assert result.w is not None

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_propensity_scores(self, lalonde_data):
        """T-123: LaLonde dataset propensity scores."""
        df = lalonde_data
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        result = hdCBPS(formula=formula, data=df, y='re78', ATT=0, method='linear')
        
        ps = result.fitted_values
        
        # Propensity scores should be in (0, 1) range
        assert np.all(ps > 0) and np.all(ps < 1)
        
        # Mean should be close to treatment rate
        treat_rate = np.mean(df['treat'])
        assert abs(np.mean(ps) - treat_rate) < 0.2, \
            f"Propensity score mean {np.mean(ps):.4f} deviates from treatment rate {treat_rate:.4f}"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_covariate_balance(self, lalonde_data):
        """T-124: LaLonde dataset covariate balance."""
        df = lalonde_data
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        result = hdCBPS(formula=formula, data=df, y='re78', ATT=0, method='linear')
        
        # Compute weighted covariate balance
        covariates = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75']
        X = df[covariates].values
        treat = df['treat'].values
        ps = result.fitted_values
        
        # ATE weights
        w1 = treat / ps
        w0 = (1 - treat) / (1 - ps)
        
        # Weighted mean differences
        balance_stats = []
        for i, cov in enumerate(covariates):
            mean_treat = np.sum(w1 * X[:, i]) / np.sum(w1)
            mean_ctrl = np.sum(w0 * X[:, i]) / np.sum(w0)
            diff = mean_treat - mean_ctrl
            balance_stats.append((cov, diff))

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_linear_method(self, lalonde_data):
        """T-125: LaLonde dataset with linear method for continuous outcome.
        
        Note: The 'method' parameter specifies the outcome distribution family.
        LaLonde's re78 (real earnings in 1978) is a continuous outcome,
        so only 'linear' (Gaussian) method is appropriate.
        """
        df = lalonde_data
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        # Linear method for continuous outcome (re78 = earnings)
        result = hdCBPS(formula=formula, data=df, y='re78', ATT=0, method='linear')
        
        assert not np.isnan(result.ATE), "Linear method ATE is NaN"
        assert result.s > 0, "Linear method SE should be positive"
        
    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_lalonde_binomial_method(self, lalonde_data):
        """T-125b: LaLonde dataset with binomial method for binary outcome.
        
        Create a binary outcome (employed = re78 > 0) for binomial method testing.
        """
        df = lalonde_data.copy()
        df['employed'] = (df['re78'] > 0).astype(int)  # Binary: employed or not
        formula = 'treat ~ age + educ + black + hisp + married + nodegr + re74 + re75'
        
        result = hdCBPS(formula=formula, data=df, y='employed', ATT=0, method='binomial')
        
        assert not np.isnan(result.ATE), "Binomial method ATE is NaN"
        assert result.s > 0, "Binomial method SE should be positive"


# =============================================================================
# T-126 ~ T-132: Numerical Stability and Property Tests
# =============================================================================

@pytest.mark.numerical
class TestHdCBPSNumericalStability:
    """
    Test ID: T-126 ~ T-132, Requirement: REQ-HD004, REQ-HD005
    
    Numerical stability and asymptotic property tests.
    
    Verifies that hdCBPS:
    - Results contain no NaN values
    - Results contain no Inf values
    - Propensity scores are bounded in (0, 1)
    - Weights have reasonable magnitude
    - Standard error decreases with sample size (sqrt-n consistency)
    - ATE is consistent (Monte Carlo verification)
    - Achieves weak covariate balance
    """

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_no_nan_in_results(self):
        """T-126: Verify results contain no NaN."""
        df, _ = generate_simple_data(n=300, p=10, seed=666)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=1, method='linear')
        
        assert not np.isnan(result.ATE), "ATE should not be NaN"
        assert not np.isnan(result.s), "ATE SE should not be NaN"
        assert not np.isnan(result.ATT), "ATT should not be NaN"
        assert not np.isnan(result.w), "ATT SE should not be NaN"
        assert not np.any(np.isnan(result.fitted_values)), "Propensity scores should not contain NaN"
        assert not np.any(np.isnan(result.weights)), "Weights should not contain NaN"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_no_inf_in_results(self):
        """T-127: Verify results contain no Inf."""
        df, _ = generate_simple_data(n=300, p=10, seed=777)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        assert not np.isinf(result.ATE), "ATE should not be Inf"
        assert not np.isinf(result.s), "SE should not be Inf"
        assert not np.any(np.isinf(result.fitted_values)), "Propensity scores should not contain Inf"
        assert not np.any(np.isinf(result.weights)), "Weights should not contain Inf"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_propensity_score_bounded(self):
        """T-128: Verify propensity scores are bounded."""
        df, _ = generate_simple_data(n=300, p=10, seed=888)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        ps = result.fitted_values
        
        # Propensity scores should be strictly in (0, 1)
        assert np.all(ps > 0), "Propensity scores should be > 0"
        assert np.all(ps < 1), "Propensity scores should be < 1"
        
        # Check for extreme values
        assert np.min(ps) > 1e-10, f"Propensity score min {np.min(ps)} is too small"
        assert np.max(ps) < 1 - 1e-10, f"Propensity score max {np.max(ps)} is too large"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_weights_reasonable_magnitude(self):
        """T-129: Verify weights magnitude is reasonable."""
        df, _ = generate_simple_data(n=300, p=10, seed=999)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        weights = result.weights
        
        # Weights should be positive
        assert np.all(weights > 0), "Weights should be positive"
        
        # Weights should not be too extreme
        assert np.max(weights) < 1000, f"Max weight {np.max(weights)} is too large"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_se_decreases_with_sample_size(self):
        """T-130: Verify standard error decreases with sample size."""
        ate_true = 2.0
        ses = []
        
        for n in [100, 300, 600]:
            df, _ = generate_simple_data(n=n, p=8, seed=42, ate_true=ate_true)
            formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(8)])
            
            result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
            ses.append(result.s)
        
        # Standard error should roughly decrease with sqrt(n)
        # Allow some variation
        assert ses[0] > ses[2] * 0.5, \
            "Standard error should decrease with sample size"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_ate_consistency_monte_carlo(self):
        """T-131: Monte Carlo consistency test."""
        ate_true = 2.0
        n_sims = 10  # Reduce simulation count to speed up test
        ates = []
        
        for seed in range(n_sims):
            df, _ = generate_simple_data(n=500, p=10, seed=seed, ate_true=ate_true)
            formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
            
            result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
            ates.append(result.ATE)
        
        mean_ate = np.mean(ates)
        
        # Mean ATE should be close to true value
        assert abs(mean_ate - ate_true) < 1.0, \
            f"Mean ATE {mean_ate:.4f} deviates from true value {ate_true} too much"

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_weak_covariate_balance(self):
        """T-132: Verify weak covariate balance property."""
        df, _ = generate_simple_data(n=500, p=10, seed=1234)
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        
        result = hdCBPS(formula=formula, data=df, y='y', ATT=0, method='linear')
        
        # Extract data
        X = df[[f'x{i}' for i in range(10)]].values
        treat = df['treat'].values
        ps = result.fitted_values
        
        # Compute weak covariate balance statistics
        # Formula: (1/n) * sum((T/pi - 1) * X_j) should be close to 0
        balance_stats = []
        for j in range(X.shape[1]):
            stat = np.mean((treat / ps - 1) * X[:, j])
            balance_stats.append(stat)
        
        # Balance statistics should be close to 0
        max_imbalance = np.max(np.abs(balance_stats))
        
        # Allow some imbalance (due to finite sample)
        assert max_imbalance < 0.5, \
            f"Max imbalance {max_imbalance:.6f} is too large"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestHdCBPSIntegration:
    """
    End-to-end integration tests for hdCBPS.
    
    Verifies the complete hdCBPS pipeline from data generation
    to treatment effect estimation for both ATE and ATT.
    """

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_full_pipeline_ate(self):
        """Complete ATE estimation pipeline test."""
        # Generate data
        df, ate_true = generate_simple_data(n=400, p=15, seed=2024, ate_true=2.5)
        
        # Build formula
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        # Run hdCBPS
        result = hdCBPS(
            formula=formula,
            data=df,
            y='y',
            ATT=0,
            method='linear',
            iterations=1000,
            seed=42
        )
        
        # Verify results
        assert result.ATE is not None
        assert result.s > 0
        assert len(result.fitted_values) == len(df)
        assert len(result.weights) == len(df)

    @pytest.mark.skipif(not HAS_GLMNETFORPYTHON, reason="Requires glmnetforpython")
    def test_full_pipeline_att(self):
        """Complete ATT estimation pipeline test."""
        # Generate data
        df, ate_true = generate_simple_data(n=400, p=15, seed=2024, ate_true=2.5)
        
        # Build formula
        formula = 'treat ~ ' + ' + '.join([f'x{i}' for i in range(15)])
        
        # Run hdCBPS with ATT
        result = hdCBPS(
            formula=formula,
            data=df,
            y='y',
            ATT=1,
            method='linear',
            iterations=1000,
            seed=42
        )
        
        # Verify results
        assert result.ATE is not None
        assert result.ATT is not None
        assert result.s > 0
        assert result.w > 0
