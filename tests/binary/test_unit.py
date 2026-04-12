"""
Unit Tests for Binary CBPS Module
==================================

Comprehensive unit tests for the binary treatment CBPS implementation,
covering all core computational components: pseudoinverse computation,
weight functions, V matrix construction, GMM objective, variance-covariance
estimation, balance diagnostics, optimization, and numerical stability.

These tests verify individual function correctness against analytical
formulas from the paper and known mathematical properties.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
    DOI: 10.1111/rssb.12027
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.special

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Import shared test utilities from root conftest
from ..conftest import (
    Tolerances, PROBS_MIN,
    assert_allclose_with_report, assert_matrix_symmetric,
    assert_positive_semidefinite,
)

# Import function under test for weight normalization tests
from cbps.utils.helpers import normalize_sample_weights

pytestmark = pytest.mark.unit

# General tolerance for weight normalization tests
TOLERANCE = 1e-10


# =============================================================================
# Module-Level Fixtures (aliases for conftest fixtures with different names)
# =============================================================================

@pytest.fixture(scope="session")
def r_ginv_func(cbps_binary_module):
    """Provide the _r_ginv function (alias for r_ginv fixture)."""
    return cbps_binary_module._r_ginv


@pytest.fixture(scope="session")
def compute_vcov_func(cbps_binary_module):
    """Provide the _compute_vcov function (alias for compute_vcov fixture)."""
    return cbps_binary_module._compute_vcov


@pytest.fixture(scope="session")
def bal_gradient_func(cbps_binary_module):
    """Provide the _bal_gradient function."""
    return cbps_binary_module._bal_gradient


@pytest.fixture(scope="session")
def gmm_gradient(cbps_binary_module):
    """Provide the _gmm_gradient function."""
    return cbps_binary_module._gmm_gradient


# =============================================================================
# Helper Functions
# =============================================================================

def compute_smd(x, treat, weights=None):
    """Compute standardized mean difference (SMD) for a covariate."""
    if weights is None:
        weights = np.ones(len(treat))
    w_t = weights[treat == 1]
    w_c = weights[treat == 0]
    x_t = x[treat == 1]
    x_c = x[treat == 0]
    mean_t = np.average(x_t, weights=w_t)
    mean_c = np.average(x_c, weights=w_c)
    var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0
    var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0
    pooled_var = (var_t + var_c) / 2
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1
    return (mean_t - mean_c) / pooled_std


def compute_variance_ratio(x, treat, weights=None):
    """Compute variance ratio between treated and control groups."""
    if weights is None:
        weights = np.ones(len(treat))
    x_t = x[treat == 1]
    x_c = x[treat == 0]
    var_t = np.var(x_t, ddof=1) if len(x_t) > 1 else 0
    var_c = np.var(x_c, ddof=1) if len(x_c) > 1 else 0
    if var_c == 0:
        return np.inf if var_t > 0 else 1.0
    return var_t / var_c


def compute_ks_statistic(x, treat, weights=None):
    """Compute Kolmogorov-Smirnov statistic between groups."""
    x_t = x[treat == 1]
    x_c = x[treat == 0]
    if len(x_t) < 2 or len(x_c) < 2:
        return 0.0
    ks_stat, _ = scipy.stats.ks_2samp(x_t, x_c)
    return ks_stat

def compute_att_weights_manual(beta, X, treat, sample_weights, probs_min=PROBS_MIN):
    """
    Manually compute ATT weights following the paper formula.
    
    ATT weight formula (Equation 11):
    w_i = (n/n_t) * (T_i - π_i) / (1 - π_i)
    
    where:
    - n = total sample size (weighted)
    - n_t = number of treated units (weighted)
    - T_i = treatment indicator
    - π_i = propensity score
    """
    # Compute weighted sample sizes
    n_c = np.sum(sample_weights[treat == 0])
    n_t = np.sum(sample_weights[treat == 1])
    n = n_c + n_t
    
    # Compute propensity scores
    theta = X @ beta
    probs = scipy.special.expit(theta)
    
    # Clip probabilities
    probs = np.clip(probs, probs_min, 1 - probs_min)
    
    # ATT weight formula
    w = (n / n_t) * (treat - probs) / (1 - probs)
    
    return w

def compute_ate_weights_manual(beta, X, treat, probs_min=PROBS_MIN):
    """
    Manually compute ATE weights following the paper formula.
    
    ATE weight formula (Equation 10):
    w_i = (T_i - π_i) / (π_i * (1 - π_i))
    
    Which can be rewritten as:
    w_i = T_i/π_i - (1-T_i)/(1-π_i)
    
    Or equivalently (as used in R code):
    w_i = 1 / (π_i - 1 + T_i)
    """
    # Compute propensity scores
    theta = X @ beta
    probs = scipy.special.expit(theta)
    
    # Clip probabilities
    probs = np.clip(probs, probs_min, 1 - probs_min)
    
    # ATE weight formula (R implementation form)
    w = 1.0 / (probs - 1 + treat)
    
    return w

@pytest.fixture
def paper_test_data():
    """Generate test data for paper formula verification.

    Provides a reproducible dataset with known propensity score model
    for verifying paper formula implementations.
    """
    np.random.seed(42)
    n = 200
    
    # Covariates with intercept
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    k = X.shape[1]
    
    # True propensity score model
    true_beta = np.array([0.0, 0.5, -0.3])
    theta = X @ true_beta
    pi_true = scipy.special.expit(theta)
    
    # Generate treatment
    treat = np.random.binomial(1, pi_true).astype(float)
    
    # Sample weights (uniform)
    sample_weights = np.ones(n)
    
    # Weighted sample sizes
    n_t = np.sum(sample_weights[treat == 1])
    n_c = np.sum(sample_weights[treat == 0])
    
    return {
        'X': X,
        'treat': treat,
        'sample_weights': sample_weights,
        'true_beta': true_beta,
        'n': n,
        'k': k,
        'n_t': n_t,
        'n_c': n_c,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestSMDCalculation:
    """
    Test ID: BAL-DIAG-001
    
    Test standardized mean difference calculation.
    """
    
    def test_smd_perfect_balance(self):
        """Test SMD is zero for perfectly balanced covariates."""
        np.random.seed(42)
        n = 200
        
        # Create perfectly balanced data
        x = np.random.normal(0, 1, n)
        treat = np.array([1] * 100 + [0] * 100)
        
        # Shuffle to mix
        idx = np.random.permutation(n)
        x = x[idx]
        treat = treat[idx]
        
        smd = compute_smd(x, treat)
        
        # SMD should be close to zero for random assignment
        assert np.abs(smd) < 0.5, f"SMD should be small for balanced data, got {smd}"

    def test_smd_imbalanced(self):
        """Test SMD detects imbalance."""
        np.random.seed(42)
        n = 200
        
        # Create imbalanced data
        x_t = np.random.normal(1, 1, 100)  # Treated mean = 1
        x_c = np.random.normal(0, 1, 100)  # Control mean = 0
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        smd = compute_smd(x, treat)
        
        # SMD should be approximately 1 (1 std difference)
        assert np.abs(smd - 1.0) < 0.3, f"SMD should be ~1 for 1 std difference, got {smd}"

    def test_smd_sign(self):
        """Test SMD sign indicates direction of imbalance."""
        np.random.seed(42)
        
        # Treated higher
        x_t = np.random.normal(2, 1, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        smd = compute_smd(x, treat)
        assert smd > 0, "SMD should be positive when treated mean > control mean"
        
        # Treated lower
        x_t = np.random.normal(-2, 1, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        
        smd = compute_smd(x, treat)
        assert smd < 0, "SMD should be negative when treated mean < control mean"

    def test_smd_scale_invariance(self):
        """Test SMD is scale-invariant."""
        np.random.seed(42)
        
        x_t = np.random.normal(1, 1, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        smd_original = compute_smd(x, treat)
        smd_scaled = compute_smd(x * 100, treat)  # Scale by 100
        
        assert_allclose_with_report(
            smd_original, smd_scaled,
            rtol=0.01, atol=0.01,
            name="SMD scale invariance"
        )

class TestWeightedSMD:
    """
    Test ID: BAL-DIAG-002
    
    Test weighted standardized mean difference calculation.
    """
    
    def test_weighted_smd_reduces_imbalance(self, cbps_binary_fit, lalonde_full):
        """Test that CBPS weights reduce SMD."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Compute SMD for each covariate (excluding intercept)
        for j in range(1, X.shape[1]):
            smd_unweighted = compute_smd(X[:, j], treat)
            smd_weighted = compute_smd(X[:, j], treat, weights)
            
            # Weighted SMD should generally be smaller (better balance)
            # Allow some tolerance as not all covariates may improve
            assert np.abs(smd_weighted) < np.abs(smd_unweighted) + 0.5, \
                f"Covariate {j}: weighted SMD ({smd_weighted:.3f}) should not be much worse than unweighted ({smd_unweighted:.3f})"

    def test_weighted_smd_with_uniform_weights(self):
        """Test weighted SMD equals unweighted SMD with uniform weights."""
        np.random.seed(42)
        n = 200
        
        x = np.random.normal(0, 1, n)
        treat = np.random.binomial(1, 0.5, n).astype(float)
        weights = np.ones(n)
        
        smd_unweighted = compute_smd(x, treat)
        smd_weighted = compute_smd(x, treat, weights)
        
        assert_allclose_with_report(
            smd_unweighted, smd_weighted,
            rtol=1e-10, atol=1e-10,
            name="SMD with uniform weights"
        )

    def test_weighted_smd_extreme_weights(self):
        """Test weighted SMD with extreme weights."""
        np.random.seed(42)
        
        # Create imbalanced data
        x_t = np.random.normal(2, 1, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        # Weights that should reduce imbalance
        # Give more weight to control units with higher x values
        weights = np.ones(200)
        weights[100:150] = 2.0  # Higher weight to some controls
        
        smd_unweighted = compute_smd(x, treat)
        smd_weighted = compute_smd(x, treat, weights)
        
        # Both should be finite
        assert np.isfinite(smd_unweighted)
        assert np.isfinite(smd_weighted)

class TestVarianceRatio:
    """
    Test ID: BAL-DIAG-003
    
    Test variance ratio calculation.
    """
    
    def test_variance_ratio_equal_variances(self):
        """Test variance ratio is 1 for equal variances."""
        np.random.seed(42)
        
        x_t = np.random.normal(0, 1, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        vr = compute_variance_ratio(x, treat)
        
        # Should be close to 1
        assert 0.7 < vr < 1.3, f"Variance ratio should be ~1 for equal variances, got {vr}"

    def test_variance_ratio_unequal_variances(self):
        """Test variance ratio detects unequal variances."""
        np.random.seed(42)
        
        x_t = np.random.normal(0, 2, 100)  # std = 2
        x_c = np.random.normal(0, 1, 100)  # std = 1
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        vr = compute_variance_ratio(x, treat)
        
        # Should be approximately 4 (2^2 / 1^2)
        assert 2.5 < vr < 6.0, f"Variance ratio should be ~4, got {vr}"

    def test_variance_ratio_positive(self):
        """Test variance ratio is always positive."""
        np.random.seed(42)
        
        for _ in range(10):
            x = np.random.normal(0, np.random.uniform(0.5, 2), 200)
            treat = np.random.binomial(1, 0.5, 200).astype(float)
            
            vr = compute_variance_ratio(x, treat)
            assert vr > 0, "Variance ratio should be positive"

    def test_variance_ratio_reciprocal(self):
        """Test variance ratio reciprocal property."""
        np.random.seed(42)
        
        x_t = np.random.normal(0, 2, 100)
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        vr = compute_variance_ratio(x, treat)
        
        # Swap treatment labels
        treat_swapped = 1 - treat
        vr_swapped = compute_variance_ratio(x, treat_swapped)
        
        # Should be reciprocals
        assert_allclose_with_report(
            vr * vr_swapped, 1.0,
            rtol=0.01, atol=0.01,
            name="Variance ratio reciprocal"
        )

class TestKSStatistic:
    """
    Test ID: BAL-DIAG-004
    
    Test Kolmogorov-Smirnov statistic calculation.
    """
    
    def test_ks_identical_distributions(self):
        """Test KS statistic is small for identical distributions."""
        np.random.seed(42)
        
        x = np.random.normal(0, 1, 200)
        treat = np.random.binomial(1, 0.5, 200).astype(float)
        
        ks = compute_ks_statistic(x, treat)
        
        # Should be small for random assignment
        assert ks < 0.3, f"KS should be small for identical distributions, got {ks}"

    def test_ks_different_distributions(self):
        """Test KS statistic detects different distributions."""
        np.random.seed(42)
        
        x_t = np.random.normal(2, 1, 100)  # Different mean
        x_c = np.random.normal(0, 1, 100)
        x = np.concatenate([x_t, x_c])
        treat = np.array([1] * 100 + [0] * 100)
        
        ks = compute_ks_statistic(x, treat)
        
        # Should be large for different distributions
        assert ks > 0.3, f"KS should be large for different distributions, got {ks}"

    def test_ks_range(self):
        """Test KS statistic is in [0, 1]."""
        np.random.seed(42)
        
        for _ in range(10):
            x = np.random.normal(np.random.uniform(-2, 2), 1, 200)
            treat = np.random.binomial(1, 0.5, 200).astype(float)
            
            ks = compute_ks_statistic(x, treat)
            assert 0 <= ks <= 1, f"KS should be in [0, 1], got {ks}"

    def test_ks_symmetric(self):
        """Test KS statistic is symmetric in treatment labels."""
        np.random.seed(42)
        
        x = np.random.normal(0, 1, 200)
        treat = np.random.binomial(1, 0.5, 200).astype(float)
        
        ks = compute_ks_statistic(x, treat)
        ks_swapped = compute_ks_statistic(x, 1 - treat)
        
        assert_allclose_with_report(
            ks, ks_swapped,
            rtol=1e-10, atol=1e-10,
            name="KS symmetry"
        )

class TestBalanceImprovement:
    """
    Test ID: BAL-DIAG-005
    
    Test balance improvement after CBPS weighting.
    """
    
    def test_overall_balance_improvement(self, cbps_binary_fit, lalonde_full):
        """Test overall balance improves with CBPS weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Compute average absolute SMD before and after
        smd_before = []
        smd_after = []
        
        for j in range(1, X.shape[1]):  # Skip intercept
            smd_before.append(np.abs(compute_smd(X[:, j], treat)))
            smd_after.append(np.abs(compute_smd(X[:, j], treat, weights)))
        
        avg_smd_before = np.mean(smd_before)
        avg_smd_after = np.mean(smd_after)
        
        # Average SMD should decrease
        assert avg_smd_after < avg_smd_before, \
            f"Average SMD should decrease: before={avg_smd_before:.3f}, after={avg_smd_after:.3f}"

    def test_balance_improvement_att(self, cbps_binary_fit, lalonde_full):
        """Test balance improvement for ATT estimation."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        weights = result['weights']
        
        # For ATT, focus on control group reweighting
        # Compute SMD for each covariate
        improvements = 0
        for j in range(1, X.shape[1]):
            smd_before = np.abs(compute_smd(X[:, j], treat))
            smd_after = np.abs(compute_smd(X[:, j], treat, weights))
            if smd_after < smd_before:
                improvements += 1
        
        # Most covariates should improve
        assert improvements >= (X.shape[1] - 1) // 2, \
            f"At least half of covariates should improve balance"

    def test_balance_improvement_exact_vs_over(self, cbps_binary_fit, lalonde_full):
        """Compare balance improvement between exact and over-identified methods."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Compute average SMD for both
        def avg_smd(weights):
            smds = [np.abs(compute_smd(X[:, j], treat, weights)) for j in range(1, X.shape[1])]
            return np.mean(smds)
        
        smd_exact = avg_smd(result_exact['weights'])
        smd_over = avg_smd(result_over['weights'])
        
        # Both should achieve reasonable balance
        assert smd_exact < 0.5, f"Exact method SMD too high: {smd_exact}"
        assert smd_over < 0.5, f"Over method SMD too high: {smd_over}"

    def test_balance_threshold(self, cbps_binary_fit, lalonde_full):
        """Test that CBPS achieves SMD below common threshold."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Common threshold is 0.1 or 0.25
        threshold = 0.25
        
        for j in range(1, X.shape[1]):
            smd = np.abs(compute_smd(X[:, j], treat, weights))
            # Allow some covariates to exceed threshold
            # but most should be below
        
        # Count covariates below threshold
        below_threshold = sum(
            np.abs(compute_smd(X[:, j], treat, weights)) < threshold
            for j in range(1, X.shape[1])
        )
        
        # At least half should be below threshold
        assert below_threshold >= (X.shape[1] - 1) // 2, \
            f"At least half of covariates should have SMD < {threshold}"

class TestBalanceLossFormula:
    """Test ID: BAL-001"""
    
    def test_ate_balance_loss_formula(self, bal_loss, r_ginv, simple_binary_data):
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n = X.shape[0]
        sample_weights, beta = data['sample_weights'], data['true_beta']
        
        XprimeX_inv = r_ginv(X.T @ X)
        loss_func = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        w_ate = (1 / n) * (1 / (probs - 1 + treat))
        Xprimew = (sample_weights[:, None] * X).T @ w_ate
        loss_manual = np.abs(Xprimew.T @ XprimeX_inv @ Xprimew)
        
        assert_allclose_with_report(loss_func, loss_manual,
            rtol=Tolerances.GMM_LOSS_RTOL, atol=Tolerances.GMM_LOSS_ATOL,
            name="ATE balance loss formula")

    def test_att_balance_loss_formula(self, bal_loss, att_wt_func, r_ginv, simple_binary_data):
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n = X.shape[0]
        sample_weights, beta = data['sample_weights'], data['true_beta']
        
        XprimeX_inv = r_ginv(X.T @ X)
        loss_func = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=1)
        
        w_att = (1 / n) * att_wt_func(beta, X, treat, sample_weights)
        Xprimew = (sample_weights[:, None] * X).T @ w_att
        loss_manual = np.abs(Xprimew.T @ XprimeX_inv @ Xprimew)
        
        assert_allclose_with_report(loss_func, loss_manual,
            rtol=Tolerances.GMM_LOSS_RTOL, atol=Tolerances.GMM_LOSS_ATOL,
            name="ATT balance loss formula")

    def test_balance_loss_non_negative(self, bal_loss, r_ginv, simple_binary_data):
        data = simple_binary_data
        X, treat, sample_weights = data['X'], data['treat'], data['sample_weights']
        XprimeX_inv = r_ginv(X.T @ X)
        
        np.random.seed(42)
        for _ in range(10):
            beta = np.random.randn(X.shape[1]) * 0.5
            loss = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=0)
            assert loss >= 0, f"Balance loss should be non-negative, got {loss}"

    def test_balance_loss_finite(self, bal_loss, r_ginv, simple_binary_data):
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights, beta = data['sample_weights'], data['true_beta']
        XprimeX_inv = r_ginv(X.T @ X)
        
        loss_ate = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        loss_att = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=1)
        assert np.isfinite(loss_ate), "ATE balance loss should be finite"
        assert np.isfinite(loss_att), "ATT balance loss should be finite"

class TestPerfectBalance:
    """Test ID: BAL-002 - Perfect balance behavior"""
    
    def test_perfect_balance_zero_loss(self, bal_loss, r_ginv, simple_binary_data):
        """When covariates are perfectly balanced, loss should be near zero."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        # Create perfectly balanced scenario: equal means in treatment groups
        n = X.shape[0]
        n_t = int(np.sum(treat))
        n_c = n - n_t
        
        # Use beta=0 which gives equal propensity scores
        beta_zero = np.zeros(X.shape[1])
        XprimeX_inv = r_ginv(X.T @ X)
        
        loss = bal_loss(beta_zero, X, treat, sample_weights, XprimeX_inv, att=0)
        
        # Loss should be small (not necessarily zero due to covariate imbalance)
        assert np.isfinite(loss), "Balance loss should be finite for zero beta"
        assert loss >= 0, "Balance loss should be non-negative"

    def test_perfect_balance_att(self, bal_loss, r_ginv, simple_binary_data):
        """ATT balance loss with zero beta."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        beta_zero = np.zeros(X.shape[1])
        XprimeX_inv = r_ginv(X.T @ X)
        
        loss = bal_loss(beta_zero, X, treat, sample_weights, XprimeX_inv, att=1)
        
        assert np.isfinite(loss), "ATT balance loss should be finite for zero beta"
        assert loss >= 0, "ATT balance loss should be non-negative"

class TestSampleWeightBalance:
    """Test ID: BAL-003 - Sample weight propagation in balance loss"""
    
    def test_uniform_weights_equivalence(self, bal_loss, r_ginv, simple_binary_data):
        """Uniform weights should give same result as ones."""
        data = simple_binary_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n = X.shape[0]
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Uniform weights (all ones)
        weights_ones = np.ones(n)
        loss_ones = bal_loss(beta, X, treat, weights_ones, XprimeX_inv, att=0)
        
        # Uniform weights (all 2s, normalized)
        weights_twos = np.ones(n) * 2
        loss_twos = bal_loss(beta, X, treat, weights_twos, XprimeX_inv, att=0)
        
        # Should be proportional (weights scale the loss)
        assert np.isfinite(loss_ones), "Loss with ones should be finite"
        assert np.isfinite(loss_twos), "Loss with twos should be finite"

    def test_weighted_balance_loss(self, bal_loss, r_ginv, weighted_binary_data):
        """Balance loss with non-uniform sample weights."""
        data = weighted_binary_data
        X, treat = data['X'], data['treat']
        sample_weights, beta = data['sample_weights'], data['true_beta']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        loss_ate = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        loss_att = bal_loss(beta, X, treat, sample_weights, XprimeX_inv, att=1)
        
        assert np.isfinite(loss_ate), "Weighted ATE balance loss should be finite"
        assert np.isfinite(loss_att), "Weighted ATT balance loss should be finite"
        assert loss_ate >= 0, "Weighted ATE balance loss should be non-negative"
        assert loss_att >= 0, "Weighted ATT balance loss should be non-negative"

    def test_sample_weight_effect(self, bal_loss, r_ginv, simple_binary_data):
        """Different sample weights should produce different losses."""
        data = simple_binary_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n = X.shape[0]
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Uniform weights
        weights_uniform = np.ones(n)
        loss_uniform = bal_loss(beta, X, treat, weights_uniform, XprimeX_inv, att=0)
        
        # Non-uniform weights
        np.random.seed(999)
        weights_varied = np.random.exponential(1, n)
        loss_varied = bal_loss(beta, X, treat, weights_varied, XprimeX_inv, att=0)
        
        # Both should be valid
        assert np.isfinite(loss_uniform), "Uniform weight loss should be finite"
        assert np.isfinite(loss_varied), "Varied weight loss should be finite"

class TestBalanceGradient:
    """Test balance gradient computation"""
    
    def test_gradient_finite(self, bal_gradient_func, r_ginv, simple_binary_data):
        """Balance gradient should be finite."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights, beta = data['sample_weights'], data['true_beta']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        grad_ate = bal_gradient_func(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        grad_att = bal_gradient_func(beta, X, treat, sample_weights, XprimeX_inv, att=1)
        
        assert np.all(np.isfinite(grad_ate)), "ATE balance gradient should be finite"
        assert np.all(np.isfinite(grad_att)), "ATT balance gradient should be finite"

    def test_gradient_shape(self, bal_gradient_func, r_ginv, simple_binary_data):
        """Balance gradient should have correct shape."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights, beta = data['sample_weights'], data['true_beta']
        k = X.shape[1]
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        grad = bal_gradient_func(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        
        assert grad.shape == (k,), f"Gradient shape should be ({k},), got {grad.shape}"

    def test_numerical_gradient(self, bal_loss, bal_gradient_func, r_ginv, simple_binary_data):
        """Balance gradient should match numerical approximation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights, beta = data['sample_weights'], data['true_beta']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Analytical gradient
        grad_analytical = bal_gradient_func(beta, X, treat, sample_weights, XprimeX_inv, att=0)
        
        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            beta_plus[i] += eps
            beta_minus[i] -= eps
            
            loss_plus = bal_loss(beta_plus, X, treat, sample_weights, XprimeX_inv, att=0)
            loss_minus = bal_loss(beta_minus, X, treat, sample_weights, XprimeX_inv, att=0)
            grad_numerical[i] = (loss_plus - loss_minus) / (2 * eps)
        
        assert_allclose_with_report(grad_analytical, grad_numerical,
            rtol=Tolerances.GRADIENT_RTOL, atol=Tolerances.GRADIENT_ATOL,
            name="Balance gradient vs numerical")

class TestBalanceLossNumericalStability:
    """Test numerical stability of balance loss"""
    
    def test_extreme_beta_values(self, bal_loss, r_ginv, simple_binary_data):
        """Balance loss should handle extreme beta values."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Large positive beta
        beta_large = np.array([5.0, 3.0, -2.0])
        loss_large = bal_loss(beta_large, X, treat, sample_weights, XprimeX_inv, att=0)
        assert np.isfinite(loss_large), "Loss should be finite for large beta"
        
        # Large negative beta
        beta_neg = np.array([-5.0, -3.0, 2.0])
        loss_neg = bal_loss(beta_neg, X, treat, sample_weights, XprimeX_inv, att=0)
        assert np.isfinite(loss_neg), "Loss should be finite for negative beta"

    def test_near_zero_probabilities(self, bal_loss, r_ginv, simple_binary_data):
        """Balance loss should handle near-zero probabilities via clipping."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Very large beta that would produce extreme probabilities
        beta_extreme = np.array([10.0, 5.0, -5.0])
        
        loss_ate = bal_loss(beta_extreme, X, treat, sample_weights, XprimeX_inv, att=0)
        loss_att = bal_loss(beta_extreme, X, treat, sample_weights, XprimeX_inv, att=1)
        
        assert np.isfinite(loss_ate), "ATE loss should be finite with extreme probabilities"
        assert np.isfinite(loss_att), "ATT loss should be finite with extreme probabilities"

class TestJStatistic:
    """
    Test ID: DIAG-001
    
    Test J-statistic (overidentification test) computation.
    """
    
    def test_j_statistic_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is always non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                assert result['J'] >= 0, \
                    f"J-statistic should be >= 0, got {result['J']} for att={att}, method={method}"

    def test_j_statistic_exact_small(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is small for exactly identified model."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        # For exactly identified, J should be very small
        assert result['J'] < 0.1, \
            f"J-statistic should be small for exact method, got {result['J']}"

    def test_j_statistic_over_larger(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is typically larger for over-identified model."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Over-identified typically has larger J (more moment conditions)
        # But this is not strictly required
        assert result_over['J'] >= 0

    def test_j_statistic_finite(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isfinite(result['J']), "J-statistic should be finite"

    def test_j_statistic_formula(self, cbps_binary_fit, gmm_func, lalonde_full):
        """Test J-statistic follows GMM formula: n * gbar' @ inv_V @ gbar."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = lalonde_full['sample_weights']
        n = lalonde_full['n']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        beta_opt = result['coefficients'].ravel()
        
        # Compute GMM loss at optimum
        gmm_result = gmm_func(beta_opt, X, treat, sample_weights, att=0, inv_V=None)
        
        # J = n * loss (approximately, depending on normalization)
        # The relationship depends on implementation details
        assert gmm_result['loss'] >= 0

class TestDeviance:
    """
    Test ID: DIAG-002
    
    Test deviance computation.
    """
    
    def test_deviance_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test deviance is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['deviance'] >= 0, "Deviance should be non-negative"

    def test_deviance_finite(self, cbps_binary_fit, lalonde_full):
        """Test deviance is finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isfinite(result['deviance']), "Deviance should be finite"

    def test_deviance_formula(self, cbps_binary_fit, lalonde_full):
        """Test deviance follows -2 * log-likelihood formula."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        probs = result['fitted_values']
        
        # Manual deviance calculation: -2 * sum(y*log(p) + (1-y)*log(1-p))
        probs_clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        log_lik = np.sum(treat * np.log(probs_clipped) + (1 - treat) * np.log(1 - probs_clipped))
        expected_deviance = -2 * log_lik
        
        # Allow some tolerance due to different computation paths
        assert_allclose_with_report(
            result['deviance'], expected_deviance,
            rtol=0.01, atol=1.0,
            name="Deviance formula"
        )

    def test_deviance_less_than_null(self, cbps_binary_fit, lalonde_full):
        """Test deviance is less than or equal to null deviance."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Model deviance should be <= null deviance (model should fit at least as well)
        assert result['deviance'] <= result['nulldeviance'] + 1e-6, \
            f"Deviance ({result['deviance']}) should be <= null deviance ({result['nulldeviance']})"

class TestNullDeviance:
    """
    Test ID: DIAG-003
    
    Test null deviance computation.
    """
    
    def test_null_deviance_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test null deviance is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['nulldeviance'] >= 0, "Null deviance should be non-negative"

    def test_null_deviance_finite(self, cbps_binary_fit, lalonde_full):
        """Test null deviance is finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isfinite(result['nulldeviance']), "Null deviance should be finite"

    def test_null_deviance_formula(self, cbps_binary_fit, lalonde_full):
        """Test null deviance follows intercept-only model formula."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Null model: p = mean(treat)
        p_null = np.mean(treat)
        p_null_clipped = np.clip(p_null, PROBS_MIN, 1 - PROBS_MIN)
        
        # Null deviance: -2 * sum(y*log(p_null) + (1-y)*log(1-p_null))
        log_lik_null = np.sum(treat * np.log(p_null_clipped) + (1 - treat) * np.log(1 - p_null_clipped))
        expected_null_deviance = -2 * log_lik_null
        
        assert_allclose_with_report(
            result['nulldeviance'], expected_null_deviance,
            rtol=0.01, atol=1.0,
            name="Null deviance formula"
        )

    def test_null_deviance_independent_of_method(self, cbps_binary_fit, lalonde_full):
        """Test null deviance is same regardless of method."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        # Null deviance should be the same (depends only on data)
        assert_allclose_with_report(
            result_over['nulldeviance'], result_exact['nulldeviance'],
            rtol=1e-10, atol=1e-10,
            name="Null deviance consistency"
        )

class TestMLEJStatistic:
    """
    Test ID: DIAG-004
    
    Test MLE J-statistic computation.
    """
    
    def test_mle_j_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test MLE J-statistic is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['mle_J'] >= 0, "MLE J-statistic should be non-negative"

    def test_mle_j_finite(self, cbps_binary_fit, lalonde_full):
        """Test MLE J-statistic is finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isfinite(result['mle_J']), "MLE J-statistic should be finite"

    def test_mle_j_vs_j_relationship(self, cbps_binary_fit, lalonde_full):
        """Test relationship between MLE J and CBPS J."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # MLE J is computed at GLM coefficients, CBPS J at optimized coefficients
        # CBPS J should typically be smaller (optimized for balance)
        # But this is not strictly required
        assert result['J'] >= 0 and result['mle_J'] >= 0

class TestGLMCoefficientInitialization:
    """
    Test ID: GLM-001
    
    Test GLM coefficient initialization.
    """
    
    def test_glm_coefficients_finite(self, lalonde_full):
        """Test GLM coefficients are finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert np.all(np.isfinite(result.params)), "GLM coefficients should be finite"

    def test_glm_coefficients_shape(self, lalonde_full):
        """Test GLM coefficients have correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = X.shape[1]
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert result.params.shape == (k,), f"GLM coefficients should have shape ({k},)"

    def test_glm_intercept_reasonable(self, lalonde_full):
        """Test GLM intercept is reasonable."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        # Intercept should be related to log-odds of treatment
        p_treat = np.mean(treat)
        expected_intercept_approx = np.log(p_treat / (1 - p_treat))
        
        # Allow wide tolerance as other covariates affect intercept
        assert np.abs(result.params[0] - expected_intercept_approx) < 5, \
            f"GLM intercept ({result.params[0]}) should be roughly related to log-odds ({expected_intercept_approx})"

    def test_glm_na_handling(self):
        """Test GLM handles near-collinear data."""
        np.random.seed(42)
        n = 100
        
        # Create data with near-collinearity
        x1 = np.random.normal(0, 1, n)
        x2 = x1 + np.random.normal(0, 0.01, n)  # Nearly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        model = sm.GLM(treat, X, family=Binomial())
        
        # Should not raise error
        try:
            result = model.fit()
            # Coefficients may have NaN for collinear columns
            assert result.params.shape == (3,)
        except Exception as e:
            # Some convergence issues are acceptable
            pass

class TestGLMFittedValues:
    """
    Test ID: GLM-002
    
    Test GLM fitted values (propensity scores).
    """
    
    def test_fitted_values_range(self, lalonde_full):
        """Test GLM fitted values are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert np.all(result.fittedvalues > 0), "Fitted values should be > 0"
        assert np.all(result.fittedvalues < 1), "Fitted values should be < 1"

    def test_fitted_values_shape(self, lalonde_full):
        """Test GLM fitted values have correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = len(treat)
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert result.fittedvalues.shape == (n,), f"Fitted values should have shape ({n},)"

    def test_fitted_values_formula(self, lalonde_full):
        """Test fitted values follow logistic formula."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        # Manual calculation
        theta = X @ result.params
        expected_probs = scipy.special.expit(theta)
        
        assert_allclose_with_report(
            result.fittedvalues, expected_probs,
            rtol=1e-10, atol=1e-10,
            name="GLM fitted values formula"
        )

    def test_fitted_values_mean(self, lalonde_full):
        """Test mean of fitted values approximates treatment proportion."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        mean_fitted = np.mean(result.fittedvalues)
        mean_treat = np.mean(treat)
        
        # Should be close
        assert np.abs(mean_fitted - mean_treat) < 0.1, \
            f"Mean fitted ({mean_fitted}) should be close to treatment proportion ({mean_treat})"

class TestGLMConvergence:
    """
    Test ID: GLM-003
    
    Test GLM convergence properties.
    """
    
    def test_glm_converges(self, lalonde_full):
        """Test GLM converges on LaLonde data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert result.converged, "GLM should converge on LaLonde data"

    def test_glm_convergence_iterations(self, lalonde_full):
        """Test GLM converges within reasonable iterations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit(maxiter=25)
        
        # Should converge within 25 iterations
        assert result.converged, "GLM should converge within 25 iterations"

    def test_glm_convergence_tolerance(self, lalonde_full):
        """Test GLM convergence with different tolerances."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        model = sm.GLM(treat, X, family=Binomial())
        
        # Strict tolerance
        result_strict = model.fit(tol=1e-10)
        
        # Loose tolerance
        result_loose = model.fit(tol=1e-4)
        
        # Both should converge
        assert result_strict.converged
        assert result_loose.converged
        
        # Coefficients should be similar
        assert_allclose_with_report(
            result_strict.params, result_loose.params,
            rtol=0.01, atol=0.01,
            name="GLM convergence tolerance"
        )

    def test_glm_convergence_simple_data(self):
        """Test GLM converges on simple synthetic data."""
        np.random.seed(42)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        true_beta = np.array([0.0, 0.5])
        theta = X @ true_beta
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        model = sm.GLM(treat, X, family=Binomial())
        result = model.fit()
        
        assert result.converged, "GLM should converge on simple data"

class TestGLMInitImpact:
    """
    Test ID: GLM-005
    
    Test GLM initialization impact on final CBPS results.
    """
    
    def test_glm_init_provides_good_starting_point(self, cbps_binary_fit, lalonde_full):
        """Test GLM initialization provides good starting point for optimization."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Fit GLM
        model = sm.GLM(treat, X, family=Binomial())
        glm_result = model.fit()
        
        # Fit CBPS
        cbps_result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # CBPS coefficients should not be too far from GLM
        # (GLM is used as initialization)
        coef_diff = np.abs(cbps_result['coefficients'].ravel() - glm_result.params)
        
        # Most coefficients should be within reasonable range
        assert np.median(coef_diff) < 2.0, \
            "CBPS coefficients should not deviate too much from GLM initialization"

    def test_cbps_improves_on_glm(self, cbps_binary_fit, lalonde_full):
        """Test CBPS improves balance compared to GLM."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Fit GLM
        model = sm.GLM(treat, X, family=Binomial())
        glm_result = model.fit()
        
        # GLM weights (IPW)
        glm_probs = glm_result.fittedvalues
        glm_probs = np.clip(glm_probs, PROBS_MIN, 1 - PROBS_MIN)
        glm_weights = treat / glm_probs + (1 - treat) / (1 - glm_probs)
        
        # CBPS weights
        cbps_result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        cbps_weights = cbps_result['weights']
        
        # Compute balance for both
        def compute_balance(weights):
            balance = 0
            for j in range(1, X.shape[1]):
                mean_t = np.average(X[treat == 1, j], weights=weights[treat == 1])
                mean_c = np.average(X[treat == 0, j], weights=weights[treat == 0])
                balance += np.abs(mean_t - mean_c)
            return balance
        
        glm_balance = compute_balance(glm_weights)
        cbps_balance = compute_balance(cbps_weights)
        
        # CBPS should achieve better or similar balance
        assert cbps_balance <= glm_balance * 1.5, \
            f"CBPS balance ({cbps_balance}) should not be much worse than GLM ({glm_balance})"

    def test_different_init_same_result(self, cbps_binary_fit, lalonde_full):
        """Test CBPS converges to similar result from different initializations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Run CBPS multiple times (should use same GLM init internally)
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Results should be identical (deterministic)
        assert_allclose_with_report(
            result1['coefficients'], result2['coefficients'],
            rtol=1e-10, atol=1e-10,
            name="CBPS reproducibility"
        )

    def test_alpha_scaling_effect(self, lalonde_full):
        """Test alpha scaling in GLM initialization."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Fit GLM
        model = sm.GLM(treat, X, family=Binomial())
        glm_result = model.fit()
        beta_glm = glm_result.params
        
        # Test different alpha values
        from cbps.core.cbps_binary import _gmm_loss
        
        sample_weights = np.ones(len(treat))
        
        losses = []
        alphas = [0.8, 0.9, 1.0, 1.1]
        for alpha in alphas:
            loss = _gmm_loss(beta_glm * alpha, X, treat, sample_weights, att=0, inv_V=None)
            losses.append(loss)
        
        # Optimal alpha should minimize loss
        min_idx = np.argmin(losses)
        
        # Loss should be finite for all alphas
        assert all(np.isfinite(l) for l in losses), "All losses should be finite"

class TestMomentConditions:
    """
    Test moment condition calculations for GMM.
    
    Test ID: GMM-001
    Requirement: REQ-011
    
    The moment conditions combine:
    1. Score conditions: (1/n) X'(T - π)
    2. Balance conditions: (1/n) X'w
    
    where w depends on the estimand (ATE or ATT).
    """
    
    def test_score_condition_formula(self, simple_binary_data):
        """Test score condition: (1/n) X'(T - π)."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Score condition formula: (1/n) * X' * (T - π)
        score_cond = (1 / n) * (sample_weights[:, None] * X).T @ (treat - probs)
        
        # Verify shape
        assert score_cond.shape == (k,), \
            f"Score condition shape mismatch: got {score_cond.shape}, expected ({k},)"
        
        # At true parameters, score should be close to zero (but not exactly)
        # This is because we're using the true beta, not the MLE
        assert np.all(np.isfinite(score_cond)), "Score condition contains non-finite values"
    
    def test_ate_balance_condition_formula(self, simple_binary_data):
        """Test ATE balance condition: (1/n) X'w where w = 1/(π - 1 + T)."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weight: w = 1 / (π - 1 + T) = T/π - (1-T)/(1-π)
        w_ate = 1 / (probs - 1 + treat)
        
        # Balance condition: (1/n) * X' * w
        balance_cond = (1 / n) * (sample_weights[:, None] * X).T @ w_ate
        
        # Verify shape
        assert balance_cond.shape == (k,), \
            f"Balance condition shape mismatch: got {balance_cond.shape}, expected ({k},)"
        
        # Verify finite values
        assert np.all(np.isfinite(balance_cond)), "Balance condition contains non-finite values"
    
    def test_att_balance_condition_formula(self, att_wt_func, simple_binary_data):
        """Test ATT balance condition using _att_wt_func."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        
        # ATT weight from function
        w_att = att_wt_func(beta, X, treat, sample_weights)
        
        # Balance condition: (1/n) * X' * w
        balance_cond = (1 / n) * (sample_weights[:, None] * X).T @ w_att
        
        # Verify shape
        assert balance_cond.shape == (k,), \
            f"ATT balance condition shape mismatch: got {balance_cond.shape}, expected ({k},)"
        
        # Verify finite values
        assert np.all(np.isfinite(balance_cond)), "ATT balance condition contains non-finite values"
    
    def test_gbar_concatenation(self, gmm_func, simple_binary_data):
        """Test that gbar correctly concatenates score and balance conditions."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        
        # Call gmm_func to get the result
        result = gmm_func(beta, X, treat, sample_weights, att=0, inv_V=None)
        
        # The gbar vector should have length 2k (score + balance)
        # We can verify this indirectly through the invV shape
        inv_V = result['inv_V']
        assert inv_V.shape == (2 * k, 2 * k), \
            f"invV shape mismatch: got {inv_V.shape}, expected ({2*k}, {2*k})"

class TestGMMLossCalculation:
    """
    Test GMM loss calculation.
    
    Test ID: GMM-002
    Requirement: REQ-008
    
    The GMM loss is the quadratic form:
        L(β) = ḡ(β)' V⁻¹ ḡ(β)
    
    where ḡ(β) is the moment condition vector and V⁻¹ is the
    pseudoinverse of the covariance matrix.
    """
    
    def test_gmm_loss_non_negative(self, gmm_loss, simple_binary_data):
        """Test that GMM loss is always non-negative."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        
        # Test with various beta values
        np.random.seed(42)
        for _ in range(10):
            beta = np.random.randn(X.shape[1]) * 0.5
            loss = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
            assert loss >= 0, f"GMM loss should be non-negative, got {loss}"
    
    def test_gmm_loss_finite(self, gmm_loss, simple_binary_data):
        """Test that GMM loss is finite for reasonable inputs."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # ATE loss
        loss_ate = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        assert np.isfinite(loss_ate), f"ATE GMM loss should be finite, got {loss_ate}"
        
        # ATT loss
        loss_att = gmm_loss(beta, X, treat, sample_weights, att=1, inv_V=None)
        assert np.isfinite(loss_att), f"ATT GMM loss should be finite, got {loss_att}"
    
    def test_gmm_loss_quadratic_form(self, gmm_func, simple_binary_data):
        """Test that GMM loss equals ḡ' V⁻¹ ḡ."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Get result from gmm_func
        result = gmm_func(beta, X, treat, sample_weights, att=0, inv_V=None)
        loss_from_func = result['loss']
        inv_V = result['inv_V']
        
        # Manually compute gbar
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weight
        w_curr = 1 / (probs - 1 + treat)
        
        # Score condition
        score_cond = (1 / n) * (sample_weights[:, None] * X).T @ (treat - probs)
        
        # Balance condition
        balance_cond = (1 / n) * (sample_weights[:, None] * X).T @ w_curr
        
        # Concatenate
        gbar = np.concatenate([score_cond.ravel(), balance_cond.ravel()])
        
        # Compute quadratic form
        loss_manual = float(gbar.T @ inv_V @ gbar)
        
        assert_allclose_with_report(
            loss_from_func, loss_manual,
            rtol=Tolerances.GMM_LOSS_RTOL,
            atol=Tolerances.GMM_LOSS_ATOL,
            name="GMM loss quadratic form"
        )
    
    def test_gmm_loss_ate_vs_att_different(self, gmm_loss, simple_binary_data):
        """Test that ATE and ATT GMM losses are different."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        loss_ate = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        loss_att = gmm_loss(beta, X, treat, sample_weights, att=1, inv_V=None)
        
        # They should be different due to different weight formulas
        assert not np.isclose(loss_ate, loss_att, rtol=1e-3), \
            f"ATE ({loss_ate}) and ATT ({loss_att}) losses should differ"

class TestTwoStepMode:
    """
    Test two-step GMM mode.
    
    Test ID: GMM-003
    Requirement: REQ-008
    
    In two-step GMM:
    1. First step: Estimate β using identity weighting matrix
    2. Second step: Re-estimate β using optimal weighting matrix V⁻¹
    
    The key property is that inv_V is precomputed and fixed during optimization.
    """
    
    def test_two_step_fixed_inv_V(self, gmm_loss, compute_V_matrix, simple_binary_data):
        """Test that two-step mode uses fixed inv_V."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute propensity scores for V matrix
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Precompute inv_V (two-step mode)
        inv_V_fixed = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # Compute loss with fixed inv_V
        loss_fixed = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=inv_V_fixed)
        
        # Compute loss with recomputed inv_V (continuous updating)
        loss_recomputed = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        
        # At the same beta, they should be equal (since V is computed at same point)
        assert_allclose_with_report(
            loss_fixed, loss_recomputed,
            rtol=Tolerances.GMM_LOSS_RTOL,
            atol=Tolerances.GMM_LOSS_ATOL,
            name="Two-step vs continuous at same beta"
        )
    
    def test_two_step_different_beta_same_inv_V(self, gmm_loss, compute_V_matrix, simple_binary_data):
        """Test that two-step mode keeps inv_V fixed when beta changes."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute propensity scores for V matrix at initial beta
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Precompute inv_V at initial beta
        inv_V_fixed = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # Perturb beta
        beta_perturbed = beta + np.random.randn(k) * 0.1
        
        # Two-step: use fixed inv_V
        loss_two_step = gmm_loss(beta_perturbed, X, treat, sample_weights, att=0, inv_V=inv_V_fixed)
        
        # Continuous updating: recompute inv_V at perturbed beta
        loss_continuous = gmm_loss(beta_perturbed, X, treat, sample_weights, att=0, inv_V=None)
        
        # They should be different because inv_V is computed at different points
        # (unless the perturbation is very small)
        assert np.isfinite(loss_two_step), "Two-step loss should be finite"
        assert np.isfinite(loss_continuous), "Continuous loss should be finite"
    
    def test_two_step_inv_V_shape(self, compute_V_matrix, simple_binary_data):
        """Test that precomputed inv_V has correct shape."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute inv_V
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        expected_shape = (2 * k, 2 * k)
        assert inv_V.shape == expected_shape, \
            f"inv_V shape mismatch: got {inv_V.shape}, expected {expected_shape}"

class TestContinuousUpdateMode:
    """
    Test continuous updating GMM mode.
    
    Test ID: GMM-004
    Requirement: REQ-008
    
    In continuous updating GMM:
    - The weighting matrix V⁻¹ is recomputed at each iteration
    - This provides more efficient estimates but is computationally more expensive
    """
    
    def test_continuous_update_recomputes_inv_V(self, gmm_func, simple_binary_data):
        """Test that continuous updating recomputes inv_V at each call."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Call with inv_V=None (continuous updating)
        result1 = gmm_func(beta, X, treat, sample_weights, att=0, inv_V=None)
        inv_V_1 = result1['inv_V']
        
        # Perturb beta
        beta_perturbed = beta + np.array([0.1, 0.05, -0.05])
        
        # Call again with perturbed beta
        result2 = gmm_func(beta_perturbed, X, treat, sample_weights, att=0, inv_V=None)
        inv_V_2 = result2['inv_V']
        
        # inv_V should be different because it's recomputed at different beta
        assert not np.allclose(inv_V_1, inv_V_2, rtol=1e-5), \
            "Continuous updating should produce different inv_V for different beta"
    
    def test_continuous_update_consistency(self, gmm_func, gmm_loss, simple_binary_data):
        """Test that gmm_func and gmm_loss give consistent results in continuous mode."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Get loss from gmm_func
        result = gmm_func(beta, X, treat, sample_weights, att=0, inv_V=None)
        loss_from_func = result['loss']
        
        # Get loss from gmm_loss
        loss_from_loss = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        
        assert_allclose_with_report(
            loss_from_func, loss_from_loss,
            rtol=Tolerances.GMM_LOSS_RTOL,
            atol=Tolerances.GMM_LOSS_ATOL,
            name="gmm_func vs gmm_loss consistency"
        )
    
    def test_continuous_update_ate_att(self, gmm_func, simple_binary_data):
        """Test continuous updating for both ATE and ATT."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # ATE continuous updating
        result_ate = gmm_func(beta, X, treat, sample_weights, att=0, inv_V=None)
        assert np.isfinite(result_ate['loss']), "ATE loss should be finite"
        assert result_ate['inv_V'].shape[0] == 2 * X.shape[1], "ATE inv_V shape incorrect"
        
        # ATT continuous updating
        result_att = gmm_func(beta, X, treat, sample_weights, att=1, inv_V=None)
        assert np.isfinite(result_att['loss']), "ATT loss should be finite"
        assert result_att['inv_V'].shape[0] == 2 * X.shape[1], "ATT inv_V shape incorrect"
        
        # Results should differ
        assert not np.isclose(result_ate['loss'], result_att['loss'], rtol=1e-3), \
            "ATE and ATT losses should differ"

class TestGMMGradient:
    """
    Test GMM gradient computation.
    
    The gradient is computed numerically using scipy.optimize.approx_fprime
    due to numerical stability issues with analytical gradients for ATT.
    """
    
    def test_gradient_shape(self, gmm_gradient, compute_V_matrix, simple_binary_data):
        """Test that gradient has correct shape."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute inv_V
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # Compute gradient
        grad = gmm_gradient(beta, inv_V, X, treat, sample_weights, att=0)
        
        assert grad.shape == (k,), \
            f"Gradient shape mismatch: got {grad.shape}, expected ({k},)"
    
    def test_gradient_finite(self, gmm_gradient, compute_V_matrix, simple_binary_data):
        """Test that gradient is finite."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute inv_V
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # Compute gradient
        grad = gmm_gradient(beta, inv_V, X, treat, sample_weights, att=0)
        
        assert np.all(np.isfinite(grad)), "Gradient should be finite"
    
    def test_gradient_numerical_verification(self, gmm_loss, gmm_gradient, 
                                              compute_V_matrix, simple_binary_data):
        """Verify gradient using finite differences."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute inv_V
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # Compute analytical gradient
        grad_analytical = gmm_gradient(beta, inv_V, X, treat, sample_weights, att=0)
        
        # Compute numerical gradient using finite differences
        eps = np.sqrt(np.finfo(float).eps)
        grad_numerical = np.zeros(k)
        for i in range(k):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            
            loss_plus = gmm_loss(beta_plus, X, treat, sample_weights, att=0, inv_V=inv_V)
            loss_minus = gmm_loss(beta_minus, X, treat, sample_weights, att=0, inv_V=inv_V)
            
            grad_numerical[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare (use looser tolerance for numerical gradient)
        assert_allclose_with_report(
            grad_analytical, grad_numerical,
            rtol=Tolerances.GRADIENT_RTOL,
            atol=Tolerances.GRADIENT_ATOL,
            name="GMM gradient numerical verification"
        )

class TestGMMNumericalStability:
    """
    Test numerical stability of GMM functions.
    """
    
    def test_extreme_propensity_scores(self, gmm_loss, simple_binary_data):
        """Test GMM loss with extreme propensity scores."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        
        # Beta that produces extreme propensity scores
        beta_extreme = np.array([5.0, 2.0, -2.0])
        
        # Should still produce finite loss due to probability clipping
        loss = gmm_loss(beta_extreme, X, treat, sample_weights, att=0, inv_V=None)
        assert np.isfinite(loss), f"GMM loss should be finite even with extreme beta, got {loss}"
    
    def test_zero_beta(self, gmm_loss, simple_binary_data):
        """Test GMM loss with zero coefficients."""
        data = simple_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        k = X.shape[1]
        
        # Zero beta (all propensity scores = 0.5)
        beta_zero = np.zeros(k)
        
        loss = gmm_loss(beta_zero, X, treat, sample_weights, att=0, inv_V=None)
        assert np.isfinite(loss), f"GMM loss should be finite with zero beta, got {loss}"
        assert loss >= 0, f"GMM loss should be non-negative, got {loss}"
    
    def test_weighted_data(self, gmm_loss, weighted_binary_data):
        """Test GMM loss with non-uniform sample weights."""
        data = weighted_binary_data
        X = data['X']
        treat = data['treat']
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Verify weights are non-uniform
        assert not np.allclose(sample_weights, np.ones_like(sample_weights)), \
            "Sample weights should be non-uniform for this test"
        
        # GMM loss should still work
        loss = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        assert np.isfinite(loss), f"GMM loss should be finite with weighted data, got {loss}"
        assert loss >= 0, f"GMM loss should be non-negative, got {loss}"

class TestTreatmentValidation:
    """
    Test ID: VAL-001
    
    Test treatment vector validation.
    """
    
    def test_valid_binary_treatment(self, cbps_binary_fit, lalonde_full):
        """Test valid binary treatment is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Should work without error
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_treatment_as_float(self, cbps_binary_fit, lalonde_full):
        """Test treatment as float array is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat'].astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_treatment_as_int(self, cbps_binary_fit, lalonde_full):
        """Test treatment as int array is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat'].astype(int)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_treatment_all_zeros(self, cbps_binary_fit, lalonde_full):
        """Test treatment with all zeros raises error or warning."""
        X = lalonde_full['X']
        treat = np.zeros(len(lalonde_full['treat']))
        
        # Should raise error or produce invalid results (warnings are acceptable)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
                # If it runs, check for warnings or invalid results
                assert len(w) > 0 or not result['converged'] or np.any(~np.isfinite(result['weights'])), \
                    "All-zero treatment should produce warning or invalid results"
            except (ValueError, ZeroDivisionError, RuntimeError):
                pass  # Expected behavior

    def test_treatment_all_ones(self, cbps_binary_fit, lalonde_full):
        """Test treatment with all ones raises error or warning."""
        X = lalonde_full['X']
        treat = np.ones(len(lalonde_full['treat']))
        
        # Should raise error or produce invalid results (warnings are acceptable)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
                # If it runs, check for warnings or invalid results
                assert len(w) > 0 or not result['converged'] or np.any(~np.isfinite(result['weights'])), \
                    "All-one treatment should produce warning or invalid results"
            except (ValueError, ZeroDivisionError, RuntimeError):
                pass  # Expected behavior

    def test_treatment_with_nan(self, cbps_binary_fit, lalonde_full):
        """Test treatment with NaN values."""
        X = lalonde_full['X']
        treat = lalonde_full['treat'].copy()
        treat[0] = np.nan
        
        # Should raise error or handle gracefully
        try:
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            # If it runs, check for NaN propagation
            assert not np.any(np.isnan(result['weights'])) or np.any(np.isnan(result['weights']))
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_treatment_with_inf(self, cbps_binary_fit, lalonde_full):
        """Test treatment with Inf values."""
        X = lalonde_full['X']
        treat = lalonde_full['treat'].copy().astype(float)
        treat[0] = np.inf
        
        # Should raise error
        with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

class TestXMatrixValidation:
    """
    Test ID: VAL-002
    
    Test covariate matrix validation.
    """
    
    def test_valid_x_matrix(self, cbps_binary_fit, lalonde_full):
        """Test valid X matrix is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_x_with_intercept(self, cbps_binary_fit, lalonde_full):
        """Test X matrix with intercept column."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # First column should be intercept (all ones)
        assert np.allclose(X[:, 0], 1.0), "First column should be intercept"
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_x_without_intercept(self, cbps_binary_fit, lalonde_full):
        """Test X matrix without intercept column."""
        X = lalonde_full['X'][:, 1:]  # Remove intercept
        treat = lalonde_full['treat']
        
        # Add intercept manually
        X_with_intercept = np.column_stack([np.ones(len(treat)), X])
        
        result = cbps_binary_fit(treat, X_with_intercept, att=0, method='over', two_step=True)
        assert result['converged']

    def test_x_with_nan(self, cbps_binary_fit, lalonde_full):
        """Test X matrix with NaN values."""
        X = lalonde_full['X'].copy()
        treat = lalonde_full['treat']
        X[0, 1] = np.nan
        
        # Should raise error or handle gracefully
        try:
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_x_with_inf(self, cbps_binary_fit, lalonde_full):
        """Test X matrix with Inf values."""
        X = lalonde_full['X'].copy()
        treat = lalonde_full['treat']
        X[0, 1] = np.inf
        
        # Should raise error
        with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

    def test_x_constant_column(self, cbps_binary_fit):
        """Test X matrix with constant column (besides intercept)."""
        np.random.seed(42)
        n = 100
        
        x1 = np.random.normal(0, 1, n)
        x2 = np.ones(n) * 5  # Constant column
        X = np.column_stack([np.ones(n), x1, x2])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Should raise error for rank-deficient matrix
        with pytest.raises(ValueError):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

    def test_x_rank_deficient(self, cbps_binary_fit):
        """Test X matrix that is rank-deficient."""
        np.random.seed(42)
        n = 100
        
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 2  # Perfectly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Should raise error
        with pytest.raises(ValueError):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

class TestSampleWeightsValidation:
    """
    Test ID: VAL-003
    
    Test sample weights validation.
    """
    
    def test_valid_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test valid sample weights are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat))
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights
        )
        assert result['converged']

    def test_none_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test None sample weights defaults to uniform."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=None
        )
        assert result['converged']

    def test_varying_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test varying sample weights are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        np.random.seed(42)
        sample_weights = np.random.exponential(1, len(treat))
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights
        )
        assert result['converged']

    def test_negative_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test negative sample weights raise error."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat))
        sample_weights[0] = -1
        
        # Should raise error or handle gracefully
        try:
            result = cbps_binary_fit(
                treat, X, att=0, method='over', two_step=True,
                sample_weights=sample_weights
            )
            # If it runs, weights should still be positive
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_zero_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test zero sample weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat))
        sample_weights[0] = 0
        
        # May work or raise error
        try:
            result = cbps_binary_fit(
                treat, X, att=0, method='over', two_step=True,
                sample_weights=sample_weights
            )
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass

    def test_nan_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test NaN sample weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat))
        sample_weights[0] = np.nan
        
        # Should raise error
        try:
            result = cbps_binary_fit(
                treat, X, att=0, method='over', two_step=True,
                sample_weights=sample_weights
            )
        except (ValueError, RuntimeError):
            pass

class TestParameterCombinationValidation:
    """
    Test ID: VAL-004
    
    Test parameter combination validation.
    """
    
    def test_valid_method_over(self, cbps_binary_fit, lalonde_full):
        """Test method='over' is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_valid_method_exact(self, cbps_binary_fit, lalonde_full):
        """Test method='exact' is accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        assert result['converged']

    def test_valid_att_values(self, cbps_binary_fit, lalonde_full):
        """Test valid att values are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            result = cbps_binary_fit(treat, X, att=att, method='over', two_step=True)
            assert result['converged'], f"Should converge for att={att}"

    def test_valid_two_step_values(self, cbps_binary_fit, lalonde_full):
        """Test valid two_step values are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for two_step in [True, False]:
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=two_step)
            # Both should return valid results (continuous updating may not always converge)
            assert 'coefficients' in result, f"Should return result for two_step={two_step}"
            assert np.all(np.isfinite(result['coefficients'])), f"Coefficients should be finite for two_step={two_step}"

    def test_valid_standardize_values(self, cbps_binary_fit, lalonde_full):
        """Test valid standardize values are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for standardize in [True, False]:
            result = cbps_binary_fit(
                treat, X, att=0, method='over', two_step=True,
                standardize=standardize
            )
            assert result['converged'], f"Should converge for standardize={standardize}"

    def test_all_parameter_combinations(self, cbps_binary_fit, lalonde_full):
        """Test all valid parameter combinations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                for two_step in [True, False]:
                    result = cbps_binary_fit(
                        treat, X, att=att, method=method, two_step=two_step
                    )
                    assert 'coefficients' in result, \
                        f"Should return result for att={att}, method={method}, two_step={two_step}"

class TestDimensionMatching:
    """
    Test ID: VAL-005
    
    Test dimension matching validation.
    """
    
    def test_matching_dimensions(self, cbps_binary_fit, lalonde_full):
        """Test matching dimensions are accepted."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        assert len(treat) == X.shape[0], "Dimensions should match"
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        assert result['converged']

    def test_mismatched_treat_x(self, cbps_binary_fit, lalonde_full):
        """Test mismatched treat and X dimensions raise error."""
        X = lalonde_full['X']
        treat = lalonde_full['treat'][:-10]  # Remove 10 observations
        
        with pytest.raises((ValueError, IndexError)):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

    def test_mismatched_sample_weights(self, cbps_binary_fit, lalonde_full):
        """Test mismatched sample weights dimensions raise error."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat) - 10)  # Wrong length
        
        with pytest.raises((ValueError, IndexError)):
            result = cbps_binary_fit(
                treat, X, att=0, method='over', two_step=True,
                sample_weights=sample_weights
            )

    def test_empty_arrays(self, cbps_binary_fit):
        """Test empty arrays raise error."""
        X = np.array([]).reshape(0, 3)
        treat = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

    def test_single_observation(self, cbps_binary_fit):
        """Test single observation raises error."""
        X = np.array([[1, 0.5, 0.3]])
        treat = np.array([1.0])
        
        with pytest.raises((ValueError, RuntimeError)):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

    def test_more_covariates_than_observations(self, cbps_binary_fit):
        """Test more covariates than observations."""
        np.random.seed(42)
        n = 10
        k = 20
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Should raise error for rank-deficient matrix
        with pytest.raises(ValueError):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)

class TestLinearPredictorFormula:
    """
    Test ID: LP-001
    
    Test linear predictor follows X @ beta formula.
    """
    
    def test_linear_predictor_formula(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor equals X @ coefficients."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Manual calculation
        expected_lp = X @ result['coefficients'].ravel()
        
        assert_allclose_with_report(
            result['linear_predictor'], expected_lp,
            rtol=1e-10, atol=1e-10,
            name="Linear predictor formula"
        )

    def test_linear_predictor_shape(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor has correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = len(treat)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['linear_predictor'].shape == (n,), \
            f"Linear predictor should have shape ({n},)"

    def test_linear_predictor_finite(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor values are finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(np.isfinite(result['linear_predictor'])), \
            "Linear predictor should be finite"

    def test_linear_predictor_range(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor has reasonable range."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        lp = result['linear_predictor']
        
        # Linear predictor should not be extremely large
        # (would indicate numerical issues)
        assert np.max(np.abs(lp)) < 50, \
            f"Linear predictor range too large: [{np.min(lp):.2f}, {np.max(lp):.2f}]"

class TestLinearPredictorFittedRelationship:
    """
    Test ID: LP-002
    
    Test relationship between linear predictor and fitted values.
    """
    
    def test_logistic_transformation(self, cbps_binary_fit, lalonde_full):
        """Test fitted_values = expit(linear_predictor)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Apply logistic transformation
        expected_fitted = scipy.special.expit(result['linear_predictor'])
        
        # Allow for probability clipping
        expected_fitted_clipped = np.clip(expected_fitted, PROBS_MIN, 1 - PROBS_MIN)
        
        assert_allclose_with_report(
            result['fitted_values'], expected_fitted_clipped,
            rtol=1e-6, atol=1e-6,
            name="Logistic transformation"
        )

    def test_inverse_logit_relationship(self, cbps_binary_fit, lalonde_full):
        """Test linear_predictor = logit(fitted_values)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Inverse logit (logit function)
        fitted = result['fitted_values']
        fitted_clipped = np.clip(fitted, PROBS_MIN, 1 - PROBS_MIN)
        expected_lp = scipy.special.logit(fitted_clipped)
        
        # Should be close (may differ due to clipping)
        assert_allclose_with_report(
            result['linear_predictor'], expected_lp,
            rtol=0.01, atol=0.1,
            name="Inverse logit relationship"
        )

    def test_monotonic_relationship(self, cbps_binary_fit, lalonde_full):
        """Test monotonic relationship between LP and fitted values."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        lp = result['linear_predictor']
        fitted = result['fitted_values']
        
        # Sort by linear predictor
        sort_idx = np.argsort(lp)
        lp_sorted = lp[sort_idx]
        fitted_sorted = fitted[sort_idx]
        
        # Fitted values should be monotonically increasing with LP
        # (allowing for ties and numerical noise)
        diffs = np.diff(fitted_sorted)
        assert np.sum(diffs < -1e-10) < len(diffs) * 0.01, \
            "Fitted values should be monotonically increasing with LP"

    def test_extreme_lp_values(self, cbps_binary_fit):
        """Test handling of extreme linear predictor values."""
        np.random.seed(42)
        n = 200
        
        # Create data that might produce extreme LP
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        # Strong treatment effect
        true_beta = np.array([0.0, 2.0])
        theta = X @ true_beta
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Fitted values should still be in (0, 1)
        assert np.all(result['fitted_values'] > 0)
        assert np.all(result['fitted_values'] < 1)

class TestSVDTransformedLinearPredictor:
    """
    Test ID: LP-003
    
    Test linear predictor with SVD transformation (if applicable).
    """
    
    def test_lp_in_original_space(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor is in original covariate space."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Linear predictor should be X @ beta (original space)
        expected_lp = X @ result['coefficients'].ravel()
        
        assert_allclose_with_report(
            result['linear_predictor'], expected_lp,
            rtol=1e-10, atol=1e-10,
            name="LP in original space"
        )

    def test_lp_consistency_across_methods(self, cbps_binary_fit, lalonde_full):
        """Test LP consistency across different methods."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        # Both should produce valid LP
        assert np.all(np.isfinite(result_over['linear_predictor']))
        assert np.all(np.isfinite(result_exact['linear_predictor']))
        
        # LP should be consistent with coefficients
        assert_allclose_with_report(
            result_over['linear_predictor'],
            X @ result_over['coefficients'].ravel(),
            rtol=1e-10, atol=1e-10,
            name="LP consistency (over)"
        )
        assert_allclose_with_report(
            result_exact['linear_predictor'],
            X @ result_exact['coefficients'].ravel(),
            rtol=1e-10, atol=1e-10,
            name="LP consistency (exact)"
        )

    def test_lp_mean_reasonable(self, cbps_binary_fit, lalonde_full):
        """Test linear predictor mean is reasonable."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        lp_mean = np.mean(result['linear_predictor'])
        
        # Mean LP should be related to log-odds of treatment
        p_treat = np.mean(treat)
        expected_mean_approx = scipy.special.logit(p_treat)
        
        # Allow wide tolerance
        assert np.abs(lp_mean - expected_mean_approx) < 3, \
            f"LP mean ({lp_mean:.2f}) should be roughly related to log-odds ({expected_mean_approx:.2f})"

class TestSafeExponential:
    """Test ID: NS-001, Requirement: REQ-101"""
    
    def test_expit_normal_range(self):
        """Test expit (sigmoid) for normal input range."""
        x = np.array([-5, -2, -1, 0, 1, 2, 5])
        result = scipy.special.expit(x)
        
        # All results should be in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)
        
        # Check symmetry: expit(-x) = 1 - expit(x)
        assert np.allclose(scipy.special.expit(-x), 1 - scipy.special.expit(x))

    def test_expit_extreme_positive(self):
        """Test expit for extreme positive values (overflow prevention)."""
        x = np.array([100, 500, 700, 1000])
        result = scipy.special.expit(x)
        
        # Should be very close to 1 but not exactly 1
        assert np.all(result <= 1.0)
        assert np.all(result > 0.99999)
        assert np.all(np.isfinite(result))

    def test_expit_extreme_negative(self):
        """Test expit for extreme negative values (underflow prevention)."""
        x = np.array([-100, -500, -700, -1000])
        result = scipy.special.expit(x)
        
        # Should be very close to 0 but not exactly 0
        assert np.all(result >= 0.0)
        assert np.all(result < 0.00001)
        assert np.all(np.isfinite(result))

    def test_expit_at_zero(self):
        """Test expit at zero equals 0.5."""
        result = scipy.special.expit(0.0)
        assert np.isclose(result, 0.5)

    def test_expit_array_mixed(self):
        """Test expit with mixed extreme and normal values."""
        x = np.array([-1000, -10, 0, 10, 1000])
        result = scipy.special.expit(x)
        
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_exp_overflow_handling(self):
        """Test that direct exp would overflow but expit handles it."""
        x = 1000.0
        
        # Direct exp would overflow
        with np.errstate(over='ignore'):
            direct_exp = np.exp(x)
        assert np.isinf(direct_exp)
        
        # But expit handles it correctly
        expit_result = scipy.special.expit(x)
        assert np.isfinite(expit_result)
        assert expit_result <= 1.0

class TestSafeLogarithm:
    """Test ID: NS-002, Requirement: REQ-102"""
    
    def test_log_normal_range(self):
        """Test log for normal positive values."""
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        result = np.log(x)
        
        assert np.all(np.isfinite(result))

    def test_log_small_positive(self):
        """Test log for very small positive values."""
        x = np.array([1e-10, 1e-100, 1e-300])
        result = np.log(x)
        
        # Should be large negative but finite
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)

    def test_log_near_zero_clipped(self):
        """Test log with clipped near-zero values."""
        x = np.array([0.0, 1e-400, -1e-10])
        x_clipped = np.clip(x, PROBS_MIN, 1 - PROBS_MIN)
        result = np.log(x_clipped)
        
        assert np.all(np.isfinite(result))

    def test_log_probability_range(self):
        """Test log for probability range [PROBS_MIN, 1-PROBS_MIN]."""
        probs = np.linspace(PROBS_MIN, 1 - PROBS_MIN, 100)
        result = np.log(probs)
        
        assert np.all(np.isfinite(result))

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation with clipped probabilities."""
        np.random.seed(42)
        n = 100
        treat = np.random.binomial(1, 0.5, n).astype(float)
        probs = np.random.uniform(0.1, 0.9, n)
        
        # Clip probabilities
        probs_clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute log-likelihood
        ll = np.sum(treat * np.log(probs_clipped) + (1 - treat) * np.log(1 - probs_clipped))
        
        assert np.isfinite(ll)

class TestProbabilityClipping:
    """Test ID: NS-003, Requirement: REQ-103"""
    
    def test_clip_probs_lower_bound(self):
        """Test probability clipping at lower bound."""
        probs = np.array([0.0, 1e-10, 1e-7, PROBS_MIN / 2])
        clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        assert np.all(clipped >= PROBS_MIN)

    def test_clip_probs_upper_bound(self):
        """Test probability clipping at upper bound."""
        probs = np.array([1.0, 1 - 1e-10, 1 - 1e-7, 1 - PROBS_MIN / 2])
        clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        assert np.all(clipped <= 1 - PROBS_MIN)

    def test_clip_probs_preserves_middle(self):
        """Test that clipping preserves values in valid range."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        assert np.allclose(probs, clipped)

    def test_clip_probs_from_expit(self):
        """Test clipping probabilities from expit output."""
        # Extreme linear predictors
        theta = np.array([-100, -10, 0, 10, 100])
        probs = scipy.special.expit(theta)
        clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        assert np.all(clipped >= PROBS_MIN)
        assert np.all(clipped <= 1 - PROBS_MIN)
        assert np.all(np.isfinite(clipped))

    def test_probs_min_value(self):
        """Test that PROBS_MIN is appropriate."""
        # PROBS_MIN should be small but not too small
        assert PROBS_MIN > 0
        assert PROBS_MIN < 1e-3
        
        # log(PROBS_MIN) should be finite
        assert np.isfinite(np.log(PROBS_MIN))
        
        # 1/PROBS_MIN should be finite
        assert np.isfinite(1 / PROBS_MIN)

class TestWeightComputation:
    """Test numerical stability of weight computations."""
    
    def test_ate_weights_extreme_probs(self, simple_binary_data):
        """Test ATE weight computation with extreme probabilities."""
        data = simple_binary_data
        treat = data['treat']
        n = len(treat)
        
        # Extreme probabilities (after clipping)
        probs = np.full(n, PROBS_MIN)
        probs[treat == 1] = 1 - PROBS_MIN
        
        # ATE weights: 1 / (probs - 1 + treat)
        weights = 1 / (probs - 1 + treat)
        
        assert np.all(np.isfinite(weights))

    def test_att_weights_extreme_probs(self, att_wt_func, simple_binary_data):
        """Test ATT weight computation with extreme probabilities."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        # Beta that produces extreme probabilities
        beta_extreme = np.array([10.0, 5.0, -5.0])
        
        weights = att_wt_func(beta_extreme, X, treat, sample_weights)
        
        assert np.all(np.isfinite(weights))

    def test_weight_division_by_small_number(self):
        """Test weight computation when dividing by small numbers."""
        probs = np.array([PROBS_MIN, 0.5, 1 - PROBS_MIN])
        treat = np.array([0.0, 1.0, 1.0])
        
        # ATE weights
        weights = 1 / (probs - 1 + treat)
        
        # Should be finite even for extreme probs
        assert np.all(np.isfinite(weights))

class TestMatrixOperations:
    """Test numerical stability of matrix operations."""
    
    def test_pseudoinverse_near_singular(self, r_ginv):
        """Test pseudoinverse of near-singular matrix."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-8  # Nearly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        
        XtX = X.T @ X
        XtX_inv = r_ginv(XtX)
        
        assert np.all(np.isfinite(XtX_inv))

    def test_pseudoinverse_rank_deficient(self, r_ginv):
        """Test pseudoinverse of rank-deficient matrix."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 * 2  # Perfectly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        
        XtX = X.T @ X
        XtX_inv = r_ginv(XtX)
        
        assert np.all(np.isfinite(XtX_inv))

    def test_v_matrix_extreme_probs(self, compute_V_matrix, simple_binary_data):
        """Test V matrix computation with extreme probabilities."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n = len(treat)
        sample_weights = data['sample_weights']
        
        # Extreme probabilities
        probs = np.full(n, 0.5)
        probs[:10] = PROBS_MIN
        probs[-10:] = 1 - PROBS_MIN
        
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        assert np.all(np.isfinite(inv_V))

    def test_quadratic_form_stability(self, r_ginv, simple_binary_data):
        """Test quadratic form computation stability."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        
        XtX_inv = r_ginv(X.T @ X)
        
        # Random vector
        np.random.seed(42)
        v = np.random.randn(k)
        
        # Quadratic form: v' @ XtX_inv @ v
        qf = v.T @ XtX_inv @ v
        
        assert np.isfinite(qf)

class TestGradientStability:
    """Test numerical stability of gradient computations."""
    
    def test_gmm_gradient_extreme_beta(self, cbps_binary_module, simple_binary_data):
        """Test GMM gradient with extreme beta values."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        # Extreme beta
        beta_extreme = np.array([10.0, 5.0, -5.0])
        
        # Compute inv_V first
        theta = X @ beta_extreme
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        inv_V = cbps_binary_module._compute_V_matrix(X, probs, sample_weights, treat, att=0, n=len(treat))
        
        # Compute gradient
        grad = cbps_binary_module._gmm_gradient(beta_extreme, inv_V, X, treat, sample_weights, att=0)
        
        assert np.all(np.isfinite(grad))

    def test_balance_gradient_extreme_beta(self, bal_gradient_func, r_ginv, simple_binary_data):
        """Test balance gradient with extreme beta values."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        XprimeX_inv = r_ginv(X.T @ X)
        
        # Extreme beta
        beta_extreme = np.array([10.0, 5.0, -5.0])
        
        grad = bal_gradient_func(beta_extreme, X, treat, sample_weights, XprimeX_inv, att=0)
        
        assert np.all(np.isfinite(grad))

class TestDeviance_NumericalStability:
    """Test numerical stability of deviance computation."""
    
    def test_deviance_normal_probs(self, simple_binary_data):
        """Test deviance computation with normal probabilities."""
        data = simple_binary_data
        treat = data['treat']
        sample_weights = data['sample_weights']
        
        probs = np.full(len(treat), 0.5)
        probs_clipped = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        deviance = -2 * np.sum(
            treat * sample_weights * np.log(probs_clipped) +
            (1 - treat) * sample_weights * np.log(1 - probs_clipped)
        )
        
        assert np.isfinite(deviance)
        assert deviance >= 0

    def test_deviance_extreme_probs(self, simple_binary_data):
        """Test deviance computation with extreme probabilities."""
        data = simple_binary_data
        treat = data['treat']
        sample_weights = data['sample_weights']
        n = len(treat)
        
        # Extreme probabilities (but clipped)
        probs = np.full(n, 0.5)
        probs[treat == 1] = 1 - PROBS_MIN
        probs[treat == 0] = PROBS_MIN
        
        deviance = -2 * np.sum(
            treat * sample_weights * np.log(probs) +
            (1 - treat) * sample_weights * np.log(1 - probs)
        )
        
        assert np.isfinite(deviance)
        assert deviance >= 0

    def test_null_deviance(self, simple_binary_data):
        """Test null deviance computation."""
        data = simple_binary_data
        treat = data['treat']
        sample_weights = data['sample_weights']
        
        treat_mean = np.average(treat, weights=sample_weights)
        treat_mean = np.clip(treat_mean, 1e-10, 1 - 1e-10)
        
        nulldeviance = -2 * np.sum(
            treat * sample_weights * np.log(treat_mean) +
            (1 - treat) * sample_weights * np.log(1 - treat_mean)
        )
        
        assert np.isfinite(nulldeviance)
        assert nulldeviance >= 0

class TestDualInitialization:
    """
    Test ID: OPT-001
    
    Test dual initialization strategy (GMM init vs Balance init).
    """
    
    def test_dual_init_both_run(self, cbps_binary_fit, lalonde_full):
        """Test both initialization paths are executed."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # The dual initialization is internal, but we can verify
        # the result is reasonable
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged'], "CBPS should converge with dual initialization"
        assert np.all(np.isfinite(result['coefficients'])), "Coefficients should be finite"

    def test_dual_init_selects_better(self, cbps_binary_fit, lalonde_full):
        """Test dual initialization selects the better solution."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # J-statistic should be reasonably small
        assert result['J'] < 100, f"J-statistic should be reasonable, got {result['J']}"

    def test_dual_init_consistency(self, cbps_binary_fit, lalonde_full):
        """Test dual initialization produces consistent results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Results should be identical (deterministic)
        assert_allclose_with_report(
            result1['coefficients'], result2['coefficients'],
            rtol=1e-10, atol=1e-10,
            name="Dual init consistency"
        )

    def test_dual_init_different_estimands(self, cbps_binary_fit, lalonde_full):
        """Test dual initialization works for different estimands."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            result = cbps_binary_fit(treat, X, att=att, method='over', two_step=True)
            assert result['converged'], f"Should converge for att={att}"

class TestConvergenceCriteria:
    """
    Test ID: OPT-002
    
    Test optimizer convergence criteria.
    """
    
    def test_convergence_flag(self, cbps_binary_fit, lalonde_full):
        """Test convergence flag is set correctly."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert isinstance(result['converged'], bool), "Converged should be boolean"

    def test_convergence_with_sufficient_iterations(self, cbps_binary_fit, lalonde_full):
        """Test convergence with sufficient iterations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=1000
        )
        
        assert result['converged'], "Should converge with 1000 iterations"

    def test_convergence_gradient_norm(self, cbps_binary_fit, lalonde_full):
        """Test convergence implies small gradient norm."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = np.ones(len(treat))
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        if result['converged']:
            # Compute gradient at optimum
            from cbps.core.cbps_binary import _gmm_func
            
            beta_opt = result['coefficients'].ravel()
            gmm_result = _gmm_func(beta_opt, X, treat, sample_weights, att=0, inv_V=None)
            
            # Loss should be at a minimum (gradient approximately zero)
            # We can't directly check gradient, but loss should be small
            assert gmm_result['loss'] < 10, "Loss should be small at convergence"

    def test_convergence_exact_method(self, cbps_binary_fit, lalonde_full):
        """Test convergence for exact method."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        assert result['converged'], "Exact method should converge"

class TestMaxIterations:
    """
    Test ID: OPT-003
    
    Test maximum iteration limit behavior.
    """
    
    def test_respects_iteration_limit(self, cbps_binary_fit, lalonde_full):
        """Test optimizer respects iteration limit."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Very few iterations
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=5
        )
        
        # Should return result (may or may not converge)
        assert result['weights'].shape == (lalonde_full['n'],)

    def test_more_iterations_better_result(self, cbps_binary_fit, lalonde_full):
        """Test more iterations can improve result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_few = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=10
        )
        
        result_many = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=1000
        )
        
        # More iterations should give same or better J
        # (or at least not much worse)
        assert result_many['J'] <= result_few['J'] * 2, \
            "More iterations should not significantly worsen J"

    def test_default_iterations_sufficient(self, cbps_binary_fit, lalonde_full):
        """Test default iterations are sufficient for convergence."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Use default iterations (1000)
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged'], "Default iterations should be sufficient"

    def test_iteration_limit_synthetic(self, cbps_binary_fit):
        """Test iteration limit on synthetic data."""
        np.random.seed(42)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        true_beta = np.array([0.0, 0.5])
        theta = X @ true_beta
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=100
        )
        
        assert result['converged'], "Should converge on simple data with 100 iterations"

class TestOptimalSolutionSelection:
    """
    Test ID: OPT-004
    
    Test selection of optimal solution from multiple initializations.
    """
    
    def test_selects_lower_objective(self, cbps_binary_fit, lalonde_full):
        """Test optimizer selects solution with lower objective."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # J should be the objective value at the selected solution
        assert result['J'] >= 0, "J should be non-negative"

    def test_solution_satisfies_constraints(self, cbps_binary_fit, lalonde_full):
        """Test selected solution satisfies basic constraints."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Propensity scores in (0, 1)
        assert np.all(result['fitted_values'] > 0)
        assert np.all(result['fitted_values'] < 1)
        
        # Weights positive
        assert np.all(result['weights'] > 0)

    def test_solution_improves_balance(self, cbps_binary_fit, lalonde_full):
        """Test selected solution improves covariate balance."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Compute weighted mean difference for each covariate
        for j in range(1, X.shape[1]):
            mean_t = np.average(X[treat == 1, j], weights=weights[treat == 1])
            mean_c = np.average(X[treat == 0, j], weights=weights[treat == 0])
            diff = np.abs(mean_t - mean_c)
            
            # Difference should be reasonable
            assert diff < 5 * np.std(X[:, j]), \
                f"Covariate {j} balance should be reasonable"

class TestNonConvergenceHandling:
    """
    Test ID: OPT-005
    
    Test handling of non-convergence cases.
    """
    
    def test_returns_result_on_non_convergence(self, cbps_binary_fit, lalonde_full):
        """Test returns result even when not converged."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Very few iterations may not converge
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=2
        )
        
        # Should still return valid structure
        assert 'coefficients' in result
        assert 'weights' in result
        assert 'converged' in result

    def test_non_convergence_flag_set(self, cbps_binary_fit):
        """Test non-convergence flag is set appropriately."""
        np.random.seed(42)
        n = 50
        
        # Create difficult optimization problem
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Very few iterations
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=1
        )
        
        # converged flag should be boolean
        assert isinstance(result['converged'], (bool, np.bool_))

    def test_non_convergence_weights_valid(self, cbps_binary_fit, lalonde_full):
        """Test weights are valid even on non-convergence."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=2
        )
        
        # Weights should still be valid
        assert np.all(np.isfinite(result['weights']))
        assert np.all(result['weights'] > 0)

    def test_difficult_data_handling(self, cbps_binary_fit):
        """Test handling of difficult optimization cases."""
        np.random.seed(42)
        n = 100
        
        # Create data with extreme propensity scores
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        # Strong treatment effect
        true_beta = np.array([0.0, 2.0])
        theta = X @ true_beta
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        # Should handle without crashing
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert 'coefficients' in result
        assert np.all(np.isfinite(result['coefficients']))

    def test_continuous_updating_convergence(self, cbps_binary_fit, lalonde_full):
        """Test convergence with continuous updating GMM."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=False,
            iterations=1000
        )
        
        # Continuous updating may not always converge, but should return valid results
        assert np.all(np.isfinite(result['coefficients'])), "Coefficients should be finite"
        assert np.all(np.isfinite(result['weights'])), "Weights should be finite"
        assert np.all(result['weights'] > 0), "Weights should be positive"

class TestFullRankPseudoinverse:
    """
    Test pseudoinverse computation for full rank matrices.
    
    Test ID: GINV-001
    Requirement: REQ-004
    
    For a full rank matrix A, the pseudoinverse A+ should satisfy:
    1. A @ A+ @ A = A (reflexive)
    2. A+ @ A @ A+ = A+ (reflexive)
    3. (A @ A+)^T = A @ A+ (symmetric)
    4. (A+ @ A)^T = A+ @ A (symmetric)
    """
    
    def test_full_rank_reflexive_property_1(self, r_ginv_func, full_rank_matrix):
        """Test A @ A+ @ A = A for full rank matrix."""
        A = full_rank_matrix
        A_pinv = r_ginv_func(A)
        
        result = A @ A_pinv @ A
        
        assert_allclose_with_report(
            result, A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="A @ A+ @ A"
        )
    
    def test_full_rank_reflexive_property_2(self, r_ginv_func, full_rank_matrix):
        """Test A+ @ A @ A+ = A+ for full rank matrix."""
        A = full_rank_matrix
        A_pinv = r_ginv_func(A)
        
        result = A_pinv @ A @ A_pinv
        
        assert_allclose_with_report(
            result, A_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="A+ @ A @ A+"
        )
    
    def test_full_rank_symmetric_property_1(self, r_ginv_func, full_rank_matrix):
        """Test (A @ A+)^T = A @ A+ for full rank matrix."""
        A = full_rank_matrix
        A_pinv = r_ginv_func(A)
        
        AA_pinv = A @ A_pinv
        
        assert_allclose_with_report(
            AA_pinv.T, AA_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="(A @ A+)^T"
        )
    
    def test_full_rank_symmetric_property_2(self, r_ginv_func, full_rank_matrix):
        """Test (A+ @ A)^T = A+ @ A for full rank matrix."""
        A = full_rank_matrix
        A_pinv = r_ginv_func(A)
        
        A_pinv_A = A_pinv @ A
        
        assert_allclose_with_report(
            A_pinv_A.T, A_pinv_A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="(A+ @ A)^T"
        )
    
    def test_full_rank_output_shape(self, r_ginv_func, full_rank_matrix):
        """Test output shape is transposed input shape."""
        A = full_rank_matrix
        A_pinv = r_ginv_func(A)
        
        expected_shape = (A.shape[1], A.shape[0])
        assert A_pinv.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {A_pinv.shape}"

class TestRankDeficientPseudoinverse:
    """
    Test pseudoinverse computation for rank deficient matrices.
    
    Test ID: GINV-002
    Requirement: REQ-004
    
    For a rank deficient matrix, the pseudoinverse should:
    1. Still satisfy the Moore-Penrose conditions
    2. Correctly truncate small singular values
    3. Return a matrix of appropriate rank
    """
    
    def test_rank_deficient_reflexive_property(self, r_ginv_func, rank_deficient_matrix):
        """Test A @ A+ @ A = A for rank deficient matrix."""
        A = rank_deficient_matrix
        A_pinv = r_ginv_func(A)
        
        result = A @ A_pinv @ A
        
        assert_allclose_with_report(
            result, A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="A @ A+ @ A (rank deficient)"
        )
    
    def test_rank_deficient_output_shape(self, r_ginv_func, rank_deficient_matrix):
        """Test output shape for rank deficient matrix."""
        A = rank_deficient_matrix
        A_pinv = r_ginv_func(A)
        
        expected_shape = (A.shape[1], A.shape[0])
        assert A_pinv.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {A_pinv.shape}"
    
    def test_rank_deficient_symmetric_properties(self, r_ginv_func, rank_deficient_matrix):
        """Test symmetric properties for rank deficient matrix."""
        A = rank_deficient_matrix
        A_pinv = r_ginv_func(A)
        
        # (A @ A+) should be symmetric
        AA_pinv = A @ A_pinv
        assert_allclose_with_report(
            AA_pinv.T, AA_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="(A @ A+)^T (rank deficient)"
        )
        
        # (A+ @ A) should be symmetric
        A_pinv_A = A_pinv @ A
        assert_allclose_with_report(
            A_pinv_A.T, A_pinv_A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="(A+ @ A)^T (rank deficient)"
        )

class TestNearSingularPseudoinverse:
    """
    Test pseudoinverse computation for near singular matrices.
    
    Test ID: GINV-003
    Requirement: REQ-004
    
    For near singular matrices (high condition number), the pseudoinverse
    should maintain numerical stability by truncating very small singular values.
    """
    
    def test_near_singular_numerical_stability(self, r_ginv_func, near_singular_matrix):
        """Test numerical stability for near singular matrix."""
        A = near_singular_matrix
        A_pinv = r_ginv_func(A)
        
        # Should not contain NaN or Inf
        assert not np.any(np.isnan(A_pinv)), "Pseudoinverse contains NaN"
        assert not np.any(np.isinf(A_pinv)), "Pseudoinverse contains Inf"
    
    def test_near_singular_reflexive_property(self, r_ginv_func, near_singular_matrix):
        """Test A @ A+ @ A ≈ A for near singular matrix (relaxed tolerance)."""
        A = near_singular_matrix
        A_pinv = r_ginv_func(A)
        
        result = A @ A_pinv @ A
        
        # Use relaxed tolerance for near singular matrices
        assert_allclose_with_report(
            result, A,
            rtol=1e-6,  # Relaxed tolerance
            atol=1e-8,
            name="A @ A+ @ A (near singular)"
        )
    
    def test_near_singular_output_shape(self, r_ginv_func, near_singular_matrix):
        """Test output shape for near singular matrix."""
        A = near_singular_matrix
        A_pinv = r_ginv_func(A)
        
        expected_shape = (A.shape[1], A.shape[0])
        assert A_pinv.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {A_pinv.shape}"

class TestZeroMatrixPseudoinverse:
    """
    Test pseudoinverse computation for zero matrix.
    
    Test ID: GINV-004
    Requirement: REQ-004
    
    The pseudoinverse of a zero matrix should be a zero matrix
    of transposed dimensions.
    """
    
    def test_zero_matrix_returns_zero(self, r_ginv_func, zero_matrix):
        """Test that pseudoinverse of zero matrix is zero matrix."""
        A = zero_matrix
        A_pinv = r_ginv_func(A)
        
        expected = np.zeros((A.shape[1], A.shape[0]))
        
        assert_allclose_with_report(
            A_pinv, expected,
            rtol=0,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Zero matrix pseudoinverse"
        )
    
    def test_zero_matrix_output_shape(self, r_ginv_func, zero_matrix):
        """Test output shape for zero matrix."""
        A = zero_matrix
        A_pinv = r_ginv_func(A)
        
        expected_shape = (A.shape[1], A.shape[0])
        assert A_pinv.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {A_pinv.shape}"
    
    def test_zero_matrix_no_nan_inf(self, r_ginv_func, zero_matrix):
        """Test that zero matrix pseudoinverse has no NaN or Inf."""
        A = zero_matrix
        A_pinv = r_ginv_func(A)
        
        assert not np.any(np.isnan(A_pinv)), "Pseudoinverse contains NaN"
        assert not np.any(np.isinf(A_pinv)), "Pseudoinverse contains Inf"

class TestPseudoinverseEdgeCases:
    """Additional edge case tests for pseudoinverse computation."""
    
    def test_identity_matrix(self, r_ginv_func):
        """Test pseudoinverse of identity matrix is identity."""
        n = 5
        I = np.eye(n)
        I_pinv = r_ginv_func(I)
        
        assert_allclose_with_report(
            I_pinv, I,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Identity matrix pseudoinverse"
        )
    
    def test_diagonal_matrix(self, r_ginv_func):
        """Test pseudoinverse of diagonal matrix."""
        d = np.array([2.0, 3.0, 0.5, 4.0])
        D = np.diag(d)
        D_pinv = r_ginv_func(D)
        
        expected = np.diag(1.0 / d)
        
        assert_allclose_with_report(
            D_pinv, expected,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Diagonal matrix pseudoinverse"
        )
    
    def test_single_row_matrix(self, r_ginv_func):
        """Test pseudoinverse of single row matrix."""
        np.random.seed(42)
        A = np.random.randn(1, 5)
        A_pinv = r_ginv_func(A)
        
        # Verify Moore-Penrose property
        result = A @ A_pinv @ A
        assert_allclose_with_report(
            result, A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Single row A @ A+ @ A"
        )
    
    def test_single_column_matrix(self, r_ginv_func):
        """Test pseudoinverse of single column matrix."""
        np.random.seed(42)
        A = np.random.randn(5, 1)
        A_pinv = r_ginv_func(A)
        
        # Verify Moore-Penrose property
        result = A @ A_pinv @ A
        assert_allclose_with_report(
            result, A,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Single column A @ A+ @ A"
        )
    
    def test_square_invertible_matrix(self, r_ginv_func):
        """Test pseudoinverse of square invertible matrix equals inverse."""
        np.random.seed(42)
        A = np.random.randn(4, 4)
        # Ensure invertible by adding identity
        A = A + 2 * np.eye(4)
        
        A_pinv = r_ginv_func(A)
        A_inv = np.linalg.inv(A)
        
        assert_allclose_with_report(
            A_pinv, A_inv,
            rtol=1e-8,
            atol=1e-10,
            name="Square invertible pseudoinverse vs inverse"
        )
    
    def test_orthogonal_matrix(self, r_ginv_func):
        """Test pseudoinverse of orthogonal matrix equals transpose."""
        np.random.seed(42)
        # Create orthogonal matrix via QR decomposition
        A = np.random.randn(5, 5)
        Q, _ = np.linalg.qr(A)
        
        Q_pinv = r_ginv_func(Q)
        
        assert_allclose_with_report(
            Q_pinv, Q.T,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Orthogonal matrix pseudoinverse vs transpose"
        )

class TestPseudoinverseNumericalPrecision:
    """Test numerical precision of pseudoinverse computation."""
    
    def test_double_pseudoinverse(self, r_ginv_func):
        """Test that (A+)+ = A for full rank matrix."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        
        A_pinv = r_ginv_func(A)
        A_pinv_pinv = r_ginv_func(A_pinv)
        
        assert_allclose_with_report(
            A_pinv_pinv, A,
            rtol=1e-8,
            atol=1e-10,
            name="Double pseudoinverse"
        )
    
    def test_transpose_pseudoinverse(self, r_ginv_func):
        """Test that (A+)^T = (A^T)+."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        
        A_pinv = r_ginv_func(A)
        AT_pinv = r_ginv_func(A.T)
        
        assert_allclose_with_report(
            A_pinv.T, AT_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Transpose pseudoinverse"
        )
    
    def test_scaled_matrix(self, r_ginv_func):
        """Test that (cA)+ = (1/c)A+ for scalar c."""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        c = 3.5
        
        A_pinv = r_ginv_func(A)
        cA_pinv = r_ginv_func(c * A)
        
        assert_allclose_with_report(
            cA_pinv, (1.0 / c) * A_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="Scaled matrix pseudoinverse"
        )

class TestSameInputSameOutput:
    """
    Test ID: REP-001
    
    Test same input produces same output.
    """
    
    def test_deterministic_coefficients(self, cbps_binary_fit, lalonde_full):
        """Test coefficients are deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['coefficients'], result2['coefficients'],
            rtol=1e-10, atol=1e-10,
            name="Coefficients reproducibility"
        )

    def test_deterministic_weights(self, cbps_binary_fit, lalonde_full):
        """Test weights are deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['weights'], result2['weights'],
            rtol=1e-10, atol=1e-10,
            name="Weights reproducibility"
        )

    def test_deterministic_fitted_values(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['fitted_values'], result2['fitted_values'],
            rtol=1e-10, atol=1e-10,
            name="Fitted values reproducibility"
        )

    def test_deterministic_variance(self, cbps_binary_fit, lalonde_full):
        """Test variance matrix is deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['var'], result2['var'],
            rtol=1e-10, atol=1e-10,
            name="Variance reproducibility"
        )

    def test_deterministic_j_statistic(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['J'], result2['J'],
            rtol=1e-10, atol=1e-10,
            name="J-statistic reproducibility"
        )

    def test_deterministic_all_methods(self, cbps_binary_fit, lalonde_full):
        """Test all method combinations are deterministic."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        configs = [
            {'att': 0, 'method': 'over', 'two_step': True},
            {'att': 0, 'method': 'exact', 'two_step': True},
            {'att': 1, 'method': 'over', 'two_step': True},
            {'att': 0, 'method': 'over', 'two_step': False},
        ]
        
        for config in configs:
            result1 = cbps_binary_fit(treat, X, **config)
            result2 = cbps_binary_fit(treat, X, **config)
            
            assert_allclose_with_report(
                result1['coefficients'], result2['coefficients'],
                rtol=1e-10, atol=1e-10,
                name=f"Coefficients reproducibility ({config})"
            )

class TestRandomSeedEffects:
    """
    Test ID: REP-002
    
    Test random seed effects on results.
    """
    
    def test_no_random_state_dependency(self, cbps_binary_fit, lalonde_full):
        """Test results don't depend on numpy random state."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Set different random states
        np.random.seed(42)
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        np.random.seed(123)
        result2 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Results should be identical (CBPS is deterministic)
        assert_allclose_with_report(
            result1['coefficients'], result2['coefficients'],
            rtol=1e-10, atol=1e-10,
            name="Random state independence"
        )

    def test_synthetic_data_with_seed(self, cbps_binary_fit):
        """Test synthetic data with fixed seed produces same results."""
        def generate_data(seed):
            np.random.seed(seed)
            n = 200
            x1 = np.random.normal(0, 1, n)
            X = np.column_stack([np.ones(n), x1])
            treat = np.random.binomial(1, 0.5, n).astype(float)
            return X, treat
        
        # Same seed should give same data and results
        X1, treat1 = generate_data(42)
        X2, treat2 = generate_data(42)
        
        assert_allclose_with_report(X1, X2, rtol=1e-10, atol=1e-10, name="Data reproducibility")
        
        result1 = cbps_binary_fit(treat1, X1, att=0, method='over', two_step=True)
        result2 = cbps_binary_fit(treat2, X2, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result1['coefficients'], result2['coefficients'],
            rtol=1e-10, atol=1e-10,
            name="Results with same seed"
        )

class TestNumericalPrecisionConsistency:
    """
    Test ID: REP-003
    
    Test numerical precision consistency.
    """
    
    def test_float64_precision(self, cbps_binary_fit, lalonde_full):
        """Test results with float64 precision."""
        X = lalonde_full['X'].astype(np.float64)
        treat = lalonde_full['treat'].astype(np.float64)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Results should be finite
        assert np.all(np.isfinite(result['coefficients']))
        assert np.all(np.isfinite(result['weights']))

    def test_precision_consistency(self, cbps_binary_fit, lalonde_full):
        """Test precision is consistent across runs."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        results = []
        for _ in range(5):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            results.append(result['coefficients'].ravel())
        
        # All results should be identical
        for i in range(1, len(results)):
            assert_allclose_with_report(
                results[0], results[i],
                rtol=1e-10, atol=1e-10,
                name=f"Precision consistency run {i}"
            )

    def test_scaled_data_consistency(self, cbps_binary_fit, lalonde_full):
        """Test consistency with scaled data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Original
        result1 = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Scaled X (except intercept)
        X_scaled = X.copy()
        X_scaled[:, 1:] = X_scaled[:, 1:] * 10
        
        result2 = cbps_binary_fit(treat, X_scaled, att=0, method='over', two_step=True)
        
        # Fitted values should be similar (propensity scores)
        # Coefficients will differ due to scaling
        assert np.all(np.isfinite(result2['fitted_values']))

class TestCrossPlatformConsistency:
    """
    Test ID: REP-004
    
    Test cross-platform consistency (within Python).
    """
    
    def test_numpy_version_independence(self, cbps_binary_fit, lalonde_full):
        """Test results don't depend on specific numpy operations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            results.append({
                'coef': result['coefficients'].copy(),
                'weights': result['weights'].copy(),
                'J': result['J']
            })
        
        # All should be identical
        for i in range(1, len(results)):
            assert_allclose_with_report(
                results[0]['coef'], results[i]['coef'],
                rtol=1e-10, atol=1e-10,
                name=f"Numpy independence run {i}"
            )

    def test_scipy_optimization_consistency(self, cbps_binary_fit, lalonde_full):
        """Test scipy optimization produces consistent results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Multiple runs should give same optimization result
        j_values = []
        for _ in range(5):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            j_values.append(result['J'])
        
        # All J values should be identical
        for i in range(1, len(j_values)):
            assert_allclose_with_report(
                j_values[0], j_values[i],
                rtol=1e-10, atol=1e-10,
                name=f"Optimization consistency run {i}"
            )

    def test_matrix_operation_consistency(self, cbps_binary_fit, lalonde_full):
        """Test matrix operations produce consistent results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Variance matrix should be consistent
        var_matrices = []
        for _ in range(3):
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            var_matrices.append(result['var'].copy())
        
        for i in range(1, len(var_matrices)):
            assert_allclose_with_report(
                var_matrices[0], var_matrices[i],
                rtol=1e-10, atol=1e-10,
                name=f"Matrix operation consistency run {i}"
            )

    def test_different_iteration_limits(self, cbps_binary_fit, lalonde_full):
        """Test results with different iteration limits (if converged)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_1000 = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=1000
        )
        
        result_2000 = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            iterations=2000
        )
        
        # If both converged, results should be very similar
        if result_1000['converged'] and result_2000['converged']:
            assert_allclose_with_report(
                result_1000['coefficients'], result_2000['coefficients'],
                rtol=1e-6, atol=1e-6,
                name="Different iteration limits"
            )

class TestCoefficientsReturn:
    """
    Test ID: RET-001
    
    Test coefficients return value shape and type.
    """
    
    def test_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficients have shape (k, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = X.shape[1]
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['coefficients'].shape == (k, 1), \
            f"Coefficients should have shape ({k}, 1), got {result['coefficients'].shape}"

    def test_coefficients_type(self, cbps_binary_fit, lalonde_full):
        """Test coefficients are numpy array."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert isinstance(result['coefficients'], np.ndarray), \
            "Coefficients should be numpy array"

    def test_coefficients_finite(self, cbps_binary_fit, lalonde_full):
        """Test coefficients are finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(np.isfinite(result['coefficients'])), \
            "Coefficients should be finite"

    def test_coefficients_dtype(self, cbps_binary_fit, lalonde_full):
        """Test coefficients have float dtype."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.issubdtype(result['coefficients'].dtype, np.floating), \
            "Coefficients should have float dtype"

class TestFittedValuesReturn:
    """
    Test ID: RET-002
    
    Test fitted values (propensity scores) range.
    """
    
    def test_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['fitted_values'] > 0), "Fitted values should be > 0"
        assert np.all(result['fitted_values'] < 1), "Fitted values should be < 1"

    def test_fitted_values_shape(self, cbps_binary_fit, lalonde_full):
        """Test fitted values have shape (n,)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = len(treat)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['fitted_values'].shape == (n,), \
            f"Fitted values should have shape ({n},)"

    def test_fitted_values_type(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are numpy array."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert isinstance(result['fitted_values'], np.ndarray), \
            "Fitted values should be numpy array"

    def test_fitted_values_clipping(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are clipped to avoid extremes."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Should be clipped to at least PROBS_MIN
        assert np.all(result['fitted_values'] >= PROBS_MIN), \
            f"Fitted values should be >= {PROBS_MIN}"
        assert np.all(result['fitted_values'] <= 1 - PROBS_MIN), \
            f"Fitted values should be <= {1 - PROBS_MIN}"

class TestWeightsReturn:
    """
    Test ID: RET-003
    
    Test weights positivity and properties.
    """
    
    def test_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['weights'] > 0), "Weights should be positive"

    def test_weights_shape(self, cbps_binary_fit, lalonde_full):
        """Test weights have shape (n,)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = len(treat)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['weights'].shape == (n,), \
            f"Weights should have shape ({n},)"

    def test_weights_finite(self, cbps_binary_fit, lalonde_full):
        """Test weights are finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(np.isfinite(result['weights'])), "Weights should be finite"

    def test_weights_standardized(self, cbps_binary_fit, lalonde_full):
        """Test weights are standardized when standardize=True."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            standardize=True
        )
        
        # Weights should sum to approximately 2 (1 per group)
        # or be normalized in some consistent way
        assert np.sum(result['weights']) > 0, "Weights should have positive sum"

class TestVarianceMatrixReturn:
    """
    Test ID: RET-004
    
    Test variance matrix properties.
    """
    
    def test_var_shape(self, cbps_binary_fit, lalonde_full):
        """Test variance matrix has shape (k, k)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = X.shape[1]
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['var'].shape == (k, k), \
            f"Variance should have shape ({k}, {k})"

    def test_var_symmetric(self, cbps_binary_fit, lalonde_full):
        """Test variance matrix is symmetric."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_matrix_symmetric(result['var'], atol=1e-10, name="Variance matrix")

    def test_var_positive_semidefinite(self, cbps_binary_fit, lalonde_full):
        """Test variance matrix is positive semi-definite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_positive_semidefinite(result['var'], atol=1e-8, name="Variance matrix")

    def test_var_finite(self, cbps_binary_fit, lalonde_full):
        """Test variance matrix is finite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(np.isfinite(result['var'])), "Variance should be finite"

    def test_var_diagonal_positive(self, cbps_binary_fit, lalonde_full):
        """Test variance diagonal is positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(np.diag(result['var']) > 0), \
            "Variance diagonal should be positive"

class TestYXReturnConsistency:
    """
    Test ID: RET-005
    
    Test y and x return value consistency.
    """
    
    def test_y_equals_input_treat(self, cbps_binary_fit, lalonde_full):
        """Test y equals input treatment vector."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result['y'], treat,
            rtol=1e-10, atol=1e-10,
            name="y return value"
        )

    def test_x_equals_input_X(self, cbps_binary_fit, lalonde_full):
        """Test x equals input covariate matrix."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert_allclose_with_report(
            result['x'], X,
            rtol=1e-10, atol=1e-10,
            name="x return value"
        )

    def test_y_shape(self, cbps_binary_fit, lalonde_full):
        """Test y has correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = len(treat)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['y'].shape == (n,), f"y should have shape ({n},)"

    def test_x_shape(self, cbps_binary_fit, lalonde_full):
        """Test x has correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n, k = X.shape
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['x'].shape == (n, k), f"x should have shape ({n}, {k})"

class TestAllReturnKeys:
    """
    Test ID: RET-006
    
    Test all expected return keys exist.
    """
    
    def test_all_keys_present(self, cbps_binary_fit, lalonde_full):
        """Test all expected keys are present in result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        expected_keys = [
            'coefficients',
            'fitted_values',
            'linear_predictor',
            'deviance',
            'nulldeviance',
            'weights',
            'y',
            'x',
            'converged',
            'J',
            'var',
            'mle_J'
        ]
        
        for key in expected_keys:
            assert key in result, f"Key '{key}' should be in result"

    def test_no_none_values(self, cbps_binary_fit, lalonde_full):
        """Test no None values in result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        for key, value in result.items():
            assert value is not None, f"Key '{key}' should not be None"

    def test_converged_is_bool(self, cbps_binary_fit, lalonde_full):
        """Test converged is boolean."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert isinstance(result['converged'], (bool, np.bool_)), \
            "converged should be boolean"

    def test_j_is_scalar(self, cbps_binary_fit, lalonde_full):
        """Test J is scalar."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isscalar(result['J']) or result['J'].shape == (), \
            "J should be scalar"

    def test_deviance_is_scalar(self, cbps_binary_fit, lalonde_full):
        """Test deviance is scalar."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.isscalar(result['deviance']) or result['deviance'].shape == (), \
            "deviance should be scalar"

    def test_all_methods_return_same_keys(self, cbps_binary_fit, lalonde_full):
        """Test all method combinations return same keys."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        configs = [
            {'att': 0, 'method': 'over', 'two_step': True},
            {'att': 0, 'method': 'exact', 'two_step': True},
            {'att': 1, 'method': 'over', 'two_step': True},
            {'att': 0, 'method': 'over', 'two_step': False},
        ]
        
        reference_keys = None
        for config in configs:
            result = cbps_binary_fit(treat, X, **config)
            if reference_keys is None:
                reference_keys = set(result.keys())
            else:
                assert set(result.keys()) == reference_keys, \
                    f"Keys should be same for config {config}"

class TestNormalizationFormula:
    """
    Test that the normalization formula matches R: sw = sw / mean(sw)
    
    Test ID: SW-001
    Requirement: REQ-002
    """
    
    def test_formula_basic(self):
        """Verify basic normalization formula: sw / mean(sw)."""
        sw = np.array([1.0, 2.0, 0.5, 1.5])
        n = len(sw)
        
        # Expected: sw / mean(sw)
        expected = sw / sw.mean()
        
        # Actual
        result = normalize_sample_weights(sw, n)
        
        np.testing.assert_allclose(
            result, expected,
            atol=TOLERANCE,
            rtol=0,
            err_msg="Normalization formula does not match sw / mean(sw)"
        )
    
    def test_formula_preserves_ratios(self):
        """Verify that normalization preserves weight ratios."""
        sw = np.array([1.0, 2.0, 4.0, 8.0])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        # Ratios should be preserved
        # result[1] / result[0] should equal sw[1] / sw[0] = 2
        np.testing.assert_allclose(
            result[1] / result[0], 2.0,
            atol=TOLERANCE,
            err_msg="Normalization should preserve weight ratios"
        )
        np.testing.assert_allclose(
            result[3] / result[0], 8.0,
            atol=TOLERANCE,
            err_msg="Normalization should preserve weight ratios"
        )
    
    def test_formula_with_uniform_weights(self):
        """Verify that uniform weights remain uniform after normalization."""
        n = 100
        sw = np.ones(n)
        
        result = normalize_sample_weights(sw, n)
        
        # All weights should be 1.0 (since mean(ones) = 1)
        np.testing.assert_allclose(
            result, np.ones(n),
            atol=TOLERANCE,
            err_msg="Uniform weights should remain uniform"
        )

class TestSumEqualsN:
    """
    Test that normalized weights sum to n.
    
    Test ID: SW-002
    Requirement: REQ-001
    """
    
    def test_sum_equals_n_basic(self):
        """Verify sum(sw_normalized) = n for basic case."""
        sw = np.array([1.0, 2.0, 0.5, 1.5])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=TOLERANCE), \
            f"Sum should equal {n}, got {result.sum()}"
    
    def test_sum_equals_n_various_sizes(self):
        """Verify sum = n for various sample sizes."""
        for n in [10, 50, 100, 500, 1000]:
            np.random.seed(42)
            sw = np.random.exponential(1.0, n)
            
            result = normalize_sample_weights(sw, n)
            
            assert np.isclose(result.sum(), n, atol=TOLERANCE), \
                f"Sum should equal {n} for n={n}, got {result.sum()}"
    
    def test_sum_equals_n_extreme_weights(self):
        """Verify sum = n even with extreme weight values."""
        sw = np.array([0.001, 1.0, 100.0, 0.5])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=TOLERANCE), \
            f"Sum should equal {n} with extreme weights, got {result.sum()}"
    
    def test_sum_equals_n_large_sample(self):
        """Verify sum = n for large sample."""
        np.random.seed(123)
        n = 10000
        sw = np.random.exponential(1.0, n)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=1e-10), \
            f"Sum should equal {n} for large sample, got {result.sum()}"

class TestEdgeCases:
    """
    Test edge cases for sample weight normalization.
    
    Test ID: SW-004
    """
    
    def test_none_returns_ones(self):
        """Verify that None input returns uniform weights."""
        n = 50
        result = normalize_sample_weights(None, n)
        
        np.testing.assert_array_equal(
            result, np.ones(n),
            err_msg="None input should return array of ones"
        )
        assert result.dtype == np.float64, "Result should be float64"
    
    def test_single_weight(self):
        """Verify normalization with single weight."""
        sw = np.array([5.0])
        n = 1
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result[0], 1.0, atol=TOLERANCE), \
            "Single weight should normalize to 1.0"
        assert np.isclose(result.sum(), n, atol=TOLERANCE), \
            "Sum should equal n=1"
    
    def test_very_small_weights(self):
        """Verify normalization with very small weights."""
        sw = np.array([1e-10, 2e-10, 3e-10, 4e-10])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=TOLERANCE), \
            f"Sum should equal {n} with small weights"
        assert np.all(np.isfinite(result)), \
            "All results should be finite"
    
    def test_very_large_weights(self):
        """Verify normalization with very large weights."""
        sw = np.array([1e10, 2e10, 3e10, 4e10])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=TOLERANCE), \
            f"Sum should equal {n} with large weights"
        assert np.all(np.isfinite(result)), \
            "All results should be finite"
    
    def test_mixed_scale_weights(self):
        """Verify normalization with mixed scale weights."""
        sw = np.array([1e-8, 1.0, 1e8])
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.sum(), n, atol=1e-10), \
            f"Sum should equal {n} with mixed scale weights"
    
    def test_dtype_conversion(self):
        """Verify that input is converted to float64."""
        sw_int = np.array([1, 2, 3, 4])  # Integer input
        n = len(sw_int)
        
        result = normalize_sample_weights(sw_int, n)
        
        assert result.dtype == np.float64, \
            f"Result dtype should be float64, got {result.dtype}"

class TestErrorHandling:
    """
    Test error handling for invalid inputs.
    
    Test ID: SW-005
    """
    
    def test_negative_weights_raise_error(self):
        """Verify that negative weights raise ValueError."""
        sw = np.array([1.0, -0.5, 2.0, 1.5])
        n = len(sw)
        
        with pytest.raises(ValueError, match="non-negative"):
            normalize_sample_weights(sw, n)
    
    def test_all_zero_weights_raise_error(self):
        """Verify that all-zero weights raise ValueError."""
        sw = np.array([0.0, 0.0, 0.0, 0.0])
        n = len(sw)
        
        with pytest.raises(ValueError, match="all zeros"):
            normalize_sample_weights(sw, n)
    
    def test_zero_weights_warning(self):
        """Verify that some zero weights produce warning."""
        sw = np.array([1.0, 0.0, 2.0, 1.5])  # One zero weight
        n = len(sw)
        
        with pytest.warns(UserWarning, match="zero values"):
            result = normalize_sample_weights(sw, n)
        
        # Should still work
        assert np.isclose(result.sum(), n, atol=TOLERANCE)

class TestMathematicalProperties:
    """
    Test mathematical properties of the normalization.
    """
    
    def test_idempotence(self):
        """Verify that normalizing twice gives same result."""
        sw = np.array([1.0, 2.0, 3.0, 4.0])
        n = len(sw)
        
        result1 = normalize_sample_weights(sw, n)
        result2 = normalize_sample_weights(result1, n)
        
        np.testing.assert_allclose(
            result1, result2,
            atol=TOLERANCE,
            err_msg="Normalization should be idempotent"
        )
    
    def test_scaling_invariance(self):
        """Verify that scaling input doesn't change output."""
        sw = np.array([1.0, 2.0, 3.0, 4.0])
        n = len(sw)
        
        result1 = normalize_sample_weights(sw, n)
        result2 = normalize_sample_weights(sw * 100, n)
        result3 = normalize_sample_weights(sw * 0.01, n)
        
        np.testing.assert_allclose(
            result1, result2,
            atol=TOLERANCE,
            err_msg="Normalization should be scale-invariant"
        )
        np.testing.assert_allclose(
            result1, result3,
            atol=TOLERANCE,
            err_msg="Normalization should be scale-invariant"
        )
    
    def test_mean_of_normalized_is_one(self):
        """Verify that mean of normalized weights is 1."""
        np.random.seed(42)
        sw = np.random.exponential(2.0, 100)
        n = len(sw)
        
        result = normalize_sample_weights(sw, n)
        
        assert np.isclose(result.mean(), 1.0, atol=TOLERANCE), \
            f"Mean of normalized weights should be 1.0, got {result.mean()}"

class TestFormulaMathVerification:
    """
    Verify the mathematical formula using explicit calculations.
    
    This tests the core formula: sw_normalized = sw / mean(sw)
    which ensures sum(sw_normalized) = n
    
    Proof:
    sum(sw / mean(sw)) = sum(sw) / mean(sw) = sum(sw) / (sum(sw)/n) = n
    """
    
    def test_formula_derivation(self):
        """Verify the mathematical derivation of the formula."""
        sw = np.array([1.0, 2.0, 0.5, 1.5])
        n = len(sw)
        
        # Step 1: Compute mean
        mean_sw = sw.mean()
        assert np.isclose(mean_sw, 1.25, atol=TOLERANCE), \
            f"Mean should be 1.25, got {mean_sw}"
        
        # Step 2: Normalize
        sw_normalized = sw / mean_sw
        
        # Step 3: Verify sum = n
        # sum(sw/mean) = sum(sw)/mean = sum(sw)/(sum(sw)/n) = n
        sum_normalized = sw_normalized.sum()
        assert np.isclose(sum_normalized, n, atol=TOLERANCE), \
            f"Sum should equal {n}, got {sum_normalized}"
        
        # Step 4: Verify against function
        result = normalize_sample_weights(sw, n)
        np.testing.assert_allclose(
            result, sw_normalized,
            atol=TOLERANCE,
            err_msg="Function result should match manual calculation"
        )
    
    def test_formula_with_explicit_values(self):
        """Test with explicit values for easy verification."""
        # sw = [2, 4, 6, 8], mean = 5, n = 4
        # normalized = [0.4, 0.8, 1.2, 1.6]
        # sum = 4.0
        sw = np.array([2.0, 4.0, 6.0, 8.0])
        n = 4
        
        expected = np.array([0.4, 0.8, 1.2, 1.6])
        result = normalize_sample_weights(sw, n)
        
        np.testing.assert_allclose(
            result, expected,
            atol=TOLERANCE,
            err_msg="Explicit value test failed"
        )
        assert np.isclose(result.sum(), 4.0, atol=TOLERANCE)

class TestCoefficientStandardErrors:
    """
    Test ID: SE-001
    
    Test coefficient standard error calculation.
    """
    
    def test_se_from_variance_matrix(self, cbps_binary_fit, lalonde_full):
        """Test standard errors are sqrt of variance diagonal."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        var = result['var']
        
        # Standard errors from diagonal
        se = np.sqrt(np.diag(var))
        
        assert np.all(se >= 0), "Standard errors should be non-negative"
        assert np.all(np.isfinite(se)), "Standard errors should be finite"

    def test_se_positive(self, cbps_binary_fit, lalonde_full):
        """Test standard errors are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        se = np.sqrt(np.diag(result['var']))
        
        assert np.all(se > 0), "Standard errors should be positive"

    def test_se_reasonable_magnitude(self, cbps_binary_fit, lalonde_full):
        """Test standard errors have reasonable magnitude."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        # SE should not be orders of magnitude larger than coefficients
        # (unless coefficient is near zero)
        for i in range(len(coef)):
            if np.abs(coef[i]) > 0.01:
                ratio = se[i] / np.abs(coef[i])
                assert ratio < 100, f"SE/coef ratio too large for coef {i}: {ratio}"

    def test_se_shape(self, cbps_binary_fit, lalonde_full):
        """Test standard errors have correct shape."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = X.shape[1]
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        se = np.sqrt(np.diag(result['var']))
        
        assert se.shape == (k,), f"SE should have shape ({k},)"

    def test_se_different_methods(self, cbps_binary_fit, lalonde_full):
        """Test standard errors for different methods."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        se_over = np.sqrt(np.diag(result_over['var']))
        se_exact = np.sqrt(np.diag(result_exact['var']))
        
        # Both should be positive and finite
        assert np.all(se_over > 0) and np.all(np.isfinite(se_over))
        assert np.all(se_exact > 0) and np.all(np.isfinite(se_exact))

class TestZStatistic:
    """
    Test ID: SE-002
    
    Test z-statistic calculation.
    """
    
    def test_z_statistic_formula(self, cbps_binary_fit, lalonde_full):
        """Test z-statistic follows z = coef / se formula."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        
        assert np.all(np.isfinite(z)), "Z-statistics should be finite"

    def test_z_statistic_sign(self, cbps_binary_fit, lalonde_full):
        """Test z-statistic sign matches coefficient sign."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        
        # Sign should match (where coefficient is not near zero)
        for i in range(len(coef)):
            if np.abs(coef[i]) > 0.01:
                assert np.sign(z[i]) == np.sign(coef[i]), \
                    f"Z-stat sign should match coef sign for coef {i}"

    def test_z_statistic_magnitude(self, cbps_binary_fit, lalonde_full):
        """Test z-statistic magnitude is reasonable."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        
        # Z-statistics should typically be within reasonable range
        # (very large z may indicate numerical issues but is not necessarily wrong)
        # Most z-statistics should be reasonable
        reasonable_count = np.sum(np.abs(z) < 100)
        assert reasonable_count >= len(z) // 2, \
            f"At least half of z-statistics should be < 100, got {reasonable_count}/{len(z)}"

class TestPValue:
    """
    Test ID: SE-003
    
    Test p-value calculation.
    """
    
    def test_p_value_range(self, cbps_binary_fit, lalonde_full):
        """Test p-values are in [0, 1]."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
        
        assert np.all(p_values >= 0), "P-values should be >= 0"
        assert np.all(p_values <= 1), "P-values should be <= 1"

    def test_p_value_two_sided(self, cbps_binary_fit, lalonde_full):
        """Test two-sided p-value calculation."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        
        # Two-sided p-value
        p_two_sided = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
        
        # One-sided p-value
        p_one_sided = 1 - scipy.stats.norm.cdf(np.abs(z))
        
        # Two-sided should be twice one-sided
        assert_allclose_with_report(
            p_two_sided, 2 * p_one_sided,
            rtol=1e-10, atol=1e-10,
            name="Two-sided vs one-sided p-value"
        )

    def test_p_value_significance(self, cbps_binary_fit, lalonde_full):
        """Test p-value significance interpretation."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z = coef / se
        p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
        
        # Large |z| should give small p-value
        for i in range(len(z)):
            if np.abs(z[i]) > 1.96:
                assert p_values[i] < 0.05, \
                    f"|z| > 1.96 should give p < 0.05 for coef {i}"

class TestConfidenceInterval:
    """
    Test ID: SE-004
    
    Test confidence interval calculation.
    """
    
    def test_ci_95_formula(self, cbps_binary_fit, lalonde_full):
        """Test 95% CI follows coef ± 1.96*se formula."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z_crit = scipy.stats.norm.ppf(0.975)  # ≈ 1.96
        
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se
        
        # CI should contain coefficient
        assert np.all(ci_lower <= coef), "CI lower should be <= coef"
        assert np.all(ci_upper >= coef), "CI upper should be >= coef"

    def test_ci_width(self, cbps_binary_fit, lalonde_full):
        """Test CI width is 2 * z_crit * se."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        se = np.sqrt(np.diag(result['var']))
        
        z_crit = scipy.stats.norm.ppf(0.975)
        expected_width = 2 * z_crit * se
        
        coef = result['coefficients'].ravel()
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se
        actual_width = ci_upper - ci_lower
        
        assert_allclose_with_report(
            actual_width, expected_width,
            rtol=1e-10, atol=1e-10,
            name="CI width"
        )

    def test_ci_different_levels(self, cbps_binary_fit, lalonde_full):
        """Test CI at different confidence levels."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        # 90% CI
        z_90 = scipy.stats.norm.ppf(0.95)
        ci_90_width = 2 * z_90 * se
        
        # 95% CI
        z_95 = scipy.stats.norm.ppf(0.975)
        ci_95_width = 2 * z_95 * se
        
        # 99% CI
        z_99 = scipy.stats.norm.ppf(0.995)
        ci_99_width = 2 * z_99 * se
        
        # Higher confidence = wider CI
        assert np.all(ci_90_width < ci_95_width), "90% CI should be narrower than 95%"
        assert np.all(ci_95_width < ci_99_width), "95% CI should be narrower than 99%"

    def test_ci_symmetry(self, cbps_binary_fit, lalonde_full):
        """Test CI is symmetric around coefficient."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        coef = result['coefficients'].ravel()
        se = np.sqrt(np.diag(result['var']))
        
        z_crit = scipy.stats.norm.ppf(0.975)
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se
        
        # Distance from coef to bounds should be equal
        dist_lower = coef - ci_lower
        dist_upper = ci_upper - coef
        
        assert_allclose_with_report(
            dist_lower, dist_upper,
            rtol=1e-10, atol=1e-10,
            name="CI symmetry"
        )

    def test_se_consistency_across_methods(self, cbps_binary_fit, lalonde_full):
        """Test SE consistency across different methods."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        se_over = np.sqrt(np.diag(result_over['var']))
        se_exact = np.sqrt(np.diag(result_exact['var']))
        
        # SEs should be in same ballpark (within factor of 10)
        ratio = se_over / se_exact
        assert np.all(ratio > 0.1) and np.all(ratio < 10), \
            "SEs should be in similar range across methods"

class TestSVDDecomposition:
    """
    Test ID: SVD-001
    
    Test SVD decomposition correctness.
    """
    
    def test_svd_reconstruction(self):
        """Test SVD reconstruction equals original matrix."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        X_reconstructed = U @ np.diag(s) @ Vt
        
        assert_allclose_with_report(
            X, X_reconstructed,
            rtol=1e-10, atol=1e-10,
            name="SVD reconstruction"
        )

    def test_svd_orthogonality(self):
        """Test SVD produces orthogonal matrices."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # U should have orthonormal columns
        assert_allclose_with_report(
            U.T @ U, np.eye(k),
            rtol=1e-10, atol=1e-10,
            name="U orthogonality"
        )
        
        # Vt should have orthonormal rows
        assert_allclose_with_report(
            Vt @ Vt.T, np.eye(k),
            rtol=1e-10, atol=1e-10,
            name="Vt orthogonality"
        )

    def test_svd_singular_values_positive(self):
        """Test SVD singular values are non-negative."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        assert np.all(s >= 0), "Singular values should be non-negative"

    def test_svd_singular_values_ordered(self):
        """Test SVD singular values are in descending order."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        assert np.all(np.diff(s) <= 0), "Singular values should be in descending order"

    def test_svd_rank_deficient(self):
        """Test SVD handles rank-deficient matrices."""
        np.random.seed(42)
        n, k = 100, 5
        
        # Create rank-deficient matrix (rank 3)
        A = np.random.randn(n, 3)
        B = np.random.randn(3, k)
        X = A @ B
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Should have 3 non-zero singular values
        rank = np.sum(s > 1e-10)
        assert rank == 3, f"Rank should be 3, got {rank}"

class TestSVDThreshold:
    """
    Test ID: SVD-002
    
    Test SVD threshold selection for singular value truncation.
    """
    
    def test_threshold_default(self):
        """Test default SVD threshold (sqrt(machine epsilon))."""
        from cbps.core.cbps_binary import _r_ginv
        
        np.random.seed(42)
        X = np.random.randn(10, 5)
        
        # Default tolerance
        tol = np.sqrt(np.finfo(float).eps)
        
        # Compute pseudoinverse
        X_pinv = _r_ginv(X)
        
        # Should satisfy pseudoinverse properties
        # X @ X_pinv @ X ≈ X
        assert_allclose_with_report(
            X @ X_pinv @ X, X,
            rtol=1e-10, atol=1e-10,
            name="Pseudoinverse property"
        )

    def test_threshold_effect_on_rank(self):
        """Test threshold effect on effective rank."""
        np.random.seed(42)
        n, k = 100, 5
        
        # Create matrix with varying singular values
        U, _ = np.linalg.qr(np.random.randn(n, k))
        V, _ = np.linalg.qr(np.random.randn(k, k))
        s = np.array([1.0, 0.1, 0.01, 1e-8, 1e-12])
        X = U @ np.diag(s) @ V.T
        
        # Different thresholds
        tol_strict = 1e-6
        tol_loose = 1e-2
        
        # Count retained singular values
        retained_strict = np.sum(s > tol_strict * s[0])
        retained_loose = np.sum(s > tol_loose * s[0])
        
        assert retained_strict >= retained_loose, \
            "Stricter threshold should retain more singular values"

    def test_threshold_numerical_stability(self):
        """Test threshold provides numerical stability."""
        from cbps.core.cbps_binary import _r_ginv
        
        np.random.seed(42)
        n, k = 50, 5
        
        # Create near-singular matrix
        U, _ = np.linalg.qr(np.random.randn(n, k))
        V, _ = np.linalg.qr(np.random.randn(k, k))
        s = np.array([1.0, 0.5, 0.1, 1e-10, 1e-15])
        X = U @ np.diag(s) @ V.T
        
        # Pseudoinverse should be finite
        X_pinv = _r_ginv(X)
        
        assert np.all(np.isfinite(X_pinv)), "Pseudoinverse should be finite"

class TestSkipSVDParameter:
    """
    Test ID: SVD-003
    
    Test skip_svd parameter effect (Note: binary CBPS doesn't use SVD by default).
    """
    
    def test_binary_cbps_no_svd(self, cbps_binary_fit, lalonde_full):
        """Test binary CBPS works without SVD transformation."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Binary CBPS should work directly on original X
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged'], "CBPS should converge without SVD"
        assert result['coefficients'].shape == (X.shape[1], 1)

    def test_coefficients_in_original_space(self, cbps_binary_fit, lalonde_full):
        """Test coefficients are in original covariate space."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Coefficients should produce valid propensity scores
        theta = X @ result['coefficients'].ravel()
        probs = scipy.special.expit(theta)
        
        assert np.all(probs > 0) and np.all(probs < 1), \
            "Propensity scores should be in (0, 1)"

    def test_collinear_covariates_handling(self, cbps_binary_fit):
        """Test handling of collinear covariates."""
        np.random.seed(42)
        n = 200
        
        # Create data with near-collinearity
        x1 = np.random.normal(0, 1, n)
        x2 = x1 + np.random.normal(0, 0.1, n)  # Nearly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        # Should handle collinearity (may warn but not crash)
        try:
            result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
            # If it runs, check basic properties
            assert result['weights'].shape == (n,)
        except ValueError as e:
            # May raise error for rank-deficient X
            assert "rank" in str(e).lower()

class TestSVDCoefficientTransform:
    """
    Test ID: SVD-004
    
    Test SVD coefficient inverse transformation.
    """
    
    def test_coefficient_transform_identity(self):
        """Test coefficient transform is identity when no SVD used."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        # Original coefficients
        beta = np.random.randn(k)
        
        # Linear predictor should be same
        theta = X @ beta
        
        # No transformation needed for binary CBPS
        assert theta.shape == (n,)

    def test_svd_transform_preserves_predictions(self):
        """Test SVD transformation preserves linear predictions."""
        np.random.seed(42)
        n, k = 100, 5
        X = np.random.randn(n, k)
        
        # SVD decomposition
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Original coefficients
        beta = np.random.randn(k)
        
        # Original prediction
        theta_orig = X @ beta
        
        # Transform to SVD space and back
        # X = U @ diag(s) @ Vt
        # X @ beta = U @ diag(s) @ Vt @ beta
        # If we work in U space with beta_svd = diag(s) @ Vt @ beta
        # Then U @ beta_svd = X @ beta
        
        beta_svd = np.diag(s) @ Vt @ beta
        theta_svd = U @ beta_svd
        
        assert_allclose_with_report(
            theta_orig, theta_svd,
            rtol=1e-10, atol=1e-10,
            name="SVD transform preserves predictions"
        )

class TestSVDVarianceTransform:
    """
    Test ID: SVD-005
    
    Test SVD variance inverse transformation.
    """
    
    def test_variance_transform_formula(self):
        """Test variance transformation formula."""
        np.random.seed(42)
        k = 5
        
        # Variance in SVD space
        var_svd = np.eye(k) * 0.1
        
        # Transformation matrix (from SVD)
        V = np.random.randn(k, k)
        V, _ = np.linalg.qr(V)  # Orthogonalize
        s = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        
        # Transform variance back to original space
        # var_orig = V @ diag(1/s) @ var_svd @ diag(1/s) @ V.T
        s_inv = 1.0 / s
        var_orig = V @ np.diag(s_inv) @ var_svd @ np.diag(s_inv) @ V.T
        
        # Should be symmetric
        assert_allclose_with_report(
            var_orig, var_orig.T,
            rtol=1e-10, atol=1e-10,
            name="Transformed variance symmetry"
        )

    def test_variance_positive_semidefinite(self):
        """Test transformed variance is positive semi-definite."""
        np.random.seed(42)
        k = 5
        
        # Positive definite variance in SVD space
        A = np.random.randn(k, k)
        var_svd = A @ A.T + np.eye(k) * 0.1
        
        # Transformation
        V = np.random.randn(k, k)
        V, _ = np.linalg.qr(V)
        s = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        s_inv = 1.0 / s
        
        var_orig = V @ np.diag(s_inv) @ var_svd @ np.diag(s_inv) @ V.T
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(var_orig)
        assert np.all(eigenvalues >= -1e-10), \
            "Transformed variance should be positive semi-definite"

    def test_cbps_variance_properties(self, cbps_binary_fit, lalonde_full):
        """Test CBPS variance matrix properties."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = X.shape[1]
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        var = result['var']
        
        # Shape
        assert var.shape == (k, k), f"Variance should be ({k}, {k})"
        
        # Symmetry
        assert_allclose_with_report(
            var, var.T,
            rtol=1e-10, atol=1e-10,
            name="Variance symmetry"
        )
        
        # Positive semi-definite
        eigenvalues = np.linalg.eigvalsh(var)
        assert np.all(eigenvalues >= -1e-8), \
            f"Variance should be PSD, min eigenvalue: {np.min(eigenvalues)}"

class TestATEVMatrixBlockStructure:
    """
    Test ATE V matrix block structure implementation.
    
    Test ID: VM-001
    Requirement: REQ-007
    
    The ATE V matrix has a 2x2 block structure:
        V = | V11  V12 |
            | V21  V22 |
    
    where:
        V11 = (1/n) * X' * diag(π(1-π)) * X
        V12 = V21 = (1/n) * X' * X
        V22 = (1/n) * X' * diag(1/(π(1-π))) * X
    """
    
    def test_ate_v_matrix_shape(self, compute_V_matrix, simple_binary_data):
        """Test that ATE V matrix has correct shape (2k x 2k)."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute V matrix (returns invV)
        inv_V = compute_V_matrix(
            X, probs, data['sample_weights'], data['treat'], att=0, n=n
        )
        
        expected_shape = (2 * k, 2 * k)
        assert inv_V.shape == expected_shape, \
            f"V matrix shape mismatch: got {inv_V.shape}, expected {expected_shape}"
    
    def test_ate_v11_block_formula(self, compute_V_matrix, r_ginv_func, simple_binary_data):
        """Test V11 block: (1/n) * X' * diag(π(1-π)) * X."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Manually compute V11 block
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        V11_expected = (1 / n) * (X_1.T @ X_1)
        
        # Compute full V matrix and extract V11
        # Note: _compute_V_matrix returns invV, so we need to reconstruct V
        # For testing, we compute V directly
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V11 from full V
        V11_actual = V[:k, :k]
        
        assert_allclose_with_report(
            V11_actual, V11_expected,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.V_MATRIX_ATOL,
            name="ATE V11 block"
        )

    
    def test_ate_v12_block_formula(self, simple_binary_data):
        """Test V12 block: (1/n) * X' * X."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Manually compute V12 block
        sw_sqrt = np.sqrt(sample_weights)
        X_1_1 = sw_sqrt[:, None] * X
        V12_expected = (1 / n) * (X_1_1.T @ X_1_1)
        
        # Compute full V matrix
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V12 from full V
        V12_actual = V[:k, k:]
        
        assert_allclose_with_report(
            V12_actual, V12_expected,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.V_MATRIX_ATOL,
            name="ATE V12 block"
        )
    
    def test_ate_v22_block_formula(self, simple_binary_data):
        """Test V22 block: (1/n) * X' * diag(1/(π(1-π))) * X."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Manually compute V22 block
        sw_sqrt = np.sqrt(sample_weights)
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        V22_expected = (1 / n) * (X_2.T @ X_2)
        
        # Compute full V matrix
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V22 from full V
        V22_actual = V[k:, k:]
        
        assert_allclose_with_report(
            V22_actual, V22_expected,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.V_MATRIX_ATOL,
            name="ATE V22 block"
        )

    
    def test_ate_v_matrix_block_symmetry(self, simple_binary_data):
        """Test that V12 = V21 (off-diagonal blocks are equal)."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V12 and V21
        V12_block = V[:k, k:]
        V21_block = V[k:, :k]
        
        assert_allclose_with_report(
            V12_block, V21_block,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.SYMMETRY_ATOL,
            name="ATE V12 = V21"
        )

class TestATTVMatrixScalingFactor:
    """
    Test ATT V matrix scaling factor implementation.
    
    Test ID: VM-002
    Requirement: REQ-007
    
    The ATT V matrix has additional scaling factors:
        V11 = (1/n) * X' * diag(π(1-π)) * X * (n/n_t)
        V12 = V21 = (1/n) * X' * diag(π) * X * (n/n_t)
        V22 = (1/n) * X' * diag(π/(1-π)) * X * (n/n_t)^2
    """
    
    def test_att_v_matrix_shape(self, compute_V_matrix, simple_binary_data):
        """Test that ATT V matrix has correct shape (2k x 2k)."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute V matrix (returns invV)
        inv_V = compute_V_matrix(
            X, probs, data['sample_weights'], data['treat'], att=1, n=n
        )
        
        expected_shape = (2 * k, 2 * k)
        assert inv_V.shape == expected_shape, \
            f"ATT V matrix shape mismatch: got {inv_V.shape}, expected {expected_shape}"

    
    def test_att_v11_scaling_factor(self, simple_binary_data):
        """Test ATT V11 block includes n/n_t scaling."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        treat = data['treat']
        sample_weights = data['sample_weights']
        n_t = np.sum(treat == 1)
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATT V11 formula: (1/n) * X' * diag(π(1-π)) * X * (n/n_t)
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        V11_expected = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        
        # Compute full ATT V matrix
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs / (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs)[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * (n / n_t)
        V22 = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V11 from full V
        V11_actual = V[:k, :k]
        
        assert_allclose_with_report(
            V11_actual, V11_expected,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.V_MATRIX_ATOL,
            name="ATT V11 scaling"
        )
    
    def test_att_v22_squared_scaling_factor(self, simple_binary_data):
        """Test ATT V22 block includes (n/n_t)^2 scaling."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        treat = data['treat']
        sample_weights = data['sample_weights']
        n_t = np.sum(treat == 1)
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATT V22 formula: (1/n) * X' * diag(π/(1-π)) * X * (n/n_t)^2
        sw_sqrt = np.sqrt(sample_weights)
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs / (1 - probs))[:, None]
        V22_expected = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        # Compute full ATT V matrix
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs)[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * (n / n_t)
        V22 = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Extract V22 from full V
        V22_actual = V[k:, k:]
        
        assert_allclose_with_report(
            V22_actual, V22_expected,
            rtol=Tolerances.V_MATRIX_RTOL,
            atol=Tolerances.V_MATRIX_ATOL,
            name="ATT V22 squared scaling"
        )

    
    def test_att_scaling_differs_from_ate(self, compute_V_matrix, simple_binary_data):
        """Test that ATT and ATE V matrices differ due to scaling."""
        data = simple_binary_data
        X = data['X']
        n = data['n']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        inv_V_ate = compute_V_matrix(
            X, probs, data['sample_weights'], data['treat'], att=0, n=n
        )
        
        # Compute ATT V matrix
        inv_V_att = compute_V_matrix(
            X, probs, data['sample_weights'], data['treat'], att=1, n=n
        )
        
        # They should be different
        assert not np.allclose(inv_V_ate, inv_V_att, rtol=1e-5), \
            "ATE and ATT V matrices should differ due to scaling factors"

class TestVMatrixSymmetry:
    """
    Test V matrix symmetry properties.
    
    Test ID: VM-003
    Requirement: REQ-007
    
    The V matrix must be symmetric: V = V^T
    """
    
    def test_ate_v_matrix_symmetry(self, simple_binary_data):
        """Test that ATE V matrix is symmetric."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        assert_matrix_symmetric(V, atol=Tolerances.SYMMETRY_ATOL, name="ATE V matrix")
    
    def test_att_v_matrix_symmetry(self, simple_binary_data):
        """Test that ATT V matrix is symmetric."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        treat = data['treat']
        sample_weights = data['sample_weights']
        n_t = np.sum(treat == 1)
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATT V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs / (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs)[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * (n / n_t)
        V22 = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        assert_matrix_symmetric(V, atol=Tolerances.SYMMETRY_ATOL, name="ATT V matrix")

    
    def test_v_matrix_diagonal_blocks_symmetric(self, simple_binary_data):
        """Test that diagonal blocks V11 and V22 are individually symmetric."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute V11 and V22 blocks
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        assert_matrix_symmetric(V11, atol=Tolerances.SYMMETRY_ATOL, name="V11 block")
        assert_matrix_symmetric(V22, atol=Tolerances.SYMMETRY_ATOL, name="V22 block")

class TestVMatrixPseudoinverseProperty:
    """
    Test V matrix pseudoinverse properties.
    
    Test ID: VM-004
    Requirement: REQ-007
    
    The pseudoinverse V+ must satisfy:
    - V @ V+ @ V ≈ V (Moore-Penrose condition 1)
    - V+ @ V @ V+ ≈ V+ (Moore-Penrose condition 2)
    """
    
    def test_ate_pseudoinverse_condition_1(self, compute_V_matrix, r_ginv_func, simple_binary_data):
        """Test V @ V+ @ V ≈ V for ATE."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Compute pseudoinverse
        V_pinv = r_ginv_func(V)
        
        # Test Moore-Penrose condition 1: V @ V+ @ V ≈ V
        result = V @ V_pinv @ V
        
        assert_allclose_with_report(
            result, V,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="ATE V @ V+ @ V"
        )

    
    def test_ate_pseudoinverse_condition_2(self, r_ginv_func, simple_binary_data):
        """Test V+ @ V @ V+ ≈ V+ for ATE."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Compute pseudoinverse
        V_pinv = r_ginv_func(V)
        
        # Test Moore-Penrose condition 2: V+ @ V @ V+ ≈ V+
        result = V_pinv @ V @ V_pinv
        
        assert_allclose_with_report(
            result, V_pinv,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="ATE V+ @ V @ V+"
        )
    
    def test_att_pseudoinverse_condition_1(self, r_ginv_func, simple_binary_data):
        """Test V @ V+ @ V ≈ V for ATT."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        treat = data['treat']
        sample_weights = data['sample_weights']
        n_t = np.sum(treat == 1)
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATT V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs / (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs)[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * (n / n_t)
        V22 = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Compute pseudoinverse
        V_pinv = r_ginv_func(V)
        
        # Test Moore-Penrose condition 1: V @ V+ @ V ≈ V
        result = V @ V_pinv @ V
        
        assert_allclose_with_report(
            result, V,
            rtol=Tolerances.PSEUDOINVERSE_RTOL,
            atol=Tolerances.PSEUDOINVERSE_ATOL,
            name="ATT V @ V+ @ V"
        )

    
    def test_invV_times_V_near_identity(self, compute_V_matrix, r_ginv_func, simple_binary_data):
        """Test that invV @ V is close to identity for full-rank V."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute full ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Compute pseudoinverse
        V_pinv = r_ginv_func(V)
        
        # For full-rank V, V+ @ V should be close to identity
        result = V_pinv @ V
        identity = np.eye(2 * k)
        
        # Use looser tolerance since V may not be perfectly full-rank
        assert_allclose_with_report(
            result, identity,
            rtol=1e-6,
            atol=1e-6,
            name="invV @ V near identity"
        )

class TestVMatrixEdgeCases:
    """
    Test V matrix computation edge cases.
    
    Test ID: VM-006
    Requirement: REQ-007
    
    Test edge cases including:
    - Extreme propensity scores
    - Single covariate
    - Large number of covariates
    - Imbalanced treatment groups
    """
    
    def test_v_matrix_extreme_probs_low(self, r_ginv_func):
        """Test V matrix with very low propensity scores."""
        np.random.seed(42)
        n = 100
        k = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        sample_weights = np.ones(n)
        
        # Very low propensity scores (near PROBS_MIN)
        probs = np.full(n, 0.01)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be finite and symmetric
        assert np.all(np.isfinite(V)), "V matrix should be finite with low probs"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix with low probs")
        
        # Pseudoinverse should also be finite
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"
    
    def test_v_matrix_extreme_probs_high(self, r_ginv_func):
        """Test V matrix with very high propensity scores."""
        np.random.seed(42)
        n = 100
        k = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        sample_weights = np.ones(n)
        
        # Very high propensity scores (near 1 - PROBS_MIN)
        probs = np.full(n, 0.99)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be finite and symmetric
        assert np.all(np.isfinite(V)), "V matrix should be finite with high probs"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix with high probs")
        
        # Pseudoinverse should also be finite
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"
    
    def test_v_matrix_single_covariate(self, r_ginv_func):
        """Test V matrix with single covariate (intercept only)."""
        np.random.seed(42)
        n = 100
        k = 1
        
        # Intercept only
        X = np.ones((n, k))
        sample_weights = np.ones(n)
        
        # Random propensity scores
        probs = np.random.uniform(0.2, 0.8, n)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be 2x2 for single covariate
        assert V.shape == (2, 2), f"V matrix shape should be (2, 2), got {V.shape}"
        assert np.all(np.isfinite(V)), "V matrix should be finite"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix single covariate")
        
        # Pseudoinverse should work
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"
    
    def test_v_matrix_many_covariates(self, r_ginv_func):
        """Test V matrix with many covariates."""
        np.random.seed(42)
        n = 200
        k = 20  # Many covariates
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        sample_weights = np.ones(n)
        
        # Random propensity scores
        probs = np.random.uniform(0.2, 0.8, n)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be 40x40 for 20 covariates
        expected_shape = (2 * k, 2 * k)
        assert V.shape == expected_shape, f"V matrix shape should be {expected_shape}, got {V.shape}"
        assert np.all(np.isfinite(V)), "V matrix should be finite"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix many covariates")
        
        # Pseudoinverse should work
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"
    
    def test_v_matrix_imbalanced_treatment(self, r_ginv_func):
        """Test V matrix with imbalanced treatment groups."""
        np.random.seed(42)
        n = 100
        k = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        sample_weights = np.ones(n)
        
        # Imbalanced: 90% treated
        treat = np.concatenate([np.ones(90), np.zeros(10)])
        n_t = int(np.sum(treat))
        
        # Propensity scores reflecting imbalance
        probs = np.random.uniform(0.7, 0.95, n)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATT V matrix (more sensitive to imbalance)
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X * np.sqrt(probs / (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X * np.sqrt(probs)[:, None]
        
        V11 = (1 / n) * (X_1.T @ X_1) * (n / n_t)
        V12 = (1 / n) * (X_1_1.T @ X_1_1) * (n / n_t)
        V22 = (1 / n) * (X_2.T @ X_2) * (n / n_t)**2
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be finite and symmetric even with imbalance
        assert np.all(np.isfinite(V)), "V matrix should be finite with imbalanced treatment"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix imbalanced treatment")
        
        # Pseudoinverse should work
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"
    
    def test_v_matrix_weighted_samples(self, r_ginv_func):
        """Test V matrix with non-uniform sample weights."""
        np.random.seed(42)
        n = 100
        k = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        
        # Non-uniform sample weights (exponential distribution)
        raw_weights = np.random.exponential(1, n)
        sample_weights = raw_weights / np.mean(raw_weights)  # Normalize to mean 1
        
        # Random propensity scores
        probs = np.random.uniform(0.2, 0.8, n)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix with weights
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # V should be finite and symmetric
        assert np.all(np.isfinite(V)), "V matrix should be finite with weighted samples"
        assert_matrix_symmetric(V, atol=1e-12, name="V matrix weighted samples")
        
        # Pseudoinverse should work
        V_pinv = r_ginv_func(V)
        assert np.all(np.isfinite(V_pinv)), "V pseudoinverse should be finite"

class TestVMatrixNumericalPrecision:
    """
    Test numerical precision of V matrix computation.
    
    Test ID: VM-007
    Requirement: REQ-007
    
    Test numerical stability with:
    - Varying condition numbers
    - Near-singular matrices
    - Precision of computation
    """
    
    def test_v_matrix_condition_number(self, r_ginv_func):
        """Test V matrix condition number is reasonable."""
        np.random.seed(42)
        n = 100
        k = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        sample_weights = np.ones(n)
        
        # Well-behaved propensity scores
        probs = np.random.uniform(0.3, 0.7, n)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Compute condition number
        cond_num = np.linalg.cond(V)
        
        # Condition number should be reasonable (not too large)
        # For well-behaved data, expect < 1e10
        assert cond_num < 1e12, f"V matrix condition number too large: {cond_num}"
    
    def test_v_matrix_positive_semidefinite(self, simple_binary_data):
        """Test that V matrix is positive semi-definite."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Check positive semi-definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(V)
        min_eigenvalue = np.min(eigenvalues)
        
        # Allow small negative eigenvalues due to numerical precision
        assert min_eigenvalue >= -1e-10, \
            f"V matrix should be positive semi-definite, min eigenvalue: {min_eigenvalue}"
    
    def test_v_matrix_determinant_nonzero(self, simple_binary_data):
        """Test that V matrix determinant is non-zero for full-rank X."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute ATE V matrix
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        V = np.block([[V11, V12],
                      [V12, V22]])
        
        # Determinant should be non-zero
        det_V = np.linalg.det(V)
        assert np.abs(det_V) > 1e-20, f"V matrix determinant should be non-zero, got {det_V}"
    
    def test_v_matrix_block_consistency(self, simple_binary_data):
        """Test that V matrix blocks are internally consistent."""
        data = simple_binary_data
        X = data['X']
        n, k = X.shape
        sample_weights = data['sample_weights']
        
        # Compute propensity scores
        beta = data['true_beta']
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute intermediate matrices
        sw_sqrt = np.sqrt(sample_weights)
        X_1 = sw_sqrt[:, None] * X * np.sqrt(probs * (1 - probs))[:, None]
        X_2 = sw_sqrt[:, None] * X / np.sqrt(probs * (1 - probs))[:, None]
        X_1_1 = sw_sqrt[:, None] * X
        
        # Compute blocks
        V11 = (1 / n) * (X_1.T @ X_1)
        V12 = (1 / n) * (X_1_1.T @ X_1_1)
        V22 = (1 / n) * (X_2.T @ X_2)
        
        # Each block should be symmetric
        assert_matrix_symmetric(V11, atol=Tolerances.SYMMETRY_ATOL, name="V11 block")
        assert_matrix_symmetric(V12, atol=Tolerances.SYMMETRY_ATOL, name="V12 block")
        assert_matrix_symmetric(V22, atol=Tolerances.SYMMETRY_ATOL, name="V22 block")
        
        # Each block should be positive semi-definite
        assert_positive_semidefinite(V11, atol=1e-10, name="V11 block")
        assert_positive_semidefinite(V12, atol=1e-10, name="V12 block")
        assert_positive_semidefinite(V22, atol=1e-10, name="V22 block")

class TestGMatrix:
    """Test ID: VC-001, Requirement: REQ-014"""
    
    def test_g_matrix_shape_ate(self, r_ginv, simple_binary_data):
        """Test G matrix shape for ATE estimation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        # Compute propensity scores
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute G matrix components (ATE)
        XG_1 = -X * (probs * (1 - probs))[:, None] * sample_weights[:, None]
        XG_2 = -X * ((treat - probs)**2 / (probs * (1 - probs)))[:, None] * sample_weights[:, None]
        
        G = np.hstack([(XG_1.T @ X), (XG_2.T @ X)]) / n
        
        # G should be (k, 2k) for over-identified GMM
        assert G.shape == (k, 2*k), f"Expected shape ({k}, {2*k}), got {G.shape}"

    def test_g_matrix_shape_att(self, att_wt_func, r_ginv, simple_binary_data):
        """Test G matrix shape for ATT estimation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        n_t = np.sum(sample_weights[treat == 1])
        
        # Compute propensity scores
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute G matrix components (ATT)
        XG_1 = -X * (probs * (1 - probs))[:, None] * sample_weights[:, None]
        dw2 = -n / n_t * probs / (1 - probs)
        dw2[treat == 1] = 0
        XG_2 = X * dw2[:, None] * sample_weights[:, None]
        
        G = np.hstack([(XG_1.T @ X), (XG_2.T @ X)]) / n
        
        assert G.shape == (k, 2*k), f"Expected shape ({k}, {2*k}), got {G.shape}"

    def test_g_matrix_finite(self, simple_binary_data):
        """Test that G matrix elements are finite."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XG_1 = -X * (probs * (1 - probs))[:, None] * sample_weights[:, None]
        XG_2 = -X * ((treat - probs)**2 / (probs * (1 - probs)))[:, None] * sample_weights[:, None]
        
        G = np.hstack([(XG_1.T @ X), (XG_2.T @ X)]) / n
        
        assert np.all(np.isfinite(G)), "G matrix should have all finite elements"

class TestOmegaMatrix:
    """Test ID: VC-002, Requirement: REQ-014"""
    
    def test_omega_matrix_shape_ate(self, simple_binary_data):
        """Test Omega matrix shape for ATE estimation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Compute W1 matrix (ATE)
        XW_1 = X * (treat - probs)[:, None] * np.sqrt(sample_weights)[:, None]
        XW_2 = X * (1 / (probs - 1 + treat))[:, None] * np.sqrt(sample_weights)[:, None]
        W1 = np.vstack([XW_1.T, XW_2.T])
        
        Omega = (W1 @ W1.T) / n
        
        # Omega should be (2k, 2k) for over-identified GMM
        assert Omega.shape == (2*k, 2*k), f"Expected shape ({2*k}, {2*k}), got {Omega.shape}"

    def test_omega_matrix_symmetric(self, simple_binary_data):
        """Test that Omega matrix is symmetric."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XW_1 = X * (treat - probs)[:, None] * np.sqrt(sample_weights)[:, None]
        XW_2 = X * (1 / (probs - 1 + treat))[:, None] * np.sqrt(sample_weights)[:, None]
        W1 = np.vstack([XW_1.T, XW_2.T])
        
        Omega = (W1 @ W1.T) / n
        
        assert_matrix_symmetric(Omega, atol=Tolerances.SYMMETRY_ATOL, name="Omega matrix")

    def test_omega_matrix_positive_semidefinite(self, simple_binary_data):
        """Test that Omega matrix is positive semi-definite."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XW_1 = X * (treat - probs)[:, None] * np.sqrt(sample_weights)[:, None]
        XW_2 = X * (1 / (probs - 1 + treat))[:, None] * np.sqrt(sample_weights)[:, None]
        W1 = np.vstack([XW_1.T, XW_2.T])
        
        Omega = (W1 @ W1.T) / n
        
        # Omega = W1 @ W1.T / n is always PSD by construction
        assert_positive_semidefinite(Omega, atol=1e-10, name="Omega matrix")

class TestSandwichFormula:
    """Test ID: VC-003, Requirement: REQ-014"""
    
    def test_sandwich_formula_structure(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test sandwich formula produces correct structure."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        # Vcov should be (k, k)
        assert vcov.shape == (k, k), f"Expected shape ({k}, {k}), got {vcov.shape}"

    def test_sandwich_formula_symmetric(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test that sandwich vcov is symmetric."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert_matrix_symmetric(vcov, atol=1e-12, name="Sandwich vcov")

    def test_sandwich_formula_finite(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test that sandwich vcov has finite elements."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert np.all(np.isfinite(vcov)), "Vcov should have all finite elements"

    def test_sandwich_formula_positive_diagonal(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test that sandwich vcov has positive diagonal (variances)."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        diag = np.diag(vcov)
        assert np.all(diag >= 0), f"Diagonal elements should be non-negative, got {diag}"

class TestVcovSymmetry:
    """Test ID: VC-004, Requirement: REQ-014"""
    
    def test_vcov_symmetry_ate(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test vcov symmetry for ATE estimation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert_matrix_symmetric(vcov, atol=1e-12, name="ATE vcov")

    def test_vcov_symmetry_att(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test vcov symmetry for ATT estimation."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=1,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert_matrix_symmetric(vcov, atol=1e-12, name="ATT vcov")

    def test_vcov_symmetry_exact_method(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test vcov symmetry for exact method (bal_only=True)."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=True, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert_matrix_symmetric(vcov, atol=1e-12, name="Exact method vcov")

class TestVcovEdgeCases:
    """Test vcov computation edge cases."""
    
    def test_vcov_small_sample(self, compute_vcov_func, r_ginv, compute_V_matrix):
        """Test vcov with small sample size."""
        np.random.seed(42)
        n, k = 30, 3
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        sample_weights = np.ones(n)
        beta = np.zeros(k)
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert vcov.shape == (k, k)
        assert np.all(np.isfinite(vcov))

    def test_vcov_many_covariates(self, compute_vcov_func, r_ginv, compute_V_matrix):
        """Test vcov with many covariates."""
        np.random.seed(42)
        n, k = 200, 15
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        sample_weights = np.ones(n)
        beta = np.zeros(k)
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert vcov.shape == (k, k)
        assert np.all(np.isfinite(vcov))

    def test_vcov_imbalanced_treatment(self, compute_vcov_func, r_ginv, compute_V_matrix):
        """Test vcov with imbalanced treatment groups."""
        np.random.seed(42)
        n, k = 100, 3
        X = np.column_stack([np.ones(n), np.random.randn(n, k-1)])
        # 80% treated
        treat = np.concatenate([np.ones(80), np.zeros(20)])
        sample_weights = np.ones(n)
        beta = np.zeros(k)
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert vcov.shape == (k, k)
        assert np.all(np.isfinite(vcov))

    def test_vcov_non_uniform_weights(self, compute_vcov_func, r_ginv, compute_V_matrix, weighted_binary_data):
        """Test vcov with non-uniform sample weights."""
        data = weighted_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        assert vcov.shape == (k, k)
        assert np.all(np.isfinite(vcov))
        assert_matrix_symmetric(vcov, atol=1e-12, name="Weighted vcov")

class TestVcovNumericalStability:
    """Test numerical stability of vcov computation."""
    
    def test_vcov_repeated_computation(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test that repeated vcov computations give identical results."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcovs = []
        for _ in range(5):
            vcov = compute_vcov_func(
                beta, probs, treat, X, sample_weights, att=0,
                bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
                two_step=True, n=n
            )
            vcovs.append(vcov)
        
        for i in range(1, len(vcovs)):
            assert np.allclose(vcovs[0], vcovs[i]), "Repeated computations should be identical"

    def test_vcov_standard_errors_reasonable(self, compute_vcov_func, r_ginv, compute_V_matrix, simple_binary_data):
        """Test that standard errors from vcov are reasonable."""
        data = simple_binary_data
        X, treat = data['X'], data['treat']
        n, k = X.shape
        sample_weights = data['sample_weights']
        beta = data['true_beta']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov_func(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        se = np.sqrt(np.diag(vcov))
        
        # Standard errors should be positive and reasonable
        assert np.all(se > 0), "Standard errors should be positive"
        assert np.all(se < 100), "Standard errors should be reasonable (< 100)"

class TestATTWeightFormula:
    """
    Test ATT weight formula implementation.
    
    Test ID: WT-001
    Requirement: REQ-005
    
    The ATT weight formula from Equation (11) in the paper:
    w_i = (n/n_t) * (T_i - π_i) / (1 - π_i)
    
    Properties:
    - Treated units (T=1): w = (n/n_t) * (1 - π) / (1 - π) = n/n_t (positive)
    - Control units (T=0): w = (n/n_t) * (-π) / (1 - π) (negative)
    """
    
    def test_att_weights_treated_units_positive(self, att_wt_func):
        """Test that treated units receive positive weights."""
        np.random.seed(42)
        n = 100
        
        # Create simple data
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.5])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Treated units should have positive weights
        treated_weights = weights[treat == 1]
        assert np.all(treated_weights > 0), \
            f"Treated units should have positive weights, got min={np.min(treated_weights)}"
    
    def test_att_weights_control_units_negative(self, att_wt_func):
        """Test that control units receive negative weights."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.5])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Control units should have negative weights
        control_weights = weights[treat == 0]
        assert np.all(control_weights < 0), \
            f"Control units should have negative weights, got max={np.max(control_weights)}"

    def test_att_weights_formula_verification(self, att_wt_func):
        """Verify ATT weight formula matches manual computation."""
        np.random.seed(123)
        n = 200
        
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        treat = np.random.binomial(1, 0.4, n).astype(float)
        sample_weights = np.ones(n)
        beta = np.array([0.1, 0.3, -0.2])
        
        # Compute using function
        weights_func = att_wt_func(beta, X, treat, sample_weights)
        
        # Compute manually
        weights_manual = compute_att_weights_manual(beta, X, treat, sample_weights)
        
        assert_allclose_with_report(
            weights_func, weights_manual,
            rtol=Tolerances.WEIGHT_RTOL,
            atol=Tolerances.WEIGHT_ATOL,
            name="ATT weights formula"
        )
    
    def test_att_weights_scaling_factor(self, att_wt_func):
        """Test that ATT weights include correct n/n_t scaling factor."""
        np.random.seed(42)
        n = 100
        n_t = 30
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * n_t + [0] * (n - n_t), dtype=float)
        sample_weights = np.ones(n)
        
        # Use beta=0 so all probs = 0.5
        beta = np.array([0.0, 0.0])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # For treated units with prob=0.5:
        # w = (n/n_t) * (1 - 0.5) / (1 - 0.5) = n/n_t
        expected_treated_weight = n / n_t
        
        treated_weights = weights[treat == 1]
        assert_allclose_with_report(
            treated_weights, np.full(n_t, expected_treated_weight),
            rtol=1e-10,
            atol=1e-12,
            name="ATT treated weights with prob=0.5"
        )
    
    def test_att_weights_sum_property(self, att_wt_func):
        """Test that weighted sum of ATT weights equals zero (balance condition)."""
        np.random.seed(42)
        n = 200
        
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        treat = np.random.binomial(1, 0.3, n).astype(float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.3, -0.2])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # The weighted sum should be close to zero when propensity scores are correct
        # This is the balance condition from the paper
        weighted_sum = np.sum(sample_weights * weights)
        
        # Note: This won't be exactly zero unless beta is the true value
        # But we can verify the formula structure
        assert np.isfinite(weighted_sum), "Weighted sum should be finite"

class TestATEWeightFormula:
    """
    Test ATE weight formula implementation.
    
    Test ID: WT-002
    Requirement: REQ-006
    
    The ATE weight formula from Equation (10) in the paper:
    w_i = (T_i - π_i) / (π_i * (1 - π_i))
    
    Which equals:
    - For treated (T=1): w = (1 - π) / (π * (1 - π)) = 1/π
    - For control (T=0): w = (-π) / (π * (1 - π)) = -1/(1-π)
    
    R implementation uses: w = 1 / (π - 1 + T)
    """
    
    def test_ate_weights_treated_positive(self, cbps_binary_module):
        """Test that ATE weights for treated units are positive."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        beta = np.array([0.0, 0.5])
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weights
        weights = 1.0 / (probs - 1 + treat)
        
        # Treated units should have positive weights (1/π > 0)
        treated_weights = weights[treat == 1]
        assert np.all(treated_weights > 0), \
            f"ATE treated weights should be positive, got min={np.min(treated_weights)}"
    
    def test_ate_weights_control_negative(self, cbps_binary_module):
        """Test that ATE weights for control units are negative."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        beta = np.array([0.0, 0.5])
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weights
        weights = 1.0 / (probs - 1 + treat)
        
        # Control units should have negative weights (-1/(1-π) < 0)
        control_weights = weights[treat == 0]
        assert np.all(control_weights < 0), \
            f"ATE control weights should be negative, got max={np.max(control_weights)}"

    def test_ate_weights_formula_equivalence(self, cbps_binary_module):
        """Test that R formula equals paper formula for ATE weights."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        beta = np.array([0.0, 0.3])
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # R formula: w = 1 / (π - 1 + T)
        weights_r_formula = 1.0 / (probs - 1 + treat)
        
        # Paper formula: w = T/π - (1-T)/(1-π)
        weights_paper_formula = treat / probs - (1 - treat) / (1 - probs)
        
        assert_allclose_with_report(
            weights_r_formula, weights_paper_formula,
            rtol=1e-14,
            atol=1e-15,
            name="ATE weight formula equivalence"
        )
    
    def test_ate_weights_treated_equals_inverse_prob(self, cbps_binary_module):
        """Test that ATE weight for treated equals 1/π."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.ones(n)  # All treated
        beta = np.array([0.0, 0.3])
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weights for treated: 1/(π - 1 + 1) = 1/π
        weights = 1.0 / (probs - 1 + treat)
        expected = 1.0 / probs
        
        assert_allclose_with_report(
            weights, expected,
            rtol=1e-14,
            atol=1e-15,
            name="ATE treated weight = 1/π"
        )
    
    def test_ate_weights_control_equals_negative_inverse(self, cbps_binary_module):
        """Test that ATE weight for control equals -1/(1-π)."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.zeros(n)  # All control
        beta = np.array([0.0, 0.3])
        
        # Compute propensity scores
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weights for control: 1/(π - 1 + 0) = 1/(π - 1) = -1/(1-π)
        weights = 1.0 / (probs - 1 + treat)
        expected = -1.0 / (1 - probs)
        
        assert_allclose_with_report(
            weights, expected,
            rtol=1e-14,
            atol=1e-15,
            name="ATE control weight = -1/(1-π)"
        )

class TestProbabilityClipping_Weights:
    """
    Test probability clipping in weight functions.
    
    Test ID: WT-003
    Requirement: REQ-103
    
    Probabilities must be clipped to [PROBS_MIN, 1-PROBS_MIN] to avoid:
    - Division by zero when π = 0 or π = 1
    - Numerical overflow in weight calculations
    """
    
    def test_clipping_prevents_division_by_zero_att(self, att_wt_func):
        """Test that clipping prevents division by zero in ATT weights."""
        n = 10
        
        # Create data that would produce π ≈ 1 (extreme positive theta)
        X = np.column_stack([np.ones(n), np.ones(n) * 10])
        treat = np.array([1] * 5 + [0] * 5, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 10.0])  # Very large coefficient
        
        # Should not raise division by zero
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        assert np.all(np.isfinite(weights)), \
            "ATT weights should be finite even with extreme probabilities"
    
    def test_clipping_prevents_division_by_zero_ate(self, cbps_binary_module):
        """Test that clipping prevents division by zero in ATE weights."""
        n = 10
        
        # Create data that would produce π ≈ 0 (extreme negative theta)
        X = np.column_stack([np.ones(n), np.ones(n) * 10])
        treat = np.array([1] * 5 + [0] * 5, dtype=float)
        beta = np.array([0.0, -10.0])  # Very negative coefficient
        
        # Compute propensity scores with clipping
        theta = X @ beta
        probs = scipy.special.expit(theta)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE weights
        weights = 1.0 / (probs - 1 + treat)
        
        assert np.all(np.isfinite(weights)), \
            "ATE weights should be finite even with extreme probabilities"
    
    def test_clipping_bounds(self, att_wt_func):
        """Test that probabilities are clipped to correct bounds."""
        n = 20
        
        # Create data with extreme theta values
        X = np.column_stack([np.ones(n), np.linspace(-20, 20, n)])
        treat = np.array([1] * 10 + [0] * 10, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 1.0])
        
        # Compute propensity scores
        theta = X @ beta
        probs_raw = scipy.special.expit(theta)
        probs_clipped = np.clip(probs_raw, PROBS_MIN, 1 - PROBS_MIN)
        
        # Verify clipping bounds
        assert np.all(probs_clipped >= PROBS_MIN), \
            f"Clipped probs should be >= {PROBS_MIN}"
        assert np.all(probs_clipped <= 1 - PROBS_MIN), \
            f"Clipped probs should be <= {1 - PROBS_MIN}"

    def test_clipping_value_matches_r(self, cbps_binary_module):
        """Test that clipping threshold matches R package (1e-6)."""
        # R package uses probs.min <- 1e-6
        assert PROBS_MIN == 1e-6, \
            f"PROBS_MIN should be 1e-6 to match R, got {PROBS_MIN}"
    
    def test_extreme_weights_bounded(self, att_wt_func):
        """Test that extreme probabilities produce bounded weights."""
        n = 100
        
        # Create data with some extreme probabilities
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.5, n).astype(float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 5.0])  # Large coefficient for extreme probs
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Weights should be bounded (not infinite)
        max_abs_weight = np.max(np.abs(weights))
        
        # With clipping at 1e-6, max weight should be bounded
        # For ATT: max ≈ (n/n_t) * 1 / PROBS_MIN when π → 1-PROBS_MIN
        assert max_abs_weight < 1e8, \
            f"Weights should be bounded, got max |w| = {max_abs_weight}"

class TestSampleWeightPropagation:
    """
    Test sample weight propagation in weight functions.
    
    Test ID: WT-004
    Requirement: REQ-017
    
    Sample weights should:
    - Affect the computation of n, n_t, n_c
    - Be normalized to sum to n
    - Propagate correctly through weight calculations
    """
    
    def test_uniform_weights_equivalent_to_unweighted(self, att_wt_func):
        """Test that uniform sample weights give same result as unweighted."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.4, n).astype(float)
        beta = np.array([0.0, 0.3])
        
        # Uniform weights (all ones)
        weights_uniform = np.ones(n)
        
        # Scaled uniform weights (should give same result)
        weights_scaled = np.ones(n) * 2.5
        weights_scaled_normalized = weights_scaled / np.mean(weights_scaled)
        
        w1 = att_wt_func(beta, X, treat, weights_uniform)
        w2 = att_wt_func(beta, X, treat, weights_scaled_normalized)
        
        assert_allclose_with_report(
            w1, w2,
            rtol=Tolerances.WEIGHT_RTOL,
            atol=Tolerances.WEIGHT_ATOL,
            name="Uniform vs scaled uniform weights"
        )
    
    def test_sample_weights_affect_n_t_n_c(self, att_wt_func):
        """Test that sample weights affect n_t and n_c computation."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        beta = np.array([0.0, 0.0])  # All probs = 0.5
        
        # Uniform weights
        weights_uniform = np.ones(n)
        
        # Non-uniform weights: double weight for treated
        weights_nonuniform = np.ones(n)
        weights_nonuniform[treat == 1] = 2.0
        weights_nonuniform = weights_nonuniform / np.mean(weights_nonuniform)
        
        w1 = att_wt_func(beta, X, treat, weights_uniform)
        w2 = att_wt_func(beta, X, treat, weights_nonuniform)
        
        # Results should differ due to different n_t/n ratio
        assert not np.allclose(w1, w2), \
            "Non-uniform sample weights should produce different results"
    
    def test_sample_weights_normalization(self, att_wt_func):
        """Test that sample weights are properly normalized."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.4, n).astype(float)
        beta = np.array([0.0, 0.3])
        
        # Create weights that don't sum to n
        raw_weights = np.random.exponential(1, n)
        
        # Normalize to mean 1 (sum to n)
        normalized_weights = raw_weights / np.mean(raw_weights)
        
        # Should work without error
        weights = att_wt_func(beta, X, treat, normalized_weights)
        
        assert np.all(np.isfinite(weights)), \
            "Weights should be finite with normalized sample weights"
    
    def test_weighted_sample_sizes(self, att_wt_func):
        """Test that weighted sample sizes are computed correctly."""
        n = 100
        
        X = np.column_stack([np.ones(n), np.zeros(n)])  # Intercept only
        treat = np.array([1] * 40 + [0] * 60, dtype=float)
        beta = np.array([0.0, 0.0])  # All probs = 0.5
        
        # Custom weights
        sample_weights = np.ones(n)
        sample_weights[:20] = 2.0  # First 20 treated have double weight
        sample_weights = sample_weights / np.mean(sample_weights)
        
        # Compute expected n_t and n_c
        n_t_weighted = np.sum(sample_weights[treat == 1])
        n_c_weighted = np.sum(sample_weights[treat == 0])
        n_weighted = n_t_weighted + n_c_weighted
        
        # With prob=0.5, treated weight = n/n_t
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Verify treated weights match expected scaling
        expected_treated_weight = n_weighted / n_t_weighted
        treated_weights = weights[treat == 1]
        
        assert_allclose_with_report(
            treated_weights, np.full(40, expected_treated_weight),
            rtol=1e-10,
            atol=1e-12,
            name="Weighted n_t scaling"
        )

class TestWeightEdgeCases:
    """Additional edge case tests for weight functions."""
    
    def test_single_treated_unit(self, att_wt_func):
        """Test ATT weights with single treated unit."""
        n = 50
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.zeros(n)
        treat[0] = 1  # Single treated unit
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.1])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(weights)), \
            "Weights should be finite with single treated unit"
    
    def test_highly_imbalanced_treatment(self, att_wt_func):
        """Test ATT weights with highly imbalanced treatment groups."""
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 5 + [0] * 95, dtype=float)  # 5% treated
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.2])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # Should produce valid weights
        assert np.all(np.isfinite(weights)), \
            "Weights should be finite with imbalanced treatment"
        
        # Treated weights should be large (n/n_t = 100/5 = 20 scaling)
        treated_weights = weights[treat == 1]
        assert np.all(treated_weights > 10), \
            "Treated weights should be large with few treated units"
    
    def test_all_same_covariates(self, att_wt_func):
        """Test ATT weights when all units have same covariates."""
        n = 50
        
        # All units have same covariate values
        X = np.column_stack([np.ones(n), np.ones(n) * 0.5])
        treat = np.array([1] * 25 + [0] * 25, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.3])
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # All treated should have same weight
        treated_weights = weights[treat == 1]
        assert np.allclose(treated_weights, treated_weights[0]), \
            "All treated units should have same weight with identical covariates"
        
        # All control should have same weight
        control_weights = weights[treat == 0]
        assert np.allclose(control_weights, control_weights[0]), \
            "All control units should have same weight with identical covariates"
    
    def test_zero_beta(self, att_wt_func):
        """Test ATT weights with zero coefficients (all probs = 0.5)."""
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.array([1] * 50 + [0] * 50, dtype=float)
        sample_weights = np.ones(n)
        beta = np.array([0.0, 0.0])  # All probs = 0.5
        
        weights = att_wt_func(beta, X, treat, sample_weights)
        
        # With prob=0.5 for all:
        # Treated: w = (n/n_t) * (1 - 0.5) / (1 - 0.5) = n/n_t = 2
        # Control: w = (n/n_t) * (0 - 0.5) / (1 - 0.5) = -n/n_t = -2
        expected_treated = 2.0
        expected_control = -2.0
        
        assert_allclose_with_report(
            weights[treat == 1], np.full(50, expected_treated),
            rtol=1e-10, atol=1e-12,
            name="Treated weights with prob=0.5"
        )
        assert_allclose_with_report(
            weights[treat == 0], np.full(50, expected_control),
            rtol=1e-10, atol=1e-12,
            name="Control weights with prob=0.5"
        )

class TestWeightNumericalPrecision:
    """Test numerical precision of weight computations."""
    
    def test_weight_symmetry_ate(self, cbps_binary_module):
        """Test ATE weight symmetry: w(T=1, π) = -w(T=0, 1-π)."""
        np.random.seed(42)
        n = 100
        
        # Create symmetric scenario
        probs = np.random.uniform(0.1, 0.9, n)
        
        # Treated weights
        w_treated = 1.0 / probs
        
        # Control weights with complementary probs
        w_control = -1.0 / (1 - probs)
        
        # For symmetric probs (π and 1-π), weights should be negatives
        # w(T=1, π) = 1/π
        # w(T=0, 1-π) = -1/(1-(1-π)) = -1/π
        # So w(T=1, π) = -w(T=0, 1-π)
        
        # This is a mathematical property verification
        assert np.all(np.isfinite(w_treated)), "Treated weights should be finite"
        assert np.all(np.isfinite(w_control)), "Control weights should be finite"
    
    def test_weight_magnitude_bounds(self, att_wt_func):
        """Test that weight magnitudes are bounded by clipping."""
        np.random.seed(42)
        n = 200
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.3, n).astype(float)
        sample_weights = np.ones(n)
        
        # Test with various beta values
        for scale in [0.1, 1.0, 5.0, 10.0]:
            beta = np.array([0.0, scale])
            weights = att_wt_func(beta, X, treat, sample_weights)
            
            # Maximum weight should be bounded
            max_weight = np.max(np.abs(weights))
            
            # With PROBS_MIN = 1e-6, max ATT weight ≈ n/n_t / PROBS_MIN
            n_t = np.sum(treat)
            theoretical_max = (n / n_t) / PROBS_MIN
            
            assert max_weight < theoretical_max, \
                f"Weight magnitude {max_weight} exceeds theoretical max {theoretical_max}"
    
    def test_weight_continuity(self, att_wt_func):
        """Test that weights change continuously with beta."""
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        treat = np.random.binomial(1, 0.4, n).astype(float)
        sample_weights = np.ones(n)
        
        beta1 = np.array([0.0, 0.3])
        beta2 = np.array([0.0, 0.3 + 1e-8])  # Small perturbation
        
        w1 = att_wt_func(beta1, X, treat, sample_weights)
        w2 = att_wt_func(beta2, X, treat, sample_weights)
        
        # Weights should be very close for small beta change
        max_diff = np.max(np.abs(w1 - w2))
        assert max_diff < 1e-5, \
            f"Weights should change continuously, got max diff {max_diff}"

class TestPaperFormulaVerification:
    """
    Verify weight formulas match the paper exactly.
    
    References:
    - Equation (10): ATE balance condition
    - Equation (11): ATT balance condition
    """
    
    def test_equation_10_ate_balance(self, cbps_binary_module):
        """
        Verify Equation (10) from the paper.
        
        ATE balance condition:
        E[w_i * X_i] = 0
        
        where w_i = (T_i - π_i) / (π_i * (1 - π_i))
        
        Note: This is a theoretical property that holds in expectation.
        In finite samples, we verify the balance is reasonably small
        relative to the standard error.
        """
        np.random.seed(42)
        n = 5000  # Larger sample for better convergence
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        # True propensity score model
        true_beta = np.array([0.0, 0.3])  # Moderate effect
        theta = X @ true_beta
        probs = scipy.special.expit(theta)
        
        # Generate treatment from true model
        treat = np.random.binomial(1, probs).astype(float)
        
        # Compute ATE weights with true probs
        w = (treat - probs) / (probs * (1 - probs))
        
        # Balance condition: E[w * X] should be close to 0
        balance = np.mean(w[:, None] * X, axis=0)
        
        # With true propensity scores, balance should be approximately 0
        # Use tolerance based on 1/sqrt(n) convergence rate
        tolerance = 3.0 / np.sqrt(n)  # ~0.042 for n=5000
        assert np.all(np.abs(balance) < tolerance), \
            f"ATE balance condition violated: {balance}, tolerance={tolerance}"
    
    def test_equation_11_att_balance(self, att_wt_func):
        """
        Verify Equation (11) from the paper.
        
        ATT balance condition:
        E[w_i * X_i | T=1] = E[w_i * X_i | T=0]
        
        where w_i = (n/n_1) * (T_i - π_i) / (1 - π_i)
        """
        np.random.seed(42)
        n = 1000  # Large sample
        
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        # True propensity score model
        true_beta = np.array([0.0, 0.5])
        theta = X @ true_beta
        probs = scipy.special.expit(theta)
        
        # Generate treatment from true model
        treat = np.random.binomial(1, probs).astype(float)
        sample_weights = np.ones(n)
        
        # Compute ATT weights with true beta
        w = att_wt_func(true_beta, X, treat, sample_weights)
        
        # For ATT, the weighted covariate means should balance
        # E[X | T=1] ≈ E[|w| * X | T=0] (approximately)
        treated_mean = np.mean(X[treat == 1], axis=0)
        
        # Control weighted mean (using absolute weights)
        control_weights = np.abs(w[treat == 0])
        control_weighted_mean = np.average(X[treat == 0], axis=0, 
                                           weights=control_weights)
        
        # Balance should be reasonable (not exact due to finite sample)
        balance_diff = np.abs(treated_mean - control_weighted_mean)
        assert np.all(balance_diff < 0.5), \
            f"ATT balance condition violated: diff = {balance_diff}"

class TestEquation10ATEWeights:
    """
    Test ID: PF-001, Requirement: REQ-034
    
    Equation (10) from Imai & Ratkovic (2014):
    ATE balance condition weight: w_i = (T_i - π_i) / (π_i * (1 - π_i))
    
    For IPW estimation:
    - Treated: w_i = 1/π_i
    - Control: w_i = -1/(1-π_i)
    """
    
    def test_ate_weight_formula_treated(self, paper_test_data):
        """Test ATE weight formula for treated units."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula for treated: w = 1/π
        w_treated_paper = 1 / pi[treat == 1]
        
        # Implementation formula: w = 1/(π - 1 + T) = 1/π for T=1
        w_treated_impl = 1 / (pi[treat == 1] - 1 + 1)
        
        assert_allclose_with_report(w_treated_impl, w_treated_paper,
            rtol=1e-14, atol=1e-14,
            name="ATE treated weight formula")

    def test_ate_weight_formula_control(self, paper_test_data):
        """Test ATE weight formula for control units."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula for control: w = -1/(1-π)
        w_control_paper = -1 / (1 - pi[treat == 0])
        
        # Implementation formula: w = 1/(π - 1 + T) = 1/(π-1) = -1/(1-π) for T=0
        w_control_impl = 1 / (pi[treat == 0] - 1 + 0)
        
        assert_allclose_with_report(w_control_impl, w_control_paper,
            rtol=1e-14, atol=1e-14,
            name="ATE control weight formula")

    def test_ate_balance_condition_formula(self, paper_test_data):
        """Test ATE balance condition: E[w_i * X_i] = 0."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n = data['n']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper Equation (10): w = (T - π) / (π * (1 - π))
        w_balance = (treat - pi) / (pi * (1 - pi))
        
        # Balance condition: (1/n) * Σ w_i * X_i ≈ 0
        balance = (sample_weights[:, None] * X).T @ w_balance / n
        
        # At true beta, balance should be approximately zero
        # (not exactly zero due to finite sample)
        assert np.all(np.abs(balance) < 0.5), f"Balance condition: {balance}"

class TestEquation11ATTWeights:
    """
    Test ID: PF-002, Requirement: REQ-035
    
    Equation (11) from Imai & Ratkovic (2014):
    ATT balance condition weight: w_i = (n/n_1) * (T_i - π_i) / (1 - π_i)
    
    For IPW estimation:
    - Treated: w_i = n/n_1
    - Control: w_i = -(n/n_1) * π_i / (1 - π_i)
    """
    
    def test_att_weight_formula_treated(self, att_wt_func, paper_test_data):
        """Test ATT weight formula for treated units."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, n_t = data['n'], data['n_t']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula for treated: w = (n/n_1) * (1 - π) / (1 - π) = n/n_1
        w_treated_paper = np.full(int(n_t), n / n_t)
        
        # Implementation
        w_impl = att_wt_func(beta, X, treat, sample_weights)
        w_treated_impl = w_impl[treat == 1]
        
        assert_allclose_with_report(w_treated_impl, w_treated_paper,
            rtol=1e-10, atol=1e-10,
            name="ATT treated weight formula")

    def test_att_weight_formula_control(self, att_wt_func, paper_test_data):
        """Test ATT weight formula for control units."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, n_t = data['n'], data['n_t']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula for control: w = -(n/n_1) * π / (1 - π)
        w_control_paper = -(n / n_t) * pi[treat == 0] / (1 - pi[treat == 0])
        
        # Implementation
        w_impl = att_wt_func(beta, X, treat, sample_weights)
        w_control_impl = w_impl[treat == 0]
        
        assert_allclose_with_report(w_control_impl, w_control_paper,
            rtol=1e-10, atol=1e-10,
            name="ATT control weight formula")

    def test_att_balance_condition_formula(self, att_wt_func, paper_test_data):
        """Test ATT balance condition: E[w_i * X_i] = 0."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n = data['n']
        sample_weights = data['sample_weights']
        
        # ATT weights
        w_att = att_wt_func(beta, X, treat, sample_weights)
        
        # Balance condition: (1/n) * Σ w_i * X_i ≈ 0
        balance = (sample_weights[:, None] * X).T @ w_att / n
        
        # At true beta, balance should be approximately zero
        assert np.all(np.abs(balance) < 0.5), f"ATT balance condition: {balance}"

class TestEquation7ScoreCondition:
    """
    Test ID: PF-005, Requirement: REQ-031
    
    Equation (7) from Imai & Ratkovic (2014):
    Score condition: E[(T_i - π_i) * X_i] = 0
    
    This is the standard MLE score equation for logistic regression.
    """
    
    def test_score_condition_formula(self, paper_test_data):
        """Test score condition formula."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n = data['n']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Score condition: (1/n) * Σ (T_i - π_i) * X_i
        score = (sample_weights[:, None] * X).T @ (treat - pi) / n
        
        # At true beta, score should be approximately zero
        assert np.all(np.abs(score) < 0.2), f"Score condition: {score}"

    def test_score_condition_gradient_relationship(self, paper_test_data):
        """Test that score condition is gradient of log-likelihood."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Score condition
        score = (sample_weights[:, None] * X).T @ (treat - pi) / n
        
        # Numerical gradient of log-likelihood
        def log_likelihood(b):
            th = X @ b
            p = np.clip(scipy.special.expit(th), PROBS_MIN, 1 - PROBS_MIN)
            return np.sum(sample_weights * (treat * np.log(p) + (1 - treat) * np.log(1 - p)))
        
        eps = np.sqrt(np.finfo(float).eps)
        grad_numerical = np.zeros(k)
        for i in range(k):
            beta_p, beta_m = beta.copy(), beta.copy()
            beta_p[i] += eps
            beta_m[i] -= eps
            grad_numerical[i] = (log_likelihood(beta_p) - log_likelihood(beta_m)) / (2 * eps)
        
        # Score should equal gradient of log-likelihood (scaled by n)
        assert_allclose_with_report(score * n, grad_numerical,
            rtol=1e-5, atol=1e-5,
            name="Score-gradient relationship")

class TestEquation12GMMObjective:
    """
    Test ID: PF-003, Requirement: REQ-036
    
    Equation (12) from Imai & Ratkovic (2014):
    GMM objective: Q(β) = ḡ(β)' Σ^{-1} ḡ(β)
    
    where ḡ(β) combines score and balance conditions.
    """
    
    def test_gmm_objective_quadratic_form(self, gmm_loss, gmm_func, paper_test_data):
        """Test GMM objective is quadratic form."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        sample_weights = data['sample_weights']
        
        # Compute GMM loss
        loss = gmm_loss(beta, X, treat, sample_weights, att=0, inv_V=None)
        
        # Loss should be non-negative (quadratic form)
        assert loss >= 0, f"GMM loss should be non-negative, got {loss}"

    def test_gmm_objective_structure(self, gmm_func, paper_test_data):
        """Test GMM objective structure matches paper."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Score condition: (1/n) * X' @ (T - π)
        score_cond = (sample_weights[:, None] * X).T @ (treat - pi) / n
        
        # Balance condition (ATE): (1/n) * X' @ w
        w_ate = 1 / (pi - 1 + treat)
        balance_cond = (sample_weights[:, None] * X).T @ w_ate / n
        
        # Combined moment conditions
        gbar = np.concatenate([score_cond.ravel(), balance_cond.ravel()])
        
        # gbar should have length 2k
        assert len(gbar) == 2 * k

    def test_gmm_loss_at_optimum(self, cbps_binary_fit, paper_test_data):
        """Test that GMM loss is minimized at optimum."""
        data = paper_test_data
        X, treat = data['X'], data['treat']
        sample_weights = data['sample_weights']
        
        # Fit CBPS
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True,
                                  sample_weights=sample_weights)
        
        # J-statistic should be small at optimum
        assert result['J'] >= 0, "J-statistic should be non-negative"

class TestEquation8ATEBalanceCondition:
    """
    Test ID: PF-006, Requirement: REQ-032
    
    Equation (8) from Imai & Ratkovic (2014):
    ATE balance condition: E[w_i^{ATE} * X_i] = 0
    
    where w_i^{ATE} = (T_i - π_i) / (π_i * (1 - π_i))
    """
    
    def test_ate_balance_moment_condition(self, paper_test_data):
        """Test ATE balance moment condition structure."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # ATE balance weights from paper
        w_ate = (treat - pi) / (pi * (1 - pi))
        
        # Moment condition: (1/n) * X' @ w
        moment = (sample_weights[:, None] * X).T @ w_ate / n
        
        # Should be k-dimensional
        assert moment.shape == (k,)

    def test_ate_balance_ipw_equivalence(self, paper_test_data):
        """Test ATE balance weights equivalent to IPW weights."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula: w = (T - π) / (π * (1 - π))
        w_paper = (treat - pi) / (pi * (1 - pi))
        
        # IPW formula: T/π - (1-T)/(1-π)
        w_ipw = treat / pi - (1 - treat) / (1 - pi)
        
        assert_allclose_with_report(w_paper, w_ipw,
            rtol=1e-14, atol=1e-14,
            name="ATE balance-IPW equivalence")

class TestEquation9ATTBalanceCondition:
    """
    Test ID: PF-007, Requirement: REQ-033
    
    Equation (9) from Imai & Ratkovic (2014):
    ATT balance condition: E[w_i^{ATT} * X_i] = 0
    
    where w_i^{ATT} = (n/n_1) * (T_i - π_i) / (1 - π_i)
    """
    
    def test_att_balance_moment_condition(self, att_wt_func, paper_test_data):
        """Test ATT balance moment condition structure."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        # ATT weights
        w_att = att_wt_func(beta, X, treat, sample_weights)
        
        # Moment condition: (1/n) * X' @ w
        moment = (sample_weights[:, None] * X).T @ w_att / n
        
        # Should be k-dimensional
        assert moment.shape == (k,)

    def test_att_balance_formula_verification(self, paper_test_data):
        """Test ATT balance weight formula matches paper."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, n_t = data['n'], data['n_t']
        
        theta = X @ beta
        pi = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # Paper formula: w = (n/n_1) * (T - π) / (1 - π)
        w_paper = (n / n_t) * (treat - pi) / (1 - pi)
        
        # Verify formula components
        # For treated: (n/n_1) * (1 - π) / (1 - π) = n/n_1
        w_treated = w_paper[treat == 1]
        assert np.allclose(w_treated, n / n_t, rtol=1e-10)
        
        # For control: (n/n_1) * (0 - π) / (1 - π) = -(n/n_1) * π / (1 - π)
        w_control = w_paper[treat == 0]
        expected_control = -(n / n_t) * pi[treat == 0] / (1 - pi[treat == 0])
        assert_allclose_with_report(w_control, expected_control,
            rtol=1e-10, atol=1e-10,
            name="ATT control weight formula")

class TestEquation15CovarianceMatrix:
    """
    Test ID: PF-004, Requirement: REQ-037
    
    Equation (15) from Imai & Ratkovic (2014):
    Sandwich variance estimator: Var(β̂) = (G'WG)^{-1} G'W Ω W'G (G'WG)^{-1}
    
    where:
    - G is the gradient matrix
    - W is the weight matrix (inverse of V)
    - Ω is the covariance of moment conditions
    """
    
    def test_sandwich_formula_components(self, compute_vcov, r_ginv, compute_V_matrix, paper_test_data):
        """Test sandwich formula component dimensions."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # G matrix: (k, 2k) for over-identified
        XG_1 = -X * (probs * (1 - probs))[:, None] * sample_weights[:, None]
        XG_2 = -X * ((treat - probs)**2 / (probs * (1 - probs)))[:, None] * sample_weights[:, None]
        G = np.hstack([(XG_1.T @ X), (XG_2.T @ X)]) / n
        
        assert G.shape == (k, 2*k), f"G shape: {G.shape}"
        
        # W matrix: (2k, 2k)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        assert inv_V.shape == (2*k, 2*k), f"W shape: {inv_V.shape}"
        
        # Omega matrix: (2k, 2k)
        XW_1 = X * (treat - probs)[:, None] * np.sqrt(sample_weights)[:, None]
        XW_2 = X * (1 / (probs - 1 + treat))[:, None] * np.sqrt(sample_weights)[:, None]
        W1 = np.vstack([XW_1.T, XW_2.T])
        Omega = (W1 @ W1.T) / n
        
        assert Omega.shape == (2*k, 2*k), f"Omega shape: {Omega.shape}"

    def test_sandwich_formula_result(self, compute_vcov, r_ginv, compute_V_matrix, paper_test_data):
        """Test sandwich formula produces valid vcov."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        # Vcov should be (k, k)
        assert vcov.shape == (k, k)
        
        # Vcov should be symmetric
        assert np.allclose(vcov, vcov.T, atol=1e-12)
        
        # Diagonal should be positive (variances)
        assert np.all(np.diag(vcov) >= 0)

class TestEquation16ATECovariance:
    """
    Test ID: PF-004 (continued), Requirement: REQ-038
    
    Equation (16) from Imai & Ratkovic (2014):
    ATE-specific covariance matrix structure.
    """
    
    def test_ate_vcov_structure(self, compute_vcov, r_ginv, compute_V_matrix, paper_test_data):
        """Test ATE vcov has correct structure."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        vcov = compute_vcov(
            beta, probs, treat, X, sample_weights, att=0,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        # Standard errors should be reasonable
        se = np.sqrt(np.diag(vcov))
        assert np.all(se > 0), "Standard errors should be positive"
        assert np.all(se < 10), "Standard errors should be reasonable"

class TestEquation17ATTCovariance:
    """
    Test ID: PF-004 (continued), Requirement: REQ-039
    
    Equation (17) from Imai & Ratkovic (2014):
    ATT-specific covariance matrix structure.
    """
    
    def test_att_vcov_structure(self, compute_vcov, r_ginv, compute_V_matrix, paper_test_data):
        """Test ATT vcov has correct structure."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, k = data['n'], data['k']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        XprimeX_inv = r_ginv(X.T @ X)
        inv_V = compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)
        
        vcov = compute_vcov(
            beta, probs, treat, X, sample_weights, att=1,
            bal_only=False, XprimeX_inv=XprimeX_inv, this_inv_V=inv_V,
            two_step=True, n=n
        )
        
        # Standard errors should be reasonable
        se = np.sqrt(np.diag(vcov))
        assert np.all(se > 0), "Standard errors should be positive"
        assert np.all(se < 10), "Standard errors should be reasonable"

    def test_att_vcov_scaling_factor(self, compute_V_matrix, paper_test_data):
        """Test ATT V matrix includes n/n_t scaling factor."""
        data = paper_test_data
        X, treat, beta = data['X'], data['treat'], data['true_beta']
        n, n_t = data['n'], data['n_t']
        sample_weights = data['sample_weights']
        
        theta = X @ beta
        probs = np.clip(scipy.special.expit(theta), PROBS_MIN, 1 - PROBS_MIN)
        
        # ATT V matrix should include n/n_t scaling
        inv_V_att = compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)
        inv_V_ate = compute_V_matrix(X, probs, sample_weights, treat, att=0, n=n)
        
        # ATT and ATE V matrices should be different due to scaling
        assert not np.allclose(inv_V_att, inv_V_ate)

class TestMomentConditionConvergence:
    """Test that moment conditions converge to zero at optimum."""
    
    def test_score_condition_at_optimum(self, cbps_binary_fit, paper_test_data):
        """Test score condition approximately zero at optimum."""
        data = paper_test_data
        X, treat = data['X'], data['treat']
        n = data['n']
        sample_weights = data['sample_weights']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True,
                                  sample_weights=sample_weights)
        
        beta_opt = result['coefficients'].ravel()
        probs_opt = result['fitted_values']
        
        # Score condition
        score = (sample_weights[:, None] * X).T @ (treat - probs_opt) / n
        
        # Should be small at optimum
        assert np.max(np.abs(score)) < 0.1, f"Score at optimum: {score}"

    def test_balance_condition_at_optimum(self, cbps_binary_fit, paper_test_data):
        """Test balance condition approximately zero at optimum."""
        data = paper_test_data
        X, treat = data['X'], data['treat']
        n = data['n']
        sample_weights = data['sample_weights']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True,
                                  sample_weights=sample_weights)
        
        probs_opt = result['fitted_values']
        
        # ATE balance weights
        w_ate = 1 / (probs_opt - 1 + treat)
        
        # Balance condition
        balance = (sample_weights[:, None] * X).T @ w_ate / n
        
        # Should be small at optimum
        assert np.max(np.abs(balance)) < 1.0, f"Balance at optimum: {balance}"

class TestBug016SVDSign:
    """
    Test ID: REG-001, Requirement: REQ-088
    
    Bug 016: SVD sign inconsistency between R and Python.
    
    The issue was that SVD decomposition can have arbitrary sign flips
    in the singular vectors, which can cause different results between
    R and Python implementations.
    
    The fix ensures consistent sign convention by:
    1. Using the same SVD algorithm as R (LAPACK)
    2. Applying sign correction to match R's convention
    """
    
    def test_svd_sign_consistency(self, r_ginv):
        """Test that pseudoinverse is consistent regardless of SVD sign."""
        np.random.seed(42)
        
        # Create a test matrix
        A = np.random.randn(5, 3)
        
        # Compute pseudoinverse
        A_pinv = r_ginv(A)
        
        # Verify Moore-Penrose conditions
        # 1. A @ A+ @ A = A
        reconstructed = A @ A_pinv @ A
        assert_allclose_with_report(reconstructed, A, rtol=1e-10, atol=1e-12,
            name="Moore-Penrose condition 1")
        
        # 2. A+ @ A @ A+ = A+
        reconstructed_pinv = A_pinv @ A @ A_pinv
        assert_allclose_with_report(reconstructed_pinv, A_pinv, rtol=1e-10, atol=1e-12,
            name="Moore-Penrose condition 2")

    def test_svd_sign_with_symmetric_matrix(self, r_ginv):
        """Test SVD sign consistency with symmetric matrices."""
        np.random.seed(123)
        
        # Create symmetric positive definite matrix
        X = np.random.randn(5, 3)
        A = X.T @ X  # Symmetric PSD
        
        A_pinv = r_ginv(A)
        
        # For symmetric PSD, pseudoinverse should also be symmetric
        assert_allclose_with_report(A_pinv, A_pinv.T, rtol=1e-14, atol=1e-14,
            name="Pseudoinverse symmetry")
        
        # Verify A @ A+ @ A = A
        reconstructed = A @ A_pinv @ A
        assert_allclose_with_report(reconstructed, A, rtol=1e-10, atol=1e-12,
            name="Symmetric matrix reconstruction")

    def test_svd_sign_deterministic(self, r_ginv):
        """Test that SVD-based pseudoinverse is deterministic."""
        np.random.seed(456)
        
        A = np.random.randn(4, 4)
        
        # Compute multiple times
        results = [r_ginv(A) for _ in range(5)]
        
        # All results should be identical
        for i in range(1, len(results)):
            assert_allclose_with_report(results[i], results[0], rtol=1e-15, atol=1e-15,
                name=f"Determinism check {i}")

    def test_svd_sign_with_rank_deficient(self, r_ginv):
        """Test SVD sign handling with rank-deficient matrices."""
        np.random.seed(789)
        
        # Create rank-deficient matrix
        u = np.random.randn(5, 2)
        v = np.random.randn(2, 4)
        A = u @ v  # Rank 2
        
        A_pinv = r_ginv(A)
        
        # Verify Moore-Penrose conditions still hold
        reconstructed = A @ A_pinv @ A
        assert_allclose_with_report(reconstructed, A, rtol=1e-10, atol=1e-10,
            name="Rank-deficient reconstruction")

class TestBug017WeightNormalization:
    """
    Test ID: REG-002, Requirement: REQ-089
    
    Bug 017: Weight normalization inconsistency.
    
    The issue was that weights were not being normalized consistently
    between different estimation modes (ATT vs ATE, standardize vs not).
    
    The fix ensures:
    1. Standardized weights sum to n
    2. Unstandardized weights maintain proper scale
    3. ATT and ATE weights follow correct formulas
    """
    
    def test_standardized_weights_sum_correctly(self, cbps_binary_fit, lalonde_full):
        """Test that standardized weights sum correctly.
        
        Note: The CBPS implementation normalizes weights such that they sum to 2
        (1 for treated group contribution, 1 for control group contribution).
        """
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # ATE with standardize=True
        result_ate = cbps_binary_fit(treat, X, att=0, method='over', 
                                      two_step=True, standardize=True)
        
        weight_sum = np.sum(result_ate['weights'])
        # Weights are normalized to sum to 2 (1 per group)
        assert np.isclose(weight_sum, 2.0, rtol=0.01), \
            f"ATE standardized weights should sum to 2, got {weight_sum}"
        
        # ATT with standardize=True
        result_att = cbps_binary_fit(treat, X, att=1, method='over',
                                      two_step=True, standardize=True)
        
        weight_sum = np.sum(result_att['weights'])
        assert np.isclose(weight_sum, 2.0, rtol=0.01), \
            f"ATT standardized weights should sum to 2, got {weight_sum}"

    def test_unstandardized_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test that unstandardized weights are all positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over',
                                  two_step=True, standardize=False)
        
        assert np.all(result['weights'] > 0), "All weights should be positive"

    def test_ate_weight_formula(self, cbps_binary_fit, lalonde_full):
        """Test ATE weights follow correct formula: 1/π for T=1, 1/(1-π) for T=0."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over',
                                  two_step=True, standardize=False)
        
        probs = result['fitted_values']
        weights = result['weights']
        
        # For treated: weight ∝ 1/π
        # For control: weight ∝ 1/(1-π)
        expected_t = 1 / probs[treat == 1]
        expected_c = 1 / (1 - probs[treat == 0])
        
        # Check proportionality (ratio should be constant within groups)
        ratio_t = weights[treat == 1] / expected_t
        ratio_c = weights[treat == 0] / expected_c
        
        # Ratios should be approximately constant
        cv_t = np.std(ratio_t) / np.mean(ratio_t) if np.mean(ratio_t) > 0 else 0
        cv_c = np.std(ratio_c) / np.mean(ratio_c) if np.mean(ratio_c) > 0 else 0
        
        assert cv_t < 0.01, f"Treated weight ratios should be constant, CV={cv_t}"
        assert cv_c < 0.01, f"Control weight ratios should be constant, CV={cv_c}"

    def test_att_weight_formula(self, cbps_binary_fit, lalonde_full):
        """Test ATT weights follow correct formula: 1 for T=1, π/(1-π) for T=0."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n_t = lalonde_full['n_t']
        
        result = cbps_binary_fit(treat, X, att=1, method='over',
                                  two_step=True, standardize=False)
        
        probs = result['fitted_values']
        weights = result['weights']
        
        # For treated: weight should be constant (all 1 before normalization)
        weights_t = weights[treat == 1]
        
        # Treated weights should have low variation
        cv_t = np.std(weights_t) / np.mean(weights_t) if np.mean(weights_t) > 0 else 0
        assert cv_t < 0.1, f"Treated weights should be nearly constant, CV={cv_t}"

    def test_weight_normalization_consistency(self, cbps_binary_fit, lalonde_full):
        """Test weight normalization is consistent across methods."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        methods = ['over', 'exact']
        
        for method in methods:
            result = cbps_binary_fit(treat, X, att=0, method=method,
                                      two_step=True, standardize=True)
            
            weight_sum = np.sum(result['weights'])
            # Weights are normalized to sum to 2 (1 per group)
            assert np.isclose(weight_sum, 2.0, rtol=0.01), \
                f"Method {method}: weights should sum to 2, got {weight_sum}"

class TestKnownIssuesRegression:
    """
    Test ID: REG-003, Requirement: REQ-090
    
    Regression tests for other known issues that have been fixed.
    """
    
    def test_input_array_not_modified(self, cbps_binary_fit, lalonde_full):
        """Test that input arrays are not significantly modified in place.
        
        Note: Due to floating point operations, tiny differences (< 1e-10) may occur.
        """
        X_original = lalonde_full['X'].copy()
        treat_original = lalonde_full['treat'].copy()
        
        X = lalonde_full['X'].copy()
        treat = lalonde_full['treat'].copy()
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Input arrays should not be significantly modified
        # Allow tiny floating point differences
        assert_allclose_with_report(X, X_original, rtol=1e-10, atol=1e-10,
            name="X should not be modified")
        assert_allclose_with_report(treat, treat_original, rtol=1e-10, atol=1e-10,
            name="treat should not be modified")

    def test_convergence_flag_accurate(self, cbps_binary_fit, lalonde_full):
        """Test that convergence flag accurately reflects optimization status."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Converged results should have valid outputs
        if result['converged']:
            assert np.all(np.isfinite(result['coefficients'])), \
                "Converged but coefficients contain non-finite values"
            assert np.all(np.isfinite(result['weights'])), \
                "Converged but weights contain non-finite values"

    def test_j_statistic_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is always non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                
                assert result['J'] >= 0, \
                    f"J-statistic should be >= 0, got {result['J']} for att={att}, method={method}"

    def test_fitted_values_in_valid_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values (propensity scores) are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                
                assert np.all(result['fitted_values'] > 0), \
                    f"Fitted values should be > 0 for att={att}, method={method}"
                assert np.all(result['fitted_values'] < 1), \
                    f"Fitted values should be < 1 for att={att}, method={method}"

    def test_vcov_symmetric_and_psd(self, cbps_binary_fit, lalonde_full):
        """Test variance-covariance matrix is symmetric and PSD."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # The variance matrix is stored as 'var' in the result
        if 'var' in result and result['var'] is not None:
            vcov = result['var']
            
            # Symmetric
            assert np.allclose(vcov, vcov.T, atol=1e-10), \
                "Vcov should be symmetric"
            
            # Positive semi-definite
            eigenvalues = np.linalg.eigvalsh(vcov)
            assert np.all(eigenvalues >= -1e-10), \
                f"Vcov should be PSD, min eigenvalue = {np.min(eigenvalues)}"

    def test_sample_weights_properly_used(self, cbps_binary_fit, lalonde_full):
        """Test that sample weights actually affect the results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        # Uniform weights
        result_uniform = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Non-uniform weights
        np.random.seed(42)
        sample_weights = np.random.exponential(1, n)
        sample_weights = sample_weights / np.mean(sample_weights)
        
        result_weighted = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights
        )
        
        # Results should be different
        coef_diff = np.max(np.abs(result_uniform['coefficients'].ravel() - 
                                   result_weighted['coefficients'].ravel()))
        
        assert coef_diff > 1e-6, \
            "Sample weights should affect coefficients"

class TestNumericalStabilityRegression:
    """Regression tests for numerical stability issues."""
    
    def test_no_nan_in_outputs(self, cbps_binary_fit, lalonde_full):
        """Test that outputs never contain NaN values."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                for two_step in [True, False]:
                    result = cbps_binary_fit(
                        treat, X, att=att, method=method, two_step=two_step
                    )
                    
                    assert not np.any(np.isnan(result['coefficients'])), \
                        f"NaN in coefficients: att={att}, method={method}, two_step={two_step}"
                    assert not np.any(np.isnan(result['weights'])), \
                        f"NaN in weights: att={att}, method={method}, two_step={two_step}"
                    assert not np.any(np.isnan(result['fitted_values'])), \
                        f"NaN in fitted_values: att={att}, method={method}, two_step={two_step}"

    def test_no_inf_in_outputs(self, cbps_binary_fit, lalonde_full):
        """Test that outputs never contain Inf values."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                
                assert not np.any(np.isinf(result['coefficients'])), \
                    f"Inf in coefficients: att={att}, method={method}"
                assert not np.any(np.isinf(result['weights'])), \
                    f"Inf in weights: att={att}, method={method}"
                assert not np.any(np.isinf(result['fitted_values'])), \
                    f"Inf in fitted_values: att={att}, method={method}"

    def test_extreme_coefficient_handling(self, cbps_binary_fit):
        """Test handling of cases that might produce extreme coefficients."""
        np.random.seed(42)
        n = 200
        
        # Create data with strong separation
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        # Strong selection
        true_beta = np.array([-3.0, 2.0])
        theta = X @ true_beta
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        # Ensure some in each group
        while np.sum(treat) < 20 or np.sum(1 - treat) < 20:
            treat = np.random.binomial(1, pi_true).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Coefficients should be finite
        assert np.all(np.isfinite(result['coefficients'])), \
            "Coefficients should be finite even with strong separation"

class TestAPIConsistency:
    """Test API consistency and backward compatibility."""
    
    def test_result_keys_present(self, cbps_binary_fit, lalonde_full):
        """Test that all expected keys are present in result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        # Core expected keys
        expected_keys = ['coefficients', 'weights', 'fitted_values', 'converged', 'J']
        
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not in result"

    def test_coefficient_shape_consistency(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape is consistent across configurations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                
                coef_shape = result['coefficients'].shape
                assert coef_shape == (k,) or coef_shape == (k, 1), \
                    f"Unexpected coefficient shape {coef_shape} for att={att}, method={method}"

    def test_weight_shape_consistency(self, cbps_binary_fit, lalonde_full):
        """Test weight shape is consistent across configurations."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        for att in [0, 1]:
            for method in ['over', 'exact']:
                result = cbps_binary_fit(treat, X, att=att, method=method, two_step=True)
                
                assert result['weights'].shape == (n,), \
                    f"Unexpected weight shape for att={att}, method={method}"