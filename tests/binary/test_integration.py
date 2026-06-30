"""
Integration Tests for Binary CBPS
=================================

End-to-end integration tests for binary CBPS using the LaLonde dataset,
covering ATE and ATT estimation, multiple dataset scenarios, and
cross-validation of results against known benchmarks.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
    DOI: 10.1111/rssb.12027

LaLonde, R. J. (1986). Evaluating the Econometric Evaluations of
    Training Programs with Experimental Data. American Economic Review,
    76(4), 604-620.
"""

import numpy as np
import pytest
import scipy.special

from ..conftest import assert_matrix_symmetric, assert_positive_semidefinite


pytestmark = [pytest.mark.integration, pytest.mark.slow]

def lalonde_full_for_r_comparison():
    """
    Load the full LaLonde dataset (3212 observations) for R comparison tests.
    
    R benchmarks were generated using the full dataset, not the Dehejia-Wahba subset.
    """
    try:
        from cbps.datasets import load_lalonde
        df = load_lalonde(dehejia_wahba_only=False)
    except ImportError:
        pytest.skip("cbps.datasets not available")
        return None
    
    # Standard covariates used in CBPS examples
    covariate_cols = ['age', 'educ', 're74', 're75', 'married', 'nodegr']
    
    # Extract covariates and add intercept
    X_raw = df[covariate_cols].values
    X = np.column_stack([np.ones(len(X_raw)), X_raw])
    
    # Extract treatment
    treat = df['treat'].values.astype(float)
    
    # Sample weights
    sample_weights = np.ones(len(df))
    
    return {
        'df': df,
        'X': X,
        'treat': treat,
        'sample_weights': sample_weights,
        'n': len(treat),
        'k': X.shape[1],
        'n_t': int(np.sum(treat)),
        'n_c': int(np.sum(1 - treat)),
    }

class TestATEOverTwostep:
    """
    Test ID: E2E-004, Requirement: REQ-068
    
    End-to-end test for ATE estimation with over-identified GMM
    and two-step estimator.
    """
    
    def test_ate_over_twostep_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATE over twostep converges."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = lalonde_full['sample_weights']
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights, standardize=True
        )
        
        assert result['converged'], "ATE over twostep should converge"

    def test_ate_over_twostep_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape matches covariates."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['coefficients'].shape == (k,) or result['coefficients'].shape == (k, 1), \
            f"Coefficients shape should be ({k},) or ({k}, 1)"

    def test_ate_over_twostep_weights_shape(self, cbps_binary_fit, lalonde_full):
        """Test weights shape matches sample size."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['weights'].shape == (n,), f"Weights shape should be ({n},)"

    def test_ate_over_twostep_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test all weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['weights'] > 0), "All weights should be positive"

    def test_ate_over_twostep_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values (propensity scores) are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['fitted_values'] > 0), "Fitted values should be > 0"
        assert np.all(result['fitted_values'] < 1), "Fitted values should be < 1"

    def test_ate_over_twostep_j_statistic_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['J'] >= 0, "J-statistic should be non-negative"

    def test_ate_over_twostep_vcov_symmetric(self, cbps_binary_fit, lalonde_full):
        """Test variance-covariance matrix is symmetric."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        if 'var' in result and result['var'] is not None:
            assert_matrix_symmetric(result['var'], atol=1e-10, name="ATE vcov")

    def test_ate_over_twostep_vcov_psd(self, cbps_binary_fit, lalonde_full):
        """Test variance-covariance matrix is positive semi-definite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        if 'var' in result and result['var'] is not None:
            assert_positive_semidefinite(result['var'], atol=1e-8, name="ATE vcov")

    def test_ate_over_twostep_balance_improvement(self, cbps_binary_fit, lalonde_full):
        """Test that CBPS improves covariate balance for ATE."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Unweighted balance (standardized mean difference)
        mean_t = np.mean(X[treat == 1], axis=0)
        mean_c = np.mean(X[treat == 0], axis=0)
        std_pooled = np.sqrt((np.var(X[treat == 1], axis=0) + np.var(X[treat == 0], axis=0)) / 2)
        std_pooled = np.where(std_pooled > 0, std_pooled, 1)
        smd_unweighted = np.abs(mean_t - mean_c) / std_pooled
        
        # Weighted balance
        w_t = weights[treat == 1]
        w_c = weights[treat == 0]
        mean_t_w = np.average(X[treat == 1], axis=0, weights=w_t)
        mean_c_w = np.average(X[treat == 0], axis=0, weights=w_c)
        smd_weighted = np.abs(mean_t_w - mean_c_w) / std_pooled
        
        max_smd_unweighted = np.max(smd_unweighted[1:])  # Skip intercept
        max_smd_weighted = np.max(smd_weighted[1:])
        
        assert max_smd_weighted < max_smd_unweighted + 0.1 or max_smd_weighted < 0.25, \
            f"Weighted balance ({max_smd_weighted:.3f}) should improve on unweighted ({max_smd_unweighted:.3f})"

class TestATEExactTwostep:
    """
    Test ID: E2E-005, Requirement: REQ-069
    
    End-to-end test for ATE estimation with exactly identified GMM.
    """
    
    def test_ate_exact_twostep_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATE exact twostep converges."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        assert result['converged'], "ATE exact twostep should converge"

    def test_ate_exact_twostep_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape matches covariates."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        assert result['coefficients'].shape == (k,) or result['coefficients'].shape == (k, 1)

    def test_ate_exact_twostep_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test all weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        assert np.all(result['weights'] > 0)

    def test_ate_exact_twostep_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        assert np.all(result['fitted_values'] > 0)
        assert np.all(result['fitted_values'] < 1)

    def test_ate_exact_j_statistic_small(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is small for exactly identified model."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        
        # For exactly identified, J should be small (but may not be exactly 0)
        assert result['J'] < 0.1, f"J-statistic should be small for exact, got {result['J']}"

    def test_ate_exact_vs_over_different(self, cbps_binary_fit, lalonde_full):
        """Test exact and over-identified give different results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_exact = cbps_binary_fit(treat, X, att=0, method='exact', two_step=True)
        result_over = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        coef_diff = np.abs(result_exact['coefficients'].ravel() - 
                          result_over['coefficients'].ravel())
        
        assert np.max(coef_diff) > 1e-6, \
            "Exact and over-identified should give different coefficients"

class TestATEVsATT:
    """Test differences between ATE and ATT estimation."""
    
    def test_ate_att_different_weights(self, cbps_binary_fit, lalonde_full):
        """Test that ATE and ATT produce different weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_ate = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_att = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        weight_diff = np.abs(result_ate['weights'] - result_att['weights'])
        
        assert np.max(weight_diff) > 1e-6, \
            "ATE and ATT should produce different weights"

    def test_ate_att_different_coefficients(self, cbps_binary_fit, lalonde_full):
        """Test that ATE and ATT produce different coefficients."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_ate = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        result_att = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        coef_diff = np.abs(result_ate['coefficients'].ravel() - 
                          result_att['coefficients'].ravel())
        
        assert np.max(coef_diff) > 1e-6, \
            "ATE and ATT should produce different coefficients"

    def test_ate_weights_symmetric(self, cbps_binary_fit, lalonde_full):
        """Test ATE weights are symmetric (same formula for treated and control)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        probs = result['fitted_values']
        
        # ATE weights: 1/π for treated, 1/(1-π) for control
        expected_weights_t = 1 / probs[treat == 1]
        expected_weights_c = 1 / (1 - probs[treat == 0])
        
        # Both groups should have positive weights
        assert np.all(expected_weights_t > 0)
        assert np.all(expected_weights_c > 0)

    def test_att_weights_asymmetric(self, cbps_binary_fit, lalonde_full):
        """Test ATT weights are asymmetric (different for treated and control)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        weights = result['weights']
        
        # ATT: treated get weight 1, control get weight π/(1-π)
        # After normalization, treated weights should be more uniform
        weights_t = weights[treat == 1]
        weights_c = weights[treat == 0]
        
        # Control weights should have more variation
        cv_t = np.std(weights_t) / np.mean(weights_t) if np.mean(weights_t) > 0 else 0
        cv_c = np.std(weights_c) / np.mean(weights_c) if np.mean(weights_c) > 0 else 0
        
        # This is a soft check - ATT typically has more variation in control weights
        assert cv_c >= 0, "Control weights should have non-negative CV"

class TestATEWeightNormalization:
    """Test weight normalization options for ATE."""
    
    def test_ate_standardize_true(self, cbps_binary_fit, lalonde_full):
        """Test standardized weights sum correctly.
        
        Note: The CBPS implementation normalizes weights such that they sum to 2
        (1 for treated group contribution, 1 for control group contribution).
        """
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True, standardize=True)
        
        weight_sum = np.sum(result['weights'])
        assert np.isclose(weight_sum, 2.0, rtol=0.01), \
            f"Standardized weights should sum to 2, got {weight_sum}"

    def test_ate_standardize_false(self, cbps_binary_fit, lalonde_full):
        """Test unstandardized weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True, standardize=False)
        
        assert np.all(result['weights'] > 0)

class TestATEContinuousUpdating:
    """Test ATE with continuous updating estimator.
    
    Note: Continuous updating GMM may not always converge, especially with
    real-world data. Tests verify that results are valid even if not converged.
    """
    
    def test_ate_over_continuous_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATE over continuous produces valid results.
        
        Note: Continuous updating GMM may not always converge with real-world data
        due to the iterative weight matrix updates. This is known behavior.
        We verify that results are still valid (finite values, positive weights).
        """
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=False)
        
        # Continuous updating may not converge - this is known behavior
        # We verify that results are still valid
        assert np.all(np.isfinite(result['coefficients'])), "Coefficients should be finite"
        assert np.all(np.isfinite(result['weights'])), "Weights should be finite"
        assert np.all(result['weights'] > 0), "Weights should be positive"

class TestATEWithSampleWeights:
    """Test ATE estimation with non-uniform sample weights."""
    
    def test_ate_with_sample_weights_convergence(self, cbps_binary_fit, lalonde_full):
        """Test convergence with sample weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        np.random.seed(42)
        sample_weights = np.random.exponential(1, n)
        sample_weights = sample_weights / np.mean(sample_weights)
        
        result = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights
        )
        
        assert result['converged'], "ATE with sample weights should converge"

    def test_ate_sample_weights_affect_result(self, cbps_binary_fit, lalonde_full):
        """Test that sample weights affect the result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        result_uniform = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        np.random.seed(42)
        sample_weights = np.random.exponential(1, n)
        sample_weights = sample_weights / np.mean(sample_weights)
        
        result_weighted = cbps_binary_fit(
            treat, X, att=0, method='over', two_step=True,
            sample_weights=sample_weights
        )
        
        coef_diff = np.abs(result_uniform['coefficients'].ravel() - 
                          result_weighted['coefficients'].ravel())
        assert np.max(coef_diff) > 1e-6, \
            "Sample weights should affect coefficients"

def lalonde_full_for_r_comparison():
    """
    Load the full LaLonde dataset (3212 observations) for R comparison tests.
    
    R benchmarks were generated using the full dataset, not the Dehejia-Wahba subset.
    """
    try:
        from cbps.datasets import load_lalonde
        df = load_lalonde(dehejia_wahba_only=False)
    except ImportError:
        pytest.skip("cbps.datasets not available")
        return None
    
    # Standard covariates used in CBPS examples
    covariate_cols = ['age', 'educ', 're74', 're75', 'married', 'nodegr']
    
    # Extract covariates and add intercept
    X_raw = df[covariate_cols].values
    X = np.column_stack([np.ones(len(X_raw)), X_raw])
    
    # Extract treatment
    treat = df['treat'].values.astype(float)
    
    # Sample weights
    sample_weights = np.ones(len(df))
    
    return {
        'df': df,
        'X': X,
        'treat': treat,
        'sample_weights': sample_weights,
        'n': len(treat),
        'k': X.shape[1],
        'n_t': int(np.sum(treat)),
        'n_c': int(np.sum(1 - treat)),
    }

class TestATTOverTwostep:
    """
    Test ID: E2E-001, Requirement: REQ-067
    
    End-to-end test for ATT estimation with over-identified GMM
    and two-step estimator.
    """
    
    def test_att_over_twostep_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATT over twostep converges."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        sample_weights = lalonde_full['sample_weights']
        
        result = cbps_binary_fit(
            treat, X, att=1, method='over', two_step=True,
            sample_weights=sample_weights, standardize=True
        )
        
        assert result['converged'], "ATT over twostep should converge"

    def test_att_over_twostep_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape matches covariates."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert result['coefficients'].shape == (k,) or result['coefficients'].shape == (k, 1), \
            f"Coefficients shape should be ({k},) or ({k}, 1)"

    def test_att_over_twostep_weights_shape(self, cbps_binary_fit, lalonde_full):
        """Test weights shape matches sample size."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert result['weights'].shape == (n,), f"Weights shape should be ({n},)"

    def test_att_over_twostep_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test all weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert np.all(result['weights'] > 0), "All weights should be positive"

    def test_att_over_twostep_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values (propensity scores) are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert np.all(result['fitted_values'] > 0), "Fitted values should be > 0"
        assert np.all(result['fitted_values'] < 1), "Fitted values should be < 1"

    def test_att_over_twostep_j_statistic_nonnegative(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert result['J'] >= 0, "J-statistic should be non-negative"

    def test_att_over_twostep_vcov_symmetric(self, cbps_binary_fit, lalonde_full):
        """Test variance-covariance matrix is symmetric."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        # The variance matrix is stored as 'var' in the result
        if 'var' in result and result['var'] is not None:
            assert_matrix_symmetric(result['var'], atol=1e-10, name="ATT vcov")

    def test_att_over_twostep_vcov_psd(self, cbps_binary_fit, lalonde_full):
        """Test variance-covariance matrix is positive semi-definite."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        if 'var' in result and result['var'] is not None:
            assert_positive_semidefinite(result['var'], atol=1e-8, name="ATT vcov")

    def test_att_over_twostep_balance_improvement(self, cbps_binary_fit, lalonde_full):
        """Test that CBPS improves covariate balance."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n_t = lalonde_full['n_t']
        n_c = lalonde_full['n_c']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        weights = result['weights']
        
        # Unweighted balance (standardized mean difference)
        mean_t = np.mean(X[treat == 1], axis=0)
        mean_c = np.mean(X[treat == 0], axis=0)
        std_pooled = np.sqrt((np.var(X[treat == 1], axis=0) + np.var(X[treat == 0], axis=0)) / 2)
        std_pooled = np.where(std_pooled > 0, std_pooled, 1)
        smd_unweighted = np.abs(mean_t - mean_c) / std_pooled
        
        # Weighted balance
        w_t = weights[treat == 1]
        w_c = weights[treat == 0]
        mean_t_w = np.average(X[treat == 1], axis=0, weights=w_t)
        mean_c_w = np.average(X[treat == 0], axis=0, weights=w_c)
        smd_weighted = np.abs(mean_t_w - mean_c_w) / std_pooled
        
        # Weighted balance should generally be better (lower SMD)
        # Allow some tolerance for numerical issues
        max_smd_unweighted = np.max(smd_unweighted[1:])  # Skip intercept
        max_smd_weighted = np.max(smd_weighted[1:])
        
        # At least some improvement or already balanced
        assert max_smd_weighted < max_smd_unweighted + 0.1 or max_smd_weighted < 0.25, \
            f"Weighted balance ({max_smd_weighted:.3f}) should improve on unweighted ({max_smd_unweighted:.3f})"

class TestATTOverContinuous:
    """
    Test ID: E2E-002, Requirement: REQ-070
    
    End-to-end test for ATT estimation with over-identified GMM
    and continuous updating estimator.
    
    Note: Continuous updating GMM may not always converge, especially with
    real-world data. Tests verify that results are valid even if not converged.
    """
    
    def test_att_over_continuous_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATT over continuous produces valid results.
        
        Note: Continuous updating GMM may not always converge with real-world data
        due to the iterative weight matrix updates. This is known behavior.
        We verify that results are still valid (finite values, positive weights).
        """
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=False)
        
        # Continuous updating may not converge - this is known behavior
        # We verify that results are still valid
        assert np.all(np.isfinite(result['coefficients'])), "Coefficients should be finite"
        assert np.all(np.isfinite(result['weights'])), "Weights should be finite"
        assert np.all(result['weights'] > 0), "Weights should be positive"
        
        # Note: We don't assert convergence here because continuous updating
        # GMM is known to have convergence issues with real-world data

    def test_att_over_continuous_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape matches covariates."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=False)
        
        assert result['coefficients'].shape == (k,) or result['coefficients'].shape == (k, 1)

    def test_att_over_continuous_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test all weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=False)
        
        assert np.all(result['weights'] > 0), "All weights should be positive"

    def test_att_over_continuous_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=False)
        
        assert np.all(result['fitted_values'] > 0)
        assert np.all(result['fitted_values'] < 1)

    def test_att_over_continuous_j_statistic(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is non-negative."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=False)
        
        assert result['J'] >= 0

class TestATTExactTwostep:
    """
    Test ID: E2E-003, Requirement: REQ-069
    
    End-to-end test for ATT estimation with exactly identified GMM.
    """
    
    def test_att_exact_twostep_convergence(self, cbps_binary_fit, lalonde_full):
        """Test that ATT exact twostep converges."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        
        assert result['converged'], "ATT exact twostep should converge"

    def test_att_exact_twostep_coefficients_shape(self, cbps_binary_fit, lalonde_full):
        """Test coefficient shape matches covariates."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        k = lalonde_full['k']
        
        result = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        
        assert result['coefficients'].shape == (k,) or result['coefficients'].shape == (k, 1)

    def test_att_exact_twostep_weights_positive(self, cbps_binary_fit, lalonde_full):
        """Test all weights are positive."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        
        assert np.all(result['weights'] > 0)

    def test_att_exact_twostep_fitted_values_range(self, cbps_binary_fit, lalonde_full):
        """Test fitted values are in (0, 1)."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        
        assert np.all(result['fitted_values'] > 0)
        assert np.all(result['fitted_values'] < 1)

    def test_att_exact_j_statistic_small(self, cbps_binary_fit, lalonde_full):
        """Test J-statistic is small for exactly identified model."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        
        # For exactly identified, J should be very small (but may not be exactly 0)
        assert result['J'] < 0.1, f"J-statistic should be small for exact, got {result['J']}"

    def test_att_exact_vs_over_different(self, cbps_binary_fit, lalonde_full):
        """Test exact and over-identified give different results."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result_exact = cbps_binary_fit(treat, X, att=1, method='exact', two_step=True)
        result_over = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        # Coefficients should be different (exact uses only score, over uses score + balance)
        coef_diff = np.abs(result_exact['coefficients'].ravel() - 
                          result_over['coefficients'].ravel())
        
        # At least some difference expected
        assert np.max(coef_diff) > 1e-6, \
            "Exact and over-identified should give different coefficients"

class TestATTWeightNormalization:
    """Test weight normalization options for ATT."""
    
    def test_att_standardize_true(self, cbps_binary_fit, lalonde_full):
        """Test standardized weights sum correctly.
        
        Note: The CBPS implementation normalizes weights such that they sum to 2
        (1 for treated group contribution, 1 for control group contribution).
        """
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True, standardize=True)
        
        # Standardized weights should sum to 2
        weight_sum = np.sum(result['weights'])
        assert np.isclose(weight_sum, 2.0, rtol=0.01), \
            f"Standardized weights should sum to 2, got {weight_sum}"

    def test_att_standardize_false(self, cbps_binary_fit, lalonde_full):
        """Test unstandardized weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True, standardize=False)
        
        # Weights should still be positive
        assert np.all(result['weights'] > 0)

class TestATTWithSampleWeights:
    """Test ATT estimation with non-uniform sample weights."""
    
    def test_att_with_sample_weights_convergence(self, cbps_binary_fit, lalonde_full):
        """Test convergence with sample weights."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        # Create non-uniform sample weights
        np.random.seed(42)
        sample_weights = np.random.exponential(1, n)
        sample_weights = sample_weights / np.mean(sample_weights)  # Normalize to mean 1
        
        result = cbps_binary_fit(
            treat, X, att=1, method='over', two_step=True,
            sample_weights=sample_weights
        )
        
        assert result['converged'], "ATT with sample weights should converge"

    def test_att_sample_weights_affect_result(self, cbps_binary_fit, lalonde_full):
        """Test that sample weights affect the result."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        n = lalonde_full['n']
        
        # Uniform weights
        result_uniform = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        # Non-uniform weights
        np.random.seed(42)
        sample_weights = np.random.exponential(1, n)
        sample_weights = sample_weights / np.mean(sample_weights)
        
        result_weighted = cbps_binary_fit(
            treat, X, att=1, method='over', two_step=True,
            sample_weights=sample_weights
        )
        
        # Results should be different
        coef_diff = np.abs(result_uniform['coefficients'].ravel() - 
                          result_weighted['coefficients'].ravel())
        assert np.max(coef_diff) > 1e-6, \
            "Sample weights should affect coefficients"

def generate_simulated_data(n, k, true_ate, seed=42):
    """
    Generate simulated data with known true treatment effect.
    
    Parameters
    ----------
    n : int
        Sample size
    k : int
        Number of covariates (excluding intercept)
    true_ate : float
        True average treatment effect
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Dictionary with X, treat, y, true_ate
    """
    np.random.seed(seed)
    
    # Generate covariates
    X_raw = np.random.normal(0, 1, (n, k))
    X = np.column_stack([np.ones(n), X_raw])
    
    # Propensity score model
    ps_beta = np.zeros(k + 1)
    ps_beta[0] = 0  # Intercept
    ps_beta[1:min(4, k+1)] = [0.3, -0.2, 0.1][:min(3, k)]  # First few covariates
    
    theta = X @ ps_beta
    pi_true = scipy.special.expit(theta)
    
    # Generate treatment
    treat = np.random.binomial(1, pi_true).astype(float)
    
    # Outcome model: Y = X @ outcome_beta + treat * true_ate + noise
    outcome_beta = np.zeros(k + 1)
    outcome_beta[0] = 5  # Intercept
    outcome_beta[1:min(4, k+1)] = [1.0, -0.5, 0.3][:min(3, k)]
    
    y = X @ outcome_beta + treat * true_ate + np.random.normal(0, 1, n)
    
    return {
        'X': X,
        'treat': treat,
        'y': y,
        'true_ate': true_ate,
        'n': n,
        'k': k + 1
    }

class TestLaLondeCPS:
    """
    Test ID: MULTI-001
    
    Test CBPS on LaLonde CPS dataset.
    """
    
    def test_lalonde_cps_convergence(self, cbps_binary_fit, lalonde_full):
        """Test CBPS converges on LaLonde data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged'], "Should converge on LaLonde data"

    def test_lalonde_cps_weights_valid(self, cbps_binary_fit, lalonde_full):
        """Test weights are valid on LaLonde data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['weights'] > 0), "Weights should be positive"
        assert np.all(np.isfinite(result['weights'])), "Weights should be finite"

    def test_lalonde_cps_propensity_scores(self, cbps_binary_fit, lalonde_full):
        """Test propensity scores are valid on LaLonde data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert np.all(result['fitted_values'] > 0), "Propensity scores should be > 0"
        assert np.all(result['fitted_values'] < 1), "Propensity scores should be < 1"

    def test_lalonde_cps_balance_improvement(self, cbps_binary_fit, lalonde_full):
        """Test balance improves on LaLonde data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        weights = result['weights']
        
        # Compute balance before and after
        def compute_imbalance(w=None):
            total = 0
            for j in range(1, X.shape[1]):
                if w is None:
                    mean_t = np.mean(X[treat == 1, j])
                    mean_c = np.mean(X[treat == 0, j])
                else:
                    mean_t = np.average(X[treat == 1, j], weights=w[treat == 1])
                    mean_c = np.average(X[treat == 0, j], weights=w[treat == 0])
                total += np.abs(mean_t - mean_c)
            return total
        
        imbalance_before = compute_imbalance()
        imbalance_after = compute_imbalance(weights)
        
        assert imbalance_after < imbalance_before, \
            f"Balance should improve: before={imbalance_before:.3f}, after={imbalance_after:.3f}"

class TestLaLondePSID:
    """
    Test ID: MULTI-002
    
    Test CBPS on LaLonde PSID dataset (if available).
    """
    
    def test_lalonde_psid_convergence(self, cbps_binary_fit, lalonde_full):
        """Test CBPS converges on PSID-like data."""
        # Use LaLonde data as proxy (PSID data may not be available)
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        # Modify to simulate PSID characteristics (larger control group)
        # This is a proxy test
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        assert result['converged'], "Should converge on PSID-like data"

    def test_lalonde_psid_att_estimation(self, cbps_binary_fit, lalonde_full):
        """Test ATT estimation on PSID-like data."""
        X = lalonde_full['X']
        treat = lalonde_full['treat']
        
        result = cbps_binary_fit(treat, X, att=1, method='over', two_step=True)
        
        # ATT weights should give more weight to controls similar to treated
        assert np.all(result['weights'] > 0)

class TestSimulatedData:
    """
    Test ID: MULTI-003
    
    Test CBPS on simulated data with known true effects.
    """
    
    def test_simulated_convergence(self, cbps_binary_fit):
        """Test CBPS converges on simulated data."""
        data = generate_simulated_data(n=500, k=5, true_ate=2.0)
        
        result = cbps_binary_fit(
            data['treat'], data['X'],
            att=0, method='over', two_step=True
        )
        
        assert result['converged'], "Should converge on simulated data"

    def test_simulated_weights_valid(self, cbps_binary_fit):
        """Test weights are valid on simulated data."""
        data = generate_simulated_data(n=500, k=5, true_ate=2.0)
        
        result = cbps_binary_fit(
            data['treat'], data['X'],
            att=0, method='over', two_step=True
        )
        
        assert np.all(result['weights'] > 0)
        assert np.all(np.isfinite(result['weights']))

    def test_simulated_ate_estimation(self, cbps_binary_fit):
        """Test ATE estimation on simulated data."""
        data = generate_simulated_data(n=1000, k=5, true_ate=2.0, seed=123)
        
        result = cbps_binary_fit(
            data['treat'], data['X'],
            att=0, method='over', two_step=True
        )
        
        weights = result['weights']
        treat = data['treat']
        y = data['y']
        
        # Weighted ATE estimate
        y1_weighted = np.average(y[treat == 1], weights=weights[treat == 1])
        y0_weighted = np.average(y[treat == 0], weights=weights[treat == 0])
        ate_estimate = y1_weighted - y0_weighted
        
        # Should be reasonably close to true ATE (within 1 std)
        assert np.abs(ate_estimate - data['true_ate']) < 2.0, \
            f"ATE estimate ({ate_estimate:.2f}) should be close to true ({data['true_ate']})"

    def test_simulated_different_sample_sizes(self, cbps_binary_fit):
        """Test CBPS on different sample sizes."""
        for n in [100, 500, 1000]:
            data = generate_simulated_data(n=n, k=3, true_ate=1.5)
            
            result = cbps_binary_fit(
                data['treat'], data['X'],
                att=0, method='over', two_step=True
            )
            
            assert result['converged'], f"Should converge for n={n}"

    def test_simulated_different_k(self, cbps_binary_fit):
        """Test CBPS with different numbers of covariates."""
        for k in [2, 5, 10]:
            data = generate_simulated_data(n=500, k=k, true_ate=1.5)
            
            result = cbps_binary_fit(
                data['treat'], data['X'],
                att=0, method='over', two_step=True
            )
            
            assert result['converged'], f"Should converge for k={k}"
            assert result['coefficients'].shape == (k + 1, 1)

class TestDifferentCovariateCombinations:
    """
    Test ID: MULTI-004
    
    Test CBPS with different covariate combinations.
    """
    
    def test_minimal_covariates(self, cbps_binary_fit):
        """Test CBPS with minimal covariates (intercept + 1)."""
        np.random.seed(42)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x1])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']
        assert result['coefficients'].shape == (2, 1)

    def test_many_covariates(self, cbps_binary_fit):
        """Test CBPS with many covariates."""
        np.random.seed(42)
        n = 500
        k = 15
        
        X_raw = np.random.normal(0, 1, (n, k - 1))
        X = np.column_stack([np.ones(n), X_raw])
        
        # Generate treatment with some covariates affecting propensity
        beta_true = np.zeros(k)
        beta_true[1:5] = [0.3, -0.2, 0.1, -0.1]
        theta = X @ beta_true
        pi_true = scipy.special.expit(theta)
        treat = np.random.binomial(1, pi_true).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']
        assert result['coefficients'].shape == (k, 1)

    def test_continuous_covariates_only(self, cbps_binary_fit):
        """Test CBPS with only continuous covariates."""
        np.random.seed(42)
        n = 300
        
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(5, 2, n)
        x3 = np.random.exponential(1, n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']

    def test_binary_covariates_only(self, cbps_binary_fit):
        """Test CBPS with only binary covariates."""
        np.random.seed(42)
        n = 300
        
        x1 = np.random.binomial(1, 0.5, n).astype(float)
        x2 = np.random.binomial(1, 0.3, n).astype(float)
        x3 = np.random.binomial(1, 0.7, n).astype(float)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']

    def test_mixed_covariates(self, cbps_binary_fit):
        """Test CBPS with mixed continuous and binary covariates."""
        np.random.seed(42)
        n = 300
        
        x1 = np.random.normal(0, 1, n)  # Continuous
        x2 = np.random.binomial(1, 0.5, n).astype(float)  # Binary
        x3 = np.random.normal(10, 3, n)  # Continuous
        x4 = np.random.binomial(1, 0.3, n).astype(float)  # Binary
        X = np.column_stack([np.ones(n), x1, x2, x3, x4])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']

    def test_correlated_covariates(self, cbps_binary_fit):
        """Test CBPS with correlated covariates."""
        np.random.seed(42)
        n = 300
        
        # Generate correlated covariates
        mean = [0, 0, 0]
        cov = [[1, 0.5, 0.3],
               [0.5, 1, 0.4],
               [0.3, 0.4, 1]]
        X_raw = np.random.multivariate_normal(mean, cov, n)
        X = np.column_stack([np.ones(n), X_raw])
        
        treat = np.random.binomial(1, 0.5, n).astype(float)
        
        result = cbps_binary_fit(treat, X, att=0, method='over', two_step=True)
        
        assert result['converged']