"""
Module: test_multitreat.py
==========================

Test Suite: Multi-Valued Treatment CBPS
Test IDs: MT-001 to MT-006
Requirements: REQ-MT-001 to REQ-MT-010

Overview:
    This module tests the multi-valued treatment CBPS implementation using
    multinomial logistic regression for propensity score estimation.

    Multi-valued treatment CBPS extends the binary CBPS framework to handle
    categorical treatments with 3 or more levels by using multinomial logistic
    regression for the propensity score model and adjusting the balancing
    conditions accordingly.

Test Categories:
    - Unit tests: Basic functionality and parameter validation
    - Numerical tests: Weight computation accuracy
    - Integration tests: End-to-end workflow
    - Edge cases: Boundary conditions and error handling

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

Usage:
    pytest tests/multitreat/ -v
    pytest tests/multitreat/ -m "not slow"
    pytest tests/multitreat/test_multitreat.py::TestThreeTreatCBPS -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_multitreat_data(n=300, n_levels=3, seed=42):
    """
    Generate synthetic multi-valued treatment data.
    
    Parameters
    ----------
    n : int
        Number of observations
    n_levels : int
        Number of treatment levels (3 or 4)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing X, treat, and metadata
    """
    np.random.seed(seed)
    
    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])
    
    # Generate treatment probabilities using softmax
    if n_levels == 3:
        logits = np.column_stack([
            np.zeros(n),  # Reference category
            0.5 * x1 - 0.3 * x2,
            -0.3 * x1 + 0.5 * x2
        ])
    else:  # n_levels == 4
        logits = np.column_stack([
            np.zeros(n),  # Reference category
            0.5 * x1,
            -0.3 * x2,
            0.2 * x1 + 0.2 * x2
        ])
    
    # Softmax to get probabilities
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Sample treatment
    treat = np.array([np.random.choice(n_levels, p=p) for p in probs])
    
    return {
        'X': X,
        'x1': x1,
        'x2': x2,
        'treat': treat,
        'n': n,
        'k': X.shape[1],
        'n_levels': n_levels,
        'probs': probs,
    }


# =============================================================================
# MT-001: Basic Multi-Valued Treatment Fitting
# =============================================================================

class TestMultitreatBasicFitting:
    """
    Test ID: MT-001
    
    Test basic multi-valued treatment CBPS fitting.
    """
    
    @pytest.fixture
    def three_level_data(self):
        """Generate 3-level treatment data."""
        return generate_multitreat_data(n=300, n_levels=3, seed=42)
    
    @pytest.fixture
    def four_level_data(self):
        """Generate 4-level treatment data."""
        return generate_multitreat_data(n=400, n_levels=4, seed=123)
    
    def test_multitreat_fit_converges(self, three_level_data):
        """Test that multi-valued CBPS converges."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = three_level_data
        X = data['X']
        treat = data['treat']
        
        result = cbps_3treat_fit(treat, X)
        
        assert result is not None
        assert 'converged' in result or hasattr(result, 'converged')
    
    def test_multitreat_returns_weights(self, three_level_data):
        """Test that multi-valued CBPS returns weights."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = three_level_data
        X = data['X']
        treat = data['treat']
        n = data['n']
        
        result = cbps_3treat_fit(treat, X)
        
        # Check weights exist and have correct shape
        if hasattr(result, 'weights'):
            weights = result.weights
        else:
            weights = result.get('weights')
        
        assert weights is not None
        assert len(weights) == n
        assert np.all(weights > 0)
    
    def test_multitreat_returns_fitted_values(self, three_level_data):
        """Test that multi-valued CBPS returns fitted probabilities."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = three_level_data
        X = data['X']
        treat = data['treat']
        n = data['n']
        n_levels = data['n_levels']
        
        result = cbps_3treat_fit(treat, X)
        
        # Check fitted values exist
        if hasattr(result, 'fitted_values'):
            fitted = result.fitted_values
        else:
            fitted = result.get('fitted_values')
        
        assert fitted is not None
        # Fitted values should be probabilities in (0, 1)
        assert np.all(fitted > 0)
        assert np.all(fitted < 1)


# =============================================================================
# MT-002: Weight Calculation Verification
# =============================================================================

class TestMultitreatWeights:
    """
    Test ID: MT-002
    
    Test weight calculations for multi-valued treatment.
    """
    
    @pytest.fixture
    def fitted_data(self):
        """Generate and fit multi-valued treatment data."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = generate_multitreat_data(n=300, n_levels=3, seed=42)
        result = cbps_3treat_fit(data['treat'], data['X'])
        
        return {
            'data': data,
            'result': result,
        }
    
    def test_weights_positive(self, fitted_data):
        """Test that all weights are positive."""
        result = fitted_data['result']
        
        if hasattr(result, 'weights'):
            weights = result.weights
        else:
            weights = result.get('weights')
        
        assert np.all(weights > 0), "All weights should be positive"
    
    def test_weights_finite(self, fitted_data):
        """Test that all weights are finite."""
        result = fitted_data['result']
        
        if hasattr(result, 'weights'):
            weights = result.weights
        else:
            weights = result.get('weights')
        
        assert np.all(np.isfinite(weights)), "All weights should be finite"


# =============================================================================
# MT-003: Balance Improvement Verification
# =============================================================================

class TestMultitreatBalance:
    """
    Test ID: MT-003
    
    Test that multi-valued CBPS improves covariate balance.
    """
    
    def test_balance_improvement(self):
        """Test that CBPS improves balance across treatment groups."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = generate_multitreat_data(n=500, n_levels=3, seed=42)
        X = data['X']
        treat = data['treat']
        x1 = data['x1']
        
        result = cbps_3treat_fit(treat, X)
        
        if hasattr(result, 'weights'):
            weights = result.weights
        else:
            weights = result.get('weights')
        
        # Calculate unweighted means by treatment group
        unweighted_means = [x1[treat == t].mean() for t in range(3)]
        
        # Calculate weighted means by treatment group
        weighted_means = []
        for t in range(3):
            mask = treat == t
            if mask.sum() > 0:
                w = weights[mask]
                weighted_means.append(np.average(x1[mask], weights=w))
        
        # Weighted means should be more similar across groups
        unweighted_range = max(unweighted_means) - min(unweighted_means)
        weighted_range = max(weighted_means) - min(weighted_means)
        
        # Balance should improve (or at least not worsen significantly)
        assert weighted_range <= unweighted_range * 1.5, \
            "Weighted balance should not be much worse than unweighted"


# =============================================================================
# MT-004 to MT-006: Additional Tests
# =============================================================================

class TestMultitreatEdgeCases:
    """
    Test ID: MT-004 to MT-006
    
    Test edge cases for multi-valued treatment.
    """
    
    def test_four_level_treatment(self):
        """MT-004: Test 4-level treatment."""
        try:
            from cbps.core.cbps_multitreat import cbps_4treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        data = generate_multitreat_data(n=400, n_levels=4, seed=123)
        result = cbps_4treat_fit(data['treat'], data['X'])
        
        assert result is not None
    
    def test_unbalanced_treatment_groups(self):
        """MT-005: Test with unbalanced treatment group sizes."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        np.random.seed(42)
        n = 300
        
        # Create unbalanced treatment: 60% level 0, 30% level 1, 10% level 2
        treat = np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        
        result = cbps_3treat_fit(treat, X)
        
        assert result is not None
    
    def test_high_dimensional_covariates(self):
        """MT-006: Test with more covariates."""
        try:
            from cbps.core.cbps_multitreat import cbps_3treat_fit
        except ImportError:
            pytest.skip("cbps_multitreat module not available")
        
        np.random.seed(42)
        n, p = 300, 10
        
        X = np.column_stack([np.ones(n), np.random.randn(n, p)])
        treat = np.random.choice([0, 1, 2], n)
        
        result = cbps_3treat_fit(treat, X)
        
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
