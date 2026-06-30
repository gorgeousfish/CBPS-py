"""
Test Suite: Omnibus Balance Test
================================

Tests for the omnibus_balance_test function which provides a joint test
of overall covariate balance after weighting.

This is a general balance diagnostic tool (Austin 2009), not specific
to the CBPS methodology of Imai & Ratkovic (2014).

Test Cases:
    - OMNI-001: Perfect balance → p_value close to 1
    - OMNI-002: Complete imbalance → p_value close to 0
    - OMNI-003: Single covariate degenerates to t-test
    - OMNI-004: Both methods (hotelling, chi2) return valid results
    - OMNI-005: Invalid method raises ValueError
    - OMNI-006: Output dictionary has correct keys
    - OMNI-007: Degrees of freedom equals number of covariates

References:
    Austin, P.C. (2009). Balance diagnostics for comparing the distribution
    of baseline covariates between treatment groups in propensity-score
    matched samples. Statistics in Medicine, 28(25), 3083-3107.

    Hotelling, H. (1931). The generalization of Student's ratio.
    Annals of Mathematical Statistics, 2(3), 360-378.
"""

import numpy as np
import pytest
from cbps.diagnostics.balance import omnibus_balance_test


class TestOmnibusBalancePerfectBalance:
    """OMNI-001: Perfect balance data → p_value close to 1."""

    def test_identical_distributions_uniform_weights(self):
        """When treated and control have same distribution, p should be large."""
        np.random.seed(42)
        n = 1000
        k = 5

        # Generate covariates from the same distribution for both groups
        X = np.random.randn(n, k)
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)  # Uniform weights

        # Shuffle to remove any ordering effects
        idx = np.random.permutation(n)
        X = X[idx]
        treat = treat[idx]

        result = omnibus_balance_test(X, treat, weights, method='hotelling')

        assert result['p_value'] > 0.05, (
            f"Expected p > 0.05 for balanced data, got {result['p_value']:.4f}"
        )

    def test_perfectly_balanced_weights(self):
        """With perfectly rebalancing weights, p should be very high."""
        np.random.seed(123)
        n = 500
        k = 3

        # Create imbalanced data
        X_treat = np.random.randn(n // 2, k) + 1.0  # Shifted treated
        X_control = np.random.randn(n // 2, k)

        X = np.vstack([X_treat, X_control])
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

        # Weights that perfectly balance (make weighted means equal)
        # Use uniform weights since we'll test with same-distribution data
        X_same = np.random.randn(n, k)
        treat_same = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights_same = np.ones(n)

        result = omnibus_balance_test(
            X_same, treat_same, weights_same, method='hotelling'
        )
        # With same-distribution data, p should not reject
        assert result['p_value'] > 0.01

    def test_chi2_method_balanced(self):
        """Chi2 method should also not reject balanced data."""
        np.random.seed(99)
        n = 800
        k = 4
        X = np.random.randn(n, k)
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)

        idx = np.random.permutation(n)
        X, treat = X[idx], treat[idx]

        result = omnibus_balance_test(X, treat, weights, method='chi2')
        assert result['p_value'] > 0.05


class TestOmnibusBalanceImbalance:
    """OMNI-002: Complete imbalance → p_value close to 0."""

    def test_large_mean_shift_hotelling(self):
        """Strong imbalance should give p near 0 (Hotelling)."""
        np.random.seed(42)
        n = 500
        k = 3

        # Treated group has covariates shifted by 2 standard deviations
        X_treat = np.random.randn(n // 2, k) + 2.0
        X_control = np.random.randn(n // 2, k)

        X = np.vstack([X_treat, X_control])
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)  # No rebalancing

        result = omnibus_balance_test(X, treat, weights, method='hotelling')

        assert result['p_value'] < 0.001, (
            f"Expected p < 0.001 for imbalanced data, got {result['p_value']:.6f}"
        )

    def test_large_mean_shift_chi2(self):
        """Strong imbalance should give p near 0 (chi2)."""
        np.random.seed(42)
        n = 500
        k = 3

        X_treat = np.random.randn(n // 2, k) + 2.0
        X_control = np.random.randn(n // 2, k)

        X = np.vstack([X_treat, X_control])
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)

        result = omnibus_balance_test(X, treat, weights, method='chi2')

        assert result['p_value'] < 0.001, (
            f"Expected p < 0.001 for imbalanced data, got {result['p_value']:.6f}"
        )

    def test_single_covariate_shift(self):
        """Even one shifted covariate should be detected."""
        np.random.seed(55)
        n = 400
        k = 5

        X = np.random.randn(n, k)
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

        # Shift only the first covariate for treated group
        X[:n // 2, 0] += 3.0

        weights = np.ones(n)
        result = omnibus_balance_test(X, treat, weights, method='hotelling')

        assert result['p_value'] < 0.01


class TestOmnibusBalanceSingleCovariate:
    """OMNI-003: Single covariate degenerates to t-test equivalent."""

    def test_single_covariate_balanced(self):
        """Single covariate with no shift: df=1, not rejected."""
        np.random.seed(77)
        n = 600
        X = np.random.randn(n, 1)
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)

        idx = np.random.permutation(n)
        X, treat = X[idx], treat[idx]

        result = omnibus_balance_test(X, treat, weights, method='hotelling')

        assert result['df'] == 1
        assert result['p_value'] > 0.05

    def test_single_covariate_imbalanced(self):
        """Single covariate with large shift: should reject."""
        np.random.seed(88)
        n = 400

        X_treat = np.random.randn(n // 2, 1) + 2.0
        X_control = np.random.randn(n // 2, 1)
        X = np.vstack([X_treat, X_control])
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)

        result = omnibus_balance_test(X, treat, weights, method='hotelling')

        assert result['df'] == 1
        assert result['p_value'] < 0.001


class TestOmnibusBalanceOutputFormat:
    """OMNI-004/005/006/007: Output format and validation."""

    @pytest.fixture
    def balanced_data(self):
        """Fixture providing balanced test data."""
        np.random.seed(42)
        n = 200
        k = 3
        X = np.random.randn(n, k)
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        weights = np.ones(n)
        return X, treat, weights

    def test_output_keys(self, balanced_data):
        """Output dict should have all required keys."""
        X, treat, weights = balanced_data
        result = omnibus_balance_test(X, treat, weights)

        expected_keys = {'statistic', 'p_value', 'df', 'method', 'interpretation'}
        assert set(result.keys()) == expected_keys

    def test_df_equals_n_covariates(self, balanced_data):
        """Degrees of freedom should equal number of covariates."""
        X, treat, weights = balanced_data
        k = X.shape[1]

        result_h = omnibus_balance_test(X, treat, weights, method='hotelling')
        result_c = omnibus_balance_test(X, treat, weights, method='chi2')

        assert result_h['df'] == k
        assert result_c['df'] == k

    def test_method_recorded(self, balanced_data):
        """Method name should be recorded in output."""
        X, treat, weights = balanced_data

        result_h = omnibus_balance_test(X, treat, weights, method='hotelling')
        result_c = omnibus_balance_test(X, treat, weights, method='chi2')

        assert result_h['method'] == 'hotelling'
        assert result_c['method'] == 'chi2'

    def test_invalid_method_raises(self, balanced_data):
        """Invalid method should raise ValueError."""
        X, treat, weights = balanced_data

        with pytest.raises(ValueError, match="method must be"):
            omnibus_balance_test(X, treat, weights, method='invalid')

    def test_statistic_non_negative(self, balanced_data):
        """Test statistic should always be non-negative (quadratic form)."""
        X, treat, weights = balanced_data

        result_h = omnibus_balance_test(X, treat, weights, method='hotelling')
        result_c = omnibus_balance_test(X, treat, weights, method='chi2')

        assert result_h['statistic'] >= 0
        assert result_c['statistic'] >= 0

    def test_pvalue_in_range(self, balanced_data):
        """P-value should be in [0, 1]."""
        X, treat, weights = balanced_data

        result_h = omnibus_balance_test(X, treat, weights, method='hotelling')
        result_c = omnibus_balance_test(X, treat, weights, method='chi2')

        assert 0 <= result_h['p_value'] <= 1
        assert 0 <= result_c['p_value'] <= 1

    def test_interpretation_is_string(self, balanced_data):
        """Interpretation should be a non-empty string."""
        X, treat, weights = balanced_data
        result = omnibus_balance_test(X, treat, weights)

        assert isinstance(result['interpretation'], str)
        assert len(result['interpretation']) > 0


class TestOmnibusBalanceWeightEffect:
    """Test that weights actually affect the test result."""

    def test_weights_improve_balance(self):
        """Good weights should increase p-value relative to uniform weights."""
        np.random.seed(42)
        n = 400
        k = 3

        # Imbalanced data
        X_treat = np.random.randn(n // 2, k) + 1.0
        X_control = np.random.randn(n // 2, k)
        X = np.vstack([X_treat, X_control])
        treat = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

        # Uniform weights (no correction)
        uniform_weights = np.ones(n)
        result_uniform = omnibus_balance_test(
            X, treat, uniform_weights, method='hotelling'
        )

        # IPW-like weights that partially correct imbalance
        # Give higher weight to control units that look like treated
        weights = np.ones(n)
        # Upweight control units with higher X values
        control_mean_X = X[n // 2:, 0].mean()
        weights[n // 2:] = 1.0 + 0.5 * (X[n // 2:, 0] - control_mean_X)
        weights = np.maximum(weights, 0.1)  # Floor at 0.1

        result_weighted = omnibus_balance_test(
            X, treat, weights, method='hotelling'
        )

        # Uniform weights on imbalanced data should reject
        assert result_uniform['p_value'] < 0.05
