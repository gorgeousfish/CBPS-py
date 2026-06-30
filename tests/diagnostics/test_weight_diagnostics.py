"""Tests for weight quality diagnostics (P1-11).

Tests the Kish (1965) ESS computation and weight quality indicators
implemented in cbps.diagnostics.weights_diag.
"""

import numpy as np
import pytest

from cbps.diagnostics.weights_diag import weight_diagnostics


class TestWeightDiagnosticsUniform:
    """Uniform weights should yield perfect diagnostics."""

    def test_uniform_weights_ess_equals_n(self):
        """Uniform weights → ESS = n, ratio = 1.0."""
        n = 100
        weights = np.ones(n)
        result = weight_diagnostics(weights)

        assert result['ess'] == pytest.approx(n, rel=1e-10)
        assert result['ess_ratio'] == pytest.approx(1.0, rel=1e-10)

    def test_uniform_weights_no_warning(self):
        """Uniform weights → warning_level = 'ok'."""
        weights = np.ones(50)
        result = weight_diagnostics(weights)
        assert result['warning_level'] == 'ok'

    def test_uniform_weights_ratio_is_one(self):
        """Uniform weights → max/min ratio = 1."""
        weights = np.ones(30)
        result = weight_diagnostics(weights)
        assert result['weight_ratio'] == pytest.approx(1.0)

    def test_uniform_weights_no_extreme(self):
        """Uniform weights → no extreme weights."""
        weights = np.ones(50)
        result = weight_diagnostics(weights)
        assert result['n_extreme'] == 0

    def test_uniform_weights_cv_zero(self):
        """Uniform weights → CV = 0."""
        weights = np.ones(50)
        result = weight_diagnostics(weights)
        assert result['cv'] == pytest.approx(0.0, abs=1e-15)


class TestWeightDiagnosticsExtreme:
    """Extreme weights should trigger warnings."""

    def test_one_extreme_weight_reduces_ess(self):
        """One extreme weight → ESS < n."""
        n = 100
        weights = np.ones(n)
        weights[0] = 1000.0  # One extreme weight
        result = weight_diagnostics(weights)

        assert result['ess'] < n
        assert result['n_extreme'] > 0

    def test_extreme_weights_caution(self):
        """Moderately variable weights → 'caution'."""
        # ESS/n should be between 0.2 and 0.5
        np.random.seed(42)
        # Create weights with ESS/n ~ 0.3
        weights = np.ones(100)
        weights[:30] = 10.0  # Make 30% much heavier
        result = weight_diagnostics(weights)
        # ESS = (sum_w)^2 / sum_w^2
        # sum_w = 30*10 + 70*1 = 370
        # sum_w^2 = 30*100 + 70*1 = 3070
        # ESS = 370^2 / 3070 = 136900 / 3070 = 44.59
        # ESS/n = 44.59/100 = 0.4459 → caution
        assert result['warning_level'] == 'caution'

    def test_extreme_weights_severe(self):
        """Very extreme weights → 'severe'."""
        weights = np.ones(100)
        weights[0] = 10000.0  # Single dominant weight
        result = weight_diagnostics(weights)

        # ESS = (sum_w)^2 / sum_w^2
        # sum_w = 10099, sum_w^2 = 10^8 + 99 ≈ 10^8
        # ESS ≈ 10099^2/10^8 ≈ 1.02 → ESS/n ≈ 0.01
        assert result['warning_level'] == 'severe'
        assert result['ess_ratio'] < 0.2


class TestWeightDiagnosticsEdgeCases:
    """Edge cases should not crash."""

    def test_all_zero_weights(self):
        """All-zero weights → handles gracefully."""
        weights = np.zeros(50)
        result = weight_diagnostics(weights)

        assert result['ess'] == 0.0
        assert result['ess_ratio'] == 0.0
        assert result['warning_level'] == 'severe'

    def test_empty_weights(self):
        """Empty array → handles gracefully."""
        weights = np.array([])
        result = weight_diagnostics(weights)

        assert result['ess'] == 0.0
        assert result['warning_level'] == 'severe'

    def test_single_weight(self):
        """Single observation → ESS = 1."""
        weights = np.array([5.0])
        result = weight_diagnostics(weights)
        assert result['ess'] == pytest.approx(1.0)
        assert result['ess_ratio'] == pytest.approx(1.0)

    def test_negative_weights_handled(self):
        """Negative weights don't crash; ESS computed on |w|, n_negative reported."""
        weights = np.array([-1.0, 2.0, 3.0, -0.5, 4.0])
        with pytest.warns(UserWarning, match="negative weight"):
            result = weight_diagnostics(weights)
        assert 'ess' in result
        assert result['n_negative'] == 2
        # ESS on |w|: [1, 2, 3, 0.5, 4], sum=10.5, sum_sq=30.25
        # ESS = 10.5^2 / 30.25 = 110.25/30.25 = 3.645...
        expected_ess = 10.5**2 / 30.25
        assert result['ess'] == pytest.approx(expected_ess, rel=1e-10)

    def test_negative_weights_max_captures_absolute(self):
        """weight_max captures largest absolute weight including negatives."""
        weights = np.array([-5.0, 1.0, 2.0, 3.0])
        with pytest.warns(UserWarning):
            result = weight_diagnostics(weights)
        assert result['weight_max'] == pytest.approx(5.0)

    def test_all_positive_no_warning(self):
        """All positive weights do not trigger negative weight warning."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = weight_diagnostics(weights)
        assert result['n_negative'] == 0


class TestWeightDiagnosticsGrouped:
    """Group-specific diagnostics with treatment indicator."""

    def test_group_diagnostics_returned(self):
        """When treat is provided, group diagnostics are computed."""
        weights = np.ones(100)
        treat = np.array([0] * 50 + [1] * 50)
        result = weight_diagnostics(weights, treat=treat)

        assert result['group_diagnostics'] is not None
        assert 0 in result['group_diagnostics']
        assert 1 in result['group_diagnostics']

    def test_group_ess_equals_group_n_for_uniform(self):
        """Uniform weights → group ESS = group n."""
        weights = np.ones(100)
        treat = np.array([0] * 60 + [1] * 40)
        result = weight_diagnostics(weights, treat=treat)

        assert result['group_diagnostics'][0]['ess'] == pytest.approx(60.0)
        assert result['group_diagnostics'][1]['ess'] == pytest.approx(40.0)

    def test_no_treat_no_group_diagnostics(self):
        """Without treat, group_diagnostics is None."""
        weights = np.ones(50)
        result = weight_diagnostics(weights)
        assert result['group_diagnostics'] is None
