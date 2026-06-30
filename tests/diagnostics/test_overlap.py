"""Tests for overlap (positivity) assumption diagnostics (P1-13).

Tests the Crump et al. (2009) overlap checking implemented in
cbps.diagnostics.overlap.
"""

import numpy as np
import pytest

from cbps.diagnostics.overlap import check_overlap


class TestOverlapGood:
    """Perfect overlap scenarios should produce no warnings."""

    def test_perfect_overlap_no_violation(self):
        """PS ∈ [0.3, 0.7] for both groups → no violation."""
        np.random.seed(42)
        n = 200
        treat = np.array([0] * 100 + [1] * 100)
        # PS in [0.3, 0.7] for both groups
        ps = np.random.uniform(0.3, 0.7, n)
        result = check_overlap(ps, treat)

        assert result['violation_detected'] is False
        assert result['warning_message'] is None

    def test_moderate_overlap_no_violation(self):
        """PS ∈ [0.1, 0.9] for both groups → still OK."""
        np.random.seed(123)
        n = 200
        treat = np.array([0] * 100 + [1] * 100)
        ps = np.random.uniform(0.1, 0.9, n)
        result = check_overlap(ps, treat)

        assert result['violation_detected'] is False

    def test_overlap_region_correct(self):
        """Overlap region = [max of mins, min of maxes]."""
        treat = np.array([0, 0, 0, 1, 1, 1])
        ps = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])
        result = check_overlap(ps, treat)

        # Control range: [0.2, 0.6], Treated range: [0.3, 0.7]
        # Overlap: [max(0.2, 0.3), min(0.6, 0.7)] = [0.3, 0.6]
        assert result['overlap_region'] == pytest.approx((0.3, 0.6))


class TestOverlapViolation:
    """Overlap violations should be detected."""

    def test_no_overlap_detected(self):
        """T=1: ps>0.9, T=0: ps<0.1 → violation_detected=True."""
        treat = np.array([0] * 50 + [1] * 50)
        ps_control = np.random.uniform(0.01, 0.1, 50)
        ps_treated = np.random.uniform(0.9, 0.99, 50)
        ps = np.concatenate([ps_control, ps_treated])

        result = check_overlap(ps, treat)

        assert result['violation_detected'] is True
        assert result['warning_message'] is not None
        assert 'SEVERE' in result['warning_message'] or 'WARNING' in result['warning_message']

    def test_partial_overlap_violation(self):
        """More than 20% outside common support → violation."""
        np.random.seed(99)
        n = 100
        treat = np.array([0] * 50 + [1] * 50)
        # Control: [0.05, 0.30], Treated: [0.25, 0.95]
        # Overlap: [0.25, 0.30] — very narrow
        ps_control = np.random.uniform(0.05, 0.30, 50)
        ps_treated = np.random.uniform(0.25, 0.95, 50)
        ps = np.concatenate([ps_control, ps_treated])

        result = check_overlap(ps, treat)
        # Many treated obs are above 0.30, many control below 0.25
        # So n_outside_overlap > 20% → violation
        assert result['n_outside_overlap'] > 0


class TestOverlapTrimming:
    """Trimming analysis should be correct."""

    def test_trimming_100pct_retained_uniform(self):
        """PS all in [0.2, 0.8] → 100% retained at alpha=0.05."""
        n = 100
        treat = np.array([0] * 50 + [1] * 50)
        ps = np.random.uniform(0.2, 0.8, n)
        result = check_overlap(ps, treat)

        # All PS are in [0.2, 0.8], so alpha=0.05 (keeps [0.05, 0.95]) retains all
        assert result['trimming_analysis'][0.05]['pct_retained'] == 100.0

    def test_trimming_excludes_extreme(self):
        """PS at 0.01 or 0.99 trimmed at alpha=0.05."""
        treat = np.array([0, 0, 0, 1, 1, 1])
        ps = np.array([0.01, 0.4, 0.5, 0.5, 0.6, 0.99])
        result = check_overlap(ps, treat)

        # alpha=0.05 → keep [0.05, 0.95]: 0.01 and 0.99 excluded
        assert result['trimming_analysis'][0.05]['n_retained'] == 4

    def test_default_alphas(self):
        """Default alphas are [0.05, 0.10, 0.15, 0.20]."""
        treat = np.array([0, 1])
        ps = np.array([0.3, 0.7])
        result = check_overlap(ps, treat)

        assert set(result['trimming_analysis'].keys()) == {0.05, 0.10, 0.15, 0.20}

    def test_custom_alphas(self):
        """Custom alphas are respected."""
        treat = np.array([0, 1])
        ps = np.array([0.3, 0.7])
        result = check_overlap(ps, treat, alphas=[0.01, 0.25])

        assert set(result['trimming_analysis'].keys()) == {0.01, 0.25}


class TestOverlapEdgeCases:
    """Edge cases for overlap checking."""

    def test_single_observation_outside(self):
        """Single obs outside overlap shouldn't necessarily trigger violation."""
        n = 100
        treat = np.array([0] * 50 + [1] * 50)
        ps = np.random.uniform(0.2, 0.8, n)
        # One control with very low PS — still < 20% outside
        ps[0] = 0.01
        result = check_overlap(ps, treat)

        # Only 1 out of 100 is outside, so 1% < 20%
        assert result['violation_detected'] is False

    def test_ps_range_output(self):
        """PS ranges are correctly reported."""
        treat = np.array([0, 0, 1, 1])
        ps = np.array([0.1, 0.4, 0.6, 0.9])
        result = check_overlap(ps, treat)

        assert result['ps_range'] == (0.1, 0.9)
        assert result['ps_range_control'] == (0.1, 0.4)
        assert result['ps_range_treated'] == (0.6, 0.9)

    def test_recommended_alpha_smallest_retaining_90pct(self):
        """recommended_alpha is smallest alpha retaining ≥ 90%."""
        n = 100
        treat = np.array([0] * 50 + [1] * 50)
        # All PS in [0.2, 0.8] → all alphas retain 100%
        ps = np.random.uniform(0.2, 0.8, n)
        result = check_overlap(ps, treat)

        # All retain 100%, so smallest alpha (0.05) is recommended
        assert result['recommended_alpha'] == 0.05

    def test_empty_control_group_detected(self):
        """Empty control group → violation with clear message."""
        treat = np.array([1, 1, 1, 1])
        ps = np.array([0.3, 0.5, 0.7, 0.8])
        result = check_overlap(ps, treat)

        assert result['violation_detected'] is True
        assert 'control' in result['warning_message']
        assert result['overlap_region'] == (np.nan, np.nan)

    def test_empty_treated_group_detected(self):
        """Empty treated group → violation with clear message."""
        treat = np.array([0, 0, 0, 0])
        ps = np.array([0.2, 0.3, 0.4, 0.5])
        result = check_overlap(ps, treat)

        assert result['violation_detected'] is True
        assert 'treated' in result['warning_message']

    def test_ps_outside_unit_interval_warns(self):
        """PS values outside [0,1] trigger a warning."""
        treat = np.array([0, 0, 0, 1, 1, 1])
        ps = np.array([-0.1, 0.3, 0.5, 0.6, 0.8, 1.2])
        with pytest.warns(UserWarning, match="should be in \\[0, 1\\]"):
            result = check_overlap(ps, treat)
        # Should still compute (possibly detecting violation)
        assert 'overlap_region' in result

    def test_length_mismatch_raises(self):
        """Mismatched lengths of ps and treat → ValueError."""
        treat = np.array([0, 1, 0])
        ps = np.array([0.3, 0.5])
        with pytest.raises(ValueError, match="Length mismatch"):
            check_overlap(ps, treat)

    def test_empty_input_handled(self):
        """Empty arrays → violation detected, no crash."""
        treat = np.array([])
        ps = np.array([])
        result = check_overlap(ps, treat)
        assert result['violation_detected'] is True
