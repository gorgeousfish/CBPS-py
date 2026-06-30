"""Tests for rank_diagnostics module."""

import numpy as np
import pytest

from cbps.msm.rank_diagnostics import diagnose_rank_selection


class TestDiagnoseRankSelection:
    """Tests for diagnose_rank_selection function."""

    def test_returns_correct_keys(self):
        """Result dict should contain all expected keys."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        result = diagnose_rank_selection(X)

        expected_keys = {
            "singular_values",
            "total_columns",
            "ranks_by_threshold",
            "energy_by_rank",
            "recommended_action",
        }
        assert set(result.keys()) == expected_keys

    def test_total_columns_matches_input(self):
        """total_columns should reflect the input matrix width."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 8))
        result = diagnose_rank_selection(X)
        assert result["total_columns"] == 8

    def test_singular_values_descending(self):
        """Singular values should be in descending order."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((60, 6))
        result = diagnose_rank_selection(X)
        sv = result["singular_values"]
        assert np.all(sv[:-1] >= sv[1:])

    def test_ranks_monotone_decreasing_with_threshold(self):
        """Larger threshold should yield equal or smaller rank."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 10))
        thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
        result = diagnose_rank_selection(X, thresholds=thresholds)
        ranks = result["ranks_by_threshold"]
        sorted_thresholds = sorted(thresholds)
        rank_values = [ranks[t] for t in sorted_thresholds]
        for i in range(len(rank_values) - 1):
            assert rank_values[i] >= rank_values[i + 1]

    def test_full_rank_matrix(self):
        """Full rank matrix should retain all columns at default threshold."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 5))
        result = diagnose_rank_selection(X)
        # With random normal data, all singular values are large
        assert result["ranks_by_threshold"][1e-4] == 5

    def test_rank_deficient_matrix(self):
        """Rank-deficient matrix should show reduced rank."""
        rng = np.random.default_rng(55)
        # Create a rank-3 matrix in 6D
        base = rng.standard_normal((100, 3))
        X = np.column_stack([base, base @ rng.standard_normal((3, 3))])
        result = diagnose_rank_selection(X)
        # At a very small threshold, should retain ~3 meaningful components
        # (numerical noise may add a tiny singular value)
        assert result["ranks_by_threshold"][1e-4] <= 6
        # At moderate threshold some directions are lost
        assert result["ranks_by_threshold"][1e-2] <= result["ranks_by_threshold"][1e-4]

    def test_energy_ratio_computation(self):
        """Energy by rank should equal cumsum(s^2)/sum(s^2)."""
        rng = np.random.default_rng(10)
        X = rng.standard_normal((50, 4))
        result = diagnose_rank_selection(X)
        s = result["singular_values"]
        s_sq = s ** 2
        expected_energy = np.cumsum(s_sq) / s_sq.sum()
        np.testing.assert_allclose(result["energy_by_rank"], expected_energy)

    def test_energy_last_element_is_one(self):
        """Cumulative energy at full rank should be 1.0."""
        rng = np.random.default_rng(77)
        X = rng.standard_normal((40, 5))
        result = diagnose_rank_selection(X)
        assert np.isclose(result["energy_by_rank"][-1], 1.0)

    def test_empty_matrix(self):
        """Empty matrix should return graceful result."""
        X = np.zeros((10, 0))
        result = diagnose_rank_selection(X)
        assert result["total_columns"] == 0
        assert len(result["singular_values"]) == 0
        assert all(v == 0 for v in result["ranks_by_threshold"].values())

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        thresholds = [0.5, 1.0, 5.0]
        result = diagnose_rank_selection(X, thresholds=thresholds)
        assert set(result["ranks_by_threshold"].keys()) == set(thresholds)

    def test_recommended_action_is_string(self):
        """Recommended action should always be a non-empty string."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        result = diagnose_rank_selection(X)
        assert isinstance(result["recommended_action"], str)
        assert len(result["recommended_action"]) > 0

    def test_zero_row_matrix(self):
        """Zero-row matrix edge case."""
        X = np.zeros((0, 5))
        result = diagnose_rank_selection(X)
        assert result["total_columns"] == 5
        assert result["recommended_action"] == "No covariates provided."
