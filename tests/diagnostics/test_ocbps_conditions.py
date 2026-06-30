"""Tests for oCBPS condition verification (P1-18/P1-21).

Tests cover:
1. Conditions satisfied → all_conditions_met = True.
2. Dimension insufficient → identification_ok = False.
3. Output format validation.
"""

import numpy as np
import pytest

from cbps.diagnostics.ocbps_conditions import verify_ocbps_conditions


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def _make_balanced_result(rng, n, k):
    """Create a mock CBPS result where balance is achieved."""
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    treat = rng.binomial(1, 0.5, n).astype(float)
    # Uniform weights → balance approximately holds for large n
    weights = np.ones(n)
    ps = np.full(n, 0.5)
    return {
        "weights": weights,
        "ps": ps,
        "J": 0.5,
        "J_pval": 0.8,
        "n_moments": 2 * k,
    }, X, treat


class TestConditionsMet:
    """When all conditions satisfied, all_conditions_met should be True."""

    def test_balanced_uniform_weights(self, rng):
        n, k = 1000, 4
        result, X, treat = _make_balanced_result(rng, n, k)

        out = verify_ocbps_conditions(result, X, treat)

        assert out["identification_ok"] is True
        assert out["overlap_ok"] is True
        assert out["all_conditions_met"] is True
        assert len(out["warnings"]) == 0

    def test_j_test_not_rejected(self, rng):
        n, k = 500, 3
        result, X, treat = _make_balanced_result(rng, n, k)

        out = verify_ocbps_conditions(result, X, treat)

        assert out["j_test_result"] is not None
        assert out["j_test_result"]["reject"] is False


class TestDimensionInsufficient:
    """When moment conditions < k, identification_ok should be False."""

    def test_under_identified(self, rng):
        n, k = 500, 10
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = rng.binomial(1, 0.5, n).astype(float)

        result = {
            "weights": np.ones(n),
            "ps": np.full(n, 0.5),
            "n_moments": 5,  # 5 + 1 = 6 < k = 10
        }

        out = verify_ocbps_conditions(result, X, treat)

        assert out["identification_ok"] is False
        assert out["all_conditions_met"] is False
        assert any("Identification" in w for w in out["warnings"])

    def test_just_identified_ok(self, rng):
        n, k = 500, 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = rng.binomial(1, 0.5, n).astype(float)

        result = {
            "weights": np.ones(n),
            "ps": np.full(n, 0.5),
            "n_moments": k,  # k + 1 >= k → OK
        }

        out = verify_ocbps_conditions(result, X, treat)

        assert out["identification_ok"] is True


class TestOverlapViolation:
    """Extreme propensity scores should trigger overlap warning."""

    def test_extreme_ps(self, rng):
        n, k = 200, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = rng.binomial(1, 0.5, n).astype(float)

        ps = np.full(n, 0.5)
        ps[0] = 0.001  # extreme
        ps[1] = 0.999  # extreme

        result = {
            "weights": np.ones(n),
            "ps": ps,
        }

        out = verify_ocbps_conditions(result, X, treat)

        assert out["overlap_ok"] is False
        assert any("Overlap" in w or "overlap" in w for w in out["warnings"])


class TestJTestRejection:
    """J-test rejection should set all_conditions_met=False."""

    def test_j_test_rejected(self, rng):
        n, k = 500, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = rng.binomial(1, 0.5, n).astype(float)

        result = {
            "weights": np.ones(n),
            "ps": np.full(n, 0.5),
            "J": 25.0,
            "J_pval": 0.001,  # rejects
            "n_moments": 2 * k,
        }

        out = verify_ocbps_conditions(result, X, treat)

        assert out["j_test_result"]["reject"] is True
        assert out["all_conditions_met"] is False


class TestOutputFormat:
    """Result dict should have all expected keys with correct types."""

    def test_all_keys_present(self, rng):
        n, k = 200, 3
        result, X, treat = _make_balanced_result(rng, n, k)

        out = verify_ocbps_conditions(result, X, treat)

        expected_keys = {
            "identification_ok", "balance_achieved", "j_test_result",
            "overlap_ok", "all_conditions_met", "warnings"
        }
        assert set(out.keys()) == expected_keys

    def test_types_correct(self, rng):
        n, k = 200, 3
        result, X, treat = _make_balanced_result(rng, n, k)

        out = verify_ocbps_conditions(result, X, treat)

        assert isinstance(out["identification_ok"], bool)
        assert isinstance(out["balance_achieved"], bool)
        assert isinstance(out["overlap_ok"], bool)
        assert isinstance(out["all_conditions_met"], bool)
        assert isinstance(out["warnings"], list)

    def test_missing_weights_raises(self, rng):
        n, k = 100, 3
        X = rng.standard_normal((n, k))
        treat = rng.binomial(1, 0.5, n).astype(float)

        with pytest.raises(ValueError, match="weights"):
            verify_ocbps_conditions({}, X, treat)
