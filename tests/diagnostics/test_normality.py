"""Tests for normality diagnostics (P1-17).

Tests cover:
1. Normal data should NOT reject normality.
2. Skewed data SHOULD reject normality.
3. Large samples (n>5000) should switch to D'Agostino-Pearson test.
4. Output format validation.
"""

import numpy as np
import pytest

from cbps.diagnostics.normality import test_treatment_normality as check_normality


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestNormalDataNotRejected:
    """Normal treatment residuals should not be rejected."""

    def test_normal_data_not_rejected(self, rng):
        n, k = 500, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        treat = X @ beta_true + rng.standard_normal(n)  # truly normal residuals

        result = check_normality(treat, X)

        assert result["reject_normality"] is False
        assert result["p_value"] > 0.05
        assert result["test_used"] == "shapiro-wilk"
        assert result["warning_message"] is None

    def test_normal_data_moderate_sample(self, rng):
        """n=2000, still shapiro-wilk, normal data passes."""
        n, k = 2000, 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        treat = X @ beta_true + rng.standard_normal(n) * 2.0

        result = check_normality(treat, X)

        assert result["reject_normality"] is False
        assert result["test_used"] == "shapiro-wilk"


class TestSkewedDataRejected:
    """Non-normal (skewed) data should be rejected."""

    def test_exponential_residuals(self, rng):
        n, k = 500, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        # Exponential residuals are heavily right-skewed
        treat = X @ beta_true + rng.exponential(2.0, size=n)

        result = check_normality(treat, X)

        assert result["reject_normality"] is True
        assert result["p_value"] < 0.05
        assert result["warning_message"] is not None
        assert "npCBPS" in result["warning_message"]

    def test_bimodal_residuals(self, rng):
        n, k = 500, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        # Bimodal residuals
        residuals = np.where(
            rng.random(n) < 0.5,
            rng.normal(-3, 0.5, n),
            rng.normal(3, 0.5, n),
        )
        treat = X @ beta_true + residuals

        result = check_normality(treat, X)

        assert result["reject_normality"] is True


class TestLargeSampleSwitch:
    """For n > 5000, should use D'Agostino-Pearson test."""

    def test_switches_to_dagostino(self, rng):
        n, k = 6000, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        treat = X @ beta_true + rng.standard_normal(n)

        result = check_normality(treat, X)

        assert result["test_used"] == "dagostino-pearson"

    def test_large_sample_normal_not_rejected(self, rng):
        n, k = 8000, 4
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        treat = X @ beta_true + rng.standard_normal(n) * 1.5

        result = check_normality(treat, X)

        assert result["test_used"] == "dagostino-pearson"
        assert result["reject_normality"] is False

    def test_large_sample_skewed_rejected(self, rng):
        n, k = 7000, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = rng.standard_normal(k)
        treat = X @ beta_true + rng.exponential(2.0, size=n)

        result = check_normality(treat, X)

        assert result["test_used"] == "dagostino-pearson"
        assert result["reject_normality"] is True


class TestOutputFormat:
    """Result dict should have all expected keys with correct types."""

    def test_all_keys_present(self, rng):
        n, k = 200, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = X @ np.array([1.0, 0.5]) + rng.standard_normal(n)

        result = check_normality(treat, X)

        expected_keys = {
            "statistic", "p_value", "test_used",
            "reject_normality", "skewness", "kurtosis", "warning_message"
        }
        assert set(result.keys()) == expected_keys

    def test_types_correct(self, rng):
        n, k = 200, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        treat = X @ np.array([1.0, 0.5]) + rng.standard_normal(n)

        result = check_normality(treat, X)

        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["test_used"], str)
        assert isinstance(result["reject_normality"], bool)
        assert isinstance(result["skewness"], float)
        assert isinstance(result["kurtosis"], float)

    def test_dimension_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            check_normality(
                rng.standard_normal(100),
                rng.standard_normal((50, 3)),
            )


class TestInputValidationGuards:
    """Tests for NaN/Inf, small sample, constant residuals guards."""

    def test_nan_input_treat(self):
        X = np.random.randn(50, 3)
        treat = np.random.randn(50)
        treat[0] = np.nan
        result = check_normality(treat, X)
        assert result['warning_message'] is not None
        assert 'NaN' in result['warning_message'] or 'nan' in result['warning_message'].lower()

    def test_inf_input_X(self):
        X = np.random.randn(50, 3)
        X[0, 0] = np.inf
        treat = np.random.randn(50)
        result = check_normality(treat, X)
        assert result['warning_message'] is not None

    def test_very_small_sample(self):
        X = np.array([[1.0, 0.5], [1.0, -0.5]])
        treat = np.array([1.0, 2.0])
        result = check_normality(treat, X)
        assert result['warning_message'] is not None
        assert 'too small' in result['warning_message']

    def test_constant_residuals(self):
        X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
        treat = X @ np.array([1.0, 2.0, 3.0])  # perfect linear
        result = check_normality(treat, X)
        assert result['warning_message'] is not None
        assert 'linear function' in result['warning_message']
