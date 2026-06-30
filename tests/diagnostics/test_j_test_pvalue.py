"""Tests for J-statistic asymptotic p-value (P1-12).

Tests the j_test_pvalue function that computes the Hansen J-test
p-value for GMM overidentification, as described in Imai & Ratkovic (2014).
"""

import numpy as np
import pytest
from scipy.stats import chi2

from cbps.core.results import j_test_pvalue


class TestJTestPvalueBasic:
    """Basic J-test p-value computation."""

    def test_j_zero_gives_pvalue_one(self):
        """J=0, df>0 → p_value = 1.0 (perfect model fit)."""
        p = j_test_pvalue(J=0.0, n_moment_conditions=10, n_parameters=5)
        assert p == pytest.approx(1.0)

    def test_large_j_gives_small_pvalue(self):
        """Large J value → p_value close to 0 (model rejected)."""
        p = j_test_pvalue(J=100.0, n_moment_conditions=10, n_parameters=5)
        assert p < 0.001

    def test_just_identified_returns_none(self):
        """df = 0 (just-identified) → returns None."""
        p = j_test_pvalue(J=5.0, n_moment_conditions=5, n_parameters=5)
        assert p is None

    def test_negative_df_returns_none(self):
        """n_parameters > n_moments → returns None."""
        p = j_test_pvalue(J=5.0, n_moment_conditions=3, n_parameters=5)
        assert p is None

    def test_negative_j_raises_valueerror(self):
        """Negative J (impossible for quadratic form) → raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            j_test_pvalue(J=-5.0, n_moment_conditions=10, n_parameters=5)

    def test_nan_j_raises_valueerror(self):
        """NaN J → raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            j_test_pvalue(J=float('nan'), n_moment_conditions=10, n_parameters=5)


class TestJTestPvalueConsistency:
    """Verify consistency with scipy.stats.chi2."""

    def test_matches_scipy_chi2(self):
        """P-value matches direct scipy computation."""
        J = 7.5
        n_moments = 12
        n_params = 6
        df = n_moments - n_params  # df = 6

        expected = 1.0 - chi2.cdf(J, df)
        actual = j_test_pvalue(J, n_moments, n_params)

        assert actual == pytest.approx(expected, rel=1e-12)

    def test_various_df_values(self):
        """Verify across different degrees of freedom."""
        test_cases = [
            (3.0, 8, 5),   # df=3
            (10.0, 15, 5),  # df=10
            (1.5, 6, 4),   # df=2
            (20.0, 30, 10), # df=20
        ]
        for J, n_m, n_p in test_cases:
            df = n_m - n_p
            expected = 1.0 - chi2.cdf(J, df)
            actual = j_test_pvalue(J, n_m, n_p)
            assert actual == pytest.approx(expected, rel=1e-12), \
                f"Failed for J={J}, df={df}"

    def test_critical_value_at_5pct(self):
        """At chi2 critical value, p-value ≈ 0.05."""
        df = 5
        critical = chi2.ppf(0.95, df)  # 11.07
        p = j_test_pvalue(critical, n_moment_conditions=10, n_parameters=5)
        assert p == pytest.approx(0.05, abs=1e-10)


class TestJTestBinaryCBPS:
    """Tests specific to binary CBPS J-test interpretation."""

    def test_binary_over_identified_df_equals_k(self):
        """For binary CBPS method='over': df = k (k score + k balance - k params)."""
        # In binary CBPS with 'over' method:
        # n_moment_conditions = 2k, n_parameters = k, df = k
        k = 7  # e.g., intercept + 6 covariates
        J = 5.0
        p = j_test_pvalue(J, n_moment_conditions=2*k, n_parameters=k)

        # Should equal chi2 with df=k=7
        expected = 1.0 - chi2.cdf(J, k)
        assert p == pytest.approx(expected, rel=1e-12)

    def test_binary_exact_identified(self):
        """For binary CBPS method='exact': df = 0, returns None."""
        k = 5
        p = j_test_pvalue(J=3.0, n_moment_conditions=k, n_parameters=k)
        assert p is None

    def test_monotonicity_in_j(self):
        """Larger J → smaller p-value (monotone decreasing)."""
        df_args = dict(n_moment_conditions=10, n_parameters=5)
        p1 = j_test_pvalue(J=1.0, **df_args)
        p2 = j_test_pvalue(J=5.0, **df_args)
        p3 = j_test_pvalue(J=20.0, **df_args)

        assert p1 > p2 > p3
