"""
Tests for separation detection warning messages in CBPS binary.

Verifies that:
1. All four severity levels (COMPLETE, QUASI, MODERATE, MINOR) are exercised
2. Warning messages contain expected format and suggestions
3. Normal data does not trigger false positive separation warnings
4. Edge boundary conditions (exactly 10%, 50%, 100%) are handled correctly

Testing strategy:
- Directly construct `probs_opt_raw` arrays with precise boundary percentages
- Call the extracted `_classify_separation` pure function
- Assert exact severity level transitions at threshold boundaries
"""

import warnings

import numpy as np
import pytest

from cbps import CBPS
from cbps.core.cbps_binary import PROBS_MIN, _classify_separation, _compute_final_weights


# ---------------------------------------------------------------------------
# Helper: construct probs_opt_raw with exact boundary percentage
# ---------------------------------------------------------------------------

def _make_probs_with_boundary_pct(boundary_pct, n=1000):
    """Construct a probs_opt_raw array with exactly boundary_pct% at boundaries.

    Parameters
    ----------
    boundary_pct : float
        Desired percentage (0-100) of observations at probability boundary.
    n : int
        Total number of observations.

    Returns
    -------
    probs_opt_raw : np.ndarray
        Array of shape (n,) where exactly `n_boundary` entries are at or
        beyond PROBS_MIN / (1 - PROBS_MIN).
    """
    n_boundary = int(round(boundary_pct / 100.0 * n))
    # Interior probs: safe values away from boundaries
    probs_interior = np.full(n - n_boundary, 0.5)
    # Boundary probs: half at low boundary, half at high boundary
    n_low = n_boundary // 2
    n_high = n_boundary - n_low
    probs_low = np.full(n_low, PROBS_MIN * 0.1)  # Well below PROBS_MIN
    probs_high = np.full(n_high, 1.0 - PROBS_MIN * 0.1)  # Well above 1-PROBS_MIN
    probs = np.concatenate([probs_interior, probs_low, probs_high])
    return probs


# ===========================================================================
# Test class: Precise boundary threshold tests via _classify_separation
# ===========================================================================

class TestClassifySeparationThresholds:
    """Directly test _classify_separation with precise boundary percentages."""

    def test_zero_boundary_returns_none(self):
        """boundary_pct = 0% -> no warning (returns None)."""
        probs = _make_probs_with_boundary_pct(0.0, n=1000)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is None

    def test_below_10_percent_returns_minor(self):
        """boundary_pct = 9.9% -> MINOR."""
        # With n=1000, 9.9% = 99 observations at boundary
        probs = _make_probs_with_boundary_pct(9.9, n=1000)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MINOR"

    def test_exactly_10_percent_returns_moderate(self):
        """boundary_pct = 10.0% -> MODERATE SEPARATION (threshold: >=10%)."""
        # With n=1000, 10.0% = exactly 100 observations at boundary
        probs = _make_probs_with_boundary_pct(10.0, n=1000)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MODERATE SEPARATION"

    def test_just_below_10_percent_is_minor(self):
        """boundary_pct = 9/100 = 9% -> MINOR."""
        # Use n=100 for integer-precise control
        probs = _make_probs_with_boundary_pct(9.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MINOR"

    def test_just_above_10_percent_is_moderate(self):
        """boundary_pct = 11/100 = 11% -> MODERATE SEPARATION."""
        probs = _make_probs_with_boundary_pct(11.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MODERATE SEPARATION"

    def test_49_percent_returns_moderate(self):
        """boundary_pct = 49% -> MODERATE SEPARATION (below 50% threshold)."""
        probs = _make_probs_with_boundary_pct(49.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MODERATE SEPARATION"

    def test_exactly_50_percent_returns_quasi(self):
        """boundary_pct = 50.0% -> QUASI-SEPARATION (threshold: >=50%)."""
        probs = _make_probs_with_boundary_pct(50.0, n=1000)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "QUASI-SEPARATION"

    def test_just_below_50_percent_is_moderate(self):
        """boundary_pct = 49/100 = 49% -> MODERATE SEPARATION."""
        probs = _make_probs_with_boundary_pct(49.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MODERATE SEPARATION"

    def test_just_above_50_percent_is_quasi(self):
        """boundary_pct = 51/100 = 51% -> QUASI-SEPARATION."""
        probs = _make_probs_with_boundary_pct(51.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "QUASI-SEPARATION"

    def test_99_percent_returns_quasi(self):
        """boundary_pct = 99% -> QUASI-SEPARATION (below 100% threshold)."""
        probs = _make_probs_with_boundary_pct(99.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "QUASI-SEPARATION"

    def test_exactly_100_percent_returns_complete(self):
        """boundary_pct = 100% -> COMPLETE SEPARATION (threshold: >=100%)."""
        probs = _make_probs_with_boundary_pct(100.0, n=1000)
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "COMPLETE SEPARATION"

    def test_single_observation_at_boundary(self):
        """1 out of 1000 at boundary = 0.1% -> MINOR."""
        n = 1000
        probs = np.full(n, 0.5)
        probs[0] = PROBS_MIN * 0.1  # One observation at low boundary
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "MINOR"

    def test_all_at_low_boundary(self):
        """All observations at low boundary -> COMPLETE SEPARATION."""
        n = 100
        probs = np.full(n, PROBS_MIN * 0.5)
        beta = np.array([-50.0, 0.0, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "COMPLETE SEPARATION"

    def test_all_at_high_boundary(self):
        """All observations at high boundary -> COMPLETE SEPARATION."""
        n = 100
        probs = np.full(n, 1.0 - PROBS_MIN * 0.5)
        beta = np.array([50.0, 0.0, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None
        severity, msg = result
        assert severity == "COMPLETE SEPARATION"


# ===========================================================================
# Test class: Verify exact boundary_pct computation
# ===========================================================================

class TestBoundaryPctComputation:
    """Verify boundary_pct is computed correctly based on PROBS_MIN threshold."""

    def test_probs_exactly_at_probs_min_counts_as_boundary(self):
        """Prob exactly equal to PROBS_MIN is at boundary (<=)."""
        n = 100
        probs = np.full(n, 0.5)
        probs[0] = PROBS_MIN  # Exactly at PROBS_MIN
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None  # Should be detected

    def test_probs_just_above_probs_min_not_boundary(self):
        """Prob slightly above PROBS_MIN is NOT at boundary."""
        n = 100
        probs = np.full(n, 0.5)
        probs[0] = PROBS_MIN * 2  # Above PROBS_MIN but still small
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is None  # Not at boundary

    def test_probs_exactly_at_one_minus_probs_min_counts_as_boundary(self):
        """Prob exactly equal to 1 - PROBS_MIN is at boundary (>=)."""
        n = 100
        probs = np.full(n, 0.5)
        probs[0] = 1.0 - PROBS_MIN  # Exactly at high boundary
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is not None

    def test_probs_just_below_one_minus_probs_min_not_boundary(self):
        """Prob slightly below 1 - PROBS_MIN is NOT at boundary."""
        n = 100
        probs = np.full(n, 0.5)
        probs[0] = 1.0 - PROBS_MIN * 2  # Below the high boundary
        beta = np.array([0.0, 0.5, 0.0])
        result = _classify_separation(probs, beta)
        assert result is None


# ===========================================================================
# Test class: Warning message format
# ===========================================================================

class TestWarningMessageFormat:
    """Verify warning message structure and formatting."""

    def test_structured_header_format(self):
        """All warnings use [CBPS Separation Warning - LEVEL] header."""
        for pct in [5.0, 25.0, 60.0, 100.0]:
            probs = _make_probs_with_boundary_pct(pct, n=100)
            beta = np.array([0.0, 0.5, 0.0])
            result = _classify_separation(probs, beta)
            assert result is not None
            severity, msg = result
            assert msg.startswith("[CBPS Separation Warning -"), (
                f"Unexpected header at {pct}%: {msg[:60]}"
            )

    def test_contains_observation_count(self):
        """Warning reports number and percentage of boundary observations."""
        probs = _make_probs_with_boundary_pct(25.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        _, msg = _classify_separation(probs, beta)
        assert "observations" in msg
        assert "%" in msg
        assert "at probability boundary" in msg

    def test_contains_low_high_breakdown(self):
        """Warning reports low and high boundary counts separately."""
        probs = _make_probs_with_boundary_pct(25.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        _, msg = _classify_separation(probs, beta)
        assert "Low boundary" in msg
        assert "High boundary" in msg

    def test_no_double_blank_lines(self):
        """Warning message should not contain double blank lines."""
        for pct in [5.0, 25.0, 60.0, 100.0]:
            probs = _make_probs_with_boundary_pct(pct, n=100)
            beta = np.array([0.0, 0.5, 0.0])
            _, msg = _classify_separation(probs, beta)
            assert "\n\n" not in msg, (
                f"Double blank line at {pct}%: {repr(msg)}"
            )

    def test_suggestions_are_numbered(self):
        """All suggestions should be numbered starting from 1."""
        probs = _make_probs_with_boundary_pct(60.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        _, msg = _classify_separation(probs, beta)
        assert "  1." in msg
        assert "  2." in msg

    def test_extreme_coefficients_line_present_when_applicable(self):
        """Extreme coef line appears when |beta| > threshold."""
        probs = _make_probs_with_boundary_pct(25.0, n=100)
        beta = np.array([0.0, 50.0, 0.0])  # |50| > 10
        _, msg = _classify_separation(probs, beta)
        assert "Extreme coefficients" in msg

    def test_no_extreme_coefficients_line_when_not_applicable(self):
        """Extreme coef line absent when |beta| <= threshold."""
        probs = _make_probs_with_boundary_pct(25.0, n=100)
        beta = np.array([0.0, 5.0, 0.0])  # |5| < 10
        _, msg = _classify_separation(probs, beta)
        assert "Extreme coefficients" not in msg

    def test_correct_suggestions_for_complete(self):
        """COMPLETE level includes expected suggestions."""
        probs = _make_probs_with_boundary_pct(100.0, n=100)
        beta = np.array([50.0, 0.0, 0.0])
        severity, msg = _classify_separation(probs, beta)
        assert severity == "COMPLETE SEPARATION"
        assert "Check for perfect predictors" in msg
        assert "hdCBPS with LASSO" in msg
        assert "Firth" in msg
        assert "Verify data coding" in msg

    def test_correct_suggestions_for_quasi(self):
        """QUASI level includes expected suggestions."""
        probs = _make_probs_with_boundary_pct(60.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        severity, msg = _classify_separation(probs, beta)
        assert severity == "QUASI-SEPARATION"
        assert "VIF" in msg
        assert "Crump et al. 2009" in msg
        assert "1st/99th percentile" in msg

    def test_correct_suggestions_for_moderate(self):
        """MODERATE level includes expected suggestions."""
        probs = _make_probs_with_boundary_pct(25.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        severity, msg = _classify_separation(probs, beta)
        assert severity == "MODERATE SEPARATION"
        assert "covariate balance" in msg
        assert "ESS" in msg

    def test_correct_suggestions_for_minor(self):
        """MINOR level includes expected suggestions."""
        probs = _make_probs_with_boundary_pct(5.0, n=100)
        beta = np.array([0.0, 0.5, 0.0])
        severity, msg = _classify_separation(probs, beta)
        assert severity == "MINOR"
        assert "usually acceptable" in msg


# ===========================================================================
# Test class: _compute_final_weights integration (emits warnings correctly)
# ===========================================================================

class TestComputeFinalWeightsWarning:
    """Verify _compute_final_weights properly emits warnings via _classify_separation."""

    def _get_separation_warning(self, beta_opt, X, treat, sample_weights,
                                att=0, standardize=True):
        """Run _compute_final_weights and return the separation warning message."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_final_weights(beta_opt, X, treat, sample_weights, att, standardize)
            sep_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "CBPS Separation Warning" in str(x.message)
            ]
            if sep_warnings:
                return str(sep_warnings[0].message)
        return None

    def test_large_coef_triggers_warning(self):
        """Large coefficient pushes all probs to boundary -> warning emitted."""
        n = 100
        rng = np.random.default_rng(42)
        treat = rng.binomial(1, 0.5, n).astype(float)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        sw = np.ones(n)
        beta_opt = np.array([200.0, 0.0, 0.0])  # All probs → 1.0

        msg = self._get_separation_warning(beta_opt, X, treat, sw)
        assert msg is not None
        assert "COMPLETE SEPARATION" in msg

    def test_small_coef_no_warning(self):
        """Small coefficient keeps probs in safe range -> no warning."""
        n = 100
        rng = np.random.default_rng(42)
        treat = rng.binomial(1, 0.5, n).astype(float)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        sw = np.ones(n)
        beta_opt = np.array([0.0, 0.3, 0.1])

        msg = self._get_separation_warning(beta_opt, X, treat, sw)
        assert msg is None


# ===========================================================================
# Test class: No false positives (end-to-end via CBPS API)
# ===========================================================================

class TestNoFalsePositive:
    """Verify well-behaved data does not trigger severe separation warnings."""

    def test_well_overlapping_data_no_severe_warning(self):
        """Mild treatment assignment should not trigger severe warnings."""
        np.random.seed(42)
        n = 500
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        logit = 0.3 * x1 + 0.2 * x2
        prob = 1 / (1 + np.exp(-logit))
        treat = (np.random.rand(n) < prob).astype(int)
        X = np.column_stack([x1, x2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = CBPS(treatment=treat, covariates=X)

            severe_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and ("COMPLETE SEPARATION" in str(x.message)
                     or "QUASI-SEPARATION" in str(x.message)
                     or "MODERATE SEPARATION" in str(x.message))
            ]
            assert len(severe_warnings) == 0, (
                f"False positive separation warning on well-overlapping data: "
                f"{str(severe_warnings[0].message)[:100]}"
            )

    def test_balanced_randomized_design_no_warning(self):
        """Random assignment (no relationship with X) should not warn."""
        np.random.seed(99)
        n = 300
        treat = np.random.binomial(1, 0.5, n)
        X = np.random.randn(n, 3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = CBPS(treatment=treat, covariates=X)

            severe_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and ("COMPLETE SEPARATION" in str(x.message)
                     or "QUASI-SEPARATION" in str(x.message))
            ]
            assert len(severe_warnings) == 0


# ===========================================================================
# Test class: Integration-level (end-to-end via CBPS API)
# ===========================================================================

class TestIntegrationSeparation:
    """End-to-end tests via the public CBPS interface."""

    def test_complete_separation_via_api(self):
        """Perfect predictor triggers COMPLETE SEPARATION via CBPS()."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        treat = (x > 0).astype(int)
        X = np.column_stack([x, np.random.randn(n)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = CBPS(treatment=treat, covariates=X)

            sep_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "CBPS Separation Warning" in str(x.message)
            ]
            assert len(sep_warnings) > 0, (
                "Expected separation warning for perfectly separated data"
            )
            msg = str(sep_warnings[0].message)
            assert "COMPLETE SEPARATION" in msg
            assert "Suggested actions:" in msg
