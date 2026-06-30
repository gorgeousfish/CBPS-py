"""
Test Suite for WeightNormalizer and Weight Standardization
===========================================================

Tests the unified ``WeightNormalizer`` class and verifies backward
compatibility with the existing ``standardize_weights`` function.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cbps.utils.weights import (
    WeightNormalizer,
    compute_ate_weights,
    compute_att_weights,
    standardize_weights,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def binary_data_balanced():
    """Balanced binary treatment data (50/50 split)."""
    np.random.seed(123)
    n = 200
    treat = np.array([1] * 100 + [0] * 100, dtype=float)
    logit = np.random.randn(n) * 0.5
    probs = 1 / (1 + np.exp(-logit))
    probs = np.clip(probs, 0.05, 0.95)
    return {"treat": treat, "probs": probs, "n": n}


@pytest.fixture
def binary_data_imbalanced():
    """Imbalanced binary treatment data (20/80 split)."""
    np.random.seed(456)
    n = 200
    treat = np.array([1] * 40 + [0] * 160, dtype=float)
    logit = -1.0 + np.random.randn(n) * 0.3
    probs = 1 / (1 + np.exp(-logit))
    probs = np.clip(probs, 0.05, 0.95)
    return {"treat": treat, "probs": probs, "n": n}


@pytest.fixture
def sample_weights_uniform():
    """Uniform sample weights."""
    return None


@pytest.fixture
def sample_weights_nonuniform():
    """Non-uniform sample weights (sum to n=200)."""
    np.random.seed(789)
    sw = np.random.exponential(1.0, 200)
    sw = sw / sw.sum() * 200
    return sw


# =============================================================================
# Test WeightNormalizer.normalize_ate
# =============================================================================


class TestNormalizeATE:
    """
    Test ATE standardization via WeightNormalizer.

    Test IDs: WNORM-ATE-001 to WNORM-ATE-007
    """

    @pytest.mark.unit
    def test_wnorm_ate001_treated_sum_one(self, binary_data_balanced):
        """WNORM-ATE-001: Treated group weights sum to 1 after normalization."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert_allclose(
            normed[data["treat"] == 1].sum(), 1.0, atol=1e-12,
            err_msg="Treated group should sum to 1"
        )

    @pytest.mark.unit
    def test_wnorm_ate002_control_sum_one(self, binary_data_balanced):
        """WNORM-ATE-002: Control group weights sum to 1 after normalization."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert_allclose(
            normed[data["treat"] == 0].sum(), 1.0, atol=1e-12,
            err_msg="Control group should sum to 1"
        )

    @pytest.mark.unit
    def test_wnorm_ate003_imbalanced_sums(self, binary_data_imbalanced):
        """WNORM-ATE-003: Group sums still 1 with imbalanced treatment."""
        data = binary_data_imbalanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert_allclose(normed[data["treat"] == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[data["treat"] == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.unit
    def test_wnorm_ate004_all_positive(self, binary_data_balanced):
        """WNORM-ATE-004: All normalized ATE weights are non-negative."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert np.all(normed >= 0), "Normalized ATE weights must be non-negative"

    @pytest.mark.unit
    def test_wnorm_ate005_none_sample_weights(self, binary_data_balanced):
        """WNORM-ATE-005: Default behavior when sample_weights is None."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        # Explicit None
        normed_none = WeightNormalizer.normalize_ate(
            raw_w, data["treat"], sample_weights=None
        )
        # Explicit ones
        normed_ones = WeightNormalizer.normalize_ate(
            raw_w, data["treat"], sample_weights=np.ones(data["n"])
        )
        assert_allclose(normed_none, normed_ones, atol=1e-14)

    @pytest.mark.unit
    def test_wnorm_ate006_with_sample_weights(
        self, binary_data_balanced, sample_weights_nonuniform
    ):
        """WNORM-ATE-006: Sample weights are applied correctly."""
        data = binary_data_balanced
        sw = sample_weights_nonuniform
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"], sample_weights=sw)
        # Groups should still sum to 1 after sw application + normalization
        assert_allclose(normed[data["treat"] == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[data["treat"] == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.unit
    def test_wnorm_ate007_does_not_modify_input(self, binary_data_balanced):
        """WNORM-ATE-007: Input weights array is not mutated."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        original = raw_w.copy()
        WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert_allclose(raw_w, original, atol=1e-15)


# =============================================================================
# Test WeightNormalizer.normalize_att
# =============================================================================


class TestNormalizeATT:
    """
    Test ATT standardization via WeightNormalizer.

    Test IDs: WNORM-ATT-001 to WNORM-ATT-006
    """

    @pytest.mark.unit
    def test_wnorm_att001_treated_sum_one(self, binary_data_balanced):
        """WNORM-ATT-001: Treated group weights sum to 1 after ATT normalization."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert_allclose(
            normed[data["treat"] == 1].sum(), 1.0, atol=1e-12,
            err_msg="Treated group should sum to 1"
        )

    @pytest.mark.unit
    def test_wnorm_att002_control_sum_one(self, binary_data_balanced):
        """WNORM-ATT-002: Control group weights sum to 1 after ATT normalization."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert_allclose(
            normed[data["treat"] == 0].sum(), 1.0, atol=1e-12,
            err_msg="Control group should sum to 1"
        )

    @pytest.mark.unit
    def test_wnorm_att003_all_nonnegative(self, binary_data_balanced):
        """WNORM-ATT-003: All normalized ATT weights are non-negative."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert np.all(normed >= 0), "ATT normalized weights must be non-negative"

    @pytest.mark.unit
    def test_wnorm_att004_imbalanced(self, binary_data_imbalanced):
        """WNORM-ATT-004: ATT normalization works with imbalanced treatment."""
        data = binary_data_imbalanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert_allclose(normed[data["treat"] == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[data["treat"] == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.unit
    def test_wnorm_att005_none_sample_weights(self, binary_data_balanced):
        """WNORM-ATT-005: Default behavior when sample_weights is None."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed_none = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"], sample_weights=None
        )
        normed_ones = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"], sample_weights=np.ones(data["n"])
        )
        assert_allclose(normed_none, normed_ones, atol=1e-14)

    @pytest.mark.unit
    def test_wnorm_att006_does_not_modify_input(self, binary_data_balanced):
        """WNORM-ATT-006: Input weights array is not mutated."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        original = raw_w.copy()
        WeightNormalizer.normalize_att(raw_w, data["treat"], data["probs"])
        assert_allclose(raw_w, original, atol=1e-15)


# =============================================================================
# Test WeightNormalizer.validate
# =============================================================================


class TestValidate:
    """
    Test weight validation logic.

    Test IDs: WNORM-VAL-001 to WNORM-VAL-004
    """

    @pytest.mark.unit
    def test_wnorm_val001_valid_weights(self):
        """WNORM-VAL-001: Valid weights pass validation."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert WeightNormalizer.validate(w) is True

    @pytest.mark.unit
    def test_wnorm_val002_nan_raises(self):
        """WNORM-VAL-002: NaN weights raise ValueError."""
        w = np.array([0.5, np.nan, 0.25, 0.25])
        with pytest.raises(ValueError, match="NaN"):
            WeightNormalizer.validate(w)

    @pytest.mark.unit
    def test_wnorm_val003_inf_raises(self):
        """WNORM-VAL-003: Inf weights raise ValueError."""
        w = np.array([0.5, np.inf, 0.25, 0.25])
        with pytest.raises(ValueError, match="Inf"):
            WeightNormalizer.validate(w)

    @pytest.mark.unit
    def test_wnorm_val004_negative_inf_raises(self):
        """WNORM-VAL-004: -Inf weights raise ValueError."""
        w = np.array([0.5, -np.inf, 0.25, 0.25])
        with pytest.raises(ValueError, match="Inf"):
            WeightNormalizer.validate(w)

    @pytest.mark.unit
    def test_wnorm_val005_negative_weights_warns(self):
        """WNORM-VAL-005: Negative weights emit warning when allow_negative=False."""
        w = np.array([0.5, -0.1, 0.3, 0.3])
        with pytest.warns(UserWarning, match="negative weight"):
            WeightNormalizer.validate(w, allow_negative=False)

    @pytest.mark.unit
    def test_wnorm_val006_negative_weights_no_warn_when_allowed(self):
        """WNORM-VAL-006: No warning when allow_negative=True."""
        w = np.array([0.5, -0.1, 0.3, 0.3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warning
            result = WeightNormalizer.validate(w, allow_negative=True)
        assert result is True

    @pytest.mark.unit
    def test_wnorm_val007_nan_still_raises_with_allow_negative(self):
        """WNORM-VAL-007: NaN detection still works when allow_negative=True."""
        w = np.array([0.5, np.nan, -0.1, 0.3])
        with pytest.raises(ValueError, match="NaN"):
            WeightNormalizer.validate(w, allow_negative=True)

    @pytest.mark.unit
    def test_wnorm_val008_inf_still_raises_with_allow_negative(self):
        """WNORM-VAL-008: Inf detection still works when allow_negative=True."""
        w = np.array([0.5, np.inf, -0.1, 0.3])
        with pytest.raises(ValueError, match="Inf"):
            WeightNormalizer.validate(w, allow_negative=True)


# =============================================================================
# Test normalize_ate / normalize_att output non-negativity
# =============================================================================


class TestNormalizeOutputNonNegativity:
    """
    Test that normalize_ate/normalize_att outputs are always non-negative
    under normal conditions, and that warnings are emitted for edge cases.

    Test IDs: WNORM-NONNEG-001 to WNORM-NONNEG-004
    """

    @pytest.mark.unit
    def test_wnorm_nonneg001_ate_output_nonnegative(self, binary_data_balanced):
        """WNORM-NONNEG-001: normalize_ate output is all >= 0."""
        data = binary_data_balanced
        raw_w = compute_ate_weights(data["treat"], data["probs"])
        normed = WeightNormalizer.normalize_ate(raw_w, data["treat"])
        assert np.all(normed >= 0), "All ATE normalized weights must be >= 0"

    @pytest.mark.unit
    def test_wnorm_nonneg002_att_output_nonnegative(self, binary_data_balanced):
        """WNORM-NONNEG-002: normalize_att output is all >= 0."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert np.all(normed >= 0), "All ATT normalized weights must be >= 0"

    @pytest.mark.unit
    def test_wnorm_nonneg003_ate_warns_on_negative_input(self):
        """WNORM-NONNEG-003: normalize_ate warns if input produces negative output."""
        # Simulate a pathological case: manually create negative raw weights
        treat = np.array([1, 1, 0, 0], dtype=float)
        # These negative weights would not normally occur from compute_ate_weights
        # but test the defensive check in normalize_ate
        raw_w = np.array([1.0, -0.5, 1.0, 1.0])
        with pytest.warns(UserWarning, match="negative weight"):
            WeightNormalizer.normalize_ate(raw_w, treat)

    @pytest.mark.unit
    def test_wnorm_nonneg004_att_output_nonneg_imbalanced(
        self, binary_data_imbalanced
    ):
        """WNORM-NONNEG-004: normalize_att output non-negative with imbalanced data."""
        data = binary_data_imbalanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)
        normed = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )
        assert np.all(normed >= 0), "All ATT normalized weights must be >= 0"


# =============================================================================
# Numerical Stability (extreme propensity scores)
# =============================================================================


class TestNumericalStability:
    """
    Test behavior under extreme propensity scores.

    Test IDs: WNORM-NUM-001 to WNORM-NUM-004
    """

    @pytest.mark.numerical
    def test_wnorm_num001_near_zero_probs_ate(self):
        """WNORM-NUM-001: ATE normalization with probs near 0."""
        treat = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        probs = np.array([0.01, 0.02, 0.01, 0.02, 0.03, 0.04])
        raw_w = compute_ate_weights(treat, probs)
        normed = WeightNormalizer.normalize_ate(raw_w, treat)
        assert np.all(np.isfinite(normed))
        assert_allclose(normed[treat == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[treat == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.numerical
    def test_wnorm_num002_near_one_probs_ate(self):
        """WNORM-NUM-002: ATE normalization with probs near 1."""
        treat = np.array([1, 1, 1, 1, 0, 0], dtype=float)
        probs = np.array([0.99, 0.98, 0.97, 0.96, 0.99, 0.98])
        raw_w = compute_ate_weights(treat, probs)
        normed = WeightNormalizer.normalize_ate(raw_w, treat)
        assert np.all(np.isfinite(normed))
        assert_allclose(normed[treat == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[treat == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.numerical
    def test_wnorm_num003_near_zero_probs_att(self):
        """WNORM-NUM-003: ATT normalization with probs near 0."""
        treat = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        probs = np.array([0.01, 0.02, 0.01, 0.02, 0.03, 0.04])
        sw = np.ones(6)
        raw_w = compute_att_weights(treat, probs, sw)
        normed = WeightNormalizer.normalize_att(raw_w, treat, probs)
        assert np.all(np.isfinite(normed))
        assert_allclose(normed[treat == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[treat == 0].sum(), 1.0, atol=1e-12)

    @pytest.mark.numerical
    def test_wnorm_num004_near_one_probs_att(self):
        """WNORM-NUM-004: ATT normalization with probs near 1."""
        treat = np.array([1, 1, 1, 1, 0, 0], dtype=float)
        probs = np.array([0.99, 0.98, 0.97, 0.96, 0.99, 0.98])
        sw = np.ones(6)
        raw_w = compute_att_weights(treat, probs, sw)
        normed = WeightNormalizer.normalize_att(raw_w, treat, probs)
        assert np.all(np.isfinite(normed))
        assert_allclose(normed[treat == 1].sum(), 1.0, atol=1e-12)
        assert_allclose(normed[treat == 0].sum(), 1.0, atol=1e-12)


# =============================================================================
# Backward Compatibility with standardize_weights
# =============================================================================


class TestBackwardCompatibility:
    """
    Ensure WeightNormalizer produces results consistent with the existing
    ``standardize_weights`` function.

    Test IDs: WNORM-COMPAT-001 to WNORM-COMPAT-004
    """

    @pytest.mark.integration
    def test_wnorm_compat001_ate_group_sums_match(self, binary_data_balanced):
        """WNORM-COMPAT-001: standardize_weights ATE group sums match normalizer."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_ate_weights(data["treat"], data["probs"])

        # Old interface
        old_result = standardize_weights(
            raw_w, data["treat"], data["probs"], sw, att=0, standardize=True
        )
        # New interface
        new_result = WeightNormalizer.normalize_ate(raw_w, data["treat"])

        # Both should sum to 1 per group
        assert_allclose(
            old_result[data["treat"] == 1].sum(), 1.0, atol=1e-10
        )
        assert_allclose(
            new_result[data["treat"] == 1].sum(), 1.0, atol=1e-12
        )
        assert_allclose(
            old_result[data["treat"] == 0].sum(), 1.0, atol=1e-10
        )
        assert_allclose(
            new_result[data["treat"] == 0].sum(), 1.0, atol=1e-12
        )

    @pytest.mark.integration
    def test_wnorm_compat002_att_group_sums_match(self, binary_data_balanced):
        """WNORM-COMPAT-002: standardize_weights ATT group sums match normalizer."""
        data = binary_data_balanced
        sw = np.ones(data["n"])
        raw_w = compute_att_weights(data["treat"], data["probs"], sw)

        # Old interface
        old_result = standardize_weights(
            raw_w, data["treat"], data["probs"], sw, att=1, standardize=True
        )
        # New interface
        new_result = WeightNormalizer.normalize_att(
            raw_w, data["treat"], data["probs"]
        )

        # Old result has sample_weights baked in; both groups sum should be close
        old_treat_sum = old_result[data["treat"] == 1].sum()
        new_treat_sum = new_result[data["treat"] == 1].sum()
        # standardize_weights multiplies by sample_weights at the end
        # With uniform sw=1, old treat sum = 1.0
        assert_allclose(old_treat_sum, 1.0, atol=1e-10)
        assert_allclose(new_treat_sum, 1.0, atol=1e-12)

    @pytest.mark.integration
    def test_wnorm_compat003_standardize_weights_unchanged_api(self):
        """WNORM-COMPAT-003: standardize_weights API signature unchanged."""
        import inspect
        sig = inspect.signature(standardize_weights)
        params = list(sig.parameters.keys())
        assert params == ["weights", "treat", "probs", "sample_weights", "att",
                          "standardize"]

    @pytest.mark.integration
    def test_wnorm_compat004_compute_functions_unchanged(self):
        """WNORM-COMPAT-004: compute_ate_weights / compute_att_weights APIs intact."""
        import inspect

        ate_sig = inspect.signature(compute_ate_weights)
        assert list(ate_sig.parameters.keys()) == ["treat", "probs"]

        att_sig = inspect.signature(compute_att_weights)
        assert list(att_sig.parameters.keys()) == [
            "treat", "probs", "sample_weights"
        ]


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
