"""Tests for NumericalConfig central constants module.

Verifies that the unified configuration:
- Contains correct default values aligned with R CBPS v0.23
- Is immutable (frozen dataclass)
- Supports custom configuration creation
"""
import pytest
from dataclasses import FrozenInstanceError

from cbps.constants import NumericalConfig, DEFAULT_CONFIG


class TestDefaultValues:
    """Verify DEFAULT_CONFIG defaults match R CBPS v0.23 and documented values."""

    def test_probs_min(self):
        """R CBPS v0.23: probs.min = 1e-6."""
        assert DEFAULT_CONFIG.probs_min == 1e-6

    def test_probs_trim_msm(self):
        """CBMSM probability clipping threshold."""
        assert DEFAULT_CONFIG.probs_trim_msm == 1e-4

    def test_const_col_threshold(self):
        """Constant column detection threshold."""
        assert DEFAULT_CONFIG.const_col_threshold == 1e-10

    def test_ndeps(self):
        """R optim() default ndeps = 1e-3."""
        assert DEFAULT_CONFIG.ndeps == 1e-3

    def test_glm_tol(self):
        """GLM IRLS convergence tolerance."""
        assert DEFAULT_CONFIG.glm_tol == 1e-8

    def test_optim_xtol(self):
        """Parameter convergence tolerance."""
        assert DEFAULT_CONFIG.optim_xtol == 1e-12

    def test_svd_threshold_msm(self):
        """MSM singular value cutoff."""
        assert DEFAULT_CONFIG.svd_threshold_msm == 1e-4

    def test_log_clip_range(self):
        """Log clip range for numerical stability."""
        assert DEFAULT_CONFIG.log_clip_range == 50.0


class TestProbsMaxProperty:
    """Verify the computed probs_max property."""

    def test_probs_max_default(self):
        """probs_max = 1 - probs_min."""
        assert DEFAULT_CONFIG.probs_max == 1.0 - 1e-6

    def test_probs_max_custom(self):
        """probs_max reflects custom probs_min."""
        config = NumericalConfig(probs_min=0.01)
        assert config.probs_max == pytest.approx(0.99)


class TestImmutability:
    """Verify frozen=True prevents mutation."""

    def test_cannot_set_probs_min(self):
        with pytest.raises(FrozenInstanceError):
            DEFAULT_CONFIG.probs_min = 0.5  # type: ignore

    def test_cannot_set_ndeps(self):
        with pytest.raises(FrozenInstanceError):
            DEFAULT_CONFIG.ndeps = 0.1  # type: ignore

    def test_cannot_set_new_attribute(self):
        with pytest.raises(FrozenInstanceError):
            DEFAULT_CONFIG.new_attr = 42  # type: ignore


class TestCustomConfig:
    """Verify custom NumericalConfig instances can be created."""

    def test_custom_probs_min(self):
        config = NumericalConfig(probs_min=1e-4)
        assert config.probs_min == 1e-4

    def test_custom_ndeps(self):
        config = NumericalConfig(ndeps=1e-5)
        assert config.ndeps == 1e-5

    def test_custom_preserves_other_defaults(self):
        config = NumericalConfig(probs_min=1e-4)
        assert config.ndeps == 1e-3
        assert config.glm_tol == 1e-8
        assert config.svd_threshold_msm == 1e-4

    def test_fully_custom(self):
        config = NumericalConfig(
            probs_min=1e-5,
            probs_trim_msm=1e-3,
            const_col_threshold=1e-8,
            ndeps=1e-4,
            glm_tol=1e-10,
            optim_xtol=1e-14,
            svd_threshold_msm=1e-3,
            log_clip_range=100.0,
        )
        assert config.probs_min == 1e-5
        assert config.probs_trim_msm == 1e-3
        assert config.const_col_threshold == 1e-8
        assert config.ndeps == 1e-4
        assert config.glm_tol == 1e-10
        assert config.optim_xtol == 1e-14
        assert config.svd_threshold_msm == 1e-3
        assert config.log_clip_range == 100.0


class TestGlobalInstance:
    """Verify the module-level DEFAULT_CONFIG is a proper singleton."""

    def test_is_numerical_config(self):
        assert isinstance(DEFAULT_CONFIG, NumericalConfig)

    def test_default_instance_values_match_class_defaults(self):
        fresh = NumericalConfig()
        assert fresh == DEFAULT_CONFIG
