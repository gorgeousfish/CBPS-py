"""Central numerical stability constants for CBPS package.

All defaults are aligned with R CBPS v0.23 where applicable.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalConfig:
    """Immutable numerical configuration for CBPS algorithms.

    Parameters are grouped by function:
    - Propensity score clipping
    - Optimization tolerances
    - Matrix computation thresholds

    References
    ----------
    R CBPS v0.23: probs.min = 1e-6 (CBPSBinary.R line 4)
    R optim(): ndeps default = 1e-3
    """

    # Propensity Score Clipping
    probs_min: float = 1e-6  # P(T|X) lower bound [R: probs.min]
    probs_trim_msm: float = 1e-4  # CBMSM probability trim threshold

    # Column Detection
    const_col_threshold: float = 1e-10  # Std below this = constant column

    # Optimization
    ndeps: float = 1e-3  # Finite difference step [R: optim default]
    glm_tol: float = 1e-8  # GLM IRLS convergence tolerance
    optim_xtol: float = 1e-12  # Parameter convergence tolerance

    # SVD / Matrix
    svd_threshold_msm: float = 1e-4  # MSM singular value cutoff
    log_clip_range: float = 50.0  # |x|>50 → clip before exp(x)

    @property
    def probs_max(self) -> float:
        """Upper bound for propensity score clipping: 1 - probs_min."""
        return 1.0 - self.probs_min


# Global default instance
DEFAULT_CONFIG = NumericalConfig()
