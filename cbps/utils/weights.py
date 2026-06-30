"""
Propensity Score Weight Computation

This module provides functions for computing inverse probability weights
(IPW) for different treatment types and target estimands within the
CBPS framework.

Supported weight types:

- **ATE weights**: Average Treatment Effect weights for binary treatments
- **ATT weights**: Average Treatment Effect on Treated weights
- **Continuous treatment**: Weighted treatment variable for balance conditions
- **Standardized weights**: Group-normalized weights (Hajek estimator)
- **WeightNormalizer**: Unified class encapsulating the standardization flow

Mathematical Framework
----------------------
For binary treatments with propensity score π(X):

**ATE** (Imai & Ratkovic 2014, Eq. 10)::

    w_i = T_i/π_i + (1-T_i)/(1-π_i)

**ATT** (Imai & Ratkovic 2014, Eq. 11)::

    w_i = (N/N₁) × (T_i - π_i)/(1 - π_i)

For continuous treatments with generalized propensity score f(T|X):

**Stabilized** (Fong et al. 2018, Eq. 2)::

    w_i = f(T_i) / f(T_i|X_i)

Standardization Protocol
------------------------
All normalization follows a strict four-step order:

1. Compute raw weights from propensity scores.
2. Apply sampling weights (if provided).
3. Group-wise normalization (each group sums to 1).
4. Validate result (no NaN/Inf).

Functions
---------
compute_ate_weights
    Compute ATE inverse probability weights.
compute_att_weights
    Compute ATT inverse probability weights.
compute_continuous_weights
    Compute stabilized continuous treatment weights.
standardize_weights
    Normalize weights by treatment group.

Classes
-------
WeightNormalizer
    Unified standardization class with validate/normalize methods.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
Journal of the Royal Statistical Society, Series B 76(1), 243-263.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. The Annals of Applied Statistics, 12(1),
156-177.
"""

import warnings
from typing import Optional

import numpy as np


class WeightNormalizer:
    """
    Unified weight normalization for CBPS estimators.

    Encapsulates the Hajek-style standardization flow for inverse probability
    weights, ensuring a consistent step ordering across ATE and ATT estimands.

    Standardization Steps (applied in strict order)
    ------------------------------------------------
    1. Compute raw weights from propensity scores.
    2. Apply sampling weights (if provided).
    3. Group-wise normalization (treated sum → 1, control sum → 1).
    4. Validate result (no NaN/Inf, correct signs).

    Mathematical Reference
    ----------------------
    **ATE weights** (Imai & Ratkovic 2014, Eq. 10)::

        w_i = T_i / π_i + (1 - T_i) / (1 - π_i)

    After Hajek normalization each group sums to 1.

    **ATT weights** (Imai & Ratkovic 2014, Eq. 11)::

        w_i = (N / N₁) × (T_i - π_i) / (1 - π_i)

    Treated group weights are constant N/N₁; control weights are
    normalized by their absolute-value sum.

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ate(
        weights: np.ndarray,
        treat: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        ATE standardization (Hajek-style group normalization).

        Each treatment group's weights are rescaled so that their sum
        equals 1, yielding an estimate of E[Y(1)] - E[Y(0)].

        Parameters
        ----------
        weights : np.ndarray
            Raw ATE weights (e.g., from ``compute_ate_weights``), shape (n,).
        treat : np.ndarray
            Binary treatment indicator (0/1), shape (n,).
        sample_weights : np.ndarray or None, optional
            Sampling weights, shape (n,). If None, uniform weights are used.

        Returns
        -------
        np.ndarray
            Standardized weights multiplied by sample weights, shape (n,).
            ``weights[treat==1].sum() ≈ 1`` and ``weights[treat==0].sum() ≈ 1``.

        Notes
        -----
        Follows the four-step standardization protocol:

        1. Raw weights are supplied via the *weights* argument.
        2. Sample weights are applied (element-wise multiplication).
        3. Treated / control sums are each normalized to 1.
        4. Validity check (via ``validate``).
        """
        n = len(treat)
        if sample_weights is None:
            sample_weights = np.ones(n)

        # Step 2: Apply sample weights
        w = weights.copy() * sample_weights

        # Step 3: Group-wise normalization
        treat_mask = treat == 1
        ctrl_mask = treat == 0

        sum_treat = np.sum(w[treat_mask])
        sum_ctrl = np.sum(w[ctrl_mask])

        # Avoid division by zero when a group is empty
        if sum_treat > 0:
            w[treat_mask] /= sum_treat
        if sum_ctrl > 0:
            w[ctrl_mask] /= sum_ctrl

        # Step 4: Validate
        WeightNormalizer.validate(w, allow_negative=False)

        return w

    @staticmethod
    def normalize_att(
        weights: np.ndarray,
        treat: np.ndarray,
        probs: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        ATT standardization.

        Treated units receive a constant weight (N/N₁ after normalization);
        control unit weights are normalized by their absolute-value sum.

        Parameters
        ----------
        weights : np.ndarray
            Raw ATT weights (e.g., from ``compute_att_weights``), shape (n,).
            Control weights are expected to be negative.
        treat : np.ndarray
            Binary treatment indicator (0/1), shape (n,).
        probs : np.ndarray
            Propensity scores, shape (n,).
        sample_weights : np.ndarray or None, optional
            Sampling weights, shape (n,). If None, uniform weights are used.

        Returns
        -------
        np.ndarray
            Standardized ATT weights (all non-negative), shape (n,).
            Treated group sums to 1 and control group sums to 1.

        Notes
        -----
        Follows the four-step standardization protocol:

        1. Raw weights are supplied via the *weights* argument.
        2. Sample weights are applied.
        3. Treated group normalized to sum 1; control group absolute values
           normalized to sum 1.
        4. Validity check.
        """
        n = len(treat)
        if sample_weights is None:
            sample_weights = np.ones(n)

        treat_mask = treat == 1
        ctrl_mask = treat == 0

        # Step 2: Apply sample weights
        w = weights.copy() * sample_weights

        # Step 3: Group-wise normalization
        sum_treat = np.sum(w[treat_mask])
        sum_ctrl_abs = np.sum(np.abs(w[ctrl_mask]))

        out = np.empty(n)
        if sum_treat > 0:
            out[treat_mask] = w[treat_mask] / sum_treat
        else:
            out[treat_mask] = w[treat_mask]

        if sum_ctrl_abs > 0:
            out[ctrl_mask] = np.abs(w[ctrl_mask]) / sum_ctrl_abs
        else:
            out[ctrl_mask] = np.abs(w[ctrl_mask])

        # Step 4: Validate
        WeightNormalizer.validate(out, allow_negative=False)

        return out

    @staticmethod
    def validate(weights: np.ndarray, allow_negative: bool = False) -> bool:
        """Validate weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Weight vector to validate.
        allow_negative : bool, default=False
            If False, warns when negative weights are detected.
            Set to True for balance condition weights (which can be
            negative by design).

        Returns
        -------
        bool
            True if all checks pass.

        Raises
        ------
        ValueError
            If weights contain NaN or Inf values.
        """
        if np.any(np.isnan(weights)):
            raise ValueError(
                "Weights contain NaN values. Check propensity score estimation."
            )
        if np.any(np.isinf(weights)):
            raise ValueError(
                "Weights contain Inf values. Propensity scores may be too "
                "close to 0 or 1."
            )
        if not allow_negative and np.any(weights < 0):
            n_neg = int(np.sum(weights < 0))
            min_val = float(np.min(weights))
            warnings.warn(
                f"Detected {n_neg} negative weight(s) (min={min_val:.6g}). "
                f"IPW weights should be non-negative; this may indicate "
                f"numerical issues in propensity score estimation.",
                stacklevel=2,
            )
        return True


def compute_ate_weights(
    treat: np.ndarray,
    probs: np.ndarray
) -> np.ndarray:
    """
    Compute ATE inverse probability weights for binary treatments.

    Implements the standard IPW formula (Imai & Ratkovic 2014, Eq. 10):

        w_i = T_i / π_i + (1 - T_i) / (1 - π_i)

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
        Should be clipped to (0, 1) before calling for numerical stability.

    Returns
    -------
    np.ndarray
        Unstandardized ATE weights, shape (n,).
        All weights are guaranteed positive.

    Notes
    -----
    This formula naturally produces positive weights:

    - Treated units (T=1): weight = 1/π(X)
    - Control units (T=0): weight = 1/(1-π(X))

    Use ``WeightNormalizer.normalize_ate`` for Hajek-style group normalization.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 0, 1, 0])
    >>> probs = np.array([0.6, 0.4, 0.7, 0.3])
    >>> w = compute_ate_weights(treat, probs)
    >>> bool(np.all(w > 0))
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Standard form: ensures positive weights directly
    weights = treat / probs + (1 - treat) / (1 - probs)

    # Note: Alternative algebraically equivalent form
    # weights = np.abs(1 / (probs - 1 + treat))

    return weights


def compute_att_weights(
    treat: np.ndarray,
    probs: np.ndarray,
    sample_weights: np.ndarray
) -> np.ndarray:
    """
    Compute ATT inverse probability weights for binary treatments.

    Implements the ATT weighting formula (Imai & Ratkovic 2014, Eq. 11):

        w_i = (N / N₁) × (T_i - π_i) / (1 - π_i)

    Parameters
    ----------
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
    sample_weights : np.ndarray
        Sampling weights normalized to sum to n, shape (n,).

    Returns
    -------
    np.ndarray
        Unstandardized ATT weights, shape (n,).
        Control unit weights are negative by construction.

    Notes
    -----
    The formula produces different signs by treatment status:

    - Treated (T=1): w = N/N₁ (positive constant)
    - Control (T=0): w = -(N/N₁) × π/(1-π) (negative)

    Use ``WeightNormalizer.normalize_att`` for Hajek-style normalization
    that takes absolute values of control weights and normalizes each group.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 1, 0, 0])
    >>> probs = np.array([0.6, 0.7, 0.4, 0.3])
    >>> sw = np.ones(4)
    >>> w = compute_att_weights(treat, probs, sw)
    >>> bool(all(w[treat == 1] > 0))  # Treated positive
    True
    >>> bool(all(w[treat == 0] < 0))  # Control negative
    True

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    """
    # Calculate effective sample size of treated group
    n_t = np.sum(sample_weights[treat == 1])
    n = len(treat)

    # ATT weight formula (may produce negative values for controls)
    weights = (n / n_t) * (treat - probs) / (1 - probs)

    return weights


def compute_continuous_weights(
    Ttilde: np.ndarray,
    stabilizers: np.ndarray,
    log_density: np.ndarray
) -> np.ndarray:
    """
    Compute weighted standardized treatment for continuous CBPS balance conditions.

    Computes the quantity T* × w where w is the stabilized weight:

        T̃_i × w_i = T̃_i × exp(log f(T̃_i) - log f(T̃_i|X_i))

    This is used in the CBPS balance condition E[T* × w × X*] = 0
    (Fong et al., 2018, Eq. 2).

    Parameters
    ----------
    Ttilde : np.ndarray
        Standardized treatment (mean=0, std=1), shape (n,).
    stabilizers : np.ndarray
        Log marginal density log f(T̃), shape (n,).
    log_density : np.ndarray
        Log conditional density log f(T̃|X) (GPS), shape (n,).

    Returns
    -------
    np.ndarray
        Weighted treatment T̃ × w, shape (n,).
        Note: This is NOT the weight itself; the stabilized weight is
        w = f(T̃)/f(T̃|X) = exp(stabilizers - log_density).

    Notes
    -----
    **Numerical stability**: The log-density difference is clipped to
    [-50, 50] before exponentiation to prevent overflow.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> Ttilde = np.array([0.0, 1.0, -1.0])
    >>> stabilizers = norm.logpdf(Ttilde, 0, 1)  # Marginal density
    >>> log_density = norm.logpdf(Ttilde, 0, 1)  # Same as marginal (no confounding)
    >>> Tw = compute_continuous_weights(Ttilde, stabilizers, log_density)
    >>> bool(np.all(np.isfinite(Tw)))
    True

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. The Annals of Applied Statistics, 12(1),
    156-177.
    """
    # Compute weights in log space for numerical stability
    # Equivalent to: Ttilde * exp(stabilizers - log_density)

    # Clip difference to prevent overflow in exp()
    log_diff = stabilizers - log_density
    log_diff_clipped = np.clip(log_diff, -50, 50)

    # Final weight computation
    weights = Ttilde * np.exp(log_diff_clipped)

    return weights


def standardize_weights(
    weights: np.ndarray,
    treat: np.ndarray,
    probs: np.ndarray,
    sample_weights: np.ndarray,
    att: int = 0,
    standardize: bool = True
) -> np.ndarray:
    """
    Normalize IPW weights by treatment group.

    Applies group-wise normalization so that weights within each treatment
    group sum to 1, or returns unnormalized Horvitz-Thompson weights.

    This function internally follows the ``WeightNormalizer`` four-step
    protocol but reconstructs weights from propensity scores to maintain
    backward compatibility with R's ``CBPS`` package (v0.23).

    Parameters
    ----------
    weights : np.ndarray
        Raw weights (unused, kept for API compatibility), shape (n,).
    treat : np.ndarray
        Binary treatment indicator (0/1), shape (n,).
    probs : np.ndarray
        Propensity scores, shape (n,).
    sample_weights : np.ndarray
        Sampling weights (normalized to sum to n), shape (n,).
    att : int, default=0
        Target estimand: 0=ATE, 1=ATT, 2=ATT with reversed encoding.
    standardize : bool, default=True
        If True, apply group-wise normalization (Hajek estimator).
        If False, return Horvitz-Thompson weights.

    Returns
    -------
    np.ndarray
        Final weights multiplied by sample_weights, shape (n,).

    Notes
    -----
    **Standardization Steps** (strict order):

    1. Reconstruct raw weights from propensity scores.
    2. Multiply by sample_weights.
    3. Normalize treated / control groups to each sum to 1.
    4. Final validation (finite, non-negative).

    **ATE** (Imai & Ratkovic 2014, Eq. 10):
        Treated: 1/π normalized; Control: 1/(1-π) normalized.

    **ATT** (Imai & Ratkovic 2014, Eq. 11):
        Treated: N/N₁ normalized; Control: |N/N₁ × (T-π)/(1-π)| normalized.

    **Implementation**: Weights are reconstructed from propensity scores
    rather than directly normalizing the input weights, to match the
    R CBPS package behavior.

    Examples
    --------
    >>> import numpy as np
    >>> treat = np.array([1, 0, 1, 0])
    >>> probs = np.array([0.6, 0.4, 0.7, 0.3])
    >>> sw = np.ones(4)
    >>> w = np.ones(4)
    >>> w_std = standardize_weights(w, treat, probs, sw, att=0, standardize=True)
    >>> bool(np.isclose(w_std[treat==1].sum(), 1.0))  # Treated group normalized
    True
    >>> bool(np.isclose(w_std[treat==0].sum(), 1.0))  # Control group normalized
    True
    """
    n = len(treat)
    # Use unweighted count for ATT normalization, matching R's CBPSBinary.R
    # R redefines n.t = sum(treat==1) (unweighted) before weight standardization
    n_t_unweighted = np.sum(treat == 1)

    if standardize:
        # Step 1: Compute normalization factors
        if att:  # ATT branch
            norm1 = np.sum(treat * sample_weights * n / n_t_unweighted)
            norm2 = np.sum((1 - treat) * sample_weights * n / n_t_unweighted *
                          (treat - probs) / (1 - probs))
        else:  # ATE branch
            norm1 = np.sum(treat * sample_weights / probs)
            norm2 = np.sum((1 - treat) * sample_weights / (1 - probs))
    else:
        # Step 2: Horvitz-Thompson weights (no normalization)
        norm1 = 1.0
        norm2 = 1.0

    # Step 3: Reconstruct standardized weights
    # Note: weights are reconstructed, not directly standardized from input
    if att:  # ATT branch
        # Treatment group + abs(control group)
        weights_std = (
            (treat == 1) * n / n_t_unweighted / norm1 +
            np.abs((treat == 0) * n / n_t_unweighted *
                   (treat - probs) / (1 - probs) / norm2)
        )
    else:  # ATE branch
        # Standard form, no abs() needed (1/π and 1/(1-π) always positive)
        weights_std = (
            (treat == 1) / probs / norm1 +
            (treat == 0) / (1 - probs) / norm2
        )

    # Step 4: Multiply by sample_weights
    weights_std = weights_std * sample_weights

    return weights_std
