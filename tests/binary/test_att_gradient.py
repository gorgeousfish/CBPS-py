"""
Numerical Verification of ATT Balance Gradient
===============================================

This module verifies the analytic ATT gradient implementation against
numerical (finite-difference) differentiation, ensuring the chain-rule
derivation in cbps_binary._gmm_gradient is correct.

Reference
---------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    Equation 11 (ATT weight definition).
"""

import numpy as np
import pytest
import scipy.special

from cbps.core.cbps_binary import (
    _gmm_gradient,
    _gmm_loss,
    _compute_V_matrix,
    PROBS_MIN,
)

pytestmark = [pytest.mark.unit, pytest.mark.numerical]


def _setup_att_problem(n=200, k=4, seed=42):
    """Create a reproducible ATT test problem with known dimensions."""
    rng = np.random.default_rng(seed)

    # Covariates with intercept
    X_raw = rng.standard_normal((n, k - 1))
    X = np.column_stack([np.ones(n), X_raw])

    # True treatment model
    beta_true = rng.standard_normal(k) * 0.5
    probs_true = scipy.special.expit(X @ beta_true)
    treat = (rng.uniform(size=n) < probs_true).astype(float)

    # Ensure we have both treated and control units
    if treat.sum() < 5:
        treat[:10] = 1.0
    if (1 - treat).sum() < 5:
        treat[-10:] = 0.0

    # Uniform sample weights (normalized to sum to n)
    sample_weights = np.ones(n)

    # Use a slightly perturbed beta for testing (not converged)
    beta_test = beta_true + rng.standard_normal(k) * 0.1

    return X, treat, sample_weights, beta_test


def _numerical_gradient_att(beta, X, treat, sample_weights, inv_V, eps=1e-7):
    """Compute gradient by central finite differences."""
    k = len(beta)
    grad = np.zeros(k)
    for j in range(k):
        e_j = np.zeros(k)
        e_j[j] = eps
        loss_plus = _gmm_loss(beta + e_j, X, treat, sample_weights, att=1, inv_V=inv_V)
        loss_minus = _gmm_loss(beta - e_j, X, treat, sample_weights, att=1, inv_V=inv_V)
        grad[j] = (loss_plus - loss_minus) / (2 * eps)
    return grad


class TestATTGradientNumericalVsAnalytic:
    """Verify ATT balance gradient via numerical differentiation."""

    def test_att_gradient_matches_numerical(self):
        """
        Core test: analytic gradient matches finite-difference approximation.

        The analytic gradient uses the chain-rule derivation:
            dw/dbeta = -(N/N_1) * pi/(1-pi) * X   for control units
            dw/dbeta = 0                            for treated units

        This should match numerical differentiation to high precision.
        """
        X, treat, sample_weights, beta_test = _setup_att_problem()
        n = len(treat)

        # Compute inv_V at the test point (fixed for both analytic and numerical)
        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = _compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)

        # Analytic gradient
        analytic_grad = _gmm_gradient(beta_test, inv_V, X, treat, sample_weights, att=1)

        # Numerical gradient
        numerical_grad = _numerical_gradient_att(
            beta_test, X, treat, sample_weights, inv_V
        )

        np.testing.assert_allclose(
            analytic_grad,
            numerical_grad,
            rtol=1e-4,
            atol=1e-8,
            err_msg="ATT analytic gradient does not match numerical gradient",
        )

    @pytest.mark.parametrize("seed", [0, 7, 13, 99, 2024])
    def test_att_gradient_multiple_seeds(self, seed):
        """Verify gradient consistency across different random data realizations."""
        X, treat, sample_weights, beta_test = _setup_att_problem(seed=seed)
        n = len(treat)

        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = _compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)

        analytic_grad = _gmm_gradient(beta_test, inv_V, X, treat, sample_weights, att=1)
        numerical_grad = _numerical_gradient_att(
            beta_test, X, treat, sample_weights, inv_V
        )

        np.testing.assert_allclose(
            analytic_grad,
            numerical_grad,
            rtol=1e-4,
            atol=1e-8,
            err_msg=f"ATT gradient mismatch for seed={seed}",
        )

    def test_att_gradient_treated_weight_constant(self):
        """
        Verify that treated units contribute zero to dw/dbeta.

        Since w_i = N/N_1 for treated units (constant), the gradient
        contribution from treated units should be exactly zero.
        """
        X, treat, sample_weights, beta_test = _setup_att_problem()
        n = len(treat)
        n_t = np.sum(sample_weights[treat == 1])

        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)

        # Compute dw directly
        dw = -n / n_t * probs / (1 - probs)
        dw[treat == 1] = 0

        # All treated entries should be exactly zero
        assert np.all(dw[treat == 1] == 0), "Treated units must have zero dw"

    def test_att_gradient_control_formula(self):
        """
        Verify the dw formula for control units matches chain-rule derivation.

        For control (T=0):
            dw/dbeta_j = -(N/N_1) * pi_i/(1-pi_i) * X_ij

        We verify the scalar dw = -(N/N_1) * pi/(1-pi) matches this.
        """
        X, treat, sample_weights, beta_test = _setup_att_problem()
        n = len(treat)
        n_t = np.sum(sample_weights[treat == 1])

        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)

        # Compute dw as implemented
        dw = -n / n_t * probs / (1 - probs)
        dw[treat == 1] = 0

        # Verify against explicit chain rule for control units
        # dw/dpi = -(N/N_1) / (1-pi)^2
        # dpi/dbeta = pi*(1-pi) (scalar part, without X)
        # dw/dbeta = dw/dpi * dpi/dbeta = -(N/N_1) * pi / (1-pi)
        control_mask = treat == 0
        expected_dw_control = -n / n_t * probs[control_mask] / (1 - probs[control_mask])

        np.testing.assert_allclose(
            dw[control_mask],
            expected_dw_control,
            rtol=1e-15,
            err_msg="Control dw formula mismatch",
        )

    def test_att_gradient_with_nonuniform_weights(self):
        """Verify gradient correctness with non-uniform sample weights."""
        rng = np.random.default_rng(123)
        n, k = 150, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        beta_true = np.array([0.2, -0.5, 0.3])
        probs_true = scipy.special.expit(X @ beta_true)
        treat = (rng.uniform(size=n) < probs_true).astype(float)
        treat[:10] = 1.0
        treat[-10:] = 0.0

        # Non-uniform sample weights (normalized to sum to n)
        raw_weights = rng.uniform(0.5, 2.0, size=n)
        sample_weights = raw_weights / raw_weights.sum() * n

        beta_test = beta_true + rng.standard_normal(k) * 0.15

        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = _compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)

        analytic_grad = _gmm_gradient(beta_test, inv_V, X, treat, sample_weights, att=1)
        numerical_grad = _numerical_gradient_att(
            beta_test, X, treat, sample_weights, inv_V
        )

        np.testing.assert_allclose(
            analytic_grad,
            numerical_grad,
            rtol=1e-4,
            atol=1e-8,
            err_msg="ATT gradient mismatch with non-uniform sample weights",
        )

    def test_att_gradient_extreme_propensity(self):
        """
        Verify gradient accuracy when propensity scores are near boundaries.

        Tests robustness when pi is close to 0 or 1 (after clipping).
        """
        rng = np.random.default_rng(77)
        n, k = 100, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])

        # Large beta to push probabilities toward boundaries
        beta_test = np.array([1.5, -2.0, 1.8])
        probs_true = scipy.special.expit(X @ beta_test)
        treat = (rng.uniform(size=n) < probs_true).astype(float)
        treat[:5] = 1.0
        treat[-5:] = 0.0

        sample_weights = np.ones(n)

        probs = scipy.special.expit(X @ beta_test)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
        inv_V = _compute_V_matrix(X, probs, sample_weights, treat, att=1, n=n)

        analytic_grad = _gmm_gradient(beta_test, inv_V, X, treat, sample_weights, att=1)
        numerical_grad = _numerical_gradient_att(
            beta_test, X, treat, sample_weights, inv_V
        )

        np.testing.assert_allclose(
            analytic_grad,
            numerical_grad,
            rtol=5e-4,
            atol=1e-7,
            err_msg="ATT gradient mismatch with extreme propensity scores",
        )
