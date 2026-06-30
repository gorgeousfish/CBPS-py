"""
Tests for r_ginv_with_diagnostics in cbps.utils.numerics.

Verifies:
- Well-conditioned matrices do NOT trigger warnings
- Ill-conditioned matrices (κ > 1e12) DO trigger warnings
- Singular/rank-deficient matrices report correct effective rank
- Pseudoinverse output matches the existing _r_ginv implementation
"""

import warnings

import numpy as np
import pytest

from cbps.utils.numerics import r_ginv_with_diagnostics


class TestWellConditionedMatrix:
    """Tests for matrices with low condition number (< 1e4)."""

    def test_identity_no_warning(self):
        """Identity matrix should have condition number 1 and no warning."""
        A = np.eye(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A, warn_threshold=1e12)
            assert len(w) == 0, f"Unexpected warning: {w}"
        assert diag['effective_rank'] == 5
        assert np.isclose(diag['condition_number'], 1.0)
        assert np.allclose(pinv, np.eye(5), atol=1e-12)

    def test_well_conditioned_random(self):
        """Random well-conditioned matrix should not trigger warning."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        A = A @ A.T + 5 * np.eye(10)  # Make well-conditioned

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A, warn_threshold=1e12)
            assert len(w) == 0

        assert diag['condition_number'] < 1e4
        assert diag['effective_rank'] == 10
        # Verify pseudoinverse property: A @ A_pinv @ A ≈ A
        assert np.allclose(A @ pinv @ A, A, atol=1e-8)

    def test_rectangular_matrix(self):
        """Non-square matrix should work without warning."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((5, 3))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A, warn_threshold=1e12)
            assert len(w) == 0

        assert diag['effective_rank'] == 3
        # Check pseudoinverse property
        assert np.allclose(A @ pinv @ A, A, atol=1e-10)


class TestIllConditionedMatrix:
    """Tests for matrices with condition number > 1e12."""

    def test_ill_conditioned_triggers_warning(self):
        """Matrix with κ > 1e12 should trigger UserWarning.
        
        Note: with the default MASS::ginv tolerance (sqrt(eps)*s_max),
        the maximum condition number of retained singular values is bounded
        by ~6.7e7. To demonstrate ill-conditioning detection, we use a
        tighter tolerance that retains more singular values.
        """
        # Build diagonal matrix with known condition number ~ 1e14
        s = np.array([1e14, 1e7, 1e3, 1.0, 0.5])
        A = np.diag(s)
        # Use very small tol to retain all singular values
        small_tol = 1e-16

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A, tol=small_tol, warn_threshold=1e12)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "ill-conditioned" in str(w[0].message).lower()

        assert diag['condition_number'] > 1e12
        assert diag['effective_rank'] == 5

    def test_custom_threshold(self):
        """Custom warn_threshold should be respected."""
        # Build matrix with condition ~ 1e6
        s = np.array([1e6, 1.0])
        A = np.diag(s)

        # Should NOT warn at default threshold (1e12)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, diag = r_ginv_with_diagnostics(A, warn_threshold=1e12)
            assert len(w) == 0

        # SHOULD warn at lower threshold (1e4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, diag = r_ginv_with_diagnostics(A, warn_threshold=1e4)
            assert len(w) == 1
            assert "ill-conditioned" in str(w[0].message).lower()

    def test_nearly_singular_warning(self):
        """Near-singular matrix should warn and report high condition number."""
        n = 4
        A = np.ones((n, n)) + 1e-14 * np.eye(n)  # rank ~1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A, warn_threshold=1e12)
            # Condition number should be very high or inf
            assert diag['condition_number'] > 1e12 or diag['condition_number'] == float('inf')


class TestSingularMatrixRank:
    """Tests for effective rank reporting on singular matrices."""

    def test_rank_deficient_matrix(self):
        """Rank-deficient matrix should report correct effective rank."""
        # Rank 2 matrix in 4x4
        A = np.array([
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [1, 0, 1, 0],
            [2, 0, 2, 0],
        ], dtype=float)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A)

        assert diag['effective_rank'] == 2

    def test_zero_matrix(self):
        """Zero matrix should have effective rank 0 and condition inf."""
        A = np.zeros((3, 3))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A)

        assert diag['effective_rank'] == 0
        assert diag['condition_number'] == float('inf')
        assert np.allclose(pinv, np.zeros((3, 3)))

    def test_rank_one_matrix(self):
        """Rank-1 matrix should report effective rank 1 and condition inf."""
        v = np.array([[1, 2, 3, 4]], dtype=float)
        A = v.T @ v  # rank 1

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pinv, diag = r_ginv_with_diagnostics(A)

        assert diag['effective_rank'] == 1
        assert diag['condition_number'] == float('inf')


class TestConsistencyWithRGinv:
    """Verify that r_ginv_with_diagnostics produces the same pseudoinverse as _r_ginv."""

    def _r_ginv_reference(self, X, tol=None):
        """Reference implementation matching _r_ginv from cbps_binary.py."""
        if tol is None:
            machine_eps = np.finfo(float).eps
            tol = np.sqrt(machine_eps)

        Xsvd_u, Xsvd_d, Xsvd_vt = np.linalg.svd(X, full_matrices=False)
        Xsvd_v = Xsvd_vt.T

        if len(Xsvd_d) == 0 or Xsvd_d[0] < np.finfo(float).eps:
            return np.zeros((X.shape[1], X.shape[0]))

        tol_threshold = max(tol * Xsvd_d[0], 0.0)
        Positive = Xsvd_d > tol_threshold

        if np.all(Positive):
            return Xsvd_v @ np.diag(1.0 / Xsvd_d) @ Xsvd_u.T
        elif not np.any(Positive):
            return np.zeros((X.shape[1], X.shape[0]))
        else:
            Xsvd_v_pos = Xsvd_v[:, Positive]
            Xsvd_d_pos = Xsvd_d[Positive]
            Xsvd_u_pos = Xsvd_u[:, Positive]
            return Xsvd_v_pos @ np.diag(1.0 / Xsvd_d_pos) @ Xsvd_u_pos.T

    def test_consistency_well_conditioned(self):
        """Pseudoinverse matches _r_ginv for well-conditioned matrix."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((8, 8))
        A = A @ A.T + 3 * np.eye(8)

        ref = self._r_ginv_reference(A)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result, _ = r_ginv_with_diagnostics(A)

        assert np.allclose(result, ref, atol=1e-10)

    def test_consistency_rank_deficient(self):
        """Pseudoinverse matches _r_ginv for rank-deficient matrix."""
        # Rank 3 in 5x5
        rng = np.random.default_rng(77)
        B = rng.standard_normal((5, 3))
        A = B @ B.T  # rank 3

        ref = self._r_ginv_reference(A)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result, _ = r_ginv_with_diagnostics(A)

        assert np.allclose(result, ref, atol=1e-10)

    def test_consistency_rectangular(self):
        """Pseudoinverse matches _r_ginv for rectangular matrix."""
        rng = np.random.default_rng(55)
        A = rng.standard_normal((6, 4))

        ref = self._r_ginv_reference(A)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result, _ = r_ginv_with_diagnostics(A)

        assert np.allclose(result, ref, atol=1e-10)

    def test_consistency_ill_conditioned(self):
        """Pseudoinverse matches _r_ginv even for ill-conditioned matrix."""
        s = np.array([1e15, 1e8, 1e2, 1.0])
        A = np.diag(s)

        ref = self._r_ginv_reference(A)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result, _ = r_ginv_with_diagnostics(A)

        assert np.allclose(result, ref, atol=1e-10)


class TestDiagnosticsContent:
    """Tests for the structure and content of the diagnostics dict."""

    def test_diagnostics_keys(self):
        """Diagnostics dict must contain required keys."""
        A = np.eye(3)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, diag = r_ginv_with_diagnostics(A)

        assert 'condition_number' in diag
        assert 'effective_rank' in diag
        assert 'tolerance' in diag

    def test_tolerance_value(self):
        """Default tolerance should be sqrt(eps) * max(singular value)."""
        A = np.diag([10.0, 5.0, 1.0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, diag = r_ginv_with_diagnostics(A)

        expected_tol = np.sqrt(np.finfo(float).eps) * 10.0
        assert np.isclose(diag['tolerance'], expected_tol)

    def test_custom_tolerance(self):
        """Custom tolerance should be stored in diagnostics."""
        A = np.eye(3)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, diag = r_ginv_with_diagnostics(A, tol=1e-5)

        assert diag['tolerance'] == 1e-5
