"""Tests for P2-5 (sparse matrix support) and P2-8 (logging system).

Tests cover:
- Logger does not produce output by default (NullHandler)
- set_verbosity(1) enables INFO output
- set_verbosity(2) enables DEBUG output
- Sparse matrix inputs are correctly converted to dense
- Conversion preserves numerical results
"""
import logging
import warnings

import numpy as np
import pytest

from cbps.logging_config import logger, set_verbosity
from cbps.utils.validation import ensure_dense


# ===========================================================================
# P2-8: Logging System Tests
# ===========================================================================

class TestLoggingSystem:
    """Tests for the CBPS logging infrastructure."""

    def setup_method(self):
        """Reset logger state before each test."""
        # Remove all handlers except NullHandler
        logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

    def test_logger_no_default_output(self, capfd):
        """Logger should not produce output by default (NullHandler only)."""
        logger.info("This should not appear")
        logger.debug("This should not appear either")
        captured = capfd.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_set_verbosity_level_1(self, capfd):
        """set_verbosity(1) should enable INFO messages."""
        set_verbosity(1)
        logger.info("Test info message")
        captured = capfd.readouterr()
        assert "Test info message" in captured.err
        assert "[CBPS] INFO" in captured.err

    def test_set_verbosity_level_2(self, capfd):
        """set_verbosity(2) should enable DEBUG messages."""
        set_verbosity(2)
        logger.debug("Test debug message")
        captured = capfd.readouterr()
        assert "Test debug message" in captured.err
        assert "[CBPS] DEBUG" in captured.err

    def test_set_verbosity_level_0_suppresses(self, capfd):
        """set_verbosity(0) should suppress INFO/DEBUG messages."""
        set_verbosity(1)  # first enable
        set_verbosity(0)  # then suppress
        logger.info("This should not appear")
        captured = capfd.readouterr()
        assert "This should not appear" not in captured.err

    def test_warning_always_visible(self, capfd):
        """WARNING level messages should always be visible after set_verbosity."""
        set_verbosity(0)
        logger.warning("Test warning")
        captured = capfd.readouterr()
        # WARNING goes through since level >= WARNING
        assert "Test warning" in captured.err

    def test_no_duplicate_handlers(self):
        """Calling set_verbosity multiple times should not add duplicate handlers."""
        set_verbosity(1)
        set_verbosity(1)
        set_verbosity(2)
        # Count non-NullHandler StreamHandlers
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1

    def test_set_verbosity_importable_from_cbps(self):
        """set_verbosity should be importable from the top-level cbps package."""
        from cbps import set_verbosity as sv
        assert callable(sv)


# ===========================================================================
# P2-5: Sparse Matrix Support Tests
# ===========================================================================

class TestSparseDenseConversion:
    """Tests for sparse-to-dense auto-conversion."""

    def test_dense_array_passthrough(self):
        """Dense numpy array should pass through unchanged."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = ensure_dense(X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, X)

    def test_sparse_csr_converted(self):
        """scipy.sparse CSR matrix should be converted to dense."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        X_dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        X_sparse = scipy_sparse.csr_matrix(X_dense)
        result = ensure_dense(X_sparse)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, X_dense)

    def test_sparse_csc_converted(self):
        """scipy.sparse CSC matrix should be converted to dense."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        X_dense = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        X_sparse = scipy_sparse.csc_matrix(X_dense)
        result = ensure_dense(X_sparse)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, X_dense)

    def test_sparse_large_dimension_warning(self):
        """Sparse matrix with >1000 columns should emit a warning."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        # Create a 10x1500 sparse matrix
        X_sparse = scipy_sparse.random(10, 1500, density=0.01, format='csr')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ensure_dense(X_sparse)
            assert len(w) == 1
            assert "hdCBPS" in str(w[0].message)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 1500)

    def test_sparse_small_dimension_no_warning(self):
        """Sparse matrix with <=1000 columns should not emit a warning."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        X_sparse = scipy_sparse.random(100, 50, density=0.1, format='csr')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ensure_dense(X_sparse)
            sparse_warnings = [x for x in w if "hdCBPS" in str(x.message)]
            assert len(sparse_warnings) == 0
        assert isinstance(result, np.ndarray)

    def test_list_input_converted(self):
        """List input should be converted to numpy array."""
        X = [[1, 2], [3, 4]]
        result = ensure_dense(X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(X))

    def test_sparse_input_to_cbps_binary(self):
        """Sparse matrix input to cbps_binary_fit should work correctly."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from cbps.core.cbps_binary import cbps_binary_fit

        # Create simple test data
        np.random.seed(42)
        n = 200
        X_dense = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        treat = (np.random.randn(n) + X_dense[:, 1] > 0).astype(float)

        # Fit with dense input
        result_dense = cbps_binary_fit(treat, X_dense, verbose=0)

        # Fit with sparse input (same data)
        X_sparse = scipy_sparse.csr_matrix(X_dense)
        result_sparse = cbps_binary_fit(treat, X_sparse, verbose=0)

        # Results should be numerically identical
        np.testing.assert_allclose(
            result_dense['coefficients'],
            result_sparse['coefficients'],
            rtol=1e-10
        )
        np.testing.assert_allclose(
            result_dense['fitted_values'],
            result_sparse['fitted_values'],
            rtol=1e-10
        )
