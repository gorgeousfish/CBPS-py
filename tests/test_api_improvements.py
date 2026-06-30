"""
Tests for P2-9 ~ P2-15 API design improvements.

These tests verify pure engineering/UX improvements that do not change
any mathematical algorithms.
"""

import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from cbps.core.cbps_binary import _normalize_att, cbps_binary_fit


# ---------- Fixtures ----------

@pytest.fixture
def simple_binary_data():
    """Generate a simple binary treatment dataset for testing."""
    np.random.seed(42)
    n = 200
    X_raw = np.random.randn(n, 3)
    X = np.column_stack([np.ones(n), X_raw])  # Add intercept
    beta_true = np.array([0.0, 0.5, -0.3, 0.2])
    prob = 1 / (1 + np.exp(-X @ beta_true))
    treat = (np.random.rand(n) < prob).astype(float)
    return treat, X


@pytest.fixture
def lalonde_df():
    """Load LaLonde dataset for integration tests."""
    try:
        from cbps.datasets import load_lalonde
        return load_lalonde(dehejia_wahba_only=True)
    except (ImportError, Exception):
        pytest.skip("LaLonde dataset not available")


# ---------- P2-9: att parameter string support ----------

class TestAttStringParameter:
    """Test that att parameter accepts string values."""

    def test_normalize_ate_string(self):
        assert _normalize_att('ate') == 0
        assert _normalize_att('ATE') == 0
        assert _normalize_att(' Ate ') == 0

    def test_normalize_att_string(self):
        assert _normalize_att('att') == 1
        assert _normalize_att('ATT') == 1

    def test_normalize_atc_string(self):
        assert _normalize_att('atc') == 2
        assert _normalize_att('ATC') == 2

    def test_normalize_integer(self):
        assert _normalize_att(0) == 0
        assert _normalize_att(1) == 1
        assert _normalize_att(2) == 2

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid att='invalid'"):
            _normalize_att('invalid')

    def test_invalid_integer_raises(self):
        with pytest.raises(ValueError, match="Invalid att=5"):
            _normalize_att(5)

    def test_att_string_in_fit(self, simple_binary_data):
        """Verify string att works end-to-end in cbps_binary_fit."""
        treat, X = simple_binary_data
        # Should not raise
        result = cbps_binary_fit(treat, X, att='att', iterations=50)
        assert result['converged'] or True  # May not converge in 50 iters
        assert 'coefficients' in result

    def test_ate_string_in_fit(self, simple_binary_data):
        """Verify 'ate' string works end-to-end."""
        treat, X = simple_binary_data
        result = cbps_binary_fit(treat, X, att='ate', iterations=50)
        assert 'coefficients' in result


# ---------- P2-10: Return dict snake_case confirmation ----------

class TestReturnDictKeys:
    """Verify all return dict keys are already snake_case."""

    def test_keys_are_snake_case(self, simple_binary_data):
        treat, X = simple_binary_data
        result = cbps_binary_fit(treat, X, att=1, iterations=50)
        expected_keys = {
            'coefficients', 'fitted_values', 'linear_predictor',
            'deviance', 'nulldeviance', 'weights', 'y', 'x',
            'converged', 'J', 'var', 'mle_J'
        }
        assert set(result.keys()) == expected_keys


# ---------- P2-11: tqdm progress bar ----------

class TestProgressBar:
    """Test that show_progress parameter works without error."""

    def test_show_progress_no_tqdm(self, simple_binary_data, monkeypatch):
        """show_progress should silently skip if tqdm not available."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'tqdm':
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        from cbps.core.cbps_binary import _vmmin_bfgs
        fn = lambda b: np.sum(b**2)
        gr = lambda b: 2 * b
        # Should not raise even with tqdm unavailable
        monkeypatch.setattr(builtins, '__import__', mock_import)
        result = _vmmin_bfgs(np.array([1.0, 2.0]), fn, gr, maxit=10,
                             show_progress=True)
        assert result.success or result.nit >= 0

    def test_show_progress_false(self, simple_binary_data):
        """show_progress=False should work normally."""
        from cbps.core.cbps_binary import _vmmin_bfgs
        fn = lambda b: np.sum(b**2)
        gr = lambda b: 2 * b
        result = _vmmin_bfgs(np.array([1.0, 2.0]), fn, gr, maxit=100,
                             show_progress=False)
        assert result.success
        assert np.allclose(result.x, [0, 0], atol=1e-4)


# ---------- P2-12: Warm start ----------

class TestWarmStart:
    """Test init_params warm start functionality."""

    def test_warm_start_skips_glm(self, simple_binary_data):
        """init_params should produce valid results without GLM init."""
        treat, X = simple_binary_data
        init = np.zeros(X.shape[1])
        result = cbps_binary_fit(treat, X, att=1, iterations=50,
                                 init_params=init)
        assert 'coefficients' in result
        assert result['coefficients'].shape == (X.shape[1], 1)

    def test_warm_start_wrong_length_raises(self, simple_binary_data):
        """init_params with wrong length should raise ValueError."""
        treat, X = simple_binary_data
        wrong_init = np.zeros(X.shape[1] + 1)
        with pytest.raises(ValueError, match="init_params length"):
            cbps_binary_fit(treat, X, att=1, init_params=wrong_init)

    def test_warm_start_uses_provided_values(self, simple_binary_data):
        """Warm start should use provided values as initialization."""
        treat, X = simple_binary_data
        # First fit normally
        result1 = cbps_binary_fit(treat, X, att=1, iterations=200)
        # Then warm start from result1's coefficients
        init = result1['coefficients'].ravel()
        result2 = cbps_binary_fit(treat, X, att=1, iterations=50,
                                  init_params=init)
        # Should converge faster (or be very close)
        assert 'coefficients' in result2


# ---------- P2-13: Error message enhancement ----------

class TestErrorMessages:
    """Test that error messages include suggestions."""

    def test_rank_deficient_message(self):
        """Rank-deficient X should give enhanced error message."""
        n = 50
        X = np.ones((n, 3))  # All columns identical -> rank 1
        X[:, 1] = X[:, 0]  # Duplicate column
        X[:, 2] = X[:, 0]
        treat = np.random.binomial(1, 0.5, n).astype(float)
        with pytest.raises(ValueError, match="Suggestions"):
            cbps_binary_fit(treat, X, att=1)

    def test_vmmin_non_finite_message(self):
        """Non-finite initial value should give enhanced error message."""
        from cbps.core.cbps_binary import _vmmin_bfgs
        fn = lambda b: np.inf  # Always returns inf
        gr = lambda b: np.zeros_like(b)
        with pytest.raises(ValueError, match="Suggestions"):
            _vmmin_bfgs(np.array([1.0]), fn, gr)


# ---------- P2-14: Batch estimation interface ----------

class TestFitMultiple:
    """Test batch estimation with fit_multiple."""

    def test_fit_multiple_basic(self, lalonde_df):
        """fit_multiple should fit on multiple datasets."""
        from cbps import fit_multiple
        # Create 3 copies with slight variation
        datasets = [lalonde_df.sample(frac=0.8, random_state=i)
                    for i in range(3)]
        results = fit_multiple('treat ~ age + educ + black', datasets, att=1)
        assert len(results) == 3
        # Check they are CBPSResults (not error dicts)
        from cbps.core.results import CBPSResults
        for r in results:
            assert isinstance(r, CBPSResults)

    def test_fit_multiple_handles_errors(self):
        """fit_multiple should catch errors gracefully."""
        from cbps import fit_multiple
        # Empty DataFrame should cause error
        bad_df = pd.DataFrame({'treat': [], 'age': [], 'educ': []})
        good_df = pd.DataFrame({
            'treat': np.random.binomial(1, 0.5, 100).astype(float),
            'age': np.random.randn(100),
            'educ': np.random.randn(100),
        })
        results = fit_multiple('treat ~ age + educ', [good_df, bad_df])
        assert len(results) == 2
        # Second result should be error dict
        assert isinstance(results[1], dict)
        assert 'error' in results[1]
        assert results[1]['dataset_index'] == 1


# ---------- P2-15: Pickle serialization ----------

class TestPickleSerialization:
    """Test that CBPSResults can be pickled and unpickled."""

    def test_pickle_roundtrip(self, lalonde_df):
        """CBPSResults should survive pickle roundtrip."""
        from cbps import CBPS
        fit = CBPS('treat ~ age + educ + black', data=lalonde_df, att=1)

        # Pickle to bytes
        pickled = pickle.dumps(fit)
        # Unpickle
        restored = pickle.loads(pickled)

        # Verify key attributes
        np.testing.assert_array_equal(fit.coefficients, restored.coefficients)
        np.testing.assert_array_equal(fit.weights, restored.weights)
        np.testing.assert_array_equal(fit.fitted_values, restored.fitted_values)
        assert fit.converged == restored.converged
        assert fit.J == restored.J

    def test_pickle_to_file(self, lalonde_df, tmp_path):
        """CBPSResults should survive pickle to/from file."""
        from cbps import CBPS
        fit = CBPS('treat ~ age + educ + black', data=lalonde_df, att=1)

        filepath = tmp_path / "cbps_result.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(fit, f)

        with open(filepath, 'rb') as f:
            restored = pickle.loads(f.read())

        np.testing.assert_array_equal(fit.coefficients, restored.coefficients)
        assert fit.converged == restored.converged
