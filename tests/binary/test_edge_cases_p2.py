"""Edge case and boundary condition tests for CBPS binary estimator.

Tests P2-16 through P2-22: separation, extreme weights, large scale,
determinism, single covariate, collinearity, and missing values.
"""
import numpy as np
import pytest
import warnings

from cbps.core.cbps_binary import cbps_binary_fit


# ---------------------------------------------------------------------------
# P2-16: Complete Separation Detection
# ---------------------------------------------------------------------------

def test_complete_separation_detected():
    """When treatment perfectly predicted by covariates, separation should be detected."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    treat = (x > 0).astype(float)  # 完美预测
    X = np.column_stack([np.ones(n), x])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cbps_binary_fit(treat, X, att=0)
        # 应触发分离警告
        sep_warnings = [x for x in w if 'Separation' in str(x.message) or 'separation' in str(x.message)]
        assert len(sep_warnings) > 0, "Complete separation should trigger warning"


# ---------------------------------------------------------------------------
# P2-17: Extreme Weights Handling
# ---------------------------------------------------------------------------

def test_extreme_weights_handling():
    """When propensity scores are very close to 0/1, weights should remain finite."""
    np.random.seed(123)
    n = 500
    # 构造导致极端PS的数据（强预测力）
    x = np.random.randn(n) * 3  # 大方差→极端PS
    treat = (np.random.rand(n) < 1 / (1 + np.exp(-x * 2))).astype(float)
    X = np.column_stack([np.ones(n), x])

    result = cbps_binary_fit(treat, X, att=0)

    # 权重必须有限
    assert np.all(np.isfinite(result['weights'])), "Weights must be finite"
    # 权重不应全为0
    assert np.any(result['weights'] != 0), "Weights should not all be zero"


# ---------------------------------------------------------------------------
# P2-18: Large Scale Stress Test (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_large_scale_n100k():
    """Verify algorithm handles n=100k without memory error."""
    np.random.seed(99)
    n = 100_000
    k = 5
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    beta_true = np.array([0.0, 0.5, -0.3, 0.2, 0.1])
    ps = 1 / (1 + np.exp(-X @ beta_true))
    treat = (np.random.rand(n) < ps).astype(float)

    result = cbps_binary_fit(treat, X, att=0, iterations=100)

    assert result['converged'] or result['J'] < 100  # 允许不完全收敛
    assert result['coefficients'].shape == (k, 1)
    assert len(result['weights']) == n


# ---------------------------------------------------------------------------
# P2-19: Deterministic with Fixed Seed
# ---------------------------------------------------------------------------

def test_deterministic_with_fixed_seed():
    """Same data + same algorithm should produce identical results."""
    np.random.seed(777)
    n = 200
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    treat = (np.random.rand(n) > 0.5).astype(float)

    result1 = cbps_binary_fit(treat, X, att=0)
    result2 = cbps_binary_fit(treat, X, att=0)

    np.testing.assert_array_almost_equal(
        result1['coefficients'], result2['coefficients'], decimal=10
    )
    np.testing.assert_array_almost_equal(
        result1['weights'], result2['weights'], decimal=10
    )


# ---------------------------------------------------------------------------
# P2-20: Single Covariate Boundary
# ---------------------------------------------------------------------------

def test_single_covariate():
    """Algorithm should work with only intercept + 1 covariate (k=2)."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    treat = (np.random.rand(n) < 1 / (1 + np.exp(-x))).astype(float)
    X = np.column_stack([np.ones(n), x])  # k=2

    result = cbps_binary_fit(treat, X, att=0)
    assert result['coefficients'].shape == (2, 1)
    assert np.all(np.isfinite(result['fitted_values']))


# ---------------------------------------------------------------------------
# P2-21: High Collinearity
# ---------------------------------------------------------------------------

def test_high_collinearity():
    """Near-collinear covariates should trigger warnings but not crash."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 1e-6  # 几乎完全共线
    X = np.column_stack([np.ones(n), x1, x2])
    treat = (np.random.rand(n) > 0.5).astype(float)

    # 应该要么成功要么抛出有意义的错误（不应崩溃）
    try:
        result = cbps_binary_fit(treat, X, att=0)
        # 如果成功，结果应有限
        assert np.all(np.isfinite(result['fitted_values']))
    except (ValueError, np.linalg.LinAlgError) as e:
        # 秩亏检测应给出有意义的错误
        assert 'rank' in str(e).lower() or 'singular' in str(e).lower()


# ---------------------------------------------------------------------------
# P2-22: Missing Values Input
# ---------------------------------------------------------------------------

def test_nan_input_raises_error():
    """NaN in input should raise ValueError with helpful message."""
    np.random.seed(42)
    n = 100
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    treat = (np.random.rand(n) > 0.5).astype(float)

    # NaN in X - must raise ValueError specifically (not LinAlgError or other)
    X_nan = X.copy()
    X_nan[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf|missing|finite"):
        cbps_binary_fit(treat, X_nan, att=0)

    # NaN in treat
    treat_nan = treat.copy()
    treat_nan[10] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf|missing|finite|Treatment"):
        cbps_binary_fit(treat_nan, X, att=0)


def test_inf_input_raises_error():
    """Inf in input should raise ValueError with helpful message."""
    np.random.seed(42)
    n = 100
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    treat = (np.random.rand(n) > 0.5).astype(float)

    X_inf = X.copy()
    X_inf[0, 1] = np.inf
    with pytest.raises(ValueError, match="NaN|Inf|missing|finite"):
        cbps_binary_fit(treat, X_inf, att=0)
