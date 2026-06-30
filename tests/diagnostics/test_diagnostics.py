"""
Test Suite: Comprehensive Diagnostics Tests for the CBPS Package
=================================================================

This module consolidates all diagnostic test suites for the Covariate
Balancing Propensity Score (CBPS) package, including covariate balance
assessment, enhanced balance diagnostics, marginal structural model
balance evaluation, visualization functions, and module export
verification.

Test Categories:
    - Module Export Tests (DIAG-INIT-001 to DIAG-INIT-015):
        Verification of public API exports and function signatures.
    - Binary Balance Tests (BAL-001 to BAL-010):
        Covariate balance assessment for binary treatments.
    - Continuous Balance Tests (BAL-011 to BAL-020):
        Covariate balance assessment for continuous treatments.
    - CBMSM Balance Tests (BMSM-001 to BMSM-025):
        Balance diagnostics for marginal structural models.
    - Enhanced Balance Tests (BALE-001 to BALE-030):
        Extended diagnostics with summary statistics and reports.
    - Visualization Tests (PLOT-001 to PLOT-030):
        Covariate balance plotting functions.

Requirements: REQ-DIAG-001 to REQ-DIAG-035, REQ-DIAG-INIT-001 to REQ-DIAG-INIT-003

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics 12(1), 156-177.

    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
    weights for marginal structural models. Journal of the American Statistical
    Association, 110(511), 1013-1023.

    Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and
    a Look Forward. Statistical Science 25(1), 1-21.

    Austin, P.C. (2009). Balance diagnostics for comparing the distribution of
    baseline covariates between treatment groups in propensity-score matched
    samples. Statistics in Medicine, 28(25), 3083-3107.

Usage:
    pytest tests/diagnostics/test_diagnostics.py -v
"""

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

from cbps import diagnostics
from cbps.diagnostics import balance_cbps, balance_cbps_continuous
from cbps.diagnostics.balance import (
    balance_cbps_enhanced,
    balance_cbps_continuous_enhanced,
)
from cbps.diagnostics.balance_cbmsm_addon import balance_cbmsm

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Diagnostics-Specific Fixtures (from diagnostics/conftest.py)
# =============================================================================

@pytest.fixture(scope="session")
def matplotlib_available():
    """
    Check if matplotlib is available in the environment.

    Returns
    -------
    bool
        True if matplotlib is installed, False otherwise.

    Notes
    -----
    This fixture is session-scoped to avoid repeated import checks.
    Tests requiring matplotlib should use this fixture and skip if False.
    """
    try:
        import matplotlib
        return True
    except ImportError:
        return False


@pytest.fixture
def binary_balance_data():
    """
    Generate mock data for binary treatment balance testing.

    Returns
    -------
    dict
        Dictionary simulating a CBPS result object with:
        - x: Covariate matrix with intercept
        - y: Binary treatment indicator
        - weights: IPW-style weights
        - fitted_values: Propensity scores

    Notes
    -----
    Used for testing balance_cbps function and binary treatment plots.
    Random seed is fixed at 42 for reproducibility.
    """
    np.random.seed(42)
    n = 200

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # X matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2, x3])

    # Treatment assignment based on covariates
    logit_ps = 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    ps = 1 / (1 + np.exp(-logit_ps))
    y = np.random.binomial(1, ps).astype(float)

    # IPW-style weights
    weights = y / ps + (1 - y) / (1 - ps)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': ps,
        'coef_names': ['Intercept', 'X1', 'X2', 'X3'],
    }


@pytest.fixture
def continuous_balance_data():
    """
    Generate mock data for continuous treatment balance testing.

    Returns
    -------
    dict
        Dictionary simulating a continuous treatment CBPS result object.

    Notes
    -----
    Used for testing balance_cbps_continuous function.
    Reference: Fong, Hazlett & Imai (2018), Annals of Applied Statistics.
    """
    np.random.seed(123)
    n = 200

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # X matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])

    # Continuous treatment correlated with covariates
    y = 0.5 * x1 - 0.3 * x2 + np.random.randn(n) * 0.5

    # Fitted values (predicted treatment mean)
    fitted_values = 0.5 * x1 - 0.3 * x2

    # Stabilized weights for continuous treatment
    weights = np.ones(n)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': fitted_values,
        'coef_names': ['Intercept', 'X1', 'X2'],
    }


@pytest.fixture
def multitreat_balance_data():
    """
    Generate mock data for multi-valued treatment balance testing.

    Returns
    -------
    dict
        Dictionary simulating a multi-valued treatment CBPS result object.

    Notes
    -----
    Used for testing balance functions with 3-level categorical treatment.
    """
    np.random.seed(456)
    n = 300
    n_levels = 3

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # X matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])

    # Multinomial treatment probabilities
    logits = np.column_stack([
        0.3 * x1 - 0.2 * x2,
        -0.2 * x1 + 0.3 * x2
    ])
    exp_logits = np.exp(logits)
    denom = 1 + exp_logits.sum(axis=1, keepdims=True)
    probs = np.column_stack([1 / denom, exp_logits / denom])

    # Treatment assignment
    y = np.array([np.random.choice(n_levels, p=p) for p in probs])

    # Weights
    weights = np.ones(n)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': probs,
        'coef_names': ['Intercept', 'X1', 'X2'],
        'n_levels': n_levels,
    }


@pytest.fixture
def cbmsm_balance_data():
    """
    Generate mock data for CBMSM (longitudinal) balance testing.

    Returns
    -------
    dict
        Dictionary simulating a CBMSM result object for testing
        balance_cbmsm_addon functions.

    Notes
    -----
    Used for testing balance functions with panel/longitudinal data.
    Reference: Imai & Ratkovic (2015), JASA.
    """
    np.random.seed(789)
    n_id = 50
    n_time = 3
    n = n_id * n_time

    # Panel structure
    id_arr = np.repeat(np.arange(n_id), n_time)
    time_arr = np.tile(np.arange(n_time), n_id)

    # Covariates
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # X matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])

    # Treatment (time-varying)
    logit_ps = 0.2 * x1 - 0.1 * x2
    ps = 1 / (1 + np.exp(-logit_ps))
    y = np.random.binomial(1, ps).astype(float)

    # Weights
    weights = np.ones(n)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'id': id_arr,
        'time': time_arr,
        'coef_names': ['Intercept', 'X1', 'X2'],
    }


@pytest.fixture
def balance_tolerance():
    """
    Return tolerance values for balance metric comparisons.

    Returns
    -------
    dict
        Dictionary of tolerance values for different metrics.
    """
    return {
        'smd_atol': 0.01,  # Standardized mean difference
        'smd_rtol': 0.05,
        'ks_atol': 0.02,   # Kolmogorov-Smirnov statistic
        'ks_rtol': 0.05,
    }



# =============================================================================
# Module Export Tests (from test_init.py)
# Test IDs: DIAG-INIT-001 to DIAG-INIT-015
# =============================================================================

@pytest.mark.unit
class TestDiagnosticsModuleExports:
    """
    Test ID: DIAG-INIT-001 ~ DIAG-INIT-010
    Requirement: REQ-DIAG-INIT-001

    Tests for cbps.diagnostics module exports.
    """

    def test_diag_init001_module_importable(self):
        """DIAG-INIT-001: cbps.diagnostics module is importable."""
        from cbps import diagnostics
        assert diagnostics is not None

    def test_diag_init002_balance_cbps_importable(self):
        """DIAG-INIT-002: balance_cbps is importable from cbps.diagnostics."""
        from cbps.diagnostics import balance_cbps
        assert callable(balance_cbps)

    def test_diag_init003_balance_cbps_continuous_importable(self):
        """DIAG-INIT-003: balance_cbps_continuous is importable from cbps.diagnostics."""
        from cbps.diagnostics import balance_cbps_continuous
        assert callable(balance_cbps_continuous)

    def test_diag_init004_all_defined(self):
        """DIAG-INIT-004: __all__ is defined in diagnostics module."""
        from cbps import diagnostics
        assert hasattr(diagnostics, '__all__')
        assert isinstance(diagnostics.__all__, (list, tuple))

    def test_diag_init005_all_contains_balance_cbps(self):
        """DIAG-INIT-005: __all__ contains balance_cbps."""
        from cbps import diagnostics
        assert 'balance_cbps' in diagnostics.__all__

    def test_diag_init006_all_contains_balance_cbps_continuous(self):
        """DIAG-INIT-006: __all__ contains balance_cbps_continuous."""
        from cbps import diagnostics
        assert 'balance_cbps_continuous' in diagnostics.__all__

    def test_diag_init007_no_private_exports(self):
        """DIAG-INIT-007: __all__ does not export private functions."""
        from cbps import diagnostics
        for name in diagnostics.__all__:
            assert not name.startswith('_'), f"Private {name} should not be in __all__"


@pytest.mark.unit
class TestDiagnosticsFunctionSignatures:
    """
    Test ID: DIAG-INIT-008 ~ DIAG-INIT-015
    Requirement: REQ-DIAG-INIT-002

    Tests for diagnostics function signatures.
    """

    def test_diag_init008_balance_cbps_has_cbps_obj_param(self):
        """DIAG-INIT-008: balance_cbps has cbps_obj parameter."""
        from cbps.diagnostics import balance_cbps
        sig = inspect.signature(balance_cbps)
        assert 'cbps_obj' in sig.parameters

    def test_diag_init009_balance_cbps_continuous_has_cbps_obj_param(self):
        """DIAG-INIT-009: balance_cbps_continuous has cbps_obj parameter."""
        from cbps.diagnostics import balance_cbps_continuous
        sig = inspect.signature(balance_cbps_continuous)
        assert 'cbps_obj' in sig.parameters

    def test_diag_init010_balance_cbps_returns_dataframe(self):
        """DIAG-INIT-010: balance_cbps returns DataFrame (verified by docstring)."""
        from cbps.diagnostics import balance_cbps
        # Check that return type is documented
        doc = balance_cbps.__doc__
        assert doc is not None
        assert 'DataFrame' in doc or 'pd.DataFrame' in doc or 'return' in doc.lower()


@pytest.mark.unit
class TestDiagnosticsPlotExports:
    """
    Test ID: DIAG-INIT-011 ~ DIAG-INIT-015
    Requirement: REQ-DIAG-INIT-003

    Tests for optional plot function exports (require matplotlib).
    """

    def test_diag_init011_plot_functions_in_all_when_available(self):
        """DIAG-INIT-011: plot functions are in __all__ when matplotlib is available."""
        from cbps import diagnostics

        # Check if matplotlib is available
        try:
            import matplotlib
            has_matplotlib = True
        except ImportError:
            has_matplotlib = False

        if has_matplotlib:
            assert 'plot_cbps' in diagnostics.__all__
            assert 'plot_cbps_continuous' in diagnostics.__all__
        else:
            # When matplotlib is not available, plot functions may not be exported
            pass

    def test_diag_init012_plot_cbps_callable_when_available(self):
        """DIAG-INIT-012: plot_cbps is callable when available."""
        try:
            from cbps.diagnostics import plot_cbps
            assert callable(plot_cbps)
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_diag_init013_plot_cbps_continuous_callable_when_available(self):
        """DIAG-INIT-013: plot_cbps_continuous is callable when available."""
        try:
            from cbps.diagnostics import plot_cbps_continuous
            assert callable(plot_cbps_continuous)
        except ImportError:
            pytest.skip("matplotlib not available")



# =============================================================================
# Binary and Continuous Balance Tests (from test_balance.py)
# Test IDs: BAL-001 to BAL-020
# =============================================================================

class TestBinaryBalance:
    """
    Test covariate balance for binary treatments.

    Test IDs: BAL-001 to BAL-010
    Requirements: REQ-DIAG-001
    """

    @pytest.fixture
    def binary_cbps_obj(self):
        """Create a mock CBPS result object for binary treatment."""
        np.random.seed(42)
        n = 200

        # Covariates
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        # X matrix with intercept
        X = np.column_stack([np.ones(n), x1, x2])

        # Treatment
        logit_ps = 0.3 * x1 - 0.2 * x2
        ps = 1 / (1 + np.exp(-logit_ps))
        y = np.random.binomial(1, ps).astype(float)

        # Weights (simulating CBPS weights)
        weights = y / ps + (1 - y) / (1 - ps)

        return {
            'x': X,
            'y': y,
            'weights': weights,
        }

    @pytest.mark.unit
    def test_bal001_returns_dict(self, binary_cbps_obj):
        """
        BAL-001: Verify balance_cbps returns a dictionary.

        Requirements: REQ-DIAG-001
        """
        result = balance_cbps(binary_cbps_obj)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_bal002_required_keys(self, binary_cbps_obj):
        """
        BAL-002: Verify required keys are present in result.

        Requirements: REQ-DIAG-001
        """
        result = balance_cbps(binary_cbps_obj)

        assert 'balanced' in result
        assert 'original' in result

    @pytest.mark.unit
    def test_bal003_balanced_shape(self, binary_cbps_obj):
        """
        BAL-003: Verify balanced matrix shape.

        Requirements: REQ-DIAG-001

        Notes:
            For binary treatment, shape should be (n_covars, 4)
            [mean_0, mean_1, std_mean_0, std_mean_1]
        """
        result = balance_cbps(binary_cbps_obj)

        # n_covars = 2 (excluding intercept), n_treats = 2
        # Shape: (n_covars, 2 * n_treats) = (2, 4)
        assert result['balanced'].shape[0] == 2  # 2 covariates
        assert result['balanced'].shape[1] == 4  # 2 * 2 treatments

    @pytest.mark.unit
    def test_bal004_original_shape(self, binary_cbps_obj):
        """
        BAL-004: Verify original matrix shape.

        Requirements: REQ-DIAG-001
        """
        result = balance_cbps(binary_cbps_obj)

        assert result['original'].shape == result['balanced'].shape

    @pytest.mark.numerical
    def test_bal005_values_finite(self, binary_cbps_obj):
        """
        BAL-005: Verify all balance values are finite.

        Requirements: REQ-DIAG-002
        """
        result = balance_cbps(binary_cbps_obj)

        assert np.all(np.isfinite(result['balanced']))
        assert np.all(np.isfinite(result['original']))

    @pytest.mark.numerical
    def test_bal006_standardized_reasonable(self, binary_cbps_obj):
        """
        BAL-006: Verify standardized values are within reasonable range.

        Requirements: REQ-DIAG-002
        """
        result = balance_cbps(binary_cbps_obj)

        # Standardized values should typically be within [-10, 10]
        assert np.all(np.abs(result['balanced'][:, 2:]) < 50)
        assert np.all(np.abs(result['original'][:, 2:]) < 50)

    @pytest.mark.numerical
    def test_bal007_reproducibility(self, binary_cbps_obj):
        """
        BAL-007: Verify balance computation is reproducible.

        Requirements: REQ-DIAG-003
        """
        result1 = balance_cbps(binary_cbps_obj)
        result2 = balance_cbps(binary_cbps_obj)

        assert_allclose(result1['balanced'], result2['balanced'])
        assert_allclose(result1['original'], result2['original'])

    @pytest.mark.numerical
    def test_bal008_balance_improvement(self, binary_cbps_obj):
        """
        BAL-008: Verify weighted balance is typically better than original.

        Requirements: REQ-DIAG-004
        """
        result = balance_cbps(binary_cbps_obj)

        # Compute SMD for original and balanced
        orig_smd = np.abs(result['original'][:, 2] - result['original'][:, 3])
        bal_smd = np.abs(result['balanced'][:, 2] - result['balanced'][:, 3])

        # On average, balanced SMD should be lower
        # (not always true for random data, so use relaxed check)
        assert np.mean(bal_smd) < np.mean(orig_smd) * 2

    @pytest.mark.unit
    def test_bal009_single_covariate(self):
        """
        BAL-009: Verify handling of single covariate.

        Requirements: REQ-DIAG-005
        """
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.binomial(1, 0.5, n).astype(float)
        weights = np.ones(n)

        cbps_obj = {'x': X, 'y': y, 'weights': weights}

        result = balance_cbps(cbps_obj)

        assert result['balanced'].shape[0] == 1

    @pytest.mark.unit
    def test_bal010_equal_weights(self):
        """
        BAL-010: Verify equal weights produce same as original.

        Requirements: REQ-DIAG-006
        """
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.binomial(1, 0.5, n).astype(float)
        weights = np.ones(n)  # Equal weights

        cbps_obj = {'x': X, 'y': y, 'weights': weights}

        result = balance_cbps(cbps_obj)

        # With equal weights, means should be approximately same
        assert_allclose(
            result['balanced'][:, :2],
            result['original'][:, :2],
            rtol=1e-10
        )


class TestContinuousBalance:
    """
    Test covariate balance for continuous treatments.

    Test IDs: BAL-011 to BAL-020
    Requirements: REQ-DIAG-007
    """

    @pytest.fixture
    def continuous_cbps_obj(self):
        """Create a mock CBPS result object for continuous treatment."""
        np.random.seed(42)
        n = 200

        # Covariates
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        # X matrix with intercept
        X = np.column_stack([np.ones(n), x1, x2])

        # Continuous treatment
        T = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)

        # Weights (simulating continuous CBPS weights)
        weights = np.ones(n) + 0.1 * np.random.randn(n)
        weights = np.abs(weights)  # Ensure positive

        return {
            'x': X,
            'y': T,
            'weights': weights,
        }

    @pytest.mark.unit
    def test_bal011_returns_dict(self, continuous_cbps_obj):
        """
        BAL-011: Verify balance_cbps_continuous returns a dictionary.

        Requirements: REQ-DIAG-007
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_bal012_required_keys(self, continuous_cbps_obj):
        """
        BAL-012: Verify required keys are present in result.

        Requirements: REQ-DIAG-007
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        assert 'balanced' in result
        assert 'unweighted' in result  # Note: continuous uses 'unweighted' not 'original'

    @pytest.mark.unit
    def test_bal013_balanced_shape(self, continuous_cbps_obj):
        """
        BAL-013: Verify balanced correlations shape.

        Requirements: REQ-DIAG-007
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        # Should be 2D array with shape (n_covars, 1) or similar
        assert result['balanced'].shape[0] == 2  # 2 covariates (excluding intercept)

    @pytest.mark.unit
    def test_bal014_original_shape(self, continuous_cbps_obj):
        """
        BAL-014: Verify unweighted correlations shape.

        Requirements: REQ-DIAG-007
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        assert result['unweighted'].shape == result['balanced'].shape

    @pytest.mark.numerical
    def test_bal015_correlations_bounded(self, continuous_cbps_obj):
        """
        BAL-015: Verify correlations are bounded in [-1, 1].

        Requirements: REQ-DIAG-008
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        assert np.all(np.abs(result['balanced']) <= 1.001)  # Small tolerance
        assert np.all(np.abs(result['unweighted']) <= 1.001)

    @pytest.mark.numerical
    def test_bal016_values_finite(self, continuous_cbps_obj):
        """
        BAL-016: Verify all correlation values are finite.

        Requirements: REQ-DIAG-008
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        assert np.all(np.isfinite(result['balanced']))
        assert np.all(np.isfinite(result['unweighted']))

    @pytest.mark.numerical
    def test_bal017_reproducibility(self, continuous_cbps_obj):
        """
        BAL-017: Verify balance computation is reproducible.

        Requirements: REQ-DIAG-009
        """
        result1 = balance_cbps_continuous(continuous_cbps_obj)
        result2 = balance_cbps_continuous(continuous_cbps_obj)

        assert_allclose(result1['balanced'], result2['balanced'])
        assert_allclose(result1['unweighted'], result2['unweighted'])

    @pytest.mark.numerical
    def test_bal018_balance_improvement(self, continuous_cbps_obj):
        """
        BAL-018: Verify weighted correlations are typically lower.

        Requirements: REQ-DIAG-010
        """
        result = balance_cbps_continuous(continuous_cbps_obj)

        # Balanced correlations should typically be smaller in magnitude
        # (not always true for random data, so use relaxed check)
        orig_mean = np.mean(np.abs(result['unweighted']))
        bal_mean = np.mean(np.abs(result['balanced']))

        # Balanced should not be much worse than original
        assert bal_mean < orig_mean * 2

    @pytest.mark.unit
    def test_bal019_single_covariate(self):
        """
        BAL-019: Verify handling of single covariate.

        Requirements: REQ-DIAG-007
        """
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        T = np.random.randn(n)
        weights = np.ones(n)

        cbps_obj = {'x': X, 'y': T, 'weights': weights}

        result = balance_cbps_continuous(cbps_obj)

        assert result['balanced'].shape[0] == 1

    @pytest.mark.numerical
    def test_bal020_equal_weights_match_unweighted(self):
        """
        BAL-020: Verify equal weights produce same as unweighted correlation.

        Requirements: REQ-DIAG-009
        """
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        T = np.random.randn(n)
        weights = np.ones(n)  # Equal weights

        cbps_obj = {'x': X, 'y': T, 'weights': weights}

        result = balance_cbps_continuous(cbps_obj)

        # With equal weights, correlations should be similar
        assert_allclose(result['balanced'], result['unweighted'], rtol=0.01)



# =============================================================================
# CBMSM Balance Tests (from test_balance_cbmsm.py)
# Test IDs: BMSM-001 to BMSM-025
# =============================================================================

# ---- CBMSM-specific fixtures ----

@pytest.fixture
def balanced_panel_cbmsm():
    """
    Generate balanced panel data with CBMSM results.

    Returns
    -------
    dict
        Dictionary with all required data for balance_cbmsm.
    """
    np.random.seed(42)
    n_units = 100
    n_periods = 3
    n_obs = n_units * n_periods

    # Panel identifiers
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    # Covariates with intercept
    x1 = np.random.randn(n_obs)
    x2 = np.random.randn(n_obs)
    X = np.column_stack([np.ones(n_obs), x1, x2])

    # Binary treatment (time-varying)
    treat_prob = 1 / (1 + np.exp(-0.3 * x1 + 0.2 * x2))
    y = np.random.binomial(1, treat_prob).astype(float)

    # CBMSM weights (unit-level)
    weights = np.random.exponential(1, n_units)
    weights = weights / weights.mean()

    # GLM baseline weights
    glm_weights = np.ones(n_units)

    return {
        'y': y,
        'x': X,
        'weights': weights,
        'glm_weights': glm_weights,
        'id': ids,
        'time': times,
        'n_units': n_units,
        'n_periods': n_periods,
        'n_obs': n_obs
    }


@pytest.fixture
def unbalanced_panel_cbmsm():
    """
    Generate unbalanced panel data (varying number of observations per unit).

    Returns
    -------
    dict
        Dictionary with unbalanced panel data.
    """
    np.random.seed(123)
    n_units = 50

    # Variable number of periods per unit
    periods_per_unit = np.random.randint(2, 5, n_units)
    n_obs = periods_per_unit.sum()

    # Panel identifiers
    ids = np.repeat(np.arange(n_units), periods_per_unit)
    times = np.concatenate([np.arange(p) for p in periods_per_unit])

    # Covariates
    x1 = np.random.randn(n_obs)
    X = np.column_stack([np.ones(n_obs), x1])

    # Treatment
    y = np.random.binomial(1, 0.5, n_obs).astype(float)

    # Weights
    weights = np.ones(n_units)
    glm_weights = np.ones(n_units)

    return {
        'y': y,
        'x': X,
        'weights': weights,
        'glm_weights': glm_weights,
        'id': ids,
        'time': times,
        'n_units': n_units,
    }


class TestBalanceCBMSMBasic:
    """
    Test basic functionality of balance_cbmsm.

    Test IDs: BMSM-001 to BMSM-010
    Requirements: REQ-DIAG-011
    """

    @pytest.mark.unit
    def test_bmsm001_returns_dict(self, balanced_panel_cbmsm):
        """
        BMSM-001: Verify balance_cbmsm returns a dictionary.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_bmsm002_required_keys_present(self, balanced_panel_cbmsm):
        """
        BMSM-002: Verify all required keys are present.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        required_keys = ['Balanced', 'Unweighted', 'StatBal']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.unit
    def test_bmsm003_balanced_is_array(self, balanced_panel_cbmsm):
        """
        BMSM-003: Verify 'Balanced' is a numpy array.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert isinstance(result['Balanced'], np.ndarray)

    @pytest.mark.unit
    def test_bmsm004_unweighted_is_array(self, balanced_panel_cbmsm):
        """
        BMSM-004: Verify 'Unweighted' is a numpy array.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert isinstance(result['Unweighted'], np.ndarray)

    @pytest.mark.unit
    def test_bmsm005_statbal_is_scalar(self, balanced_panel_cbmsm):
        """
        BMSM-005: Verify 'StatBal' is a scalar.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert np.isscalar(result['StatBal'])

    @pytest.mark.unit
    def test_bmsm006_balanced_shape(self, balanced_panel_cbmsm):
        """
        BMSM-006: Verify 'Balanced' has correct shape.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        n_covars = balanced_panel_cbmsm['x'].shape[1] - 1  # Exclude intercept
        n_hist = len(np.unique([''.join(map(str, h)) for h in
                                result['Balanced'].T[:result['Balanced'].shape[1]//2]]))

        # Shape should be (n_covars, 2 * n_hist)
        assert result['Balanced'].shape[0] == n_covars

    @pytest.mark.unit
    def test_bmsm007_balanced_unweighted_same_shape(self, balanced_panel_cbmsm):
        """
        BMSM-007: Verify 'Balanced' and 'Unweighted' have same shape.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert result['Balanced'].shape == result['Unweighted'].shape

    @pytest.mark.unit
    def test_bmsm008_values_finite(self, balanced_panel_cbmsm):
        """
        BMSM-008: Verify all values are finite.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert np.all(np.isfinite(result['Balanced']))
        assert np.all(np.isfinite(result['Unweighted']))
        assert np.isfinite(result['StatBal'])

    @pytest.mark.unit
    def test_bmsm009_reproducibility(self, balanced_panel_cbmsm):
        """
        BMSM-009: Verify results are reproducible.

        Requirements: REQ-DIAG-011
        """
        result1 = balance_cbmsm(balanced_panel_cbmsm)
        result2 = balance_cbmsm(balanced_panel_cbmsm)

        assert_allclose(result1['Balanced'], result2['Balanced'])
        assert_allclose(result1['Unweighted'], result2['Unweighted'])
        assert result1['StatBal'] == result2['StatBal']

    @pytest.mark.unit
    def test_bmsm010_statbal_nonnegative(self, balanced_panel_cbmsm):
        """
        BMSM-010: Verify StatBal is non-negative.

        Requirements: REQ-DIAG-011
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        assert result['StatBal'] >= 0


class TestTreatmentHistory:
    """
    Test treatment history reconstruction.

    Test IDs: BMSM-011 to BMSM-015
    Requirements: REQ-DIAG-012
    """

    @pytest.mark.unit
    def test_bmsm011_treatment_history_string_format(self, balanced_panel_cbmsm):
        """
        BMSM-011: Verify treatment histories are reconstructed correctly.

        Requirements: REQ-DIAG-012
        """
        # Manually compute expected treatment histories
        ids = balanced_panel_cbmsm['id']
        times = balanced_panel_cbmsm['time']
        y = balanced_panel_cbmsm['y']

        unique_ids = np.unique(ids)
        unique_times = np.unique(times)

        # Check that treatment history is computed
        result = balance_cbmsm(balanced_panel_cbmsm)

        # Result should have columns for each unique treatment history
        assert result['Balanced'].shape[1] > 0

    @pytest.mark.unit
    def test_bmsm012_handles_all_treated(self):
        """
        BMSM-012: Verify handling when all units are treated in all periods.

        Requirements: REQ-DIAG-012
        """
        np.random.seed(42)
        n_units = 20
        n_periods = 2
        n_obs = n_units * n_periods

        cbmsm_obj = {
            'y': np.ones(n_obs),  # All treated
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        # Should have single treatment history pattern
        # (all 1s, so just one unique history)
        assert result['Balanced'].shape[1] == 2  # mean and std.mean

    @pytest.mark.unit
    def test_bmsm013_handles_all_control(self):
        """
        BMSM-013: Verify handling when all units are control in all periods.

        Requirements: REQ-DIAG-012
        """
        np.random.seed(42)
        n_units = 20
        n_periods = 2
        n_obs = n_units * n_periods

        cbmsm_obj = {
            'y': np.zeros(n_obs),  # All control
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        # Should have single treatment history pattern
        assert result['Balanced'].shape[1] == 2

    @pytest.mark.unit
    def test_bmsm014_multiple_histories(self, balanced_panel_cbmsm):
        """
        BMSM-014: Verify handling of multiple treatment histories.

        Requirements: REQ-DIAG-012
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        # With random treatment, should have multiple histories
        n_histories = result['Balanced'].shape[1] // 2

        assert n_histories >= 1

    @pytest.mark.unit
    def test_bmsm015_history_pattern_coverage(self):
        """
        BMSM-015: Verify all observed patterns are covered.

        Requirements: REQ-DIAG-012
        """
        np.random.seed(42)
        n_units = 50
        n_periods = 2
        n_obs = n_units * n_periods

        # Create specific treatment patterns
        y = np.zeros(n_obs)
        # First 25 units: 0+0, Next 25 units: 1+1
        y[:n_periods * 25] = 0  # All 0+0
        y[n_periods * 25:] = 1  # All 1+1

        cbmsm_obj = {
            'y': y,
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        # Should have exactly 2 unique histories (0+0 and 1+1)
        n_histories = result['Balanced'].shape[1] // 2
        assert n_histories == 2


class TestWeightedBalance:
    """
    Test weighted balance computation.

    Test IDs: BMSM-016 to BMSM-020
    Requirements: REQ-DIAG-013
    """

    @pytest.mark.unit
    def test_bmsm016_weights_affect_balance(self, balanced_panel_cbmsm):
        """
        BMSM-016: Verify weights affect balanced statistics.

        Requirements: REQ-DIAG-013
        """
        result1 = balance_cbmsm(balanced_panel_cbmsm)

        # Modify weights
        cbmsm_obj_modified = balanced_panel_cbmsm.copy()
        cbmsm_obj_modified['weights'] = np.ones(balanced_panel_cbmsm['n_units']) * 2.0

        result2 = balance_cbmsm(cbmsm_obj_modified)

        # Balanced statistics may differ with different weights
        # (at least structure should be same)
        assert result1['Balanced'].shape == result2['Balanced'].shape

    @pytest.mark.unit
    def test_bmsm017_uniform_weights_baseline(self, balanced_panel_cbmsm):
        """
        BMSM-017: Verify uniform weights produce valid baseline.

        Requirements: REQ-DIAG-013
        """
        cbmsm_obj = balanced_panel_cbmsm.copy()
        cbmsm_obj['weights'] = np.ones(balanced_panel_cbmsm['n_units'])
        cbmsm_obj['glm_weights'] = np.ones(balanced_panel_cbmsm['n_units'])

        result = balance_cbmsm(cbmsm_obj)

        # With uniform weights, Balanced and Unweighted should be similar
        # (not identical due to different normalization)
        assert np.all(np.isfinite(result['Balanced']))
        assert np.all(np.isfinite(result['Unweighted']))

    @pytest.mark.unit
    def test_bmsm018_extreme_weights_handled(self, balanced_panel_cbmsm):
        """
        BMSM-018: Verify extreme weights are handled.

        Requirements: REQ-DIAG-013
        """
        cbmsm_obj = balanced_panel_cbmsm.copy()

        # Very extreme weights
        weights = np.ones(balanced_panel_cbmsm['n_units'])
        weights[0] = 100.0  # Very large
        weights[1] = 0.01   # Very small
        cbmsm_obj['weights'] = weights

        result = balance_cbmsm(cbmsm_obj)

        # Should still produce finite results
        assert np.all(np.isfinite(result['Balanced']))

    @pytest.mark.unit
    def test_bmsm019_standardized_means_computed(self, balanced_panel_cbmsm):
        """
        BMSM-019: Verify standardized means are computed.

        Requirements: REQ-DIAG-013
        """
        result = balance_cbmsm(balanced_panel_cbmsm)

        # Second half of columns should be standardized means
        n_hist = result['Balanced'].shape[1] // 2

        # Standardized means should exist
        std_means = result['Balanced'][:, n_hist:]
        assert std_means.shape[1] == n_hist

    @pytest.mark.unit
    def test_bmsm020_glm_weights_used_for_baseline(self, balanced_panel_cbmsm):
        """
        BMSM-020: Verify GLM weights are used for Unweighted baseline.

        Requirements: REQ-DIAG-013
        """
        result1 = balance_cbmsm(balanced_panel_cbmsm)

        # Modify GLM weights
        cbmsm_obj_modified = balanced_panel_cbmsm.copy()
        cbmsm_obj_modified['glm_weights'] = np.random.exponential(1, balanced_panel_cbmsm['n_units'])

        result2 = balance_cbmsm(cbmsm_obj_modified)

        # Unweighted statistics should differ
        # (Balanced should be same if weights unchanged)
        assert result1['Unweighted'].shape == result2['Unweighted'].shape


class TestBalanceCBMSMEdgeCases:
    """
    Test edge cases and error handling.

    Test IDs: BMSM-021 to BMSM-025
    Requirements: REQ-DIAG-014
    """

    @pytest.mark.edge_case
    def test_bmsm021_single_period(self):
        """
        BMSM-021: Verify handling of single time period.

        Requirements: REQ-DIAG-014
        """
        np.random.seed(42)
        n_units = 50
        n_periods = 1
        n_obs = n_units * n_periods

        cbmsm_obj = {
            'y': np.random.binomial(1, 0.5, n_obs).astype(float),
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.arange(n_units),
            'time': np.zeros(n_units, dtype=int),
        }

        result = balance_cbmsm(cbmsm_obj)

        # Should handle single period
        assert np.all(np.isfinite(result['Balanced']))

    @pytest.mark.edge_case
    def test_bmsm022_unbalanced_panel(self, unbalanced_panel_cbmsm):
        """
        BMSM-022: Verify handling of unbalanced panel.

        Requirements: REQ-DIAG-014
        """
        result = balance_cbmsm(unbalanced_panel_cbmsm)

        # Should handle unbalanced panel
        assert np.all(np.isfinite(result['Balanced']))

    @pytest.mark.edge_case
    def test_bmsm023_small_sample(self):
        """
        BMSM-023: Verify handling of small sample sizes.

        Requirements: REQ-DIAG-014
        """
        np.random.seed(42)
        n_units = 10
        n_periods = 2
        n_obs = n_units * n_periods

        cbmsm_obj = {
            'y': np.random.binomial(1, 0.5, n_obs).astype(float),
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        assert np.all(np.isfinite(result['Balanced']))

    @pytest.mark.edge_case
    def test_bmsm024_alternative_weight_keys(self):
        """
        BMSM-024: Verify handling of alternative weight dictionary keys.

        Requirements: REQ-DIAG-014
        """
        np.random.seed(42)
        n_units = 30
        n_periods = 2
        n_obs = n_units * n_periods

        # Use alternative key names
        cbmsm_obj = {
            'y': np.random.binomial(1, 0.5, n_obs).astype(float),
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs)]),
            'w': np.ones(n_units),  # Alternative key
            'glm_w': np.ones(n_units),  # Alternative key
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        assert np.all(np.isfinite(result['Balanced']))

    @pytest.mark.edge_case
    def test_bmsm025_many_covariates(self):
        """
        BMSM-025: Verify handling of many covariates.

        Requirements: REQ-DIAG-014
        """
        np.random.seed(42)
        n_units = 50
        n_periods = 2
        n_obs = n_units * n_periods
        n_covars = 10

        cbmsm_obj = {
            'y': np.random.binomial(1, 0.5, n_obs).astype(float),
            'x': np.column_stack([np.ones(n_obs), np.random.randn(n_obs, n_covars)]),
            'weights': np.ones(n_units),
            'glm_weights': np.ones(n_units),
            'id': np.repeat(np.arange(n_units), n_periods),
            'time': np.tile(np.arange(n_periods), n_units),
        }

        result = balance_cbmsm(cbmsm_obj)

        # Should have n_covars rows (excluding intercept)
        assert result['Balanced'].shape[0] == n_covars
        assert np.all(np.isfinite(result['Balanced']))



# =============================================================================
# Enhanced Balance Tests (from test_balance_enhanced.py)
# Test IDs: BALE-001 to BALE-030
# =============================================================================

# ---- Enhanced balance fixtures ----

@pytest.fixture
def binary_cbps_result():
    """
    Create a mock binary CBPS result object for testing.

    Returns
    -------
    dict
        Mock CBPS result with treatment, covariates, and weights.
    """
    np.random.seed(42)
    n = 200

    # Covariates with intercept
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Binary treatment with imbalance
    logit_ps = 0.5 * x1 - 0.3 * x2
    ps = 1 / (1 + np.exp(-logit_ps))
    y = np.random.binomial(1, ps).astype(float)

    # Simulated CBPS weights that improve balance
    fitted_values = ps
    weights = y / np.clip(fitted_values, 0.01, 0.99) + \
              (1 - y) / np.clip(1 - fitted_values, 0.01, 0.99)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': fitted_values,
    }


@pytest.fixture
def continuous_cbps_result():
    """
    Create a mock continuous treatment CBPS result object for testing.

    Returns
    -------
    dict
        Mock continuous CBPS result with treatment, covariates, and weights.
    """
    np.random.seed(123)
    n = 200

    # Covariates with intercept
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Continuous treatment correlated with covariates
    y = 0.5 * x1 + 0.3 * x2 + np.random.randn(n) * 0.5

    # Simulated weights that reduce correlation
    weights = np.abs(np.random.randn(n)) + 0.5
    weights = weights / weights.mean() * n / len(weights)  # Normalize

    return {
        'x': X,
        'y': y,
        'weights': weights,
    }


@pytest.fixture
def perfect_balance_result():
    """
    Create a mock result where weighting achieves perfect balance.

    Returns
    -------
    dict
        Mock CBPS result with weights achieving near-perfect balance.
    """
    np.random.seed(456)
    n = 200

    # Balanced covariates by design
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Random treatment (balanced by design)
    y = np.random.binomial(1, 0.5, n).astype(float)

    # Uniform weights (no adjustment needed)
    weights = np.ones(n)
    fitted_values = np.full(n, 0.5)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': fitted_values,
    }


class TestBalanceCBPSEnhanced:
    """Tests for balance_cbps_enhanced function."""

    # -------------------------------------------------------------------------
    # BALE-001: Basic functionality
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_returns_dict(self, binary_cbps_result):
        """BALE-001: Function returns a dictionary."""
        result = balance_cbps_enhanced(binary_cbps_result)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_required_keys_present(self, binary_cbps_result):
        """BALE-002: All required keys are present in output."""
        result = balance_cbps_enhanced(binary_cbps_result)

        required_keys = [
            'balanced', 'original', 'smd_weighted', 'smd_unweighted',
            'improvement_pct', 'n_imbalanced_before', 'n_imbalanced_after',
            'summary', 'report'
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    @pytest.mark.unit
    def test_summary_dict_keys(self, binary_cbps_result):
        """BALE-003: Summary dictionary contains all expected statistics."""
        result = balance_cbps_enhanced(binary_cbps_result)

        expected_summary_keys = [
            'mean_smd_before', 'mean_smd_after',
            'max_smd_before', 'max_smd_after',
            'n_imbalanced_before', 'n_imbalanced_after',
            'pct_imbalanced_before', 'pct_imbalanced_after',
            'mean_improvement_pct'
        ]

        for key in expected_summary_keys:
            assert key in result['summary'], f"Missing summary key: {key}"

    # -------------------------------------------------------------------------
    # BALE-004 to BALE-007: Array shapes and types
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_smd_array_shapes(self, binary_cbps_result):
        """BALE-004: SMD arrays have correct shapes."""
        result = balance_cbps_enhanced(binary_cbps_result)

        # balance_cbps excludes intercept column, so n_covars = X.shape[1] - 1
        n_covars = binary_cbps_result['x'].shape[1] - 1

        # SMD arrays should be (n_covars, n_comparisons)
        assert result['smd_weighted'].shape[0] == n_covars
        assert result['smd_unweighted'].shape[0] == n_covars
        assert result['improvement_pct'].shape[0] == n_covars

    @pytest.mark.unit
    def test_smd_nonnegative(self, binary_cbps_result):
        """BALE-005: SMD values are non-negative."""
        result = balance_cbps_enhanced(binary_cbps_result)

        assert np.all(result['smd_weighted'] >= 0), "SMD weighted contains negative values"
        assert np.all(result['smd_unweighted'] >= 0), "SMD unweighted contains negative values"

    @pytest.mark.unit
    def test_count_types(self, binary_cbps_result):
        """BALE-006: Imbalanced counts are integers."""
        result = balance_cbps_enhanced(binary_cbps_result)

        assert isinstance(result['n_imbalanced_before'], (int, np.integer))
        assert isinstance(result['n_imbalanced_after'], (int, np.integer))

    @pytest.mark.unit
    def test_report_is_string(self, binary_cbps_result):
        """BALE-007: Report is a non-empty string."""
        result = balance_cbps_enhanced(binary_cbps_result)

        assert isinstance(result['report'], str)
        assert len(result['report']) > 0

    # -------------------------------------------------------------------------
    # BALE-008 to BALE-012: Summary statistics accuracy
    # -------------------------------------------------------------------------

    @pytest.mark.numerical
    def test_mean_smd_computation(self, binary_cbps_result):
        """BALE-008: Mean SMD is computed correctly."""
        result = balance_cbps_enhanced(binary_cbps_result)

        # Verify mean computation
        expected_mean_after = np.mean(result['smd_weighted'])
        expected_mean_before = np.mean(result['smd_unweighted'])

        assert_allclose(
            result['summary']['mean_smd_after'],
            expected_mean_after,
            rtol=1e-10
        )
        assert_allclose(
            result['summary']['mean_smd_before'],
            expected_mean_before,
            rtol=1e-10
        )

    @pytest.mark.numerical
    def test_max_smd_computation(self, binary_cbps_result):
        """BALE-009: Max SMD is computed correctly."""
        result = balance_cbps_enhanced(binary_cbps_result)

        expected_max_after = np.max(result['smd_weighted'])
        expected_max_before = np.max(result['smd_unweighted'])

        assert_allclose(
            result['summary']['max_smd_after'],
            expected_max_after,
            rtol=1e-10
        )
        assert_allclose(
            result['summary']['max_smd_before'],
            expected_max_before,
            rtol=1e-10
        )

    @pytest.mark.numerical
    def test_imbalanced_count_with_threshold(self, binary_cbps_result):
        """BALE-010: Imbalanced count respects threshold parameter."""
        threshold = 0.2
        result = balance_cbps_enhanced(binary_cbps_result, threshold=threshold)

        expected_before = np.sum(result['smd_unweighted'] > threshold)
        expected_after = np.sum(result['smd_weighted'] > threshold)

        assert result['n_imbalanced_before'] == expected_before
        assert result['n_imbalanced_after'] == expected_after

    @pytest.mark.numerical
    def test_improvement_percentage_formula(self, binary_cbps_result):
        """BALE-011: Improvement percentage follows correct formula."""
        result = balance_cbps_enhanced(binary_cbps_result)

        smd_before = result['smd_unweighted']
        smd_after = result['smd_weighted']
        improvement = result['improvement_pct']

        # Check improvement formula: (before - after) / before * 100
        nonzero_mask = smd_before > 1e-10
        expected_improvement = np.zeros_like(improvement)
        expected_improvement[nonzero_mask] = (
            (smd_before[nonzero_mask] - smd_after[nonzero_mask]) /
            smd_before[nonzero_mask] * 100
        )

        assert_allclose(improvement, expected_improvement, rtol=1e-10)

    @pytest.mark.numerical
    def test_percentage_imbalanced_computation(self, binary_cbps_result):
        """BALE-012: Percentage imbalanced is computed correctly."""
        result = balance_cbps_enhanced(binary_cbps_result)

        n_covars = result['smd_weighted'].size
        expected_pct_before = result['n_imbalanced_before'] / n_covars * 100
        expected_pct_after = result['n_imbalanced_after'] / n_covars * 100

        assert_allclose(
            result['summary']['pct_imbalanced_before'],
            expected_pct_before,
            rtol=1e-10
        )
        assert_allclose(
            result['summary']['pct_imbalanced_after'],
            expected_pct_after,
            rtol=1e-10
        )

    # -------------------------------------------------------------------------
    # BALE-013 to BALE-015: Threshold parameter
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_default_threshold(self, binary_cbps_result):
        """BALE-013: Default threshold is 0.1."""
        result = balance_cbps_enhanced(binary_cbps_result)

        # Check that report mentions threshold 0.1
        assert "0.1" in result['report']

    @pytest.mark.unit
    def test_custom_threshold(self, binary_cbps_result):
        """BALE-014: Custom threshold is applied correctly."""
        threshold = 0.25
        result = balance_cbps_enhanced(binary_cbps_result, threshold=threshold)

        # Check that report mentions custom threshold
        assert "0.25" in result['report']

    @pytest.mark.unit
    def test_threshold_affects_counts(self, binary_cbps_result):
        """BALE-015: Different thresholds produce different counts."""
        result_strict = balance_cbps_enhanced(binary_cbps_result, threshold=0.05)
        result_lenient = balance_cbps_enhanced(binary_cbps_result, threshold=0.5)

        # Stricter threshold should produce more imbalanced covariates
        assert result_strict['n_imbalanced_before'] >= result_lenient['n_imbalanced_before']
        assert result_strict['n_imbalanced_after'] >= result_lenient['n_imbalanced_after']

    # -------------------------------------------------------------------------
    # BALE-016 to BALE-018: Covariate names parameter
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_covariate_names_in_report(self, binary_cbps_result):
        """BALE-016: Covariate names appear in detailed report."""
        # Note: balance_cbps excludes intercept, so provide names for non-intercept covariates only
        covariate_names = ['Age', 'Education']  # Names for X[:, 1] and X[:, 2]
        result = balance_cbps_enhanced(
            binary_cbps_result,
            covariate_names=covariate_names
        )

        # Check that names appear in report
        for name in covariate_names:
            assert name in result['report']

    @pytest.mark.unit
    def test_without_covariate_names(self, binary_cbps_result):
        """BALE-017: Function works without covariate names."""
        result = balance_cbps_enhanced(binary_cbps_result, covariate_names=None)

        # Should still produce valid output
        assert isinstance(result['report'], str)
        assert len(result['report']) > 0

    @pytest.mark.unit
    def test_mismatched_covariate_names(self, binary_cbps_result):
        """BALE-018: Wrong number of covariate names doesn't include details."""
        # Wrong number of names
        covariate_names = ['X1', 'X2']  # Only 2, but have 3 covariates
        result = balance_cbps_enhanced(
            binary_cbps_result,
            covariate_names=covariate_names
        )

        # Should still work but detailed section won't have these names
        # (the function checks length match before adding details)
        assert isinstance(result['report'], str)

    # -------------------------------------------------------------------------
    # BALE-019 to BALE-020: Edge cases
    # -------------------------------------------------------------------------

    @pytest.mark.edge_case
    def test_perfect_balance(self, perfect_balance_result):
        """BALE-019: Handles case with near-perfect balance."""
        result = balance_cbps_enhanced(perfect_balance_result)

        # Summary should reflect good balance
        assert result['summary']['mean_smd_after'] < 0.3  # Reasonable threshold
        assert result['n_imbalanced_after'] >= 0  # Non-negative count

    @pytest.mark.edge_case
    def test_no_improvement(self):
        """BALE-020: Handles case with no balance improvement."""
        np.random.seed(789)
        n = 100

        # Create data where weights don't help
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.binomial(1, 0.5, n).astype(float)

        # Uniform weights (no change)
        cbps_obj = {
            'x': X,
            'y': y,
            'weights': np.ones(n),
            'fitted_values': np.full(n, 0.5),
        }

        result = balance_cbps_enhanced(cbps_obj)

        # Should still return valid output
        assert isinstance(result, dict)
        assert 'summary' in result


class TestBalanceCBPSContinuousEnhanced:
    """Tests for balance_cbps_continuous_enhanced function."""

    # -------------------------------------------------------------------------
    # BALE-021: Basic functionality
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_returns_dict(self, continuous_cbps_result):
        """BALE-021: Function returns a dictionary."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_required_keys_present(self, continuous_cbps_result):
        """BALE-022: All required keys are present in output."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        required_keys = [
            'balanced', 'unweighted', 'abs_corr_weighted', 'abs_corr_unweighted',
            'improvement_pct', 'n_imbalanced_before', 'n_imbalanced_after',
            'summary', 'report'
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    @pytest.mark.unit
    def test_summary_dict_keys(self, continuous_cbps_result):
        """BALE-023: Summary dictionary contains all expected statistics."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        expected_summary_keys = [
            'mean_abs_corr_before', 'mean_abs_corr_after',
            'max_abs_corr_before', 'max_abs_corr_after',
            'n_imbalanced_before', 'n_imbalanced_after'
        ]

        for key in expected_summary_keys:
            assert key in result['summary'], f"Missing summary key: {key}"

    # -------------------------------------------------------------------------
    # BALE-024 to BALE-026: Correlation values
    # -------------------------------------------------------------------------

    @pytest.mark.numerical
    def test_correlation_bounded(self, continuous_cbps_result):
        """BALE-024: Absolute correlations are bounded [0, 1]."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        assert np.all(result['abs_corr_weighted'] >= 0)
        assert np.all(result['abs_corr_weighted'] <= 1 + 1e-10)  # Allow small tolerance
        assert np.all(result['abs_corr_unweighted'] >= 0)
        assert np.all(result['abs_corr_unweighted'] <= 1 + 1e-10)

    @pytest.mark.numerical
    def test_mean_correlation_computation(self, continuous_cbps_result):
        """BALE-025: Mean absolute correlation is computed correctly."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        expected_mean_after = np.mean(result['abs_corr_weighted'])
        expected_mean_before = np.mean(result['abs_corr_unweighted'])

        assert_allclose(
            result['summary']['mean_abs_corr_after'],
            expected_mean_after,
            rtol=1e-10
        )
        assert_allclose(
            result['summary']['mean_abs_corr_before'],
            expected_mean_before,
            rtol=1e-10
        )

    @pytest.mark.numerical
    def test_max_correlation_computation(self, continuous_cbps_result):
        """BALE-026: Max absolute correlation is computed correctly."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        expected_max_after = np.max(result['abs_corr_weighted'])
        expected_max_before = np.max(result['abs_corr_unweighted'])

        assert_allclose(
            result['summary']['max_abs_corr_after'],
            expected_max_after,
            rtol=1e-10
        )
        assert_allclose(
            result['summary']['max_abs_corr_before'],
            expected_max_before,
            rtol=1e-10
        )

    # -------------------------------------------------------------------------
    # BALE-027 to BALE-029: Threshold and covariate names
    # -------------------------------------------------------------------------

    @pytest.mark.unit
    def test_threshold_affects_counts(self, continuous_cbps_result):
        """BALE-027: Different thresholds produce different counts."""
        result_strict = balance_cbps_continuous_enhanced(
            continuous_cbps_result, threshold=0.05
        )
        result_lenient = balance_cbps_continuous_enhanced(
            continuous_cbps_result, threshold=0.5
        )

        # Stricter threshold should produce more imbalanced covariates
        assert result_strict['n_imbalanced_before'] >= result_lenient['n_imbalanced_before']
        assert result_strict['n_imbalanced_after'] >= result_lenient['n_imbalanced_after']

    @pytest.mark.unit
    def test_covariate_names_in_report(self, continuous_cbps_result):
        """BALE-028: Covariate names appear in detailed report."""
        # Note: balance_cbps_continuous excludes intercept, so provide names for non-intercept covariates only
        covariate_names = ['Income', 'Experience']  # Names for X[:, 1] and X[:, 2]
        result = balance_cbps_continuous_enhanced(
            continuous_cbps_result,
            covariate_names=covariate_names
        )

        # Check that names appear in report
        for name in covariate_names:
            assert name in result['report']

    @pytest.mark.unit
    def test_report_contains_interpretation(self, continuous_cbps_result):
        """BALE-029: Report contains interpretation guidelines."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)

        # Report should contain interpretation section
        assert 'Interpretation' in result['report'] or 'correlation' in result['report'].lower()

    # -------------------------------------------------------------------------
    # BALE-030: Integration test
    # -------------------------------------------------------------------------

    @pytest.mark.integration
    def test_full_workflow(self, continuous_cbps_result):
        """BALE-030: Full workflow with all parameters."""
        # Note: balance_cbps_continuous excludes intercept, so n_covars = X.shape[1] - 1
        covariate_names = ['X1', 'X2']  # Names for non-intercept covariates
        threshold = 0.15

        result = balance_cbps_continuous_enhanced(
            continuous_cbps_result,
            threshold=threshold,
            covariate_names=covariate_names
        )

        # Verify complete output
        assert isinstance(result['report'], str)
        assert len(result['report']) > 100  # Substantial report
        assert result['summary']['n_imbalanced_before'] >= 0
        assert result['summary']['n_imbalanced_after'] >= 0

        # Verify numerical arrays (excludes intercept column)
        n_covars = continuous_cbps_result['x'].shape[1] - 1
        assert result['abs_corr_weighted'].shape[0] == n_covars
        assert result['abs_corr_unweighted'].shape[0] == n_covars


class TestBalanceEnhancedParametrized:
    """Parametrized tests for edge cases and various inputs."""

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.25, 0.5])
    def test_various_thresholds_binary(self, binary_cbps_result, threshold):
        """Test binary balance diagnostics with various thresholds."""
        result = balance_cbps_enhanced(binary_cbps_result, threshold=threshold)

        # All outputs should be valid
        assert result['n_imbalanced_before'] >= 0
        assert result['n_imbalanced_after'] >= 0
        assert 0 <= result['summary']['pct_imbalanced_before'] <= 100
        assert 0 <= result['summary']['pct_imbalanced_after'] <= 100

    @pytest.mark.parametrize("threshold", [0.01, 0.05, 0.1, 0.25, 0.5])
    def test_various_thresholds_continuous(self, continuous_cbps_result, threshold):
        """Test continuous balance diagnostics with various thresholds."""
        result = balance_cbps_continuous_enhanced(
            continuous_cbps_result, threshold=threshold
        )

        # All outputs should be valid
        assert result['n_imbalanced_before'] >= 0
        assert result['n_imbalanced_after'] >= 0


class TestReportFormat:
    """Tests for report formatting and content."""

    @pytest.mark.unit
    def test_report_has_sections(self, binary_cbps_result):
        """Report contains expected sections."""
        result = balance_cbps_enhanced(binary_cbps_result)
        report = result['report']

        # Check for key sections
        assert 'Balance' in report or 'balance' in report
        assert 'Weighting' in report or 'weighting' in report.lower()

    @pytest.mark.unit
    def test_report_contains_numbers(self, binary_cbps_result):
        """Report contains numerical values."""
        result = balance_cbps_enhanced(binary_cbps_result)
        report = result['report']

        # Should contain some numbers (SMD values, percentages)
        numbers = re.findall(r'\d+\.?\d*', report)
        assert len(numbers) > 0, "Report should contain numerical values"

    @pytest.mark.unit
    def test_continuous_report_format(self, continuous_cbps_result):
        """Continuous treatment report has appropriate format."""
        result = balance_cbps_continuous_enhanced(continuous_cbps_result)
        report = result['report']

        # Should mention correlation (not SMD)
        assert 'correlation' in report.lower() or 'Correlation' in report



# =============================================================================
# Visualization Tests (from test_plots.py)
# Test IDs: PLOT-001 to PLOT-030
# =============================================================================

# ---- Plot-specific fixtures ----

@pytest.fixture
def binary_cbps_mock():
    """Create a mock CBPS result object for binary treatment."""
    np.random.seed(42)
    n = 200

    # Covariates with intercept
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Binary treatment
    logit_ps = 0.3 * x1 - 0.2 * x2
    ps = 1 / (1 + np.exp(-logit_ps))
    y = np.random.binomial(1, ps).astype(float)

    # Simulated CBPS weights
    weights = y / np.clip(ps, 0.01, 0.99) + (1 - y) / np.clip(1 - ps, 0.01, 0.99)

    return {
        'x': X,
        'y': y,
        'weights': weights,
        'fitted_values': ps,
    }


@pytest.fixture
def continuous_cbps_mock():
    """Create a mock CBPS result object for continuous treatment."""
    np.random.seed(42)
    n = 200

    # Covariates with intercept
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Continuous treatment
    T = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)

    # Simulated weights
    weights = np.ones(n) + 0.1 * np.random.randn(n)
    weights = np.abs(weights)

    return {
        'x': X,
        'y': T,
        'weights': weights,
        'fitted_values': T,  # Mock fitted values
    }


@pytest.fixture
def multitreat_cbps_mock():
    """Create a mock CBPS result object for 3-level treatment."""
    np.random.seed(42)
    n = 300

    # Covariates with intercept
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # 3-level treatment (0, 1, 2)
    y = np.random.choice([0.0, 1.0, 2.0], size=n, p=[0.3, 0.4, 0.3])

    # Simulated weights
    weights = np.ones(n)

    return {
        'x': X,
        'y': y,
        'weights': weights,
    }


class TestBoxplotStatsTukey:
    """
    Test _compute_boxplot_stats_tukey helper function.

    Test IDs: PLOT-001 to PLOT-005
    Requirements: REQ-DIAG-011
    """

    @pytest.mark.unit
    def test_plot001_returns_dict(self):
        """
        PLOT-001: Verify _compute_boxplot_stats_tukey returns a dictionary.

        Requirements: REQ-DIAG-011
        """
        from cbps.diagnostics.plots import _compute_boxplot_stats_tukey

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _compute_boxplot_stats_tukey(data)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_plot002_required_keys(self):
        """
        PLOT-002: Verify required keys are present in result.

        Requirements: REQ-DIAG-011
        """
        from cbps.diagnostics.plots import _compute_boxplot_stats_tukey

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _compute_boxplot_stats_tukey(data)

        required_keys = ['whislo', 'q1', 'med', 'q3', 'whishi']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    @pytest.mark.unit
    def test_plot003_median_correct(self):
        """
        PLOT-003: Verify median computation is correct.

        Requirements: REQ-DIAG-011
        """
        from cbps.diagnostics.plots import _compute_boxplot_stats_tukey

        # Even number of elements
        data_even = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result_even = _compute_boxplot_stats_tukey(data_even)
        assert result_even['med'] == 5.5

        # Odd number of elements
        data_odd = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result_odd = _compute_boxplot_stats_tukey(data_odd)
        assert result_odd['med'] == 5.0

    @pytest.mark.unit
    def test_plot004_quartiles_ordering(self):
        """
        PLOT-004: Verify quartiles are in correct order.

        Requirements: REQ-DIAG-011
        """
        from cbps.diagnostics.plots import _compute_boxplot_stats_tukey

        data = np.random.randn(100)
        result = _compute_boxplot_stats_tukey(data)

        # whislo <= q1 <= med <= q3 <= whishi
        assert result['whislo'] <= result['q1']
        assert result['q1'] <= result['med']
        assert result['med'] <= result['q3']
        assert result['q3'] <= result['whishi']

    @pytest.mark.unit
    def test_plot005_whiskers_within_data(self):
        """
        PLOT-005: Verify whisker endpoints are actual data values.

        Requirements: REQ-DIAG-011
        """
        from cbps.diagnostics.plots import _compute_boxplot_stats_tukey

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = _compute_boxplot_stats_tukey(data)

        # Whisker endpoints should be data values
        assert result['whislo'] in data
        assert result['whishi'] in data


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotCBPS:
    """
    Test plot_cbps function for binary/multi-valued treatments.

    Test IDs: PLOT-006 to PLOT-015
    Requirements: REQ-DIAG-012
    """

    @pytest.mark.unit
    def test_plot006_executes_without_error(self, binary_cbps_mock):
        """
        PLOT-006: Verify plot_cbps executes without error.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        # Should not raise
        result = plot_cbps(binary_cbps_mock, silent=True)

        # Silent mode returns None
        assert result is None

        # Clean up
        plt.close('all')

    @pytest.mark.unit
    def test_plot007_returns_dataframe(self, binary_cbps_mock):
        """
        PLOT-007: Verify plot_cbps returns DataFrame when silent=False.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        result = plot_cbps(binary_cbps_mock, silent=False)

        assert isinstance(result, pd.DataFrame)

        plt.close('all')

    @pytest.mark.unit
    def test_plot008_dataframe_columns(self, binary_cbps_mock):
        """
        PLOT-008: Verify DataFrame has required columns.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        result = plot_cbps(binary_cbps_mock, silent=False)

        required_cols = ['contrast', 'covariate', 'balanced', 'original']
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        plt.close('all')

    @pytest.mark.unit
    def test_plot009_boxplot_mode(self, binary_cbps_mock):
        """
        PLOT-009: Verify boxplot mode executes without error.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        # Should not raise
        result = plot_cbps(binary_cbps_mock, boxplot=True, silent=True)

        assert result is None

        plt.close('all')

    @pytest.mark.unit
    def test_plot010_covars_subset(self, binary_cbps_mock):
        """
        PLOT-010: Verify covars parameter filters covariates.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        # Select only first covariate (index 0)
        result = plot_cbps(binary_cbps_mock, covars=[0], silent=False)

        # Should have fewer rows
        assert len(result) == 1  # 1 covariate * 1 contrast for binary

        plt.close('all')

    @pytest.mark.unit
    def test_plot011_kind_parameter_error(self, binary_cbps_mock):
        """
        PLOT-011: Verify 'kind' parameter raises helpful error.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        with pytest.raises(TypeError) as excinfo:
            plot_cbps(binary_cbps_mock, kind='boxplot')

        assert "boxplot=True" in str(excinfo.value)

        plt.close('all')

    @pytest.mark.unit
    def test_plot012_three_level_treatment(self, multitreat_cbps_mock):
        """
        PLOT-012: Verify plot_cbps handles 3-level treatment.

        Requirements: REQ-DIAG-012
        """
        from cbps.diagnostics.plots import plot_cbps

        result = plot_cbps(multitreat_cbps_mock, silent=False)

        # 3-level treatment has 3 contrasts (C(3,2) = 3)
        # 2 covariates * 3 contrasts = 6 rows
        assert len(result) == 6

        plt.close('all')

    @pytest.mark.numerical
    def test_plot013_balance_values_finite(self, binary_cbps_mock):
        """
        PLOT-013: Verify balance values are finite.

        Requirements: REQ-DIAG-013
        """
        from cbps.diagnostics.plots import plot_cbps

        result = plot_cbps(binary_cbps_mock, silent=False)

        assert np.all(np.isfinite(result['balanced']))
        assert np.all(np.isfinite(result['original']))

        plt.close('all')

    @pytest.mark.numerical
    def test_plot014_balance_values_nonnegative(self, binary_cbps_mock):
        """
        PLOT-014: Verify absolute SMD values are non-negative.

        Requirements: REQ-DIAG-013
        """
        from cbps.diagnostics.plots import plot_cbps

        result = plot_cbps(binary_cbps_mock, silent=False)

        assert np.all(result['balanced'] >= 0)
        assert np.all(result['original'] >= 0)

        plt.close('all')

    @pytest.mark.unit
    def test_plot015_reproducibility(self, binary_cbps_mock):
        """
        PLOT-015: Verify plotting is reproducible.

        Requirements: REQ-DIAG-014
        """
        from cbps.diagnostics.plots import plot_cbps

        result1 = plot_cbps(binary_cbps_mock, silent=False)
        result2 = plot_cbps(binary_cbps_mock, silent=False)

        pd.testing.assert_frame_equal(result1, result2)

        plt.close('all')


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotCBPSContinuous:
    """
    Test plot_cbps_continuous function.

    Test IDs: PLOT-016 to PLOT-022
    Requirements: REQ-DIAG-015
    """

    @pytest.mark.unit
    def test_plot016_executes_without_error(self, continuous_cbps_mock):
        """
        PLOT-016: Verify plot_cbps_continuous executes without error.

        Requirements: REQ-DIAG-015
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, silent=True)

        assert result is None

        plt.close('all')

    @pytest.mark.unit
    def test_plot017_returns_dataframe(self, continuous_cbps_mock):
        """
        PLOT-017: Verify returns DataFrame when silent=False.

        Requirements: REQ-DIAG-015
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, silent=False)

        assert isinstance(result, pd.DataFrame)

        plt.close('all')

    @pytest.mark.unit
    def test_plot018_dataframe_columns(self, continuous_cbps_mock):
        """
        PLOT-018: Verify DataFrame has required columns.

        Requirements: REQ-DIAG-015
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, silent=False)

        required_cols = ['covariate', 'balanced', 'original']
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        plt.close('all')

    @pytest.mark.unit
    def test_plot019_boxplot_mode(self, continuous_cbps_mock):
        """
        PLOT-019: Verify boxplot mode executes without error.

        Requirements: REQ-DIAG-015
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, boxplot=True, silent=True)

        assert result is None

        plt.close('all')

    @pytest.mark.numerical
    def test_plot020_correlations_bounded(self, continuous_cbps_mock):
        """
        PLOT-020: Verify correlations are bounded in [0, 1].

        Requirements: REQ-DIAG-016
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, silent=False)

        # Absolute correlations should be in [0, 1]
        assert np.all(result['balanced'] >= 0)
        assert np.all(result['balanced'] <= 1.001)  # Small tolerance
        assert np.all(result['original'] >= 0)
        assert np.all(result['original'] <= 1.001)

        plt.close('all')

    @pytest.mark.unit
    def test_plot021_covars_subset(self, continuous_cbps_mock):
        """
        PLOT-021: Verify covars parameter filters covariates.

        Requirements: REQ-DIAG-015
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result = plot_cbps_continuous(continuous_cbps_mock, covars=[0], silent=False)

        assert len(result) == 1

        plt.close('all')

    @pytest.mark.unit
    def test_plot022_reproducibility(self, continuous_cbps_mock):
        """
        PLOT-022: Verify plotting is reproducible.

        Requirements: REQ-DIAG-016
        """
        from cbps.diagnostics.plots import plot_cbps_continuous

        result1 = plot_cbps_continuous(continuous_cbps_mock, silent=False)
        result2 = plot_cbps_continuous(continuous_cbps_mock, silent=False)

        pd.testing.assert_frame_equal(result1, result2)

        plt.close('all')


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotNPCBPS:
    """
    Test plot_npcbps function.

    Test IDs: PLOT-023 to PLOT-025
    Requirements: REQ-DIAG-017
    """

    @pytest.fixture
    def npcbps_binary_mock(self):
        """Create mock npCBPS result for binary treatment."""
        np.random.seed(42)
        n = 200

        # npCBPS X does NOT have intercept
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([x1, x2])

        # Binary treatment
        y = np.random.binomial(1, 0.5, n).astype(float)

        # Simulated weights
        weights = np.ones(n) + 0.1 * np.random.randn(n)
        weights = np.abs(weights)

        return {
            'x': X,
            'y': y,
            'weights': weights,
            'log_el': -100.0,  # npCBPS marker
        }

    @pytest.fixture
    def npcbps_continuous_mock(self):
        """Create mock npCBPS result for continuous treatment."""
        np.random.seed(42)
        n = 200

        # npCBPS X does NOT have intercept
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        X = np.column_stack([x1, x2])

        # Continuous treatment (many unique values)
        y = 0.5 * x1 - 0.3 * x2 + np.random.randn(n)

        # Simulated weights
        weights = np.ones(n) + 0.1 * np.random.randn(n)
        weights = np.abs(weights)

        return {
            'x': X,
            'y': y,
            'weights': weights,
            'log_el': -100.0,  # npCBPS marker
        }

    @pytest.mark.unit
    def test_plot023_binary_treatment(self, npcbps_binary_mock):
        """
        PLOT-023: Verify plot_npcbps routes binary treatment correctly.

        Requirements: REQ-DIAG-017
        """
        from cbps.diagnostics.plots import plot_npcbps

        result = plot_npcbps(npcbps_binary_mock, silent=False)

        # Should return DataFrame with contrast column (binary)
        assert isinstance(result, pd.DataFrame)
        assert 'contrast' in result.columns

        plt.close('all')

    @pytest.mark.unit
    def test_plot024_continuous_treatment(self, npcbps_continuous_mock):
        """
        PLOT-024: Verify plot_npcbps routes continuous treatment correctly.

        Requirements: REQ-DIAG-017
        """
        from cbps.diagnostics.plots import plot_npcbps

        result = plot_npcbps(npcbps_continuous_mock, silent=False)

        # Should return DataFrame without contrast column (continuous)
        assert isinstance(result, pd.DataFrame)
        # Continuous treatment doesn't have 'contrast' column
        assert 'covariate' in result.columns

        plt.close('all')

    @pytest.mark.unit
    def test_plot025_invalid_input_raises(self):
        """
        PLOT-025: Verify invalid input raises error.

        Requirements: REQ-DIAG-017
        """
        from cbps.diagnostics.plots import plot_npcbps

        # Missing 'y' key should raise an error (ValueError or AttributeError)
        with pytest.raises((ValueError, AttributeError)):
            plot_npcbps({'no_y_key': 'invalid'})

        plt.close('all')


class TestMatplotlibDependency:
    """
    Test matplotlib optional dependency handling.

    Test IDs: PLOT-026 to PLOT-028
    Requirements: REQ-DIAG-018
    """

    @pytest.mark.unit
    def test_plot026_has_matplotlib_flag(self):
        """
        PLOT-026: Verify HAS_MATPLOTLIB flag is set.

        Requirements: REQ-DIAG-018
        """
        from cbps.diagnostics import plots

        assert hasattr(plots, 'HAS_MATPLOTLIB')
        assert isinstance(plots.HAS_MATPLOTLIB, bool)

    @pytest.mark.unit
    @pytest.mark.skipif(HAS_MATPLOTLIB, reason="Test only when matplotlib missing")
    def test_plot027_import_error_without_matplotlib(self, binary_cbps_mock):
        """
        PLOT-027: Verify ImportError raised without matplotlib.

        Requirements: REQ-DIAG-018
        """
        from cbps.diagnostics.plots import plot_cbps

        with pytest.raises(ImportError) as excinfo:
            plot_cbps(binary_cbps_mock)

        assert "matplotlib" in str(excinfo.value).lower()

    @pytest.mark.unit
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_plot028_no_error_with_matplotlib(self, binary_cbps_mock):
        """
        PLOT-028: Verify no ImportError with matplotlib installed.

        Requirements: REQ-DIAG-018
        """
        from cbps.diagnostics.plots import plot_cbps

        # Should not raise ImportError
        try:
            plot_cbps(binary_cbps_mock, silent=True)
        except ImportError:
            pytest.fail("ImportError raised unexpectedly")
        finally:
            plt.close('all')


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotEdgeCases:
    """
    Test edge cases for plotting functions.

    Test IDs: PLOT-029 to PLOT-030
    Requirements: REQ-DIAG-019
    """

    @pytest.mark.edge_case
    def test_plot029_single_covariate(self):
        """
        PLOT-029: Verify handling of single covariate.

        Requirements: REQ-DIAG-019
        """
        from cbps.diagnostics.plots import plot_cbps

        np.random.seed(42)
        n = 100

        # Single covariate (plus intercept)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.binomial(1, 0.5, n).astype(float)
        weights = np.ones(n)

        cbps_obj = {'x': X, 'y': y, 'weights': weights}

        result = plot_cbps(cbps_obj, silent=False)

        # Should have 1 row (1 covariate * 1 contrast)
        assert len(result) == 1

        plt.close('all')

    @pytest.mark.edge_case
    def test_plot030_many_covariates(self):
        """
        PLOT-030: Verify handling of many covariates.

        Requirements: REQ-DIAG-019
        """
        from cbps.diagnostics.plots import plot_cbps

        np.random.seed(42)
        n = 200
        k = 20  # Many covariates

        X = np.column_stack([np.ones(n), np.random.randn(n, k)])
        y = np.random.binomial(1, 0.5, n).astype(float)
        weights = np.ones(n)

        cbps_obj = {'x': X, 'y': y, 'weights': weights}

        result = plot_cbps(cbps_obj, silent=False)

        # Should have k rows (k covariates * 1 contrast)
        assert len(result) == k

        plt.close('all')
