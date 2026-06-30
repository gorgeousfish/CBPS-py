"""
CBPS Test Suite Configuration
=============================

Root-level pytest configuration for the CBPS Python package test suite.

This module provides:
    - Shared pytest markers configuration
    - Common fixtures available to all test modules
    - Test collection modifiers
    - Warning filters for clean test output

Test Categories (Markers):
    - unit: Unit tests for individual functions/classes
    - integration: End-to-end integration tests
    - slow: Slow tests (runtime > 10 seconds)
    - paper_reproduction: Paper reproduction tests (Imai & Ratkovic 2014, Fong 2018)
    - edge_case: Edge case tests (extreme probabilities, collinearity, small samples)
    - numerical: Numerical precision tests

Usage:
    Run all tests: pytest
    Skip slow tests: pytest -m "not slow"
    Run only unit tests: pytest -m unit
    Run only paper reproduction tests: pytest -m paper_reproduction

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.
    
    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics 12(1), 156-177.
"""

import warnings
import numpy as np
import pytest


# =============================================================================
# Warning Filters for Clean Test Output
# =============================================================================
# These warnings are expected during testing and are intentionally filtered
# to produce cleaner test output for JOSS/JSS submission review.

# Filter informational warnings that are expected during normal CBPS operation
warnings.filterwarnings(
    "ignore",
    message="Treatment variable is numeric.*Interpreting as binary treatment",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Treatment vector is numeric with.*unique values.*Interpreting as.*treatment",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="ATT parameter.*is not supported for continuous treatments",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Parameter conflict: method='exact'.*incompatible with two_step=True",
    category=UserWarning
)

# Filter numerical conditioning warnings that are expected in edge case tests
warnings.filterwarnings(
    "ignore",
    message="V matrix has poor conditioning",
    category=UserWarning
)

# Note: Warnings that indicate actual problems (e.g., convergence failures,
# extreme propensity scores) are NOT filtered and will still be shown.


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    Note: These markers are synchronized with pyproject.toml [tool.pytest.ini_options].
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions/classes"
    )
    config.addinivalue_line(
        "markers", "integration: End-to-end integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (runtime > 10 seconds)"
    )
    config.addinivalue_line(
        "markers", "paper_reproduction: Paper reproduction tests (Imai & Ratkovic 2014, Fong 2018)"
    )
    config.addinivalue_line(
        "markers", "edge_case: Edge case tests (extreme probabilities, collinearity, small samples)"
    )
    config.addinivalue_line(
        "markers", "numerical: Numerical precision tests"
    )


# =============================================================================
# Session-Level Setup
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """
    Set up the test session.
    
    This fixture:
    - Sets random seed for reproducibility
    - Configures numpy print options for readable output
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure numpy for readable output
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    
    yield
    
    # Cleanup (if needed)
    pass


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def random_state():
    """Provide a fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def tolerance_config():
    """
    Provide tolerance configuration for numerical comparisons.
    
    Returns
    -------
    dict
        Dictionary of tolerance values for different comparison types.
    """
    return {
        'coefficient_rtol': 1e-3,
        'coefficient_atol': 1e-3,
        'weight_rtol': 1e-10,
        'weight_atol': 1e-10,
        'gradient_rtol': 1e-6,
        'gradient_atol': 1e-8,
        'vcov_rtol': 1e-8,
        'vcov_atol': 1e-8,
    }


# =============================================================================
# Numerical Tolerance Constants (shared across binary, nonparametric, etc.)
# =============================================================================

class Tolerances:
    """
    Numerical tolerance constants for test comparisons.

    Calibrated based on machine epsilon (~2.2e-16), expected numerical
    precision of different operations, and acceptable differences between
    Python and R implementations.
    """
    PSEUDOINVERSE_RTOL = 1e-10
    PSEUDOINVERSE_ATOL = 1e-12
    WEIGHT_RTOL = 1e-10
    WEIGHT_ATOL = 1e-10
    V_MATRIX_RTOL = 1e-10
    V_MATRIX_ATOL = 1e-10
    GMM_LOSS_RTOL = 1e-8
    GMM_LOSS_ATOL = 1e-10
    GRADIENT_RTOL = 1e-6
    GRADIENT_ATOL = 1e-8
    OPTIMIZATION_RTOL = 1e-6
    OPTIMIZATION_ATOL = 1e-6
    VCOV_RTOL = 1e-8
    VCOV_ATOL = 1e-8
    COEFFICIENT_RTOL = 0.1
    COEFFICIENT_ATOL = 0.1
    SYMMETRY_ATOL = 1e-15


@pytest.fixture(scope="session")
def tolerances():
    """Provide tolerance constants for tests."""
    return Tolerances()


# Probability clipping threshold (matches R package)
PROBS_MIN = 1e-6


@pytest.fixture(scope="session")
def probs_min():
    """Provide the probability clipping threshold."""
    return PROBS_MIN


# =============================================================================
# Assertion Helper Functions (shared across test modules)
# =============================================================================

def assert_allclose_with_report(actual, expected, rtol=1e-7, atol=0, name="array"):
    """Assert arrays are close with detailed error reporting."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(expected != 0, diff / np.abs(expected), 0)
    max_rel_diff = np.max(rel_diff)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name} comparison failed:\n"
            f"  Max absolute difference: {max_diff:.2e} at index {max_diff_idx}\n"
            f"  Max relative difference: {max_rel_diff:.2e}\n"
            f"  Tolerance: rtol={rtol:.2e}, atol={atol:.2e}\n"
            f"  Actual value at max diff: {actual[max_diff_idx]}\n"
            f"  Expected value at max diff: {expected[max_diff_idx]}"
        )


def assert_matrix_symmetric(matrix, atol=1e-15, name="matrix"):
    """Assert that a matrix is symmetric."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AssertionError(f"{name} is not square: shape {matrix.shape}")
    diff = np.abs(matrix - matrix.T)
    max_diff = np.max(diff)
    if max_diff > atol:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"{name} is not symmetric:\n"
            f"  Max asymmetry: {max_diff:.2e} at index {max_idx}\n"
            f"  Tolerance: {atol:.2e}"
        )


def assert_positive_semidefinite(matrix, atol=1e-10, name="matrix"):
    """Assert that a matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue < -atol:
        raise AssertionError(
            f"{name} is not positive semi-definite:\n"
            f"  Minimum eigenvalue: {min_eigenvalue:.2e}\n"
            f"  Tolerance: {atol:.2e}"
        )


@pytest.fixture
def assert_helpers():
    """Provide assertion helper functions to tests."""
    return {
        'allclose_with_report': assert_allclose_with_report,
        'matrix_symmetric': assert_matrix_symmetric,
        'positive_semidefinite': assert_positive_semidefinite,
    }


# =============================================================================
# CBPS Module Import Fixtures (shared across binary tests)
# =============================================================================

@pytest.fixture(scope="session")
def cbps_binary_module():
    """Import and provide the cbps_binary module."""
    try:
        from cbps.core import cbps_binary
        return cbps_binary
    except ImportError as e:
        pytest.skip(f"cbps_binary module not available: {e}")


@pytest.fixture(scope="session")
def r_ginv(cbps_binary_module):
    """Provide the _r_ginv function."""
    return cbps_binary_module._r_ginv


@pytest.fixture(scope="session")
def att_wt_func(cbps_binary_module):
    """Provide the _att_wt_func function."""
    return cbps_binary_module._att_wt_func


@pytest.fixture(scope="session")
def compute_V_matrix(cbps_binary_module):
    """Provide the _compute_V_matrix function."""
    return cbps_binary_module._compute_V_matrix


@pytest.fixture(scope="session")
def compute_vcov(cbps_binary_module):
    """Provide the _compute_vcov function."""
    return cbps_binary_module._compute_vcov


@pytest.fixture(scope="session")
def gmm_func(cbps_binary_module):
    """Provide the _gmm_func function."""
    return cbps_binary_module._gmm_func


@pytest.fixture(scope="session")
def gmm_loss(cbps_binary_module):
    """Provide the _gmm_loss function."""
    return cbps_binary_module._gmm_loss


@pytest.fixture(scope="session")
def bal_loss(cbps_binary_module):
    """Provide the _bal_loss function."""
    return cbps_binary_module._bal_loss


@pytest.fixture(scope="session")
def cbps_binary_fit(cbps_binary_module):
    """Provide the cbps_binary_fit function."""
    return cbps_binary_module.cbps_binary_fit


# =============================================================================
# R Intermediate Values Fixture
# =============================================================================

@pytest.fixture(scope="session")
def r_intermediate_values():
    """Load R intermediate values for comparison tests, if available."""
    import json
    from pathlib import Path
    json_path = Path(__file__).parent / "binary" / "r_intermediate_values.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


# =============================================================================
# LaLonde Dataset Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def lalonde_data():
    """Load the LaLonde (Dehejia-Wahba) dataset."""
    try:
        from cbps.datasets import load_lalonde
        return load_lalonde(dehejia_wahba_only=True)
    except ImportError:
        pytest.skip("LaLonde dataset not found")


@pytest.fixture(scope="session")
def lalonde_X_treat(lalonde_data):
    """Extract covariate matrix X and treatment vector from LaLonde data."""
    import scipy.special
    covariate_cols = ['age', 'educ', 're74', 're75', 'married', 'nodegr']
    X_raw = lalonde_data[covariate_cols].values
    X = np.column_stack([np.ones(len(X_raw)), X_raw])
    treat = lalonde_data['treat'].values.astype(float)
    return X, treat


@pytest.fixture(scope="session")
def lalonde_full(lalonde_data, lalonde_X_treat):
    """Complete LaLonde dataset bundle for testing."""
    X, treat = lalonde_X_treat
    sample_weights = np.ones(len(treat))
    return {
        'df': lalonde_data,
        'X': X,
        'treat': treat,
        'sample_weights': sample_weights,
        'n': len(treat),
        'k': X.shape[1],
        'n_t': int(np.sum(treat)),
        'n_c': int(np.sum(1 - treat)),
    }


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================

@pytest.fixture
def simple_binary_data():
    """Generate simple synthetic binary treatment data for unit tests."""
    import scipy.special
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    true_beta = np.array([0.0, 0.5, -0.3])
    theta = X @ true_beta
    pi_true = scipy.special.expit(theta)
    treat = np.random.binomial(1, pi_true).astype(float)
    while np.sum(treat) < 20 or np.sum(treat) > 80:
        treat = np.random.binomial(1, pi_true).astype(float)
    sample_weights = np.ones(n)
    return {
        'X': X, 'treat': treat, 'sample_weights': sample_weights,
        'true_beta': true_beta, 'n': n, 'k': X.shape[1],
        'n_t': int(np.sum(treat)), 'n_c': int(np.sum(1 - treat)),
    }


@pytest.fixture
def weighted_binary_data():
    """Generate synthetic data with non-uniform sample weights."""
    import scipy.special
    np.random.seed(123)
    n = 150
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])
    true_beta = np.array([0.0, 0.4, -0.2])
    theta = X @ true_beta
    pi_true = scipy.special.expit(theta)
    treat = np.random.binomial(1, pi_true).astype(float)
    raw_weights = np.random.exponential(1, n)
    sample_weights = raw_weights / np.mean(raw_weights)
    return {
        'X': X, 'treat': treat, 'sample_weights': sample_weights,
        'true_beta': true_beta, 'n': n, 'k': X.shape[1],
        'n_t': int(np.sum(treat)), 'n_c': int(np.sum(1 - treat)),
    }


# =============================================================================
# Matrix Test Fixtures
# =============================================================================

@pytest.fixture
def full_rank_matrix():
    """Generate a well-conditioned full-rank matrix for pseudoinverse tests."""
    np.random.seed(42)
    return np.random.randn(5, 3)


@pytest.fixture
def rank_deficient_matrix():
    """Generate a rank-deficient matrix for pseudoinverse tests."""
    np.random.seed(42)
    u = np.random.randn(5, 2)
    v = np.random.randn(2, 4)
    return u @ v


@pytest.fixture
def near_singular_matrix():
    """Generate a near-singular matrix for numerical stability tests."""
    np.random.seed(42)
    n = 5
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.array([1.0, 0.5, 0.1, 0.01, 1e-12])
    return U @ np.diag(s) @ V.T


@pytest.fixture
def zero_matrix():
    """Generate a zero matrix for edge case tests."""
    return np.zeros((3, 3))
