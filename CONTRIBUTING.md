# Contributing to cbps

Thank you for your interest in contributing to the cbps Python package! We welcome contributions from everyone and are grateful for every contribution, whether it's a bug report, feature suggestion, documentation improvement, or code contribution.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## First-Time Contributors

New to open source? We're happy to help you get started! Here are some ways to make your first contribution:

- **Fix a typo** in the documentation or docstrings
- **Improve an example** or add a new one
- **Report a bug** you've encountered
- **Add a test case** for an existing function

Look for issues labeled [`good first issue`](https://github.com/gorgeousfish/cbps-python/labels/good%20first%20issue) for beginner-friendly tasks.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on our [GitHub Issues](https://github.com/gorgeousfish/cbps-python/issues) page with:

1. A clear, descriptive title
2. A detailed description of the problem
3. Steps to reproduce the issue
4. Expected behavior vs. actual behavior
5. Your environment (Python version, OS, package version)
6. If possible, a minimal code example that reproduces the issue

### Suggesting Features

We welcome feature suggestions! Please open an issue with:

1. A clear description of the feature
2. The motivation/use case for the feature
3. Any relevant references (papers, other implementations, etc.)

### Contributing Code

1. **Open an issue first**: Before submitting a pull request with new features or significant changes, please open an issue to discuss your proposed changes. This helps avoid duplicate work and ensures your contribution aligns with the project's direction.

2. **Fork the repository**: Create your own fork of the project.

3. **Create a feature branch**: Use a descriptive branch name.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**: Follow the code style guidelines below.

5. **Write tests**: All new code should include appropriate tests.

6. **Submit a pull request**: Include a clear description of the changes and reference any related issues.

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/gorgeousfish/cbps-python.git
cd cbps-python

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style Guidelines

### Formatting

We use automated tools to ensure consistent code style:

- **Black** for code formatting (88 characters per line)
- **isort** for import sorting (with black profile)
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black cbps/
isort cbps/

# Check code style
black --check cbps/
isort --check cbps/
flake8 cbps/
mypy cbps/
```

### Type Annotations

All functions must have complete type annotations. Use `# type: ignore` sparingly and only when necessary (e.g., for third-party libraries without type stubs).

### Documentation

- All public functions, classes, and methods must have docstrings
- Use NumPy-style docstrings
- Include type information in docstrings for complex types
- Add examples where appropriate

Example:

```python
def compute_weights(
    probs: np.ndarray,
    treatment: np.ndarray,
    estimand: str = "ATE"
) -> np.ndarray:
    """Compute inverse probability weights.

    Parameters
    ----------
    probs : np.ndarray
        Estimated propensity scores, shape (n,).
    treatment : np.ndarray
        Binary treatment indicator, shape (n,).
    estimand : str, optional
        Target estimand, either "ATE" or "ATT". Default is "ATE".

    Returns
    -------
    np.ndarray
        Computed weights, shape (n,).

    Examples
    --------
    >>> probs = np.array([0.3, 0.7, 0.5])
    >>> treatment = np.array([0, 1, 1])
    >>> weights = compute_weights(probs, treatment, estimand="ATE")
    """
```

## Numerical Precision Requirements

**Critical**: This package maintains high numerical precision (±1e-6) for all core algorithms. This is essential for reproducing results from the original R CBPS package.

### Key Constraints

1. **Float64 only**: All floating-point operations must use `numpy.float64`
   ```python
   X = np.array(data, dtype=np.float64)
   ```

2. **Generalized inverse**: Use `scipy.linalg.pinv(V, rcond=None)` for numerical stability
   ```python
   invV = scipy.linalg.pinv(V, rcond=None)
   ```

3. **GLM initialization**: Use `statsmodels.GLM` for propensity score estimation
   ```python
   glm_fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(tol=1e-8, maxiter=25)
   ```

4. **Probability clipping**: Use `np.clip(probs, 1e-6, 1-1e-6)`
   ```python
   probs = np.clip(probs, 1e-6, 1-1e-6)
   ```

5. **Sample weights normalization**: First step in all modules
   ```python
   sw = sw / sw.mean()  # Ensures sw.sum() = n
   ```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cbps --cov-report=html

# Run specific test file
pytest tests/test_cbps.py

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "r_benchmark"  # Only R benchmark tests
```

### Writing Tests

- All new code should have tests with coverage ≥90%
- Use `numpy.testing.assert_allclose(atol=1e-6, rtol=0)` for numerical comparisons
- Set random seeds for reproducibility: `np.random.seed(12345)`
- Use pytest markers appropriately:
  - `@pytest.mark.slow` for tests taking >10 seconds
  - `@pytest.mark.r_benchmark` for R comparison tests
  - `@pytest.mark.integration` for end-to-end tests

Example test:

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

def test_cbps_lalonde():
    """Test CBPS on LaLonde data."""
    from cbps.datasets import load_lalonde
    import cbps

    data = load_lalonde(dehejia_wahba_only=True)
    fit = cbps.CBPS(formula="treat ~ age + educ", data=data, att=1)

    # Verify convergence and basic properties
    assert fit.converged
    assert len(fit.coefficients) == 3  # intercept + 2 covariates

@pytest.mark.r_benchmark
def test_cbps_matches_r():
    """Test that Python results match R CBPS package."""
    # ... comparison with R results
    assert_allclose(python_coef, r_coef, atol=1e-6)
```

## Pull Request Process

1. **Ensure all checks pass**:
   ```bash
   black cbps/
   isort cbps/
   flake8 cbps/
   mypy cbps/
   pytest
   ```

2. **Update documentation** if you've changed APIs or added features.

3. **Add changelog entry** in `CHANGELOG.md` under "Unreleased" section.

4. **Commit with clear messages** following conventional commits:
   ```bash
   git commit -m "feat(cbps): add support for clustered standard errors"
   git commit -m "fix(multitreat): correct weight normalization"
   git commit -m "docs: update installation instructions"
   ```

5. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Requirements**:
   - All CI checks pass (GitHub Actions)
   - Code coverage does not decrease
   - No new linter warnings
   - Numerical tests pass (±1e-6 precision)
   - Documentation updated if needed

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/gorgeousfish/cbps-python/discussions) or Issue
- **Bug Reports**: Use [GitHub Issues](https://github.com/gorgeousfish/cbps-python/issues)
- **Documentation**: See [https://cbps-python.readthedocs.io](https://cbps-python.readthedocs.io)

## Maintainers

- **Cai Xuanyu** - xuanyuCAI@outlook.com
- **Xu Wenli** - wlxu@cityu.edu.mo

## Attribution

Contributors will be acknowledged in the project's documentation. We use the all-contributors specification to recognize all types of contributions.

Thank you for contributing to cbps!
