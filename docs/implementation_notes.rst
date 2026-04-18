Implementation Notes
====================

This document provides important implementation details and technical considerations for the CBPS Python package.

Numerical Precision
-------------------

The package is validated against the reference R ``CBPS`` implementation
(v0.23). Agreement tolerances observed in the test suite are:

- **Binary / Multi-valued CBPS**: coefficients and weights within ±1e-6;
  Hansen's J-statistic within ±1e-4; asymptotic standard errors within
  ±1e-5.
- **Continuous CBPS**: coefficients and weights within ±1e-6; weighted
  correlations with covariates and ATE estimates within ±1e-4.
- **CBMSM / CBIV / npCBPS / hdCBPS**: weights and point estimates
  validated within ±1e-6 (or the tightest tolerance the underlying
  optimizer supports; see individual test files under ``tests/``).
- **High-dimensional / ill-conditioned designs**: the package falls back
  to SVD-based solves and pinv regularisation when the design matrix is
  near-singular, to keep the reported agreement bands stable.

npCBPS: Empirical Likelihood Optimization
------------------------------------------

The ``npCBPS`` function uses empirical likelihood optimization, which involves a **non-convex objective function**.

**Key Characteristics:**

- Optimizer: ``scipy.optimize.minimize(method='BFGS')``
- Multiple local optima may exist
- Solution quality verified by weight sum constraint

**Convergence Verification:**

.. code-block:: python

   from cbps import npCBPS
   
   fit = npCBPS('treat ~ age + educ', data=df)
   
   # Check convergence diagnostics
   print(f"Weight sum: {fit.sumw0:.4f}")  # Should be ≈ 1.0
   print(f"Deviation: {abs(fit.sumw0 - 1.0):.4f}")  # Should be < 0.05
   print(f"Converged: {fit.converged}")

hdCBPS: Cross-Validation for Variable Selection
-------------------------------------------------

The ``hdCBPS`` function uses cross-validation for LASSO variable selection. For reproducible results, set a random seed before calling the function.

**Key Characteristics:**

- Uses ``glmnetforpython`` for LASSO regularization
- Cross-validation fold assignment depends on NumPy random state
- Set ``np.random.seed()`` for reproducible results

**Example:**

.. code-block:: python

   import numpy as np
   from cbps import hdCBPS
   
   # Set seed for reproducibility
   np.random.seed(12345)
   fit = hdCBPS(formula='treat ~ x1 + x2 + ... + x100', data=data, y='outcome', ATT=0)

Automatic Treatment Type Detection
----------------------------------

The package detects treatment type from the data but applies different rules
to binary versus multi-valued cases:

- **Binary treatment**: Arrays with exactly two unique values that are a subset
  of :math:`\{0, 1\}` (integer, boolean, or floating-point) are auto-detected
  and routed to the binary backend. Float 0/1 arrays are accepted but trigger a
  ``UserWarning`` recommending explicit ``int`` or ``pd.Categorical``
  representation.
- **Multi-valued treatment (3–4 levels)**: Must be an explicit
  ``pd.Categorical`` (or categorical dtype in a DataFrame). Numeric arrays
  with 3–4 unique values are **not** auto-detected as categorical; they are
  treated as continuous and a warning is issued recommending explicit
  conversion.
- **Continuous treatment**: Any numeric array not matching the binary
  criterion above (including floating-point arrays and integer arrays with
  more than two distinct values that are not wrapped in ``pd.Categorical``).

**Examples:**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from cbps import CBPS

   # Auto-detected as binary
   treat_binary = np.array([0, 1, 0, 1, 1, 0] * 30)
   fit_bin = CBPS(treatment=treat_binary, covariates=X, att=1)

   # Multi-valued: MUST be pd.Categorical; prefer the formula interface
   df = pd.DataFrame({'treat': pd.Categorical([0, 1, 2] * 60),
                      'x1': np.random.randn(180),
                      'x2': np.random.randn(180)})
   fit_multi = CBPS(formula='treat ~ x1 + x2', data=df, att=0)

   # Continuous treatment (float array)
   treat_cont = np.random.uniform(0, 10, 180)
   fit_cont = CBPS(treatment=treat_cont, covariates=X, att=0)

CBMSM: Time-Invariant Parameters
--------------------------------

For marginal structural models with ``time_vary=False``, the coefficient vector is correctly
replicated across all time periods, ensuring propensity scores vary appropriately.

**Verification:**

.. code-block:: python

   from cbps import CBMSM
   
   fit = CBMSM(
       formula='treat ~ treat_lag + covariate',
       id='id', time='time', data=df,
       time_vary=False  # Time-invariant parameters
   )
   
   # Propensity scores should vary across periods
   print(f"Weight variation: {fit.weights.std():.4f}")

Function Summary
----------------

+-------------------+------------------------------------------+
| Function          | Key Implementation Notes                 |
+===================+==========================================+
| CBPS (binary)     | Logistic GMM with balance conditions     |
+-------------------+------------------------------------------+
| CBPS (continuous) | GPS with correlation-based balance       |
+-------------------+------------------------------------------+
| CBMSM             | Longitudinal MSM weights                 |
+-------------------+------------------------------------------+
| npCBPS            | Non-convex empirical likelihood          |
+-------------------+------------------------------------------+
| hdCBPS            | LASSO CV with numpy random seed control  |
+-------------------+------------------------------------------+
| CBIV              | Instrumental variable compliance weights |
+-------------------+------------------------------------------+

Quality Assurance
-----------------

All implementations are:

1. **Tested** with comprehensive unit and integration tests
2. **Documented** with theoretical references
3. **Validated** against published research results

For questions or issues, please open an issue on GitHub.
