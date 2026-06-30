Implementation Notes
====================

This document provides important implementation details and technical considerations for the CBPS Python package.

Numerical Precision
-------------------

The package maintains high numerical precision across all estimation methods:

- **Binary/Multi-valued CBPS**: Coefficients and weights precise to ±1e-6
- **Continuous CBPS**: GPS estimation with correlation-based balance
- **High-dimensional matrices**: Robust SVD-based operations for ill-conditioned cases

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

The package automatically detects treatment type from the data:

- **Discrete treatment**: Integer arrays with ≤4 unique values
- **Continuous treatment**: Arrays with >4 unique values or floating-point type

**Example:**

.. code-block:: python

   import numpy as np
   from cbps import CBPS
   
   # Integer array with 3 unique values → Discrete treatment
   treat = np.array([0, 1, 2, 0, 1, 2] * 30)
   fit = CBPS(treatment=treat, covariates=X, att=1)
   
   # Floating-point array → Continuous treatment
   treat_cont = np.random.uniform(0, 10, 180)
   fit = CBPS(treatment=treat_cont, covariates=X, att=0)

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
