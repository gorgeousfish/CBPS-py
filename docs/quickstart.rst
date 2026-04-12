Quick Start Guide
=================

This guide provides an introduction to the CBPS package for Python.
For detailed tutorials, see :doc:`tutorials/index`.

5-Minute Quick Start
--------------------

A minimal working example:

.. code-block:: python

   from cbps import CBPS, balance
   from cbps.datasets import load_lalonde

   # Load the LaLonde job training dataset
   data = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS weights for ATT (Average Treatment Effect on the Treated)
   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1
   )

   # Assess covariate balance
   bal = balance(fit)
   print(bal)

   # Extract weights for downstream analysis
   weights = fit.weights
   print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

Basic Usage Patterns
--------------------

Formula Interface (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The formula interface accepts patsy-style formulas with pandas DataFrames:

.. code-block:: python

   from cbps import CBPS
   import pandas as pd
   import numpy as np

   # Prepare data as DataFrame
   np.random.seed(42)
   data = pd.DataFrame({
       'treat': np.random.binomial(1, 0.4, 200),
       'age': np.random.normal(45, 12, 200),
       'education': np.random.normal(12, 3, 200),
       'income': np.random.lognormal(10, 0.5, 200)
   })

   # Estimate CBPS
   fit = CBPS(
       formula='treat ~ age + education + income',
       data=data,
       att=1  # ATT estimation (use att=0 for ATE)
   )

Array Interface
^^^^^^^^^^^^^^^

NumPy arrays can be used directly:

.. code-block:: python

   import numpy as np
   from cbps import CBPS

   # Prepare data as arrays
   np.random.seed(42)
   n = 200
   treat = np.random.binomial(1, 0.4, n)
   X = np.column_stack([
       np.random.normal(45, 12, n),    # age
       np.random.normal(12, 3, n)      # education
   ])

   # Add intercept column for array interface
   X_with_intercept = np.column_stack([np.ones(n), X])

   # Estimate CBPS
   fit = CBPS(
       treatment=treat,
       covariates=X_with_intercept,
       att=1
   )

Common Tasks
------------

Estimate ATT (Average Treatment Effect on the Treated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cbps import CBPS
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)

   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1  # ATT
   )

   print(f"Converged: {fit.converged}")
   print(f"J-statistic: {fit.J:.4f}")

Estimate ATE (Average Treatment Effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cbps import CBPS
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)

   fit_ate = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=0  # ATE
   )

   print(f"Converged: {fit_ate.converged}")

Assess Covariate Balance
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cbps import CBPS, balance
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)
   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1
   )

   bal = balance(fit)
   print(bal)

   # Interpretation guidelines:
   # SMD < 0.1: Excellent balance
   # SMD < 0.2: Acceptable balance
   # SMD > 0.2: Poor balance, consider model respecification

Visualize Results
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cbps import CBPS, plot_cbps
   from cbps.datasets import load_lalonde
   import matplotlib.pyplot as plt

   data = load_lalonde(dehejia_wahba_only=True)
   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1
   )

   # Create balance diagnostic plot
   plot_cbps(fit)
   plt.tight_layout()
   plt.show()

   # To retrieve balance data as DataFrame, use silent=False
   balance_df = plot_cbps(fit, silent=False)
   print(balance_df.head())

Statistical Summary
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cbps import CBPS
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)
   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1
   )

   summary = fit.summary()
   print(summary)

Extract Weights for Outcome Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import statsmodels.api as sm
   from cbps import CBPS
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)
   fit = CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=data,
       att=1
   )

   # Retrieve CBPS weights
   weights = fit.weights

   # Weighted least squares regression for outcome analysis
   X_outcome = sm.add_constant(data[['treat', 'age', 'educ']])
   y_outcome = data['re78']

   wls_model = sm.WLS(y_outcome, X_outcome, weights=weights)
   result = wls_model.fit()
   print(result.summary())

Advanced Usage
--------------

Continuous Treatment (Generalized Propensity Score)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For continuous treatment variables, CBPS estimates generalized propensity scores
using a linear model with covariate balance constraints:

.. code-block:: python

   from cbps import CBPS, balance
   from cbps.datasets import load_continuous_simulation

   # Load simulated continuous treatment data
   data, metadata = load_continuous_simulation(dgp=1)

   # Estimate GPS for continuous treatment
   # Treatment column is 'T', covariates are 'X1' through 'X10'
   fit = CBPS(
       formula='T ~ X1 + X2 + X3 + X4 + X5',
       data=data,
       att=0,            # ATE (only valid estimand for continuous treatments)
       method='over'
   )

   # Assess balance via weighted correlations
   bal = balance(fit)
   print(bal)
   # Target: |correlation| < 0.1 indicates good balance

Marginal Structural Models (Longitudinal Data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For panel data with time-varying treatments, CBMSM estimates stabilized
inverse probability weights:

.. code-block:: python

   from cbps import CBMSM
   from cbps.datasets import load_blackwell

   # Load Blackwell political campaign data
   data = load_blackwell()

   # Estimate MSM weights
   fit = CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName',
       time='time',
       data=data,
       type='MSM',
       time_vary=False
   )

   # Retrieve unit-level weights
   msm_weights = fit.weights
   print(f"Weight range: [{msm_weights.min():.2f}, {msm_weights.max():.2f}]")

High-Dimensional CBPS (LASSO Regularization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For settings where the number of covariates is large relative to sample size,
hdCBPS combines CBPS with LASSO penalization:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from cbps import hdCBPS

   # Generate high-dimensional data
   np.random.seed(12345)
   n, p = 200, 50
   X = np.random.randn(n, p)
   treat = np.random.binomial(1, 0.5, n)
   outcome = 0.5 * treat + X[:, 0] + np.random.randn(n)

   # Construct DataFrame
   df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
   df['treat'] = treat
   df['outcome'] = outcome

   # Estimate hdCBPS (requires outcome variable)
   fit = hdCBPS(
       formula='treat ~ ' + ' + '.join([f'x{i}' for i in range(p)]),
       data=df,
       y='outcome',
       ATT=0  # 0 for ATE, 1 for ATT
   )

   print(f"ATE estimate: {fit.ATE:.3f}")
   print(f"Standard error: {fit.s:.3f}")

.. note::

   hdCBPS requires the ``glmnetforpython`` package. On Apple Silicon (M1/M2/M3),
   install from source with a Fortran compiler (see Installation guide).

Next Steps
----------

- **Tutorials**: See :doc:`tutorials/index` for comprehensive guides
- **API Reference**: See :doc:`api/index` for detailed documentation
- **Theory**: See :doc:`theory` for mathematical background
- **Technical Notes**: See :doc:`implementation_notes` for implementation details

