Advanced Usage Guide
====================

This guide covers advanced usage patterns and best practices for the CBPS Python package.

Quick Reference: API Overview
-----------------------------

Core Functions
~~~~~~~~~~~~~~

======================== ================================================
Function                 Description
======================== ================================================
``cbps.CBPS()``          Main estimator for binary/multi-valued/continuous treatments
``cbps.CBMSM()``         Marginal structural models for longitudinal data
``cbps.npCBPS()``        Nonparametric CBPS using empirical likelihood
``cbps.hdCBPS()``        High-dimensional CBPS with LASSO variable selection
``cbps.CBIV()``          Instrumental variable CBPS for noncompliance
``cbps.AsyVar()``        Asymptotic variance estimation
``cbps.balance()``       Covariate balance assessment
``cbps.vcov_outcome()``  Variance adjustment for outcome models
======================== ================================================

Key Parameters
~~~~~~~~~~~~~~

======================== ============================== =======================
Parameter                Description                    Example
======================== ============================== =======================
``att``                  Target estimand (0=ATE, 1=ATT) ``att=1``
``two_step``             Two-step GMM estimator         ``two_step=True``
``sample_weights``       Survey sampling weights        ``sample_weights=w``
``probs_min``            Minimum probability bound      ``probs_min=0.01``
``baseline_formula``     Baseline outcome model         ``baseline_formula="~age"``
``diff_formula``         Treatment effect model         ``diff_formula="~educ"``
``time_vary``            Time-varying parameters        ``time_vary=True``
``msm_variance``         MSM variance method            ``msm_variance="approx"``
======================== ============================== =======================

**Naming Convention**: All parameters use Python's snake_case convention.

Object Methods
~~~~~~~~~~~~~~

======================== ============================== ==============================
Method                   Description                    Usage
======================== ============================== ==============================
``fit.summary()``        Statistical summary            ``print(fit.summary())``
``fit.vcov()``           Variance-covariance matrix     ``vcov = fit.vcov()``
``print(fit)``           String representation          ``print(fit)``
``fit.predict()`` (*)    Propensity score prediction    ``ps = fit.predict(newdata=new_df)``
``fit.balance()``        Balance diagnostics            ``bal = fit.balance()``
``fit.plot()`` (**)      Diagnostic plots               ``fit.plot(kind='deviance')``
======================== ============================== ==============================

(*) :meth:`~cbps.core.results.CBPSResults.predict` takes a keyword argument
``newdata`` (default ``None``, which reuses the training sample) and
``type`` (``'response'`` or ``'link'``). Pass a ``pd.DataFrame`` with the
same columns as the fitted ``formula`` to score a new sample.

(**) ``kind`` currently supports only ``'deviance'``; other values raise
``ValueError``. Deviance plots are only defined for binary treatment.

Standalone Functions
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 45 40

   * - Function
     - Description
     - Usage
   * - ``cbps.balance(fit)``
     - Covariate balance assessment
     - ``bal = cbps.balance(fit)``
   * - ``cbps.AsyVar(Y, CBPS_obj=fit)``
     - Asymptotic variance / CI for the ATE
     - ``result = cbps.AsyVar(Y=Y, CBPS_obj=fit)``
   * - ``cbps.vcov_outcome(fit, Y, Z, delta)``
     - Adjusted vcov for weighted outcome regression
     - ``V = cbps.vcov_outcome(fit, Y, Z, delta)``
   * - ``cbps.plot_cbps(fit)``
     - Love plot (binary / multi-valued)
     - ``cbps.plot_cbps(fit)``
   * - ``cbps.plot_cbps_continuous(fit)``
     - Weighted-correlation plot (continuous)
     - ``cbps.plot_cbps_continuous(fit)``
   * - ``cbps.plot_cbmsm(fit)``
     - Balance plot for marginal structural models
     - ``cbps.plot_cbmsm(fit)``
   * - ``cbps.plot_npcbps(fit)``
     - Balance plot for nonparametric CBPS
     - ``cbps.plot_npcbps(fit)``

Data Structures
~~~~~~~~~~~~~~~

======================== ============================== =============================================
Type                     Description                    Notes
======================== ============================== =============================================
``pd.DataFrame``         Primary data input             Required for formula interface
``np.ndarray``           Array interface input          For direct array input
``pd.Categorical``       Categorical treatment          Required for multi-valued (3–4 level) treatment
======================== ============================== =============================================

Code Examples
-------------

Example 1: Binary Treatment CBPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import cbps
    from cbps.datasets import load_lalonde

    # Load LaLonde job training dataset
    df = load_lalonde(dehejia_wahba_only=True)

    # Estimate ATT using over-identified GMM
    fit = cbps.CBPS(
        formula='treat ~ age + educ + re74 + re75',
        data=df,
        att=1,           # ATT estimation
        method='over'    # Over-identified GMM
    )

    # View statistical summary
    print(fit.summary())

    # Assess covariate balance
    bal = cbps.balance(fit)
    print(bal['balanced'])

    # Visualize balance diagnostics (requires matplotlib)
    import matplotlib.pyplot as plt
    cbps.plot_cbps(fit)
    plt.show()

Example 2: Continuous Treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For continuous treatments, CBPS estimates the generalized propensity score (GPS)
by minimizing weighted correlations between the treatment and covariates.

.. code-block:: python

    import cbps
    from cbps.datasets import load_continuous_simulation

    # Load simulated continuous treatment data
    data, metadata = load_continuous_simulation(dgp=1)

    # Estimate generalized propensity score (GPS)
    fit = cbps.CBPS(
        formula='T ~ X1 + X2 + X3 + X4 + X5',
        data=data,
        att=0,           # ATE estimation (only valid estimand for continuous)
        method='over'
    )

    # Check balance via weighted correlations (target: |r| < 0.1)
    bal = cbps.balance(fit)
    print(bal['balanced'])     # Weighted correlations
    print(bal['unweighted'])   # Baseline correlations

Example 3: Marginal Structural Model (CBMSM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import cbps
    from cbps.datasets import load_blackwell

    # Load longitudinal data
    df = load_blackwell()

    # Estimate MSM weights
    fit = cbps.CBMSM(
        formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
        id='demName',
        time='time',
        data=df,
        type='MSM',
        time_vary=False
    )

    # Access weights
    print(f"Weight range: [{fit.weights.min():.2f}, {fit.weights.max():.2f}]")

Example 4: scikit-learn Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`cbps.sklearn.CBPSEstimator` wraps the binary / multi-valued
``CBPS()`` estimator as a sklearn-compatible classifier so it can sit inside
a :class:`~sklearn.pipeline.Pipeline` together with standard preprocessors.

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    from cbps.sklearn import CBPSEstimator
    from cbps.datasets import load_lalonde

    df = load_lalonde()
    covs = ['age', 'educ', 'black', 'hisp', 'married',
            'nodegr', 're74', 're75']
    X = df[covs].values
    T = df['treat'].values
    Y = df['re78'].values

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('cbps', CBPSEstimator(att=0, method='over')),
    ])
    pipe.fit(X, T)

    est = pipe.named_steps['cbps']
    print(f"Converged: {est.cbps_result_.converged}")

    # Retrieve IPW weights for a downstream weighted outcome regression
    w = est.get_weights()
    ate_model = LinearRegression().fit(T.reshape(-1, 1), Y, sample_weight=w)
    print(f"IPW-weighted ATE: {ate_model.coef_[0]:.2f}")

.. warning::

   :meth:`~cbps.sklearn.CBPSEstimator.predict_proba` and
   :meth:`~cbps.sklearn.CBPSEstimator.predict` only return the stored
   training-sample propensity scores; they raise :class:`ValueError` on
   arrays with a different sample count. As a consequence,
   :class:`~sklearn.model_selection.GridSearchCV` and
   :func:`~sklearn.model_selection.cross_val_score` with default scoring do
   **not** produce meaningful test-fold scores. For out-of-sample
   propensity-score prediction, use the formula-interface result's
   :meth:`cbps.core.results.CBPSResults.predict` method instead.

Numerical Considerations
------------------------

npCBPS: Non-Convex Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The npCBPS function uses empirical likelihood optimization with BFGS algorithm.
Due to the non-convex nature of the objective function (Fong et al. 2018, Section 3.3.2),
results depend on initialization and optimization parameters.

**Convergence verification**:

.. code-block:: python

    import cbps
    from cbps.datasets import load_lalonde
    
    df = load_lalonde(dehejia_wahba_only=True)
    fit = cbps.npCBPS('treat ~ age + educ + re74', data=df)
    
    # Check weight sum (should be close to 1.0)
    print(f"Weight sum: {fit.sumw0:.4f}")
    
    # Verify convergence (tolerance is 5% for npCBPS)
    if abs(fit.sumw0 - 1.0) > 0.05:
        print("Warning: Poor convergence")

hdCBPS: Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~

LASSO cross-validation uses NumPy random fold assignment. Set a random seed 
for reproducible results.

.. note::
   hdCBPS requires the ``glmnetforpython`` package. See the installation guide
   for setup instructions on Apple Silicon.

.. code-block:: python

    import cbps
    import numpy as np
    import pandas as pd
    
    # Set seed for reproducibility
    np.random.seed(12345)
    
    # Create sample data
    n, p = 200, 20
    X = np.random.randn(n, p)
    treat = np.random.binomial(1, 0.5, n)
    outcome = 0.5 * treat + X[:, 0] + np.random.randn(n)
    
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
    df['treat'] = treat
    df['outcome'] = outcome
    
    # Estimate hdCBPS (requires outcome variable)
    fit = cbps.hdCBPS(
        formula='treat ~ ' + ' + '.join([f'x{i}' for i in range(p)]),
        data=df,
        y='outcome',  # Outcome variable required for hdCBPS
        ATT=0         # 0 for ATE, 1 for ATT
    )
    
    print(f"ATE estimate: {fit.ATE:.3f} (SE: {fit.s:.3f})")

Common Usage Patterns
---------------------

Intercept Handling
~~~~~~~~~~~~~~~~~~

The formula interface adds an intercept automatically via patsy. The array
interface also adds an intercept automatically if the first column of
``covariates`` is not a constant vector. Supplying an intercept column
explicitly is equivalent:

.. code-block:: python

    import cbps
    import numpy as np
    from cbps.datasets import load_lalonde

    df = load_lalonde(dehejia_wahba_only=True)

    # Formula interface (automatic intercept via patsy)
    fit_formula = cbps.CBPS('treat ~ age + educ', data=df)

    # Array interface — intercept is inserted automatically
    treat = df['treat'].values
    X = df[['age', 'educ']].values
    fit_auto = cbps.CBPS(treatment=treat, covariates=X)

    # Array interface — intercept supplied explicitly
    X_with_intercept = np.column_stack([np.ones(len(treat)), X])
    fit_manual = cbps.CBPS(treatment=treat, covariates=X_with_intercept)

.. note::
   The auto-insertion only triggers when no constant column is detected. Pass
   ``verbose=1`` to see a ``UserWarning`` when an intercept is added on your
   behalf.

Treatment Type Detection
~~~~~~~~~~~~~~~~~~~~~~~~

Treatment type is automatically detected based on data characteristics:

- **Binary**: Numeric arrays with exactly 2 unique values in {0, 1}
- **Multi-valued**: ``pd.Categorical`` with 3-4 levels (explicit conversion required)
- **Continuous**: Any numeric array not matching binary criteria

.. note::
   For multi-valued treatments, explicit ``pd.Categorical`` conversion is required.
   Numeric arrays with 3-4 unique values will be treated as continuous (with a warning).

.. code-block:: python

    import cbps
    import numpy as np
    import pandas as pd
    
    # Prepare sample data
    n = 100
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])  # Intercept + 2 covariates
    
    # WARNING: Numeric 3-level array is treated as CONTINUOUS (not multi-valued)
    # This triggers a warning recommending explicit Categorical conversion
    treat = np.array([0, 1, 2] * 33 + [0])
    # fit = cbps.CBPS(treatment=treat, covariates=X, att=0)  # Would use continuous CBPS

    # CORRECT: Explicit Categorical for discrete multi-valued treatment
    df = pd.DataFrame({'treatment': pd.Categorical([0, 1, 2] * 33 + [0]),
                       'X1': np.random.randn(n),
                       'X2': np.random.randn(n)})
    fit = cbps.CBPS(formula='treatment ~ X1 + X2', data=df, att=0)

Best Practices
--------------

1. Use Formula Interface
~~~~~~~~~~~~~~~~~~~~~~~~

The formula interface is recommended for most use cases:

.. code-block:: python

    import cbps
    from cbps.datasets import load_lalonde
    
    df = load_lalonde(dehejia_wahba_only=True)
    
    # Recommended: formula interface
    fit = cbps.CBPS('treat ~ age + educ + I(re75==0)', data=df)

2. Set Random Seeds
~~~~~~~~~~~~~~~~~~~

For reproducibility with stochastic algorithms:

.. code-block:: python

    import numpy as np
    np.random.seed(12345)

3. Check Convergence
~~~~~~~~~~~~~~~~~~~~

Always verify optimization success:

.. code-block:: python

    import cbps
    from cbps.datasets import load_lalonde
    
    df = load_lalonde(dehejia_wahba_only=True)
    fit = cbps.CBPS('treat ~ age + educ', data=df)
    
    if not fit.converged:
        print("Warning: Optimization did not converge")
        print(f"J-statistic: {fit.J:.4f}")

4. Use Diagnostic Tools
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import cbps
    import matplotlib.pyplot as plt
    from cbps.datasets import load_lalonde
    
    df = load_lalonde(dehejia_wahba_only=True)
    fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)
    
    # Check balance
    bal = cbps.balance(fit)
    print(bal['balanced'])

    # Visualize results (requires matplotlib)
    cbps.plot_cbps(fit)
    plt.show()

    # Statistical summary
    print(fit.summary())

Architecture Overview
---------------------

Internal Module Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

The CBPS package uses a modular architecture where the main ``CBPS()`` function
routes to specialized implementations based on treatment type:

- ``cbps_binary_fit`` - Binary treatment estimation (0/1 treatment)
- ``cbps_continuous_fit`` - Continuous treatment estimation (GPS)
- ``cbps_3treat_fit``, ``cbps_4treat_fit`` - Multi-valued treatment estimation (3-4 levels)
- ``cbps_optimal_2treat`` - Optimal CBPS with dual balancing conditions

**Usage note**: Always call ``cbps.CBPS()`` directly; low-level functions are
for internal use only and their signatures may change without notice.

Additional Resources
--------------------

- **API Documentation**: See :doc:`api/index` for the autodoc-generated API reference
- **Tutorials**: See :doc:`tutorials/index` for the three replication tutorials
- **Theory**: See :doc:`theory` for mathematical background

Getting Help
------------

If you encounter issues:

1. Check the :doc:`quickstart` guide
2. Review the troubleshooting section in ``README.md``
3. Consult the :doc:`api/index` reference
4. Open an issue on GitHub with:

   - Code that reproduces the issue
   - Error message or unexpected output
   - Data sample (if possible)

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - Parameter naming
     - Snake_case convention (e.g., ``att``, ``two_step``)
   * - Object methods
     - ``fit.summary()``, ``fit.vcov()``, ``fit.predict()``
   * - Standalone functions
     - ``cbps.balance(fit)``, ``cbps.AsyVar(Y, CBPS_obj=fit)``,
       ``cbps.vcov_outcome(fit, Y, Z, delta)``,
       ``cbps.plot_cbps(fit)``, ``cbps.plot_cbps_continuous(fit)``,
       ``cbps.plot_cbmsm(fit)``, ``cbps.plot_npcbps(fit)``
   * - Data structures
     - ``pd.DataFrame``, ``np.ndarray``, ``pd.Categorical``
   * - Treatment detection
     - Automatic based on data characteristics