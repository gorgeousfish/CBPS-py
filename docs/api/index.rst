API Reference
=============

.. currentmodule:: cbps

.. note::

   Current version: |version|. Requires Python ≥ 3.10.

Comprehensive API documentation for the CBPS Python package, providing complete coverage of all estimation functions, diagnostic tools, and inference methods for causal inference with covariate balancing propensity scores.

This package implements a unified framework for propensity score estimation across diverse treatment modalities, from binary assignments to continuous interventions, with specialized extensions for high-dimensional settings, longitudinal data, and nonparametric estimation.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   core
   highdim
   nonparametric
   diagnostics
   inference
   iv
   msm
   datasets
   config

Method Selection Guide
----------------------

.. list-table:: Choosing the Right CBPS Method
   :header-rows: 1
   :widths: 22 14 14 14 36

   * - Scenario
     - Treatment
     - Dimension
     - Function
     - Key Parameters
   * - Standard observational study
     - Binary
     - Low (p < n)
     - :func:`CBPS`
     - ``att=0/1``, ``method='over'``
   * - Categorical treatment (3–4 levels)
     - Multi-valued
     - Low
     - :func:`CBPS`
     - Auto-detected from data
   * - Dose-response estimation
     - Continuous
     - Low
     - :func:`CBPS`
     - ``att=0``, ``method='over'``
   * - Efficiency-optimal estimation
     - Binary
     - Low
     - :func:`CBPS`
     - ``baseline_formula``, ``diff_formula``
   * - Many potential confounders
     - Binary
     - High (p ≫ n)
     - :func:`hdCBPS`
     - ``y='outcome'`` (required)
   * - Unknown treatment mechanism
     - Binary/Continuous
     - Low
     - :func:`npCBPS`
     - ``corprior=None``
   * - Longitudinal/panel data
     - Time-varying binary
     - Low
     - :func:`CBMSM`
     - ``id``, ``time``, ``type='MSM'``
   * - Treatment noncompliance
     - Binary with IV
     - Low
     - :func:`CBIV`
     - ``Z`` (instrument), ``twosided``


Module Relationships
~~~~~~~~~~~~~~~~~~~~

The typical CBPS workflow follows this pattern:

1. **Estimation**: :func:`CBPS` / :func:`hdCBPS` / :func:`npCBPS` / :func:`CBMSM` / :func:`CBIV`
2. **Diagnostics**: :func:`balance` → :func:`~cbps.diagnostics.plots.plot_cbps` / :func:`~cbps.diagnostics.plots.plot_cbps_continuous`
3. **Inference**: :func:`AsyVar` (binary treatment) / :func:`vcov_outcome` (continuous treatment)

Core API Overview
-----------------

The CBPS package provides a comprehensive suite of functions organized by functionality:

**Primary Estimation Functions**

The main entry points for different causal inference scenarios:

- :func:`CBPS` — Universal estimator supporting binary, multi-valued, and continuous treatments
- :func:`CBMSM` — Marginal structural models for longitudinal data with time-varying confounding
- :func:`hdCBPS` — High-dimensional CBPS with automated variable selection
- :func:`npCBPS` — Nonparametric estimation using empirical likelihood
- :func:`CBIV` — Instrumental variable extensions for noncompliance scenarios

**Statistical Inference**

Tools for variance estimation and confidence interval computation:

- :func:`AsyVar` — Asymptotic variance estimation via sandwich variance estimator (binary treatment)
- :func:`vcov_outcome` — Variance adjustment for outcome models with estimated weights (continuous treatment)

**Diagnostic and Visualization**

Comprehensive tools for model assessment and validation:

- :func:`balance` — Covariate balance assessment with standardized differences
- :func:`~cbps.diagnostics.plots.plot_cbps` — Publication-ready diagnostic plots for binary/multi-valued treatments
- :func:`~cbps.diagnostics.plots.plot_cbps_continuous` — Specialized plots for continuous treatment CBPS
- :func:`~cbps.diagnostics.weights_diag.weight_diagnostics` — Weight quality diagnostics (ESS, extremes)
- :func:`~cbps.diagnostics.overlap.check_overlap` — Positivity (overlap) assumption check
- :func:`~cbps.diagnostics.normality.test_treatment_normality` — Normality test for continuous treatment
- :func:`~cbps.diagnostics.ocbps_conditions.verify_ocbps_conditions` — oCBPS condition verification

**Configuration and Utilities**

Package-level configuration and batch processing:

- :func:`~cbps.logging_config.set_verbosity` — Control package verbosity level
- :func:`fit_multiple` — Batch estimation across multiple datasets
- :class:`~cbps.constants.NumericalConfig` — Numerical stability configuration

Result Object Methods
---------------------

All fitted CBPS objects provide a consistent interface:

- :meth:`~cbps.core.results.CBPSResults.summary` — Comprehensive statistical summary with coefficient estimates, standard errors, and diagnostics
- :meth:`~cbps.core.results.CBPSResults.vcov` — Variance-covariance matrix of estimated parameters
- :meth:`~cbps.core.results.CBPSResults.balance` — Detailed balance statistics before and after weighting
- :meth:`~cbps.core.results.CBPSResults.plot` — Diagnostic visualizations when matplotlib is available
- :meth:`~cbps.core.results.CBPSResults.predict` — Compute propensity scores for new observations

Quick Reference
---------------

**Binary Treatment Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   # Load LaLonde experimental data
   df = load_lalonde(dehejia_wahba_only=True)

   # Estimate ATT with over-identified GMM
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=df,
       att=1,                    # Target estimand
       method='over',            # Over-identified GMM
       two_step=True,           # Two-step estimator
       standardize=True         # Weight standardization
   )

   # Extract results
   import numpy as np
   coefficients = fit.coefficients
   weights = fit.weights
   se = np.sqrt(np.diag(fit.var))  # Standard errors from variance-covariance matrix
   balance_stats = fit.balance()

**Continuous Treatment Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_continuous_simulation

   # Load simulation data (DGP1: correctly specified models)
   data, metadata = load_continuous_simulation(dgp=1)

   # Estimate generalized propensity score
   fit = cbps.CBPS(
       formula='T ~ X1 + X2 + X3 + X4 + X5',
       data=data,
       method='over'            # GMM identification
   )

   # GPS values and weights
   gps_values = fit.fitted_values
   weights = fit.weights

**High-Dimensional Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   import numpy as np

   # Load LaLonde data
   df = load_lalonde(dehejia_wahba_only=True)

   # Variable selection with hdCBPS
   # Note: hdCBPS requires an outcome variable for ATE estimation
   fit = cbps.hdCBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=df,
       y='re78',       # Outcome variable (required)
       ATT=0           # 0 for ATE, 1 for ATT
   )

   # Access estimates
   ate_estimate = fit.ATE
   ate_se = fit.s
   print(f"ATE: {ate_estimate:.2f} (SE: {ate_se:.2f})")

**Longitudinal Data Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_blackwell

   # Load Blackwell campaign advertising data
   df = load_blackwell()

   # Marginal structural model
   fit = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName',
       time='time',
       data=df,
       type='MSM',
       time_vary=False
   )

   # MSM weights for causal effect estimation
   msm_weights = fit.weights
   print(f"Units: {fit.n_units}, Periods: {fit.n_periods}")

Parameter Reference
-------------------

Common parameters across CBPS estimators:

**Data Specification**

- ``formula`` (str) — Wilkinson-Rogers formula specifying treatment and covariates (e.g., ``'treat ~ x1 + x2'``)
- ``data`` (DataFrame) — Dataset containing all variables
- ``treatment`` (array) — Treatment variable (array interface)
- ``covariates`` (DataFrame/array) — Covariate matrix (array interface)

**Estimation Options**

- ``att`` (int or str) — Target estimand: 0/‘ate’ for ATE, 1/‘att’ for ATT, 2/‘atc’ for ATC. The high-level ``CBPS()`` accepts integers only; the lower-level ``cbps_fit()`` also accepts strings. Note: :func:`hdCBPS` uses ``ATT`` (uppercase) for this parameter.
- ``method`` (str) — GMM identification: ``'exact'`` or ``'over'``
- ``two_step`` (bool) — Use two-step GMM (``True``) or continuous updating (``False``)
- ``standardize`` (bool) — Standardize weights to sum to 1 within each treatment group
- ``iterations`` (int) — Maximum optimization iterations (default: 1000)

**Advanced Options**

- ``sample_weights`` (array) — Survey sampling weights
- ``baseline_formula`` (str) — Formula for baseline outcome model (optimal CBPS)
- ``diff_formula`` (str) — Formula for treatment effect model (optimal CBPS)
- ``na_action`` (str) — Missing value handling: ``'warn'``, ``'fail'``, or ``'ignore'``
- ``verbose`` (int) — Verbosity level: 0 (silent), 1 (basic), 2 (detailed)

For detailed parameter descriptions and additional options, see the module-specific documentation pages.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
