Tutorials
=========

This section contains comprehensive step-by-step tutorials covering all major CBPS functionality.
Each tutorial is provided as an interactive Jupyter Notebook with complete code examples,
visualizations, and detailed explanations.

.. note::
   All tutorials are available as Jupyter Notebooks in the ``examples/`` directory.
   You can download and run them locally to experiment with the code.

Tutorial Overview
-----------------

We provide 4 comprehensive tutorials covering different aspects of CBPS:

1. **Binary Treatment CBPS** - Basic propensity score estimation for binary treatments
2. **Continuous Treatment CBPS** - Generalized propensity score for continuous treatments
3. **Marginal Structural Models** - Longitudinal data with time-varying treatments
4. **High-Dimensional CBPS** - LASSO variable selection for high-dimensional settings

Tutorial 1: Binary Treatment CBPS
----------------------------------

**File:** ``examples/tutorial_binary.ipynb``

**Dataset:** LaLonde (445 observations)

**Topics covered:**

- Data loading and exploration
- ATT vs. ATE estimation
- Method comparison (exact vs. over-identified GMM)
- Covariate balance assessment
- Weight distribution visualization
- Treatment effect estimation with AsyVar

**Learning outcomes:**

- Understand the difference between ATT and ATE
- Learn how to assess covariate balance
- Interpret J-statistics and convergence diagnostics
- Estimate treatment effects with proper variance adjustment

Tutorial 2: Continuous Treatment CBPS
--------------------------------------

**File:** ``examples/tutorial_continuous.ipynb``

**Dataset:** Political ads data (16,265 observations)

**Topics covered:**

- Continuous treatment variable exploration
- Generalized Propensity Score (GPS) estimation
- Covariate balance using correlations 
- GPS weight distribution and effective sample size
- Outcome regression with GPS weights
- Dose-response curve analysis

**Learning outcomes:**

- Understand GPS vs. binary propensity scores
- Learn balance metrics for continuous treatments
- Estimate dose-response relationships
- Use robust variance estimation for weighted regression

Tutorial 3: Marginal Structural Models (CBMSM)
-----------------------------------------------

**File:** ``examples/tutorial_msm.ipynb``

**Dataset:** Blackwell longitudinal data (570 observations, 5 time periods)

**Topics covered:**

- Longitudinal data structure exploration
- Time-varying vs. time-invariant treatment models
- MSM weight estimation with CBMSM
- Propensity score behavior across time periods
- MSM weights examination by period

**Learning outcomes:**

- Understand marginal structural models for longitudinal data
- Learn the difference between time-varying and time-invariant models

Tutorial 4: High-Dimensional CBPS (hdCBPS)
-------------------------------------------

**File:** ``examples/tutorial_hdcbps.ipynb``

**Dataset:** Simulated high-dimensional data (p=100, n=200)

**Topics covered:**

- High-dimensional data simulation (p >> n)
- Why standard CBPS fails in high dimensions
- LASSO variable selection with cross-validation
- Variable selection accuracy metrics (precision/recall/F1)
- Propensity score estimation and validation
- Balance assessment for selected variables
- Effective sample size calculation

**Learning outcomes:**

- Understand when to use hdCBPS (p ≈ n or p >> n)
- Learn LASSO variable selection for propensity scores
- Assess variable selection accuracy
- Ensure reproducibility with random seed control

Running the Tutorials
----------------------

To run the tutorials locally:

1. Install Jupyter:

   .. code-block:: bash

      pip install jupyter

2. Navigate to the examples directory:

   .. code-block:: bash

      cd CBPS_python/examples

3. Start Jupyter Notebook:

   .. code-block:: bash

      jupyter notebook

4. Open any tutorial file (e.g., ``tutorial_binary.ipynb``)

Additional Examples
-------------------

For focused examples demonstrating specific functionality, see the Python scripts
in the ``examples/`` directory. These include:

- ``cbps_basic.py`` - Basic CBPS usage
- ``cbmsm_basic.py`` - Marginal structural models
- ``npcbps_basic.py`` - Nonparametric CBPS
- ``hdcbps_basic.py`` - High-dimensional CBPS
- ``cbiv_basic.py`` - Instrumental variables
- ``balance_basic.py`` - Balance diagnostics
- ``plot_cbps.py`` - Visualization
- And more...

See the ``examples/README.md`` file for a complete list.

