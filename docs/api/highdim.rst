High-Dimensional CBPS (hdCBPS)
===============================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

hdCBPS
------

.. autofunction:: cbps.hdCBPS

High-dimensional Covariate Balancing Propensity Score estimation for robust causal
inference when the number of covariates may exceed the sample size.

**Key Features**:

- Handles high-dimensional settings where :math:`p \gg n`
- LASSO-based variable selection with cross-validated regularization
- Double robustness: consistent when either propensity score or outcome model is correct
- Semiparametric efficiency under correct model specification
- Sample boundedness: ATE estimates lie within the range of observed outcomes

**When to Use hdCBPS**:

- High-dimensional covariate space (potentially more covariates than observations)
- Settings with many irrelevant or weakly relevant confounders
- Requirement for automatic variable selection in treatment effect estimation
- Sparse propensity score and/or outcome models

**Requirements**:

hdCBPS requires ``glmnetforpython`` for the LASSO variable selection step.

**Installation**

.. code-block:: bash

   # Option 1: From source (recommended for Apple Silicon M1/M2/M3)
   brew install gcc  # Install gfortran compiler
   git clone https://github.com/thierrymoudiki/glmnetforpython.git
   cd glmnetforpython
   pip install -e .
   
   # Option 2: Standard pip installation
   pip install glmnetforpython

**Note**: On Apple Silicon, source installation is recommended. If you encounter
``not a mach-o file`` errors, ensure gfortran is installed via Homebrew.

**Mathematical Framework**:

The hdCBPS algorithm follows the methodology of Ning, Peng, and Imai (2020):

**Covariate Balancing Properties** (Ning et al., 2020, Definition 1):

The hdCBPS methodology distinguishes two types of covariate balancing:

- **Strong covariate balancing**: :math:`\sum_{i=1}^n (T_i/\hat{\pi}_i - 1) X_i = 0`,
  which balances the mean of every component of :math:`X_i`. This is infeasible when
  :math:`d > n`.
- **Weak covariate balancing**: :math:`\sum_{i=1}^n (T_i/\hat{\pi}_i - 1) \alpha^{*\top} X_i = 0`,
  which only requires balancing the linear combination of covariates that predicts the
  outcome. This is sufficient for removing bias from propensity score estimation.

The hdCBPS algorithm achieves the weak covariate balancing property by calibrating
propensity scores using outcome-predictive covariates selected via LASSO.

**Step 1** (Propensity Score LASSO): Obtain initial propensity score estimates via
penalized M-estimation:

.. math::

   \hat{\beta} = \arg\min_{\beta} \left\{ -Q_n(\beta) + \lambda \|\beta\|_1 \right\}

where :math:`Q_n(\beta)` is the generalized quasi-likelihood function.

**Step 2** (Outcome Model LASSO): Fit penalized outcome models to identify relevant
predictors:

.. math::

   \tilde{\alpha} = \arg\min_{\alpha} \left\{ L_n(\alpha) + \lambda' \|\alpha\|_1 \right\}

**Step 3** (Covariate Balancing): Calibrate propensity scores by minimizing the GMM
objective to balance selected covariates:

.. math::

   \tilde{\gamma} = \arg\min_{\gamma} \|g_n(\gamma)\|_2^2

where :math:`g_n(\gamma) = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{T_i}{\pi(\gamma^T X_{i\tilde{S}} + \hat{\beta}_{\tilde{S}^c}^T X_{i\tilde{S}^c})} - 1 \right) X_{i\tilde{S}}`.

**Step 4** (Treatment Effect Estimation): Compute the Horvitz-Thompson estimator:

.. math::

   \hat{\mu}_1 = \frac{1}{n} \sum_{i=1}^{n} \frac{T_i Y_i}{\tilde{\pi}_i}

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   import numpy as np
   
   # Load LaLonde experimental data
   df = load_lalonde(dehejia_wahba_only=True)
   
   # Create high-dimensional scenario by adding noise covariates
   np.random.seed(12345)
   n = len(df)
   p_noise = 30
   noise_cols = {f'noise_{i}': np.random.randn(n) for i in range(p_noise)}
   df_hd = df.assign(**noise_cols)
   
   # Build formula with all covariates
   base_vars = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75']
   noise_vars = [f'noise_{i}' for i in range(p_noise)]
   all_vars = base_vars + noise_vars
   formula = 'treat ~ ' + ' + '.join(all_vars)
   
   # Estimate hdCBPS
   fit = cbps.hdCBPS(
       formula=formula,
       data=df_hd,
       y='re78',   # Outcome variable (required)
       ATT=0       # 0 for ATE, 1 for ATT
   )
   
   # Access results
   print(f"ATE estimate: {fit.ATE:.2f}")
   print(f"Standard error: {fit.s:.2f}")
   print(f"Converged: {fit.converged}")

**Theoretical Properties**

*Asymptotic Normality* (Ning et al., 2020, Theorem 1): Under regularity conditions,
the hdCBPS estimator :math:`\hat{\mu}_1` satisfies:

.. math::

   \sqrt{n}(\hat{\mu}_1 - \mu_1^*) \xrightarrow{d} N(0, V)

where :math:`V` attains the semiparametric efficiency bound when the propensity
score model is correctly specified and the outcome model is linear in covariates.

*Double Robustness* (Ning et al., 2020, Propositions 1–2): The hdCBPS estimator
remains :math:`\sqrt{n}`-consistent and asymptotically normal under either of the
following conditions:

- The propensity score model is correctly specified (even if the outcome model is wrong)
- The outcome model is correctly specified (even if the propensity score model is wrong)

This double robustness property provides valid confidence intervals in both cases,
a key advantage over methods that require both models to be correct.

**Variance Estimation**:

Standard errors for ATE are computed by extending the sandwich variance estimator
(Equation 11 in Ning et al., 2020) to account for both treatment groups:

.. math::

   \hat{V}_{\text{ATE}} = \frac{1}{n} \left[ \sum_{i: T_i=1} \frac{(Y_i - \tilde{\alpha}_1^T X_i)^2}{\tilde{\pi}_i^2} + \sum_{i: T_i=0} \frac{(Y_i - \tilde{\alpha}_0^T X_i)^2}{(1-\tilde{\pi}_i)^2} + \sum_{i=1}^{n} (\tilde{\alpha}_1^T X_i - \tilde{\alpha}_0^T X_i - \hat{\mu})^2 \right]

where :math:`\tilde{\alpha}_1` and :math:`\tilde{\alpha}_0` are the LASSO outcome model
coefficients for the treated and control groups respectively.

Confidence intervals are valid under correct specification of either the propensity
score model or the outcome model (Corollary 1 in the paper).

**See Also**:

- :func:`cbps.CBPS` — Standard CBPS for low-dimensional settings
- :func:`cbps.npCBPS` — Nonparametric CBPS via empirical likelihood
- :func:`cbps.AsyVar` — Asymptotic variance estimation for treatment effects
- :func:`cbps.balance` — Covariate balance assessment

Result Object
-------------

HDCBPSResults
~~~~~~~~~~~~~

.. autoclass:: cbps.highdim.hdcbps.HDCBPSResults
   :members:
   :undoc-members:
   :show-inheritance:

The result object returned by ``hdCBPS()``.

**Treatment Effect Estimates**:

- ``ATE``: Average Treatment Effect estimate (Horvitz-Thompson estimator)
- ``ATT``: Average Treatment Effect on the Treated (if ``ATT=1`` was specified)
- ``s``: Standard error of ATE (sandwich estimator)
- ``w``: Standard error of ATT (if ATT was requested)

**Propensity Score Estimates**:

- ``fitted_values``: Calibrated propensity scores, shape ``(n,)``
- ``weights``: Inverse probability weights for outcome modeling, shape ``(n,)``
- ``coefficients0``: Calibrated propensity score coefficients (control optimization)
- ``coefficients1``: Calibrated propensity score coefficients (treatment optimization)

**Variable Selection Information**:

- ``selected_indices_treat``: Indices of variables selected by LASSO for the treated outcome model
- ``selected_indices_control``: Indices of variables selected by LASSO for the control outcome model
- ``n_selected_treat``: Number of variables selected for the treated outcome model
- ``n_selected_control``: Number of variables selected for the control outcome model

**Convergence Diagnostics**:

- ``converged``: Whether the GMM optimization converged within tolerance
- ``iterations_used``: Dictionary containing iteration counts for each optimization

**Example**

.. code-block:: python

   fit = cbps.hdCBPS(
       formula='treat ~ age + educ + re74 + re75',
       data=df,
       y='re78',
       ATT=1
   )
   
   # Treatment effect estimates with confidence intervals
   print(f"ATE: {fit.ATE:.2f} (SE: {fit.s:.2f})")
   print(f"95% CI: [{fit.ATE - 1.96*fit.s:.2f}, {fit.ATE + 1.96*fit.s:.2f}]")
   
   if fit.ATT is not None:
       print(f"ATT: {fit.ATT:.2f} (SE: {fit.w:.2f})")
   
   # Variable selection summary
   print(f"Variables selected (treated): {fit.n_selected_treat}")
   print(f"Variables selected (control): {fit.n_selected_control}")
   
   # Use weights for downstream analysis
   weights = fit.weights

**Troubleshooting**:

If you encounter errors:

1. **"glmnetforpython not found"**: Install from source (see Installation section)

2. **"not a mach-o file" (Apple Silicon)**: Install gfortran via Homebrew:

   .. code-block:: bash

      brew install gcc
      pip install glmnetforpython

3. **Convergence issues**: Increase ``iterations`` parameter or check data quality

References
----------

Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score.
*Biometrika*, 107(3), 533-554. https://doi.org/10.1093/biomet/asaa020

