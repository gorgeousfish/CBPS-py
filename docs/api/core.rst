Core CBPS Functions
====================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

This module contains the core CBPS estimation functions for binary, multi-valued,
and continuous treatments. The implementation follows the generalized method of
moments (GMM) framework described in Imai & Ratkovic (2014), with extensions for
continuous treatments (Fong, Hazlett & Imai, 2018), optimal balancing conditions
(Fan et al., 2022), and multi-valued treatments (Imai & Ratkovic, 2014, Section 4).

Main Function
-------------

CBPS
~~~~

.. autofunction:: cbps.CBPS

The primary function for estimating covariate balancing propensity scores across diverse treatment scenarios. This unified interface automatically detects treatment types and applies the appropriate estimation method:

**Treatment Type Support**

- **Binary Treatments** (0/1): Logistic model-based CBPS for ATE and ATT estimation
- **Multi-valued Treatments** (3-4 levels): Multinomial logistic CBPS for categorical treatments
- **Continuous Treatments**: Generalized propensity scores via parametric models

**Key Features**

- Intelligent treatment type detection based on data characteristics
- Dual interface: Wilkinson-Rogers formula syntax and direct array input
- Flexible GMM estimation: exactly-identified and over-identified specifications
- Advanced optimization: two-step and continuous updating GMM estimators
- Robust numerical stability with comprehensive error handling

**Mathematical Framework**

The CBPS estimator solves the following GMM optimization problem:

.. math::
   \hat{\beta} = \arg\min_{\beta} \, g_n(\beta)' W_n g_n(\beta)

where the moment conditions combine treatment prediction and covariate balance:

.. math::
   g_n(\beta) = \begin{bmatrix}
      \frac{1}{n}\sum_{i=1}^n [T_i - e(X_i,\beta)] \\
      \frac{1}{n}\sum_{i=1}^n X_i[T_i - e(X_i,\beta)]
   \end{bmatrix}

For binary treatments with propensity score :math:`e(X_i,\beta) = \text{logit}^{-1}(X_i'\beta)`:

**Moment Condition Weights** (used internally for GMM estimation):

- **ATE**: :math:`w_i = \frac{T_i - e_i}{e_i(1-e_i)}`
- **ATT**: :math:`w_i = \frac{n}{n_1} \cdot \frac{T_i - e_i}{1-e_i}`

**Returned Weights** (``fit.weights`` for outcome analysis):

The ``weights`` attribute returns the absolute value of moment condition weights:

- **ATE**: :math:`w_i = \left|\frac{1}{e_i - 1 + T_i}\right| = \frac{T_i}{e_i} + \frac{1-T_i}{1-e_i}`
- **ATT**: :math:`w_i = \frac{n}{n_1} \cdot \frac{|T_i - e_i|}{1-e_i}`, yielding :math:`\frac{n}{n_1}` for treated and :math:`\frac{n}{n_1} \cdot \frac{e_i}{1-e_i}` for controls

.. note::
   When ``standardize=True`` (default), weights are further normalized so that
   each treatment group sums to 1. For unnormalized Horvitz-Thompson weights,
   set ``standardize=False``.

**Multi-valued Treatment Framework**

For :math:`J`-valued treatment :math:`T_i \in \{1, \ldots, J\}` with multinomial
logistic propensity scores :math:`\pi_j(X_i, \beta) = P(T_i = j \mid X_i)`, the
moment conditions generalize to (Imai & Ratkovic, 2014, Section 4):

.. math::

   g_n(\beta) = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^{J-1}
   \left[ \mathbf{1}(T_i = j) - \pi_j(X_i, \beta) \right] \tilde{X}_i

where :math:`\tilde{X}_i` is the covariate vector (possibly including an intercept).
The over-identified specification adds covariate balancing conditions for each
treatment level contrast.

**Continuous Treatment Framework**

For continuous treatment :math:`T_i` with generalized propensity score
:math:`f_\theta(T_i \mid X_i)`, the CBGPS moment conditions combine the score
condition for :math:`\sigma^2` and the covariate balancing conditions
(Fong et al., 2018, Eq. 2):

.. math::

   \mathbb{E}\left\{\mathbf{m}_\theta(T_i, X_i)\right\} = \mathbb{E}\begin{pmatrix}
   \frac{1}{\sigma^2}(T_i^* - X_i^{*\top}\beta)^2 - 1 \\[6pt]
   \sigma \exp\!\left[\frac{1}{2\sigma^2}(T_i^* - X_i^{*\top}\beta)^2
   - \frac{T_i^{*2}}{2}\right] T_i^* X_i^*
   \end{pmatrix} = 0

where :math:`T_i^*` and :math:`X_i^*` are the whitened (centered and scaled)
treatment and covariates, and :math:`\theta = (\beta, \sigma^2)`. The stabilized
weight for outcome analysis is:

.. math::

   w_i = \frac{f(T_i^*)}{f_\theta(T_i^* \mid X_i^*)}
       = \sigma \exp\!\left[\frac{1}{2\sigma^2}(T_i^* - X_i^{*\top}\beta)^2
         - \frac{T_i^{*2}}{2}\right]

**Optimal CBPS (oCBPS)**

When ``baseline_formula`` and ``diff_formula`` are specified, :func:`CBPS` uses the
optimal balancing conditions of Fan et al. (2022). The oCBPS estimating function
incorporates both baseline and differential outcome models to achieve the
semiparametric efficiency bound (Fan et al., 2022, Eq. 3.2–3.3):

.. math::

   g_{\text{opt}}(\beta) = \begin{pmatrix}
   s_\beta(T, X) \\[4pt]
   h_{\text{base}}(X) \cdot \left[\frac{T}{\pi_\beta(X)} - \frac{1-T}{1-\pi_\beta(X)}\right] \\[4pt]
   h_{\text{diff}}(X) \cdot \left[\frac{T}{\pi_\beta(X)} - \frac{1-T}{1-\pi_\beta(X)}\right]
   \end{pmatrix}

where :math:`s_\beta` is the propensity score likelihood score, :math:`h_{\text{base}}(X)`
is the baseline outcome model from ``baseline_formula``, and :math:`h_{\text{diff}}(X)` is
the treatment effect heterogeneity model from ``diff_formula``. Under correct specification
of either the propensity score or the outcome model, the resulting ATE estimator is
doubly robust (Fan et al., 2022, Theorem 3.1). When both models are correct, it attains
the semiparametric efficiency bound (Fan et al., 2022, Corollary 2.2).

**Examples**

*Binary Treatment with Formula Interface*

.. code-block:: python

   import cbps
   import pandas as pd

   # Load observational data
   from cbps.datasets import load_lalonde
   data = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS for ATT
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=data,
       att=1,                    # Target estimand
       method='over',            # Over-identified GMM
       two_step=True,           # Two-step estimator
       standardize=True         # Weight standardization
   )

   # Examine results
   print(fit.summary())

   # Extract components
   print(f"Coefficients: {fit.coefficients}")
   print(f"Max weight: {max(fit.weights):.3f}")
   print(f"J-statistic: {fit.J:.3f}")

*Continuous Treatment*

.. code-block:: python

   # Continuous treatment example
   fit_cont = cbps.CBPS(
       formula='dosage ~ baseline_score + age + sex',
       data=continuous_df,
       method='over',            # GMM identification
       two_step=False           # Continuous updating for precision
   )

   # Generalized propensity scores
   gps_values = fit_cont.fitted_values
   print(f"GPS range: [{gps_values.min():.3f}, {gps_values.max():.3f}]")

*Array Interface*

.. code-block:: python

   import numpy as np

   # Direct array input
   np.random.seed(42)
   treatment = np.array([0, 1, 0, 1] * 100)
   covariates = np.random.randn(400, 5)

   # Add intercept manually
   X_with_intercept = np.column_stack([np.ones(400), covariates])

   fit_array = cbps.CBPS(
       treatment=treatment,
       covariates=X_with_intercept,
       att=0                    # ATE estimation
   )

**References**

Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
*Journal of the Royal Statistical Society, Series B* 76(1), 243-263.
https://doi.org/10.1111/rssb.12027

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. *The Annals of Applied Statistics*,
12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
Optimal covariate balancing conditions in propensity score estimation.
*Journal of Business & Economic Statistics*, 41(1), 97-110.
https://doi.org/10.1080/07350015.2021.2002159

**Troubleshooting**

- **Optimization does not converge**: Increase ``iterations`` (default: 1000),
  try ``method='exact'`` instead of ``'over'``, or check for perfect separation
  or near-collinearity in covariates.
- **Extreme weights**: Enable ``standardize=True`` (default) to normalize weights.
  Check for positivity violations (propensity scores near 0 or 1).
- **Numerical instability**: The implementation uses SVD-based matrix operations
  for stability. If issues persist, check for constant or near-constant covariates.
- **Continuous treatment fails**: Ensure ``att=0`` (ATE only for continuous treatments).
  Check that the treatment variable has sufficient variation.

**See Also**

- :func:`cbps.balance` — Covariate balance assessment after CBPS estimation
- :func:`cbps.diagnostics.plots.plot_cbps` — Visual balance diagnostics for binary/multi-valued treatments
- :func:`cbps.diagnostics.plots.plot_cbps_continuous` — Visual balance diagnostics for continuous treatments
- :func:`cbps.AsyVar` — Asymptotic variance estimation for binary treatment ATE
- :func:`cbps.vcov_outcome` — Variance adjustment for continuous treatment outcome regression
- :func:`cbps.hdCBPS` — High-dimensional CBPS for settings with many covariates
- :func:`cbps.npCBPS` — Nonparametric CBPS via empirical likelihood
- :func:`cbps.CBMSM` — Marginal structural models for longitudinal data
- :func:`cbps.CBIV` — Instrumental variable extensions for noncompliance

Low-Level Fitting Routine
--------------------------

cbps_fit
~~~~~~~~

.. autofunction:: cbps.cbps_fit

.. deprecated:: 0.1.0
   Use :func:`cbps.CBPS` instead. ``cbps_fit`` is maintained for backward
   compatibility but may be removed in a future version.

.. note::
   This function is maintained for backward compatibility but is not recommended for direct use. The main :func:`cbps.CBPS` function handles all routing and parameter validation internally.

**Architecture Overview**

The CBPS implementation uses specialized subfunctions for different treatment types:

- ``cbps_binary_fit`` - Binary treatment estimation
- ``cbps_continuous_fit`` - Continuous treatment estimation
- ``cbps_3treat_fit``, ``cbps_4treat_fit`` - Multi-valued treatment estimation (3-4 levels)
- ``cbps_optimal_2treat`` - Optimal CBPS with dual balancing conditions

**Usage Recommendation**

The main :func:`cbps.CBPS` function is recommended for most use cases as it handles
treatment type detection, data validation, and SVD preprocessing automatically:

.. code-block:: python

   # Recommended approach
   fit = cbps.CBPS(formula='treat ~ x1 + x2', data=df)

   # Low-level array interface (for advanced users)
   # Note: cbps_fit returns a dictionary, not a CBPSResults object
   result = cbps.cbps_fit(treat, X, method='over', att=1)
   print(result['coefficients'])  # Access via dictionary keys
   print(result['converged'])     # Convergence status

Result Objects
---------------

CBPSResults
~~~~~~~~~~~

.. autoclass:: cbps.core.results.CBPSResults
   :members:
   :undoc-members:
   :show-inheritance:

The primary result class containing estimation outputs and diagnostic information.

**Core Attributes**

- ``coefficients`` (ndarray): Estimated propensity score parameters (k × 1)
- ``weights`` (ndarray): CBPS weights for causal effect estimation (n × 1)
- ``fitted_values`` (ndarray): Estimated propensity scores (n × 1)
- ``var`` (ndarray): Variance-covariance matrix (k × k); use ``np.sqrt(np.diag(var))`` for standard errors
- ``J`` (float): Hansen's J-statistic for over-identification test
- ``converged`` (bool): Optimization convergence status
- ``deviance`` (float): Model deviance at optimum

**Key Methods**

- :meth:`summary()` - Comprehensive statistical summary
- :meth:`vcov()` - Variance-covariance matrix
- :meth:`predict()` - Compute propensity scores for new data
- :meth:`balance()` - Covariate balance diagnostics

**Example Usage**

.. code-block:: python

   fit = cbps.CBPS('treat ~ age + educ + income', data=df)

   # Access fitted values
   coef = fit.coefficients
   weights = fit.weights
   prop_scores = fit.fitted_values

   # Statistical inference
   import numpy as np
   se = np.sqrt(np.diag(fit.var))  # Standard errors from variance-covariance matrix
   ci_lower = coef.ravel() - 1.96 * se
   ci_upper = coef.ravel() + 1.96 * se

   # Diagnostics
   if fit.converged:
       print(f"Converged: {fit.converged}")
       print(f"J-statistic: {fit.J:.3f}")

CBPSSummary
~~~~~~~~~~~

.. autoclass:: cbps.core.results.CBPSSummary
   :members:
   :undoc-members:
   :show-inheritance:

A formatted summary object containing coefficient tables, standard errors, and model diagnostics. Returned by the :meth:`CBPSResults.summary()` method.

**Example**

.. code-block:: python

   fit = cbps.CBPS('treat ~ x1 + x2', data=df)
   summary = fit.summary()

   # Access summary components
   coef_estimates = summary.coef       # Coefficient estimates
   standard_errors = summary.se        # Standard errors
   z_statistics = summary.zvalues      # z-statistics
   p_values = summary.pvalues          # Two-sided p-values

   # Full coefficient table (k x 4 matrix)
   # Columns: Estimate, Std.Error, z-value, Pr(>|z|)
   coef_table = summary.coefficients

   print(summary)  # Formatted output