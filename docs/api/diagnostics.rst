Diagnostic Tools
================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

This module provides tools for assessing covariate balance after propensity score
estimation. Effective covariate balance is the primary diagnostic criterion for
evaluating CBPS performance.

balance
-------

.. autofunction:: cbps.balance

Compute covariate balance statistics before and after CBPS weighting.

**Key Features**:

- Standardized mean differences (SMD) for binary and multi-valued treatments
- Weighted Pearson correlations for continuous treatments (CBGPS)
- Comparison of weighted versus unweighted (baseline) statistics
- Automatic method selection based on treatment type

**When to Use**:

- Assess covariate balance after CBPS estimation
- Compare balance improvement from inverse probability weighting
- Evaluate propensity score model adequacy
- Verify that CBPS optimization achieved its balancing objective

**Balance Metrics**:

1. **Binary and multi-valued treatments**: Absolute standardized mean difference (ASMD)

   The ASMD measures the difference in covariate means between treatment groups,
   standardized by the sample standard deviation:

   .. math::

      \text{ASMD} = \frac{|\bar{X}_1 - \bar{X}_0|}{s}

   where :math:`s` is the standard deviation computed across all observations (full sample).

2. **Continuous treatments**: Pearson correlation coefficient

   For continuous treatments, balance is assessed via the weighted correlation between
   the treatment and each covariate. Under perfect balance, these correlations equal zero:

   .. math::

      \rho_w(T, X) = \frac{\sum_i w_i (T_i - \bar{T}_w)(X_i - \bar{X}_w)}{\sqrt{\sum_i w_i (T_i - \bar{T}_w)^2} \sqrt{\sum_i w_i (X_i - \bar{X}_w)^2}}

**Example (Binary Treatment)**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   # Load the LaLonde experimental dataset
   df = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS with ATT weighting
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Compute balance statistics
   bal = cbps.balance(fit)
   print(bal['balanced'])   # Weighted balance (after CBPS)
   print(bal['original'])   # Unweighted balance (before CBPS)

   # Output structure:
   # - Columns: treatment group means and standardized means
   # - Rows: covariates (excluding intercept)

**Example (Continuous Treatment)**

.. code-block:: python

   import cbps
   import pandas as pd
   import numpy as np

   # Generate synthetic continuous treatment data
   np.random.seed(123)
   n = 500
   df = pd.DataFrame({
       'dosage': np.random.uniform(0, 100, n),
       'baseline_score': np.random.normal(50, 15, n),
       'age': np.random.normal(45, 12, n)
   })

   # Estimate CBGPS (continuous treatment requires att=0)
   fit = cbps.CBPS('dosage ~ baseline_score + age', data=df, att=0)

   # Compute correlation-based balance
   bal = cbps.balance(fit)
   print(bal['balanced'])     # Weighted correlations (target: near 0)
   print(bal['unweighted'])   # Unweighted correlations (baseline)

   # Interpretation:
   # - Weighted correlations near 0 indicate successful balance
   # - Large reduction from unweighted to weighted indicates CBPS effectiveness

**Interpreting Balance Statistics**:

For binary and multi-valued treatments (SMD):

- **ASMD < 0.1**: Excellent balance (widely accepted threshold)
- **ASMD < 0.25**: Acceptable balance
- **ASMD > 0.25**: Poor balance; consider model re-specification

For continuous treatments (correlation):

- **|r| < 0.1**: Good balance
- **|r| ≈ 0**: Excellent balance
- **Large |r|**: Poor balance; consider alternative specifications

**See Also**:

- :func:`cbps.plot_cbps` - Visual balance diagnostics
- :func:`cbps.CBPS` - Primary CBPS estimation function

plot_cbps
---------

.. autofunction:: cbps.diagnostics.plots.plot_cbps

Visualize covariate balance for binary and multi-valued treatments.

**Key Features**:

- Scatter plots showing absolute standardized mean differences (ASMD) per contrast
- Box plots summarizing ASMD distributions across covariates
- Before vs. after weighting comparison (two-panel layout)
- Customizable matplotlib styling via keyword arguments

**Plot Types**:

1. **Scatter plot** (default): Each point represents one covariate's ASMD for a given contrast
2. **Box plot** (``boxplot=True``): Summary statistics of ASMD across all covariates

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   import matplotlib.pyplot as plt

   # Load data and estimate CBPS
   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Create default scatter plot
   cbps.plot_cbps(fit)
   plt.show()

   # Create boxplot visualization
   cbps.plot_cbps(fit, boxplot=True)
   plt.show()

   # Return balance data as DataFrame
   balance_df = cbps.plot_cbps(fit, silent=False)
   print(balance_df.head())

plot_cbps_continuous
--------------------

.. autofunction:: cbps.diagnostics.plots.plot_cbps_continuous

Visualize covariate balance for continuous treatments using Pearson correlations.

**Key Features**:

- Scatter or box plots of absolute treatment-covariate correlations
- Comparison of unweighted (baseline) vs. CBGPS-weighted correlations
- Visual verification that weighted correlations approach zero

**Plot Interpretation**:

Points closer to zero on the x-axis indicate better balance. Effective CBGPS
estimation should shift correlations from the unweighted row toward zero in
the weighted row.

**Example**

.. code-block:: python

   import cbps
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt

   # Generate continuous treatment data
   np.random.seed(42)
   n = 500
   df = pd.DataFrame({
       'dose': np.random.uniform(0, 100, n),
       'age': np.random.normal(45, 12, n),
       'income': np.random.lognormal(10, 0.5, n)
   })

   # Fit CBGPS
   fit = cbps.CBPS('dose ~ age + income', data=df, att=0)

   # Plot correlation-based balance
   cbps.plot_cbps_continuous(fit)
   plt.show()

   # Return correlation data as DataFrame
   corr_df = cbps.plot_cbps_continuous(fit, silent=False)
   print(corr_df)

Weight Diagnostics
------------------

.. automodule:: cbps.diagnostics.weights_diag
   :members:
   :undoc-members:

Comprehensive diagnostics for inverse probability weights, including Kish (1965)
effective sample size (ESS), weight distribution summaries, and extreme value detection.

**Key Metrics**:

- **ESS (Effective Sample Size)**: Measures information loss from weighting; ratio close to 1 is ideal
- **Coefficient of Variation**: Summarizes weight heterogeneity
- **Extreme Weight Detection**: Identifies weights exceeding 10× the median

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   from cbps.diagnostics.weights_diag import weight_diagnostics

   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Run weight diagnostics
   diag = weight_diagnostics(fit.weights, treat=df['treat'].values)
   print(f"ESS: {diag['ess']:.1f} / {len(df)} (ratio: {diag['ess_ratio']:.3f})")
   print(f"Weight CV: {diag['cv']:.3f}")
   print(f"Extreme weights: {diag['n_extreme']}")

Overlap Check
-------------

.. automodule:: cbps.diagnostics.overlap
   :members:
   :undoc-members:

Diagnostics for the positivity (overlap) assumption using the trimming approach
of Crump et al. (2009). Assesses whether propensity score distributions have
sufficient common support for reliable causal inference.

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   from cbps.diagnostics.overlap import check_overlap

   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Check overlap
   overlap = check_overlap(fit.fitted_values, df['treat'].values)
   print(f"Overlap region: {overlap['overlap_region']}")
   print(f"Violation detected: {overlap['violation_detected']}")
   if overlap['recommended_alpha']:
       print(f"Recommended trimming: alpha={overlap['recommended_alpha']}")

Normality Test
--------------

.. automodule:: cbps.diagnostics.normality
   :members:
   :undoc-members:

Diagnostics for the conditional normality assumption required by continuous
treatment CBPS (Fong, Hazlett & Imai, 2018). Tests whether treatment residuals
(T - X'β) follow a normal distribution.

**Example**

.. code-block:: python

   import cbps
   import numpy as np
   import pandas as pd
   from cbps.diagnostics.normality import test_treatment_normality

   # Simulate continuous treatment data
   np.random.seed(42)
   n = 500
   X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
   treat = X @ np.array([1, 0.5, -0.3, 0.2]) + np.random.randn(n)

   # Test normality of treatment residuals
   result = test_treatment_normality(treat, X)
   print(f"Test: {result['test_used']}")
   print(f"Statistic: {result['statistic']:.4f}, p-value: {result['p_value']:.4f}")
   print(f"Reject normality: {result['reject_normality']}")

oCBPS Conditions
----------------

.. automodule:: cbps.diagnostics.ocbps_conditions
   :members:
   :undoc-members:

Verification of observable necessary conditions for optimal CBPS (Fan et al. 2022)
validity. Checks identification, balance, overidentification, and positivity conditions.

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   from cbps.diagnostics.ocbps_conditions import verify_ocbps_conditions

   df = load_lalonde(dehejia_wahba_only=True)

   # Fit optimal CBPS
   fit = cbps.CBPS(
       'treat ~ age + educ + re74 + re75',
       data=df,
       att=0,
       baseline_formula='~ age + educ + re74',
       diff_formula='~ re74 + re75'
   )

   # Verify conditions
   conditions = verify_ocbps_conditions(
       result={'weights': fit.weights, 'J': fit.J, 'ps': fit.fitted_values},
       X=fit.x,
       treat=df['treat'].values
   )
   print(f"Identification OK: {conditions['identification_ok']}")
   print(f"Balance achieved: {conditions['balance_achieved']}")
   print(f"Overlap OK: {conditions['overlap_ok']}")
   print(f"All conditions met: {conditions['all_conditions_met']}")

CBPSResults Methods
-------------------

The ``CBPSResults`` class provides methods for statistical inference and diagnostics.

summary
~~~~~~~

.. automethod:: cbps.core.results.CBPSResults.summary
   :no-index:

Generate a statistical summary of the fitted CBPS model.

**Returns**: ``CBPSSummary`` object containing:

- Coefficient table with estimates, standard errors, z-statistics, and p-values
- J-statistic for testing over-identifying restrictions (GMM specification test)
- Log-likelihood value
- Convergence status

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ', data=df)
   summ = fit.summary()
   print(summ)

vcov
~~~~

.. automethod:: cbps.core.results.CBPSResults.vcov
   :no-index:

Compute the variance-covariance matrix of estimated propensity score coefficients.

**Returns**: ``np.ndarray`` of shape ``(k, k)`` where ``k`` is the number of parameters.

**Example**

.. code-block:: python

   import cbps
   import numpy as np
   from cbps.datasets import load_lalonde

   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ', data=df)
   vcov_mat = fit.vcov()

   # Extract standard errors from diagonal
   se = np.sqrt(np.diag(vcov_mat))
   print(f"Standard errors: {se}")

   # Compute parameter correlation matrix
   corr = vcov_mat / np.outer(se, se)
   print(f"Correlation matrix:\n{corr}")

__str__
~~~~~~~

.. automethod:: cbps.core.results.CBPSResults.__str__
   :no-index:

Return a formatted string representation of the CBPS results (invoked by ``print()``).

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   df = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS('treat ~ age + educ', data=df)
   print(fit)

Complete Diagnostic Workflow
----------------------------

The following workflow demonstrates a complete CBPS analysis with diagnostics:

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   import matplotlib.pyplot as plt
   import numpy as np

   # Step 1: Load data
   df = load_lalonde(dehejia_wahba_only=True)

   # Step 2: Estimate CBPS
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Step 3: Check convergence
   if not fit.converged:
       print("Warning: Optimization did not converge")

   # Step 4: Print statistical summary
   print(fit.summary())

   # Step 5: Assess covariate balance
   bal = cbps.balance(fit)
   print("\nBalance diagnostics:")
   print(bal['balanced'])

   # Step 6: Visual diagnostics
   cbps.plot_cbps(fit)
   plt.show()

   # Step 7: Examine variance-covariance matrix
   vcov = fit.vcov()
   se = np.sqrt(np.diag(vcov))
   print(f"\nStandard errors: {se}")

   # Step 8: Extract weights for outcome analysis
   weights = fit.weights
   print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

**Quality Criteria**:

1. **Convergence**: ``fit.converged`` should be ``True``
2. **Balance**: ASMD < 0.1 for all covariates (or correlations near 0 for continuous treatment)
3. **J-statistic**: Non-significant p-value indicates model is not rejected
4. **Weight stability**: Extreme weight ratios may indicate positivity violations

Troubleshooting
~~~~~~~~~~~~~~~

- **ASMD remains high after CBPS**: Consider adding interaction terms or
  polynomial terms to the propensity score model. Try ``method='over'`` for
  better balance. Check for positivity violations.
- **Continuous treatment correlations not near zero**: Verify that ``att=0``
  was used (required for continuous treatments). Consider increasing model
  flexibility.
- **Balance function returns unexpected structure**: Ensure the input is a
  fitted CBPS result object (``CBPSResults``), not a raw dictionary from
  ``cbps_fit()``.

**References**:

- Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
  *Journal of the Royal Statistical Society, Series B*, 76(1), 243-263.
  https://doi.org/10.1111/rssb.12027

- Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
  score for a continuous treatment. *The Annals of Applied Statistics*, 12(1), 156-177.
  https://doi.org/10.1214/17-AOAS1101

- Stuart, E.A. (2010). Matching methods for causal inference: A review and a
  look forward. *Statistical Science*, 25(1), 1-21.
  https://doi.org/10.1214/09-STS313

**See Also**

- :func:`cbps.CBPS` — Primary CBPS estimation function
- :func:`cbps.AsyVar` — Asymptotic variance for binary treatment ATE
- :func:`cbps.vcov_outcome` — Variance adjustment for continuous treatment

