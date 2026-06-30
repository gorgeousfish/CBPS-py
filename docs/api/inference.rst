Inference Tools
===============

.. currentmodule:: cbps

.. versionadded:: 0.1.0

This module provides variance estimation and confidence interval construction
for causal effect estimates obtained through CBPS weighting. The inference
methods properly account for estimation uncertainty in propensity scores
within the GMM framework.

AsyVar
------

.. autofunction:: cbps.AsyVar

Asymptotic variance estimation for average treatment effect (ATE) estimators
under binary treatment using CBPS weights.

**Key Features**

- Sandwich variance estimator accounting for propensity score estimation
- Semiparametric efficiency bound under correct model specification
- Confidence intervals for ATE estimates
- Supports both standard CBPS and optimal CBPS (oCBPS)

**When to Use AsyVar**

- Constructing confidence intervals for treatment effects
- Quantifying uncertainty from propensity score estimation
- Binary treatment with CBPS or oCBPS weights
- Doubly-robust inference with outcome model specification

**Mathematical Framework**

For standard CBPS, the asymptotic variance follows the sandwich formula
from Theorem 2.1 of Fan et al. (2022):

.. math::

   V = \Sigma_\mu + H_0^\top (H_f^\top \Omega^{-1} H_f)^{-1} H_0
       - 2 H_0^\top H_f^{-1} \text{Cov}(\mu, g)

For optimal CBPS (oCBPS), the variance attains the semiparametric efficiency
bound (Corollary 2.2 of Fan et al. 2022):

.. math::

   V_{\text{opt}} = E\left[ \frac{\sigma_1^2(X)}{\pi(X)} +
                    \frac{\sigma_0^2(X)}{1-\pi(X)} + (L(X) - \mu)^2 \right]

where:

- :math:`\sigma_t^2(X)`: Conditional variance of potential outcome :math:`Y(t)`
- :math:`\pi(X)`: Propensity score
- :math:`L(X) = E[Y(1) - Y(0) | X]`: Conditional average treatment effect

**Efficiency Bound Conditions** (Fan et al., 2022, Corollary 2.1–2.2):

The oCBPS variance :math:`V_{\text{opt}}` attains the semiparametric efficiency bound
when the following conditions are satisfied:

1. The propensity score model :math:`\pi_\beta(X)` is correctly specified
2. The baseline outcome model :math:`E[Y(0) \mid X]` (from ``baseline_formula``) is correctly specified
3. The treatment effect heterogeneity model :math:`E[Y(1) - Y(0) \mid X]` (from ``diff_formula``) is correctly specified

Under correct specification of only the propensity score model, the estimator remains
consistent but may not achieve the efficiency bound. Under correct specification of
only the outcome models, the estimator is doubly robust (Fan et al., 2022, Theorem 3.1).

**Example (Standard CBPS)**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde
   import numpy as np

   # Load LaLonde experimental data
   df = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS for ATT
   fit = cbps.CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Outcome variable
   y = df['re78'].values

   # Asymptotic variance and confidence interval
   result = cbps.AsyVar(
       Y=y,
       CBPS_obj=fit,
       method="CBPS",
       CI=0.95
   )

   # Extract results
   mu_hat = result['mu.hat']       # ATE point estimate
   se = result['std.err']          # Standard error
   ci = result['CI.mu.hat']        # 95% confidence interval

   print(f"ATE: {mu_hat:.2f}")
   print(f"SE: {se:.2f}")
   print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

**Example (Optimal CBPS with Efficiency Bound)**

.. code-block:: python

   # Optimal CBPS achieves semiparametric efficiency when both
   # propensity score and outcome models are correctly specified
   fit = cbps.CBPS(
       formula='treat ~ age + educ',
       data=df,
       baseline_formula='~ age + educ + re74',  # E[Y(0)|X] model
       diff_formula='~ I(re75==0)',             # E[Y(1)-Y(0)|X] model
       att=0  # oCBPS requires ATE estimation
   )

   y = df['re78'].values

   # Variance estimation using semiparametric efficiency bound
   result = cbps.AsyVar(
       Y=y,
       CBPS_obj=fit,
       method="oCBPS",
       CI=0.95
   )

   print(f"ATE: {result['mu.hat']:.2f}")
   print(f"SE: {result['std.err']:.2f}")
   print(f"95% CI: [{result['CI.mu.hat'][0]:.2f}, {result['CI.mu.hat'][1]:.2f}]")

**References**

Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
Optimal covariate balancing conditions in propensity score estimation.
*Journal of Business & Economic Statistics*, 41(1), 97-110.
https://doi.org/10.1080/07350015.2021.2002159

**See Also**

- :func:`cbps.CBPS` - Main CBPS estimation function
- :func:`cbps.vcov_outcome` - Variance adjustment for continuous treatment
- :meth:`cbps.core.results.CBPSResults.vcov` - Propensity score coefficient variance

vcov_outcome
------------

.. autofunction:: cbps.vcov_outcome

Variance-covariance adjustment for weighted outcome regression with continuous
treatments. This function implements the sandwich variance estimator that
properly accounts for uncertainty in generalized propensity score estimation.

**Key Features**

- Sandwich variance estimator for weighted least squares regression
- Propagates weight estimation uncertainty to standard errors
- Designed specifically for continuous treatment CBPS
- Follows the GMM-based variance derivation of Fong et al. (2018)

**When to Use vcov_outcome**

- Estimating dose-response functions with continuous treatments
- Weighted outcome regression using continuous-treatment CBPS weights (``cbps_fit.weights``)
- Constructing confidence intervals for treatment coefficients
- Obtaining robust standard errors that account for two-stage estimation

Within the continuous-treatment implementation, ``cbps_fit.weights`` contains the stabilized regression weights used in downstream outcome models, while ``cbps_fit.fitted_values`` stores the estimated conditional density :math:`f_\theta(T \mid X)` (generalized propensity score). ``vcov_outcome`` should be paired with the same outcome model and the same weights used in the fitted weighted regression.

**Mathematical Framework**

Following Fong, Hazlett, and Imai (2018), for weighted outcome regression
:math:`Y_i = Z_i^\top \delta + \epsilon_i` with stabilized weights
:math:`w_i = f(T_i)/f_\theta(T_i | X_i)`, the adjusted variance is:

.. math::

   V(\hat{\delta}) = S_\delta^{-1} \cdot \text{E}\left[
       (s_i - S_\theta M^{-1} m_\theta)^{\otimes 2}
   \right] \cdot S_\delta^{-1\top}

where the components are defined as:

- :math:`S_\delta = E[w_i Z_i Z_i^\top]` — the weighted Hessian of the outcome model
- :math:`s_i = w_i \epsilon_i Z_i` — the weighted score for observation :math:`i`, with :math:`\epsilon_i = Y_i - Z_i^\top \delta`
- :math:`m_\theta = \partial \log f_\theta(T_i \mid X_i) / \partial \theta` — the score of the generalized propensity score model
- :math:`M = E[\partial m_\theta / \partial \theta^\top]` — the Jacobian (information matrix) of the GPS model
- :math:`S_\theta = E[s_i \cdot m_\theta^\top]` — the cross-covariance between outcome and GPS scores

The adjustment term :math:`S_\theta M^{-1} m_\theta` accounts for the estimation
uncertainty in the generalized propensity score, yielding standard errors that are
typically larger than naive WLS standard errors.

**Example (Continuous Treatment)**

.. code-block:: python

   import cbps
   from cbps.datasets import load_political_ads
   import statsmodels.api as sm
   import numpy as np

   # Load political advertising data (Fong et al. 2018)
   df, metadata = load_political_ads()

   # Select subset of columns for analysis
   covars = ['MedianHHInc', 'PerCapitaHHInc', 'PercentOver65']

   # Estimate continuous treatment CBPS for total ads
   fit = cbps.CBPS(
       formula=f'TotAds ~ {" + ".join(covars)}',
       data=df,
       att=0  # ATE estimation (only valid for continuous)
   )

   # Weighted outcome regression for Republican vote share
   outcome_vars = ['TotAds'] + covars[:2]
   Z = sm.add_constant(df[outcome_vars])
   Y = df['RepubShare'].values
   wls_fit = sm.WLS(Y, Z, weights=fit.weights).fit()

   # Variance adjustment for weight estimation uncertainty
   vcov_adj = cbps.vcov_outcome(
       cbps_fit=fit,
       Y=Y,
       Z=Z.values,
       delta=wls_fit.params
   )

   # Compare adjusted vs. naive standard errors
   se_adj = np.sqrt(np.diag(vcov_adj))
   print(f"Adjusted SE: {se_adj}")
   print(f"Naive SE:    {wls_fit.bse.values}")

.. note::

   ``vcov_outcome`` is designed exclusively for continuous treatment CBPS.
   For binary treatments, use :func:`cbps.AsyVar` to obtain treatment effect
   variance with proper adjustment for propensity score estimation.

**References**

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment. *The Annals of Applied Statistics*,
12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

**See Also**

- :func:`cbps.AsyVar` - ATE variance for binary treatments
- :func:`cbps.CBPS` - Main CBPS estimation function
- :meth:`cbps.core.results.CBPSResults.vcov` - Propensity score coefficient variance

Return Values
-------------

**AsyVar** returns a dictionary containing:

+----------------+-----------------------------------------------------------+
| Key            | Description                                               |
+================+===========================================================+
| ``mu.hat``     | ATE point estimate via inverse probability weighting      |
+----------------+-----------------------------------------------------------+
| ``asy.var``    | Asymptotic variance of sqrt(n) * (mu_hat - mu)            |
+----------------+-----------------------------------------------------------+
| ``var``        | Finite-sample variance (``asy.var / n``)                  |
+----------------+-----------------------------------------------------------+
| ``std.err``    | Standard error (``sqrt(var)``)                            |
+----------------+-----------------------------------------------------------+
| ``CI.mu.hat``  | Confidence interval bounds as ndarray of shape (2,)       |
+----------------+-----------------------------------------------------------+

**vcov_outcome** returns:

- ``ndarray`` of shape ``(p, p)``: Adjusted variance-covariance matrix for
  the ``p`` outcome model coefficients. Diagonal elements are variances;
  off-diagonal elements are covariances.

**Example**

.. code-block:: python

   # Binary treatment: AsyVar returns a dictionary
   result = cbps.AsyVar(Y=outcome, CBPS_obj=fit, method='CBPS', CI=0.95)
   print(f"ATE = {result['mu.hat']:.3f} (SE = {result['std.err']:.3f})")
   print(f"95% CI: [{result['CI.mu.hat'][0]:.3f}, {result['CI.mu.hat'][1]:.3f}]")

   # Continuous treatment: vcov_outcome returns a matrix
   vcov = cbps.vcov_outcome(cbps_fit, Y, Z, delta)
   se = np.sqrt(np.diag(vcov))  # Extract standard errors
   t_stat = delta / se          # Compute t-statistics

Troubleshooting
~~~~~~~~~~~~~~~

- **AsyVar raises error with continuous treatment**: :func:`AsyVar` is designed
  exclusively for binary treatments. For continuous treatments, use
  :func:`vcov_outcome` instead.
- **vcov_outcome raises error with binary treatment**: :func:`vcov_outcome` is
  designed exclusively for continuous treatments. For binary treatments, use
  :func:`AsyVar` instead.
- **Large standard errors from AsyVar**: Check for extreme propensity scores
  (near 0 or 1), which inflate variance. Consider trimming or using
  ``standardize=True`` in the CBPS estimation step.
- **vcov_outcome standard errors smaller than naive WLS**: This is unexpected.
  Verify that the correct CBPS fit object and outcome model are passed.
  The adjusted standard errors should generally be larger than naive ones.

