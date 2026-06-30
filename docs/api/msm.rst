Marginal Structural Models (CBMSM)
===================================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

Overview
--------

The ``CBMSM`` function implements Covariate Balancing Propensity Score estimation
for Marginal Structural Models, as developed by Imai and Ratkovic (2015).

Marginal structural models (MSMs) use inverse probability of treatment weighting
(IPTW) to estimate causal effects from longitudinal data with time-varying
treatments and confounders. Unlike standard regression, MSMs can adjust for
time-dependent confounders affected by prior treatment without introducing
post-treatment bias.

The key challenge in MSMs is estimating inverse probability weights. Standard
maximum likelihood estimation of propensity scores can lead to poor covariate
balance and sensitivity to model misspecification. CBMSM addresses this by
directly targeting covariate balance when estimating weights.

CBMSM
-----

.. autofunction:: cbps.CBMSM

**Key Features**:

- Robust to treatment model misspecification through covariate balancing
- Supports both time-invariant and time-varying treatment models
- Stabilized weights for variance reduction
- Low-rank variance approximation for computational efficiency
- GMM-based estimation with optimal weighting

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~~

For a panel of :math:`N` units observed over :math:`J` time periods, let
:math:`T_{ij} \in \{0, 1\}` denote the treatment status and :math:`X_{ij}`
the covariates for unit :math:`i` at time :math:`j`. Denote treatment
history up to time :math:`j` as :math:`\bar{T}_{ij} = (T_{i1}, \ldots, T_{ij})`.

The inverse probability weight for unit :math:`i` is:

.. math::

   w_i = \prod_{j=1}^{J} \frac{1}{P(T_{ij} \mid \bar{T}_{i,j-1}, \bar{X}_{ij})}

The stabilized weight incorporates the marginal treatment probability:

.. math::

   w_i^* = \prod_{j=1}^{J} \frac{P(T_{ij} \mid \bar{T}_{i,j-1})}{P(T_{ij} \mid \bar{T}_{i,j-1}, \bar{X}_{ij})}

CBMSM estimates these weights by solving a GMM problem where moment conditions
are derived from the covariate balancing property. At each time period :math:`j`,
weights should balance covariates across current and future treatment assignments,
conditional on past treatment history. The number of binding moment conditions
per covariate at period :math:`j` is :math:`2^J - 2^{j-1}`, yielding a total of
:math:`\sum_{j=1}^{J} K (2^J - 2^{j-1})` conditions for :math:`K` covariates
(Imai and Ratkovic 2015, Equation 17).

**Hadamard Matrix Representation** (Imai & Ratkovic, 2015, Eq. 17):

The balance conditions across all treatment history combinations can be compactly
represented using the Hadamard matrix :math:`H_J` of order :math:`2^J`. For each
covariate :math:`X_k` and time period :math:`j`, the balance condition requires:

.. math::

   \frac{1}{N} \sum_{i=1}^N w_i X_{ijk} \cdot h_{\ell}(\bar{T}_{ij}) = 0

where :math:`h_\ell(\cdot)` are the rows of the Hadamard matrix encoding treatment
history contrasts, and :math:`w_i` are the inverse probability weights.

**Low-Rank Variance Approximation** (Imai & Ratkovic, 2015, Eq. 27):

When ``msm_variance='approx'`` (default), the GMM weighting matrix uses a low-rank
approximation that assumes zero correlation across balance conditions from different
time periods:

.. math::

   \hat{\Sigma}_{\text{approx}} = \text{blockdiag}(\hat{\Sigma}_1, \ldots, \hat{\Sigma}_J)

where :math:`\hat{\Sigma}_j` is the estimated covariance of the balance conditions
at time :math:`j`. This reduces computational cost from :math:`O(K^2 \cdot 4^J)` to
:math:`O(K^2 \cdot J \cdot 2^J)` while maintaining good finite-sample performance.
When ``msm_variance='full'``, the complete covariance matrix is computed without
this block-diagonal approximation.

Estimation Methods
~~~~~~~~~~~~~~~~~~

**Two-step GMM** (``twostep=True``, default): Computes the optimal weighting
matrix using initial estimates, then re-estimates parameters. Faster and
typically sufficient for most applications.

**Continuous updating GMM** (``twostep=False``): Updates the weighting matrix
at each iteration. More computationally intensive but may provide better
finite-sample performance.

**Variance approximation** (``msm_variance``):

- ``'approx'`` (default): Low-rank approximation assuming zero correlation
  across balance conditions (Imai and Ratkovic, 2015, Eq. 27). Recommended
  for computational efficiency.
- ``'full'``: Complete covariance matrix computation. More accurate but
  computationally expensive, especially with many time periods.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import cbps
   from cbps.datasets import load_blackwell

   # Load Blackwell negative campaign advertising data
   # 114 candidates observed over 5 weeks
   df = load_blackwell()

   # Estimate MSM weights with time-invariant treatment model
   fit = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName',
       time='time',
       data=df,
       type='MSM',
       time_vary=False,
       twostep=True,
       msm_variance='approx'
   )

   # Examine results
   print(f"Sample: {fit.n_units} units x {fit.n_periods} periods")
   print(f"Converged: {fit.converged}")
   print(f"J-statistic: {fit.J:.4f}")

   # Weight diagnostics
   print(f"Weight range: [{fit.weights.min():.2f}, {fit.weights.max():.2f}]")

Time-varying vs Time-invariant Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time-invariant** (``time_vary=False``, default): A single set of propensity
score coefficients :math:`\beta` is shared across all time periods. This
assumes the treatment assignment mechanism is stable over time.

**Time-varying** (``time_vary=True``): Separate coefficients :math:`\beta_j`
are estimated for each time period :math:`j`, allowing the treatment model
to change over time.

.. code-block:: python

   # Time-invariant: coefficients shape (k,)
   fit1 = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName', time='time', data=df,
       time_vary=False
   )
   print(f"Coefficients shape: {fit1.coefficients.shape}")

   # Time-varying: coefficients shape (k, J)
   fit2 = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName', time='time', data=df,
       time_vary=True
   )
   print(f"Coefficients shape: {fit2.coefficients.shape}")

**See Also**:

- :class:`~cbps.msm.cbmsm.CBMSMResults` - Result object documentation
- :func:`~cbps.CBPS` - Cross-sectional propensity score estimation
- :func:`cbps.balance` — Covariate balance assessment
- :func:`cbps.AsyVar` — Asymptotic variance estimation (cross-sectional only)

Result Object
-------------

CBMSMResults
~~~~~~~~~~~~

.. autoclass:: cbps.msm.cbmsm.CBMSMResults
   :members:
   :undoc-members:
   :show-inheritance:

The ``CBMSMResults`` object returned by ``CBMSM()`` contains estimated weights,
model coefficients, and diagnostic information.

**Core Attributes**:

- ``weights``: Inverse probability weights :math:`1/P(T|X)`, shape ``(n_units,)``
- ``fitted_values``: Stabilized weights :math:`P(T)/P(T|X)`, shape ``(n_units,)``
- ``coefficients``: Propensity score model parameters
- ``treat_hist``: Treatment history matrix, shape ``(n_units, n_periods)``
- ``treat_cum``: Cumulative treatment counts per unit

**Diagnostic Attributes**:

- ``converged``: Whether GMM optimization converged
- ``J``: Hansen J-statistic (normalized GMM objective)
- ``n_units``: Number of units :math:`N`
- ``n_periods``: Number of time periods :math:`J`
- ``time_vary``: Whether period-specific coefficients were estimated

Using Weights in Outcome Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The estimated weights are used in weighted outcome regression to estimate
marginal structural model parameters. The ``fitted_values`` (stabilized weights)
are recommended for outcome analysis as they reduce variance.

.. code-block:: python

   import cbps
   from cbps.datasets import load_blackwell
   import statsmodels.api as sm

   df = load_blackwell()

   # Estimate MSM weights
   fit = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName',
       time='time',
       data=df,
       time_vary=False
   )

   # Construct treatment history indicators for outcome model
   # Each row is one unit; columns are treatment at each time period
   treat_hist = fit.treat_hist  # shape (114, 5)

   # Get final outcome (vote share at last period)
   df_last = df[df['time'] == df['time'].max()].copy()
   df_last = df_last.sort_values('demName')
   outcome = df_last['demprcnt'].values

   # Weighted regression of outcome on treatment history
   X_outcome = sm.add_constant(treat_hist)
   wls = sm.WLS(outcome, X_outcome, weights=fit.fitted_values)
   result = wls.fit()

   print("Marginal Structural Model Results:")
   print(f"Intercept: {result.params[0]:.3f}")
   for j in range(fit.n_periods):
       print(f"Treatment effect (period {j+1}): {result.params[j+1]:.3f}")

Weight Diagnostics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cbps
   from cbps.datasets import load_blackwell

   df = load_blackwell()
   fit = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length',
       id='demName', time='time', data=df
   )

   # Check dimensions
   print(f"Units: {fit.n_units}, Periods: {fit.n_periods}")
   print(f"Weights shape: {fit.weights.shape}")
   print(f"Fitted values shape: {fit.fitted_values.shape}")

   # Weight distribution
   print(f"Weight range: [{fit.weights.min():.2f}, {fit.weights.max():.2f}]")
   print(f"Weight mean: {fit.weights.mean():.2f}")

   # Convergence check
   if not fit.converged:
       print("Warning: Optimization did not converge")
       print("Consider adjusting model specification or using different init")

   # J-statistic (lower is better)
   print(f"J-statistic: {fit.J:.4f}")

Troubleshooting
~~~~~~~~~~~~~~~

- **Moment condition explosion with many time periods**: The number of balance
  conditions grows as :math:`O(K \cdot 2^J)` where :math:`K` is the number of
  covariates and :math:`J` is the number of time periods. For :math:`J > 6`,
  consider using ``msm_variance='approx'`` and reducing the number of covariates.
- **Optimization does not converge**: Try ``twostep=True`` (faster, usually sufficient).
  Reduce model complexity or check for time periods with very few treated units.
- **Extreme weights**: Longitudinal weights are products across time periods, so
  extreme values are more common than in cross-sectional settings. Check for
  time periods with very low treatment probabilities.
- **Stabilized vs. unstabilized weights**: Use ``fit.fitted_values`` (stabilized)
  for outcome analysis to reduce variance. Use ``fit.weights`` (unstabilized)
  only when stabilized weights are not appropriate.

Rank Selection Diagnostics
--------------------------

.. automodule:: cbps.msm.rank_diagnostics
   :members:
   :undoc-members:

Sensitivity analysis tools for SVD rank selection in CBMSM covariate matrices.
These diagnostics help assess whether results are sensitive to the singular value
threshold used for low-rank approximation.

.. warning::
   Automatic rank selection goes beyond the Imai & Ratkovic (2015) specification.
   Use these tools for sensitivity analysis only. The default fixed threshold
   (1e-4) should be used for published analyses unless justified.

**Example**

.. code-block:: python

   import numpy as np
   from cbps.msm.rank_diagnostics import diagnose_rank_selection

   # Diagnose rank sensitivity for a covariate matrix
   rng = np.random.default_rng(42)
   X = rng.standard_normal((100, 10))

   diag = diagnose_rank_selection(X)
   print(f"Total columns: {diag['total_columns']}")
   print(f"Ranks by threshold: {diag['ranks_by_threshold']}")
   print(f"Recommendation: {diag['recommended_action']}")

References
----------

Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. *Journal of the American Statistical
Association*, 110(511), 1013-1023. https://doi.org/10.1080/01621459.2014.956872

Robins, J. M., Hernan, M. A., and Brumback, B. (2000). Marginal structural
models and causal inference in epidemiology. *Epidemiology*, 11(5), 550-560.
https://doi.org/10.1097/00001648-200009000-00011

Blackwell, M. (2013). A framework for dynamic causal inference in political
science. *American Journal of Political Science*, 57(2), 504-520.
https://doi.org/10.1111/ajps.12000
