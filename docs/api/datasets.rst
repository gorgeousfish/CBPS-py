Datasets
========

.. currentmodule:: cbps.datasets

.. versionadded:: 0.1.0

This module provides functions for loading standard causal inference datasets commonly used in propensity score analysis research. All datasets are bundled with the package and require no external downloads.

Cross-Sectional Studies
-----------------------

LaLonde Dataset
~~~~~~~~~~~~~~~

.. autofunction:: load_lalonde

The LaLonde dataset originates from the National Supported Work (NSW) demonstration, a labor training program conducted in the mid-1970s. It serves as a canonical benchmark for propensity score methods due to the availability of both experimental and observational comparison groups.

**Data Characteristics**

- **Dehejia-Wahba Subsample**: 445 observations (185 treated, 260 control), 11 variables
- **Full Dataset**: 3212 observations, 12 variables

**Variables**

- ``treat``: Treatment indicator (1 = received job training)
- ``age``: Age in years
- ``educ``: Years of education
- ``black``: African American indicator
- ``hisp``: Hispanic indicator
- ``married``: Marital status indicator
- ``nodegr``: No high school degree indicator
- ``re74``, ``re75``: Real earnings in 1974, 1975 (pre-treatment)
- ``re78``: Real earnings in 1978 (outcome)

**Example**

.. code-block:: python

   from cbps.datasets import load_lalonde
   import cbps

   # Load Dehejia-Wahba subsample
   data = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS for ATT
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=data,
       att=1
   )

   # Check balance
   print(fit.balance())

**References**

LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
programs with experimental data. *American Economic Review*, 76(4), 604-620.
https://doi.org/10.2307/1806062

Dehejia, R. H. and Wahba, S. (1999). Causal effects in nonexperimental
studies: Reevaluating the evaluation of training programs. *Journal of the
American Statistical Association*, 94(448), 1053-1062.
https://doi.org/10.1080/01621459.1999.10473858

LaLonde PSID Combined
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_lalonde_psid_combined

This function provides combined NSW experimental and PSID control group data for
evaluation bias tests. The PSID control groups serve as observational
comparison units for assessing propensity score method performance.

**Data Configurations**

- ``psid_version='main'``: Full PSID controls (2490 observations)
- ``psid_version='controls2'``: Restricted PSID controls (253 observations)
- ``psid_version='controls3'``: Further restricted PSID controls (128 observations)

**Example**

.. code-block:: python

   from cbps.datasets import load_lalonde_psid_combined
   import cbps

   # Load combined data
   combined = load_lalonde_psid_combined(psid_version='main')

   # Estimate propensity scores for selection into experimental sample
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re75',
       data=combined,
       att=0  # ATE
   )

**References**

Smith, J. A. and Todd, P. E. (2005). Does matching overcome LaLonde's
critique of nonexperimental estimators? *Journal of Econometrics*, 125(1-2), 305-353.
https://doi.org/10.1016/j.jeconom.2004.04.011

Longitudinal Studies
--------------------

Blackwell Dataset
~~~~~~~~~~~~~~~~~

.. autofunction:: load_blackwell

The Blackwell dataset contains longitudinal observations of U.S. Senate and
gubernatorial candidates during the five weeks leading up to elections. It is
designed for demonstrating marginal structural model estimation with
time-varying treatments.

**Data Structure**

- **Units**: 114 candidates (identified by ``demName``)
- **Time Periods**: 5 weeks (``time`` = 1, 2, 3, 4, 5)
- **Total Observations**: 570 (114 × 5)

**Key Variables**

- ``demName``: Candidate identifier (string)
- ``d.gone.neg``: Whether candidate ran negative ads this period (0/1)
- ``d.gone.neg.l1``: Lagged treatment (previous period)
- ``d.gone.neg.l2``: Twice-lagged treatment
- ``camp.length``: Campaign duration
- ``demprcnt``: Democratic vote share (outcome)
- ``time``: Time period indicator (1-5)

**Example**

.. code-block:: python

   from cbps.datasets import load_blackwell
   import cbps

   # Load panel data
   df = load_blackwell()

   # Estimate CBMSM for time-varying treatment
   fit = cbps.CBMSM(
       formula='d.gone.neg ~ d.gone.neg.l1 + camp.length + deminc',
       id='demName',
       time='time',
       data=df,
       type='MSM'
   )

   # Extract MSM weights
   msm_weights = fit.weights

**References**

Blackwell, M. (2013). A framework for dynamic causal inference in political
science. *American Journal of Political Science*, 57(2), 504-520.
https://doi.org/10.1111/ajps.12000

Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. *Journal of the American Statistical
Association*, 110(511), 1013-1023.
https://doi.org/10.1080/01621459.2014.956872

Continuous Treatment
--------------------

Continuous Simulation
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_continuous_simulation

Simulation data from Fong, Hazlett, and Imai (2018) for validating continuous
treatment CBPS methods. The simulation framework evaluates estimation
performance under model misspecification scenarios.

**Data Generating Processes (DGPs)**

- **DGP1**: Both treatment and outcome correctly specified (linear models)
- **DGP2**: Treatment model misspecified (nonlinear term in X2)
- **DGP3**: Outcome model misspecified (nonlinear term in X2)
- **DGP4**: Doubly misspecified (both models incorrect)

**Data Characteristics**

- **Observations**: 200
- **Variables**: 12 (T, Y, X1-X10)
- **True ATE**: 1.0
- **Covariate Structure**: Multivariate normal with pairwise correlation 0.2

**Example**

.. code-block:: python

   from cbps.datasets import load_continuous_simulation
   import cbps

   # Load DGP1 simulation
   data, metadata = load_continuous_simulation(dgp=1)

   # Estimate CBGPS
   fit = cbps.CBPS(
       formula='T ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10',
       data=data,
       method='over'
   )

   # Weighted outcome regression for ATE
   import numpy as np
   import statsmodels.api as sm

   X_outcome = sm.add_constant(data['T'])
   wls = sm.WLS(data['Y'], X_outcome, weights=fit.weights)
   result = wls.fit()
   print(f"Estimated ATE: {result.params[1]:.3f}")
   print(f"True ATE: {metadata['true_ate']}")

**References**

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101

Political Ads Dataset
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_political_ads

Real-world application data from Urban and Niebler (2014) examining the
causal effects of political advertisements on campaign contributions. This
dataset demonstrates continuous treatment CBPS in practice.

**Data Characteristics**

- **Observations**: 16,265 zip codes
- **Variables**: 155
- **Treatment**: ``TotAds`` (total advertising count, range: 0-22,379)
- **Recommended Transformation**: Box-Cox with λ = -0.16

**Example**

.. code-block:: python

   from cbps.datasets import load_political_ads
   import cbps
   import numpy as np
   from scipy import stats

   # Load data
   data, metadata = load_political_ads()

   # Box-Cox transformation for approximate normality
   ads_transformed, _ = stats.boxcox(data['TotAds'] + 1)  # Add 1 for zeros
   data['TotAds_bc'] = ads_transformed

   # Estimate CBGPS on transformed treatment
   # (Select relevant covariates based on analysis goals)
   formula = 'TotAds_bc ~ MedianHHInc + PercentBlack + PercentHispanic + Urban'
   fit = cbps.CBPS(formula=formula, data=data, method='over')

**References**

Urban, C. and Niebler, S. (2014). Dollars on the sidewalk: Should U.S.
Presidential candidates advertise in uncontested states? *American Journal
of Political Science*, 58(2), 322-336.
https://doi.org/10.1111/ajps.12073

Nonparametric CBPS
------------------

npCBPS Simulation
~~~~~~~~~~~~~~~~~

.. autofunction:: load_npcbps_continuous_sim

Simulation data for validating the nonparametric CBPS (npCBPS) implementation.
Contains continuous treatment data with known benchmark results for testing.

**Data Characteristics**

- **Observations**: 500
- **Variables**: 7 (Y, T, X1-X5)
- **Covariate Structure**: Multivariate normal with correlation = 0.5

**Example**

.. code-block:: python

   from cbps.datasets import load_npcbps_continuous_sim
   import cbps

   # Load simulation data
   df = load_npcbps_continuous_sim()

   # Estimate npCBPS
   fit = cbps.npCBPS(
       formula='T ~ X1 + X2 + X3 + X4 + X5',
       data=df,
       corprior=0.1  # Prior correlation tolerance
   )

   # Access weights and balance
   weights = fit.weights
   print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

Module Contents
---------------

.. list-table:: Available Functions
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`load_lalonde`
     - LaLonde NSW job training evaluation data
   * - :func:`load_lalonde_psid_combined`
     - Combined NSW experimental and PSID control data
   * - :func:`load_blackwell`
     - Blackwell longitudinal campaign advertising data
   * - :func:`load_continuous_simulation`
     - Fong et al. (2018) continuous treatment simulation
   * - :func:`load_political_ads`
     - Urban & Niebler political advertising data
   * - :func:`load_npcbps_continuous_sim`
     - npCBPS validation simulation data

**See Also**

- :func:`cbps.CBPS` — Primary estimation function (binary, multi-valued, continuous treatments)
- :func:`cbps.CBMSM` — Marginal structural models (uses :func:`load_blackwell` data)
- :func:`cbps.npCBPS` — Nonparametric CBPS (uses :func:`load_npcbps_continuous_sim` data)
- :func:`cbps.hdCBPS` — High-dimensional CBPS (uses :func:`load_lalonde` data)
