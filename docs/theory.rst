Theoretical Background
======================

This page provides an overview of the theoretical foundations of the Covariate Balancing Propensity Score (CBPS) methodology.

Core Methodology
----------------

The CBPS method estimates propensity scores by optimizing two objectives simultaneously:

1. **Prediction of treatment assignment** (standard propensity score objective)
2. **Covariate balance** between treatment and control groups

This dual optimization is achieved through Generalized Method of Moments (GMM) estimation.

Mathematical Framework
----------------------

Binary Treatment
^^^^^^^^^^^^^^^^

For binary treatment :math:`T_i \in \{0, 1\}` and covariates :math:`X_i`, the propensity score is:

.. math::

   \pi(X_i; \beta) = P(T_i = 1 | X_i) = \frac{1}{1 + \exp(-X_i^T \beta)}

**GMM Moment Conditions:**

The CBPS estimator :math:`\hat{\beta}` solves:

.. math::

   \frac{1}{n} \sum_{i=1}^n g(T_i, X_i; \beta) = 0

where the moment conditions are:

.. math::

   g(T_i, X_i; \beta) = \begin{pmatrix}
   (T_i - \pi(X_i; \beta)) X_i \\
   \left(\frac{T_i}{\pi(X_i; \beta)} - \frac{1-T_i}{1-\pi(X_i; \beta)}\right) X_i
   \end{pmatrix}

The first set of moments ensures prediction accuracy (standard logistic regression).
The second set enforces covariate balance (weighted means are equal across groups).

**Reference:** Imai & Ratkovic (2014), JRSS-B, DOI: `10.1111/rssb.12027 <https://doi.org/10.1111/rssb.12027>`_

Continuous Treatment
^^^^^^^^^^^^^^^^^^^^

For continuous treatment :math:`T_i \in \mathbb{R}` and covariates :math:`X_i`, the Generalized Propensity Score (GPS) is:

.. math::

   r(t, X_i; \beta, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(t - X_i^T \beta)^2}{2\sigma^2}\right)

**Stabilized Weights:**

Following Robins, Hernán and Brumback (2000), the continuous CBPS uses stabilized inverse probability weights:

.. math::

   w_i = \frac{f(T_i^*)}{f(T_i^* | X_i^*; \hat{\beta}, \hat{\sigma}^2)}

where :math:`f(T_i^*)` is the marginal density (standard normal after standardization) and :math:`f(T_i^* | X_i^*)` is the conditional density (GPS).

**Balance Condition:**

The stabilized weights should satisfy:

.. math::

   \text{Corr}(X_j, T | w) \approx 0 \quad \forall j

where :math:`\text{Corr}(X_j, T | w)` is the weighted correlation between covariate :math:`X_j` and treatment :math:`T`.

**F-Statistic:**

Overall balance is assessed using:

.. math::

   F = n \cdot R^2_w / (1 - R^2_w)

where :math:`R^2_w` is the weighted R-squared from regressing :math:`T` on :math:`X`. Target: :math:`F < 10^{-4}`.

**Reference:** Fong, Hazlett & Imai (2018), The Annals of Applied Statistics, DOI: `10.1214/17-AOAS1101 <https://doi.org/10.1214/17-AOAS1101>`_

Marginal Structural Models (MSM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For longitudinal data with time-varying treatments :math:`T_{it}` at times :math:`t = 1, \ldots, T`, the MSM weight for unit :math:`i` is:

.. math::

   w_i = \prod_{t=1}^T \frac{1}{\pi(T_{it} | X_{it}, T_{i,t-1}; \beta_t)}

**Time-Invariant Parameters:**

When :math:`\beta_t = \beta` for all :math:`t` (``time_vary=False``), the propensity score model is:

.. math::

   \pi(T_{it} | X_{it}, T_{i,t-1}; \beta) = \frac{1}{1 + \exp(-[X_{it}, T_{i,t-1}]^T \beta)}

**Critical Implementation Detail:**

The coefficient vector :math:`\beta` must be correctly expanded to match the stacked covariate matrix across all time periods.

**Reference:** Imai & Ratkovic (2015), JASA, DOI: `10.1080/01621459.2014.956872 <https://doi.org/10.1080/01621459.2014.956872>`_

High-Dimensional CBPS
^^^^^^^^^^^^^^^^^^^^^

For high-dimensional settings where :math:`p \approx n` or :math:`p >> n`, hdCBPS uses LASSO variable selection:

**Step 1: LASSO Selection**

.. math::

   \hat{\beta}^{\text{LASSO}} = \arg\min_\beta \left\{ -\ell(\beta) + \lambda \|\beta\|_1 \right\}

where :math:`\ell(\beta)` is the log-likelihood and :math:`\lambda` is chosen by cross-validation.

**Step 2: CBPS on Selected Variables**

Let :math:`S = \{j : \hat{\beta}^{\text{LASSO}}_j \neq 0\}` be the selected variables. Estimate CBPS using only :math:`X_S`.

**Reference:** Ning, Peng & Imai (2020), Biometrika, DOI: `10.1093/biomet/asaa020 <https://doi.org/10.1093/biomet/asaa020>`_

Instrumental Variables
^^^^^^^^^^^^^^^^^^^^^^

For noncompliance settings with instrument :math:`Z_i`, treatment :math:`D_i`, and outcome :math:`Y_i`:

**Two-Sided Noncompliance:**

.. math::

   \text{CACE} = E[Y_i(1) - Y_i(0) | D_i(1) > D_i(0)]

where :math:`D_i(z)` is the potential treatment under instrument :math:`z`.

**CBIV Estimation:**

1. Estimate compliance probabilities using CBPS
2. Compute IV weights
3. Estimate CACE using weighted regression

**Reference:** Fong (2018), Unpublished manuscript

Estimation Methods
------------------

Just-Identified GMM (``method='exact'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Number of moment conditions = Number of parameters

- Faster computation
- No J-statistic (model is exactly identified)
- Use when you trust the model specification

Over-Identified GMM (``method='over'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Number of moment conditions > Number of parameters

- More robust to model misspecification
- Provides J-statistic for specification testing
- Recommended for most applications

Two-Step Estimation (``two_step=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``method='over'``, the two-step GMM estimator uses the optimal weighting matrix:

.. math::

   W = \left(\frac{1}{n} \sum_{i=1}^n g_i g_i^T\right)^{-1}

- Asymptotically efficient
- First step obtains initial estimates, second step uses optimal weights
- Recommended for final analysis after model selection

When to Use Each Method
-----------------------

+-------------------+---------------------------+---------------------------+
| Method            | Use When                  | Avoid When                |
+===================+===========================+===========================+
| Binary CBPS       | Binary treatment          | Continuous treatment      |
+-------------------+---------------------------+---------------------------+
| Continuous CBPS   | Continuous treatment      | Binary treatment          |
+-------------------+---------------------------+---------------------------+
| CBMSM             | Longitudinal data         | Cross-sectional data      |
+-------------------+---------------------------+---------------------------+
| hdCBPS            | p ≈ n or p >> n           | p << n                    |
+-------------------+---------------------------+---------------------------+
| npCBPS            | Nonlinear relationships   | Linear relationships      |
+-------------------+---------------------------+---------------------------+
| CBIV              | Noncompliance/IV setting  | No instrument available   |
+-------------------+---------------------------+---------------------------+

Key References
--------------

1. **Imai, K., & Ratkovic, M. (2014).** Covariate balancing propensity score. 
   *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 76(1), 243-263.
   DOI: `10.1111/rssb.12027 <https://doi.org/10.1111/rssb.12027>`_

2. **Imai, K., & Ratkovic, M. (2015).** Robust estimation of inverse probability weights for marginal structural models.
   *Journal of the American Statistical Association*, 110(511), 1013-1023.
   DOI: `10.1080/01621459.2014.956872 <https://doi.org/10.1080/01621459.2014.956872>`_

3. **Fong, C., Hazlett, C., & Imai, K. (2018).** Covariate balancing propensity score for a continuous treatment.
   *The Annals of Applied Statistics*, 12(1), 156-177.
   DOI: `10.1214/17-AOAS1101 <https://doi.org/10.1214/17-AOAS1101>`_

4. **Ning, Y., Peng, S., & Imai, K. (2020).** Robust estimation of causal effects via a high-dimensional covariate balancing propensity score.
   *Biometrika*, 107(3), 533-554.
   DOI: `10.1093/biomet/asaa020 <https://doi.org/10.1093/biomet/asaa020>`_

5. **Fong, C. (2018).** Robust and efficient estimation of causal effects with calibrated covariate balance.
   *Unpublished manuscript*.

