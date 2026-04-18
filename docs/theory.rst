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

where :math:`R^2_w` is the weighted R-squared from regressing :math:`T` on
:math:`X`. Balanced weights drive :math:`F` toward zero; as a concrete
benchmark, Fong, Hazlett and Imai (2018, Figure 3) report
:math:`F \approx 9.3 \times 10^{-5}` for CBGPS on the political-ads
application. Appropriate targets depend on sample size, covariate design,
and treatment scaling, and should be interpreted relative to the MLE
baseline for the same problem rather than used as a universal cutoff.

Marginal Structural Models (MSM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For longitudinal data with time-varying treatments :math:`T_{it}` at times :math:`t = 1, \ldots, T`, the MSM weight for unit :math:`i` is:

.. math::

   w_i = \prod_{t=1}^T \frac{1}{\pi(T_{it} | X_{it}, T_{i,t-1}; \beta_t)}

**Time-Invariant Parameters:**

When :math:`\beta_t = \beta` for all :math:`t` (``time_vary=False``), the propensity score model is:

.. math::

   \pi(T_{it} | X_{it}, T_{i,t-1}; \beta) = \frac{1}{1 + \exp(-[X_{it}, T_{i,t-1}]^T \beta)}

High-Dimensional CBPS
^^^^^^^^^^^^^^^^^^^^^

For high-dimensional settings where :math:`p \approx n` or :math:`p >> n`, hdCBPS uses LASSO variable selection:

**Step 1: LASSO Selection**

.. math::

   \hat{\beta}^{\text{LASSO}} = \arg\min_\beta \left\{ -\ell(\beta) + \lambda \|\beta\|_1 \right\}

where :math:`\ell(\beta)` is the log-likelihood and :math:`\lambda` is chosen by cross-validation.

**Step 2: CBPS on Selected Variables**

Let :math:`S = \{j : \hat{\beta}^{\text{LASSO}}_j \neq 0\}` be the selected variables. Estimate CBPS using only :math:`X_S`.

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

Optimal CBPS (oCBPS)
^^^^^^^^^^^^^^^^^^^^

Fan, Imai, Lee, Liu, Ning, and Yang (2022) extend the binary CBPS framework
with a *dual* set of balancing conditions that achieve the semiparametric
efficiency bound for the average treatment effect.

**Balancing Conditions:**

Let :math:`\pi_i = \pi(X_i; \beta)` denote the propensity score. Following
Fan et al. (2022, Eq. 3.3), the oCBPS estimator :math:`\hat{\beta}` solves
the joint GMM moment conditions

.. math::

   \frac{1}{n} \sum_{i=1}^n
   \left(\frac{T_i}{\pi_i} - \frac{1 - T_i}{1 - \pi_i}\right)
   h_1(X_i) = \mathbf{0},
   \qquad
   \frac{1}{n} \sum_{i=1}^n
   \left(\frac{T_i}{\pi_i} - 1\right) h_2(X_i) = \mathbf{0},

where :math:`h_1(X)` and :math:`h_2(X)` are user-specified basis functions.
The first condition is the standard CBPS-style balancing equation on
:math:`h_1`; the second matches the *weighted* covariates
:math:`\{(1 - \pi_i)/\pi_i \cdot h_2(X_i) : T_i = 1\}` in the treated group
to the *unweighted* covariates :math:`\{h_2(X_i) : T_i = 0\}` in the
control group. Setting :math:`h_1(X) = K(X) := E[Y(0) \mid X]` and
:math:`h_2(X) = L(X) := E[Y(1) - Y(0) \mid X]` yields the optimal
balancing function

.. math::

   \alpha^{\top} f(X) = \pi(X)\, E[Y(0) \mid X] + (1 - \pi(X))\, E[Y(1) \mid X]

from Fan et al. (2022, Corollary 2.2), producing a doubly-robust ATE
estimator whose asymptotic variance attains the semiparametric efficiency
bound of Hahn (1998).

**Sandwich Variance:**

The asymptotic variance is available in closed form. Writing
:math:`\mu = E[Y(1)] - E[Y(0)]`, Fan et al. (2022, Theorem 3.2) show

.. math::

   \sqrt{n}(\hat{\mu} - \mu) \;\xrightarrow{d}\;
   \mathcal{N}\!\left(0,\; \sigma_{\text{oCBPS}}^2\right),

with :math:`\sigma_{\text{oCBPS}}^2` implemented by :func:`cbps.AsyVar`
when its ``method`` argument is ``"oCBPS"``. The full (sub-optimal) sandwich
formula is also available via ``cbps.AsyVar(..., method="CBPS")`` (this
refers to the ``method`` argument of ``AsyVar``, not the ``method``
argument of :func:`cbps.CBPS`).

**Relationship to the Python API:**

- :func:`cbps.CBPS` with ``baseline_formula`` and ``diff_formula`` fits
  the oCBPS moment conditions above, with the ``baseline_formula`` design
  matrix playing the role of :math:`h_1(X)` and the ``diff_formula``
  design matrix the role of :math:`h_2(X)` (see
  :func:`cbps.core.cbps_optimal.cbps_optimal_2treat` in the source).
- :func:`cbps.AsyVar` computes the asymptotic variance using either the
  full sandwich or the oCBPS efficiency-bound variance, depending on its
  ``method`` argument.

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

+-------------------+--------------------------------+---------------------------+
| Method            | Use When                       | Avoid When                |
+===================+================================+===========================+
| Binary CBPS       | Binary treatment               | Continuous treatment      |
+-------------------+--------------------------------+---------------------------+
| Continuous CBPS   | Continuous treatment           | Binary treatment          |
+-------------------+--------------------------------+---------------------------+
| Optimal CBPS      | Want efficiency-bound SE       | No outcome model available|
|                   | for ATE with binary treatment  |                           |
+-------------------+--------------------------------+---------------------------+
| CBMSM             | Longitudinal data              | Cross-sectional data      |
+-------------------+--------------------------------+---------------------------+
| hdCBPS            | p ≈ n or p >> n                | p << n                    |
+-------------------+--------------------------------+---------------------------+
| npCBPS            | Nonlinear relationships        | Linear relationships      |
+-------------------+--------------------------------+---------------------------+
| CBIV              | Noncompliance/IV setting       | No instrument available   |
+-------------------+--------------------------------+---------------------------+
