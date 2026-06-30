Instrumental Variables (CBIV)
==============================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

Overview
--------

The CBIV (Covariate Balancing Propensity Score for Instrumental Variables) module
implements propensity score estimation for instrumental variable settings with
treatment noncompliance. It extends the CBPS methodology to identify and weight
compliers for local average treatment effect (LATE) estimation.

CBIV
----

.. autofunction:: cbps.CBIV

**Key Features**:

- Two-sided noncompliance: models compliers, always-takers, and never-takers
- One-sided noncompliance: models compliers and never-takers only
- Three estimation methods: MLE, over-identified GMM, and exactly-identified GMM
- Returns inverse probability weights for downstream LATE/CACE estimation

**When to Use CBIV**:

- Randomized experiments with imperfect compliance
- Encouragement designs where a randomly assigned instrument affects treatment uptake
- Settings requiring complier average causal effect (CACE/LATE) estimation

**Noncompliance Types**:

1. **Two-sided noncompliance** (``twosided=True``):
   
   - Treatment group members may refuse treatment (never-takers among treated)
   - Control group members may obtain treatment elsewhere (always-takers among control)
   - Models three principal strata: compliers, always-takers, never-takers

2. **One-sided noncompliance** (``twosided=False``):
   
   - Only one direction of noncompliance (e.g., treatment refusal only)
   - Assumes no always-takers (π_a = 0)
   - Models two principal strata: compliers and never-takers

**Principal Stratification Framework**:

Following Angrist, Imbens, and Rubin (1996), units are classified into principal strata
based on their potential treatment status under each instrument value:

- **Compliers**: :math:`D_i(Z=1) = 1, D_i(Z=0) = 0` — respond to encouragement
- **Always-takers**: :math:`D_i(Z=1) = D_i(Z=0) = 1` — always treated
- **Never-takers**: :math:`D_i(Z=1) = D_i(Z=0) = 0` — never treated
- **Defiers**: :math:`D_i(Z=1) = 0, D_i(Z=0) = 1` — excluded by monotonicity assumption

**Mathematical Framework**:

For instrument :math:`Z_i`, treatment :math:`D_i`, and covariates :math:`X_i`,
CBIV models the compliance type probabilities:

.. math::

   \pi_c(X_i) = P(\text{Complier} | X_i), \quad
   \pi_a(X_i) = P(\text{Always-taker} | X_i), \quad
   \pi_n(X_i) = P(\text{Never-taker} | X_i)

The complier average causal effect (CACE), also known as the local average
treatment effect (LATE), is identified among compliers:

.. math::

   \text{CACE} = E[Y_i(1) - Y_i(0) | \text{Complier}]

**Estimation Method Comparison**

The three estimation methods differ in their moment conditions
(Imai & Ratkovic, 2014, Section 3.3):

- **MLE** (``method='mle'``): Uses only the propensity score likelihood conditions.
  Efficient under correct model specification but sensitive to misspecification.

  .. math::

     g_{\text{mle}}(\beta) = \frac{1}{n} \sum_{i=1}^n s_\beta(D_i, Z_i, X_i)

- **Over-identified GMM** (``method='over'``): Combines propensity score and covariate
  balance conditions. More robust to model misspecification (recommended).

  .. math::

     g_{\text{over}}(\beta) = \frac{1}{n} \sum_{i=1}^n \begin{pmatrix}
     s_\beta(D_i, Z_i, X_i) \\
     X_i \left[\frac{D_i}{\pi_c(X_i)} - 1\right]
     \end{pmatrix}

- **Exactly-identified GMM** (``method='exact'``): Uses only covariate balance
  conditions without propensity score likelihood.

  .. math::

     g_{\text{exact}}(\beta) = \frac{1}{n} \sum_{i=1}^n
     X_i \left[\frac{D_i}{\pi_c(X_i)} - 1\right]

where :math:`s_\beta` is the score function of the compliance type model and
:math:`\pi_c(X_i)` is the estimated complier probability.

**Example (Two-Sided Noncompliance)**

.. code-block:: python

   import cbps
   import numpy as np

   # Simulate IV data with noncompliance
   np.random.seed(42)
   n = 500

   # Instrument (randomized encouragement)
   Z = np.random.binomial(1, 0.5, n)

   # Pre-treatment covariates (intercept added automatically)
   X = np.random.randn(n, 2)

   # Generate treatment with noncompliance
   # - Compliers (70%): D = Z (respond to encouragement)
   # - Always-takers (15%): D = 1 regardless of Z
   # - Never-takers (15%): D = 0 regardless of Z
   compliance_type = np.random.choice(
       ['complier', 'always', 'never'], n, p=[0.7, 0.15, 0.15]
   )
   Tr = np.where(compliance_type == 'complier', Z,
                 np.where(compliance_type == 'always', 1, 0))

   # Fit CBIV model
   fit = cbps.CBIV(
       Tr=Tr,              # Binary treatment (0/1)
       Z=Z,                # Binary instrument (0/1)
       X=X,                # Covariate matrix
       method='mle',       # 'over', 'exact', or 'mle'
       twosided=True       # Two-sided noncompliance
   )

   # Examine results
   print(f"Coefficients shape: {fit.coefficients.shape}")  # (k, 2)
   print(f"Weights shape: {fit.weights.shape}")            # (n,)
   print(f"All weights positive: {all(fit.weights > 0)}")

**Example (One-Sided Noncompliance)**

.. code-block:: python

   # One-sided noncompliance: assumes no always-takers (π_a = 0)
   # Appropriate for settings where control units cannot access treatment
   fit = cbps.CBIV(
       Tr=Tr,
       Z=Z,
       X=X,
       method='mle',
       twosided=False
   )

**Estimation Methods**

.. code-block:: python

   # Maximum Likelihood Estimation (MLE)
   # - Propensity score conditions only
   # - Efficient under correct model specification
   fit_mle = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='mle')

   # Over-identified GMM (recommended)
   # - Combines propensity score and covariate balance conditions
   # - More robust to model misspecification
   fit_gmm = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='over')

   # Exactly-identified GMM
   # - Covariate balance conditions only
   # - No propensity score conditions
   fit_exact = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='exact')

**See Also**:

- :func:`cbps.CBPS` — Standard CBPS for observational studies without IV
- :func:`cbps.AsyVar` — Asymptotic variance estimation for treatment effects
- :func:`cbps.balance` — Covariate balance assessment

Result Object
-------------

CBIVResults
~~~~~~~~~~~

.. autoclass:: cbps.iv.cbiv.CBIVResults
   :members:
   :undoc-members:
   :show-inheritance:

The result object returned by :func:`CBIV`. Contains estimated compliance
probabilities, inverse probability weights, and model diagnostics.

**Core Attributes**:

- ``coefficients``: Estimated compliance model coefficients

  - Two-sided: shape ``(k, 2)`` where columns are [β_c, β_a]
  - One-sided: shape ``(k,)`` for complier coefficients only

- ``fitted_values``: Estimated compliance type probabilities

  - Two-sided: shape ``(n, 3)`` with columns [π_c, π_a, π_n]
  - One-sided: shape ``(n, 1)`` with complier probabilities π_c

- ``weights``: Inverse probability weights for CACE estimation, computed as 1/π_c.
  Shape ``(n,)``.

- ``p_complier``: Property returning complier probabilities π_c as a 1D array.
  Provides a unified interface for both two-sided and one-sided models.

- ``converged``: Boolean indicating optimization convergence status.

- ``J``: Hansen's J-statistic for the over-identification test.

- ``deviance``: Model deviance (−2 × log-likelihood).

- ``bal``: Covariate balance loss from the GMM objective.

- ``method``: Estimation method used (``'mle'``, ``'over'``, or ``'exact'``).

- ``two_sided``: Boolean indicating whether two-sided noncompliance model was used.

**Example**

.. code-block:: python

   fit = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='over')

   # Access compliance probabilities
   print(f"Fitted values shape: {fit.fitted_values.shape}")

   # Inverse probability weights for CACE estimation
   weights = fit.weights
   print(f"Mean weight: {weights.mean():.4f}")

   # Model information
   print(f"Method: {fit.method}")
   print(f"Two-sided: {fit.two_sided}")
   print(f"Converged: {fit.converged}")

**CACE Estimation with Weighted Least Squares**

.. code-block:: python

   import statsmodels.api as sm

   # Assume Y is the outcome variable (shape (n,))
   # Step 1: Fit CBIV to obtain complier weights
   fit = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='mle', twosided=True)

   # Step 2: Weighted least squares regression of outcome on treatment
   # Inverse probability weights adjust for differential compliance probabilities
   outcome_model = sm.WLS(Y, sm.add_constant(Tr), weights=fit.weights)
   result = outcome_model.fit()

   # The coefficient on treatment estimates CACE
   cace = result.params[1]
   cace_se = result.bse[1]
   print(f"CACE estimate: {cace:.4f} (SE: {cace_se:.4f})")

**Accessing Compliance Probabilities**

.. code-block:: python

   # Use the p_complier property for a unified interface
   # Returns (n,) array for both two-sided and one-sided models
   fit = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='over')
   p_complier = fit.p_complier
   print(f"Mean complier probability: {p_complier.mean():.4f}")

   # For two-sided models, access all three compliance probabilities
   fit = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=True)
   # fitted_values has shape (n, 3): columns are [π_c, π_a, π_n]
   p_complier = fit.fitted_values[:, 0]  # P(Complier | X)
   p_always = fit.fitted_values[:, 1]    # P(Always-taker | X)
   p_never = fit.fitted_values[:, 2]     # P(Never-taker | X)

   # Verify probabilities sum to 1
   assert np.allclose(p_complier + p_always + p_never, 1.0)

   # For one-sided models, fitted_values has shape (n, 1)
   fit = cbps.CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=False)
   p_complier = fit.fitted_values[:, 0]  # Or use fit.p_complier

Troubleshooting
~~~~~~~~~~~~~~~

- **Convergence failure**: IV models are more challenging to estimate than standard
  CBPS. Try ``method='mle'`` first (simpler optimization), then ``method='over'``.
  Ensure sufficient sample size relative to the number of covariates.
- **All weights are identical**: This may indicate that the instrument has no effect
  on treatment uptake (weak instrument). Verify that the instrument-treatment
  relationship is strong.
- **Complier probabilities near zero**: Some units may have very low estimated
  complier probabilities, leading to extreme weights. Consider trimming weights
  or checking for violations of the monotonicity assumption.
- **Negative complier probabilities**: This should not occur with proper estimation.
  Check that the instrument ``Z`` is correctly coded as binary (0/1) and that
  treatment ``Tr`` is also binary.

References
----------

Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
*Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
https://doi.org/10.1111/rssb.12027

Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of
Causal Effects Using Instrumental Variables. *Journal of the American
Statistical Association*, 91(434), 444-455.
https://doi.org/10.1080/01621459.1996.10476902
