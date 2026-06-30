Nonparametric CBPS (npCBPS)
============================

.. currentmodule:: cbps

.. versionadded:: 0.1.0

npCBPS
------

.. autofunction:: cbps.npCBPS

Nonparametric covariate balancing generalized propensity score (npCBGPS)
using empirical likelihood, as described in Section 3.3 of Fong, Hazlett,
and Imai (2018).

**Key Features**:

- No parametric propensity score model specification required
- Direct weight estimation via empirical likelihood maximization
- Penalized imbalance approach for finite-sample balance control
- Supports both continuous and factor treatments

**When to Use npCBPS**:

- When the treatment assignment mechanism is unknown or complex
- When parametric models may be misspecified
- When a nonparametric approach that directly targets covariate balance is preferred
- When computational cost is acceptable (slower than parametric CBPS)

**Mathematical Framework**:

The npCBPS estimator solves the penalized empirical likelihood problem
(Equation 10 in Fong et al., 2018):

.. math::

   \min_{w_i, \eta} \left[ \sum_{i=1}^n \log w_i + \frac{1}{2\rho} \eta^T \eta \right]

subject to the constraints:

.. math::

   \sum_{i=1}^n w_i X_i^* T_i^* = \eta, \quad
   \sum_{i=1}^n w_i X_i^* = 0, \quad
   \sum_{i=1}^n w_i T_i^* = 0, \quad
   \sum_{i=1}^n w_i = n, \quad w_i > 0

where :math:`X^*` and :math:`T^*` are whitened covariates and standardized
treatment, :math:`\eta` is the allowed finite-sample weighted correlation,
and :math:`\rho` is the ``corprior`` parameter controlling the penalty.

**Dual Formulation** (Fong et al., 2018, Section 3.3.2):

The primal problem is solved via its dual. Introducing Lagrange multipliers
:math:`\lambda = (\lambda_1, \ldots, \lambda_{2K+1})` for the constraints, the
dual problem optimizes over :math:`(2K+1)` multipliers instead of :math:`n` weights:

.. math::

   \max_{\lambda} \sum_{i=1}^n \log\left(1 + \lambda^\top c_i\right)
   + \frac{1}{2\rho} \|\eta(\lambda)\|^2

where :math:`c_i` is the constraint vector for observation :math:`i` combining
the covariate balance, treatment balance, and normalization constraints.
Each weight is recovered as :math:`w_i = n / (1 + \lambda^\top c_i)`.

**Convergence Diagnostic (sumw0)**:

The ``sumw0`` attribute reports :math:`\sum_{i=1}^n w_i^{(0)} / n` where
:math:`w_i^{(0)}` are the unnormalized weights before rescaling to sum to :math:`n`.
Under correct convergence, this ratio should be close to 1.0 because the
normalization constraint :math:`\sum_i w_i = n` is part of the optimization.
Deviations greater than 5% indicate that the optimizer failed to satisfy the
constraints, typically due to:

- ``corprior`` being too small (overly tight balance requirement)
- Near-collinearity in the whitened covariate matrix
- Insufficient iterations in the BFGS optimizer

**Implementation Notes**:

The algorithm uses BFGS optimization for the dual problem. Key characteristics:

1. **Non-convex optimization**: The empirical likelihood objective is not generally convex, so there is no guarantee of finding the global optimum (Section 3.3.2).
2. **Dual formulation**: Optimizes over (2K+1) Lagrange multipliers instead of n weights directly.
3. **Convergence diagnostic**: Check ``sumw0`` (the sum of unnormalized weights), which should be close to 1.0 (within 5%).

**Example**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   # Load data
   df = load_lalonde()

   # Estimate weights using empirical likelihood
   fit = cbps.npCBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr',
       data=df,
       corprior=None,  # Default: 0.1/n (Section 3.3.4 recommendation)
       print_level=0   # Suppress optimization output
   )

   # Check convergence (sumw0 should be close to 1.0)
   print(f"Weight sum (sumw0): {fit.sumw0:.4f}")
   print(f"Deviation from 1.0: {abs(fit.sumw0 - 1.0):.4f}")
   print(f"Converged: {fit.converged}")

   # Access results
   print(f"Weights shape: {fit.weights.shape}")
   print(f"Weighted correlations (eta): {fit.eta}")
   print(f"Log empirical likelihood: {fit.log_el:.4f}")

**Convergence Diagnostics**

.. code-block:: python

   import numpy as np
   import cbps
   from cbps.datasets import load_lalonde

   df = load_lalonde()
   fit = cbps.npCBPS('treat ~ age + educ + black + hisp', data=df)

   # Check weight sum (should be close to 1.0, tolerance ±5%)
   if abs(fit.sumw0 - 1.0) > 0.05:
       print("Warning: Poor convergence - consider adjusting corprior")

   # Check weighted correlations (should be close to 0)
   max_corr = np.max(np.abs(fit.eta))
   print(f"Max weighted correlation: {max_corr:.6f}")

Troubleshooting
~~~~~~~~~~~~~~~

- **sumw0 deviates from 1.0 by more than 5%**: Increase ``corprior`` to relax
  the balance requirement. The default ``corprior=0.1/n`` (Section 3.3.4
  recommendation) works well in most cases, but may be too strict for
  high-dimensional or ill-conditioned data.
- **Optimization does not converge**: The empirical likelihood objective is
  non-convex (Section 3.3.2), so convergence to a local optimum is possible.
  Try different starting values or increase ``print_level`` to monitor progress.
- **Negative weights**: This should not occur as the empirical likelihood
  formulation enforces :math:`w_i > 0`. If observed, it indicates a numerical
  issue — try increasing ``corprior``.
- **Very large weights**: Some observations may receive disproportionately large
  weights. This is more common with small samples or when the treatment mechanism
  is highly nonlinear. Consider using parametric :func:`cbps.CBPS` as an alternative.

**See Also**:

- :func:`cbps.CBPS` - Parametric CBPS for binary and continuous treatments
- :func:`cbps.balance` — Covariate balance assessment after npCBPS estimation
- :func:`cbps.diagnostics.plots.plot_cbps` — Visual balance diagnostics

**References**:

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101

Result Object
-------------

NPCBPSResults
~~~~~~~~~~~~~

.. autoclass:: cbps.nonparametric.npcbps.NPCBPSResults
   :members:
   :undoc-members:
   :show-inheritance:

The result object returned by ``npCBPS()``. Contains:

**Core Attributes**:

- ``weights``: Estimated weights normalized to sum to n.
- ``eta``: Optimal weighted correlations :math:`\eta = \alpha \cdot \eta_0`, where :math:`\eta_0` is the initial (unweighted) correlation.
- ``sumw0``: Sum of unnormalized weights before normalization. Should be close to 1.0 (within 5%); large deviations indicate convergence issues.
- ``par``: Optimal scaling parameter :math:`\alpha \in [0, 1]` from the line search. Values near 0 indicate tight balance; values near 1 indicate relaxed balance.
- ``log_el``: Log empirical likelihood at the optimum.
- ``log_p_eta``: Log prior density :math:`\log f(\eta)` at the optimum, where :math:`\eta \sim N(0, \rho I_K)`.
- ``converged``: Optimization convergence status.
- ``iterations``: Number of optimization iterations.

**Example**

.. code-block:: python

   import numpy as np
   import cbps
   from cbps.datasets import load_lalonde

   df = load_lalonde()
   fit = cbps.npCBPS('treat ~ age + educ + black', data=df)

   # Weights for outcome analysis
   weights = fit.weights
   print(f"Weights sum to: {weights.sum():.1f}")

   # Balance diagnostics
   print(f"Weighted correlations (eta): {fit.eta}")
   print(f"Max absolute correlation: {np.max(np.abs(fit.eta)):.6f}")

   # Convergence check
   print(f"Weight sum (sumw0): {fit.sumw0:.4f}")
   print(f"Converged: {fit.converged}")

**Validation**

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   df = load_lalonde()
   fit = cbps.npCBPS('treat ~ age + educ + black + hisp', data=df)

   # Verify convergence
   assert abs(fit.sumw0 - 1.0) < 0.05, "Poor convergence"
   assert fit.converged, "Optimization did not converge"

   # Valid solution confirmed
   print("npCBPS estimation successful")

