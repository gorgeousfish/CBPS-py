"""
Core CBPS Algorithm Implementation

This module implements the fundamental covariate balancing propensity score (CBPS)
algorithms for various treatment modalities. The CBPS methodology extends
traditional propensity score estimation by directly incorporating covariate balance
conditions into the estimation framework through generalized method of moments
(GMM) optimization.

Algorithm Components
--------------------

The module provides implementations for the following treatment types:

* **Binary treatments** (`cbps_binary_fit`): Standard CBPS for estimating average
  treatment effects (ATE) and average treatment effects on the treated (ATT)
  using logistic regression models within the GMM framework

* **Continuous treatments** (`cbps_continuous_fit`): Generalized propensity score
  (GPS) estimation for continuous treatment variables, extending the CBPS
  methodology to parametric continuous treatment models

* **Multi-valued treatments** (`cbps_3treat_fit`, `cbps_4treat_fit`): Extension
  to categorical treatments with three or four levels using multinomial logistic
  regression models

* **Optimal CBPS** (`cbps_optimal_2treat`): Doubly robust estimation that
  incorporates outcome model information to improve efficiency while maintaining
  robustness to model misspecification

Statistical Framework
--------------------

For binary treatment assignment :math:`T \\in \\{0,1\\}` and covariates
:math:`X`, the CBPS estimator solves the following GMM optimization problem:

.. math::
    \\hat{\\beta} = \\arg\\min_{\\beta} \\, \\frac{1}{2} g_n(\\beta)' W_n g_n(\\beta)

where the moment conditions :math:`g_n(\\beta)` combine:

1. **Score condition**: :math:`E[T_i - e(X_i, \\beta)] = 0`
2. **Balance conditions**: :math:`E[X_i(T_i - e(X_i, \\beta))] = 0`

The weight matrix :math:`W_n` is chosen optimally to achieve the
Hansen (1982) efficiency bound within the class of GMM estimators.

Computational Methods
---------------------

The implementations employ numerical optimization algorithms suitable for
the non-convex objective functions that arise in CBPS estimation:

* Two-step GMM estimator for computational efficiency
* Continuous-updating GMM for improved finite-sample properties
* Newton-Raphson and BFGS algorithms with analytical gradients

References
----------
.. [1] Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
   Journal of the Royal Statistical Society, Series B 76(1), 243-263.
   https://doi.org/10.1111/rssb.12027

.. [2] Hansen, L. P. (1982). Large sample properties of generalized method
   of moments estimators. Econometrica 50(4), 1029-1054.
"""

# Binary treatment CBPS implementation
from .cbps_binary import cbps_binary_fit

# Continuous treatment CBPS (Generalized Propensity Score)
from .cbps_continuous import cbps_continuous_fit

# Multi-valued treatment CBPS for categorical treatments
from .cbps_multitreat import cbps_3treat_fit, cbps_4treat_fit

# Optimal CBPS with dual balancing conditions
from .cbps_optimal import cbps_optimal_2treat

# Result classes and summary statistics
from .results import CBPSResults, CBPSSummary

__all__ = [
    'cbps_binary_fit',
    'cbps_continuous_fit',
    'cbps_3treat_fit',
    'cbps_4treat_fit',
    'cbps_optimal_2treat',
    'CBPSResults',
    'CBPSSummary',
]

