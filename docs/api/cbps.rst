CBPS: Binary, Multi-Valued, and Continuous Treatments
======================================================

The top-level :func:`cbps.CBPS` function dispatches to specialised binary,
multi-valued, and continuous backends based on the type of the treatment
variable. For binary and multi-valued treatments the package supports both
just-identified GMM (``method='exact'``) and over-identified GMM
(``method='over'``); continuous treatments use a generalized propensity
score formulation (Fong, Hazlett & Imai, 2018).

Main Function
-------------

.. autofunction:: cbps.CBPS

Low-level Fit Interface
-----------------------

.. autofunction:: cbps.cbps_fit

Asymptotic Variance
-------------------

:func:`cbps.AsyVar` computes the asymptotic variance and confidence interval
for the ATE using one of two variance formulae derived in Fan et al. (2022):

- ``method='CBPS'`` — full sandwich formula that propagates
  propensity-score estimation uncertainty through the joint GMM influence
  function.
- ``method='oCBPS'`` — semiparametric efficiency bound (Hahn, 1998), valid
  when the ``baseline_formula`` / ``diff_formula`` oCBPS balancing
  conditions are used.

The ``method`` argument of :func:`cbps.AsyVar` is unrelated to the
``method`` argument of :func:`cbps.CBPS`, which selects between
``'over'`` and ``'exact'`` GMM estimation.

.. autofunction:: cbps.AsyVar

Result Container
----------------

.. autoclass:: cbps.core.results.CBPSResults
   :members:
   :show-inheritance:

.. autoclass:: cbps.core.results.CBPSSummary
   :members:
   :show-inheritance:
