CBIV: Instrumental Variable CBPS
=================================

:func:`cbps.CBIV` implements the covariate balancing propensity score for
instrumental variable settings with noncompliance. It estimates compliance
probabilities (compliers, always-takers, never-takers) via GMM and returns
inverse compliance weights suitable for downstream complier-average causal
effect estimation.

Main Function
-------------

.. autofunction:: cbps.CBIV

Result Container
----------------

.. autoclass:: cbps.iv.cbiv.CBIVResults
   :members:
   :show-inheritance:

.. autoclass:: cbps.iv.cbiv.CBIVSummary
   :members:
   :show-inheritance:

Numerical Warnings
------------------

.. autoclass:: cbps.iv.cbiv.CBIVNumericalWarning
   :show-inheritance:
