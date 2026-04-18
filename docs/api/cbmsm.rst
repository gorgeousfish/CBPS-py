CBMSM: Marginal Structural Models
==================================

:func:`cbps.CBMSM` implements covariate balancing propensity score estimation
for marginal structural models with time-varying treatments
(Imai & Ratkovic, 2015). Both time-invariant and time-varying coefficient
specifications are supported, and stabilised MSM weights are returned.

Main Function
-------------

.. autofunction:: cbps.CBMSM

Low-level Fit Interface
-----------------------

.. autofunction:: cbps.cbmsm_fit

Result Container
----------------

.. autoclass:: cbps.msm.cbmsm.CBMSMResults
   :members:
   :show-inheritance:

.. autoclass:: cbps.msm.cbmsm.CBMSMSummary
   :members:
   :show-inheritance:
