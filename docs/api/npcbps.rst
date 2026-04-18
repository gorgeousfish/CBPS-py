npCBPS: Nonparametric CBPS
===========================

:func:`cbps.npCBPS` is the nonparametric covariate balancing propensity score
estimator based on empirical likelihood (Fong, Hazlett & Imai, 2018, Section 3.3).
It requires no parametric specification for the propensity score and instead
directly estimates inverse probability weights that satisfy weighted balance
constraints.

Main Function
-------------

.. autofunction:: cbps.npCBPS

Low-level Fit Interface
-----------------------

.. autofunction:: cbps.npCBPS_fit

Result Container
----------------

.. autoclass:: cbps.nonparametric.npcbps.NPCBPSResults
   :members:
   :show-inheritance:

.. autoclass:: cbps.nonparametric.npcbps.NPCBPSSummary
   :members:
   :show-inheritance:

Supporting Utilities
--------------------

The nonparametric backend exposes a small number of reusable helpers that can
be useful for developing custom diagnostics.

.. autofunction:: cbps.nonparametric.cholesky_whitening

.. autofunction:: cbps.nonparametric.get_w

.. autofunction:: cbps.nonparametric.log_post

.. autofunction:: cbps.nonparametric.llog

.. autofunction:: cbps.nonparametric.llogp
