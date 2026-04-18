hdCBPS: High-Dimensional CBPS
==============================

:func:`cbps.hdCBPS` implements the high-dimensional covariate balancing
propensity score estimator of Ning, Peng & Imai (2020). The algorithm
combines LASSO variable selection with calibrated balancing conditions and
returns treatment effect estimates together with sandwich standard errors.

.. note::
   ``hdCBPS`` requires the optional ``glmnetforpython`` dependency. See the
   :doc:`../installation` page for platform-specific instructions, including
   Apple Silicon notes.

Main Function
--------------

.. autofunction:: cbps.hdCBPS

Result Container
----------------

.. autoclass:: cbps.highdim.hdcbps.HDCBPSResults
   :members:
   :show-inheritance:

.. autoclass:: cbps.highdim.hdcbps.HDCBPSSummary
   :members:
   :show-inheritance:
