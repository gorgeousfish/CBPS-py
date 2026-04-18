Diagnostics and Plotting
=========================

The ``cbps.diagnostics`` subpackage provides covariate balance assessment
utilities and (optional) plotting helpers. Balance is reported through
standardised mean differences for discrete treatments and weighted Pearson
correlations for continuous treatments.

Balance Assessment
------------------

.. autofunction:: cbps.balance

Low-level balance routines
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cbps.diagnostics.balance.balance_cbps

.. autofunction:: cbps.diagnostics.balance.balance_cbps_continuous

Continuous Treatment Diagnostics
--------------------------------

.. autofunction:: cbps.diagnostics.continuous_diagnostics.diagnose_cbgps_quality

.. autofunction:: cbps.diagnostics.continuous_diagnostics.print_balance_diagnosis

.. autofunction:: cbps.diagnostics.continuous_diagnostics.compute_weighted_correlations

.. autofunction:: cbps.diagnostics.continuous_diagnostics.compute_f_statistic

Plotting (requires matplotlib)
------------------------------

The plotting helpers are only imported when ``matplotlib`` is available.
Install the ``plots`` extra — currently from GitHub while ``cbps-python``
is not yet on PyPI — to enable them::

   pip install "cbps-python[plots] @ git+https://github.com/gorgeousfish/CBPS-py.git"

.. autofunction:: cbps.plot_cbps

.. autofunction:: cbps.plot_cbps_continuous

.. autofunction:: cbps.plot_cbmsm

.. autofunction:: cbps.plot_npcbps
