Datasets
========

The ``cbps.datasets`` subpackage bundles the benchmark datasets used in the
CBPS methodology papers. All loaders return fully resolved objects; no
network access or external downloads are required.

Cross-sectional
---------------

.. autofunction:: cbps.datasets.load_lalonde

.. autofunction:: cbps.datasets.load_lalonde_psid_combined

Longitudinal
------------

.. autofunction:: cbps.datasets.load_blackwell

Continuous Treatment
--------------------

.. autofunction:: cbps.datasets.load_continuous_simulation

.. autofunction:: cbps.datasets.load_political_ads

Nonparametric CBPS Simulations
------------------------------

.. autofunction:: cbps.datasets.load_npcbps_continuous_sim
