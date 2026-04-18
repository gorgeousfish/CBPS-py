Tutorials
=========

This section contains step-by-step replications of the three core CBPS
methodology papers. Each tutorial is provided as an interactive Jupyter
notebook in the ``examples/`` directory, together with an equivalent
standalone Python script.

.. note::
   Monte Carlo portions of these notebooks use a reduced number of replications
   so that the full notebook executes in a few minutes. Qualitative rankings
   across estimators should match the published figures; exact numerical
   agreement is not expected.

Tutorial Overview
-----------------

1. **Binary and Multi-Valued CBPS** — Kang-Schafer simulation and LaLonde
   propensity-score matching (Imai & Ratkovic, 2014).
2. **Continuous Treatment CBPS** — DGP simulation study and political ads
   application (Fong, Hazlett & Imai, 2018).
3. **Marginal Structural Models (CBMSM)** — Blackwell longitudinal data
   (Imai & Ratkovic, 2015).

Tutorial 1: Binary Treatment CBPS
----------------------------------

**Files:** ``examples/replicate_imai_ratkovic_2014.ipynb`` and
``examples/replicate_imai_ratkovic_2014.py``.

**Datasets:** Kang-Schafer (2007) simulation DGP and the LaLonde-PSID
combined sample (:func:`cbps.datasets.load_lalonde_psid_combined`).

**Topics covered:**

- Implementing the four-scenario Kang-Schafer simulation (both correct,
  PS correct, outcome correct, both wrong).
- Downstream estimators: Horvitz-Thompson, Hájek IPW, WLS, doubly-robust.
- CBPS with ``method='exact'`` (just-identified GMM) and ``method='over'``
  (over-identified GMM with J-statistic).
- Matching using CBPS propensity scores on the LaLonde-PSID sample.

Tutorial 2: Continuous Treatment CBPS
--------------------------------------

**Files:** ``examples/replicate_fong_hazlett_imai_2018.ipynb`` and
``examples/replicate_fong_hazlett_imai_2018.py``.

**Datasets:** Four-DGP simulation from Section 4 of the paper and the Urban &
Niebler (2014) political advertising data
(:func:`cbps.datasets.load_political_ads`).

**Topics covered:**

- Generalized propensity score (GPS) estimation with a continuous treatment.
- Parametric CBGPS via :func:`cbps.CBPS` and nonparametric CBGPS via
  :func:`cbps.npCBPS`.
- Weighted correlations between covariates and treatment as the primary
  balance diagnostic.
- The F-statistic from regressing treatment on covariates under the
  estimated weights (target well below 1 after weighting).
- Box-Cox transformed treatment and the signed correlation table for the
  political advertising application.

Tutorial 3: Marginal Structural Models (CBMSM)
-----------------------------------------------

**Files:** ``examples/replicate_imai_ratkovic_2015.ipynb`` and
``examples/replicate_imai_ratkovic_2015.py``.

**Dataset:** Blackwell (2013) longitudinal political campaign data across
five time periods (:func:`cbps.datasets.load_blackwell`).

**Topics covered:**

- Marginal structural models for panel data with time-varying treatments.
- Time-invariant (``time_vary=False``) and time-varying
  (``time_vary=True``) parameter specifications.
- Stabilized CBPS weights versus GLM-based stabilized weights.
- Propensity score behaviour across time periods.

Running the Tutorials
----------------------

To run the notebooks locally:

1. Install Jupyter:

   .. code-block:: bash

      pip install jupyter

2. From the root of the cloned repository, start Jupyter:

   .. code-block:: bash

      jupyter notebook examples/

3. Open any of the ``replicate_*.ipynb`` files.

Alternatively, the equivalent ``.py`` scripts can be executed directly:

.. code-block:: bash

   python examples/replicate_imai_ratkovic_2014.py
   python examples/replicate_fong_hazlett_imai_2018.py
   python examples/replicate_imai_ratkovic_2015.py

A self-contained runner for the Kang-Schafer simulation portion of the
2014 replication is available as ``examples/run_replication.py``.

Additional Worked Examples
--------------------------

For focused code patterns covering each function of the package — binary and
continuous CBPS, CBMSM, CBIV, npCBPS, hdCBPS, balance diagnostics, and
asymptotic variance estimation — refer to:

- The :doc:`../quickstart` guide.
- The :doc:`../advanced_usage` guide.
- The :doc:`../api/index` reference generated from the source docstrings.

