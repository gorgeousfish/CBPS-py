CBPS: Covariate Balancing Propensity Score
===========================================

A comprehensive Python implementation of the covariate balancing propensity score methodology for causal inference from observational studies.

The Covariate Balancing Propensity Score (CBPS) represents a paradigm shift in propensity score estimation by directly incorporating covariate balance optimization into the estimation framework (Imai & Ratkovic, 2014). Unlike traditional approaches that require iterative model specification adjustments, CBPS seamlessly integrates treatment prediction with covariate balance through generalized method of moments (GMM) estimation.

The methodology exploits the dual characterization of propensity scores: (i) as conditional probabilities of treatment assignment given covariates, and (ii) as balancing scores that ensure independence between treatment and covariates after weighting. This dual role is operationalized through moment conditions that combine standard likelihood scores with balance constraints, yielding estimators with superior finite-sample performance while maintaining asymptotic efficiency.

This package provides state-of-the-art implementations for the full spectrum of CBPS methodologies:

- **Binary Treatments** - Robust estimation for 0/1 treatment assignments using logistic models
- **Multi-valued Treatments** - Extension to categorical treatments via multinomial logistic regression
- **Continuous Treatments** - Generalized propensity scores for continuous treatment variables
- **High-dimensional Settings** - Regularized estimation with LASSO variable selection (hdCBPS)
- **Nonparametric Estimation** - Empirical likelihood approaches avoiding parametric assumptions
- **Longitudinal Data** - Marginal structural models for time-varying treatments
- **Instrumental Variables** - Extensions for treatment noncompliance scenarios

Key Innovations
----------------

- **Unified framework** — single interface with automatic treatment-type detection
- **Advanced estimation** — both just-identified and over-identified GMM
- **Numerical robustness** — SVD-based matrix operations and comprehensive edge-case handling
- **Comprehensive diagnostics** — balance assessment, Hansen J-test, and plotting helpers
- **Asymptotic inference** — sandwich variance (``cbps.AsyVar``) and outcome-regression vcov (``cbps.vcov_outcome``)

Quick Start
-----------

Install from PyPI:

.. code-block:: bash

   pip install cbps-python

A minimal example on the LaLonde NSW sample:

.. code-block:: python

   import cbps
   from cbps.datasets import load_lalonde

   data = load_lalonde(dehejia_wahba_only=True)
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=data,
       att=1,                # ATT; use att=0 for ATE
       method='over',        # over-identified GMM (recommended)
   )
   print(fit.summary())
   print(cbps.balance(fit)['balanced'])

For installation variants, the full set of estimators (CBGPS, CBMSM, npCBPS,
hdCBPS, CBIV, oCBPS), and the paper-replication notebooks, see
:doc:`installation`, :doc:`quickstart`, :doc:`advanced_usage`, and
:doc:`tutorials/index`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   :hidden:

   theory
   advanced_usage
   implementation_notes

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :hidden:

   GitHub Repository <https://github.com/gorgeousfish/CBPS-py>
   Issue Tracker <https://github.com/gorgeousfish/CBPS-py/issues>
   Documentation <https://cbps-python.readthedocs.io>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

When using this software in academic publications, see ``CITATION.cff`` in
the source repository or the *Citation* section of ``README.md`` for the
recommended software citation entries (APA and BibTeX).

License
=======

This project is licensed under the GNU Affero General Public License v3.0 or later. See the LICENSE file for details.

Copyright (c) 2025-2026 Xuanyu Cai, Wenli Xu