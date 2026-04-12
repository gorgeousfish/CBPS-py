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

- **Unified Framework**: Single interface supporting all treatment types with intelligent detection
- **Advanced Estimation**: Both exactly-identified and over-identified GMM with multiple optimization strategies
- **Numerical Robustness**: Enhanced stability through SVD-based operations and comprehensive edge case handling
- **Comprehensive Diagnostics**: Built-in tools for balance assessment, model validation, and sensitivity analysis
- **Asymptotic Inference**: Valid variance estimation and confidence intervals for all treatment effect estimands

Implementation Highlights
-------------------------

The implementation incorporates several numerical and usability enhancements:

- SVD-based matrix inversion for numerical stability in ill-conditioned problems
- Sophisticated convergence diagnostics with informative failure messages
- Automatic detection and handling of separation and overlap violations
- Intelligent treatment type recognition based on data characteristics
- Graceful degradation when optional dependencies are unavailable

Quick Start
-----------

Installation
~~~~~~~~~~~~

Basic installation from PyPI:

.. code-block:: bash

   pip install cbps-python

Full installation with all features:

.. code-block:: bash

   pip install 'cbps-python[all]'

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import cbps
   import pandas as pd

   # Load example dataset
   from cbps.datasets import load_lalonde
   data = load_lalonde(dehejia_wahba_only=True)

   # Estimate CBPS for ATT estimation
   fit = cbps.CBPS(
       formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
       data=data,
       att=1,                    # Target estimand: ATT
       method='over',            # Over-identified GMM (recommended)
       two_step=True,           # Two-step estimator for efficiency
       standardize=True         # Standardize weights for stability
   )

   # Examine results
   print(fit.summary())         # Coefficient estimates and diagnostics

   # Assess covariate balance
   balance_stats = cbps.balance(fit)
   print(balance_stats['balanced'])  # Weighted balance statistics

   # Visualize results (requires matplotlib)
   cbps.plot_cbps(fit)

For comprehensive examples covering all treatment types and advanced features, see :doc:`quickstart`.

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
   references

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :hidden:

   GitHub Repository <https://github.com/gorgeousfish/cbps-python>
   Issue Tracker <https://github.com/gorgeousfish/cbps-python/issues>
   Documentation <https://cbps-python.readthedocs.io>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

When using this software in academic publications, please cite both the methodological papers and the software package.

**Core Methodology**

Imai, K. and Ratkovic, M. (2014). "Covariate Balancing Propensity Score." *Journal of the Royal Statistical Society, Series B* 76(1), 243–263. https://doi.org/10.1111/rssb.12027

**Methodological Extensions**

- **Continuous treatments**: Fong, C., Hazlett, C., and Imai, K. (2018). "Covariate balancing propensity score for a continuous treatment: Application to the efficacy of political advertisements." *The Annals of Applied Statistics* 12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

- **High-dimensional settings**: Ning, Y., Peng, S., and Imai, K. (2020). "Robust estimation of causal effects via a high-dimensional covariate balancing propensity score." *Biometrika* 107(3), 533–554. https://doi.org/10.1093/biomet/asaa020

- **Nonparametric estimation**: Fong, C., Hazlett, C., and Imai, K. (2018). "Covariate balancing propensity score for general treatment regimes." *Journal of the American Statistical Association* 113(523), 1316–1329. https://doi.org/10.1080/01621459.2017.1385465

- **Marginal structural models**: Imai, K. and Ratkovic, M. (2015). "Robust estimation of inverse probability weights for marginal structural models." *Journal of the American Statistical Association* 110(511), 1013–1023. https://doi.org/10.1080/01621459.2014.956872

- **Optimal CBPS**: Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022). "Optimal covariate balancing conditions in propensity score estimation." *Journal of Business & Economic Statistics* 41(1), 97–110. https://doi.org/10.1080/07350015.2021.2002159

- **Instrumental variables**: Fong, C. (2018). "Covariate balancing propensity scores for instrumental variable applications." Unpublished manuscript.

**Software Implementation**

Cai, X. and Xu, W. (2025). *cbps-python: Python Implementation of Covariate Balancing Propensity Score*. Version 0.1.0. https://github.com/gorgeousfish/cbps-python

License
=======

This project is licensed under the GNU Affero General Public License v3.0 or later. See the LICENSE file for details.

Copyright (c) 2025-2026 Cai Xuanyu, Xu Wenli