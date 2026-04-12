References
==========

This page lists the key academic papers and resources related to the CBPS methodology.

.. note::
   **Accessing DOI Links:** Some publisher websites may block automated checkers with 403 errors,
   but all DOI links are valid and accessible through web browsers. If you encounter access issues:

   - Try accessing the DOI link directly in your browser
   - Use your institution's library access
   - Search the paper title on `Google Scholar <https://scholar.google.com/>`_
   - Many papers are available as preprints on author websites or arXiv

Core Papers
-----------

Binary and Multi-Valued Treatment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Imai, K., & Ratkovic, M. (2014).** Covariate balancing propensity score. 
*Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 76(1), 243-263.

- **DOI:** `10.1111/rssb.12027 <https://doi.org/10.1111/rssb.12027>`_
- **Topics:** Binary treatment, multi-valued treatment, GMM estimation, covariate balance
- **Key Contributions:**
  
  - Introduced CBPS methodology
  - Dual optimization: prediction + balance
  - Just-identified and over-identified GMM
  - Asymptotic theory and variance estimation

Marginal Structural Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Imai, K., & Ratkovic, M. (2015).** Robust estimation of inverse probability weights for marginal structural models.
*Journal of the American Statistical Association*, 110(511), 1013-1023.

- **DOI:** `10.1080/01621459.2014.956872 <https://doi.org/10.1080/01621459.2014.956872>`_
- **Topics:** Longitudinal data, time-varying treatments, MSM weights
- **Key Contributions:**
  
  - Extended CBPS to longitudinal settings
  - Time-varying and time-invariant parameter models
  - Robust MSM weight estimation
  - Balance diagnostics for panel data

Continuous Treatment
^^^^^^^^^^^^^^^^^^^^

**Fong, C., Hazlett, C., & Imai, K. (2018).** Covariate balancing propensity score for a continuous treatment: 
Application to the efficacy of political advertisements.
*The Annals of Applied Statistics*, 12(1), 156-177.

- **DOI:** `10.1214/17-AOAS1101 <https://doi.org/10.1214/17-AOAS1101>`_
- **Topics:** Continuous treatment, generalized propensity score, dose-response
- **Key Contributions:**
  
  - Generalized propensity score (GPS) for continuous treatments
  - Correlation-based balance metrics
  - F-statistic for overall balance (target < 1e-4)
  - Application to political advertising data

High-Dimensional CBPS
^^^^^^^^^^^^^^^^^^^^^^

**Ning, Y., Peng, S., & Imai, K. (2020).** Robust estimation of causal effects via a high-dimensional covariate balancing propensity score.
*Biometrika*, 107(3), 533-554.

- **DOI:** `10.1093/biomet/asaa020 <https://doi.org/10.1093/biomet/asaa020>`_
- **Topics:** High-dimensional covariates, LASSO variable selection, p >> n
- **Key Contributions:**
  
  - LASSO-based variable selection for CBPS
  - Handles p ≈ n and p >> n settings
  - Cross-validation for lambda selection
  - Theoretical guarantees for sparse models

Instrumental Variables
^^^^^^^^^^^^^^^^^^^^^^^

**Fong, C. (2018).** Robust and efficient estimation of causal effects with calibrated covariate balance.
*Unpublished manuscript*.

- **Topics:** Instrumental variables, noncompliance, CACE estimation
- **Key Contributions:**
  
  - CBPS for instrumental variable settings
  - Two-sided and one-sided noncompliance
  - MLE and GMM estimation methods
  - Complier Average Causal Effect (CACE)

Related Methodology
-------------------

Propensity Score Methods
^^^^^^^^^^^^^^^^^^^^^^^^

**Rosenbaum, P. R., & Rubin, D. B. (1983).** The central role of the propensity score in observational studies for causal effects.
*Biometrika*, 70(1), 41-55.

- **DOI:** `10.1093/biomet/70.1.41 <https://doi.org/10.1093/biomet/70.1.41>`_
- **Topics:** Propensity score, observational studies, causal inference

**Hirano, K., & Imbens, G. W. (2004).** The propensity score with continuous treatments.
In *Applied Bayesian modeling and causal inference from incomplete-data perspectives* (pp. 73-84).

- **Topics:** Continuous treatment, generalized propensity score

Covariate Balance
^^^^^^^^^^^^^^^^^

**Austin, P. C. (2011).** An introduction to propensity score methods for reducing the effects of confounding in observational studies.
*Multivariate Behavioral Research*, 46(3), 399-424.

- **DOI:** `10.1080/00273171.2011.568786 <https://doi.org/10.1080/00273171.2011.568786>`_
- **Topics:** Propensity score matching, balance diagnostics, standardized mean differences

**Stuart, E. A. (2010).** Matching methods for causal inference: A review and a look forward.
*Statistical Science*, 25(1), 1-21.

- **DOI:** `10.1214/09-STS313 <https://doi.org/10.1214/09-STS313>`_
- **Topics:** Matching methods, balance assessment, sensitivity analysis

Software and Implementation
----------------------------

Python Package
^^^^^^^^^^^^^^

**CBPS Python package:**

- **GitHub:** https://github.com/gorgeousfish/cbps-python
- **Version:** 0.1.0
- **Authors:** Cai Xuanyu, Xu Wenli
- **License:** AGPL-3.0-or-later
- **Features:** Comprehensive CBPS implementation with enhanced numerical stability

Datasets
--------

LaLonde Dataset
^^^^^^^^^^^^^^^

**LaLonde, R. J. (1986).** Evaluating the econometric evaluations of training programs with experimental data.
*The American Economic Review*, 76(4), 604-620.

- **Topics:** Job training program evaluation, experimental vs. observational data
- **Used in:** CBPS tutorials and examples

**Dehejia, R. H., & Wahba, S. (1999).** Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs.
*Journal of the American Statistical Association*, 94(448), 1053-1062.

- **DOI:** `10.1080/01621459.1999.10473858 <https://doi.org/10.1080/01621459.1999.10473858>`_
- **Topics:** Propensity score matching, LaLonde data reanalysis

Blackwell Dataset
^^^^^^^^^^^^^^^^^

**Blackwell, M. (2013).** A framework for dynamic causal inference in political science.
*American Journal of Political Science*, 57(2), 504-520.

- **DOI:** `10.1111/j.1540-5907.2012.00626.x <https://doi.org/10.1111/j.1540-5907.2012.00626.x>`_
- **Topics:** Longitudinal data, time-varying treatments, political science applications
- **Used in:** CBMSM tutorials and examples

Additional Resources
--------------------

Online Resources
^^^^^^^^^^^^^^^^

- **Kosuke Imai's Website:** `https://imai.fas.harvard.edu/ <https://imai.fas.harvard.edu/>`_
- **Causal Inference Book (Imbens & Rubin):** `https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB <https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB>`_

Citation Guidelines
-------------------

When using this Python package in your research, please cite:

1. **The relevant methodology paper(s)** from the Core Papers section above

2. **This Python package:**

   Cai, X., & Xu, W. (2025). CBPS Python: Python implementation of Covariate Balancing Propensity Score.
   Version 0.1.0. https://github.com/gorgeousfish/cbps-python

BibTeX Entries
--------------

.. code-block:: bibtex

   @article{imai2014cbps,
     title={Covariate balancing propensity score},
     author={Imai, Kosuke and Ratkovic, Marc},
     journal={Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
     volume={76},
     number={1},
     pages={243--263},
     year={2014},
     doi={10.1111/rssb.12027}
   }

   @article{imai2015msm,
     title={Robust estimation of inverse probability weights for marginal structural models},
     author={Imai, Kosuke and Ratkovic, Marc},
     journal={Journal of the American Statistical Association},
     volume={110},
     number={511},
     pages={1013--1023},
     year={2015},
     doi={10.1080/01621459.2014.956872}
   }

   @article{fong2018continuous,
     title={Covariate balancing propensity score for a continuous treatment: Application to the efficacy of political advertisements},
     author={Fong, Christian and Hazlett, Chad and Imai, Kosuke},
     journal={The Annals of Applied Statistics},
     volume={12},
     number={1},
     pages={156--177},
     year={2018},
     doi={10.1214/17-AOAS1101}
   }

   @article{ning2020highdim,
     title={Robust estimation of causal effects via a high-dimensional covariate balancing propensity score},
     author={Ning, Yang and Peng, Sida and Imai, Kosuke},
     journal={Biometrika},
     volume={107},
     number={3},
     pages={533--554},
     year={2020},
     doi={10.1093/biomet/asaa020}
   }

