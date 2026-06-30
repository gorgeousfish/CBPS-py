---
title: 'cbps: Covariate Balancing Propensity Score for Python'
tags:
  - Python
  - causal inference
  - propensity score
  - covariate balancing
  - observational studies
  - generalized method of moments
authors:
  - name: Xuanyu Cai
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Wenli Xu
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: "1"
affiliations:
  - name: Faculty of Data Science, City University of Macau, Macau SAR, China
    index: 1
date: 16 February 2026
bibliography: paper.bib
---

# Summary

In observational studies, researchers rely on the propensity score---the conditional probability of receiving treatment given observed covariates---to adjust for confounding and estimate causal effects. However, conventional propensity score methods are highly sensitive to model misspecification: even minor departures from the assumed model can lead to covariate imbalance and biased causal estimates. The Covariate Balancing Propensity Score (CBPS) method, proposed by @imai2014covariate, addresses this limitation through a Generalized Method of Moments (GMM) framework that simultaneously optimizes the predictive accuracy of the propensity score model and the balance of covariates between treatment groups. This approach has since been extended to continuous treatments [@fong2018covariate], longitudinal data via marginal structural models [@imai2015robust], high-dimensional settings [@ning2020robust], and optimal balancing conditions [@fan2022optimal].

`cbps` provides a complete Python implementation of the CBPS family of methods, covering all variants introduced in the five core papers. The package supports binary, multi-valued, continuous, longitudinal, and high-dimensional treatment variables, and offers nonparametric estimation, doubly robust estimation, instrumental variable extensions (CBIV), diagnostics, and visualization tools. `cbps` features both an R-style formula interface for ease of migration from R and a NumPy array interface for integration with the Python machine learning ecosystem. Built-in dataset loaders for classic causal inference datasets further support teaching and replication.


# Statement of Need

Since the seminal work of @rosenbaum1983central, the propensity score has become a cornerstone of causal inference in observational studies. By conditioning on or weighting by the propensity score, researchers can approximate the conditions of a randomized experiment using non-experimental data. Yet conventional propensity score methods suffer from a fundamental tension---the "propensity score tautology" [@imai2014covariate]: the very purpose of the propensity score is to balance covariates across treatment groups, but its estimation depends on correct model specification. When the model is misspecified, the resulting scores may fail to achieve balance and can even exacerbate bias. CBPS resolves this tension by jointly imposing score and balance conditions within a GMM framework.

Currently, the only complete implementation of the CBPS family resides in the R ecosystem. Meanwhile, Python has emerged as a leading language for causal inference research, with active projects such as DoWhy (https://github.com/py-why/dowhy), EconML, and CausalML forming a vibrant ecosystem. However, none of these Python tools fully implement the CBPS family: cbpys [@lal2024cbpys] provides binary-treatment ATT estimation via exponential tilting but does not cover the full GMM framework or other variants. This gap forces researchers who need CBPS to switch between R and Python, complicating their workflows and limiting the adoption of CBPS methods in the Python community.

`cbps` was developed to fill this gap. The package targets causal inference researchers, econometricians, epidemiologists, and political scientists who require robust causal estimation in observational studies. It provides a unified API covering all CBPS variants---from binary treatment to marginal structural models for longitudinal data [@robins2000marginal], high-dimensional settings, and optimal balancing conditions---so that researchers can apply the full CBPS methodology without leaving the Python ecosystem.


# State of the Field

The following table summarizes existing tools' support for CBPS family variants:

| Tool | Language | CBPS | CBGPS | CBMSM | hdCBPS | oCBPS | CBIV |
|------|----------|:----:|:-----:|:-----:|:------:|:-----:|:----:|
| CBPS [@fong2025cbps] | R | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| WeightIt [@greifer2025weightit] | R | ✓ | ✓ | Partial | ✗ | ✗ | ✗ |
| CBPS [@premik2017cbps] | Stata | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| psweight [@kranker2021improving] | Stata | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cbpys [@lal2024cbpys] | Python | Partial | ✗ | ✗ | ✗ | ✗ | ✗ |
| balance [@sarig2023balance] | Python | Partial | ✗ | ✗ | ✗ | ✗ | ✗ |
| **cbps** | **Python** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

In the R ecosystem, the CBPS package [@fong2025cbps] serves as the reference implementation covering all six variants. WeightIt [@greifer2025weightit] supports basic CBPS and CBGPS within a general weighting framework but lacks hdCBPS, oCBPS, and full CBMSM support. In Stata, @premik2017cbps implements binary-treatment CBPS within the GMM framework, and psweight [@kranker2021improving] provides propensity score weighting with weight stabilization; both are limited to binary treatments. In Python, cbpys [@lal2024cbpys] provides binary-treatment ATT estimation via exponential tilting but does not employ the full GMM framework, while the balance package [@sarig2023balance] includes only binary-treatment CBPS for survey non-response correction. Related methods such as entropy balancing [@hainmueller2012entropy] and the generalized propensity score [@hirano2003efficient] differ from CBPS in their theoretical foundations and estimation strategies.

We chose to release `cbps` as a standalone package rather than contributing to an existing project for four reasons: (1) the CBPS family's distinctive GMM optimization framework is architecturally incompatible with general-purpose causal inference libraries; (2) faithfully implementing all method variants from five core papers, including diagnostics, inference, and visualization, requires dedicated modular design; (3) numerical alignment with the R reference implementation demands independent, targeted testing infrastructure; and (4) existing Python causal inference libraries are architected around causal graph identification or heterogeneous effect estimation, making them unsuitable hosts for GMM-based propensity score methods.


# Software Design

The design of `cbps` is guided by three core decisions, each serving the goals of reproducibility and usability in academic software.

**Unified GMM framework.** All CBPS variants share the theoretical foundation of the Generalized Method of Moments [@hansen1982large]. For binary treatment, CBPS jointly solves the moment conditions:

$$\frac{1}{N}\sum_{i=1}^{N} g(T_i, X_i; \beta) = 0$$

where $g$ combines a score condition ensuring likelihood maximization with a balance condition ensuring weighted covariate balance. The resulting over-identified system can be assessed via Hansen's J-test [@hansen1982large]. We adopted this unified GMM framework---rather than implementing separate optimization routines for each method---to ensure consistency across the full spectrum: from basic CBPS [@imai2014covariate] to continuous-treatment CBGPS [@fong2018covariate], longitudinal CBMSM [@imai2015robust], high-dimensional hdCBPS [@ning2020robust], optimal balancing oCBPS [@fan2022optimal], doubly robust estimation [@tsiatis2007comment; @robins1999association], and empirical likelihood extensions [@owen2001empirical].

**Dual interface design.** The package offers both an R-style formula interface and a NumPy array interface. The formula interface lowers migration costs for R users, while the array interface facilitates integration with Python's machine learning ecosystem including scikit-learn.

**Modular architecture.** Because the mathematical foundations and optimization strategies differ substantially across CBPS variants, the codebase is organized into independent modules (core, msm, highdim, nonparametric, iv, among others) rather than a monolithic class hierarchy. This design enables independent development, testing, and maintenance of each module.


# Research Impact Statement

As a newly released package, `cbps` demonstrates its credible near-term research significance through three categories of evidence.

**Numerical validation.** The package's Monte Carlo validation tests cover the data-generating processes (DGPs) from all five core papers, using the exact simulation designs specified in the original publications. Bias, root mean squared error (RMSE), and convergence rates are verified against published results with tolerances based on Monte Carlo standard errors (3×MC SE). These tests run automatically in the continuous integration pipeline, guarding against regressions across releases.

**Software quality.** The package maintains a multi-layered testing infrastructure comprising over 2,000 test functions across 125 test files, covering unit tests, integration tests, Monte Carlo validation, and paper reproduction. Complete Sphinx API documentation is hosted on ReadTheDocs, and three Python replication scripts reproduce the core analyses of @imai2014covariate, @imai2015robust, and @fong2018covariate. Continuous integration via GitHub Actions covers multiple operating systems and Python versions.

**Reproducibility contributions.** `cbps` makes the CBPS family of methods---previously available only in R---directly accessible within the Python ecosystem. Four built-in dataset loaders---LaLonde employment training data [@lalonde1986evaluating], Blackwell longitudinal campaign data [@blackwell2013framework], continuous treatment simulation data, and npCBPS validation data---facilitate teaching and methodological comparison. The dual interface design and scikit-learn-compatible API further lower the barrier to adoption for researchers from diverse backgrounds.


# Acknowledgements

We thank Kosuke Imai, Marc Ratkovic, and Christian Fong for their foundational contributions to the CBPS methodology and R reference implementation. `cbps` relies on NumPy [@harris2020array], SciPy [@virtanen2020scipy], statsmodels [@seabold2010statsmodels], and pandas [@mckinney2010data].

# References
