"""
Statistical Inference Test Suite
================================

This package contains tests for the statistical inference components of CBPS,
including variance estimation, confidence intervals, and hypothesis testing.

Test Modules:
    - test_paper_formulas.py: Verification of paper formulas for J-statistic,
      double robustness, and asymptotic properties (INF-001 to INF-015)
    - test_asyvar.py: Asymptotic variance estimation for ATE (ASYVAR-001 to ASYVAR-025)
    - test_vcov_outcome.py: Variance adjustment for weighted outcome regression
      (VCOV-001 to VCOV-020)

References:
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society, Series B 76(1), 243-263.

    Fan, Q., Hsu, Y.-C., Lieli, R. P., and Zhang, Y. (2022). Optimal Covariate
    Balancing Conditions in Propensity Score Estimation. JBES 40(4), 1468-1482.

    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
    Score for a Continuous Treatment. Annals of Applied Statistics 12(1), 156-177.
"""
