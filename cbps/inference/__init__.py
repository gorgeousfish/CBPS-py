"""
Statistical Inference for CBPS Estimators
==========================================

This module provides variance estimation and confidence interval construction
for causal effect estimates obtained through CBPS weighting. The inference
methods properly account for estimation uncertainty in propensity score
parameters within the GMM framework.

Submodules
----------
asyvar
    Asymptotic variance estimation for binary treatment ATE using the
    sandwich variance estimator. Implements both the semiparametric
    efficiency bound (oCBPS) and the full sandwich formula (CBPS) from
    Fan et al. (2022).

vcov_outcome
    Variance-covariance adjustment for weighted outcome regressions with
    continuous treatments. Propagates weight estimation uncertainty through
    standard error calculations following Fong, Hazlett, and Imai (2018),
    Section 3.2.

References
----------
Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022).
    Optimal covariate balancing conditions in propensity score estimation.
    Journal of Business & Economic Statistics, 41(1), 97-110.
    https://doi.org/10.1080/07350015.2021.2002159

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment. The Annals of Applied Statistics,
    12(1), 156-177. https://doi.org/10.1214/17-AOAS1101

Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
"""

from cbps.inference.asyvar import asy_var
from cbps.inference.vcov_outcome import vcov_outcome

__all__ = ['asy_var', 'vcov_outcome']
