"""
Nonparametric CBPS Module.

This subpackage implements the nonparametric covariate balancing generalized
propensity score (npCBGPS) estimator from Section 3.3 of Fong, Hazlett, and
Imai (2018). The function is named :func:`npCBPS` for API consistency with
the parametric version.

Unlike parametric CBPS, this approach does not require specifying a functional
form for the propensity score. Instead, it directly estimates inverse
probability weights by maximizing the empirical likelihood subject to
covariate balance constraints.

Main API
--------
:func:`npCBPS`
    Estimate nonparametric covariate balancing weights from a formula
    and DataFrame.
:class:`NPCBPSResults`
    Container for estimation results including weights and diagnostics.

Submodules
----------
:mod:`taylor_approx`
    Modified logarithm with Taylor approximation for numerical stability.
:mod:`cholesky_whitening`
    Covariate whitening via Cholesky decomposition.
:mod:`empirical_likelihood`
    Dual optimization routines for empirical likelihood.

When to Use npCBPS
------------------
- When you are uncertain about the correct propensity score model specification.
- When you prefer a nonparametric approach that directly targets balance.
- When computational cost is acceptable (npCBPS is slower than parametric CBPS).

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. The Annals of Applied Statistics, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101
"""

from .npcbps import npCBPS, NPCBPSResults
from .taylor_approx import llog, llogp
from .cholesky_whitening import cholesky_whitening
from .empirical_likelihood import get_w, log_post

__all__ = [
    'npCBPS',
    'NPCBPSResults',
    'llog',
    'llogp',
    'cholesky_whitening',
    'get_w',
    'log_post'
]
