"""
Marginal Structural Models (MSM) Module.

This module implements the Covariate Balancing Propensity Score (CBPS)
methodology for marginal structural models, as developed by Imai and
Ratkovic (2015). MSMs enable robust causal inference with time-varying
treatments and confounders in longitudinal settings.

Functions
---------
CBMSM
    Formula-based interface for MSM weight estimation.
cbmsm_fit
    Matrix interface for advanced users.

Classes
-------
CBMSMResults
    Container for fitted weights, coefficients, and diagnostics.

Notes
-----
The module estimates inverse probability weights by solving a GMM problem
where moment conditions are derived from the covariate balancing property
of MSM weights. Key features include:

- Time-invariant or time-varying propensity score coefficients
- Stabilized weights P(T)/P(T|X) for variance reduction
- Low-rank covariance approximation for computational efficiency
- Orthogonal moment conditions based on 2^J factorial design framework

References
----------
Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
weights for marginal structural models. Journal of the American Statistical
Association, 110(511), 1013-1023. https://doi.org/10.1080/01621459.2014.956872
"""

from cbps.msm.cbmsm import CBMSM, cbmsm_fit, CBMSMResults

__all__ = [
    "CBMSM",
    "cbmsm_fit",
    "CBMSMResults",
]
