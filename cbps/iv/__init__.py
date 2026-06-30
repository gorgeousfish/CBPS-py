"""
Covariate Balancing Propensity Score for Instrumental Variables (CBIV)
=======================================================================

This module implements the Covariate Balancing Propensity Score (CBPS) methodology
for instrumental variable (IV) settings with treatment noncompliance. CBIV estimates
compliance type probabilities using generalized method of moments (GMM), simultaneously
optimizing covariate balance among compliers and prediction of treatment assignment.

In IV settings with noncompliance, units can be classified into principal strata
based on their potential treatment status under different instrument values:

- **Compliers**: Units who take treatment when encouraged (Z=1) and do not
  take treatment when not encouraged (Z=0).
- **Always-takers**: Units who take treatment regardless of encouragement.
- **Never-takers**: Units who do not take treatment regardless of encouragement.

The local average treatment effect (LATE) is identified among compliers. CBIV
provides weights (inverse of complier probability) that can be used for
downstream causal effect estimation.

Key Components
--------------
- ``CBIV``: Main function for estimating compliance type propensity scores
- ``CBIVResults``: Result container with fitted compliance probabilities and weights
- ``CBIVNumericalWarning``: Warning class for numerical stability issues

Noncompliance Models
--------------------
- **Two-sided noncompliance** (default): Models compliers, always-takers, and
  never-takers using multinomial logistic regression with three compliance types.
- **One-sided noncompliance**: Models compliers and never-takers only (assumes
  no always-takers), using binary logistic regression.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
Journal of the Royal Statistical Society: Series B (Statistical Methodology),
76(1), 243-263. https://doi.org/10.1111/rssb.12027

Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of
Causal Effects Using Instrumental Variables. Journal of the American
Statistical Association, 91(434), 444-455. https://doi.org/10.1080/01621459.1996.10476902
"""

from .cbiv import CBIV, CBIVResults, CBIVNumericalWarning

__all__ = ["CBIV", "CBIVResults", "CBIVNumericalWarning"]
