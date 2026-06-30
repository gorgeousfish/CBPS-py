"""
Dataset Loaders Module
======================

Provides functions for loading standard causal inference datasets commonly used
in propensity score analysis and causal inference research.

All datasets are bundled with the package and require no external downloads.

Available Datasets
------------------

**Cross-sectional Studies**:

- :func:`load_lalonde`: LaLonde (1986) NSW job training program evaluation data
- :func:`load_lalonde_psid_combined`: Combined NSW experimental and PSID control data

**Longitudinal Studies**:

- :func:`load_blackwell`: Blackwell (2013) longitudinal political campaign data

**Continuous Treatment**:

- :func:`load_continuous_simulation`: Fong et al. (2018) DGP simulation data
- :func:`load_political_ads`: Political advertising efficacy data (Urban & Niebler 2014)

**Nonparametric CBPS**:

- :func:`load_npcbps_continuous_sim`: Simulation data for npCBPS validation

Datasets Not Included
--------------------------------------

- **Political Socialization Panel Study** (Kam & Palmer 2008):
  Used in Ning, Peng, and Imai (2020) for high-dimensional CBPS empirical
  analysis. This dataset is not included due to copyright restrictions.
  It can be obtained from ICPSR (Study #4037):
  https://www.icpsr.umich.edu/web/ICPSR/studies/4037

References
----------
LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
programs with experimental data. *American Economic Review*, 76(4), 604-620.

Blackwell, M. (2013). A framework for dynamic causal inference in political
science. *American Journal of Political Science*, 57(2), 504-520.

Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.

Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score. *Biometrika*,
107(3), 533-554.

Kam, C. D. and Palmer, C. L. (2008). Reconsidering the effects of education
on political participation. *Journal of Politics*, 70(3), 612-631.
"""

from cbps.datasets.lalonde import (
    load_lalonde,
    load_lalonde_psid_combined,
)
from cbps.datasets.blackwell import load_blackwell
from cbps.datasets.continuous import (
    load_continuous_simulation,
    load_political_ads,
)
from cbps.datasets.npcbps_sim import load_npcbps_continuous_sim

__all__ = [
    "load_lalonde",
    "load_lalonde_psid_combined",
    "load_blackwell",
    "load_continuous_simulation",
    "load_political_ads",
    "load_npcbps_continuous_sim",
]
