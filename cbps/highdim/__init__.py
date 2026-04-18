"""
High-Dimensional Covariate Balancing Propensity Score (hdCBPS)
==============================================================

This module implements the High-Dimensional Covariate Balancing Propensity
Score (hdCBPS) methodology for robust causal inference in settings where
the number of covariates may exceed the sample size (p >> n).

Algorithm Overview
------------------
The hdCBPS algorithm proceeds in four steps as described in Ning et al. (2020):

1. **Propensity Score Estimation** (Equation 5): Fit penalized logistic
   regression (LASSO) to obtain initial propensity score coefficients.

2. **Outcome Model Estimation** (Equation 6): Fit penalized regression
   (LASSO) to estimate outcome model coefficients separately for treatment
   and control groups. This implementation uses unweighted LASSO (w_2=1).

3. **Covariate Balancing** (Equation 7): Calibrate the propensity score by
   minimizing the GMM objective to balance covariates selected in Step 2.
   This achieves the weak covariate balancing property (Equation 9).

4. **Treatment Effect Estimation**: Compute ATE/ATT using the Horvitz-Thompson
   estimator with calibrated propensity scores. Standard errors are computed
   using the sandwich variance estimator (Equation 11).

Key Features
------------
- **Double Robustness**: Consistent and asymptotically normal when either
  the propensity score model or outcome model is correctly specified.
- **Sample Boundedness**: Estimated ATE lies within the range of observed
  outcomes, ensuring stable estimates.
- **Semiparametric Efficiency**: Achieves the efficiency bound when both
  models are correctly specified.
- **High-Dimensional Support**: Handles p >> n through L1 regularization.

Requirements
------------
- **glmnetforpython**: Required for LASSO regularization with Fortran backend.
- numpy, scipy: Numerical computations.
- pandas: Data handling.

References
----------
Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score. Biometrika,
107(3), 533-554. https://doi.org/10.1093/biomet/asaa020

See Also
--------
cbps.CBPS : Standard CBPS for low-dimensional settings; automatically
    dispatches to the continuous-treatment branch when the treatment
    variable is continuous.
cbps.npCBPS : Nonparametric CBPS for continuous treatments (empirical
    likelihood).
"""

__all__ = []

# Import hdCBPS function when glmnetforpython is available
try:
    from .hdcbps import hdCBPS, HDCBPSResults
    from .lasso_utils import cv_glmnet, select_variables
    __all__.extend(['hdCBPS', 'HDCBPSResults', 'cv_glmnet', 'select_variables'])
except ImportError:
    import warnings
    warnings.warn(
        "hdCBPS requires glmnetforpython. Install with: "
        "pip install glmnetforpython",
        ImportWarning
    )

# Weight functions (available regardless of glmnetforpython)
from .weight_funcs import (
    ate_wt_func,
    ate_wt_nl_func,
    att_wt_func,
    att_wt_nl_func
)

__all__.extend([
    'ate_wt_func',
    'ate_wt_nl_func',
    'att_wt_func',
    'att_wt_nl_func'
])