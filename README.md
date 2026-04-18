# cbps

**Covariate Balancing Propensity Score for Python**

<!-- PyPI badge disabled until cbps-python is published to PyPI.
     Uncomment the next line after the first PyPI release:
[![PyPI version](https://img.shields.io/pypi/v/cbps-python.svg)](https://pypi.org/project/cbps-python/)
-->
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/cbps-python/badge/?version=latest)](https://cbps-python.readthedocs.io/en/latest/?badge=latest)
[![CITATION.cff](https://img.shields.io/badge/citation-cff-blue.svg)](CITATION.cff)

![cbps](image/image.jpg)

## Overview

Traditional propensity score estimation faces a fundamental challenge known as the **propensity score tautology**: researchers iterate between fitting models and checking covariate balance, yet the estimated propensity score is considered appropriate only if it achieves balance. Even slight model misspecification can result in substantial bias in treatment effect estimates.

**CBPS solves this problem** by directly incorporating covariate balance conditions into propensity score estimation through the Generalized Method of Moments (GMM) framework. Instead of solely maximizing likelihood, CBPS simultaneously optimizes:

1. **Predictive accuracy** of treatment assignment (score condition)
2. **Covariate balance** between treatment groups (balance condition)

This dual optimization yields propensity scores that are more robust to model misspecification while maintaining theoretical guarantees for consistent causal effect estimation.

## Features

- **Binary & Multi-valued Treatments** - Standard CBPS for discrete treatments with ATE/ATT estimation
- **Continuous Treatments** - Generalized propensity scores (CBGPS) for dose-response analysis
- **Longitudinal Data** - Marginal structural models (CBMSM) for time-varying treatments
- **High-dimensional Settings** - Regularized estimation (hdCBPS) when covariates exceed sample size
- **Nonparametric Methods** - Empirical likelihood approach (npCBPS) without distributional assumptions
- **Doubly Robust Estimation** - Optimal CBPS (oCBPS) with improved efficiency
- **Instrumental Variables** - CBPS for IV settings with treatment noncompliance
- **Model Diagnostics** - Hansen's J-test, balance statistics, and visualization tools
- **scikit-learn Integration** - `cbps.sklearn.CBPSEstimator` is a `BaseEstimator`/`ClassifierMixin` wrapper usable inside `Pipeline` (see [scikit-learn Integration](#scikit-learn-integration) for details and current limitations)
- **R Package Compatibility** - Numerical accuracy within ±1e-6 of CBPS R package v0.23

## Installation

### From GitHub

> **Note:** `cbps-python` is not yet published to PyPI. For now, install
> directly from the GitHub repository. Once the package is released,
> `pip install cbps-python` will become the default.

```bash
pip install "cbps-python @ git+https://github.com/gorgeousfish/CBPS-py.git"
```

### With Optional Dependencies

```bash
# High-dimensional CBPS support
pip install "cbps-python[hdcbps] @ git+https://github.com/gorgeousfish/CBPS-py.git"

# Visualization tools
pip install "cbps-python[plots] @ git+https://github.com/gorgeousfish/CBPS-py.git"

# scikit-learn integration
pip install "cbps-python[sklearn] @ git+https://github.com/gorgeousfish/CBPS-py.git"

# All features
pip install "cbps-python[all] @ git+https://github.com/gorgeousfish/CBPS-py.git"
```

### Development Installation

```bash
git clone https://github.com/gorgeousfish/CBPS-py.git
cd CBPS-py
pip install -e ".[dev]"
```

> **Note for Apple Silicon Users**: `hdCBPS` requires the optional
> `glmnetforpython` package, which ships a Fortran extension. On most
> platforms `pip install glmnetforpython` pulls a pre-built wheel. If the
> wheel is unavailable and compilation fails on your Mac, install a Fortran
> compiler first and re-run pip:
> ```bash
> brew install gcc               # provides gfortran
> export FC=gfortran
> pip install glmnetforpython
> ```
> If PyPI still does not provide a suitable wheel, install from source:
> ```bash
> git clone https://github.com/thierrymoudiki/glmnetforpython.git
> pip install -e ./glmnetforpython
> ```
> See the [`Installation` section of the docs](https://cbps-python.readthedocs.io/en/latest/installation.html#requirements-for-hdcbps) for the same steps with extra troubleshooting tips.

## Quick Start

Replicating the LaLonde (1986) analysis from Imai and Ratkovic (2014, Section 3.2):

```python
from cbps import CBPS, balance
from cbps.datasets import load_lalonde

# Load LaLonde (1986) NSW job training data (445 observations)
data = load_lalonde()

# Over-identified CBPS for ATE estimation (Imai & Ratkovic, 2014)
fit = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data,
    att=0,           # Average Treatment Effect
    method='over'    # Over-identified GMM (combines score + balance conditions)
)

# View results
print(fit.summary())
# Coefficients:
#                    Estimate   Std. Error    z value     Pr(>|z|)
# (Intercept)        0.740178     0.002239    330.625    0.000e+00 ***
# age                0.007589     0.085471      0.089    9.292e-01
# educ              -0.048260     0.060982     -0.791    4.287e-01
# black             -0.199980     0.000624   -320.226    0.000e+00 ***
# hisp              -0.756848     0.000051 -14837.961    0.000e+00 ***
# married            0.103707     0.000092   1129.734    0.000e+00 ***
# nodegr            -0.676270     0.000224  -3013.458    0.000e+00 ***
# re74              -0.000022     0.124988     -0.000    9.999e-01
# re75               0.000032     0.128498      0.000    9.998e-01
#
# J - statistic:  5.09e-03
# Log-Likelihood: -294.2790
#
# Diagnostics:
#   Converged:              Yes
#   Weight Summary:
#     Min:     0.0029    Max:     0.0090    Mean:   0.0045
#   Effective Sample Size:  420.7

# Check covariate balance improvement
bal = balance(fit)
print(bal['balanced'])   # Balance after CBPS weighting
print(bal['original'])   # Balance before weighting (baseline)
```

> **Key casing note.** `balance()` uses lower-case keys for the binary /
> multi-valued / continuous cases: ``'balanced'`` and ``'original'`` (binary /
> multi-valued) or ``'unweighted'`` (continuous). `CBMSM` instead returns
> capitalised keys — ``'Balanced'``, ``'Unweighted'`` and ``'StatBal'`` — to
> stay aligned with the R CBPS package.

## Method Family

The CBPS methodology has been extended to address a range of causal inference
challenges. All the variants below share a common GMM / empirical-likelihood
framework and are exposed through the `cbps` package:

```
                                 ┌──────────────────────┐
                                 │         CBPS         │
                                 │ Binary / Multi-valued│
                                 └──────────┬───────────┘
       ┌───────────────┬────────────────────┼──────────────────┬──────────────────┐
       ▼               ▼                    ▼                  ▼                  ▼
  ┌─────────┐    ┌──────────┐         ┌──────────┐       ┌──────────┐       ┌──────────┐
  │  CBGPS  │    │  CBMSM   │         │  hdCBPS  │       │   CBIV   │       │  oCBPS   │
  │Continu- │    │Longitud- │         │   High-  │       │Instrumen-│       │  Optimal │
  │ ous     │    │  inal    │         │dimension-│       │ tal Vari-│       │  / DR    │
  │Treatment│    │  / MSM   │         │    al    │       │   ables  │       │ ATE est. │
  └────┬────┘    └──────────┘         └──────────┘       └──────────┘       └──────────┘
       │
       ▼
  ┌─────────┐
  │ npCBPS  │
  │Nonparam.│   (empirical likelihood; supports continuous + discrete T)
  └─────────┘
```

`oCBPS` is reached through the same `CBPS()` entry point by passing
`baseline_formula` and `diff_formula` (Fan et al., 2022). `npCBPS()`
provides the nonparametric empirical-likelihood estimator of Fong, Hazlett
& Imai (2018), complementing the parametric continuous CBGPS that is also
exposed through `CBPS()`.

### Method Selection Guide

| Scenario | Method | Function | Key Reference |
|:---------|:-------|:---------|:--------------|
| Binary treatment, cross-sectional | CBPS | `CBPS()` | Imai & Ratkovic (2014) |
| Multi-valued treatment (3-4 levels) | CBPS | `CBPS()` | Imai & Ratkovic (2014) |
| Continuous treatment (parametric) | CBGPS | `CBPS()` | Fong et al. (2018) |
| Continuous treatment (nonparametric) | npCBPS | `npCBPS()` | Fong et al. (2018) |
| Longitudinal/panel data | CBMSM | `CBMSM()` | Imai & Ratkovic (2015) |
| High-dimensional (d >> n) | hdCBPS | `hdCBPS()` | Ning et al. (2020) |
| Doubly robust estimation | oCBPS | `CBPS(..., baseline_formula, diff_formula)` | Fan et al. (2022) |
| Instrumental variables | CBIV | `CBIV()` | Imai & Ratkovic (2014) |

---

## CBPS: Binary and Multi-valued Treatments

The core CBPS method estimates propensity scores for binary or multi-valued discrete treatments by solving GMM moment conditions that combine the score function with covariate balance constraints.

### When to Use

- Cross-sectional observational studies with binary treatment (0/1)
- Categorical treatments with 3-4 levels
- When model diagnostics (J-test) are desired

### Key Concepts

**For ATE estimation**, the balance condition ensures:

$$E\left[\frac{T \cdot X}{\pi(X)} - \frac{(1-T) \cdot X}{1-\pi(X)}\right] = 0$$

**For ATT estimation**, control observations are reweighted to match the treated:

$$E\left[T \cdot X - \frac{\pi(X)(1-T) \cdot X}{1-\pi(X)}\right] = 0$$

### Syntax

```python
CBPS(
    formula=None, data=None,
    treatment=None, covariates=None,       # array interface (alternative to formula/data)
    att=1, method='over', two_step=True, standardize=True,
    sample_weights=None,
    baseline_formula=None, diff_formula=None,   # enables oCBPS (Fan et al. 2022)
    iterations=1000, theoretical_exact=False,
    na_action='warn', verbose=0,
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `formula` | `None` | R-style formula, e.g. `'treatment ~ x1 + x2 + ...'`. Use with `data=` |
| `data` | `None` | pandas DataFrame containing the variables in `formula` |
| `treatment` / `covariates` | `None` | Array interface alternative to `formula` / `data` |
| `att` | `1` | Estimand: `0`=ATE, `1`=ATT (T=1 as treated), `2`=ATT (T=0 as treated) |
| `method` | `'over'` | `'exact'` (just-identified) or `'over'` (over-identified GMM) |
| `two_step` | `True` | Two-step GMM (`True`) or continuous updating (`False`) |
| `standardize` | `True` | Normalise weights to sum to 1 within each treatment group |
| `sample_weights` | `None` | Survey / sampling weights (length `n`); defaults to 1 everywhere |
| `baseline_formula` | `None` | Outcome baseline covariates K(X); triggers oCBPS when paired with `diff_formula` |
| `diff_formula` | `None` | Treatment-effect covariates L(X); used together with `baseline_formula` |
| `iterations` | `1000` | Maximum optimizer iterations |
| `theoretical_exact` | `False` | When `method='exact'`, use the direct equation solver instead of the balance-loss optimizer |
| `na_action` | `'warn'` | `'warn'` (drop + warn), `'fail'` (raise), or `'ignore'` (patsy default) |
| `verbose` | `0` | `0`=silent, `1`=basic, `2`=detailed diagnostics |

> **R-compatibility aliases.** For CBPS users coming from R, `CBPS()` also
> accepts the uppercase alias `ATT` (forwarded to `att`) and the R-style
> alias `twostep` (forwarded to `two_step`). Note that `CBMSM()` and `CBIV()`
> use `twostep` as their canonical parameter name — this alias convention
> applies only to `CBPS()`.
>
> **Interaction between `method` and `two_step`.** The two flags are
> independent: `method='over'` / `'exact'` selects the moment conditions
> (over-identified GMM vs. just-identified balance-only), while `two_step`
> controls whether an analytical gradient is used during balance-loss
> optimization (matching the R CBPS package, where `twostep=TRUE` enables
> the analytical gradient and `twostep=FALSE` falls back to numerical
> differentiation). Both combinations are valid.


### Example

Replicating the LaLonde analysis from Imai and Ratkovic (2014, Section 3.2):

```python
from cbps import CBPS, balance
from cbps.datasets import load_lalonde
from scipy import stats

data = load_lalonde()

# ATE estimation with over-identified GMM (CBPS2 in the paper)
fit_ate = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data,
    att=0,
    method='over'
)
print(fit_ate.summary())

# ATT estimation
fit_att = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data,
    att=1,
    method='over'
)

# Hansen's J-test for model specification (cf. Section 2.3)
# Paper reports J = 6.8 (df=22) for the linear specification
k = fit_ate.coefficients.shape[0]  # number of parameters
j_pvalue = 1 - stats.chi2.cdf(fit_ate.J, k)
print(f"J-statistic: {fit_ate.J:.4e}, df: {k}, p-value: {j_pvalue:.4f}")

# Covariate balance comparison
bal = balance(fit_ate)
print(bal['balanced'])   # Weighted balance (should show improvement)
print(bal['original'])   # Unweighted baseline
```

---

## CBGPS: Continuous Treatments

CBGPS extends the covariate balancing framework to continuous treatments by minimizing the weighted correlation between treatment and covariates.

### When to Use

- Continuous treatment variable (e.g., dosage, intensity, duration)
- Dose-response curve estimation
- When parametric assumptions about treatment distribution are acceptable

### Key Concept

For continuous treatment T with generalized propensity score f(T|X), the balance condition minimizes:

$$E\left[\frac{f(T)}{f(T|X)} \cdot T^* \cdot X^*\right] = 0$$

where T* and X* are centered and scaled versions of treatment and covariates.

### Syntax

Continuous treatment is auto-detected from the data type; the call uses the
same `CBPS()` entry point as the binary case, with continuous-specific
defaults highlighted below:

```python
CBPS(
    formula, data,
    att=0,              # only att=0 (ATE) is supported for continuous T
    method='over',      # 'over' (recommended) or 'exact'
    two_step=True,
    standardize=True,
    sample_weights=None,
    iterations=1000,
    na_action='warn',
    verbose=0,
)
```

The balance loss is built from Pearson correlations between $T^*$ and $X^*$
rather than standardized mean differences. Passing `att=1` or `att=2` with a
continuous treatment triggers a warning and is silently coerced back to `att=0`.

### Example

Replicating the empirical application from Fong, Hazlett, and Imai (2018, Section 5) — the effect of political advertising on campaign contributions using the Urban and Niebler (2014) dataset:

```python
import numpy as np
from cbps import CBPS
from cbps.datasets import load_political_ads

# Load Urban & Niebler (2014) political advertising data (n=16,265)
df_raw, meta = load_political_ads()

# Box-Cox transform treatment variable (lambda = -0.16, as in the paper)
work = df_raw.copy()
lam = meta["boxcox_lambda"]  # -0.16
work["T_bc"] = ((work["TotAds"].values + 1).clip(min=1e-10) ** lam - 1.0) / lam
work["logPop"] = np.log(work["TotalPop"].values.clip(min=1))
work["logInc"] = np.log(work["Inc"].values.clip(min=0) + 1)

# Add squared terms for non-binary covariates (paper p. 171)
work["logPop_sq"] = work["logPop"] ** 2
work["density_sq"] = work["density"] ** 2
work["logInc_sq"] = work["logInc"] ** 2
work["PercentHispanic_sq"] = work["PercentHispanic"] ** 2
work["PercentBlack_sq"] = work["PercentBlack"] ** 2
work["PercentOver65_sq"] = work["PercentOver65"] ** 2
work["per_collegegrads_sq"] = work["per_collegegrads"] ** 2

cov_cols = ["logPop", "density", "logInc", "PercentHispanic",
            "PercentBlack", "PercentOver65", "per_collegegrads", "CanCommute",
            "logPop_sq", "density_sq", "logInc_sq", "PercentHispanic_sq",
            "PercentBlack_sq", "PercentOver65_sq", "per_collegegrads_sq"]
work = work.dropna(subset=["T_bc"] + cov_cols).reset_index(drop=True)

# CBPS auto-detects continuous treatment
formula = "T_bc ~ " + " + ".join(cov_cols)
fit = CBPS(formula=formula, data=work, att=0, method="over")
print(fit.summary())
# Converged: True, J-statistic ≈ 0.0000
# CBGPS reduces covariate-treatment correlations
# (cf. Table 1 and Figure 3 in the paper, 15 covariates)
# Note: Over-identified GMM may fall back to exact method for this dataset
# due to high condition number of the covariate matrix
```

---

## npCBPS: Nonparametric CBPS

npCBPS uses empirical likelihood to estimate balancing weights without parametric assumptions about the generalized propensity score.

### When to Use

- Continuous treatment with unknown distribution
- When parametric assumptions may be violated
- Flexible, assumption-light estimation preferred

### Key Concept

Weights are chosen to maximize empirical likelihood subject to balance constraints:

$$\max \prod_{i=1}^n w_i \quad \text{s.t.} \quad \sum_i w_i X_i^* T_i^* = 0, \quad \sum_i w_i = n$$

### Syntax

```python
npCBPS(
    formula, data,
    na_action=None,    # None/'warn'/'fail'/'ignore' for missing values
    corprior=None,     # prior penalty; default 0.1/n (Fong et al., 2018)
    print_level=0,     # >0 prints optimization diagnostics
    seed=None,         # sets np.random.seed before optimisation
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `formula` | — | Treatment-and-covariates formula, e.g. `'T ~ x1 + x2'` |
| `data` | — | `pandas.DataFrame` containing the variables in `formula` |
| `na_action` | `'warn'` | Missing-value policy: `'warn'` (drop + warn), `'fail'` (raise), `'ignore'` (drop silently) |
| `corprior` | `None` (auto: `0.1/n`) | Prior penalty for the allowed weighted correlation η ~ N(0, σ² I). Larger = more tolerance for imbalance. Pass a float (e.g. `0.01`) to override the sample-size-adaptive default. |
| `print_level` | `0` | Verbosity of the empirical-likelihood optimiser |
| `seed` | `None` | Random seed forwarded to `np.random.seed` for reproducibility |

### Example

Using the Urban and Niebler (2014) data as in Fong et al. (2018, Section 5). For faster execution, we use a random subset and the 8 base covariates (without squared terms):

> **Note**: npCBPS uses iterative empirical likelihood optimization. With the full dataset (n=16,265) and 15 covariates, computation may take several minutes. The subset below (n=2,000) runs in ~30 seconds and demonstrates the same methodology.

```python
import numpy as np
from cbps import npCBPS
from cbps.datasets import load_political_ads

# Data preparation (same as CBGPS example)
df_raw, meta = load_political_ads()
work = df_raw.copy()
lam = meta["boxcox_lambda"]
work["T_bc"] = ((work["TotAds"].values + 1).clip(min=1e-10) ** lam - 1.0) / lam
work["logPop"] = np.log(work["TotalPop"].values.clip(min=1))
work["logInc"] = np.log(work["Inc"].values.clip(min=0) + 1)

cov_cols = ["logPop", "density", "logInc", "PercentHispanic",
            "PercentBlack", "PercentOver65", "per_collegegrads", "CanCommute"]
work = work.dropna(subset=["T_bc"] + cov_cols).reset_index(drop=True)

# Random subset for demonstration (full dataset also supported)
np.random.seed(42)
idx = np.random.choice(len(work), 2000, replace=False)
subset = work.iloc[idx].reset_index(drop=True)

formula = "T_bc ~ " + " + ".join(cov_cols)
fit = npCBPS(formula=formula, data=subset, corprior=0.01)
print(fit.summary())
# Converged: ✓ Yes
# Weighted Correlations: Mean=0.000101, all < 0.0002 (near-perfect balance)
# Weight Distribution: Min=0.533, Max=1.670, Mean=1.000
# Effective sample size: 1965.2
# Efficiency: 98.3%
```

---

## CBMSM: Marginal Structural Models

CBMSM extends CBPS to longitudinal settings with time-varying treatments and confounders, addressing the challenge that standard regression cannot properly adjust for time-dependent confounders affected by prior treatment.

### When to Use

- Panel/longitudinal data with repeated measurements
- Time-varying treatments
- Time-dependent confounders affected by past treatment

### Key Concept

At each time period, weights balance covariates across all potential future treatment sequences, conditional on observed treatment history:

$$E\left[w_i(\bar{T}_J, \bar{X}_J) \cdot X_{ij} \mid \bar{T}_{j-1}\right] = E[X_{ij} \mid \bar{T}_{j-1}]$$

### Syntax

```python
CBMSM(
    formula, id, time, data,
    type='MSM',              # 'MSM' or 'MultiBin'
    twostep=True,            # two-step GMM (recommended); set False for CU-GMM
    msm_variance='approx',   # 'approx' (low-rank, default) or 'full'
    time_vary=False,         # share coefficients across periods or not
    init='opt',              # 'opt' | 'glm' | 'CBPS'
    iterations=None,         # None = algorithm-specific default
)
```

| Parameter | Description |
|:----------|:------------|
| `formula` | Treatment model formula |
| `id` | Unit identifier column name or array |
| `time` | Time period column name or array |
| `data` | `pandas.DataFrame` containing treatment, covariates, id, and time |
| `type` | `'MSM'` (marginal structural) or `'MultiBin'` (multiple binary) |
| `twostep` | Two-step GMM (`True`, default) or continuous updating (`False`) |
| `msm_variance` | `'approx'` (default) or `'full'` variance estimation |
| `time_vary` | Whether **treatment model coefficients** vary across time (not whether covariates are time-varying). `False` shares one coefficient vector across periods; `True` estimates separate coefficients per period. |
| `init` | Initialization scheme: `'opt'` (default, pick best of GLM and CBPS), `'glm'`, or `'CBPS'` |
| `iterations` | Maximum optimizer iterations (default: algorithm-specific) |

### Example

Replicating the empirical application from Imai and Ratkovic (2015, Section 5) — the effect of negative campaign advertising on Democratic vote share using the Blackwell (2013) dataset:

```python
import numpy as np
import statsmodels.api as sm
from cbps import CBMSM
from cbps.datasets import load_blackwell

# Blackwell (2013): 114 U.S. Senate/gubernatorial races, J=5 weekly periods
data = load_blackwell()

# Full treatment model from Section 5 (1548 balancing conditions)
fit = CBMSM(
    formula='d.gone.neg ~ d.gone.neg.l1 + d.gone.neg.l2 + d.neg.frac.l3 + '
            'camp.length + deminc + base.poll + '
            'year.2002 + year.2004 + year.2006 + base.und + office',
    id='demName',
    time='time',
    data=data,
    type='MSM',
    time_vary=True,
    twostep=True,
    msm_variance='approx'
)
print(fit.summary())

# Estimate cumulative effect of negative advertising on vote share
# (cf. Table 3, CBPS-Approx column: Cumulative effect = -1.43)
outcome = data.loc[data["time"] == data["time"].min(), "demprcnt"].values
X_cum = sm.add_constant(fit.treat_cum.reshape(-1, 1))
m_cum = sm.WLS(outcome, X_cum, weights=fit.fitted_values).fit()
print(f"Cumulative effect: {m_cum.params[1]:.2f} (SE: {m_cum.bse[1]:.2f})")
# Expected output: Cumulative effect: -1.44 (SE: 0.43)
# Paper Table 3 reports -1.43 (0.43) for CBPS-Approx
# Note: Convergence may show False due to optimizer tolerance settings,
# but the estimates closely match the published results.
```

---

## hdCBPS: High-dimensional CBPS

hdCBPS handles settings where the number of covariates exceeds the sample size through LASSO regularization, while maintaining doubly robust properties.

### When to Use

- High-dimensional settings (d >> n)
- Many potential confounders with unknown importance
- When variable selection is needed
- Doubly robust estimation desired

### Key Concept

hdCBPS achieves the **weak covariate balancing property**:

$$\sum_i \left(\frac{T_i}{\tilde{\pi}_i} - 1\right) \alpha^{*\top} X_i \approx 0$$

where α* are outcome model coefficients. This enables root-n consistency even when d >> n.

### Syntax

```python
hdCBPS(
    formula, data, y,
    ATT=0,              # 0 = ATE, 1 = ATT
    iterations=1000,    # Nelder-Mead iterations in the GMM step
    method='linear',    # 'linear' | 'binomial' | 'poisson' (outcome family)
    seed=None,          # seeds cross-validation fold assignment
    na_action=None,     # 'warn' (default) | 'drop' | 'fail'
    verbose=0,          # reserved for future use
)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `formula` | — | Propensity score model formula (may contain many covariates) |
| `data` | — | `pandas.DataFrame` with treatment, covariates and outcome |
| `y` | — | Outcome variable name or array (required for the Step 2 LASSO) |
| `ATT` | `0` | Target estimand: `0` = ATE, `1` = ATT |
| `iterations` | `1000` | Maximum Nelder-Mead iterations used in Step 3 (GMM calibration) |
| `method` | `'linear'` | Outcome family: `'linear'` (Gaussian), `'binomial'` (logistic), `'poisson'` (count) |
| `seed` | `None` | Seeds cross-validation fold assignment for reproducible LASSO selection |
| `na_action` | `'warn'` | Missing-value policy: `'warn'` (drop + warn), `'drop'` (silent drop), `'fail'` (raise) |
| `verbose` | `0` | Verbosity level (reserved for future use) |

> **Reproducibility tip.** LASSO cross-validation fold assignment is drawn from
> `numpy.random`; pass `seed=<int>` (or call `np.random.seed(...)` before
> `hdCBPS`) to make Step 1 / Step 2 results reproducible.

### Example

```python
from cbps import hdCBPS
import pandas as pd
import numpy as np

# Simulate high-dimensional data (d > n)
np.random.seed(42)
n, d = 200, 300
X = np.random.normal(0, 1, (n, d))
beta_true = np.zeros(d)
beta_true[:5] = [1, 0.5, 0.25, 0.1, 0.05]  # Sparse true model
T = (X @ beta_true + np.random.normal(0, 1, n) > 0).astype(int)
Y = T + X[:, :3] @ [1, 0.5, 0.25] + np.random.normal(0, 1, n)

data = pd.DataFrame(X, columns=[f'X{i}' for i in range(d)])
data['T'] = T
data['Y'] = Y

# hdCBPS with automatic variable selection
import sys
sys.setrecursionlimit(5000)  # Needed for patsy with many covariates
fit = hdCBPS(
    formula='T ~ ' + ' + '.join([f'X{i}' for i in range(d)]),
    data=data,
    y='Y',
    ATT=0
)
print(f"ATE estimate: {fit.ATE:.4f}")
print(f"SE: {fit.s:.4f}")
print(f"Selected variables (treated): {fit.n_selected_treat}")
print(f"Selected variables (control): {fit.n_selected_control}")
```

> **Note**: Debug attributes (e.g., `result.debug_r_yhat1`) are now stored internally in `result._debug` dict. Direct attribute access still works but emits a `DeprecationWarning`. Use `result._debug['debug_r_yhat1']` instead.

---

## oCBPS: Optimal CBPS

Optimal CBPS improves upon standard CBPS by incorporating the outcome model structure, achieving doubly robust estimation with improved efficiency (Fan et al. 2022).

### When to Use

- When doubly robust estimation is desired
- Outcome model structure is known or estimable
- Maximum efficiency is important

### Key Concept

oCBPS solves for optimal balancing conditions that minimize asymptotic variance while maintaining consistency under either propensity score or outcome model misspecification. The optimal balancing function satisfies:

$$\alpha^\top f(X) = \pi(X) E[Y(0)|X] + (1-\pi(X)) E[Y(1)|X]$$

This gives greater weight to determinants of the mean potential outcome that is less likely to be realized.

### Syntax

Optimal CBPS is accessed through the `CBPS()` function by passing both
`baseline_formula` and `diff_formula` as keyword arguments (they are not the
third / fourth positional arguments of `CBPS()`):

```python
CBPS(formula, data, baseline_formula=..., diff_formula=..., att=0)
```

| Parameter | Description |
|:----------|:------------|
| `formula` | Propensity score model formula |
| `baseline_formula` | Outcome model baseline covariates (K(X)) |
| `diff_formula` | Treatment effect covariates (L(X)) |
| `att` | Must be 0 (ATE only for oCBPS) |

### Example

The baseline and diff formulas must satisfy the dimension constraint: m1 + m2 + 1 ≥ k, where m1 = number of baseline covariates, m2 = number of diff covariates, and k = number of propensity score parameters (including intercept).

```python
from cbps import CBPS
from cbps.datasets import load_lalonde

data = load_lalonde()

# Optimal CBPS with outcome model specification
# Propensity score model: 9 parameters (intercept + 8 covariates), k=9
# Baseline K(X): 8 covariates, m1=8
# Diff L(X): 2 covariates, m2=2
# Dimension check: m1 + m2 + 1 = 11 >= 9 = k ✓ (over-identified)
fit = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data,
    baseline_formula='~ age + educ + black + hisp + married + nodegr + re74 + re75',
    diff_formula='~ age + educ',
    att=0
)
print(fit.summary())
# Coefficients:
#                    Estimate   Std. Error    z value     Pr(>|z|)
# (Intercept)        1.175647     0.023190     50.696    0.000e+00 ***
# age                0.004057     0.137242      0.030    9.764e-01
# educ              -0.069238     0.357867     -0.193    8.466e-01
# black             -0.224812     0.017131    -13.123    0.000e+00 ***
# hisp              -0.856508     0.002234   -383.434    0.000e+00 ***
# married            0.165491     0.002664     62.114    0.000e+00 ***
# nodegr            -0.916259     0.005290   -173.202    0.000e+00 ***
# re74              -0.000035     0.000265     -0.130    8.964e-01
# re75               0.000068     0.000438      0.155    8.766e-01
#
# J - statistic:  3.94e-10
# Log-Likelihood: -293.6243
# Note: Convergence may show False due to optimizer tolerance,
# but the J-statistic near zero indicates excellent balance.
```

---

## CBIV: Instrumental Variables

CBIV extends the covariate balancing framework to instrumental variable settings where treatment noncompliance exists.

### When to Use

- Randomized experiments with noncompliance
- Observational studies with valid instruments
- Estimating local average treatment effects (LATE)

### Syntax

```python
# Formula interface (recommended)
CBIV(
    formula='treatment ~ covariates | instruments',
    data=df,
    method='over', twostep=True, twosided=True,
    iterations=1000,
    probs_min=1e-6,
    warn_clipping=True, clipping_warn_threshold=0.05,
    verbose=0,
)

# Matrix interface (equivalent; mutually exclusive with formula/data)
CBIV(Tr=treatment, Z=instrument, X=covariates, method='over', twosided=False)
```

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `formula` | `None` | IV formula: `'treat ~ x1 + x2 \| z'` (the `\|` separates covariates from instruments) |
| `data` | `None` | `pandas.DataFrame` (required with `formula`) |
| `Tr` | `None` | Binary treatment array (matrix interface) |
| `Z` | `None` | Binary instrument array (matrix interface) |
| `X` | `None` | Covariate matrix without intercept (matrix interface) |
| `method` | `'over'` | `'over'` (score + balance conditions), `'exact'` (balance only) or `'mle'` (score only) |
| `twostep` | `True` | Two-step GMM (faster, default) or continuous-updating GMM (`False`) |
| `twosided` | `True` | `True` for two-sided noncompliance (compliers + always-takers + never-takers); `False` for one-sided (no always-takers) |
| `iterations` | `1000` | Maximum optimizer iterations |
| `probs_min` | `1e-6` | Lower bound on compliance probabilities; probabilities are clipped to `[probs_min, 1 - probs_min]` |
| `warn_clipping` | `True` | Whether to warn when the share of clipped probabilities exceeds `clipping_warn_threshold` |
| `clipping_warn_threshold` | `0.05` | Clipping fraction above which the warning fires |
| `verbose` | `0` | `0` silent, `1` basic, `2` detailed optimisation diagnostics |

### Example

```python
import numpy as np
import pandas as pd
from cbps import CBIV

# Simulate IV data with one-sided noncompliance
np.random.seed(42)
n = 500
X = np.random.randn(n, 2)
Z = np.random.binomial(1, 0.5, n)  # Randomized instrument
p_comply = 1 / (1 + np.exp(-0.5 - 0.3 * X[:, 0]))
comply = np.random.binomial(1, p_comply, n)
Tr = Z * comply  # Treatment = instrument × compliance

# Formula interface
df = pd.DataFrame({
    'treat': Tr, 'z': Z, 'x1': X[:, 0], 'x2': X[:, 1]
})
fit = CBIV(formula="treat ~ x1 + x2 | z", data=df,
           method='over', twosided=False)
print(fit.summary())
# CBIV Estimation Results
# ===============================
# Sample size: 500
# Method: over
# Two-sided noncompliance: No
# Converged: Yes
#
# Model Statistics:
#   J-statistic: 0.021656
#   Complier Probabilities (π_c): Mean=0.6098
#   Complier Weights (1/π_c): Mean=1.6431

# Matrix interface (equivalent)
fit2 = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=False)
print(f"Converged: {fit2.converged}")   # True
print(f"J-statistic: {fit2.J:.4f}")     # 0.0217
```

---

## Diagnostics

### Balance Assessment

```python
from cbps import CBPS, balance
from cbps.datasets import load_lalonde

data = load_lalonde()
fit = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data, att=0, method='over'
)

# Balance assessment (cf. Table 3 in Imai & Ratkovic, 2014)
bal = balance(fit)
print(bal['balanced'])   # Balance statistics after CBPS weighting
print(bal['original'])   # Baseline unweighted statistics
# DataFrames have covariate names as row index for all estimator types
```

### Asymptotic Variance (AsyVar)

`AsyVar` computes the asymptotic variance and confidence interval for the
AIPW-based ATE using one of two closed-form variance formulae derived by
Fan et al. (2022):

- `method='CBPS'` — full sandwich formula that propagates propensity-score
  estimation uncertainty through the joint GMM influence function.
- `method='oCBPS'` — semiparametric efficiency bound (Hahn, 1998), valid
  when the oCBPS balancing conditions `baseline_formula` / `diff_formula`
  produce a root-n consistent, doubly-robust ATE estimator.

The `method` argument of `AsyVar` is independent of the `method` argument of
`CBPS()` (which controls `'over'` vs `'exact'` GMM).

```python
from cbps import CBPS, AsyVar
from cbps.datasets import load_lalonde

data = load_lalonde()
fit = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data, att=0, method='over'
)

# Efficiency-bound variance (recommended when an outcome model is available)
result = AsyVar(Y=data['re78'].values, CBPS_obj=fit, method='oCBPS')

# Preferred: snake_case keys
print(f"ATE: {result['mu_hat']:.3f} (SE: {result['std_err']:.3f})")
print(f"95% CI: [{result['ci_mu_hat'][0]:.1f}, {result['ci_mu_hat'][1]:.1f}]")

# Backward compatible: R-style keys also work
print(f"ATE: {result['mu.hat']:.3f}")  # same value as result['mu_hat']

# Full sandwich variance (no outcome-model efficiency claim needed)
result_cbps = AsyVar(Y=data['re78'].values, CBPS_obj=fit, method='CBPS')
```

### Visualization

```python
from cbps import CBPS, plot_cbps, plot_cbps_continuous
from cbps.datasets import load_lalonde, load_political_ads
import numpy as np

data = load_lalonde()
fit = CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data, att=0, method='over'
)

# Love plot for binary treatment balance (cf. Figure 1 concept in the paper)
plot_cbps(fit)

# For continuous treatment (Fong et al. 2018, Figure 3 concept)
df_raw, meta = load_political_ads()
work = df_raw.copy()
lam = meta["boxcox_lambda"]
work["T_bc"] = ((work["TotAds"].values + 1).clip(min=1e-10) ** lam - 1.0) / lam
work["logPop"] = np.log(work["TotalPop"].values.clip(min=1))
work["logInc"] = np.log(work["Inc"].values.clip(min=0) + 1)
work["logPop_sq"] = work["logPop"] ** 2
work["density_sq"] = work["density"] ** 2
work["logInc_sq"] = work["logInc"] ** 2
work["PercentHispanic_sq"] = work["PercentHispanic"] ** 2
work["PercentBlack_sq"] = work["PercentBlack"] ** 2
work["PercentOver65_sq"] = work["PercentOver65"] ** 2
work["per_collegegrads_sq"] = work["per_collegegrads"] ** 2
cov_cols = ["logPop", "density", "logInc", "PercentHispanic",
            "PercentBlack", "PercentOver65", "per_collegegrads", "CanCommute",
            "logPop_sq", "density_sq", "logInc_sq", "PercentHispanic_sq",
            "PercentBlack_sq", "PercentOver65_sq", "per_collegegrads_sq"]
work = work.dropna(subset=["T_bc"] + cov_cols).reset_index(drop=True)
fit_cont = CBPS(formula="T_bc ~ " + " + ".join(cov_cols), data=work, att=0, method="over")
plot_cbps_continuous(fit_cont)
```

### J-Statistic (Specification Test)

For over-identified CBPS, Hansen's J-statistic tests model specification:

$$J = n \cdot \bar{g}(\hat{\beta})' \hat{\Sigma}^{-1} \bar{g}(\hat{\beta}) \xrightarrow{d} \chi^2_{k}$$

A significant J-statistic suggests potential model misspecification.

```python
from scipy import stats

# J-statistic is stored in fit.J
print(f"J-statistic: {fit.J:.4f}")

# Compute p-value manually
k = fit.coefficients.shape[0]  # number of parameters
j_pvalue = 1 - stats.chi2.cdf(fit.J, k)
print(f"Degrees of freedom: {k}")
print(f"p-value: {j_pvalue:.4f}")
```

---

## API Reference

### Core Estimators

| Function | Treatment Type | Description |
|:---------|:---------------|:------------|
| `CBPS()` | Binary, Multi-valued, Continuous | Main estimator with automatic detection; also supports oCBPS via `baseline_formula`/`diff_formula` |
| `cbps_fit()` | Binary, Multi-valued, Continuous | Low-level array interface for CBPS |
| `npCBPS()` | Continuous | Nonparametric empirical likelihood |
| `npCBPS_fit()` | Continuous | Low-level array interface for npCBPS |
| `CBMSM()` | Time-varying Binary | Marginal structural models (formula interface) |
| `cbmsm_fit()` | Time-varying Binary | Low-level array interface for CBMSM |
| `hdCBPS()` | Binary | High-dimensional with LASSO regularization |
| `CBIV()` | Binary | Instrumental variables |

### Diagnostics and Inference

| Function | Description |
|:---------|:------------|
| `balance()` | Covariate balance statistics (SMD for discrete, correlation for continuous) |
| `AsyVar()` | Asymptotic variance estimation for ATE (returns both snake_case and R-style keys) |
| `vcov_outcome()` | Variance-covariance matrix for weighted outcome regression |
| `plot_cbps()` | Love plot for binary/multi-valued treatments |
| `plot_cbps_continuous()` | Correlation plot for continuous treatments |
| `plot_cbmsm()` | Balance plot for marginal structural models |
| `plot_npcbps()` | Balance plot for nonparametric CBPS |

### scikit-learn Integration

| Class | Description |
|:------|:------------|
| `cbps.sklearn.CBPSEstimator` | scikit-learn compatible wrapper around the binary / multi-valued `CBPS()` estimator; exposes `fit`, `predict_proba`, `predict`, `get_weights` and is usable inside `Pipeline`. |

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from cbps.sklearn import CBPSEstimator
from cbps.datasets import load_lalonde

df = load_lalonde()
covs = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75']
X = df[covs].values
T = df['treat'].values
Y = df['re78'].values

# Fit CBPS inside a sklearn Pipeline (preprocessing → CBPS)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('cbps', CBPSEstimator(att=0, method='over')),
])
pipe.fit(X, T)

est = pipe.named_steps['cbps']
print(f"Converged: {est.cbps_result_.converged}")
print(f"J-statistic: {est.cbps_result_.J:.4f}")

# Retrieve IPW weights for a downstream weighted outcome regression
w = est.get_weights()
ate_model = LinearRegression().fit(T.reshape(-1, 1), Y, sample_weight=w)
print(f"IPW-weighted ATE: {ate_model.coef_[0]:.2f}")
```

> **Known limitations.** `CBPSEstimator.predict_proba` / `predict` return
> the stored training-sample propensity scores only and raise
> `ValueError` on arrays with a different sample count. As a consequence,
> `GridSearchCV` / `cross_val_score` with default scoring do **not**
> produce meaningful test-fold scores. For out-of-sample propensity-score
> prediction, call the formula-interface result's
> `CBPSResults.predict(newdata=...)` method instead.

### Result Attributes (CBPSResults)

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `coefficients` | ndarray (k, 1) or (k, m) | Propensity score model coefficients |
| `coef` | ndarray (k,) | Flattened coefficient vector (alias) |
| `weights` | ndarray (n,) | Balancing weights |
| `fitted_values` | ndarray (n,) | Predicted propensity scores |
| `fitted` | ndarray (n,) | Alias for `fitted_values` |
| `linear_predictor` | ndarray (n,) | X @ beta before link transformation |
| `J` | float | Hansen's J-statistic |
| `J_stat` | float | Alias for `J` |
| `deviance` | float | Model deviance (-2 * log-likelihood) |
| `var` | ndarray (k, k) | Variance-covariance matrix of coefficients |
| `converged` | bool | Optimization convergence status |
| `residuals` | ndarray (n,) | Deviance residuals |
| `pseudo_r2` | float | McFadden's pseudo R-squared |
| `sigmasq` | float or None | Residual variance (continuous treatment only) |

> **Note**: `str(result)` and `str(result.summary())` now include a Diagnostics block showing convergence status, weight distribution summary (Min/Max/Mean), and Effective Sample Size (ESS).

### Result Methods

| Method | Description |
|:-------|:------------|
| `.summary()` | Returns `CBPSSummary` with coefficient table, SEs, z-values, p-values |
| `.vcov()` | Returns variance-covariance matrix |
| `.balance(**kwargs)` | Computes covariate balance diagnostics |
| `.predict(newdata=None, type='response')` | Predict propensity scores (or linear predictor) on new `DataFrame`; defaults to the training sample |
| `.plot(kind='deviance')` | Deviance-residual diagnostic plots (binary treatment only; currently the only supported `kind`) |

### Summary Methods for All Result Classes

All result classes now provide a consistent `summary()` method that returns a dedicated summary object (not a string). Use `print(result.summary())` for formatted output.

| Result Class | Summary Class | Key Contents |
|:-------------|:--------------|:-------------|
| `CBPSResults` | `CBPSSummary` | Coefficients, SEs, z-values, p-values, diagnostics |
| `CBMSMResults` | `CBMSMSummary` | Propensity scores, MSM weights, coefficients |
| `NPCBPSResults` | `NPCBPSSummary` | Convergence, optimization, weighted correlations, weight distribution |
| `HDCBPSResults` | `HDCBPSSummary` | ATE/ATT, variable selection, convergence |
| `CBIVResults` | `CBIVSummary` | Coefficients, J-statistic, balance |

### Summary Attributes (CBPSSummary)

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `coef` | ndarray | Coefficient estimates |
| `se` | ndarray | Standard errors |
| `zvalues` | ndarray | z-statistics |
| `pvalues` | ndarray | Two-sided p-values |

---

## Numerical Accuracy

This package maintains high numerical precision validated against the R CBPS package (v0.23):

| Component | Precision | Notes |
|:----------|:----------|:------|
| Coefficients | ±1e-6 | Core propensity score parameters |
| Weights | ±1e-6 | IPW weights |
| J-statistic | ±1e-4 | Specification test |
| Standard errors | ±1e-5 | Asymptotic variance |

Numerical accuracy is verified through extensive benchmark tests against R outputs using the LaLonde and Blackwell datasets.

---

## Datasets

The package includes classic datasets for causal inference research:

| Function | Description | Treatment Type | Reference |
|:---------|:------------|:---------------|:----------|
| `load_lalonde()` | NSW job training program evaluation | Binary | LaLonde (1986) |
| `load_lalonde_psid_combined()` | NSW experimental + PSID control data | Binary | Dehejia & Wahba (1999) |
| `load_blackwell()` | Longitudinal political campaign data | Time-varying Binary | Blackwell (2013) |
| `load_continuous_simulation()` | Simulated dose-response data (4 DGPs, 200 obs each) | Continuous | Fong et al. (2018) |
| `load_political_ads()` | Political advertising efficacy | Continuous | Urban & Niebler (2014) |
| `load_npcbps_continuous_sim()` | Nonparametric CBPS validation data | Continuous | Fong et al. (2018) |

### Example Usage

```python
from cbps.datasets import (
    load_lalonde,
    load_lalonde_psid_combined,
    load_blackwell,
    load_political_ads
)

# LaLonde (1986) job training data - 445 observations
lalonde = load_lalonde()

# Combined NSW + PSID data for selection bias studies
lalonde_psid = load_lalonde_psid_combined()

# Blackwell (2013) negative campaign advertising - longitudinal data
blackwell = load_blackwell()

# Political ads efficacy data (Urban & Niebler 2014)
df_raw, meta = load_political_ads()
```

---

## References

Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 76(1), 243-263. [doi:10.1111/rssb.12027](https://doi.org/10.1111/rssb.12027)

Imai, K., & Ratkovic, M. (2015). Robust estimation of inverse probability weights for marginal structural models. *Journal of the American Statistical Association*, 110(511), 1013-1023. [doi:10.1080/01621459.2014.956872](https://doi.org/10.1080/01621459.2014.956872)

Fong, C., Hazlett, C., & Imai, K. (2018). Covariate balancing propensity score for a continuous treatment: Application to the efficacy of political advertisements. *The Annals of Applied Statistics*, 12(1), 156-177. [doi:10.1214/17-AOAS1101](https://doi.org/10.1214/17-AOAS1101)

Ning, Y., Peng, S., & Imai, K. (2020). Robust estimation of causal effects via a high-dimensional covariate balancing propensity score. *Biometrika*, 107(3), 533-554. [doi:10.1093/biomet/asaa020](https://doi.org/10.1093/biomet/asaa020)

Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., & Yang, X. (2022). Optimal covariate balancing conditions in propensity score estimation. *Journal of Business & Economic Statistics*, 41(1), 97-110. [doi:10.1080/07350015.2021.2002159](https://doi.org/10.1080/07350015.2021.2002159)

## Authors

**Python Implementation:**

- **Xuanyu Cai**, City University of Macau
  Email: [xuanyuCAI@outlook.com](mailto:xuanyuCAI@outlook.com)
- **Wenli Xu**, City University of Macau
  Email: [wlxu@cityu.edu.mo](mailto:wlxu@cityu.edu.mo)

**Methodology:**

- **Kosuke Imai**, Harvard University
- **Marc Ratkovic**, Princeton University
- **Christian Fong**, Stanford University
- **Chad Hazlett**, UCLA
- **Yang Ning**, Cornell University
- **Jianqing Fan**, Princeton University

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this package in your research, please cite both the methodology papers and the Python implementation:

**APA Format:**

> Cai, X., & Xu, W. (2025). *cbps: Covariate Balancing Propensity Score for Python* (Version 0.1.0) [Computer software]. GitHub. https://github.com/gorgeousfish/CBPS-py
>
> Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 76(1), 243-263.
>
> Imai, K., & Ratkovic, M. (2015). Robust estimation of inverse probability weights for marginal structural models. *Journal of the American Statistical Association*, 110(511), 1013-1023.
>
> Fong, C., Hazlett, C., & Imai, K. (2018). Covariate balancing propensity score for a continuous treatment: Application to the efficacy of political advertisements. *The Annals of Applied Statistics*, 12(1), 156-177.
>
> Ning, Y., Peng, S., & Imai, K. (2020). Robust estimation of causal effects via a high-dimensional covariate balancing propensity score. *Biometrika*, 107(3), 533-554.
>
> Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., & Yang, X. (2022). Optimal covariate balancing conditions in propensity score estimation. *Journal of Business & Economic Statistics*, 41(1), 97-110.

**BibTeX:**

```bibtex
@software{cbps2025python,
  title={cbps: Covariate Balancing Propensity Score for Python},
  author={Cai, Xuanyu and Xu, Wenli},
  year={2025},
  version={0.1.0},
  url={https://github.com/gorgeousfish/CBPS-py}
}

@article{imai2014covariate,
  title={Covariate Balancing Propensity Score},
  author={Imai, Kosuke and Ratkovic, Marc},
  journal={Journal of the Royal Statistical Society Series B: Statistical Methodology},
  volume={76}, number={1}, pages={243--263},
  year={2014},
  doi={10.1111/rssb.12027}
}

@article{imai2015robust,
  title={Robust Estimation of Inverse Probability Weights for Marginal Structural Models},
  author={Imai, Kosuke and Ratkovic, Marc},
  journal={Journal of the American Statistical Association},
  volume={110}, number={511}, pages={1013--1023},
  year={2015},
  doi={10.1080/01621459.2014.956872}
}

@article{fong2018covariate,
  title={Covariate Balancing Propensity Score for a Continuous Treatment: Application to the Efficacy of Political Advertisements},
  author={Fong, Christian and Hazlett, Chad and Imai, Kosuke},
  journal={The Annals of Applied Statistics},
  volume={12}, number={1}, pages={156--177},
  year={2018},
  doi={10.1214/17-AOAS1101}
}

@article{ning2020robust,
  title={Robust Estimation of Causal Effects via a High-Dimensional Covariate Balancing Propensity Score},
  author={Ning, Yang and Peng, Sida and Imai, Kosuke},
  journal={Biometrika},
  volume={107}, number={3}, pages={533--554},
  year={2020},
  doi={10.1093/biomet/asaa020}
}

@article{fan2022optimal,
  title={Optimal Covariate Balancing Conditions in Propensity Score Estimation},
  author={Fan, Jianqing and Imai, Kosuke and Lee, Inbeom and Liu, Han and Ning, Yang and Yang, Xiaolin},
  journal={Journal of Business \& Economic Statistics},
  volume={41}, number={1}, pages={97--110},
  year={2022},
  doi={10.1080/07350015.2021.2002159}
}
```

## See Also

- Original R package by Fong, Ratkovic, Imai, Hazlett, and Yang: https://CRAN.R-project.org/package=CBPS
- Paper: Imai, K., & Ratkovic, M. (2014). Covariate balancing propensity score. https://doi.org/10.1111/rssb.12027
