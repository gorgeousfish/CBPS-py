# CBPS Python Examples

This directory contains replication scripts and notebooks that reproduce the
core empirical results from three of the five CBPS methodology papers this
package implements: Imai & Ratkovic (2014), Imai & Ratkovic (2015), and Fong,
Hazlett & Imai (2018). The other two papers — Ning, Peng & Imai (2020) on
high-dimensional CBPS and Fan et al. (2022) on optimal CBPS — are covered by
the unit-test suite rather than by full replication notebooks. Each
replication is provided in two equivalent forms — a Python script (`.py`) for
command-line use and a Jupyter notebook (`.ipynb`) for interactive
exploration. A standalone command-line runner for the Kang-Schafer simulation
is also included.

## Available Replications

### 1. `replicate_imai_ratkovic_2014` — Binary and Multi-Valued CBPS

Replicates the two main empirical analyses from
Imai, K. and Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
*Journal of the Royal Statistical Society, Series B* 76(1), 243–263.

- **Section 3.1**: Kang-Schafer (2007) simulation study (Table 1) — compares
  CBPS against standard logistic regression under four specification scenarios
  (both correct, PS correct, outcome correct, both wrong) using four
  downstream estimators (HT, Hájek IPW, WLS, doubly-robust).
- **Section 3.2**: LaLonde (1986) NSW-PSID evaluation bias analysis (Table 2)
  — CBPS-based propensity score matching applied to the NSW experimental
  sample combined with PSID controls.

**Files**: `replicate_imai_ratkovic_2014.py`, `replicate_imai_ratkovic_2014.ipynb`

### 2. `replicate_fong_hazlett_imai_2018` — Continuous Treatment CBPS

Replicates the main simulation study and empirical application from
Fong, C., Hazlett, C., and Imai, K. (2018). "Covariate Balancing Propensity
Score for a Continuous Treatment." *The Annals of Applied Statistics* 12(1),
156–177.

- **Section 4**: Four-DGP simulation study (Figures 1–2) — compares
  Unadjusted / MLE GPS / CBGPS / npCBGPS across correctly and incorrectly
  specified treatment and outcome models.
- **Section 5**: Empirical application to Urban and Niebler (2014) political
  advertising data (Table 1, Figure 3) — Box-Cox transformed treatment,
  15 covariates, signed Pearson correlations and F-statistic diagnostics.

**Files**: `replicate_fong_hazlett_imai_2018.py`,
`replicate_fong_hazlett_imai_2018.ipynb`

### 3. `replicate_imai_ratkovic_2015` — Marginal Structural Models

Replicates the empirical application from
Imai, K. and Ratkovic, M. (2015). "Robust Estimation of Inverse Probability
Weights for Marginal Structural Models." *Journal of the American Statistical
Association* 110(511), 1013–1023.

- Blackwell (2013) longitudinal campaign data, 5 time periods.
- Time-invariant and time-varying CBMSM weight estimation.
- Comparison of CBPS-based stabilized weights against GLM weights.

**Files**: `replicate_imai_ratkovic_2015.py`, `replicate_imai_ratkovic_2015.ipynb`

### 4. `run_replication.py` — Standalone Kang-Schafer Runner

Self-contained command-line script that runs the Kang-Schafer simulation
portion of the Imai & Ratkovic (2014) replication. Useful for smoke testing
the package without opening a notebook.

## Quick Start

### Running Python Scripts

```bash
# From the repository root
python examples/replicate_imai_ratkovic_2014.py
python examples/replicate_fong_hazlett_imai_2018.py
python examples/replicate_imai_ratkovic_2015.py
```

### Running Jupyter Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open any replicate_*.ipynb file in the examples/ directory
```

## Data Requirements

All replications use datasets bundled with the package under
`cbps.datasets`:

- `load_lalonde()` — LaLonde (1986) NSW job training data.
- `load_lalonde_psid_combined()` — NSW treated + PSID control merged sample.
- `load_blackwell()` — Blackwell (2013) longitudinal political data.
- `load_political_ads()` — Urban & Niebler (2014) county-level advertising data.
- `load_continuous_simulation()` — Fong et al. (2018) simulation DGPs.
- `load_npcbps_continuous_sim()` — Simulation data for npCBPS validation.

No external data files are required.

## Notes on Reproducibility

- Monte Carlo portions use a reduced number of replications (e.g.,
  `n_sim = 20` in the notebook, compared to the 500 in the paper) so the
  notebooks run end-to-end in a few minutes. Exact numerical agreement with
  the published figures is not expected — trends and qualitative rankings
  across estimators should match.
- npCBPS performs empirical-likelihood optimization whose objective is
  non-convex; results may vary slightly across runs or platforms.
- hdCBPS requires the optional `glmnetforpython` package (see the
  Installation guide for Apple Silicon instructions).

## Documentation

- **Online documentation**: <https://cbps.readthedocs.io>
- **Top-level README**: [`../README.md`](../README.md)
- **API reference**: built from source docstrings under `docs/api/` and
  rendered at <https://cbps.readthedocs.io>.

## License

Same as the CBPS Python package (AGPL-3.0-or-later).

