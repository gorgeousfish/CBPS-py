# CBPS Python Examples

This directory contains comprehensive examples and tutorials for the CBPS Python package.

## Tutorials (Jupyter Notebooks)

Interactive tutorials covering complete workflows (4/4 completed):

### 1. `tutorial_binary.ipynb` - Binary Treatment CBPS
   - LaLonde dataset exploration
   - ATT and ATE estimation
   - Method comparison (exact vs. over-identified GMM)
   - Balance assessment and visualization
   - Treatment effect estimation with AsyVar
   - Best practices and troubleshooting
   - **10 sections, ~360 lines**

### 2. `tutorial_continuous.ipynb` - Continuous Treatment CBPS (GPS)
   - Political ads data exploration
   - Generalized Propensity Score (GPS) estimation
   - Covariate balance using correlations (target |ρ| < 0.1)
   - F-statistic verification for overall balance
   - GPS weight distribution and effective sample size
   - Outcome regression with GPS weights
   - Robust variance adjustment with vcov_outcome
   - Dose-response curve analysis
   - **11 sections, ~440 lines**

### 3. `tutorial_msm.ipynb` - Marginal Structural Models (CBMSM)
   - Blackwell longitudinal data structure exploration
   - Time-varying vs. time-invariant treatment models
   - MSM weight estimation with CBMSM
   - Propensity scores across time periods
   - MSM weights examination by period
   - **7 sections, ~365 lines**

### 4. `tutorial_hdcbps.ipynb` - High-Dimensional CBPS (hdCBPS)
   - **Note:** Requires glmnetforpython package
   - High-dimensional data simulation (p=100, n=200, p/n=0.5)
   - Why standard CBPS fails in high dimensions
   - LASSO variable selection with cross-validation
   - Variable selection accuracy metrics (precision/recall/F1)
   - Propensity score estimation and validation
   - Balance assessment for selected variables
   - Effective sample size calculation
   - Installation troubleshooting (especially Apple Silicon M1/M2/M3)
   - **8 sections, ~380 lines**

**Total: 4 comprehensive tutorials**

## Basic Examples (Python Scripts)

Focused examples demonstrating specific functionality:

### Core Functions

1. `cbps_basic.py` - Basic CBPS usage for binary treatment
   - ATT estimation with over-identified GMM
   - Complete workflow from data loading to outcome analysis
   - 150 lines

2. `cbmsm_basic.py` - Marginal Structural Models
   - Time-invariant and time-varying treatment models
   - Propensity score verification across time periods
   - 150 lines

3. `npcbps_basic.py` - Nonparametric CBPS
   - Empirical likelihood-based weight estimation
   - Convergence diagnostics
   - 150 lines

4. `hdcbps_basic.py` - High-Dimensional CBPS
   - **REQUIRES:** `glmnetforpython` package
   - LASSO variable selection for p >> n
   - Sparse propensity score models
   - 150 lines

5. `cbiv_basic.py` - Instrumental Variables
   - Two-sided and one-sided noncompliance
   - MLE vs. GMM estimation methods
   - CACE (Complier Average Causal Effect) estimation
   - 287 lines

6. `asyvar_comprehensive_demo.py` - Asymptotic variance estimation
   - Treatment effect confidence intervals
   - Accounts for propensity score estimation uncertainty
   - Optimal CBPS (oCBPS) variance

### Diagnostic Functions

7. `balance_basic.py` - Balance assessment for binary treatment
   - Standardized mean differences (SMD)
   - Weighted vs. unweighted balance comparison
   - Balance table for publication
   - 150 lines

8. `balance_continuous.py` - Balance for continuous treatment
   - Pearson correlations as balance metric
   - F-statistic for overall balance
   - Interpretation guidelines
   - 150 lines

9. `plot_cbps.py` - Visualization for binary treatment
   - Scatter plots: propensity score vs. covariates
   - Box plots: covariate distributions by treatment
   - Propensity score overlap diagnostics
   - 150 lines

10. `plot_continuous.py` - Visualization for continuous treatment
    - Correlation plots (weighted vs. unweighted)
    - Treatment and weight distribution plots
    - Custom plot styling examples
    - 150 lines

### Inference Functions

11. `vcov_outcome_basic.py` - Outcome model variance adjustment
    - Robust variance estimation for weighted regression
    - Comparison with naive standard errors
    - Binary and continuous treatment examples
    - 150 lines

12. `vcov_cbps.py` - Propensity score coefficient variance
    - Variance-covariance matrix extraction
    - Standard errors and confidence intervals
    - Correlation structure and eigenvalue decomposition
    - 150 lines

13. `summary_cbps.py` - Statistical summaries
    - Coefficient table with p-values
    - J-statistic interpretation
    - Convergence diagnostics and weight summary
    - 150 lines

**Total: 13 basic examples**

## Advanced Examples (Planned)

More complex use cases and best practices (upcoming in future releases):

1. **`advanced_ate_att_comparison.py`** - ATE vs. ATT comparison
2. **`advanced_optimal_cbps.py`** - Optimal CBPS (oCBPS) with doubly-robust estimation
3. **`advanced_sensitivity_analysis.py`** - Sensitivity to unobserved confounding
4. **`advanced_subgroup_analysis.py`** - Heterogeneous treatment effects
5. **`advanced_multiple_treatments.py`** - Multi-valued treatment analysis

## Quick Start

### Running Python Scripts

```bash
# Navigate to examples directory
cd examples

# Run any example
python cbps_basic.py
python cbmsm_basic.py
python balance_basic.py
```

### Running Jupyter Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open any tutorial_*.ipynb file
```

## Data Requirements

All examples use built-in datasets from `cbps.datasets`:

- **LaLonde**: `load_lalonde()` - Job training program evaluation
- **Blackwell**: `load_blackwell()` - Longitudinal political data
- **Political Ads**: `load_political_ads()` - Continuous treatment data
- **Simulations**: Various simulation datasets for testing

No external data files required!

## Important Notes

### hdCBPS Dependency

hdCBPS **requires** the `glmnetforpython` package:

```bash
pip install glmnetforpython
```

**Apple Silicon (M1/M2/M3) users:**
```bash
brew install gcc
export FC=gfortran
pip install glmnetforpython
```

## Documentation

For complete API documentation, see:

- **API Reference:** `docs/api/`
- **README:** `../README.md`
- **Online Documentation:** https://cbps-python.readthedocs.io

## License

Same as CBPS Python package (GPL-2+)

