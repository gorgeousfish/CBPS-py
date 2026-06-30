# Monte Carlo Paper Reproduction Tests

Monte Carlo simulation tests validating the CBPS Python package against
published numerical results from five peer-reviewed papers.

## Structure

```
monte_carlo/
├── conftest.py          # DGP implementations and shared MC fixtures
├── paper_constants.py   # Numerical targets from papers (single source of truth)
├── test_imai2014.py     # Imai & Ratkovic (2014) JRSSB
├── test_ir2015.py       # Imai & Ratkovic (2015) JASA
├── test_fong2018.py     # Fong, Hazlett & Imai (2018) AoAS
├── test_ning2020.py     # Ning, Peng & Imai (2020) Biometrika
└── test_fan2022.py      # Fan et al. (2022) JBES
```

## Paper Coverage

| Paper | File | Content |
|-------|------|---------|
| Imai & Ratkovic (2014) JRSSB 76(1) | `test_imai2014.py` | Table 1 (Kang-Schafer DGP), CBIV, multi-valued treatment, LaLonde empirical |
| Imai & Ratkovic (2015) JASA 110(511) | `test_ir2015.py` | Figures 2-3 (CBMSM longitudinal treatment effects) |
| Fong et al. (2018) AoAS 12(1) | `test_fong2018.py` | Section 4 (continuous CBPS, 4 DGP scenarios, NPCBGPS) |
| Ning et al. (2020) Biometrika 107(3) | `test_ning2020.py` | Table 1, Supplementary Tables 2-6 (high-dimensional CBPS) |
| Fan et al. (2022) JBES 41(1) | `test_fan2022.py` | Tables 1-4 (optimal CBPS, inference, beta coverage) |

## Supporting Modules

- **conftest.py**: DGP implementations (exact from papers) and shared MC fixtures
- **paper_constants.py**: Single source of truth for all numerical targets and tolerances

## Tolerance Configuration

All tolerances are based on Monte Carlo Standard Error (MC SE):

```
For bias:     MC SE = SD / sqrt(n_sims), tolerance = 3 * MC SE
For coverage: MC SE = sqrt(p(1-p) / n_sims), tolerance = 5 * MC SE
For RMSE:     tolerance = max(0.25, |paper_value| * 0.12)
```

See `paper_constants.py` for complete tolerance configuration with scientific
justification for each threshold.

## Running Tests

```bash
# Quick tests (CI/CD)
pytest tests/monte_carlo/ -m "not slow" -v

# Full paper reproduction
pytest tests/monte_carlo/ -m "paper_reproduction" -v

# Specific paper
pytest tests/monte_carlo/test_imai2014.py -v
pytest tests/monte_carlo/test_ir2015.py -v
pytest tests/monte_carlo/test_fong2018.py -v
pytest tests/monte_carlo/test_ning2020.py -v
pytest tests/monte_carlo/test_fan2022.py -v
```

## References

1. Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
   JRSSB 76(1), 243-263. DOI: 10.1111/rssb.12027

2. Imai, K. and Ratkovic, M. (2015). Robust Estimation of Inverse Probability
   Weights for Marginal Structural Models. JASA 110(511), 1013-1023.
   DOI: 10.1080/01621459.2014.956872

3. Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
   Score for a Continuous Treatment. AoAS 12(1), 156-177.
   DOI: 10.1214/17-AOAS1101

4. Ning, Y., Peng, S., and Imai, K. (2020). Robust Estimation of Causal Effects
   via a High-Dimensional Covariate Balancing Propensity Score. Biometrika
   107(3), 533-554. DOI: 10.1093/biomet/asaa020

5. Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022). Optimal
   Covariate Balancing Conditions in Propensity Score Estimation. JBES 41(1),
   97-110. DOI: 10.1080/07350015.2021.2002159
