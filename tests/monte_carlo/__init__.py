"""
Monte Carlo Paper Reproduction Tests for CBPS
==============================================

This subpackage contains Monte Carlo simulation tests that verify
the statistical properties of CBPS estimators by reproducing the
simulation studies from the five core papers. These tests are essential
for JOSS submission to demonstrate that the implementation correctly
captures the theoretical properties of the CBPS methodology.

All DGPs and parameters are exact reproductions from published papers.
No simplifications or modifications unless explicitly noted.

Test Files
----------
1. test_imai2014.py
   - Paper: Imai & Ratkovic (2014) JRSSB 76(1), 243-263
   - Content: Table 1 (Kang-Schafer DGP, 4 scenarios), CBIV (Section 3.3),
     multi-valued treatment (Section 3.2), LaLonde empirical analysis,
     Theorems 1-2, double robustness
   - DOI: 10.1111/rssb.12027

2. test_ir2015.py
   - Paper: Imai & Ratkovic (2015) JASA 110(511), 1013-1023
   - Content: CBMSM for marginal structural models (Figures 2-3),
     longitudinal treatment effects, qualitative validation
   - DOI: 10.1080/01621459.2014.956872

3. test_fong2018.py
   - Paper: Fong, Hazlett & Imai (2018) AoAS 12(1), 156-177
   - Content: Section 4 continuous treatment CBPS (4 DGP scenarios),
     nonparametric CBGPS (npCBGPS), F-statistics, bias, RMSE, coverage
   - DOI: 10.1214/17-AOAS1101

4. test_ning2020.py
   - Paper: Ning, Peng & Imai (2020) Biometrika 107(3), 533-554
   - Content: Table 1 (high-dimensional CBPS, d >> n),
     Supplementary Tables 2-6 (alternative DGPs), double robustness
   - DOI: 10.1093/biomet/asaa020

5. test_fan2022.py
   - Paper: Fan, Imai, Lee, Liu, Ning & Yang (2022) JBES 41(1), 97-110
   - Content: Tables 1-4 (optimal CBPS), semiparametric efficiency bound,
     inference module validation, beta coverage
   - DOI: 10.1080/07350015.2021.2002159

Supporting Modules
------------------
- conftest.py: DGP implementations and shared MC fixtures
- paper_constants.py: Exact numerical targets from papers (single source of truth)

Test Markers
------------
@pytest.mark.paper_reproduction
    Tests that reproduce exact paper results

@pytest.mark.slow
    Tests taking >10 seconds (typically full paper replications)

@pytest.mark.numerical
    Numerical accuracy and stability tests

Usage
-----
# Run quick tests only (for CI/CD)
pytest tests/monte_carlo/ -m "not slow" -v

# Run all tests including slow paper reproductions
pytest tests/monte_carlo/ -v

# Run specific paper reproduction
pytest tests/monte_carlo/test_imai2014.py -v
pytest tests/monte_carlo/test_fong2018.py -v

References
----------
1. Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
   Journal of the Royal Statistical Society, Series B 76(1), 243-263.

2. Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse probability
   weights for marginal structural models. Journal of the American Statistical
   Association 110(511), 1013-1023.

3. Fong, C., Hazlett, C., and Imai, K. (2018). Covariate Balancing Propensity
   Score for a Continuous Treatment. Annals of Applied Statistics 12(1), 156-177.

4. Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
   via a high-dimensional covariate balancing propensity score. Biometrika
   107(3), 533-554.

5. Fan, J., Imai, K., Lee, I., Liu, H., Ning, Y., and Yang, X. (2022). Optimal
   covariate balancing conditions in propensity score estimation. Journal of
   Business & Economic Statistics 41(1), 97-110.
"""
