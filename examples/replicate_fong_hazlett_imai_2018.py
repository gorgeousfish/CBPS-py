"""
Replication: Fong, Hazlett, and Imai (2018)

"Covariate Balancing Propensity Score for a Continuous Treatment:
Application to the Efficacy of Political Advertisements."
The Annals of Applied Statistics, 12(1), 156-177.

This script replicates the main simulation study (Section 4, Figures 1-2)
and the empirical application (Section 5, Table 1, Figure 3) from the paper.
The simulation compares four estimators under four data generating processes
that vary in whether the treatment and outcome models are correctly specified.
The empirical application estimates the effect of political advertising on
vote share using data from Urban and Niebler (2014).

Usage:
    python replicate_fong_hazlett_imai_2018.py
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cbps
from cbps.datasets import load_political_ads

DGP_LABELS = {
    1: "DGP1: Both correctly specified",
    2: "DGP2: Treatment misspecified",
    3: "DGP3: Outcome misspecified",
    4: "DGP4: Both misspecified",
}
COV_COLS = [f"X{j}" for j in range(1, 11)]
SIM_FORMULA = "T ~ " + " + ".join(COV_COLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_dgp(dgp, n=200, rng=None):
    """One Monte Carlo draw following Section 4.1 of the paper."""
    rng = rng or np.random.default_rng()
    K = 10
    Sigma = np.full((K, K), 0.2); np.fill_diagonal(Sigma, 1.0)
    X = rng.multivariate_normal(np.zeros(K), Sigma, size=n)

    if dgp in (1, 3):
        T = X[:,0] + X[:,1] + 0.2*X[:,2] + 0.2*X[:,3] + 0.2*X[:,4] + rng.normal(0, 2.0, n)
    else:
        T = (X[:,1]+0.5)**2 + 0.4*X[:,2] + 0.4*X[:,3] + 0.4*X[:,4] + rng.normal(0, 1.5, n)

    if dgp in (1, 2):
        Y = X[:,1] + 0.1*X[:,3] + 0.1*X[:,4] + 0.1*X[:,5] + T + rng.normal(0, 5.0, n)
    else:
        Y = 2*(X[:,1]+0.5)**2 + T + 0.5*X[:,3] + 0.5*X[:,4] + 0.5*X[:,5] + rng.normal(0, 5.0, n)

    df = pd.DataFrame(X, columns=COV_COLS)
    df["T"], df["Y"] = T, Y
    return df


def _ate_wls(df, weights=None):
    """WLS regression of Y on T; returns the slope (ATE estimate)."""
    Xr = sm.add_constant(df["T"].values)
    m = sm.WLS(df["Y"].values, Xr, weights=weights) if weights is not None else sm.OLS(df["Y"].values, Xr)
    return m.fit().params[1]


def _f_balance(T, X, weights=None):
    """F-statistic from (weighted) regression of T on X."""
    Xc = sm.add_constant(X)
    n = len(T)
    if weights is not None:
        w = weights * n / weights.sum()
        return sm.WLS(T, Xc, weights=w).fit().fvalue
    return sm.OLS(T, Xc).fit().fvalue


def _wcorr(T, X, weights=None):
    """Weighted Pearson correlations between T and each column of X."""
    K = X.shape[1]
    out = np.zeros(K)
    if weights is None:
        for j in range(K):
            out[j] = np.corrcoef(T, X[:, j])[0, 1]
    else:
        w = weights / weights.sum()
        mT = w @ T
        for j in range(K):
            mX = w @ X[:, j]
            cov = np.sum(w * (T - mT) * (X[:, j] - mX))
            vT = np.sum(w * (T - mT)**2)
            vX = np.sum(w * (X[:, j] - mX)**2)
            out[j] = cov / np.sqrt(vT * vX) if vT > 0 and vX > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Part 1: Simulation Study (Section 4, Figures 1-2)
# ---------------------------------------------------------------------------

def run_simulation(n_sim=100, seed=42):
    """Monte Carlo comparison of four estimators across four DGPs.

    Replicates Section 4.1 (Figures 1-2) with n_sim=100 replications
    (the paper uses 500) for computational tractability.
    """
    print("Part 1: Simulation Study (Section 4, Figures 1-2)")
    print(f"  Replications: {n_sim} (paper: 500), n=200, K=10, true ATE=1.0\n")

    methods = ["Unadjusted", "MLE", "CBGPS", "npCBGPS"]
    R = {d: {m: {"f": [], "ate": []} for m in methods} for d in range(1, 5)}
    rng = np.random.default_rng(seed)

    for dgp in range(1, 5):
        print(f"  {DGP_LABELS[dgp]}")
        for _ in range(n_sim):
            df = _generate_dgp(dgp, rng=rng)
            T, X = df["T"].values, df[COV_COLS].values

            # Unadjusted
            R[dgp]["Unadjusted"]["ate"].append(_ate_wls(df))
            R[dgp]["Unadjusted"]["f"].append(_f_balance(T, X))

            # MLE (one-step GPS, no balance optimization)
            try:
                Xc_sim = sm.add_constant(X)
                ols_sim = sm.OLS(T, Xc_sim).fit()
                from scipy.stats import norm as _norm
                gps_sim = _norm.pdf(T, loc=ols_sim.fittedvalues,
                                    scale=np.sqrt(np.mean(ols_sim.resid**2)))
                marg_sim = _norm.pdf(T, loc=T.mean(), scale=T.std(ddof=1))
                w_sim = marg_sim / np.clip(gps_sim, 1e-6, None)
                w_sim = w_sim / w_sim.sum()
                R[dgp]["MLE"]["ate"].append(_ate_wls(df, w_sim))
                R[dgp]["MLE"]["f"].append(_f_balance(T, X, w_sim))
            except Exception:
                R[dgp]["MLE"]["ate"].append(np.nan)
                R[dgp]["MLE"]["f"].append(np.nan)

            # CBGPS (parametric, just-identified per paper eq. 2)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = cbps.CBPS(formula=SIM_FORMULA, data=df, att=0,
                                    method='exact', verbose=0)
                R[dgp]["CBGPS"]["ate"].append(_ate_wls(df, fit.weights))
                R[dgp]["CBGPS"]["f"].append(_f_balance(T, X, fit.weights))
            except Exception:
                R[dgp]["CBGPS"]["ate"].append(np.nan)
                R[dgp]["CBGPS"]["f"].append(np.nan)

            # npCBGPS (nonparametric)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = cbps.npCBPS(formula=SIM_FORMULA, data=df,
                                      corprior=0.1, print_level=0)
                R[dgp]["npCBGPS"]["ate"].append(_ate_wls(df, fit.weights))
                R[dgp]["npCBGPS"]["f"].append(_f_balance(T, X, fit.weights))
            except Exception:
                R[dgp]["npCBGPS"]["ate"].append(np.nan)
                R[dgp]["npCBGPS"]["f"].append(np.nan)

    # --- Summary tables ---
    def _print_table(title, key, func):
        print(f"\n  {title}")
        print(f"  {'DGP':<30s}" + "".join(f"{m:>12s}" for m in methods))
        for d in range(1, 5):
            row = DGP_LABELS[d].split(":")[0]
            vals = []
            for m in methods:
                v = [x for x in R[d][m][key] if np.isfinite(x)]
                vals.append(func(v) if v else np.nan)
            print(f"  {row:<30s}" + "".join(f"{v:>12.4f}" for v in vals))

    _print_table("Median F-statistic (cf. Figure 1)", "f", np.median)
    _print_table("ATE bias = E[ATE_hat] - 1.0 (cf. Figure 2)", "ate",
                 lambda v: np.mean(v) - 1.0)
    _print_table("ATE RMSE (cf. Figure 2)", "ate",
                 lambda v: np.sqrt(np.mean((np.array(v) - 1.0)**2)))
    return R


# ---------------------------------------------------------------------------
# Part 2: Empirical Application (Section 5, Table 1, Figure 3)
# ---------------------------------------------------------------------------

# Paper covariates (Section 5, p. 170-171): 8 original + 7 squared terms = 15
# "We add the squares of all nonbinary pretreatment covariates to the model
#  in order to balance both their first and second moments." (p. 171)
COV_DISPLAY = [
    ("logPop",            "log(Population)"),
    ("density",           "Population density"),
    ("logInc",            "log(Income+1)"),
    ("PercentHispanic",   "% Hispanic"),
    ("PercentBlack",      "% Black"),
    ("PercentOver65",     "% Over 65"),
    ("per_collegegrads",  "% College graduates"),
    ("CanCommute",        "Commute indicator"),
    ("logPop_sq",         "log(Population)^2"),
    ("density_sq",        "Population density^2"),
    ("logInc_sq",         "log(Income+1)^2"),
    ("PercentHispanic_sq", "% Hispanic^2"),
    ("PercentBlack_sq",   "% Black^2"),
    ("PercentOver65_sq",  "% Over 65^2"),
    ("per_collegegrads_sq", "% College graduates^2"),
]


def run_empirical_application():
    """Replicate Section 5 (Table 1, Figure 3).

    Data: Urban and Niebler (2014) political advertising dataset.
    Treatment: TotAds, Box-Cox transformed with lambda = -0.16.
    """
    print("\nPart 2: Empirical Application (Section 5, Table 1, Figure 3)")

    df_raw, meta = load_political_ads()
    print(f"  Dataset: Urban & Niebler (2014), n={len(df_raw)}")
    print(f"  Treatment: TotAds, Box-Cox lambda={meta['boxcox_lambda']}")

    # Prepare data following the paper
    work = df_raw.copy()
    lam = meta["boxcox_lambda"]
    work["T_bc"] = ((work["TotAds"].values + 1).clip(min=1e-10)**lam - 1.0) / lam
    work["logPop"] = np.log(work["TotalPop"].values.clip(min=1))
    work["logInc"] = np.log(work["Inc"].values.clip(min=0) + 1)

    # Add squared terms for all non-binary covariates (paper p. 171)
    work["logPop_sq"] = work["logPop"] ** 2
    work["density_sq"] = work["density"] ** 2
    work["logInc_sq"] = work["logInc"] ** 2
    work["PercentHispanic_sq"] = work["PercentHispanic"] ** 2
    work["PercentBlack_sq"] = work["PercentBlack"] ** 2
    work["PercentOver65_sq"] = work["PercentOver65"] ** 2
    work["per_collegegrads_sq"] = work["per_collegegrads"] ** 2

    cov_cols = [c for c, _ in COV_DISPLAY]
    work = work.dropna(subset=["T_bc"] + cov_cols).reset_index(drop=True)
    print(f"  Analysis sample: n={len(work)}")

    formula = "T_bc ~ " + " + ".join(cov_cols)
    T = work["T_bc"].values
    X = work[cov_cols].values

    # Fit four estimators
    corrs, fstats = {}, {}

    corrs["Unweighted"] = _wcorr(T, X)
    fstats["Unweighted"] = _f_balance(T, X)

    print("\n  Fitting MLE (standard GPS)...")
    # MLE: standard GPS via OLS, no balance conditions
    Xc_mle = sm.add_constant(X)
    ols_fit = sm.OLS(T, Xc_mle).fit()
    T_hat = ols_fit.fittedvalues
    sigma_mle = np.sqrt(np.mean(ols_fit.resid**2))
    from scipy.stats import norm
    gps_mle = norm.pdf(T, loc=T_hat, scale=sigma_mle)
    marginal_mle = norm.pdf(T, loc=T.mean(), scale=T.std(ddof=1))
    w_mle = marginal_mle / np.clip(gps_mle, 1e-6, None)
    w_mle = w_mle / w_mle.sum()
    corrs["MLE"] = _wcorr(T, X, w_mle)
    fstats["MLE"] = _f_balance(T, X, w_mle)

    print("  Fitting CBGPS (parametric)...")
    fit_cb = cbps.CBPS(formula=formula, data=work, att=0,
                       method='exact', verbose=0)
    corrs["CBGPS"] = _wcorr(T, X, fit_cb.weights)
    fstats["CBGPS"] = _f_balance(T, X, fit_cb.weights)

    print("  Fitting npCBGPS (nonparametric)...")
    fit_np = cbps.npCBPS(formula=formula, data=work,
                         corprior=0.1, print_level=0)
    corrs["npCBGPS"] = _wcorr(T, X, fit_np.weights)
    fstats["npCBGPS"] = _f_balance(T, X, fit_np.weights)

    # Table 1: Signed Pearson correlations (p. 171)
    est_names = ["Unweighted", "MLE", "CBGPS", "npCBGPS"]
    print(f"\n  Table 1: Cor(T, X_j) (cf. Table 1, p. 171)")
    print(f"  {'Covariate':<25s}" + "".join(f"{e:>12s}" for e in est_names))
    for j, (col, label) in enumerate(COV_DISPLAY):
        row = f"  {label:<25s}"
        for e in est_names:
            row += f"{corrs[e][j]:>12.3f}"
        print(row)

    # Figure 3: F-statistic (p. 172)
    print(f"\n  F-statistic from regression of T on X (cf. Figure 3, p. 172)")
    print(f"    Unweighted:  {fstats['Unweighted']:.4f}  (paper: ~29.3)")
    print(f"    MLE:         {fstats['MLE']:.4f}")
    print(f"    CBGPS:       {fstats['CBGPS']:.6f}  (paper: ~9.33e-5)")
    print(f"    npCBGPS:     {fstats['npCBGPS']:.6f}")

    # Convergence diagnostics
    print(f"\n  Convergence:")
    print(f"    CBGPS converged: {fit_cb.converged}, J={fit_cb.J:.6f}")
    print(f"    CBGPS weights: [{fit_cb.weights.min():.4f}, {fit_cb.weights.max():.4f}]")
    print(f"    npCBGPS weights: [{fit_np.weights.min():.4f}, {fit_np.weights.max():.4f}]")

    return {"corrs": corrs, "fstats": fstats}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Replication: Fong, Hazlett, and Imai (2018)")
    print("Annals of Applied Statistics, 12(1), 156-177.\n")

    run_simulation(n_sim=20, seed=42)
    run_empirical_application()

    print("\n  Note: Simulation uses 20 replications (paper uses 500).")
    print("  Exact numerical agreement is not expected due to differences in")
    print("  random draws and optimizer implementations between R and Python.")


if __name__ == "__main__":
    main()
