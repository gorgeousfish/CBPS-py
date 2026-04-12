"""
Replication of Imai and Ratkovic (2015), "Robust Estimation of Inverse
Probability Weights for Marginal Structural Models," Journal of the American
Statistical Association, 110(511), 1013-1023.

This script reproduces the empirical application from Section 5 of the paper,
which analyzes the effect of negative campaign advertising on Democratic vote
share using the Blackwell (2013) longitudinal dataset.

  Section 5 -- Negative campaign advertising and election outcomes (Table 3)
  Section 5 -- Covariate balance diagnostics (Figure 4 concept)

The dataset contains 114 U.S. Senate and gubernatorial races from 2000-2006,
observed over J=5 weekly periods leading up to each election.  The treatment
is whether a candidate ran negative advertisements in a given week, and the
outcome is the Democratic candidate's final vote share.

Usage:
    python replicate_imai_ratkovic_2015.py

References:
    Blackwell, M. (2013). A framework for dynamic causal inference in
        political science. American Journal of Political Science, 57(2),
        504-519.
    Imai, K. and Ratkovic, M. (2015). Robust estimation of inverse
        probability weights for marginal structural models. Journal of the
        American Statistical Association, 110(511), 1013-1023.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

import cbps
from cbps.datasets import load_blackwell

# Full treatment model from Section 5 (1548 balancing conditions)
FULL_FORMULA = (
    "d.gone.neg ~ d.gone.neg.l1 + d.gone.neg.l2 + d.neg.frac.l3"
    " + camp.length + deminc + base.poll"
    " + year.2002 + year.2004 + year.2006"
    " + base.und + office"
)


def load_and_describe():
    """Load the Blackwell panel data and print descriptive statistics."""
    df = load_blackwell()
    n_units = df["demName"].nunique()
    periods = sorted(df["time"].unique())

    print("Blackwell (2013) negative campaign advertising data")
    print(f"  Candidates: {n_units}, periods: {len(periods)}, "
          f"observations: {len(df)}")
    for t in periods:
        prev = df.loc[df["time"] == t, "d.gone.neg"].mean()
        print(f"  Week {t} treatment prevalence: {prev:.3f}")
    print()
    return df


def estimate_weights(df):
    """Estimate MSM weights via GLM, CBPS, and CBPS-Approximate.

    GLM weights are extracted from the glm_weights attribute of the CBMSM
    fit, which stores the standard logistic regression IPW for comparison.
    """
    fits = {}

    print("Estimating CBPS-Approximate (low-rank variance)...")
    fits["CBPS-Approx"] = cbps.CBMSM(
        formula=FULL_FORMULA, id="demName", time="time", data=df,
        type="MSM", time_vary=True, twostep=True, msm_variance="approx",
    )

    print("Estimating CBPS (full variance)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits["CBPS"] = cbps.CBMSM(
            formula=FULL_FORMULA, id="demName", time="time", data=df,
            type="MSM", time_vary=True, twostep=True, msm_variance="full",
        )

    for name, fit in fits.items():
        w = fit.fitted_values
        print(f"  {name}: converged={fit.converged}, J={fit.J:.4f}, "
              f"weights=[{w.min():.4f}, {w.max():.4f}]")
    print()
    return fits


def _get_weights(fits, method):
    """Return the appropriate weight vector for a given method name."""
    if method == "GLM":
        return fits["CBPS-Approx"].glm_weights
    return fits[method].fitted_values


def _get_fit(fits, method):
    """Return the CBMSMResults object used for a given method."""
    if method == "GLM":
        return fits["CBPS-Approx"]
    return fits[method]


def replicate_table3(df, fits):
    """Reproduce Table 3: Impact of negative advertising on vote share.

    Reports time-specific effects (beta_1 through beta_5) and the cumulative
    effect, estimated via weighted least squares under three weighting schemes.
    """
    print("Table 3: Impact of Negative Advertising on Vote Share")
    print("  (Imai and Ratkovic, 2015, Section 5)\n")

    # Outcome: first-period cross-section (one observation per candidate)
    first_t = df["time"].min()
    outcome = df.loc[df["time"] == first_t, "demprcnt"].values

    methods = ["GLM", "CBPS", "CBPS-Approx"]
    table = {}

    for method in methods:
        fit = _get_fit(fits, method)
        w = _get_weights(fits, method)

        # Time-specific effects: regress outcome on treatment history dummies
        X_hist = sm.add_constant(fit.treat_hist)
        m_hist = sm.WLS(outcome, X_hist, weights=w).fit()

        effects = {}
        for j in range(fit.treat_hist.shape[1]):
            effects[f"beta_{j+1}"] = (m_hist.params[j+1], m_hist.bse[j+1])

        # Cumulative effect: regress outcome on total treatment count
        X_cum = sm.add_constant(fit.treat_cum.reshape(-1, 1))
        m_cum = sm.WLS(outcome, X_cum, weights=w).fit()
        effects["Cumulative"] = (m_cum.params[1], m_cum.bse[1])

        table[method] = effects

    # Format output
    header = f"{'Effect':<14}" + "".join(f"  {m:>18}" for m in methods)
    print(header)
    print("-" * len(header))
    for key in [f"beta_{j}" for j in range(1, 6)] + ["Cumulative"]:
        row = f"{key:<14}"
        for method in methods:
            coef, se = table[method][key]
            row += f"  {coef:>8.3f} ({se:.3f})"
        print(row)

    print()
    print("Notes: WLS of Democratic vote share on treatment history indicators")
    print("(time-specific) or cumulative treatment count. SE in parentheses.")
    print()
    return table


def assess_balance(fits):
    """Covariate balance comparison between GLM and CBPS (cf. Figure 4).

    Reports per-covariate mean absolute standardized differences and the
    fraction of balancing conditions where CBPS achieves better balance.
    """
    print("Covariate Balance Assessment (cf. Figure 4)")
    print()

    fit = fits["CBPS-Approx"]
    bal = fit.balance()

    glm_raw = bal["Unweighted"]
    cbps_raw = bal["Balanced"]
    cov_names = bal["row_names"]

    # Standardized mean columns occupy the second half of the matrix
    n_pat = glm_raw.shape[1] // 2
    glm_std = np.abs(glm_raw[:, n_pat:])
    cbps_std = np.abs(cbps_raw[:, n_pat:])

    glm_avg = glm_std.mean(axis=1)
    cbps_avg = cbps_std.mean(axis=1)

    print(f"{'Covariate':<20} {'GLM':>10} {'CBPS':>10} {'Diff':>10}")
    print("-" * 52)
    for i, name in enumerate(cov_names):
        d = glm_avg[i] - cbps_avg[i]
        print(f"{name:<20} {glm_avg[i]:>10.4f} {cbps_avg[i]:>10.4f} "
              f"{'+' if d > 0 else ''}{d:>9.4f}")

    n_cond = glm_std.size
    n_better = int(np.sum(cbps_std < glm_std))
    pct = 100.0 * n_better / n_cond if n_cond > 0 else 0.0

    print()
    print(f"Balancing conditions: {n_cond}")
    print(f"CBPS improves balance: {n_better}/{n_cond} ({pct:.1f}%)")
    print(f"Mean abs. imbalance -- GLM: {glm_std.mean():.4f}, "
          f"CBPS: {cbps_std.mean():.4f}")
    print()


def weight_diagnostics(fits):
    """Report weight distribution summaries across estimation methods."""
    print("Weight Diagnostics")
    print()

    methods = ["GLM", "CBPS", "CBPS-Approx"]
    header = f"{'Statistic':<12}" + "".join(f"  {m:>14}" for m in methods)
    print(header)
    print("-" * len(header))

    ws = {m: _get_weights(fits, m) for m in methods}
    stats = [
        ("Min", np.min), ("Q1", lambda x: np.percentile(x, 25)),
        ("Median", np.median), ("Mean", np.mean),
        ("Q3", lambda x: np.percentile(x, 75)),
        ("Max", np.max), ("Std Dev", np.std),
    ]
    for label, fn in stats:
        row = f"{label:<12}" + "".join(f"  {fn(ws[m]):>14.4f}" for m in methods)
        print(row)
    print()


def main():
    """Replicate the empirical application from Section 5 of Imai and
    Ratkovic (2015)."""
    df = load_and_describe()
    fits = estimate_weights(df)
    weight_diagnostics(fits)
    replicate_table3(df, fits)
    assess_balance(fits)


if __name__ == "__main__":
    main()
