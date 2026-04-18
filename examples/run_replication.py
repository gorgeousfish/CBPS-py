"""Run the Imai & Ratkovic 2014 replication as a script.

Assumes the ``cbps`` package is installed into the active environment
(e.g., via ``pip install -e .`` from the repository root).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

import cbps
from cbps.datasets import load_lalonde

# ============================================================
# Kang-Schafer DGP
# ============================================================
def kang_schafer_dgp(n, rng):
    X_star = rng.standard_normal((n, 4))
    X_obs = np.column_stack([
        np.exp(X_star[:, 0] / 2),
        X_star[:, 1] / (1 + np.exp(X_star[:, 0])) + 10,
        (X_star[:, 0] * X_star[:, 2] / 25 + 0.6) ** 3,
        (X_star[:, 0] + X_star[:, 3] + 20) ** 2,
    ])
    logit_ps = -X_star[:, 0] + 0.5 * X_star[:, 1] - 0.25 * X_star[:, 2] - 0.1 * X_star[:, 3]
    ps_true = expit(logit_ps)
    T = rng.binomial(1, ps_true)
    Y = 210 + 27.4 * X_star[:, 0] + 13.7 * (X_star[:, 1] + X_star[:, 2] + X_star[:, 3])
    Y += rng.standard_normal(n)
    return X_star, X_obs, T, Y, ps_true

def horvitz_thompson(Y, T, w):
    n = len(Y)
    return np.sum(T * Y * w) / n

def ipw_hajek(Y, T, w):
    return np.sum(T * Y * w) / np.sum(T * w)

def wls_estimator(Y, T, X, w):
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    mask = T == 1
    if mask.sum() < X_aug.shape[1]:
        return np.nan
    try:
        beta = np.linalg.lstsq(
            X_aug[mask] * np.sqrt(w[mask, None]),
            Y[mask] * np.sqrt(w[mask]), rcond=None
        )[0]
        return np.mean(X_aug @ beta)
    except np.linalg.LinAlgError:
        return np.nan

def doubly_robust(Y, T, X, w):
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    mask = T == 1
    if mask.sum() < X_aug.shape[1]:
        return np.nan
    try:
        beta = np.linalg.lstsq(X_aug[mask], Y[mask], rcond=None)[0]
        mu_hat = X_aug @ beta
        n = len(Y)
        return np.mean(mu_hat) + np.sum(T * w * (Y - mu_hat)) / n
    except np.linalg.LinAlgError:
        return np.nan

def compute_weights_from_ps(ps, T):
    return np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))

def fit_logistic(X_design, T_vec):
    X_aug = np.column_stack([np.ones(len(T_vec)), X_design])
    def neg_loglik(beta):
        xb = np.clip(X_aug @ beta, -30, 30)
        return -np.sum(T_vec * xb - np.log(1 + np.exp(xb)))
    res = minimize(neg_loglik, np.zeros(X_aug.shape[1]), method='BFGS')
    return np.clip(expit(X_aug @ res.x), 1e-6, 1 - 1e-6)

def run_one_simulation(n, rng):
    X_star, X_obs, T, Y, ps_true = kang_schafer_dgp(n, rng)
    results = {}
    scenarios = {
        'Both correct':    (X_star, X_star),
        'PS correct':      (X_star, X_obs),
        'Outcome correct': (X_obs,  X_star),
        'Both wrong':      (X_obs,  X_obs),
    }
    estimators = {
        'HT':  lambda Y, T, X, w: horvitz_thompson(Y, T, w),
        'IPW': lambda Y, T, X, w: ipw_hajek(Y, T, w),
        'WLS': lambda Y, T, X, w: wls_estimator(Y, T, X, w),
        'DR':  lambda Y, T, X, w: doubly_robust(Y, T, X, w),
    }
    for scenario_name, (X_ps, X_out) in scenarios.items():
        ps_glm = fit_logistic(X_ps, T)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fit_exact = cbps.CBPS(treatment=T, covariates=X_ps, att=0,
                                      method='exact', verbose=0)
                ps_cbps1 = np.clip(fit_exact.fitted_values, 1e-6, 1 - 1e-6)
            except Exception:
                ps_cbps1 = ps_glm
            try:
                fit_over = cbps.CBPS(treatment=T, covariates=X_ps, att=0,
                                     method='over', verbose=0)
                ps_cbps2 = np.clip(fit_over.fitted_values, 1e-6, 1 - 1e-6)
            except Exception:
                ps_cbps2 = ps_glm
        method_ps = {
            'GLM':   ps_glm,
            'CBPS1': ps_cbps1,
            'CBPS2': ps_cbps2,
            'True':  np.clip(ps_true, 1e-6, 1 - 1e-6),
        }
        for mname, ps_vec in method_ps.items():
            w = compute_weights_from_ps(ps_vec, T)
            for ename, efunc in estimators.items():
                results[(scenario_name, mname, ename)] = efunc(Y, T, X_out, w)
    return results

# ============================================================
# Run a small simulation (n=200, 50 reps) for quick check
# ============================================================
print("=" * 70)
print("Table 1: Kang-Schafer Simulation (n=200, 50 replications)")
print("=" * 70)

n_sim = 50
rng = np.random.default_rng(2014)
true_mean = 210.0
scenarios = ['Both correct', 'PS correct', 'Outcome correct', 'Both wrong']
methods = ['GLM', 'CBPS1', 'CBPS2', 'True']
estimator_names = ['HT', 'IPW', 'WLS', 'DR']
all_results = {(s, m, e): [] for s in scenarios for m in methods for e in estimator_names}

for i in range(n_sim):
    if (i+1) % 10 == 0:
        print(f"  Simulation {i+1}/{n_sim}...")
    sim = run_one_simulation(200, rng)
    for key, val in sim.items():
        all_results[key].append(val)

print(f"\n{'Scenario':<20s} {'Method':<8s}  {'HT_Bias':>10s} {'HT_RMSE':>10s}  {'IPW_Bias':>10s} {'IPW_RMSE':>10s}")
print("-" * 80)
for scenario in scenarios:
    for method in methods:
        line = f'{scenario:<20s} {method:<8s}'
        for est in ['HT', 'IPW']:
            vals = np.array(all_results[(scenario, method, est)])
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                bias = np.mean(vals) - true_mean
                rmse = np.sqrt(np.mean((vals - true_mean) ** 2))
                line += f'  {bias:>10.2f} {rmse:>10.2f}'
            else:
                line += f'  {"NA":>10s} {"NA":>10s}'
        print(line)
    print()

# ============================================================
# Table 2: LaLonde
# ============================================================
print("=" * 70)
print("Table 2: LaLonde Evaluation Bias")
print("=" * 70)

from cbps.datasets.lalonde import _get_data_dir
from scipy.spatial.distance import cdist
data_dir = _get_data_dir()
full = pd.read_csv(data_dir / 'LaLonde.csv')

nsw_ctrl = full[(full['exper'] == 1) & (full['treat'] == 0)].copy()
psid = full[full['exper'] == 0].copy()

nsw_ctrl_for_ps = nsw_ctrl.copy()
nsw_ctrl_for_ps['select'] = 1
psid_for_ps = psid.copy()
psid_for_ps['select'] = 0
combined = pd.concat([nsw_ctrl_for_ps, psid_for_ps], ignore_index=True)

combined['age_sq'] = combined['age'] ** 2
combined['educ_sq'] = combined['educ'] ** 2
combined['re75_sq'] = combined['re75'] ** 2
combined['re74_sq'] = combined['re74'] ** 2
combined['hisp_re74zero'] = combined['hisp'] * (combined['re74'] == 0).astype(float)

specs = {
    'Linear': 'select ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    'Quadratic': ('select ~ age + educ + black + hisp + married + nodegr + re74 + re75'
                  ' + age_sq + educ_sq + re74_sq + re75_sq'),
    'Smith-Todd': ('select ~ age + educ + black + hisp + married + nodegr'
                   ' + re74 + re75 + age_sq + educ_sq + re74_sq + re75_sq'
                   ' + hisp_re74zero'),
}

treated_mask = combined['select'] == 1
control_mask = combined['select'] == 0
treated_idx = np.where(treated_mask)[0]
control_idx = np.where(control_mask)[0]

def nearest_neighbor_match(ps_treated, ps_control, control_indices):
    logit_t = np.log(ps_treated / (1 - ps_treated)).reshape(-1, 1)
    logit_c = np.log(ps_control / (1 - ps_control)).reshape(-1, 1)
    dist = cdist(logit_t, logit_c, metric='euclidean')
    best = np.argmin(dist, axis=1)
    return control_indices[best]

def fit_glm_formula(formula_str, data):
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    dep, indep = formula_str.split('~')
    covars = [v.strip() for v in indep.split('+')]
    X_mat = data[covars].values.astype(float)
    y_vec = data[dep.strip()].values.astype(float)
    X_aug = np.column_stack([np.ones(len(y_vec)), X_mat])
    model = sm.GLM(y_vec, X_aug, family=Binomial())
    glm_fit = model.fit(maxiter=25)
    ps = glm_fit.fittedvalues
    return np.clip(ps, 1e-6, 1 - 1e-6)

print(f"\n{'Specification':<16s} {'Method':<8s} {'Eval Bias':>12s} {'N Matched':>10s}")
print("-" * 50)

for spec_name, formula in specs.items():
    all_methods = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ps_glm = fit_glm_formula(formula, combined)
            all_methods['GLM'] = ps_glm
        except Exception as e:
            print(f'  [{spec_name}/GLM] failed: {e}')

        for mname, mkwargs in [('CBPS1', {'method': 'exact', 'two_step': True}),
                                ('CBPS2', {'method': 'over', 'two_step': True})]:
            try:
                fit = cbps.CBPS(formula=formula, data=combined,
                                att=0, verbose=0, **mkwargs)
                all_methods[mname] = np.clip(fit.fitted_values, 1e-6, 1 - 1e-6)
            except Exception as e:
                print(f'  [{spec_name}/{mname}] failed: {e}')

    for mname in ['GLM', 'CBPS1', 'CBPS2']:
        if mname not in all_methods:
            print(f"{spec_name:<16s} {mname:<8s} {'NA':>12s} {'0':>10s}")
            continue
        ps_all = all_methods[mname]
        matched_ctrl_idx = nearest_neighbor_match(
            ps_all[treated_idx], ps_all[control_idx], control_idx
        )
        re78_treated = combined.loc[treated_idx, 're78'].values
        re78_matched = combined.loc[matched_ctrl_idx, 're78'].values
        eval_bias = np.mean(re78_treated) - np.mean(re78_matched)
        n_matched = len(np.unique(matched_ctrl_idx))
        print(f"{spec_name:<16s} {mname:<8s} {eval_bias:>12.1f} {n_matched:>10d}")

# ============================================================
# Quick API test
# ============================================================
print("\n" + "=" * 70)
print("API Test: CBPS summary and balance")
print("=" * 70)

data = load_lalonde()
fit = cbps.CBPS(
    formula='treat ~ age + educ + black + hisp + married + nodegr + re74 + re75',
    data=data, att=0, method='over'
)
print(fit.summary())

from cbps import balance
bal = balance(fit)
print('\nCovariate Balance (weighted):')
print(bal['balanced'])
