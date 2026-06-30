"""测试 vmmin BFGS 实现是否匹配 R 的 optim(BFGS) 结果。

关键对比：
- Python vmmin 的 bal_loss 最终值 vs R 的 9.543865e-05
- Python vmmin 的迭代次数 vs R 的 75 function evals / 73 gradient evals
- 逆变换后的 beta_orig vs R 的 beta_orig
"""
import warnings
import numpy as np
import pandas as pd
import scipy.special
import scipy.optimize
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
import sys
sys.path.insert(0, '/Users/cxy/Desktop/cbps/CBPS_python')

from cbps.core.cbps_binary import (
    _r_ginv, _bal_loss, _bal_gradient, _vmmin_bfgs, PROBS_MIN
)

# ===== R 参考值 =====
R_bal_loss_opt = 9.543865e-05
R_beta_orig = np.array([3.444255, -0.1460451, -0.1375703, 2.260167,
                         2.136003, -2.209607, 1.393202, -7.924812e-05])
R_alpha = 0.927223
R_nfev = 75
R_ngrad = 73

# ===== 加载数据 =====
full = pd.read_csv('/Users/cxy/Desktop/cbps/CBPS_python/data/LaLonde.csv')
nsw_ctrl = full[(full['exper'] == 1) & (full['treat'] == 0)].copy()
psid = full[full['exper'] == 0].copy()
nsw_ctrl['select'] = 1
psid['select'] = 0
combined = pd.concat([nsw_ctrl, psid], ignore_index=True)

treat = combined['select'].values.astype(float)
covars = combined[['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're75']].values
X_raw = np.column_stack([np.ones(len(treat)), covars])
n = len(treat)
k = X_raw.shape[1]

# ===== SVD 预处理 =====
X_work = X_raw.copy()
x_sd = X_work[:, 1:].std(axis=0, ddof=1)
x_mean = X_work[:, 1:].mean(axis=0)
X_work[:, 1:] = (X_work[:, 1:] - x_mean) / x_sd

U, s, Vt = np.linalg.svd(X_work, full_matrices=False)
V_matrix = Vt.T
X_svd = U

# ===== GLM 初始化 =====
sample_weights = np.ones(n)
att = 0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = sm.GLM(treat, X_svd, family=Binomial())
    glm_fit = model.fit(tol=1e-8, maxiter=25)
glm_beta = glm_fit.params.copy()
glm_beta[np.isnan(glm_beta)] = 0

# ===== XprimeX_inv =====
sw_sqrt = np.sqrt(sample_weights)
XprimeX = (sw_sqrt[:, None] * X_svd).T @ (sw_sqrt[:, None] * X_svd)
XprimeX_inv = _r_ginv(XprimeX)

# ===== Alpha scaling (完整 GMM loss) =====
def gmm_loss_full(beta):
    theta = X_svd @ beta
    probs = scipy.special.expit(theta)
    probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)
    w = (probs - 1 + treat) ** (-1)
    w_del = (1/n) * (sample_weights[:, None] * X_svd).T @ w
    g_score = (1/n) * (sample_weights[:, None] * X_svd).T @ (treat - probs)
    g_bar = np.concatenate([g_score, w_del])
    X1 = np.sqrt(sample_weights)[:, None] * X_svd * np.sqrt((1-probs)*probs)[:, None]
    X2 = np.sqrt(sample_weights)[:, None] * X_svd * (1/np.sqrt(probs*(1-probs)))[:, None]
    X11 = np.sqrt(sample_weights)[:, None] * X_svd
    V_mat = np.block([
        [(1/n) * X1.T @ X1, (1/n) * X11.T @ X11],
        [(1/n) * X11.T @ X11, (1/n) * X2.T @ X2]
    ])
    invV = _r_ginv(V_mat)
    return float(g_bar.T @ invV @ g_bar)

alpha_func = lambda a: gmm_loss_full(glm_beta * a)
alpha_result = scipy.optimize.minimize_scalar(alpha_func, bounds=(0.8, 1.1), method='bounded')
alpha_opt = alpha_result.x
gmm_init = glm_beta * alpha_opt

print(f"n={n}, k={k}")
print(f"alpha: {alpha_opt:.6f} (R: {R_alpha})")

# ===== 辅助函数：SVD 逆变换 =====
def svd_to_orig(beta_svd):
    """将 SVD 空间的系数转换回原始空间"""
    d_inv = np.where(s > 1e-5, 1.0 / s, 0.0)
    beta = V_matrix @ np.diag(d_inv) @ beta_svd
    beta[1:] = beta[1:] / x_sd
    beta[0] = beta[0] - np.sum(x_mean * beta[1:])
    return beta

# ===== 测试 1: vmmin BFGS =====
print("\n" + "="*60)
print("测试 1: vmmin BFGS (R's optim 忠实翻译)")
print("="*60)

bal_loss_func = lambda b: _bal_loss(b, X_svd, treat, sample_weights, XprimeX_inv, att)
bal_grad_func = lambda b: _bal_gradient(b, X_svd, treat, sample_weights, XprimeX_inv, att)

opt_vmmin = _vmmin_bfgs(
    gmm_init,
    fn=bal_loss_func,
    gr=bal_grad_func,
    maxit=10000,
    trace=False,
)

beta_orig_vmmin = svd_to_orig(opt_vmmin.x)
orig_diff_vmmin = np.max(np.abs(beta_orig_vmmin - R_beta_orig))

print(f"  loss:     {opt_vmmin.fun:.10f}  (R: {R_bal_loss_opt:.10f})")
print(f"  nfev:     {opt_vmmin.nfev}  (R: {R_nfev})")
print(f"  njev:     {opt_vmmin.njev}  (R: {R_ngrad})")
print(f"  nit:      {opt_vmmin.nit}")
print(f"  success:  {opt_vmmin.success}")
print(f"  beta_orig: {beta_orig_vmmin}")
print(f"  R beta:    {R_beta_orig}")
print(f"  orig_diff: {orig_diff_vmmin:.8f}")

# ===== 测试 2: scipy BFGS (对比) =====
print("\n" + "="*60)
print("测试 2: scipy BFGS (gtol=1e-6, 旧方法)")
print("="*60)

opt_scipy = scipy.optimize.minimize(
    bal_loss_func, gmm_init, method='BFGS', jac=bal_grad_func,
    options={'maxiter': 10000, 'gtol': 1e-6}
)

beta_orig_scipy = svd_to_orig(opt_scipy.x)
orig_diff_scipy = np.max(np.abs(beta_orig_scipy - R_beta_orig))

print(f"  loss:     {opt_scipy.fun:.10f}  (R: {R_bal_loss_opt:.10f})")
print(f"  nfev:     {opt_scipy.nfev}")
print(f"  nit:      {opt_scipy.nit}")
print(f"  orig_diff: {orig_diff_scipy:.8f}")

# ===== 测试 3: 通过 CBPS 接口调用 =====
print("\n" + "="*60)
print("测试 3: 通过 _optimize_balance 接口调用 vmmin")
print("="*60)

from cbps.core.cbps_binary import _optimize_balance

opt_interface = _optimize_balance(
    gmm_init, X_svd, treat, sample_weights, XprimeX_inv,
    att=att, two_step=True, iterations=10000
)

beta_orig_interface = svd_to_orig(opt_interface.x)
orig_diff_interface = np.max(np.abs(beta_orig_interface - R_beta_orig))

print(f"  loss:     {opt_interface.fun:.10f}")
print(f"  nfev:     {opt_interface.nfev}")
print(f"  orig_diff: {orig_diff_interface:.8f}")

# ===== 总结 =====
print("\n" + "="*60)
print("总结")
print("="*60)
print(f"  vmmin orig_diff:  {orig_diff_vmmin:.8f}")
print(f"  scipy orig_diff:  {orig_diff_scipy:.8f}")
print(f"  改善倍数:         {orig_diff_scipy / orig_diff_vmmin:.1f}x" if orig_diff_vmmin > 0 else "  改善倍数: inf")
print(f"  vmmin loss diff:  {abs(opt_vmmin.fun - R_bal_loss_opt):.2e}")
print(f"  scipy loss diff:  {abs(opt_scipy.fun - R_bal_loss_opt):.2e}")
