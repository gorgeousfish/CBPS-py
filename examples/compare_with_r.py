"""
端到端比较：Python CBPS vs R CBPS
使用 R 包输出的倾向得分作为参考值。
"""

import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0, '..')
import cbps
from cbps.datasets.lalonde import _get_data_dir


def nearest_neighbor_match(ps_treated, ps_control, control_indices):
    logit_t = np.log(ps_treated / (1 - ps_treated)).reshape(-1, 1)
    logit_c = np.log(ps_control / (1 - ps_control)).reshape(-1, 1)
    dist = cdist(logit_t, logit_c, metric='euclidean')
    best = np.argmin(dist, axis=1)
    return control_indices[best]


def run_compare():
    data_dir = _get_data_dir()
    full = pd.read_csv(data_dir / 'LaLonde.csv')

    nsw_ctrl = full[(full['exper'] == 1) & (full['treat'] == 0)].copy()
    psid = full[full['exper'] == 0].copy()

    nsw_ctrl['select'] = 1
    psid['select'] = 0
    combined = pd.concat([nsw_ctrl, psid], ignore_index=True)

    combined['age_sq'] = combined['age'] ** 2
    combined['educ_sq'] = combined['educ'] ** 2
    combined['re75_sq'] = combined['re75'] ** 2
    combined['re74_sq'] = combined['re74'] ** 2

    treated_mask = combined['select'] == 1
    control_mask = combined['select'] == 0
    treated_idx = np.where(treated_mask)[0]
    control_idx = np.where(control_mask)[0]

    # 读取 R 结果
    r_ps = pd.read_csv('../../CBPS_R/lalonde_r_ps.csv')

    specs = {
        'Linear': {
            'formula': 'select ~ age + educ + black + hisp + married + nodegr + re75',
            'r_exact_col': 'linear_exact_ps',
            'r_over_col': 'linear_over_ps',
        },
        'Smith-Todd': {
            'formula': 'select ~ age + educ + black + hisp + married + nodegr + re74 + re75 + age_sq + educ_sq + re74_sq + re75_sq',
            'r_exact_col': 'st_exact_ps',
            'r_over_col': 'st_over_ps',
        },
    }

    for spec_name, spec_info in specs.items():
        print("=" * 80)
        print(f"规格: {spec_name}")
        print("=" * 80)

        formula = spec_info['formula']
        r_ps_exact = r_ps[spec_info['r_exact_col']].values
        r_ps_over = r_ps[spec_info['r_over_col']].values

        # Python CBPS1 (exact)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fit1 = cbps.CBPS(formula=formula, data=combined,
                                 att=0, method='exact', verbose=0)
                py_ps_exact = np.clip(fit1.fitted_values, 1e-6, 1 - 1e-6)
            except Exception as e:
                print(f"  CBPS1 失败: {e}")
                py_ps_exact = None

        # Python CBPS2 (over)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fit2 = cbps.CBPS(formula=formula, data=combined,
                                 att=0, method='over', verbose=0)
                py_ps_over = np.clip(fit2.fitted_values, 1e-6, 1 - 1e-6)
            except Exception as e:
                print(f"  CBPS2 失败: {e}")
                py_ps_over = None

        # 比较倾向得分
        for method, py_ps, r_ps_ref in [
            ('CBPS1 (exact)', py_ps_exact, r_ps_exact),
            ('CBPS2 (over)', py_ps_over, r_ps_over),
        ]:
            if py_ps is None:
                print(f"\n  {method}: Python 失败")
                continue

            diff = py_ps - r_ps_ref
            corr = np.corrcoef(py_ps, r_ps_ref)[0, 1]
            max_diff = np.max(np.abs(diff))
            mean_diff = np.mean(np.abs(diff))

            print(f"\n  {method}:")
            print(f"    Python PS 范围: [{py_ps.min():.6f}, {py_ps.max():.6f}]")
            print(f"    R PS 范围:      [{r_ps_ref.min():.6f}, {r_ps_ref.max():.6f}]")
            print(f"    相关系数: {corr:.6f}")
            print(f"    最大绝对差异: {max_diff:.6f}")
            print(f"    平均绝对差异: {mean_diff:.6f}")

            # 评估偏差比较
            for label, ps_vec in [('Python', py_ps), ('R', r_ps_ref)]:
                ps_clipped = np.clip(ps_vec, 1e-6, 1 - 1e-6)
                matched_ctrl_idx = nearest_neighbor_match(
                    ps_clipped[treated_idx], ps_clipped[control_idx], control_idx
                )
                re78_treated = combined.iloc[treated_idx]['re78'].values
                re78_matched = combined.iloc[matched_ctrl_idx]['re78'].values
                eval_bias = np.mean(re78_treated) - np.mean(re78_matched)
                print(f"    {label} 评估偏差: {eval_bias:.1f}")


if __name__ == '__main__':
    run_compare()
