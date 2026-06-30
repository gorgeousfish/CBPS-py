"""快速运行Table 2并与论文和R对比"""
import sys
sys.path.insert(0, '/Users/cxy/Desktop/cbps/CBPS_python')
sys.path.insert(0, '/Users/cxy/Desktop/cbps/CBPS_python/examples')

from replicate_imai_ratkovic_2014 import run_lalonde_analysis, print_table2

df = run_lalonde_analysis(seed=2014)
print_table2(df)

print("\n\nR包结果 (同样的1-to-1 NN matching on logit PS):")
print(f"{'Specification':<16s} {'Method':<8s} {'R Eval Bias':>12s} {'Py Eval Bias':>12s}")
print("-" * 52)

r_results = {
    ('Linear', 'GLM'): -756.5,
    ('Linear', 'CBPS1'): -2013.4,
    ('Linear', 'CBPS2'): -1036.8,
    ('Quadratic', 'GLM'): 5090.0,
    ('Quadratic', 'CBPS1'): 5090.0,
    ('Quadratic', 'CBPS2'): 5090.0,
    ('Smith-Todd', 'GLM'): -1416.9,
    ('Smith-Todd', 'CBPS1'): -1455.8,
    ('Smith-Todd', 'CBPS2'): -1378.3,
}

for _, row in df.iterrows():
    key = (row['Specification'], row['Method'])
    r_val = r_results.get(key, float('nan'))
    py_val = row['Eval Bias']
    import numpy as np
    r_str = f"{r_val:>12.1f}" if np.isfinite(r_val) else f"{'NA':>12s}"
    py_str = f"{py_val:>12.1f}" if np.isfinite(py_val) else f"{'NA':>12s}"
    print(f"{row['Specification']:<16s} {row['Method']:<8s} {r_str} {py_str}")
