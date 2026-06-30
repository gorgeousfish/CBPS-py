# Examples 目录规范化方案

## 一、现状分析

当前 `examples/` 目录共 22 个文件（4 个 notebook + 13 个 .py 脚本 + 4 个 sklearn 相关 + 1 个 README），存在以下问题：

1. **大量重复**：每个 `*_basic.py` 与对应 `tutorial_*.ipynb` 高度重叠；`sklearn_complete_demo.py` 完全包含 `sklearn_gridsearch.py` 和 `sklearn_pipeline.py`
2. **AI 味严重**：几乎所有文件都有 `print("=" * 70)` 分隔线 + "Example completed successfully!" 结尾；大量 section 是纯打印文本
3. **内部开发痕迹泄露**：`npcbps_basic.py` 中 "deferred to Story 1.12"；`cbmsm_basic.py` 中 "Verify Bug Fix"；`continuous_treatment_diagnostics_example.py` 中 "Date: 2026-01-28"
4. **无真正的论文复现**：现有文件使用论文数据集但不复现论文中的具体表格/图形
5. **notebook 未清理**：`tutorial_continuous.ipynb` 包含 ModuleNotFoundError 错误输出

## 二、五篇论文实证部分概要

### 论文 1：Imai & Ratkovic (2014), JRSS-B — 原始 CBPS
- **模拟**：Kang-Schafer (2007) 设定，4 种场景（正确/错误指定），n=200/1000，10000 次 MC
- **实证**：LaLonde 职业培训数据（NSW 实验组 297 + PSID 对照组 2490），评估偏差估计
- **可复现**：Table 1（模拟结果）、Table 2（评估偏差）、Table 3（协变量平衡）
- **数据**：✅ `load_lalonde_psid_combined()` 可用

### 论文 2：Fong, Hazlett & Imai (2018), AOAS — 连续处理 CBPS
- **模拟**：4 种 DGP（线性/非线性处理×线性/非线性结果），n=200，500 次 MC
- **实证**：Political Ads 数据（Urban & Niebler 2014），16265 个邮编，Box-Cox λ=-0.16
- **可复现**：Figure 1（F 统计量平衡）、Figure 2（ATE 估计分布）、Table 1（逐变量相关系数）、Figure 3（平衡指标）
- **数据**：✅ `load_political_ads()` 和 `load_continuous_simulation(dgp=1..4)` 可用

### 论文 3：Imai & Ratkovic (2015), JASA — MSM CBPS
- **模拟**：4 种场景（滞后结构×函数形式），J=3 时期，n=500/1000/2500/5000
- **实证**：Blackwell (2013) 负面广告数据，114 场选举，5 周，1548 个平衡条件
- **可复现**：Figure 4（平衡条件散点图）、Table 3（负面广告对得票率的影响）
- **数据**：✅ `load_blackwell()` 可用

### 论文 4：Ning, Peng & Imai (2020), Biometrika — 高维 CBPS
- **模拟**：高维设定 d >> n，正确/错误指定场景
- **实证**：Political Socialization Panel Study（Kam & Palmer 2008），1051 名高中毕业生，204 个预处理变量
- **可复现**：模拟部分可复现；实证部分需要外部数据
- **数据**：⚠️ Political Socialization Panel Study 数据**不在包内**，需要单独获取

### 论文 5：Fan, Imai, Lee, Liu, Ning & Yang (2021), JBES — 最优 CBPS
- **模拟**：正确/错误指定场景，与标准 CBPS 对比
- **实证**：LaLonde/Dehejia-Wahba 数据（NSW 297 + PSID 2490），三种协变量规格（linear/quadratic/Smith-Todd）
- **可复现**：ATT 估计与实验基准（$886）的比较
- **数据**：✅ `load_lalonde_psid_combined()` 可用

## 三、缺失数据清单

| 论文 | 数据集 | 状态 | 说明 |
|------|--------|------|------|
| hdCBPS (2020) | Political Socialization Panel Study | ❌ 缺失 | Kam & Palmer (2008) 使用的数据，1051 名 1965 届高中毕业生，204 个预处理变量。需从 ICPSR 或原作者处获取。该数据集原始来源为 Youth-Parent Socialization Panel Study (Jennings & Niemi)，ICPSR Study #4037 |

## 四、规范化后的文件结构

### 保留并重写（8 个文件）

```
examples/
├── README.md                              # 重写：简洁的索引
├── replicate_imai_ratkovic_2014.py        # 论文1复现：二元CBPS
├── replicate_fong_hazlett_imai_2018.py    # 论文2复现：连续CBPS
├── replicate_imai_ratkovic_2015.py        # 论文3复现：MSM CBPS
├── replicate_ning_peng_imai_2020.py       # 论文4复现：高维CBPS（仅模拟）
├── replicate_fan_et_al_2021.py            # 论文5复现：最优CBPS
├── quickstart.py                          # 快速入门：覆盖二元/连续/MSM
└── sklearn_integration.py                 # sklearn 集成演示
```

### 删除（14 个文件）

| 文件 | 删除原因 |
|------|----------|
| `cbps_basic.py` | 被 quickstart.py + replicate_imai_ratkovic_2014.py 覆盖 |
| `cbmsm_basic.py` | 被 replicate_imai_ratkovic_2015.py 覆盖，含 "Verify Bug Fix" |
| `hdcbps_basic.py` | 被 replicate_ning_peng_imai_2020.py 覆盖 |
| `npcbps_basic.py` | 被 replicate_fong_hazlett_imai_2018.py 覆盖，含 "Story 1.12" |
| `cbiv_basic.py` | CBIV 为未发表手稿方法，不属于五篇核心论文，移至 quickstart 中简要提及 |
| `balance_basic.py` | 被论文复现脚本中的平衡诊断覆盖 |
| `balance_continuous.py` | 被论文复现脚本中的平衡诊断覆盖 |
| `plot_cbps.py` | 被论文复现脚本中的可视化覆盖 |
| `plot_continuous.py` | 被论文复现脚本中的可视化覆盖 |
| `summary_cbps.py` | 与 vcov_cbps.py 高度重叠，被论文复现覆盖 |
| `vcov_cbps.py` | 与 summary_cbps.py 高度重叠，被论文复现覆盖 |
| `vcov_outcome_basic.py` | 被论文复现脚本覆盖 |
| `asyvar_comprehensive_demo.py` | 测试风格，被论文复现覆盖 |
| `sklearn_gridsearch.py` | 被 sklearn_integration.py 完全覆盖 |
| `sklearn_pipeline.py` | 被 sklearn_integration.py 完全覆盖 |
| `continuous_treatment_diagnostics_example.py` | 使用内部 API，含未来日期 |
| `tutorial_binary.ipynb` | 被 replicate + quickstart 覆盖 |
| `tutorial_continuous.ipynb` | 被 replicate + quickstart 覆盖，含错误输出 |
| `tutorial_hdcbps.ipynb` | 被 replicate 覆盖 |
| `tutorial_msm.ipynb` | 被 replicate + quickstart 覆盖 |

## 五、写作规范

1. **语言**：全英文，学术风格，无 AI 味
2. **格式**：每个文件开头有论文引用和简要说明，无 `print("=" * 70)` 分隔线
3. **内容**：复现论文中的具体表格/图形，标注对应论文中的 Table/Figure 编号
4. **代码风格**：简洁专业，使用 `if __name__ == "__main__"` 入口，合理的函数封装
5. **输出**：结构化的表格输出，便于与论文结果对比
