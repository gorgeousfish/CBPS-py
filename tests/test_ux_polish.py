"""
UX Polish Test Suite
====================

验证 CBPS Python 包用户体验优化的正确性。
本测试文件覆盖 ux-polish spec 中的各项需求。

测试按需求分组：
- 需求 3：Export_List 清理（移除私有函数）
"""

import pytest


# =============================================================================
# 需求 3：Export_List 清理
# =============================================================================

@pytest.mark.unit
class TestExportListCleanup:
    """验证 __all__ 导出列表不包含私有函数，同时保持向后兼容。

    需求 3.1: __all__ 不包含任何以下划线开头的函数名
    需求 3.2: from cbps import * 不导出 _detect_treatment_type
    需求 3.3: from cbps import _detect_treatment_type 显式导入仍可用
    """

    def test_all_has_no_underscore_entries(self):
        """__all__ 中不应包含任何以 '_' 开头的条目。

        Validates: Requirements 3.1
        """
        import cbps
        for name in cbps.__all__:
            assert not name.startswith('_'), (
                f"cbps.__all__ 不应包含私有名称 '{name}'，"
                f"以下划线开头的函数不应出现在公共导出列表中"
            )

    def test_detect_treatment_type_not_in_all(self):
        """_detect_treatment_type 不应出现在 __all__ 中。

        Validates: Requirements 3.1, 3.2
        """
        import cbps
        assert '_detect_treatment_type' not in cbps.__all__, (
            "'_detect_treatment_type' 不应出现在 cbps.__all__ 中"
        )

    def test_detect_treatment_type_explicit_import(self):
        """_detect_treatment_type 仍可通过显式导入使用。

        Validates: Requirements 3.3
        """
        from cbps import _detect_treatment_type
        assert callable(_detect_treatment_type), (
            "_detect_treatment_type 应为可调用对象"
        )


# =============================================================================
# 需求 5：AsyVar 键名 Python 化（Property 5）
# =============================================================================

@pytest.mark.property
class TestAsyVarKeyBidirectionalCompat:
    """Property 5: AsyVar 键名双向兼容

    验证 AsyVar 返回字典同时包含 R 风格和 snake_case 键名，
    且对应新旧键名指向相同的值对象。

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    # R 风格 → snake_case 的键名映射
    KEY_MAPPING = {
        'mu.hat': 'mu_hat',
        'asy.var': 'asy_var',
        'CI.mu.hat': 'ci_mu_hat',
        'std.err': 'std_err',
    }

    @pytest.fixture(scope="class")
    def asyvar_result(self):
        """使用 lalonde 数据集拟合 CBPS 并调用 AsyVar 获取结果。"""
        from cbps import CBPS, AsyVar
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = CBPS('treat ~ age + educ + black + hisp', data=data, att=0)
        result = AsyVar(Y=data['re78'].values, CBPS_obj=fit, method="oCBPS")
        return result

    def test_r_style_keys_present(self, asyvar_result):
        """返回字典应包含所有 R 风格键名。

        **Validates: Requirements 5.2**
        """
        for r_key in self.KEY_MAPPING.keys():
            assert r_key in asyvar_result, (
                f"AsyVar 返回字典缺少 R 风格键名 '{r_key}'"
            )

    def test_snake_case_keys_present(self, asyvar_result):
        """返回字典应包含所有 snake_case 键名。

        **Validates: Requirements 5.1**
        """
        for snake_key in self.KEY_MAPPING.values():
            assert snake_key in asyvar_result, (
                f"AsyVar 返回字典缺少 snake_case 键名 '{snake_key}'"
            )

    def test_var_key_still_exists(self, asyvar_result):
        """'var' 键（已是 snake_case）应仍然存在。

        **Validates: Requirements 5.1**
        """
        assert 'var' in asyvar_result, (
            "AsyVar 返回字典缺少 'var' 键"
        )

    def test_old_new_keys_identity(self, asyvar_result):
        """对应的 R 风格和 snake_case 键名应指向同一值对象（is 检查）。

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        for r_key, snake_key in self.KEY_MAPPING.items():
            assert asyvar_result[r_key] is asyvar_result[snake_key], (
                f"'{r_key}' 和 '{snake_key}' 应指向同一对象，"
                f"但 id({r_key})={id(asyvar_result[r_key])} != "
                f"id({snake_key})={id(asyvar_result[snake_key])}"
            )

    def test_old_new_keys_equal_values(self, asyvar_result):
        """对应的 R 风格和 snake_case 键名的值应相等。

        **Validates: Requirements 5.2, 5.3**
        """
        import numpy as np
        for r_key, snake_key in self.KEY_MAPPING.items():
            old_val = asyvar_result[r_key]
            new_val = asyvar_result[snake_key]
            if isinstance(old_val, np.ndarray):
                np.testing.assert_array_equal(old_val, new_val,
                    err_msg=f"'{r_key}' 和 '{snake_key}' 的数组值不相等")
            else:
                assert old_val == new_val, (
                    f"'{r_key}'={old_val} != '{snake_key}'={new_val}"
                )

    def test_values_are_numerically_reasonable(self, asyvar_result):
        """AsyVar 返回值应在数值上合理（非 NaN、非 Inf）。

        **Validates: Requirements 5.1, 5.3**
        """
        import numpy as np

        # mu_hat 应为有限数值
        assert np.isfinite(asyvar_result['mu_hat']), (
            f"mu_hat={asyvar_result['mu_hat']} 不是有限数值"
        )

        # asy_var 应为非负有限数值
        assert np.isfinite(asyvar_result['asy_var']), (
            f"asy_var={asyvar_result['asy_var']} 不是有限数值"
        )
        assert asyvar_result['asy_var'] >= 0, (
            f"asy_var={asyvar_result['asy_var']} 应为非负值"
        )

        # std_err 应为非负有限数值
        assert np.isfinite(asyvar_result['std_err']), (
            f"std_err={asyvar_result['std_err']} 不是有限数值"
        )
        assert asyvar_result['std_err'] >= 0, (
            f"std_err={asyvar_result['std_err']} 应为非负值"
        )

        # ci_mu_hat 应为长度 2 的数组，下界 < 上界
        ci = asyvar_result['ci_mu_hat']
        assert len(ci) == 2, f"ci_mu_hat 长度应为 2，实际为 {len(ci)}"
        assert ci[0] < ci[1], (
            f"ci_mu_hat 下界 {ci[0]} 应小于上界 {ci[1]}"
        )


# =============================================================================
# 需求 1：summary() 返回类型统一（Property 1）
# =============================================================================

@pytest.mark.property
class TestSummaryReturnTypeConsistency:
    """Property 1: summary() 返回类型一致性

    对 CBMSMResults 和 NPCBPSResults 实例验证 summary() 返回非字符串对象
    且具有 __str__ 方法，str(result.summary()) 返回非空字符串。

    **Validates: Requirements 1.1, 1.2, 1.5**
    """

    @pytest.fixture(scope="class")
    def cbmsm_result(self):
        """使用 blackwell 数据集拟合 CBMSM 模型。"""
        from cbps import CBMSM
        from cbps.datasets import load_blackwell

        blackwell = load_blackwell()
        fit = CBMSM(
            "d.gone.neg ~ d.gone.neg.l1 + camp.length",
            id="demName",
            time="time",
            data=blackwell,
        )
        return fit

    @pytest.fixture(scope="class")
    def npcbps_result(self):
        """使用 lalonde 数据集拟合 npCBPS 模型。"""
        from cbps import npCBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = npCBPS('treat ~ age + educ + black + hisp', data=data)
        return fit

    # --- CBMSMResults tests ---

    def test_cbmsm_summary_is_not_string(self, cbmsm_result):
        """CBMSMResults.summary() 应返回非字符串对象。

        **Validates: Requirements 1.1**
        """
        summary = cbmsm_result.summary()
        assert not isinstance(summary, str), (
            "CBMSMResults.summary() 不应返回字符串，"
            f"实际返回类型为 {type(summary).__name__}"
        )

    def test_cbmsm_summary_has_str_method(self, cbmsm_result):
        """CBMSMResults.summary() 返回的对象应具有 __str__ 方法。

        **Validates: Requirements 1.1, 1.5**
        """
        summary = cbmsm_result.summary()
        assert hasattr(summary, '__str__'), (
            "CBMSMResults.summary() 返回的对象缺少 __str__ 方法"
        )

    def test_cbmsm_summary_str_nonempty(self, cbmsm_result):
        """str(CBMSMResults.summary()) 应返回非空字符串。

        **Validates: Requirements 1.1, 1.5**
        """
        summary = cbmsm_result.summary()
        text = str(summary)
        assert isinstance(text, str), (
            f"str(summary) 应返回字符串，实际为 {type(text).__name__}"
        )
        assert len(text.strip()) > 0, (
            "str(CBMSMResults.summary()) 不应为空字符串"
        )

    def test_cbmsm_summary_contains_expected_keywords(self, cbmsm_result):
        """CBMSMResults.summary() 文本应包含 CBMSM 特征关键词。

        **Validates: Requirements 1.1, 1.5**
        """
        text = str(cbmsm_result.summary())
        assert "CBMSM" in text, "摘要文本应包含 'CBMSM' 标识"
        assert "Convergence" in text or "converge" in text.lower(), (
            "摘要文本应包含收敛状态信息"
        )

    def test_cbmsm_summary_type_name(self, cbmsm_result):
        """CBMSMResults.summary() 应返回 CBMSMSummary 类型。

        **Validates: Requirements 1.1**
        """
        from cbps.msm.cbmsm import CBMSMSummary

        summary = cbmsm_result.summary()
        assert isinstance(summary, CBMSMSummary), (
            f"CBMSMResults.summary() 应返回 CBMSMSummary 实例，"
            f"实际为 {type(summary).__name__}"
        )

    # --- NPCBPSResults tests ---

    def test_npcbps_summary_is_not_string(self, npcbps_result):
        """NPCBPSResults.summary() 应返回非字符串对象。

        **Validates: Requirements 1.2**
        """
        summary = npcbps_result.summary()
        assert not isinstance(summary, str), (
            "NPCBPSResults.summary() 不应返回字符串，"
            f"实际返回类型为 {type(summary).__name__}"
        )

    def test_npcbps_summary_has_str_method(self, npcbps_result):
        """NPCBPSResults.summary() 返回的对象应具有 __str__ 方法。

        **Validates: Requirements 1.2, 1.5**
        """
        summary = npcbps_result.summary()
        assert hasattr(summary, '__str__'), (
            "NPCBPSResults.summary() 返回的对象缺少 __str__ 方法"
        )

    def test_npcbps_summary_str_nonempty(self, npcbps_result):
        """str(NPCBPSResults.summary()) 应返回非空字符串。

        **Validates: Requirements 1.2, 1.5**
        """
        summary = npcbps_result.summary()
        text = str(summary)
        assert isinstance(text, str), (
            f"str(summary) 应返回字符串，实际为 {type(text).__name__}"
        )
        assert len(text.strip()) > 0, (
            "str(NPCBPSResults.summary()) 不应为空字符串"
        )

    def test_npcbps_summary_contains_expected_keywords(self, npcbps_result):
        """NPCBPSResults.summary() 文本应包含 npCBPS 特征关键词。

        **Validates: Requirements 1.2, 1.5**
        """
        text = str(npcbps_result.summary())
        assert "npCBPS" in text or "CBPS" in text, (
            "摘要文本应包含 'npCBPS' 或 'CBPS' 标识"
        )

    def test_npcbps_summary_type_name(self, npcbps_result):
        """NPCBPSResults.summary() 应返回 NPCBPSSummary 类型。

        **Validates: Requirements 1.2**
        """
        from cbps.nonparametric.npcbps import NPCBPSSummary

        summary = npcbps_result.summary()
        assert isinstance(summary, NPCBPSSummary), (
            f"NPCBPSResults.summary() 应返回 NPCBPSSummary 实例，"
            f"实际为 {type(summary).__name__}"
        )

    # --- Cross-class consistency ---

    def test_both_summaries_are_printable(self, cbmsm_result, npcbps_result):
        """两个 Result 类的 summary() 都应支持 print() 调用。

        **Validates: Requirements 1.1, 1.2, 1.5**
        """
        for name, result in [("CBMSM", cbmsm_result), ("npCBPS", npcbps_result)]:
            summary = result.summary()
            # print() 内部调用 str()，不应抛出异常
            text = str(summary)
            assert text, f"{name} 的 summary() 对象 str() 不应返回空值"


# =============================================================================
# 需求 1.6：summary() 文本内容向后兼容（Property 2）
# =============================================================================

@pytest.mark.property
class TestSummaryTextBackwardCompat:
    """Property 2: summary() 文本内容向后兼容

    对 CBMSMResults 和 NPCBPSResults 验证 str(result.summary()) 输出
    与旧实现（直接返回字符串）的内容一致。旧实现的字符串构建逻辑已被
    完整移入 Summary 类的 __str__() 方法，因此通过验证输出包含所有
    预期的结构化段落和格式元素来确保向后兼容。

    **Validates: Requirements 1.6**
    """

    # ---- Fixtures (reuse blackwell / lalonde datasets) ----

    @pytest.fixture(scope="class")
    def cbmsm_result(self):
        """使用 blackwell 数据集拟合 CBMSM 模型。"""
        from cbps import CBMSM
        from cbps.datasets import load_blackwell

        blackwell = load_blackwell()
        fit = CBMSM(
            "d.gone.neg ~ d.gone.neg.l1 + camp.length",
            id="demName",
            time="time",
            data=blackwell,
        )
        return fit

    @pytest.fixture(scope="class")
    def npcbps_result(self):
        """使用 lalonde 数据集拟合 npCBPS 模型。"""
        from cbps import npCBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = npCBPS('treat ~ age + educ + black + hisp', data=data)
        return fit

    # ================================================================
    # CBMSMResults: 验证 summary 文本包含旧实现的所有结构化段落
    # ================================================================

    def test_cbmsm_summary_contains_header(self, cbmsm_result):
        """CBMSM summary 文本应包含标题行和分隔线。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        assert "=" * 70 in text, "摘要文本应包含 '=' * 70 分隔线"
        assert "CBMSM" in text, "摘要文本应包含 'CBMSM' 标题"

    def test_cbmsm_summary_contains_basic_info(self, cbmsm_result):
        """CBMSM summary 文本应包含基本信息段落。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        assert "Number of observations" in text, (
            "摘要文本应包含 'Number of observations'"
        )
        assert "Number of time periods" in text, (
            "摘要文本应包含 'Number of time periods'"
        )
        assert "Number of units" in text, (
            "摘要文本应包含 'Number of units'"
        )
        assert "Time-varying treatment model" in text, (
            "摘要文本应包含 'Time-varying treatment model'"
        )
        assert "Convergence" in text, (
            "摘要文本应包含 'Convergence'"
        )
        assert "J-statistic" in text, (
            "摘要文本应包含 'J-statistic'"
        )

    def test_cbmsm_summary_contains_propensity_scores(self, cbmsm_result):
        """CBMSM summary 文本应包含倾向得分摘要段落。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        assert "Propensity Scores P(T|X) Summary" in text, (
            "摘要文本应包含 'Propensity Scores P(T|X) Summary'"
        )
        # 旧实现包含 Min/1Q/Median/Mean/3Q/Max 统计量
        for stat in ["Min:", "1Q:", "Median:", "Mean:", "3Q:", "Max:"]:
            assert stat in text, f"倾向得分摘要应包含 '{stat}'"

    def test_cbmsm_summary_contains_msm_weights(self, cbmsm_result):
        """CBMSM summary 文本应包含 MSM 权重摘要段落。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        assert "MSM Weights (Stabilized) Summary" in text, (
            "摘要文本应包含 'MSM Weights (Stabilized) Summary'"
        )

    def test_cbmsm_summary_contains_coefficients(self, cbmsm_result):
        """CBMSM summary 文本应包含系数段落。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        assert "Coefficients" in text, (
            "摘要文本应包含 'Coefficients'"
        )
        assert "beta[" in text, (
            "摘要文本应包含 'beta[' 系数标签"
        )

    def test_cbmsm_summary_numeric_values_present(self, cbmsm_result):
        """CBMSM summary 文本中的数值应与 result 属性一致。

        **Validates: Requirements 1.6**
        """
        import numpy as np

        result = cbmsm_result
        text = str(result.summary())

        # 验证 n_units 和 n_periods 数值出现在文本中
        assert str(result.n_units) in text, (
            f"摘要文本应包含 n_units={result.n_units}"
        )
        assert str(result.n_periods) in text, (
            f"摘要文本应包含 n_periods={result.n_periods}"
        )

        # 验证 J-statistic 数值（6位小数格式）
        j_str = f"{result.J:.6f}"
        assert j_str in text, (
            f"摘要文本应包含 J-statistic 值 '{j_str}'"
        )

        # 验证权重统计量数值
        w_min = f"{np.min(result.weights):.6f}"
        w_max = f"{np.max(result.weights):.6f}"
        assert w_min in text, f"摘要文本应包含权重最小值 '{w_min}'"
        assert w_max in text, f"摘要文本应包含权重最大值 '{w_max}'"

    def test_cbmsm_summary_separator_lines(self, cbmsm_result):
        """CBMSM summary 文本应包含正确的分隔线格式。

        **Validates: Requirements 1.6**
        """
        text = str(cbmsm_result.summary())
        # 旧实现使用 "=" * 70 作为首尾分隔线，"-" * 70 作为段落分隔线
        assert text.count("=" * 70) >= 2, (
            "摘要文本应至少包含 2 条 '=' * 70 分隔线（首尾）"
        )
        assert "-" * 70 in text, (
            "摘要文本应包含 '-' * 70 段落分隔线"
        )

    def test_cbmsm_summary_idempotent(self, cbmsm_result):
        """多次调用 summary() 应产生相同的文本输出。

        **Validates: Requirements 1.6**
        """
        text1 = str(cbmsm_result.summary())
        text2 = str(cbmsm_result.summary())
        assert text1 == text2, (
            "多次调用 summary() 应产生完全相同的文本输出"
        )

    # ================================================================
    # NPCBPSResults: 验证 summary 文本包含旧实现的所有结构化段落
    # ================================================================

    def test_npcbps_summary_contains_header(self, npcbps_result):
        """npCBPS summary 文本应包含标题行和分隔线。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "=" * 70 in text, "摘要文本应包含 '=' * 70 分隔线"
        assert "npCBPS" in text, "摘要文本应包含 'npCBPS' 标题"

    def test_npcbps_summary_contains_sample_info(self, npcbps_result):
        """npCBPS summary 文本应包含样本信息段落。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "Sample size" in text, (
            "摘要文本应包含 'Sample size'"
        )
        assert "Treatment group" in text, (
            "摘要文本应包含 'Treatment group'"
        )
        assert "Control group" in text, (
            "摘要文本应包含 'Control group'"
        )

    def test_npcbps_summary_contains_convergence(self, npcbps_result):
        """npCBPS summary 文本应包含收敛诊断段落。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "Convergence Diagnostics" in text, (
            "摘要文本应包含 'Convergence Diagnostics'"
        )
        assert "Converged" in text, (
            "摘要文本应包含 'Converged'"
        )

    def test_npcbps_summary_contains_optimization(self, npcbps_result):
        """npCBPS summary 文本应包含优化结果段落。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "Optimization Results" in text, (
            "摘要文本应包含 'Optimization Results'"
        )

    def test_npcbps_summary_contains_weight_distribution(self, npcbps_result):
        """npCBPS summary 文本应包含权重分布段落。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "Weight Distribution" in text, (
            "摘要文本应包含 'Weight Distribution'"
        )
        # 旧实现包含 Min/Q1/Median/Mean/Q3/Max/Sum 统计量
        for stat in ["Min:", "Q1:", "Median:", "Mean:", "Q3:", "Max:", "Sum:"]:
            assert stat in text, f"权重分布段落应包含 '{stat}'"

        # 旧实现包含有效样本量
        assert "Effective sample size" in text, (
            "摘要文本应包含 'Effective sample size'"
        )

    def test_npcbps_summary_contains_diagnostics(self, npcbps_result):
        """npCBPS summary 文本应包含诊断段落。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert "Diagnostics:" in text, (
            "摘要文本应包含 'Diagnostics:'"
        )

    def test_npcbps_summary_numeric_values_present(self, npcbps_result):
        """npCBPS summary 文本中的数值应与 result 属性一致。

        **Validates: Requirements 1.6**
        """
        import numpy as np

        result = npcbps_result
        text = str(result.summary())

        # 验证样本量
        n = len(result.y)
        assert str(n) in text, f"摘要文本应包含样本量 {n}"

        # 验证权重统计量
        if result.weights is not None:
            w_min = f"{result.weights.min():.6f}"
            w_max = f"{result.weights.max():.6f}"
            assert w_min in text, f"摘要文本应包含权重最小值 '{w_min}'"
            assert w_max in text, f"摘要文本应包含权重最大值 '{w_max}'"

            # 验证有效样本量数值
            ess = (result.weights.sum() ** 2) / (result.weights ** 2).sum()
            ess_str = f"{ess:.1f}"
            assert ess_str in text, (
                f"摘要文本应包含有效样本量 '{ess_str}'"
            )

    def test_npcbps_summary_separator_lines(self, npcbps_result):
        """npCBPS summary 文本应包含正确的分隔线格式。

        **Validates: Requirements 1.6**
        """
        text = str(npcbps_result.summary())
        assert text.count("=" * 70) >= 2, (
            "摘要文本应至少包含 2 条 '=' * 70 分隔线（首尾）"
        )
        assert "-" * 70 in text, (
            "摘要文本应包含 '-' * 70 段落分隔线"
        )

    def test_npcbps_summary_idempotent(self, npcbps_result):
        """多次调用 summary() 应产生相同的文本输出。

        **Validates: Requirements 1.6**
        """
        text1 = str(npcbps_result.summary())
        text2 = str(npcbps_result.summary())
        assert text1 == text2, (
            "多次调用 summary() 应产生完全相同的文本输出"
        )

    # ================================================================
    # 跨类一致性：验证两个 Summary 对象的通用向后兼容属性
    # ================================================================

    def test_summary_str_matches_print_output(self, cbmsm_result, npcbps_result):
        """str(summary) 和 print(summary) 应产生相同的文本。

        旧实现中 print(result.summary()) 直接打印字符串，
        新实现中 print() 调用 __str__()，结果应一致。

        **Validates: Requirements 1.6**
        """
        import io
        import sys

        for name, result in [("CBMSM", cbmsm_result), ("npCBPS", npcbps_result)]:
            summary = result.summary()
            str_output = str(summary)

            # 捕获 print 输出
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            print(summary)
            sys.stdout = old_stdout
            print_output = captured.getvalue()

            # print() 会在末尾追加 \n
            assert print_output == str_output + "\n", (
                f"{name}: str(summary) 和 print(summary) 的输出不一致"
            )


# =============================================================================
# 需求 2：HDCBPSResults 调试属性封装（Property 3）
# =============================================================================

@pytest.mark.property
class TestHDCBPSResultsDebugEncapsulation:
    """Property 3: HDCBPSResults 调试属性封装与向后兼容

    验证 HDCBPSResults 的 debug_* 属性已封装到 _debug 字典中，
    通过旧属性名访问时返回正确值并触发 DeprecationWarning，
    dir() 不暴露 debug_* 条目，不存在的属性正确抛出 AttributeError。

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @pytest.fixture
    def result_with_debug(self):
        """构造 HDCBPSResults 实例并手动设置 _debug 字典中的值。"""
        from cbps.highdim.hdcbps import HDCBPSResults

        result = HDCBPSResults()
        result._debug['debug_r_yhat1'] = 42.0
        result._debug['debug_sigma_1'] = [1.0, 2.0, 3.0]
        result._debug['debug_r_yhat0'] = -3.14
        result._debug['debug_sigma_0'] = [4.0, 5.0]
        return result

    # ---- 需求 2.2: 旧属性名访问返回正确值并触发 DeprecationWarning ----

    def test_debug_attr_access_returns_correct_value_scalar(self, result_with_debug):
        """通过 result.debug_r_yhat1 访问应返回与 _debug 字典相同的值。

        **Validates: Requirements 2.1, 2.2**
        """
        with pytest.warns(DeprecationWarning):
            val = result_with_debug.debug_r_yhat1
        assert val == result_with_debug._debug['debug_r_yhat1']
        assert val == 42.0

    def test_debug_attr_access_returns_correct_value_list(self, result_with_debug):
        """通过 result.debug_sigma_1 访问应返回与 _debug 字典相同的列表值。

        **Validates: Requirements 2.1, 2.2**
        """
        with pytest.warns(DeprecationWarning):
            val = result_with_debug.debug_sigma_1
        assert val is result_with_debug._debug['debug_sigma_1']
        assert val == [1.0, 2.0, 3.0]

    def test_debug_attr_access_triggers_deprecation_warning(self, result_with_debug):
        """访问 debug_* 属性应触发 DeprecationWarning。

        **Validates: Requirements 2.2**
        """
        with pytest.warns(DeprecationWarning, match="已弃用"):
            _ = result_with_debug.debug_r_yhat1

    def test_debug_attr_access_warning_mentions_attribute_name(self, result_with_debug):
        """DeprecationWarning 消息应包含被访问的属性名。

        **Validates: Requirements 2.2**
        """
        with pytest.warns(DeprecationWarning, match="debug_r_yhat0"):
            _ = result_with_debug.debug_r_yhat0

    def test_all_debug_keys_accessible_via_old_attr(self, result_with_debug):
        """_debug 字典中的所有键都应可通过旧属性名访问。

        **Validates: Requirements 2.1, 2.2**
        """
        for key, expected_val in result_with_debug._debug.items():
            with pytest.warns(DeprecationWarning):
                actual_val = getattr(result_with_debug, key)
            assert actual_val is expected_val, (
                f"result.{key} 应返回 _debug['{key}'] 的值"
            )

    # ---- 需求 2.3: dir(result) 不包含 debug_* 条目 ----

    def test_dir_excludes_debug_entries(self, result_with_debug):
        """dir(result) 不应包含任何以 debug_ 开头的条目。

        **Validates: Requirements 2.3**
        """
        dir_entries = dir(result_with_debug)
        debug_entries = [e for e in dir_entries if e.startswith('debug_')]
        assert debug_entries == [], (
            f"dir(result) 不应包含 debug_* 条目，但发现: {debug_entries}"
        )

    def test_dir_still_contains_public_attrs(self, result_with_debug):
        """dir(result) 应仍然包含公共属性如 ATE、converged 等。

        **Validates: Requirements 2.3**
        """
        dir_entries = dir(result_with_debug)
        for attr in ['ATE', 'converged', 'summary', '_debug']:
            assert attr in dir_entries, (
                f"dir(result) 应包含公共属性 '{attr}'"
            )

    # ---- 错误处理: 不存在的 debug_* 属性抛出 AttributeError ----

    def test_missing_debug_attr_raises_attribute_error(self, result_with_debug):
        """访问不存在的 debug_* 属性应抛出 AttributeError。

        **Validates: Requirements 2.2**
        """
        with pytest.warns(DeprecationWarning):
            with pytest.raises(AttributeError, match="debug_nonexistent"):
                _ = result_with_debug.debug_nonexistent

    def test_non_debug_missing_attr_raises_attribute_error(self, result_with_debug):
        """访问非 debug_ 前缀的不存在属性应正常抛出 AttributeError。

        **Validates: Requirements 2.2**
        """
        with pytest.raises(AttributeError):
            _ = result_with_debug.totally_nonexistent_attr

    def test_non_debug_missing_attr_no_deprecation_warning(self, result_with_debug):
        """访问非 debug_ 前缀的不存在属性不应触发 DeprecationWarning。

        **Validates: Requirements 2.2**
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(AttributeError):
                _ = result_with_debug.totally_nonexistent_attr

    # ---- 正常属性不受影响 ----

    def test_normal_attrs_not_affected(self, result_with_debug):
        """正常属性（如 ATE）的访问不应受 __getattr__ 影响。

        **Validates: Requirements 2.1**
        """
        import warnings
        # 设置一个正常属性值
        result_with_debug.ATE = 1.5

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # 访问正常属性不应触发任何 DeprecationWarning
            val = result_with_debug.ATE
        assert val == 1.5


# =============================================================================
# 需求 1：CBIVResults 和 HDCBPSResults 的 summary() 返回类型（Property 1 补充）
# =============================================================================

@pytest.mark.property
class TestCBIVSummaryReturnType:
    """Property 1 补充: CBIVResults.summary() 返回类型一致性

    验证 CBIVResults.summary() 返回非字符串对象且具有 __str__ 方法，
    str(result.summary()) 返回非空字符串。

    **Validates: Requirements 1.4, 1.5**
    """

    @pytest.fixture(scope="class")
    def cbiv_result(self):
        """使用 lalonde 数据集拟合 CBIV 模型。"""
        from cbps import CBIV
        from cbps.datasets import load_lalonde
        import numpy as np

        data = load_lalonde()
        np.random.seed(42)
        Z = np.random.binomial(1, 0.5, len(data))
        fit = CBIV(
            Tr=data['treat'].values,
            Z=Z,
            X=data[['age', 'educ', 'black', 'hisp']].values,
            method='over',
            twosided=False,
        )
        return fit

    def test_cbiv_summary_is_not_string(self, cbiv_result):
        """CBIVResults.summary() 应返回非字符串对象。

        **Validates: Requirements 1.4**
        """
        summary = cbiv_result.summary()
        assert not isinstance(summary, str), (
            "CBIVResults.summary() 不应返回字符串，"
            f"实际返回类型为 {type(summary).__name__}"
        )

    def test_cbiv_summary_type_name(self, cbiv_result):
        """CBIVResults.summary() 应返回 CBIVSummary 类型。

        **Validates: Requirements 1.4**
        """
        from cbps.iv.cbiv import CBIVSummary

        summary = cbiv_result.summary()
        assert isinstance(summary, CBIVSummary), (
            f"CBIVResults.summary() 应返回 CBIVSummary 实例，"
            f"实际为 {type(summary).__name__}"
        )

    def test_cbiv_summary_has_str_method(self, cbiv_result):
        """CBIVResults.summary() 返回的对象应具有 __str__ 方法。

        **Validates: Requirements 1.4, 1.5**
        """
        summary = cbiv_result.summary()
        assert hasattr(summary, '__str__'), (
            "CBIVResults.summary() 返回的对象缺少 __str__ 方法"
        )

    def test_cbiv_summary_str_nonempty(self, cbiv_result):
        """str(CBIVResults.summary()) 应返回非空字符串。

        **Validates: Requirements 1.4, 1.5**
        """
        summary = cbiv_result.summary()
        text = str(summary)
        assert isinstance(text, str) and len(text.strip()) > 0, (
            "str(CBIVResults.summary()) 不应为空字符串"
        )

    def test_cbiv_summary_contains_expected_keywords(self, cbiv_result):
        """CBIVResults.summary() 文本应包含 CBIV 特征关键词。

        **Validates: Requirements 1.4, 1.5**
        """
        text = str(cbiv_result.summary())
        assert "CBIV" in text, "摘要文本应包含 'CBIV' 标识"
        assert "J-statistic" in text, "摘要文本应包含 'J-statistic'"
        assert "Converged" in text, "摘要文本应包含 'Converged'"

    def test_cbiv_summary_matches_str(self, cbiv_result):
        """CBIVResults.summary() 文本应与 str(result) 输出一致。

        **Validates: Requirements 1.4, 1.5**
        """
        summary_text = str(cbiv_result.summary())
        str_text = str(cbiv_result)
        assert summary_text == str_text, (
            "str(result.summary()) 应与 str(result) 输出一致"
        )

    def test_cbiv_summary_idempotent(self, cbiv_result):
        """多次调用 summary() 应产生相同的文本输出。

        **Validates: Requirements 1.4**
        """
        text1 = str(cbiv_result.summary())
        text2 = str(cbiv_result.summary())
        assert text1 == text2, (
            "多次调用 summary() 应产生完全相同的文本输出"
        )

    def test_cbiv_summary_has_repr(self, cbiv_result):
        """CBIVSummary 应具有有意义的 __repr__。

        **Validates: Requirements 1.4**
        """
        summary = cbiv_result.summary()
        repr_text = repr(summary)
        assert "CBIVSummary" in repr_text, (
            f"repr(summary) 应包含 'CBIVSummary'，实际为 '{repr_text}'"
        )


@pytest.mark.property
class TestHDCBPSSummaryReturnType:
    """Property 1 补充: HDCBPSResults.summary() 返回类型一致性

    验证 HDCBPSResults.summary() 返回非字符串对象且具有 __str__ 方法，
    str(result.summary()) 返回非空字符串。

    **Validates: Requirements 1.3, 1.5**
    """

    @pytest.fixture(scope="class")
    def hdcbps_result(self):
        """使用 lalonde 数据集拟合 hdCBPS 模型。"""
        from cbps import hdCBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = hdCBPS('treat ~ age + educ + black + hisp + married + nodegr',
                     data=data, y='re78')
        return fit

    def test_hdcbps_summary_is_not_string(self, hdcbps_result):
        """HDCBPSResults.summary() 应返回非字符串对象。

        **Validates: Requirements 1.3**
        """
        summary = hdcbps_result.summary()
        assert not isinstance(summary, str), (
            "HDCBPSResults.summary() 不应返回字符串，"
            f"实际返回类型为 {type(summary).__name__}"
        )

    def test_hdcbps_summary_type_name(self, hdcbps_result):
        """HDCBPSResults.summary() 应返回 HDCBPSSummary 类型。

        **Validates: Requirements 1.3**
        """
        from cbps.highdim.hdcbps import HDCBPSSummary

        summary = hdcbps_result.summary()
        assert isinstance(summary, HDCBPSSummary), (
            f"HDCBPSResults.summary() 应返回 HDCBPSSummary 实例，"
            f"实际为 {type(summary).__name__}"
        )

    def test_hdcbps_summary_has_str_method(self, hdcbps_result):
        """HDCBPSResults.summary() 返回的对象应具有 __str__ 方法。

        **Validates: Requirements 1.3, 1.5**
        """
        summary = hdcbps_result.summary()
        assert hasattr(summary, '__str__'), (
            "HDCBPSResults.summary() 返回的对象缺少 __str__ 方法"
        )

    def test_hdcbps_summary_str_nonempty(self, hdcbps_result):
        """str(HDCBPSResults.summary()) 应返回非空字符串。

        **Validates: Requirements 1.3, 1.5**
        """
        summary = hdcbps_result.summary()
        text = str(summary)
        assert isinstance(text, str) and len(text.strip()) > 0, (
            "str(HDCBPSResults.summary()) 不应为空字符串"
        )

    def test_hdcbps_summary_contains_expected_keywords(self, hdcbps_result):
        """HDCBPSResults.summary() 文本应包含 hdCBPS 特征关键词。

        **Validates: Requirements 1.3, 1.5**
        """
        text = str(hdcbps_result.summary())
        assert "hdCBPS" in text or "CBPS" in text, (
            "摘要文本应包含 'hdCBPS' 或 'CBPS' 标识"
        )
        assert "ATE" in text, "摘要文本应包含 'ATE'"
        assert "Converged" in text or "converge" in text.lower(), (
            "摘要文本应包含收敛状态信息"
        )

    def test_hdcbps_summary_idempotent(self, hdcbps_result):
        """多次调用 summary() 应产生相同的文本输出。

        **Validates: Requirements 1.3**
        """
        text1 = str(hdcbps_result.summary())
        text2 = str(hdcbps_result.summary())
        assert text1 == text2, (
            "多次调用 summary() 应产生完全相同的文本输出"
        )

    def test_hdcbps_summary_has_repr(self, hdcbps_result):
        """HDCBPSSummary 应具有有意义的 __repr__。

        **Validates: Requirements 1.3**
        """
        summary = hdcbps_result.summary()
        repr_text = repr(summary)
        assert "HDCBPSSummary" in repr_text, (
            f"repr(summary) 应包含 'HDCBPSSummary'，实际为 '{repr_text}'"
        )


# =============================================================================
# 需求 4：CBPSResults 诊断输出增强（Property 4）
# =============================================================================

@pytest.mark.property
class TestCBPSResultsDiagnosticOutput:
    """Property 4: CBPSResults 诊断输出完整性

    验证 str(result) 和 str(result.summary()) 都包含诊断信息块：
    收敛状态、权重分布摘要、有效样本量。

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @pytest.fixture(scope="class")
    def cbps_result(self):
        """使用 lalonde 数据集拟合 CBPS 模型。"""
        from cbps import CBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = CBPS('treat ~ age + educ + black + hisp', data=data, att=0)
        return fit

    # ---- str(result) 诊断信息 ----

    def test_str_contains_converged(self, cbps_result):
        """str(result) 应包含收敛状态。

        **Validates: Requirements 4.1**
        """
        text = str(cbps_result)
        assert "Converged:" in text, (
            "str(result) 应包含 'Converged:'"
        )
        expected = "Yes" if cbps_result.converged else "No"
        assert expected in text, (
            f"str(result) 应包含收敛状态 '{expected}'"
        )

    def test_str_contains_weight_summary(self, cbps_result):
        """str(result) 应包含权重分布摘要。

        **Validates: Requirements 4.2**
        """
        text = str(cbps_result)
        assert "Weight Summary:" in text, (
            "str(result) 应包含 'Weight Summary:'"
        )
        assert "Min:" in text, "str(result) 应包含 'Min:'"
        assert "Max:" in text, "str(result) 应包含 'Max:'"
        assert "Mean:" in text, "str(result) 应包含 'Mean:'"

    def test_str_contains_ess(self, cbps_result):
        """str(result) 应包含有效样本量。

        **Validates: Requirements 4.3**
        """
        text = str(cbps_result)
        assert "Effective Sample Size:" in text, (
            "str(result) 应包含 'Effective Sample Size:'"
        )

    def test_str_ess_value_correct(self, cbps_result):
        """str(result) 中的 ESS 值应与公式 (sum(w))^2 / sum(w^2) 一致。

        **Validates: Requirements 4.3**
        """
        import numpy as np

        w = cbps_result.weights
        expected_ess = (w.sum() ** 2) / (w ** 2).sum()
        ess_str = f"{expected_ess:.1f}"
        text = str(cbps_result)
        assert ess_str in text, (
            f"str(result) 应包含 ESS 值 '{ess_str}'，实际文本中未找到"
        )

    # ---- str(summary) 诊断信息 ----

    def test_summary_str_contains_converged(self, cbps_result):
        """str(result.summary()) 应包含收敛状态。

        **Validates: Requirements 4.4**
        """
        text = str(cbps_result.summary())
        assert "Converged:" in text, (
            "str(result.summary()) 应包含 'Converged:'"
        )

    def test_summary_str_contains_weight_summary(self, cbps_result):
        """str(result.summary()) 应包含权重分布摘要。

        **Validates: Requirements 4.4**
        """
        text = str(cbps_result.summary())
        assert "Weight Summary:" in text, (
            "str(result.summary()) 应包含 'Weight Summary:'"
        )
        assert "Min:" in text
        assert "Max:" in text
        assert "Mean:" in text

    def test_summary_str_contains_ess(self, cbps_result):
        """str(result.summary()) 应包含有效样本量。

        **Validates: Requirements 4.4**
        """
        text = str(cbps_result.summary())
        assert "Effective Sample Size:" in text, (
            "str(result.summary()) 应包含 'Effective Sample Size:'"
        )

    def test_summary_str_ess_value_correct(self, cbps_result):
        """str(result.summary()) 中的 ESS 值应与公式一致。

        **Validates: Requirements 4.4**
        """
        import numpy as np

        w = cbps_result.weights
        expected_ess = (w.sum() ** 2) / (w ** 2).sum()
        ess_str = f"{expected_ess:.1f}"
        text = str(cbps_result.summary())
        assert ess_str in text, (
            f"str(result.summary()) 应包含 ESS 值 '{ess_str}'"
        )

    # ---- ESS 数值正确性：已知权重向量 ----

    def test_ess_uniform_weights(self):
        """均匀权重 [1,1,1,1] 的 ESS 应等于 4.0。

        **Validates: Requirements 4.3**
        """
        import numpy as np
        from cbps.core.results import CBPSResults

        result = CBPSResults.__new__(CBPSResults)
        result.weights = np.array([1.0, 1.0, 1.0, 1.0])
        result.converged = True
        result.coefficients = np.array([[0.0]])
        result.call_info = "test"
        result.sigmasq = None
        result.deviance = 0.0
        result.J = 0.0

        text = str(result)
        assert "Effective Sample Size:  4.0" in text, (
            f"均匀权重 ESS 应为 4.0，实际文本:\n{text}"
        )

    def test_ess_extreme_weights(self):
        """极端权重 [4, 0.001, 0.001, 0.001] 的 ESS 应接近 1.0。

        **Validates: Requirements 4.3**
        """
        import numpy as np
        from cbps.core.results import CBPSResults

        result = CBPSResults.__new__(CBPSResults)
        result.weights = np.array([4.0, 0.001, 0.001, 0.001])
        result.converged = True
        result.coefficients = np.array([[0.0]])
        result.call_info = "test"
        result.sigmasq = None
        result.deviance = 0.0
        result.J = 0.0

        w = result.weights
        expected_ess = (w.sum() ** 2) / (w ** 2).sum()
        # ESS should be close to 1.0 for extreme weights
        assert expected_ess < 2.0, (
            f"极端权重 ESS 应接近 1.0，实际为 {expected_ess}"
        )
        ess_str = f"{expected_ess:.1f}"
        text = str(result)
        assert ess_str in text, (
            f"str(result) 应包含 ESS 值 '{ess_str}'"
        )

    def test_diagnostics_block_label(self, cbps_result):
        """诊断信息块应以 'Diagnostics:' 标签开头。

        **Validates: Requirements 4.1**
        """
        text = str(cbps_result)
        assert "Diagnostics:" in text, (
            "str(result) 应包含 'Diagnostics:' 标签"
        )
        text2 = str(cbps_result.summary())
        assert "Diagnostics:" in text2, (
            "str(result.summary()) 应包含 'Diagnostics:' 标签"
        )


# =============================================================================
# 需求 6：balance() 返回值标签一致性（Property 6）
# =============================================================================

@pytest.mark.property
class TestBalanceDataFrameLabels:
    """Property 6: balance() 返回值 DataFrame 行标签

    验证 balance() 返回的字典中 DataFrame 具有非默认行索引（协变量名称），
    且行索引长度等于协变量数量（不含截距）。

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    COVARIATES = ['age', 'educ', 'black', 'hisp']

    @pytest.fixture(scope="class")
    def cbps_result(self):
        """使用 lalonde 数据集拟合 CBPS 模型。"""
        from cbps import CBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = CBPS('treat ~ age + educ + black + hisp', data=data, att=0)
        return fit

    @pytest.fixture(scope="class")
    def npcbps_result(self):
        """使用 lalonde 数据集拟合 npCBPS 模型。"""
        from cbps import npCBPS
        from cbps.datasets import load_lalonde

        data = load_lalonde()
        fit = npCBPS('treat ~ age + educ + black + hisp', data=data)
        return fit

    # ---- CBPS balance() 测试 ----

    def test_cbps_balance_has_named_index(self, cbps_result):
        """CBPS balance() 返回的 DataFrame 应具有非默认行索引。

        **Validates: Requirements 6.1**
        """
        import pandas as pd
        from cbps import balance

        result = balance(cbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert not isinstance(df.index, pd.RangeIndex), (
                    f"CBPS balance()['{key}'] 不应使用 RangeIndex"
                )

    def test_cbps_balance_index_length(self, cbps_result):
        """CBPS balance() 行索引长度应等于协变量数量。

        **Validates: Requirements 6.1**
        """
        import pandas as pd
        from cbps import balance

        result = balance(cbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert len(df.index) == len(self.COVARIATES), (
                    f"CBPS balance()['{key}'] 行索引长度应为 {len(self.COVARIATES)}，"
                    f"实际为 {len(df.index)}"
                )

    def test_cbps_balance_index_names(self, cbps_result):
        """CBPS balance() 行索引应包含协变量名称。

        **Validates: Requirements 6.1**
        """
        import pandas as pd
        from cbps import balance

        result = balance(cbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                for cov in self.COVARIATES:
                    assert cov in df.index, (
                        f"CBPS balance()['{key}'] 行索引应包含 '{cov}'"
                    )

    def test_cbps_balance_enhanced_has_named_index(self, cbps_result):
        """CBPS balance(enhanced=True) 返回的 DataFrame 应具有非默认行索引。

        **Validates: Requirements 6.1**
        """
        import pandas as pd
        from cbps import balance

        result = balance(cbps_result, enhanced=True)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert not isinstance(df.index, pd.RangeIndex), (
                    f"CBPS balance(enhanced=True)['{key}'] 不应使用 RangeIndex"
                )

    # ---- npCBPS balance() 测试 ----

    def test_npcbps_balance_has_named_index(self, npcbps_result):
        """npCBPS balance() 返回的 DataFrame 应具有非默认行索引。

        **Validates: Requirements 6.2**
        """
        import pandas as pd
        from cbps import balance

        result = balance(npcbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert not isinstance(df.index, pd.RangeIndex), (
                    f"npCBPS balance()['{key}'] 不应使用 RangeIndex"
                )

    def test_npcbps_balance_index_length(self, npcbps_result):
        """npCBPS balance() 行索引长度应等于协变量数量。

        **Validates: Requirements 6.2**
        """
        import pandas as pd
        from cbps import balance

        result = balance(npcbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert len(df.index) == len(self.COVARIATES), (
                    f"npCBPS balance()['{key}'] 行索引长度应为 {len(self.COVARIATES)}，"
                    f"实际为 {len(df.index)}"
                )

    def test_npcbps_balance_index_names(self, npcbps_result):
        """npCBPS balance() 行索引应包含协变量名称。

        **Validates: Requirements 6.2**
        """
        import pandas as pd
        from cbps import balance

        result = balance(npcbps_result)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                for cov in self.COVARIATES:
                    assert cov in df.index, (
                        f"npCBPS balance()['{key}'] 行索引应包含 '{cov}'"
                    )

    def test_npcbps_balance_enhanced_has_named_index(self, npcbps_result):
        """npCBPS balance(enhanced=True) 返回的 DataFrame 应具有非默认行索引。

        **Validates: Requirements 6.2**
        """
        import pandas as pd
        from cbps import balance

        result = balance(npcbps_result, enhanced=True)
        for key in ['balanced', 'original']:
            if key in result and isinstance(result[key], pd.DataFrame):
                df = result[key]
                assert not isinstance(df.index, pd.RangeIndex), (
                    f"npCBPS balance(enhanced=True)['{key}'] 不应使用 RangeIndex"
                )

    # ---- 跨类一致性 ----

    def test_both_balance_results_have_columns(self, cbps_result, npcbps_result):
        """CBPS 和 npCBPS 的 balance() 返回 DataFrame 都应具有列标签。

        **Validates: Requirements 6.1, 6.2**
        """
        import pandas as pd
        from cbps import balance

        for name, obj in [("CBPS", cbps_result), ("npCBPS", npcbps_result)]:
            result = balance(obj)
            for key in ['balanced', 'original']:
                if key in result and isinstance(result[key], pd.DataFrame):
                    df = result[key]
                    assert len(df.columns) > 0, (
                        f"{name} balance()['{key}'] 应具有列标签"
                    )
                    assert not all(isinstance(c, int) for c in df.columns), (
                        f"{name} balance()['{key}'] 列标签不应全为整数"
                    )
