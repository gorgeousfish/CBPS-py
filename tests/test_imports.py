"""
Module Import Verification Test Suite
======================================

This module consolidates all import tests for the CBPS Python package,
verifying that every public module and its documented exports are importable.
Each test class corresponds to a distinct subpackage, ensuring that the
package's public API surface is complete, correctly wired, and free of
circular import issues.

The tests are organized hierarchically:

1. **Top-level package** (``cbps``): Verifies the main entry-point functions
   ``CBPS``, ``cbps_fit``, ``CBMSM``, ``npCBPS``, ``hdCBPS``, ``CBIV``,
   ``AsyVar``, ``balance``, and related convenience wrappers.

2. **Core algorithms** (``cbps.core``): Binary, continuous, multi-valued,
   and optimal CBPS fitting routines together with result container classes.

3. **Marginal structural models** (``cbps.msm``): Longitudinal treatment
   weight estimation via ``CBMSM`` and ``cbmsm_fit``.

4. **Nonparametric estimation** (``cbps.nonparametric``): Empirical
   likelihood-based weight estimation without parametric assumptions.

5. **High-dimensional estimation** (``cbps.highdim``): LASSO-regularized
   CBPS weight functions for settings where p >> n.

6. **Statistical inference** (``cbps.inference``): Sandwich variance
   estimation and outcome-model variance-covariance adjustment.

7. **Instrumental variables** (``cbps.iv``): CBPS methodology adapted for
   treatment noncompliance with instrumental variable designs.

8. **Datasets** (``cbps.datasets``): Bundled empirical and simulated
   datasets for reproducible examples and validation.

9. **Diagnostics** (``cbps.diagnostics``): Covariate balance assessment
   and visualization utilities.

10. **scikit-learn integration** (``cbps.sklearn``): sklearn-compatible
    estimator wrapper for pipeline and cross-validation workflows.

11. **Utilities** (``cbps.utils``): Formula parsing, weight computation,
    data validation, and preprocessing helpers.

Test Methodology
----------------
Each test class follows a uniform structure:

- ``test_module_importable``: The subpackage can be imported without error.
- ``test_<name>_importable``: Each documented public symbol is importable
  via ``from <module> import <name>``.
- ``test_all_defined``: The module defines ``__all__`` for explicit API
  declaration.
- ``test_no_private_in_all``: No underscore-prefixed names leak into
  ``__all__``.

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
Journal of the Royal Statistical Society, Series B, 76(1), 243-263.
"""

import pytest


# =============================================================================
# Top-Level Package Imports (cbps)
# =============================================================================

@pytest.mark.unit
class TestTopLevelImports:
    """Verify that the top-level ``cbps`` package exports its public API."""

    def test_module_importable(self):
        """The top-level cbps package is importable."""
        import cbps  # noqa: F401

    def test_version_defined(self):
        """The package exposes a ``__version__`` string."""
        import cbps
        assert hasattr(cbps, '__version__')
        assert isinstance(cbps.__version__, str)

    def test_CBPS_importable(self):
        """The main ``CBPS`` function is importable."""
        from cbps import CBPS  # noqa: F401
        assert callable(CBPS)

    def test_cbps_fit_importable(self):
        """The array-interface ``cbps_fit`` function is importable."""
        from cbps import cbps_fit  # noqa: F401
        assert callable(cbps_fit)

    def test_CBMSM_importable(self):
        """The ``CBMSM`` function is importable from top level."""
        from cbps import CBMSM  # noqa: F401
        assert callable(CBMSM)

    def test_cbmsm_fit_importable(self):
        """The ``cbmsm_fit`` function is importable from top level."""
        from cbps import cbmsm_fit  # noqa: F401
        assert callable(cbmsm_fit)

    def test_npCBPS_importable(self):
        """The ``npCBPS`` function is importable from top level."""
        from cbps import npCBPS  # noqa: F401
        assert callable(npCBPS)

    def test_hdCBPS_importable(self):
        """The ``hdCBPS`` function is importable from top level."""
        from cbps import hdCBPS  # noqa: F401
        assert callable(hdCBPS)

    def test_CBIV_importable(self):
        """The ``CBIV`` function is importable from top level."""
        from cbps import CBIV  # noqa: F401
        assert callable(CBIV)

    def test_AsyVar_importable(self):
        """The ``AsyVar`` function is importable from top level."""
        from cbps import AsyVar  # noqa: F401
        assert callable(AsyVar)

    def test_balance_importable(self):
        """The ``balance`` function is importable from top level."""
        from cbps import balance  # noqa: F401
        assert callable(balance)

    def test_detect_treatment_type_importable(self):
        """The ``_detect_treatment_type`` helper is importable from top level."""
        from cbps import _detect_treatment_type  # noqa: F401
        assert callable(_detect_treatment_type)

    def test_plot_functions_importable(self):
        """Plot convenience wrappers are importable from top level."""
        from cbps import plot_cbps, plot_cbps_continuous  # noqa: F401

    def test_all_defined(self):
        """The top-level package defines ``__all__``."""
        import cbps
        assert hasattr(cbps, '__all__')
        assert isinstance(cbps.__all__, list)
        assert len(cbps.__all__) > 0

    def test_all_entries_resolvable(self):
        """Every name listed in ``__all__`` is actually accessible."""
        import cbps
        for name in cbps.__all__:
            assert hasattr(cbps, name), (
                f"cbps.__all__ lists '{name}' but it is not accessible "
                f"as an attribute of the cbps package"
            )


# =============================================================================
# Core Module Imports (cbps.core)
# =============================================================================

@pytest.mark.unit
class TestCoreImports:
    """Verify that ``cbps.core`` exports all documented fitting routines."""

    def test_module_importable(self):
        """The cbps.core subpackage is importable."""
        import cbps.core  # noqa: F401

    def test_cbps_binary_fit_importable(self):
        """``cbps_binary_fit`` is importable from cbps.core."""
        from cbps.core import cbps_binary_fit  # noqa: F401
        assert callable(cbps_binary_fit)

    def test_cbps_continuous_fit_importable(self):
        """``cbps_continuous_fit`` is importable from cbps.core."""
        from cbps.core import cbps_continuous_fit  # noqa: F401
        assert callable(cbps_continuous_fit)

    def test_cbps_3treat_fit_importable(self):
        """``cbps_3treat_fit`` is importable from cbps.core."""
        from cbps.core import cbps_3treat_fit  # noqa: F401
        assert callable(cbps_3treat_fit)

    def test_cbps_4treat_fit_importable(self):
        """``cbps_4treat_fit`` is importable from cbps.core."""
        from cbps.core import cbps_4treat_fit  # noqa: F401
        assert callable(cbps_4treat_fit)

    def test_cbps_optimal_2treat_importable(self):
        """``cbps_optimal_2treat`` is importable from cbps.core."""
        from cbps.core import cbps_optimal_2treat  # noqa: F401
        assert callable(cbps_optimal_2treat)

    def test_CBPSResults_importable(self):
        """``CBPSResults`` class is importable from cbps.core."""
        from cbps.core import CBPSResults  # noqa: F401
        assert isinstance(CBPSResults, type)

    def test_CBPSSummary_importable(self):
        """``CBPSSummary`` class is importable from cbps.core."""
        from cbps.core import CBPSSummary  # noqa: F401
        assert isinstance(CBPSSummary, type)

    def test_all_defined(self):
        """The cbps.core subpackage defines ``__all__``."""
        import cbps.core
        assert hasattr(cbps.core, '__all__')
        assert isinstance(cbps.core.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.core
        for name in cbps.core.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in cbps.core.__all__"
            )


# =============================================================================
# MSM Module Imports (cbps.msm)
# =============================================================================

@pytest.mark.unit
class TestMSMImports:
    """Verify that ``cbps.msm`` exports marginal structural model components."""

    def test_module_importable(self):
        """The cbps.msm subpackage is importable."""
        import cbps.msm  # noqa: F401

    def test_CBMSM_importable(self):
        """``CBMSM`` function is importable from cbps.msm."""
        from cbps.msm import CBMSM  # noqa: F401
        assert callable(CBMSM)

    def test_cbmsm_fit_importable(self):
        """``cbmsm_fit`` function is importable from cbps.msm."""
        from cbps.msm import cbmsm_fit  # noqa: F401
        assert callable(cbmsm_fit)

    def test_CBMSMResults_importable(self):
        """``CBMSMResults`` class is importable from cbps.msm."""
        from cbps.msm import CBMSMResults  # noqa: F401
        assert isinstance(CBMSMResults, type)

    def test_all_defined(self):
        """The cbps.msm subpackage defines ``__all__``."""
        import cbps.msm
        assert hasattr(cbps.msm, '__all__')
        assert isinstance(cbps.msm.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.msm
        for name in cbps.msm.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in cbps.msm.__all__"
            )


# =============================================================================
# Nonparametric Module Imports (cbps.nonparametric)
# =============================================================================

@pytest.mark.unit
class TestNonparametricImports:
    """Verify that ``cbps.nonparametric`` exports empirical likelihood components."""

    def test_module_importable(self):
        """The cbps.nonparametric subpackage is importable."""
        import cbps.nonparametric  # noqa: F401

    def test_npCBPS_importable(self):
        """``npCBPS`` function is importable from cbps.nonparametric."""
        from cbps.nonparametric import npCBPS  # noqa: F401
        assert callable(npCBPS)

    def test_NPCBPSResults_importable(self):
        """``NPCBPSResults`` class is importable from cbps.nonparametric."""
        from cbps.nonparametric import NPCBPSResults  # noqa: F401
        assert isinstance(NPCBPSResults, type)

    def test_llog_importable(self):
        """``llog`` Taylor-approximated log is importable."""
        from cbps.nonparametric import llog  # noqa: F401
        assert callable(llog)

    def test_llogp_importable(self):
        """``llogp`` derivative of Taylor-approximated log is importable."""
        from cbps.nonparametric import llogp  # noqa: F401
        assert callable(llogp)

    def test_cholesky_whitening_importable(self):
        """``cholesky_whitening`` is importable from cbps.nonparametric."""
        from cbps.nonparametric import cholesky_whitening  # noqa: F401
        assert callable(cholesky_whitening)

    def test_get_w_importable(self):
        """``get_w`` empirical likelihood solver is importable."""
        from cbps.nonparametric import get_w  # noqa: F401
        assert callable(get_w)

    def test_log_post_importable(self):
        """``log_post`` posterior function is importable."""
        from cbps.nonparametric import log_post  # noqa: F401
        assert callable(log_post)

    def test_all_defined(self):
        """The cbps.nonparametric subpackage defines ``__all__``."""
        import cbps.nonparametric
        assert hasattr(cbps.nonparametric, '__all__')
        assert isinstance(cbps.nonparametric.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.nonparametric
        for name in cbps.nonparametric.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.nonparametric.__all__"
            )


# =============================================================================
# High-Dimensional Module Imports (cbps.highdim)
# =============================================================================

@pytest.mark.unit
class TestHighdimImports:
    """Verify that ``cbps.highdim`` exports weight functions and optional hdCBPS."""

    def test_module_importable(self):
        """The cbps.highdim subpackage is importable."""
        import cbps.highdim  # noqa: F401

    def test_ate_wt_func_importable(self):
        """``ate_wt_func`` is importable from cbps.highdim."""
        from cbps.highdim import ate_wt_func  # noqa: F401
        assert callable(ate_wt_func)

    def test_att_wt_func_importable(self):
        """``att_wt_func`` is importable from cbps.highdim."""
        from cbps.highdim import att_wt_func  # noqa: F401
        assert callable(att_wt_func)

    def test_ate_wt_nl_func_importable(self):
        """``ate_wt_nl_func`` is importable from cbps.highdim."""
        from cbps.highdim import ate_wt_nl_func  # noqa: F401
        assert callable(ate_wt_nl_func)

    def test_att_wt_nl_func_importable(self):
        """``att_wt_nl_func`` is importable from cbps.highdim."""
        from cbps.highdim import att_wt_nl_func  # noqa: F401
        assert callable(att_wt_nl_func)

    def test_hdCBPS_importable_when_glmnet_available(self):
        """``hdCBPS`` is importable when glmnetforpython is installed."""
        import cbps.highdim
        if 'hdCBPS' in cbps.highdim.__all__:
            from cbps.highdim import hdCBPS  # noqa: F401
            assert callable(hdCBPS)
        else:
            pytest.skip("glmnetforpython not installed; hdCBPS unavailable")

    def test_all_defined(self):
        """The cbps.highdim subpackage defines ``__all__``."""
        import cbps.highdim
        assert hasattr(cbps.highdim, '__all__')
        assert isinstance(cbps.highdim.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.highdim
        for name in cbps.highdim.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.highdim.__all__"
            )


# =============================================================================
# Inference Module Imports (cbps.inference)
# =============================================================================

@pytest.mark.unit
class TestInferenceImports:
    """Verify that ``cbps.inference`` exports variance estimation functions."""

    def test_module_importable(self):
        """The cbps.inference subpackage is importable."""
        import cbps.inference  # noqa: F401

    def test_asy_var_importable(self):
        """``asy_var`` is importable from cbps.inference."""
        from cbps.inference import asy_var  # noqa: F401
        assert callable(asy_var)

    def test_vcov_outcome_importable(self):
        """``vcov_outcome`` is importable from cbps.inference."""
        from cbps.inference import vcov_outcome  # noqa: F401
        assert callable(vcov_outcome)

    def test_all_defined(self):
        """The cbps.inference subpackage defines ``__all__``."""
        import cbps.inference
        assert hasattr(cbps.inference, '__all__')
        assert isinstance(cbps.inference.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.inference
        for name in cbps.inference.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.inference.__all__"
            )


# =============================================================================
# Instrumental Variables Module Imports (cbps.iv)
# =============================================================================

@pytest.mark.unit
class TestIVImports:
    """Verify that ``cbps.iv`` exports instrumental variable CBPS components."""

    def test_module_importable(self):
        """The cbps.iv subpackage is importable."""
        import cbps.iv  # noqa: F401

    def test_CBIV_importable(self):
        """``CBIV`` function is importable from cbps.iv."""
        from cbps.iv import CBIV  # noqa: F401
        assert callable(CBIV)

    def test_CBIVResults_importable(self):
        """``CBIVResults`` class is importable from cbps.iv."""
        from cbps.iv import CBIVResults  # noqa: F401
        assert isinstance(CBIVResults, type)

    def test_CBIVNumericalWarning_importable(self):
        """``CBIVNumericalWarning`` is importable from cbps.iv."""
        from cbps.iv import CBIVNumericalWarning  # noqa: F401
        assert issubclass(CBIVNumericalWarning, Warning)

    def test_all_defined(self):
        """The cbps.iv subpackage defines ``__all__``."""
        import cbps.iv
        assert hasattr(cbps.iv, '__all__')
        assert isinstance(cbps.iv.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.iv
        for name in cbps.iv.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in cbps.iv.__all__"
            )


# =============================================================================
# Datasets Module Imports (cbps.datasets)
# =============================================================================

@pytest.mark.unit
class TestDatasetsImports:
    """Verify that ``cbps.datasets`` exports all dataset loader functions."""

    def test_module_importable(self):
        """The cbps.datasets subpackage is importable."""
        import cbps.datasets  # noqa: F401

    def test_load_lalonde_importable(self):
        """``load_lalonde`` is importable from cbps.datasets."""
        from cbps.datasets import load_lalonde  # noqa: F401
        assert callable(load_lalonde)

    def test_load_lalonde_psid_combined_importable(self):
        """``load_lalonde_psid_combined`` is importable from cbps.datasets."""
        from cbps.datasets import load_lalonde_psid_combined  # noqa: F401
        assert callable(load_lalonde_psid_combined)

    def test_load_blackwell_importable(self):
        """``load_blackwell`` is importable from cbps.datasets."""
        from cbps.datasets import load_blackwell  # noqa: F401
        assert callable(load_blackwell)

    def test_load_continuous_simulation_importable(self):
        """``load_continuous_simulation`` is importable from cbps.datasets."""
        from cbps.datasets import load_continuous_simulation  # noqa: F401
        assert callable(load_continuous_simulation)

    def test_load_political_ads_importable(self):
        """``load_political_ads`` is importable from cbps.datasets."""
        from cbps.datasets import load_political_ads  # noqa: F401
        assert callable(load_political_ads)

    def test_load_npcbps_continuous_sim_importable(self):
        """``load_npcbps_continuous_sim`` is importable from cbps.datasets."""
        from cbps.datasets import load_npcbps_continuous_sim  # noqa: F401
        assert callable(load_npcbps_continuous_sim)

    def test_all_defined(self):
        """The cbps.datasets subpackage defines ``__all__``."""
        import cbps.datasets
        assert hasattr(cbps.datasets, '__all__')
        assert isinstance(cbps.datasets.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.datasets
        for name in cbps.datasets.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.datasets.__all__"
            )


# =============================================================================
# Diagnostics Module Imports (cbps.diagnostics)
# =============================================================================

@pytest.mark.unit
class TestDiagnosticsImports:
    """Verify that ``cbps.diagnostics`` exports balance and plotting functions."""

    def test_module_importable(self):
        """The cbps.diagnostics subpackage is importable."""
        import cbps.diagnostics  # noqa: F401

    def test_balance_cbps_importable(self):
        """``balance_cbps`` is importable from cbps.diagnostics."""
        from cbps.diagnostics import balance_cbps  # noqa: F401
        assert callable(balance_cbps)

    def test_balance_cbps_continuous_importable(self):
        """``balance_cbps_continuous`` is importable from cbps.diagnostics."""
        from cbps.diagnostics import balance_cbps_continuous  # noqa: F401
        assert callable(balance_cbps_continuous)

    def test_diagnose_cbgps_quality_importable(self):
        """``diagnose_cbgps_quality`` is importable from cbps.diagnostics."""
        from cbps.diagnostics import diagnose_cbgps_quality  # noqa: F401
        assert callable(diagnose_cbgps_quality)

    def test_compute_weighted_correlations_importable(self):
        """``compute_weighted_correlations`` is importable from cbps.diagnostics."""
        from cbps.diagnostics import compute_weighted_correlations  # noqa: F401
        assert callable(compute_weighted_correlations)

    def test_plot_cbps_importable(self):
        """``plot_cbps`` is importable when matplotlib is available."""
        import cbps.diagnostics
        if 'plot_cbps' in cbps.diagnostics.__all__:
            from cbps.diagnostics import plot_cbps  # noqa: F401
            assert callable(plot_cbps)
        else:
            pytest.skip("matplotlib not installed; plot_cbps unavailable")

    def test_plot_cbps_continuous_importable(self):
        """``plot_cbps_continuous`` is importable when matplotlib is available."""
        import cbps.diagnostics
        if 'plot_cbps_continuous' in cbps.diagnostics.__all__:
            from cbps.diagnostics import plot_cbps_continuous  # noqa: F401
            assert callable(plot_cbps_continuous)
        else:
            pytest.skip("matplotlib not installed; plot_cbps_continuous unavailable")

    def test_all_defined(self):
        """The cbps.diagnostics subpackage defines ``__all__``."""
        import cbps.diagnostics
        assert hasattr(cbps.diagnostics, '__all__')
        assert isinstance(cbps.diagnostics.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.diagnostics
        for name in cbps.diagnostics.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.diagnostics.__all__"
            )


# =============================================================================
# scikit-learn Integration Module Imports (cbps.sklearn)
# =============================================================================

@pytest.mark.unit
class TestSklearnImports:
    """Verify that ``cbps.sklearn`` exports the sklearn-compatible estimator."""

    def test_module_importable(self):
        """The cbps.sklearn subpackage is importable."""
        import cbps.sklearn  # noqa: F401

    def test_CBPSEstimator_importable(self):
        """``CBPSEstimator`` class is importable from cbps.sklearn."""
        from cbps.sklearn import CBPSEstimator  # noqa: F401
        assert isinstance(CBPSEstimator, type)

    def test_CBPSEstimator_is_sklearn_estimator(self):
        """``CBPSEstimator`` inherits from sklearn BaseEstimator."""
        try:
            from sklearn.base import BaseEstimator
            from cbps.sklearn import CBPSEstimator
            assert issubclass(CBPSEstimator, BaseEstimator)
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_all_defined(self):
        """The cbps.sklearn subpackage defines ``__all__``."""
        import cbps.sklearn
        assert hasattr(cbps.sklearn, '__all__')
        assert isinstance(cbps.sklearn.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.sklearn
        for name in cbps.sklearn.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.sklearn.__all__"
            )


# =============================================================================
# Utilities Module Imports (cbps.utils)
# =============================================================================

@pytest.mark.unit
class TestUtilsImports:
    """Verify that ``cbps.utils`` exports all documented utility functions."""

    def test_module_importable(self):
        """The cbps.utils subpackage is importable."""
        import cbps.utils  # noqa: F401

    # --- Formula parsing ---

    def test_parse_formula_importable(self):
        """``parse_formula`` is importable from cbps.utils."""
        from cbps.utils import parse_formula  # noqa: F401
        assert callable(parse_formula)

    def test_parse_dual_formulas_importable(self):
        """``parse_dual_formulas`` is importable from cbps.utils."""
        from cbps.utils import parse_dual_formulas  # noqa: F401
        assert callable(parse_dual_formulas)

    def test_parse_arrays_importable(self):
        """``parse_arrays`` is importable from cbps.utils."""
        from cbps.utils import parse_arrays  # noqa: F401
        assert callable(parse_arrays)

    # --- Weight computation ---

    def test_compute_ate_weights_importable(self):
        """``compute_ate_weights`` is importable from cbps.utils."""
        from cbps.utils import compute_ate_weights  # noqa: F401
        assert callable(compute_ate_weights)

    def test_compute_att_weights_importable(self):
        """``compute_att_weights`` is importable from cbps.utils."""
        from cbps.utils import compute_att_weights  # noqa: F401
        assert callable(compute_att_weights)

    def test_compute_continuous_weights_importable(self):
        """``compute_continuous_weights`` is importable from cbps.utils."""
        from cbps.utils import compute_continuous_weights  # noqa: F401
        assert callable(compute_continuous_weights)

    def test_standardize_weights_importable(self):
        """``standardize_weights`` is importable from cbps.utils."""
        from cbps.utils import standardize_weights  # noqa: F401
        assert callable(standardize_weights)

    # --- Data utilities ---

    def test_validate_arrays_importable(self):
        """``validate_arrays`` is importable from cbps.utils."""
        from cbps.utils import validate_arrays  # noqa: F401
        assert callable(validate_arrays)

    def test_handle_missing_importable(self):
        """``handle_missing`` is importable from cbps.utils."""
        from cbps.utils import handle_missing  # noqa: F401
        assert callable(handle_missing)

    def test_encode_treatment_factor_importable(self):
        """``encode_treatment_factor`` is importable from cbps.utils."""
        from cbps.utils import encode_treatment_factor  # noqa: F401
        assert callable(encode_treatment_factor)

    def test_normalize_sample_weights_importable(self):
        """``normalize_sample_weights`` is importable from cbps.utils."""
        from cbps.utils import normalize_sample_weights  # noqa: F401
        assert callable(normalize_sample_weights)

    # --- __all__ checks ---

    def test_all_defined(self):
        """The cbps.utils subpackage defines ``__all__``."""
        import cbps.utils
        assert hasattr(cbps.utils, '__all__')
        assert isinstance(cbps.utils.__all__, list)

    def test_no_private_in_all(self):
        """No underscore-prefixed names appear in ``__all__``."""
        import cbps.utils
        for name in cbps.utils.__all__:
            assert not name.startswith('_'), (
                f"Private name '{name}' should not appear in "
                f"cbps.utils.__all__"
            )

    def test_all_entries_resolvable(self):
        """Every name listed in ``__all__`` is actually accessible."""
        import cbps.utils
        for name in cbps.utils.__all__:
            assert hasattr(cbps.utils, name), (
                f"cbps.utils.__all__ lists '{name}' but it is not accessible "
                f"as an attribute of the cbps.utils package"
            )
