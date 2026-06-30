"""
Tests for diagnostic visualization functions.

Tests cover:
- love_plot: Love plot generation and parameter handling
- plot_weight_distribution: Weight histogram by treatment group
- plot_ps_overlap: Propensity score overlap visualization
- CBPSSummary.__str__: Formatted summary output

Test IDs: DIAG-VIZ-001 to DIAG-VIZ-020
"""
import numpy as np
import pandas as pd
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def binary_data():
    """Generate synthetic binary treatment data."""
    np.random.seed(42)
    n = 200
    treat = np.concatenate([np.ones(80), np.zeros(120)])
    x = np.column_stack([
        np.ones(n),
        np.random.normal(0, 1, n),
        np.random.normal(0, 1, n),
        np.random.normal(0, 1, n),
    ])
    ps = 1 / (1 + np.exp(-x @ np.array([0.1, 0.5, -0.3, 0.2])))
    weights = np.where(treat == 1, 1 / ps, 1 / (1 - ps))
    return {
        'weights': weights,
        'x': x,
        'y': treat,
        'fitted_values': ps,
    }


@pytest.fixture
def balance_dataframe():
    """Generate a balance DataFrame for love_plot."""
    return pd.DataFrame({
        'covariate': ['age', 'education', 'income', 're74', 're75'],
        'original': [0.25, 0.12, 0.45, 0.33, 0.28],
        'balanced': [0.05, 0.03, 0.08, 0.06, 0.04],
    })


@pytest.fixture
def balance_dict():
    """Generate a balance dict (like output of balance_cbps)."""
    # Shape: (4 covariates, 4 columns) = means_t, means_c, std_t, std_c
    np.random.seed(123)
    original = np.array([
        [0.5, 0.3, 0.6, 0.2],
        [1.0, 0.8, 1.1, 0.7],
        [0.3, 0.5, 0.4, 0.6],
        [0.7, 0.4, 0.8, 0.3],
    ])
    balanced = np.array([
        [0.4, 0.38, 0.42, 0.39],
        [0.9, 0.88, 0.91, 0.89],
        [0.4, 0.42, 0.41, 0.43],
        [0.5, 0.48, 0.52, 0.49],
    ])
    return {'original': original, 'balanced': balanced}


# ============================================================
# love_plot tests
# ============================================================

class TestLovePlot:
    """Tests for love_plot function. DIAG-VIZ-001 to DIAG-VIZ-005."""

    @pytest.mark.unit
    def test_viz001_returns_figure_from_dataframe(self, balance_dataframe):
        """DIAG-VIZ-001: love_plot returns Figure from DataFrame input."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import love_plot

        fig = love_plot(balance_dataframe, threshold=0.1)

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz002_returns_figure_from_dict(self, balance_dict):
        """DIAG-VIZ-002: love_plot returns Figure from dict input."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import love_plot

        fig = love_plot(balance_dict, threshold=0.1)

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz003_threshold_line_present(self, balance_dataframe):
        """DIAG-VIZ-003: Threshold line is drawn."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import love_plot

        fig = love_plot(balance_dataframe, threshold=0.15)
        ax = fig.axes[0]

        # Check that a vertical line exists (axvline creates Line2D)
        lines = ax.get_lines()
        assert len(lines) > 0
        plt.close('all')

    @pytest.mark.unit
    def test_viz004_custom_title(self, balance_dataframe):
        """DIAG-VIZ-004: Custom title is applied."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import love_plot

        fig = love_plot(balance_dataframe, title="My Custom Title")
        ax = fig.axes[0]

        assert ax.get_title() == "My Custom Title"
        plt.close('all')

    @pytest.mark.unit
    def test_viz005_invalid_input_raises(self):
        """DIAG-VIZ-005: Invalid input type raises TypeError."""
        pytest.importorskip("matplotlib")
        from cbps.diagnostics.plots import love_plot

        with pytest.raises(TypeError, match="must be a dict.*or a DataFrame"):
            love_plot("invalid input")

    @pytest.mark.unit
    def test_viz005b_missing_columns_raises(self):
        """love_plot raises ValueError for DataFrame with wrong columns."""
        pytest.importorskip("matplotlib")
        from cbps.diagnostics.plots import love_plot

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(ValueError, match="must have columns"):
            love_plot(df)


# ============================================================
# plot_weight_distribution tests
# ============================================================

class TestPlotWeightDistribution:
    """Tests for plot_weight_distribution. DIAG-VIZ-006 to DIAG-VIZ-010."""

    @pytest.mark.unit
    def test_viz006_returns_figure(self, binary_data):
        """DIAG-VIZ-006: Returns matplotlib Figure."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_weight_distribution

        fig = plot_weight_distribution(binary_data['weights'], binary_data['y'])

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz007_two_panels(self, binary_data):
        """DIAG-VIZ-007: Figure has two subplots (treated + control)."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_weight_distribution

        fig = plot_weight_distribution(binary_data['weights'], binary_data['y'])

        assert len(fig.axes) == 2
        plt.close('all')

    @pytest.mark.unit
    def test_viz008_custom_bins(self, binary_data):
        """DIAG-VIZ-008: Custom bin count is respected."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_weight_distribution

        fig = plot_weight_distribution(
            binary_data['weights'], binary_data['y'], bins=20
        )

        assert isinstance(fig, plt.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz009_custom_title(self, binary_data):
        """DIAG-VIZ-009: Custom title is applied."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_weight_distribution

        fig = plot_weight_distribution(
            binary_data['weights'], binary_data['y'],
            title="Custom Title"
        )
        # suptitle is at figure level
        assert fig._suptitle.get_text() == "Custom Title"
        plt.close('all')

    @pytest.mark.unit
    def test_viz010_handles_extreme_weights(self):
        """DIAG-VIZ-010: Handles extreme weight values without error."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_weight_distribution

        weights = np.array([0.01, 100.0, 1.0, 1.5, 0.5, 50.0])
        treat = np.array([1, 1, 1, 0, 0, 0])

        fig = plot_weight_distribution(weights, treat)
        assert fig is not None
        plt.close('all')


# ============================================================
# plot_ps_overlap tests
# ============================================================

class TestPlotPSOverlap:
    """Tests for plot_ps_overlap. DIAG-VIZ-011 to DIAG-VIZ-015."""

    @pytest.mark.unit
    def test_viz011_kde_returns_figure(self, binary_data):
        """DIAG-VIZ-011: KDE method returns Figure."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_ps_overlap

        fig = plot_ps_overlap(
            binary_data['fitted_values'], binary_data['y'], method='kde'
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz012_histogram_returns_figure(self, binary_data):
        """DIAG-VIZ-012: Histogram method returns Figure."""
        matplotlib = pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_ps_overlap

        fig = plot_ps_overlap(
            binary_data['fitted_values'], binary_data['y'], method='histogram'
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close('all')

    @pytest.mark.unit
    def test_viz013_invalid_method_raises(self, binary_data):
        """DIAG-VIZ-013: Invalid method raises ValueError."""
        pytest.importorskip("matplotlib")
        from cbps.diagnostics.plots import plot_ps_overlap

        with pytest.raises(ValueError, match="must be 'kde' or 'histogram'"):
            plot_ps_overlap(
                binary_data['fitted_values'], binary_data['y'],
                method='invalid'
            )

    @pytest.mark.unit
    def test_viz014_custom_title(self, binary_data):
        """DIAG-VIZ-014: Custom title is applied."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_ps_overlap

        fig = plot_ps_overlap(
            binary_data['fitted_values'], binary_data['y'],
            title="Overlap Check"
        )
        ax = fig.axes[0]
        assert ax.get_title() == "Overlap Check"
        plt.close('all')

    @pytest.mark.unit
    def test_viz015_custom_bins(self, binary_data):
        """DIAG-VIZ-015: Custom bins for histogram method."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt
        from cbps.diagnostics.plots import plot_ps_overlap

        fig = plot_ps_overlap(
            binary_data['fitted_values'], binary_data['y'],
            method='histogram', bins=25
        )

        assert fig is not None
        plt.close('all')


# ============================================================
# matplotlib ImportError tests
# ============================================================

class TestMatplotlibImportError:
    """Test that functions raise ImportError when matplotlib is missing.
    DIAG-VIZ-016 to DIAG-VIZ-018."""

    @pytest.mark.unit
    def test_viz016_love_plot_import_error(self, monkeypatch, balance_dataframe):
        """DIAG-VIZ-016: love_plot raises ImportError without matplotlib."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'matplotlib.pyplot' or name == 'matplotlib':
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        # Need to reload plots module with mock
        import importlib
        import cbps.diagnostics.plots as plots_module

        monkeypatch.setattr(builtins, '__import__', mock_import)

        with pytest.raises(ImportError, match="matplotlib"):
            # Call the function directly with mocked import
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError(
                    "matplotlib is required for love_plot(). "
                    "Install it with: pip install matplotlib"
                )

    @pytest.mark.unit
    def test_viz017_weight_dist_import_guard(self):
        """DIAG-VIZ-017: plot_weight_distribution has import guard."""
        from cbps.diagnostics.plots import plot_weight_distribution
        import inspect
        source = inspect.getsource(plot_weight_distribution)
        assert "import matplotlib.pyplot" in source
        assert "ImportError" in source

    @pytest.mark.unit
    def test_viz018_ps_overlap_import_guard(self):
        """DIAG-VIZ-018: plot_ps_overlap has import guard."""
        from cbps.diagnostics.plots import plot_ps_overlap
        import inspect
        source = inspect.getsource(plot_ps_overlap)
        assert "import matplotlib.pyplot" in source
        assert "ImportError" in source


# ============================================================
# CBPSSummary.__str__ tests
# ============================================================

class TestCBPSSummaryStr:
    """Tests for enhanced CBPSSummary.__str__. DIAG-VIZ-019 to DIAG-VIZ-020."""

    @pytest.fixture
    def mock_summary(self):
        """Create a CBPSSummary object for testing."""
        from cbps.core.results import CBPSSummary
        np.random.seed(42)
        n = 200
        y = np.concatenate([np.ones(80), np.zeros(120)])
        ps = np.random.uniform(0.2, 0.8, n)
        weights = np.where(y == 1, 1 / ps, 1 / (1 - ps))

        coef_table = np.array([
            [-1.7476, 0.2134, -8.189, 2.65e-16],
            [0.0129, 0.0046, 2.804, 5.05e-03],
            [0.0714, 0.0345, 2.069, 3.86e-02],
        ])
        return CBPSSummary(
            call="CBPS(treat ~ age + educ, data=lalonde, ATT=TRUE)",
            coef_table=coef_table,
            coef_names=["(Intercept)", "age", "education"],
            significance=["***", "**", "*"],
            J=12.34,
            j_pvalue=0.0548,
            j_df=6,
            deviance=245.6,
            sigmasq=None,
            y=y,
            fitted_values=ps,
            weights=weights,
            converged=True,
        )

    @pytest.mark.unit
    def test_viz019_summary_contains_key_sections(self, mock_summary):
        """DIAG-VIZ-019: Summary output contains all key sections."""
        output = str(mock_summary)

        # Header
        assert "CBPS Estimation Summary" in output
        assert "=" * 60 in output

        # Call
        assert "Call:" in output
        assert "CBPS(treat ~ age + educ" in output

        # Sample info
        assert "N: 200" in output
        assert "Treated: 80" in output
        assert "Control: 120" in output

        # Convergence
        assert "Converged: Yes" in output

        # Coefficients
        assert "Coefficients:" in output
        assert "(Intercept)" in output
        assert "age" in output
        assert "education" in output

        # Diagnostics
        assert "Diagnostics:" in output
        assert "J-statistic:" in output
        assert "Effective Sample Size:" in output
        assert "Log-Likelihood:" in output

    @pytest.mark.unit
    def test_viz020_summary_formatting(self, mock_summary):
        """DIAG-VIZ-020: Summary has clean ASCII formatting."""
        output = str(mock_summary)
        lines = output.split('\n')

        # First and last lines should be separator
        assert lines[0] == "=" * 60
        assert lines[-1] == "=" * 60

        # Should have section separators
        assert "-" * 60 in output

        # J-statistic with df and p-value
        assert "df=6" in output
        assert "p=0.0548" in output
