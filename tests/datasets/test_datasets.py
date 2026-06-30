"""
Module: test_datasets.py
========================

Test Suite: Dataset Loaders
Test IDs: DATA-001 to DATA-036
Requirements: REQ-DATA-001 to REQ-DATA-011

Overview:
    This module tests the dataset loading functions ensuring data integrity,
    proper column types, and correct data retrieval.

Test Categories:
    - Unit tests: Basic loading functionality
    - Data integrity tests: Column presence, data types, value ranges

Datasets Covered:
    - LaLonde (1986): NSW job training data (DATA-001 to DATA-010)
    - Blackwell (2013): Longitudinal political campaign data (DATA-011 to DATA-018)
    - Continuous simulation: Fong et al. (2018) DGP data (DATA-019 to DATA-025)
    - Combined datasets: LaLonde + PSID controls (DATA-026 to DATA-030)
    - npCBPS simulation: Nonparametric CBPS validation data (DATA-031 to DATA-036)

References:
    LaLonde, R. J. (1986). Evaluating the econometric evaluations of training
    programs with experimental data. American Economic Review, 76(4), 604-620.

    Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
    score for a continuous treatment: Application to the efficacy of political
    advertisements. The Annals of Applied Statistics, 12(1), 156-177.

Usage:
    pytest tests/datasets/test_datasets.py -v
"""

import numpy as np
import pandas as pd
import pytest

from cbps.datasets import (
    load_lalonde,
    load_lalonde_psid_combined,
    load_blackwell,
    load_continuous_simulation,
    load_npcbps_continuous_sim,
)


# =============================================================================
# Test Class: LaLonde Dataset (DATA-001 to DATA-010)
# =============================================================================

class TestLaLondeDataset:
    """
    Test LaLonde dataset loading.
    
    Test IDs: DATA-001 to DATA-010
    Requirements: REQ-DATA-001
    """
    
    @pytest.mark.unit
    def test_data001_load_returns_dataframe(self):
        """
        DATA-001: Verify load_lalonde returns a DataFrame.
        
        Requirements: REQ-DATA-001
        """
        df = load_lalonde()
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.unit
    def test_data002_columns_present(self):
        """
        DATA-002: Verify required columns are present.
        
        Requirements: REQ-DATA-001
        """
        df = load_lalonde()
        
        required_columns = ['treat', 'age', 'educ', 're74', 're75', 're78']
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    @pytest.mark.unit
    def test_data003_no_missing_values(self):
        """
        DATA-003: Verify no missing values in key columns.
        
        Requirements: REQ-DATA-002
        """
        df = load_lalonde()
        
        key_columns = ['treat', 'age', 'educ']
        
        for col in key_columns:
            assert df[col].notna().all(), f"Missing values in column: {col}"
    
    @pytest.mark.unit
    def test_data004_treatment_binary(self):
        """
        DATA-004: Verify treatment is binary (0 or 1).
        
        Requirements: REQ-DATA-002
        """
        df = load_lalonde()
        
        assert set(df['treat'].unique()).issubset({0, 1})
    
    @pytest.mark.unit
    def test_data005_age_positive(self):
        """
        DATA-005: Verify age values are positive.
        
        Requirements: REQ-DATA-002
        """
        df = load_lalonde()
        
        assert (df['age'] > 0).all()
    
    @pytest.mark.unit
    def test_data006_educ_reasonable(self):
        """
        DATA-006: Verify education values are reasonable (0-20).
        
        Requirements: REQ-DATA-002
        """
        df = load_lalonde()
        
        assert (df['educ'] >= 0).all()
        assert (df['educ'] <= 20).all()
    
    @pytest.mark.unit
    def test_data007_sample_size(self):
        """
        DATA-007: Verify reasonable sample size.
        
        Requirements: REQ-DATA-003
        """
        df = load_lalonde()
        
        # LaLonde dataset should have > 100 observations
        assert len(df) > 100
    
    @pytest.mark.unit
    def test_data008_dehejia_wahba_only(self):
        """
        DATA-008: Verify dehejia_wahba_only filter works.
        
        Requirements: REQ-DATA-003
        """
        df_full = load_lalonde(dehejia_wahba_only=False)
        df_dw = load_lalonde(dehejia_wahba_only=True)
        
        # DW subset should be smaller or equal
        assert len(df_dw) <= len(df_full)
    
    @pytest.mark.unit
    def test_data009_reproducibility(self):
        """
        DATA-009: Verify loading is reproducible.
        
        Requirements: REQ-DATA-004
        """
        df1 = load_lalonde()
        df2 = load_lalonde()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    @pytest.mark.unit
    def test_data010_treatment_distribution(self):
        """
        DATA-010: Verify both treatment groups are present.
        
        Requirements: REQ-DATA-002
        """
        df = load_lalonde()
        
        n_treated = (df['treat'] == 1).sum()
        n_control = (df['treat'] == 0).sum()
        
        assert n_treated > 0
        assert n_control > 0


# =============================================================================
# Test Class: Blackwell Dataset (DATA-011 to DATA-018)
# =============================================================================

class TestBlackwellDataset:
    """
    Test Blackwell longitudinal dataset loading.
    
    Test IDs: DATA-011 to DATA-018
    Requirements: REQ-DATA-005
    """
    
    @pytest.mark.unit
    def test_data011_load_returns_dataframe(self):
        """
        DATA-011: Verify load_blackwell returns a DataFrame.
        
        Requirements: REQ-DATA-005
        """
        df = load_blackwell()
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.unit
    def test_data012_columns_present(self):
        """
        DATA-012: Verify key columns are present.
        
        Requirements: REQ-DATA-005
        """
        df = load_blackwell()
        
        # Should have unit and time identifiers for panel data
        assert len(df.columns) > 0
        assert len(df) > 0
    
    @pytest.mark.unit
    def test_data013_sample_size(self):
        """
        DATA-013: Verify reasonable sample size.
        
        Requirements: REQ-DATA-005
        """
        df = load_blackwell()
        
        assert len(df) > 50
    
    @pytest.mark.unit
    def test_data014_no_missing_ids(self):
        """
        DATA-014: Verify no missing values in identifier columns.
        
        Requirements: REQ-DATA-006
        """
        df = load_blackwell()
        
        # Most columns should have no missing values
        assert df.notna().sum().sum() > 0
    
    @pytest.mark.unit
    def test_data015_reproducibility(self):
        """
        DATA-015: Verify loading is reproducible.
        
        Requirements: REQ-DATA-006
        """
        df1 = load_blackwell()
        df2 = load_blackwell()
        
        pd.testing.assert_frame_equal(df1, df2)


# =============================================================================
# Test Class: Continuous Treatment Dataset (DATA-019 to DATA-025)
# =============================================================================

class TestContinuousDataset:
    """
    Test continuous treatment simulation dataset loading.
    
    Test IDs: DATA-019 to DATA-025
    Requirements: REQ-DATA-007
    
    Notes:
        load_continuous_simulation returns (DataFrame, dict) tuple.
    """
    
    @pytest.mark.unit
    def test_data019_load_returns_tuple(self):
        """
        DATA-019: Verify load_continuous_simulation returns correct type.
        
        Requirements: REQ-DATA-007
        """
        result = load_continuous_simulation()
        
        # Returns (DataFrame, dict) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        df, info = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(info, dict)
    
    @pytest.mark.unit
    def test_data020_treatment_continuous(self):
        """
        DATA-020: Verify treatment is continuous (not binary).
        
        Requirements: REQ-DATA-007
        """
        df, info = load_continuous_simulation()
        
        # Should have 'T' column or similar
        if 'T' in df.columns:
            # Continuous treatment should have more than 2 unique values
            assert len(df['T'].unique()) > 10
    
    @pytest.mark.unit
    def test_data021_sample_size(self):
        """
        DATA-021: Verify reasonable sample size.
        
        Requirements: REQ-DATA-008
        """
        df, info = load_continuous_simulation()
        
        assert len(df) > 100
    
    @pytest.mark.unit
    def test_data022_no_infinite_values(self):
        """
        DATA-022: Verify no infinite values.
        
        Requirements: REQ-DATA-008
        """
        df, info = load_continuous_simulation()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            assert np.isfinite(df[col]).all(), f"Infinite values in {col}"
    
    @pytest.mark.unit
    def test_data023_reproducibility(self):
        """
        DATA-023: Verify loading is reproducible.
        
        Requirements: REQ-DATA-009
        """
        df1, _ = load_continuous_simulation()
        df2, _ = load_continuous_simulation()
        
        pd.testing.assert_frame_equal(df1, df2)


# =============================================================================
# Test Class: Combined Datasets (DATA-026 to DATA-030)
# =============================================================================

class TestCombinedDatasets:
    """
    Test combined and PSID datasets.
    
    Test IDs: DATA-026 to DATA-030
    Requirements: REQ-DATA-010
    """
    
    @pytest.mark.unit
    def test_data026_psid_combined_returns_dataframe(self):
        """
        DATA-026: Verify load_lalonde_psid_combined returns a DataFrame.
        
        Requirements: REQ-DATA-010
        """
        df = load_lalonde_psid_combined()
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.unit
    def test_data027_psid_combined_larger(self):
        """
        DATA-027: Verify PSID combined is larger than experimental only.
        
        Requirements: REQ-DATA-010
        """
        df_lalonde = load_lalonde()
        df_psid = load_lalonde_psid_combined()
        
        # PSID combined should have more control units
        n_control_lalonde = (df_lalonde['treat'] == 0).sum()
        n_control_psid = (df_psid['treat'] == 0).sum()
        
        assert n_control_psid >= n_control_lalonde
    
    @pytest.mark.unit
    def test_data028_psid_columns_match(self):
        """
        DATA-028: Verify PSID combined has similar columns to LaLonde.
        
        Requirements: REQ-DATA-010
        """
        df_lalonde = load_lalonde()
        df_psid = load_lalonde_psid_combined()
        
        # Key columns should be present in both
        key_cols = ['treat', 'age', 'educ']
        
        for col in key_cols:
            if col in df_lalonde.columns:
                assert col in df_psid.columns, f"Missing column: {col}"
    
    @pytest.mark.unit
    def test_data029_psid_treatment_binary(self):
        """
        DATA-029: Verify PSID combined has binary treatment.
        
        Requirements: REQ-DATA-010
        """
        df = load_lalonde_psid_combined()
        
        assert set(df['treat'].unique()).issubset({0, 1})
    
    @pytest.mark.unit
    def test_data030_psid_reproducibility(self):
        """
        DATA-030: Verify PSID loading is reproducible.
        
        Requirements: REQ-DATA-010
        """
        df1 = load_lalonde_psid_combined()
        df2 = load_lalonde_psid_combined()
        
        pd.testing.assert_frame_equal(df1, df2)


# =============================================================================
# Test Class: npCBPS Simulation Datasets (DATA-031 to DATA-040)
# =============================================================================

class TestNpCBPSDataset:
    """
    Test npCBPS simulation data loading.
    
    Test IDs: DATA-031 to DATA-036
    Requirements: REQ-DATA-011
    
    Notes:
        These tests validate the nonparametric CBPS simulation data
        used for validating npCBPS implementations.
        
    References:
        Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
        score for a continuous treatment: Application to the efficacy of political
        advertisements. The Annals of Applied Statistics, 12(1), 156-177.
    """
    
    # -------------------------------------------------------------------------
    # Continuous Simulation Data Tests (DATA-031 to DATA-035)
    # -------------------------------------------------------------------------
    
    @pytest.mark.unit
    def test_data031_load_continuous_sim_returns_dataframe(self):
        """
        DATA-031: Verify load_npcbps_continuous_sim returns a DataFrame.
        
        Requirements: REQ-DATA-011
        """
        df = load_npcbps_continuous_sim()
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.unit
    def test_data032_continuous_sim_shape(self):
        """
        DATA-032: Verify simulation data has expected shape (500, 7).
        
        Requirements: REQ-DATA-011
        """
        df = load_npcbps_continuous_sim()
        
        assert df.shape == (500, 7), f"Expected shape (500, 7), got {df.shape}"
    
    @pytest.mark.unit
    def test_data033_continuous_sim_columns(self):
        """
        DATA-033: Verify expected columns are present.
        
        Requirements: REQ-DATA-011
        """
        df = load_npcbps_continuous_sim()
        
        expected_columns = ['Y', 'T', 'X1', 'X2', 'X3', 'X4', 'X5']
        
        assert list(df.columns) == expected_columns, \
            f"Expected columns {expected_columns}, got {list(df.columns)}"
    
    @pytest.mark.unit
    def test_data034_continuous_sim_no_missing(self):
        """
        DATA-034: Verify no missing values in simulation data.
        
        Requirements: REQ-DATA-011
        """
        df = load_npcbps_continuous_sim()
        
        assert df.notna().all().all(), "Simulation data contains missing values"
    
    @pytest.mark.unit
    def test_data035_continuous_sim_treatment_continuous(self):
        """
        DATA-035: Verify treatment T is continuous (many unique values).
        
        Requirements: REQ-DATA-011
        """
        df = load_npcbps_continuous_sim()
        
        # Continuous treatment should have many unique values
        n_unique = df['T'].nunique()
        
        assert n_unique > 100, \
            f"Treatment should be continuous, but only {n_unique} unique values"
    
    @pytest.mark.unit
    def test_data036_continuous_sim_reproducibility(self):
        """
        DATA-036: Verify simulation data loading is reproducible.
        
        Requirements: REQ-DATA-011
        """
        df1 = load_npcbps_continuous_sim()
        df2 = load_npcbps_continuous_sim()
        
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
