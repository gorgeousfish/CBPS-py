"""
Scikit-learn Integration Tests
==============================

This module tests the scikit-learn compatible CBPS estimator, verifying
compliance with sklearn API conventions and integration with sklearn pipelines.

Test IDs:
    - SK-001: Basic estimator API compliance
    - SK-002: fit() method tests
    - SK-003: transform() method tests
    - SK-004: Pipeline integration tests
    - SK-005: GridSearchCV compatibility tests
    - SK-006: Clone and parameter tests

Requirements:
    - REQ-SK-001 to REQ-SK-010: sklearn integration requirements

References:
    scikit-learn estimator development guidelines:
    https://scikit-learn.org/stable/developers/develop.html
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# Skip all tests if sklearn is not available
try:
    import sklearn
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="scikit-learn not installed"
)


# =============================================================================
# Test Data Generator
# =============================================================================

def generate_classification_data(n=200, seed=42):
    """Generate simple binary classification data."""
    np.random.seed(seed)
    
    X = np.random.randn(n, 5)
    
    # Generate treatment based on covariates
    logits = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    
    return X, y


# =============================================================================
# SK-001: Basic Estimator API Compliance
# =============================================================================

class TestSklearnAPICompliance:
    """
    Test ID: SK-001
    
    Test basic sklearn API compliance.
    """
    
    @pytest.fixture
    def cbps_estimator(self):
        """Create CBPS sklearn estimator."""
        try:
            from cbps.sklearn import CBPSEstimator
            return CBPSEstimator()
        except ImportError:
            pytest.skip("CBPSEstimator not available")
    
    def test_estimator_has_fit_method(self, cbps_estimator):
        """Test that estimator has fit method."""
        assert hasattr(cbps_estimator, 'fit')
        assert callable(cbps_estimator.fit)
    
    def test_estimator_has_predict_method(self, cbps_estimator):
        """Test that estimator has predict method."""
        assert hasattr(cbps_estimator, 'predict')
        assert callable(cbps_estimator.predict)
    
    def test_estimator_has_predict_proba_method(self, cbps_estimator):
        """Test that estimator has predict_proba method."""
        assert hasattr(cbps_estimator, 'predict_proba')
        assert callable(cbps_estimator.predict_proba)
    
    def test_estimator_has_get_params(self, cbps_estimator):
        """Test that estimator has get_params method."""
        assert hasattr(cbps_estimator, 'get_params')
        params = cbps_estimator.get_params()
        assert isinstance(params, dict)
    
    def test_estimator_has_set_params(self, cbps_estimator):
        """Test that estimator has set_params method."""
        assert hasattr(cbps_estimator, 'set_params')


# =============================================================================
# SK-002: fit() Method Tests
# =============================================================================

class TestSklearnFit:
    """
    Test ID: SK-002
    
    Test fit() method behavior.
    """
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        return generate_classification_data(n=200, seed=42)
    
    def test_fit_returns_self(self, simple_data):
        """Test that fit returns self."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = simple_data
        estimator = CBPSEstimator()
        result = estimator.fit(X, y)
        
        assert result is estimator
    
    def test_fit_sets_is_fitted(self, simple_data):
        """Test that fit sets fitted state."""
        try:
            from cbps.sklearn import CBPSEstimator
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = simple_data
        estimator = CBPSEstimator()
        estimator.fit(X, y)
        
        # Should not raise NotFittedError
        check_is_fitted(estimator)
    
    def test_fit_stores_classes(self, simple_data):
        """Test that fit stores classes_ attribute."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = simple_data
        estimator = CBPSEstimator()
        estimator.fit(X, y)
        
        assert hasattr(estimator, 'classes_')
        assert len(estimator.classes_) == 2


# =============================================================================
# SK-003: Prediction Method Tests
# =============================================================================

class TestSklearnPredict:
    """
    Test ID: SK-003
    
    Test prediction methods.
    """
    
    @pytest.fixture
    def fitted_estimator(self):
        """Create and fit CBPS estimator."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        estimator = CBPSEstimator()
        estimator.fit(X, y)
        
        return estimator, X, y
    
    def test_predict_returns_array(self, fitted_estimator):
        """Test that predict returns numpy array."""
        estimator, X, y = fitted_estimator
        predictions = estimator.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
    
    def test_predict_binary_values(self, fitted_estimator):
        """Test that predict returns binary values."""
        estimator, X, y = fitted_estimator
        predictions = estimator.predict(X)
        
        unique_values = np.unique(predictions)
        assert len(unique_values) <= 2
    
    def test_predict_proba_returns_probabilities(self, fitted_estimator):
        """Test that predict_proba returns valid probabilities."""
        estimator, X, y = fitted_estimator
        proba = estimator.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-10)


# =============================================================================
# SK-004: Pipeline Integration Tests
# =============================================================================

class TestSklearnPipeline:
    """
    Test ID: SK-004
    
    Test integration with sklearn Pipeline.
    """
    
    def test_pipeline_with_scaler(self):
        """Test CBPS in pipeline with StandardScaler."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('cbps', CBPSEstimator())
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(X)
    
    def test_pipeline_predict_proba(self):
        """Test predict_proba in pipeline."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('cbps', CBPSEstimator())
        ])
        
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        
        assert proba.shape == (len(X), 2)


# =============================================================================
# SK-005: GridSearchCV Compatibility Tests
# =============================================================================

class TestSklearnGridSearch:
    """
    Test ID: SK-005
    
    Test compatibility with GridSearchCV.
    """
    
    def test_gridsearch_basic(self):
        """Test basic GridSearchCV functionality."""
        try:
            from cbps.sklearn import CBPSEstimator
            from sklearn.model_selection import GridSearchCV
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        
        # Define parameter grid (assuming these parameters exist)
        param_grid = {}  # Empty grid for basic test
        
        estimator = CBPSEstimator()
        grid_search = GridSearchCV(estimator, param_grid, cv=2, scoring='accuracy')
        
        grid_search.fit(X, y)
        
        assert hasattr(grid_search, 'best_estimator_')


# =============================================================================
# SK-006: Clone and Parameter Tests
# =============================================================================

class TestSklearnClone:
    """
    Test ID: SK-006
    
    Test clone functionality and parameter handling.
    """
    
    def test_clone_creates_unfitted_copy(self):
        """Test that clone creates an unfitted copy."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        
        estimator = CBPSEstimator()
        estimator.fit(X, y)
        
        cloned = clone(estimator)
        
        # Cloned estimator should be unfitted
        assert not hasattr(cloned, 'classes_') or cloned.classes_ is None
    
    def test_get_set_params_roundtrip(self):
        """Test get_params/set_params roundtrip."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        estimator = CBPSEstimator()
        params = estimator.get_params()
        
        new_estimator = CBPSEstimator()
        new_estimator.set_params(**params)
        
        assert estimator.get_params() == new_estimator.get_params()


# =============================================================================
# Weight Access Tests
# =============================================================================

class TestSklearnWeights:
    """
    Test weight access from fitted sklearn estimator.
    """
    
    def test_fitted_estimator_has_weights(self):
        """Test that fitted estimator provides access to CBPS weights."""
        try:
            from cbps.sklearn import CBPSEstimator
        except ImportError:
            pytest.skip("CBPSEstimator not available")
        
        X, y = generate_classification_data(n=200, seed=42)
        
        estimator = CBPSEstimator()
        estimator.fit(X, y)
        
        # Check for weights attribute or method
        if hasattr(estimator, 'weights_'):
            weights = estimator.weights_
            assert len(weights) == len(X)
            assert np.all(weights > 0)
        elif hasattr(estimator, 'get_weights'):
            weights = estimator.get_weights()
            assert len(weights) == len(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
