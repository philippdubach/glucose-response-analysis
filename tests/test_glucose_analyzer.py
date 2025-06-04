"""Unit tests for glucose response analyzer."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models.glucose_response_analyzer import GlucoseResponseAnalyzer
from src.models.xgboost_regressor import GlucoseXGBoostRegressor

class TestGlucoseResponseAnalyzer:
    """Test cases for GlucoseResponseAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        df_meals = pd.DataFrame({
            'userID': ['user_1'] * 50 + ['user_2'] * 50,
            'Meal': ['breakfast'] * 25 + ['lunch'] * 25 + ['breakfast'] * 25 + ['lunch'] * 25,
            'time': pd.date_range('2023-01-01', periods=n_samples, freq='5min'),
            'GlucoseValue': 90 + np.random.normal(0, 10, n_samples),
            'meal_taken': [0] * n_samples
        })
        
        # Mark some meals
        df_meals.loc[24, 'meal_taken'] = 1  # First meal
        df_meals.loc[74, 'meal_taken'] = 1  # Second meal
        
        return df_meals
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return GlucoseResponseAnalyzer()
    
    def test_normalized_gaussian(self, analyzer):
        """Test normalized Gaussian function."""
        t = np.array([0, 10, 20, 30, 40])
        A, delta, sigma, b = 20, 20, 10, 90
        
        result = analyzer.normalized_gaussian(t, A, delta, sigma, b)
        
        assert len(result) == len(t)
        assert all(r >= b for r in result)  # All values should be above baseline
        assert result[2] == max(result)  # Peak should be at delta=20
    
    def test_fit_single_meal(self, analyzer, sample_data):
        """Test fitting a single meal."""
        meal_index = sample_data[sample_data['meal_taken'] == 1].index[0]
        
        result = analyzer.fit_single_meal(sample_data, meal_index)
        
        assert result['success'] == True
        assert 'A' in result
        assert 'delta' in result
        assert 'sigma' in result
        assert 'baseline' in result
        assert result['A'] > 0
        assert result['delta'] >= 0
        assert result['sigma'] > 0
    
    def test_fit_all_meals(self, analyzer, sample_data):
        """Test fitting all meals."""
        result_df = analyzer.fit_all_meals(sample_data)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert all(col in result_df.columns for col in ['A', 'delta', 'sigma', 'userID'])

class TestGlucoseXGBoostRegressor:
    """Test cases for GlucoseXGBoostRegressor."""
    
    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data."""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'A': 20 + np.random.normal(0, 5, n_samples),
            'delta': 30 + np.random.normal(0, 10, n_samples),
            'sigma': 15 + np.random.normal(0, 3, n_samples),
            'CHO': 40 + np.random.normal(0, 15, n_samples),
            'PRO': 20 + np.random.normal(0, 8, n_samples),
            'FAT': 15 + np.random.normal(0, 5, n_samples),
            'age': 45 + np.random.normal(0, 10, n_samples),
            'BMI': 24 + np.random.normal(0, 3, n_samples),
            'hour': np.random.randint(6, 22, n_samples),
            'is_breakfast': np.random.choice([0, 1], n_samples)
        })
    
    @pytest.fixture
    def regressor(self):
        """Create regressor instance."""
        return GlucoseXGBoostRegressor()
    
    def test_prepare_features(self, regressor, sample_ml_data):
        """Test feature preparation."""
        X, feature_names = regressor.prepare_features(sample_ml_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(fname in sample_ml_data.columns for fname in feature_names)
    
    def test_train_single_target(self, regressor, sample_ml_data):
        """Test training a single target."""
        X, _ = regressor.prepare_features(sample_ml_data)
        y = sample_ml_data['A']
        
        model = regressor.train_single_target(X, y, 'amplitude')
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'train_metrics_')
        assert hasattr(model, 'test_metrics_')
    
    def test_model_metrics_calculation(self, regressor):
        """Test metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = regressor._calculate_metrics(y_true, y_pred, 'test')
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r_squared' in metrics
        assert 'correlation' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert 0 <= metrics['r_squared'] <= 1

if __name__ == "__main__":
    pytest.main([__file__])