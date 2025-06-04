"""XGBoost regressor for glucose response parameter prediction."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class GlucoseXGBoostRegressor:
    """XGBoost regressor for predicting glucose response parameters (A, δ, σ)."""
    
    def __init__(self):
        self.config = config
        self.models = {}  # Will store separate models for A, delta, sigma
        self.scalers = {}  # Feature scalers for each target
        self.feature_names = None
        self.shap_explainers = {}
        
        # Model parameters from the research paper
        self.default_params = {
            'n_estimators': self.config.get('model.n_estimators', 200),
            'max_depth': self.config.get('model.max_depth', 5),
            'learning_rate': self.config.get('model.learning_rate', 0.01),
            'subsample': self.config.get('model.subsample', 0.5),
            'objective': 'reg:squarederror',
            'random_state': self.config.get('model.random_state', 42),
            'n_jobs': -1
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for model training."""
        logger.info("Preparing features for XGBoost model...")
        
        # Define feature columns based on the research paper
        demographic_features = ['age', 'gender', 'BMI', 'height', 'weight']
        meal_features = ['CHO', 'PRO', 'FAT']
        
        # Time-based features
        time_features = ['hour', 'day_of_week', 'is_breakfast', 'is_lunch', 'is_dinner', 'is_snack']
        
        # CGM statistics features (last 24h and 4h)
        cgm_24h_features = [
            'cgm_24h_mean', 'cgm_24h_std', 'cgm_24h_TIR', 'cgm_24h_IQR',
            'cgm_24h_below_70_pct', 'cgm_24h_above_140_pct', 'cgm_24h_skewness', 'cgm_24h_kurtosis'
        ]
        
        cgm_4h_features = [
            'cgm_4h_mean', 'cgm_4h_std', 'cgm_4h_TIR', 'cgm_4h_IQR',
            'cgm_4h_below_70_pct', 'cgm_4h_above_140_pct', 'cgm_4h_skewness', 'cgm_4h_kurtosis'
        ]
        
        # Glucotype features
        glucotype_features = ['GRT1_24h_pct', 'GRT2_24h_pct', 'GRT3_24h_pct',
                             'GRT1_4h_pct', 'GRT2_4h_pct', 'GRT3_4h_pct']
        
        # Autocorrelation features
        autocorr_features = ['autocorr_8h', 'autocorr_4h', 'autocorr_1h',
                           'autocorr_40min', 'autocorr_20min', 'autocorr_10min']
        
        # User-specific features
        user_features = [col for col in df.columns if col.startswith('user_')]
        
        # Combine all available features
        all_possible_features = (demographic_features + meal_features + time_features + 
                               cgm_24h_features + cgm_4h_features + glucotype_features + 
                               autocorr_features + user_features)
        
        # Select only available features
        available_features = [f for f in all_possible_features if f in df.columns]
        
        logger.info(f"Selected {len(available_features)} features for model training")
        
        return df[available_features], available_features
    
    def train_single_target(self, X: pd.DataFrame, y: pd.Series, target_name: str,
                          hyperparameter_tuning: bool = False) -> xgb.XGBRegressor:
        """Train XGBoost model for a single target parameter."""
        logger.info(f"Training XGBoost model for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.get('model.test_size', 0.2),
            random_state=self.config.get('model.random_state', 42)
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.scalers[target_name] = scaler
        
        if hyperparameter_tuning:
            model = self._hyperparameter_tuning(X_train_scaled, y_train, target_name)
        else:
            model = xgb.XGBRegressor(**self.default_params)
            model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train, target_name)
        test_metrics = self._calculate_metrics(y_test, y_pred_test, target_name)
        
        logger.info(f"{target_name} - Train R²: {train_metrics['r_squared']:.3f}, "
                   f"Test R²: {test_metrics['r_squared']:.3f}")
        logger.info(f"{target_name} - Test MAE: {test_metrics['mae']:.3f}, "
                   f"Test RMSE: {test_metrics['rmse']:.3f}")
        
        # Store model evaluation results
        model.train_metrics_ = train_metrics
        model.test_metrics_ = test_metrics
        model.X_test_ = X_test_scaled
        model.y_test_ = y_test
        
        return model
    
    def _hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              target_name: str) -> xgb.XGBRegressor:
        """Perform hyperparameter tuning using GridSearchCV."""
        logger.info(f"Performing hyperparameter tuning for {target_name}...")
        
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.001, 0.01, 0.1],
            'subsample': [0.5, 0.7, 0.9]
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.config.get('model.random_state', 42),
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config.get('model.cv_folds', 5),
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {target_name}: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          target_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate percentage MAE
        mae_pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate R-squared
        r_squared = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(y_true, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mae_pct': mae_pct,
            'r_squared': r_squared,
            'correlation': correlation,
            'p_value': p_value
        }
    
    def train_all_targets(self, df_ml: pd.DataFrame, hyperparameter_tuning: bool = False) -> Dict:
        """Train XGBoost models for all target parameters (A, δ, σ)."""
        logger.info("Training XGBoost models for all targets...")
        
        # Prepare features
        X, feature_names = self.prepare_features(df_ml)
        self.feature_names = feature_names
        
        # Define targets
        targets = {
            'A': 'amplitude',
            'delta': 'time_to_peak', 
            'sigma': 'curve_width'
        }
        
        results = {}
        
        for target_col, target_name in targets.items():
            if target_col in df_ml.columns:
                y = df_ml[target_col]
                model = self.train_single_target(X, y, target_name, hyperparameter_tuning)
                self.models[target_name] = model
                results[target_name] = {
                    'model': model,
                    'train_metrics': model.train_metrics_,
                    'test_metrics': model.test_metrics_
                }
                
                # Create SHAP explainer
                self.shap_explainers[target_name] = shap.TreeExplainer(model)
        
        logger.info("All models trained successfully")
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions for all targets."""
        predictions = {}
        
        for target_name, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[target_name].transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            predictions[target_name] = model.predict(X_scaled)
        
        return predictions
    
    def calculate_shap_values(self, X: pd.DataFrame, target_name: str) -> np.ndarray:
        """Calculate SHAP values for feature importance."""
        if target_name not in self.shap_explainers:
            raise ValueError(f"No SHAP explainer available for {target_name}")
        
        # Scale features
        X_scaled = self.scalers[target_name].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return self.shap_explainers[target_name].shap_values(X_scaled)
    
    def create_bland_altman_analysis(self, target_name: str) -> Dict:
        """Create Bland-Altman analysis for model evaluation."""
        if target_name not in self.models:
            raise ValueError(f"No model trained for {target_name}")
        
        model = self.models[target_name]
        y_true = model.y_test_
        y_pred = model.predict(model.X_test_)
        
        # Calculate differences and means
        diff = y_pred - y_true
        mean_vals = (y_pred + y_true) / 2
        
        # Calculate bias and limits of agreement
        bias = np.mean(diff)
        std_diff = np.std(diff)
        la95_upper = bias + 1.96 * std_diff
        la95_lower = bias - 1.96 * std_diff
        
        # Calculate correlation of differences with means (systematic bias)
        bias_correlation, bias_p_value = pearsonr(mean_vals, diff)
        
        return {
            'bias': bias,
            'la95_upper': la95_upper,
            'la95_lower': la95_lower,
            'std_diff': std_diff,
            'bias_correlation': bias_correlation,
            'bias_p_value': bias_p_value,
            'y_true': y_true,
            'y_pred': y_pred,
            'differences': diff,
            'means': mean_vals
        }
    
    def save_models(self, save_path: str = None):
        """Save trained models and scalers."""
        if save_path is None:
            save_path = config.get_results_path('models')
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for target_name, model in self.models.items():
            model_path = save_path / f'xgboost_{target_name}.joblib'
            scaler_path = save_path / f'scaler_{target_name}.joblib'
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[target_name], scaler_path)
        
        # Save feature names
        feature_path = save_path / 'feature_names.joblib'
        joblib.dump(self.feature_names, feature_path)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str = None):
        """Load trained models and scalers."""
        if load_path is None:
            load_path = config.get_results_path('models')
        
        load_path = Path(load_path)
        
        for target_name in ['amplitude', 'time_to_peak', 'curve_width']:
            model_path = load_path / f'xgboost_{target_name}.joblib'
            scaler_path = load_path / f'scaler_{target_name}.joblib'
            
            if model_path.exists() and scaler_path.exists():
                self.models[target_name] = joblib.load(model_path)
                self.scalers[target_name] = joblib.load(scaler_path)
        
        # Load feature names
        feature_path = load_path / 'feature_names.joblib'
        if feature_path.exists():
            self.feature_names = joblib.load(feature_path)
        
        logger.info(f"Models loaded from {load_path}")

class MultiLinearRegressor:
    """Multi-linear regression model for glucose response amplitude prediction."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for multi-linear regression (CHO, PRO, FAT and interactions)."""
        features_df = df[['CHO', 'PRO', 'FAT']].copy()
        
        # Add interaction terms
        features_df['CHO_PRO'] = features_df['CHO'] * features_df['PRO']
        features_df['CHO_FAT'] = features_df['CHO'] * features_df['FAT']
        features_df['PRO_FAT'] = features_df['PRO'] * features_df['FAT']
        
        self.feature_names = features_df.columns.tolist()
        return features_df
    
    def train(self, df_ml: pd.DataFrame) -> Dict:
        """Train multi-linear regression model."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X = self.prepare_features(df_ml)
        y = df_ml['A']  # Amplitude parameter
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Get coefficients and p-values
        from scipy import stats
        n = len(y)
        p = X.shape[1]
        
        # Calculate t-statistics and p-values
        mse = np.sum((y - y_pred) ** 2) / (n - p - 1)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_beta = np.sqrt(var_beta)
        t_stats = self.model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        results = {
            'r2': r2,
            'rse': rse,
            'coefficients': dict(zip(self.feature_names, self.model.coef_)),
            'p_values': dict(zip(self.feature_names, p_values)),
            'intercept': self.model.intercept_
        }
        
        return results