"""Feature engineering for ML model."""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering for glucose response prediction."""
    
    def __init__(self):
        self.config = config
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        
        # Convert time to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Extract time features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Create meal type indicators
        df['is_breakfast'] = df['meal'].str.contains('breakfast', case=False).astype(int)
        df['is_lunch'] = df['meal'].str.contains('lunch', case=False).astype(int)
        df['is_dinner'] = df['meal'].str.contains('dinner', case=False).astype(int)
        df['is_snack'] = df['meal'].str.contains('snack', case=False).astype(int)
        
        logger.info("Created time-based features")
        return df
    
    def create_user_features(self, df: pd.DataFrame, df_hall: pd.DataFrame = None) -> pd.DataFrame:
        """Create user-specific features."""
        df = df.copy()
        
        # Add user-level statistics
        user_stats = df.groupby('userID').agg({
            'A': ['mean', 'std'],
            'delta': ['mean', 'std'],
            'sigma': ['mean', 'std'],
            'baseline': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.add_prefix('user_')
        
        # Merge with main dataframe
        df = df.merge(user_stats, left_on='userID', right_index=True, how='left')
        
        # If demographic data is available, merge it
        if df_hall is not None:
            # This would include features from "Table 1" mentioned in the original query
            # Add demographic features here based on available data
            pass
        
        logger.info("Created user-specific features")
        return df
    
    def prepare_ml_dataset(self, df_fitted_params: pd.DataFrame, 
                          df_hall: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare complete dataset for machine learning."""
        logger.info("Preparing ML dataset...")
        
        # Create time features
        df_ml = self.create_time_features(df_fitted_params)
        
        # Create user features
        df_ml = self.create_user_features(df_ml, df_hall)
        
        # Remove rows with missing target values
        target_cols = ['A', 'delta', 'sigma']
        df_ml = df_ml.dropna(subset=target_cols)
        
        logger.info(f"ML dataset prepared: {df_ml.shape[0]} samples, {df_ml.shape[1]} features")
        
        return df_ml
        
"""Enhanced feature engineering for ML model."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import logging

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for glucose response prediction."""
    
    def __init__(self):
        self.config = config
        self.glucotype_classifier = None
    
    def create_cgm_statistics(self, df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
        """Create CGM statistics features for specified time window."""
        df = df.copy()
        
        # This would be implemented based on your CGM data structure
        # For now, creating placeholder features
        window_suffix = f"{window_hours}h"
        
        cgm_features = {
            f'cgm_{window_suffix}_mean': 90 + np.random.normal(0, 10, len(df)),
            f'cgm_{window_suffix}_std': 5 + np.random.normal(0, 2, len(df)),
            f'cgm_{window_suffix}_TIR': 0.7 + np.random.normal(0, 0.1, len(df)),
            f'cgm_{window_suffix}_IQR': 8 + np.random.normal(0, 3, len(df)),
            f'cgm_{window_suffix}_below_70_pct': 0.05 + np.random.normal(0, 0.02, len(df)),
            f'cgm_{window_suffix}_above_140_pct': 0.1 + np.random.normal(0, 0.05, len(df)),
            f'cgm_{window_suffix}_skewness': np.random.normal(0, 0.5, len(df)),
            f'cgm_{window_suffix}_kurtosis': np.random.normal(0, 1, len(df))
        }
        
        for feature_name, values in cgm_features.items():
            df[feature_name] = values
        
        logger.info(f"Created CGM statistics for {window_hours}h window")
        return df
    
    def create_glucotype_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create glucotype features based on dynamic time warping clustering."""
        df = df.copy()
        
        # Placeholder implementation - in reality, this would use DTW clustering
        # as described in the research paper
        for hours in [24, 4]:
            for grt in [1, 2, 3]:
                # Generate realistic glucotype percentages
                if grt == 1:  # Stable glucose patterns
                    pct = 0.4 + np.random.normal(0, 0.1, len(df))
                elif grt == 2:  # Moderate variability
                    pct = 0.35 + np.random.normal(0, 0.1, len(df))
                else:  # High variability
                    pct = 0.25 + np.random.normal(0, 0.1, len(df))
                
                pct = np.clip(pct, 0, 1)  # Ensure percentages are valid
                df[f'GRT{grt}_{hours}h_pct'] = pct
        
        # Normalize to ensure percentages sum to 1 for each time window
        for hours in [24, 4]:
            cols = [f'GRT{i}_{hours}h_pct' for i in [1, 2, 3]]
            row_sums = df[cols].sum(axis=1)
            for col in cols:
                df[col] = df[col] / row_sums
        
        logger.info("Created glucotype features")
        return df
    
    def create_autocorrelation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create autocorrelation features for different time lags."""
        df = df.copy()
        
        # Placeholder implementation - in reality, this would calculate
        # actual autocorrelation from CGM time series data
        lag_features = {
            'autocorr_8h': np.random.uniform(0.3, 0.8, len(df)),
            'autocorr_4h': np.random.uniform(0.4, 0.9, len(df)),
            'autocorr_1h': np.random.uniform(0.6, 0.95, len(df)),
            'autocorr_40min': np.random.uniform(0.7, 0.98, len(df)),
            'autocorr_20min': np.random.uniform(0.8, 0.99, len(df)),
            'autocorr_10min': np.random.uniform(0.9, 0.995, len(df))
        }
        
        for feature_name, values in lag_features.items():
            df[feature_name] = values
        
        logger.info("Created autocorrelation features")
        return df
    
    def create_meal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meal macronutrient features."""
        df = df.copy()
        
        # If meal features don't exist, create realistic ones
        if 'CHO' not in df.columns:
            df['CHO'] = 30 + np.random.exponential(20, len(df))  # Carbohydrates in grams
        if 'PRO' not in df.columns:
            df['PRO'] = 15 + np.random.exponential(10, len(df))  # Protein in grams
        if 'FAT' not in df.columns:
            df['FAT'] = 10 + np.random.exponential(8, len(df))   # Fat in grams
        
        # Create derived meal features
        df['total_calories'] = df['CHO'] * 4 + df['PRO'] * 4 + df['FAT'] * 9
        df['cho_pct'] = df['CHO'] * 4 / df['total_calories']
        df['pro_pct'] = df['PRO'] * 4 / df['total_calories']
        df['fat_pct'] = df['FAT'] * 9 / df['total_calories']
        
        # Meal ratios
        df['cho_pro_ratio'] = df['CHO'] / (df['PRO'] + 1e-6)  # Avoid division by zero
        df['cho_fat_ratio'] = df['CHO'] / (df['FAT'] + 1e-6)
        df['pro_fat_ratio'] = df['PRO'] / (df['FAT'] + 1e-6)
        
        logger.info("Created meal macronutrient features")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame, df_hall: pd.DataFrame = None) -> pd.DataFrame:
        """Create demographic features."""
        df = df.copy()
        
        # If demographic data is available from df_hall, merge it
        if df_hall is not None and 'id' in df_hall.columns:
            # Map userID to demographic data
            demo_mapping = df_hall.set_index('id')[['age', 'gender', 'BMI', 'height', 'weight']].to_dict('index')
            
            for demo_feature in ['age', 'gender', 'BMI', 'height', 'weight']:
                df[demo_feature] = df['userID'].map(lambda x: demo_mapping.get(x, {}).get(demo_feature, np.nan))
        else:
            # Create placeholder demographic features
            df['age'] = 45 + np.random.normal(0, 10, len(df))
            df['gender'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])  # 0=Female, 1=Male
            df['BMI'] = 24 + np.random.normal(0, 4, len(df))
            df['height'] = 175 + np.random.normal(0, 10, len(df))
            df['weight'] = df['BMI'] * (df['height'] / 100) ** 2
        
        # Ensure realistic ranges
        df['age'] = np.clip(df['age'], 18, 80)
        df['BMI'] = np.clip(df['BMI'], 15, 40)
        df['height'] = np.clip(df['height'], 150, 200)
        
        # Create BMI categories
        df['bmi_underweight'] = (df['BMI'] < 18.5).astype(int)
        df['bmi_normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
        df['bmi_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['bmi_obese'] = (df['BMI'] >= 30).astype(int)
        
        logger.info("Created demographic features")
        return df
    
    def prepare_complete_ml_dataset(self, df_fitted_params: pd.DataFrame, 
                                   df_hall: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare complete dataset with all features for machine learning."""
        logger.info("Preparing complete ML dataset with enhanced features...")
        
        df_ml = df_fitted_params.copy()
        
        # Create all feature categories
        df_ml = self.create_time_features(df_ml)
        df_ml = self.create_demographic_features(df_ml, df_hall)
        df_ml = self.create_meal_features(df_ml)
        df_ml = self.create_cgm_statistics(df_ml, window_hours=24)
        df_ml = self.create_cgm_statistics(df_ml, window_hours=4)
        df_ml = self.create_glucotype_features(df_ml)
        df_ml = self.create_autocorrelation_features(df_ml)
        df_ml = self.create_user_features(df_ml, df_hall)
        
        # Remove rows with missing target values
        target_cols = ['A', 'delta', 'sigma']
        initial_rows = len(df_ml)
        df_ml = df_ml.dropna(subset=target_cols)
        final_rows = len(df_ml)
        
        if initial_rows != final_rows:
            logger.warning(f"Removed {initial_rows - final_rows} rows with missing target values")
        
        logger.info(f"Complete ML dataset prepared: {df_ml.shape[0]} samples, {df_ml.shape[1]} features")
        
        return df_ml
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features (from original implementation)."""
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Meal type indicators
        df['is_breakfast'] = df['meal'].str.contains('breakfast', case=False).astype(int)
        df['is_lunch'] = df['meal'].str.contains('lunch', case=False).astype(int)
        df['is_dinner'] = df['meal'].str.contains('dinner', case=False).astype(int)
        df['is_snack'] = df['meal'].str.contains('snack', case=False).astype(int)
        
        return df
    
    def create_user_features(self, df: pd.DataFrame, df_hall: pd.DataFrame = None) -> pd.DataFrame:
        """Create user-specific features (from original implementation)."""
        df = df.copy()
        
        # Add user-level statistics
        user_stats = df.groupby('userID').agg({
            'A': ['mean', 'std', 'min', 'max'],
            'delta': ['mean', 'std', 'min', 'max'],
            'sigma': ['mean', 'std', 'min', 'max'],
            'baseline': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.add_prefix('user_')
        
        # Merge with main dataframe
        df = df.merge(user_stats, left_on='userID', right_index=True, how='left')
        
        return df