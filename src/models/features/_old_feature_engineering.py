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