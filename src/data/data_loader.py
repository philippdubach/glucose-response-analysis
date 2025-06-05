"""Data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class HallDataLoader:
    """Data loader for Hall 2018 glucose response dataset."""
    
    def __init__(self):
        self.config = config
        self.hall_data_path = self.config.get_data_path(self.config.get('data.hall_data_file'))
        self.meals_data_path = self.config.get_data_path(self.config.get('data.meals_data_file'))
    
    def load_hall_data(self) -> pd.DataFrame:
        """Load and preprocess Hall dataset."""
        logger.info(f"Loading Hall data from {self.hall_data_path}")
        
        df_hall_raw = pd.read_csv(self.hall_data_path)
        
        # Remove unnecessary columns
        columns_to_drop = self.config.get('data.columns_to_drop', [])
        df_hall = df_hall_raw.drop(columns=columns_to_drop, errors='ignore')
        
        logger.info(f"Loaded Hall data: {df_hall.shape[0]} records, {df_hall.shape[1]} columns")
        return df_hall
    
    def load_meals_data(self) -> pd.DataFrame:
        """Load and preprocess meals data."""
        logger.info(f"Loading meals data from {self.meals_data_path}")
        
        df_meals_raw = pd.read_csv(self.meals_data_path)
        df_meals_raw = df_meals_raw.sort_values(['Meal', 'userID', 'time'])
        
        # Group by user, meal, and time, then average glucose values
        df_meals = df_meals_raw.groupby(['userID', 'Meal', 'time']).mean().reset_index()
        
        # Convert time to datetime
        df_meals['time'] = pd.to_datetime(df_meals['time'], format='%d/%m/%Y %H:%M')
        
        logger.info(f"Loaded meals data: {df_meals.shape[0]} observations")
        return df_meals
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both Hall and meals datasets."""
        hall_data = self.load_hall_data()
        meals_data = self.load_meals_data()
        return hall_data, meals_data