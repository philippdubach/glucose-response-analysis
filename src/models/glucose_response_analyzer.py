"""Glucose response analysis and curve fitting."""

import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
import logging

from ..utils.config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)

class GlucoseResponseAnalyzer:
    """Analyzer for fitting Gaussian curves to glucose response data."""
    
    def __init__(self):
        self.config = config
        self.observation_window = self.config.get('analysis.observation_window_minutes', 150)
        self.sampling_interval = self.config.get('analysis.sampling_interval_minutes', 5)
        self.baseline_window = self.config.get('analysis.baseline_window', 2)
        
    @staticmethod
    def normalized_gaussian(t: np.ndarray, A: float, delta: float, sigma: float, b: float) -> np.ndarray:
        """
        Normalized Gaussian function representing glycemic response.
        
        Parameters:
        - t: time array
        - A: amplitude (peak glucose increase above baseline)
        - delta: time to peak (minutes)
        - sigma: standard deviation (curve width)
        - b: baseline glucose level
        """
        gauss = np.exp(-((t - delta) ** 2) / (2 * sigma ** 2))
        return A * (gauss / np.max(gauss)) + b
    
    def residuals(self, params: List[float], t: np.ndarray, y: np.ndarray, b: float) -> np.ndarray:
        """Calculate residuals for least squares optimization."""
        A, delta, sigma = params
        return self.normalized_gaussian(t, A, delta, sigma, b) - y
    
    def fit_single_meal(self, df_meals: pd.DataFrame, meal_index: int) -> Dict:
        """
        Fit Gaussian curve to a single meal response.
        
        Parameters:
        - df_meals: DataFrame with meal data
        - meal_index: Index of the meal start in the DataFrame
        
        Returns:
        - Dictionary containing fitted parameters and metadata
        """
        try:
            # Extract glucose data for observation window
            n_observations = self.observation_window // self.sampling_interval
            end_index = min(meal_index + n_observations, len(df_meals))
            glucose_data = df_meals.loc[meal_index:end_index-1, 'GlucoseValue']
            
            # Create time array
            t = np.arange(0, len(glucose_data) * self.sampling_interval, self.sampling_interval)
            
            # Calculate baseline
            baseline_start = max(0, meal_index - self.baseline_window)
            baseline_end = meal_index
            b = df_meals.loc[baseline_start:baseline_end-1, 'GlucoseValue'].mean()
            
            # Initial parameter guesses
            A_guess = max(0, glucose_data.max() - b)
            peak_idx = glucose_data.idxmax() - meal_index
            delta_guess = t[min(peak_idx, len(t)-1)] if peak_idx < len(t) else t[-1]
            sigma_guess = self.config.get('analysis.initial_sigma_guess', 20)
            
            # Optimization bounds
            min_amplitude = self.config.get('analysis.min_amplitude', 0)
            max_delta = self.config.get('analysis.max_delta_minutes', 150)
            max_sigma = self.config.get('analysis.max_sigma_minutes', 60)
            
            lower_bounds = [min_amplitude, 0, 5]
            upper_bounds = [A_guess + 20, max_delta, max_sigma]
            
            # Perform optimization
            result = least_squares(
                self.residuals,
                [A_guess, delta_guess, sigma_guess],
                args=(t, glucose_data.values, b),
                bounds=(lower_bounds, upper_bounds),
                loss='soft_l1'
            )
            
            A_opt, delta_opt, sigma_opt = result.x
            y_fit = self.normalized_gaussian(t, A_opt, delta_opt, sigma_opt, b)
            
            # Get meal metadata
            meal_info = df_meals.loc[meal_index]
            
            return {
                'success': True,
                'A': A_opt,
                'delta': delta_opt,
                'sigma': sigma_opt,
                'baseline': b,
                'userID': meal_info['userID'],
                'meal': meal_info['Meal'],
                'time': meal_info['time'],
                't': t,
                'y_true': glucose_data.values,
                'y_fit': y_fit,
                'r_squared': self._calculate_r_squared(glucose_data.values, y_fit),
                'meal_index': meal_index
            }
            
        except Exception as e:
            logger.error(f"Failed to fit meal at index {meal_index}: {str(e)}")
            return {'success': False, 'error': str(e), 'meal_index': meal_index}
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared for goodness of fit."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def fit_all_meals(self, df_meals: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Gaussian curves to all meals in the dataset.
        
        Parameters:
        - df_meals: DataFrame with preprocessed meal data
        
        Returns:
        - DataFrame with fitted parameters
        """
        meal_indices = df_meals[df_meals['meal_taken'] == 1].index.tolist()
        total_meals = len(meal_indices)
        
        logger.info(f"Fitting {total_meals} meals...")
        
        results = []
        successful_fits = 0
        
        for i, meal_idx in enumerate(meal_indices):
            result = self.fit_single_meal(df_meals, meal_idx)
            
            if result['success']:
                results.append({
                    'A': round(result['A'], 3),
                    'delta': round(result['delta'], 3),
                    'sigma': round(result['sigma'], 3),
                    'baseline': round(result['baseline'], 3),
                    'userID': result['userID'],
                    'meal': result['meal'],
                    'time': result['time'],
                    'r_squared': round(result['r_squared'], 4)
                })
                successful_fits += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{total_meals} meals...")
        
        logger.info(f"Successfully fitted {successful_fits}/{total_meals} meals")
        
        return pd.DataFrame(results)