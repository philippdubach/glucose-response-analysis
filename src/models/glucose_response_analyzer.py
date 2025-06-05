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
        self.observation_window = self.config.get(
            "analysis.observation_window_minutes", 150
        )
        self.sampling_interval = self.config.get(
            "analysis.sampling_interval_minutes", 5
        )
        self.baseline_window = self.config.get("analysis.baseline_window", 2)

    @staticmethod
    def normalized_gaussian(
        t: np.ndarray, A: float, delta: float, sigma: float, b: float
    ) -> np.ndarray:
        """
        Normalized Gaussian function representing glycemic response.

        Parameters:
        - t: time array
        - A: amplitude (peak glucose increase above baseline)
        - delta: time to peak (minutes)
        - sigma: standard deviation (curve width)
        - b: baseline glucose level
        """
        gauss = np.exp(-((t - delta) ** 2) / (2 * sigma**2))
        return A * (gauss / np.max(gauss)) + b

    def residuals(
        self, params: List[float], t: np.ndarray, y: np.ndarray, b: float
    ) -> np.ndarray:
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
        - Dictionary containing fitted parameters, metadata, and nutritional information
        """
        try:
            # Extract glucose data for observation window
            n_observations = self.observation_window // self.sampling_interval
            end_index = min(meal_index + n_observations, len(df_meals))
            glucose_data = df_meals.loc[meal_index : end_index - 1, "GlucoseValue"]

            # Create time array
            t = np.arange(
                0, len(glucose_data) * self.sampling_interval, self.sampling_interval
            )

            # Calculate baseline
            baseline_start = max(0, meal_index - self.baseline_window)
            baseline_end = meal_index
            b = df_meals.loc[baseline_start : baseline_end - 1, "GlucoseValue"].mean()

            # Initial parameter guesses
            A_guess = max(0, glucose_data.max() - b)
            peak_idx = glucose_data.idxmax() - meal_index
            delta_guess = t[min(peak_idx, len(t) - 1)] if peak_idx < len(t) else t[-1]
            sigma_guess = self.config.get("analysis.initial_sigma_guess", 20)

            # Optimization bounds
            min_amplitude = self.config.get("analysis.min_amplitude", 0)
            max_delta = self.config.get("analysis.max_delta_minutes", 150)
            max_sigma = self.config.get("analysis.max_sigma_minutes", 60)

            lower_bounds = [min_amplitude, 0, 5]
            upper_bounds = [A_guess + 20, max_delta, max_sigma]

            # Perform optimization
            result = least_squares(
                self.residuals,
                [A_guess, delta_guess, sigma_guess],
                args=(t, glucose_data.values, b),
                bounds=(lower_bounds, upper_bounds),
                loss="soft_l1",
            )

            A_opt, delta_opt, sigma_opt = result.x
            y_fit = self.normalized_gaussian(t, A_opt, delta_opt, sigma_opt, b)

            # Get meal metadata
            meal_info = df_meals.loc[meal_index]

            # Prepare result dictionary with curve fitting parameters
            result_dict = {
                "success": True,
                "A": A_opt,
                "delta": delta_opt,
                "sigma": sigma_opt,
                "baseline": b,
                "userID": meal_info["userID"],
                "meal": meal_info["Meal"],
                "time": meal_info["time"],
                "t": t,
                "y_true": glucose_data.values,
                "y_fit": y_fit,
                "r_squared": self._calculate_r_squared(glucose_data.values, y_fit),
                "meal_index": meal_index,
            }

            # Add nutritional information if available
            nutritional_cols = ["CHO", "PRO", "FAT"]
            for col in nutritional_cols:
                if col in meal_info.index:
                    result_dict[col] = meal_info[col]
                else:
                    logger.warning(f"Nutritional column '{col}' not found in meal data")

            # Add any other relevant meal metadata (excluding time-series data)
            exclude_cols = ["GlucoseValue", "meal_taken"]
            for col in meal_info.index:
                if col not in result_dict and col not in exclude_cols:
                    result_dict[col] = meal_info[col]

            return result_dict

        except Exception as e:
            logger.error(f"Failed to fit meal at index {meal_index}: {str(e)}")

            # Try to get basic meal info even if fitting fails
            try:
                meal_info = df_meals.loc[meal_index]
                return {
                    "success": False,
                    "error": str(e),
                    "meal_index": meal_index,
                    "userID": meal_info.get("userID", None),
                    "meal": meal_info.get("Meal", None),
                    "CHO": meal_info.get("CHO", None),
                    "PRO": meal_info.get("PRO", None),
                    "FAT": meal_info.get("FAT", None),
                }
            except:
                return {"success": False, "error": str(e), "meal_index": meal_index}

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
        - DataFrame with fitted parameters and nutritional information
        """
        meal_indices = df_meals[df_meals["meal_taken"] == 1].index.tolist()
        total_meals = len(meal_indices)

        logger.info(f"Fitting {total_meals} meals...")
        logger.info(f"Available columns in input data: {list(df_meals.columns)}")

        # Check if nutritional columns are available
        nutritional_cols = ["CHO", "PRO", "FAT"]
        available_nutritional = [
            col for col in nutritional_cols if col in df_meals.columns
        ]
        missing_nutritional = [
            col for col in nutritional_cols if col not in df_meals.columns
        ]

        if missing_nutritional:
            logger.warning(f"Missing nutritional columns: {missing_nutritional}")
        if available_nutritional:
            logger.info(f"Available nutritional columns: {available_nutritional}")

        results = []
        successful_fits = 0
        failed_fits = 0

        for i, meal_idx in enumerate(meal_indices):
            result = self.fit_single_meal(df_meals, meal_idx)

            if result["success"]:
                # Create clean result dictionary for DataFrame
                clean_result = {
                    # Fitted parameters
                    "A": round(result["A"], 3),
                    "delta": round(result["delta"], 3),
                    "sigma": round(result["sigma"], 3),
                    "baseline": round(result["baseline"], 3),
                    "r_squared": round(result["r_squared"], 4),
                    # Meal identification
                    "userID": result["userID"],
                    "meal": result["meal"],
                    "time": result["time"],
                }

                # Add nutritional information
                for col in nutritional_cols:
                    if col in result:
                        clean_result[col] = result[col]
                    else:
                        clean_result[col] = np.nan
                        logger.warning(
                            f"Missing {col} for meal {result.get('meal', 'unknown')} of user {result.get('userID', 'unknown')}"
                        )

                # Add any other metadata (excluding arrays and complex objects)
                for key, value in result.items():
                    if (
                        key not in clean_result
                        and key not in ["success", "t", "y_true", "y_fit", "meal_index"]
                        and not isinstance(value, (np.ndarray, list))
                    ):
                        clean_result[key] = value

                results.append(clean_result)
                successful_fits += 1
            else:
                failed_fits += 1
                logger.debug(
                    f"Failed to fit meal at index {meal_idx}: {result.get('error', 'Unknown error')}"
                )

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Processed {i + 1}/{total_meals} meals... (Success: {successful_fits}, Failed: {failed_fits})"
                )

        logger.info(
            f"Curve fitting completed: {successful_fits} successful, {failed_fits} failed out of {total_meals} total meals"
        )

        if successful_fits == 0:
            logger.error("No meals were successfully fitted!")
            return pd.DataFrame()

        df_results = pd.DataFrame(results)

        # Log summary statistics
        logger.info(f"Final dataset shape: {df_results.shape}")
        logger.info(f"Final columns: {list(df_results.columns)}")

        # Verify and report on nutritional data
        for col in nutritional_cols:
            if col in df_results.columns:
                non_null_count = df_results[col].notna().sum()
                if non_null_count > 0:
                    min_val = df_results[col].min()
                    max_val = df_results[col].max()
                    mean_val = df_results[col].mean()
                    logger.info(
                        f"{col}: {non_null_count}/{len(df_results)} meals, range {min_val:.1f}-{max_val:.1f}g (mean: {mean_val:.1f}g)"
                    )
                else:
                    logger.warning(f"{col}: No valid data found!")
            else:
                logger.error(f"{col}: Column missing from results!")

        # Verify curve fitting quality
        if "r_squared" in df_results.columns:
            mean_r2 = df_results["r_squared"].mean()
            good_fits = (df_results["r_squared"] > 0.5).sum()
            logger.info(
                f"Curve fitting quality: mean R² = {mean_r2:.3f}, {good_fits}/{len(df_results)} fits with R² > 0.5"
            )

        return df_results

    def get_meal_summary(self, df_results: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for fitted meals.

        Parameters:
        - df_results: DataFrame returned by fit_all_meals()

        Returns:
        - Dictionary with summary statistics
        """
        if df_results.empty:
            return {"error": "No data to summarize"}

        summary = {
            "total_meals": len(df_results),
            "unique_users": df_results["userID"].nunique()
            if "userID" in df_results.columns
            else 0,
            "curve_parameters": {
                "amplitude_mean": df_results["A"].mean()
                if "A" in df_results.columns
                else None,
                "amplitude_std": df_results["A"].std()
                if "A" in df_results.columns
                else None,
                "time_to_peak_mean": df_results["delta"].mean()
                if "delta" in df_results.columns
                else None,
                "time_to_peak_std": df_results["delta"].std()
                if "delta" in df_results.columns
                else None,
                "curve_width_mean": df_results["sigma"].mean()
                if "sigma" in df_results.columns
                else None,
                "curve_width_std": df_results["sigma"].std()
                if "sigma" in df_results.columns
                else None,
            },
            "fitting_quality": {
                "mean_r_squared": df_results["r_squared"].mean()
                if "r_squared" in df_results.columns
                else None,
                "good_fits_count": (df_results["r_squared"] > 0.5).sum()
                if "r_squared" in df_results.columns
                else None,
                "excellent_fits_count": (df_results["r_squared"] > 0.8).sum()
                if "r_squared" in df_results.columns
                else None,
            },
        }

        # Add nutritional summary if available
        nutritional_cols = ["CHO", "PRO", "FAT"]
        summary["nutritional_data"] = {}
        for col in nutritional_cols:
            if col in df_results.columns:
                summary["nutritional_data"][col] = {
                    "mean": df_results[col].mean(),
                    "std": df_results[col].std(),
                    "min": df_results[col].min(),
                    "max": df_results[col].max(),
                    "count": df_results[col].notna().sum(),
                }

        return summary
