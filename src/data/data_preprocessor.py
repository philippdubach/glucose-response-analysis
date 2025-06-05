"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HallDataPreprocessor:
    """Preprocessor for Hall glucose response dataset."""

    def __init__(self):
        self.config = config

    def filter_subjects(
        self, df_hall: pd.DataFrame, df_meals: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter out diabetic/prediabetic subjects and incomplete datasets."""
        logger.info("Filtering subjects...")

        initial_count = len(df_meals)

        # Remove diabetic/prediabetic subjects
        if self.config.get("data.exclude_diabetic", True):
            diabetic_prediabetic = df_hall[df_hall["type"] != "non-diabetic"]
            diabetic_ids = diabetic_prediabetic["id"].unique()
            df_meals = df_meals[~df_meals["userID"].isin(diabetic_ids)]
            logger.info(f"Removed {len(diabetic_ids)} diabetic/prediabetic subjects")

        # Remove incomplete datasets
        incomplete_ids = self.config.get("data.incomplete_datasets", [])
        df_meals = df_meals[~df_meals["userID"].isin(incomplete_ids)]
        logger.info(f"Removed {len(incomplete_ids)} incomplete datasets")

        final_count = len(df_meals)
        logger.info(f"Filtered data: {initial_count} -> {final_count} observations")

        return df_meals

    def mark_meal_times(self, df_meals: pd.DataFrame) -> pd.DataFrame:
        """Mark meal times in the dataset."""
        logger.info("Marking meal times...")

        df_meals = df_meals.copy()
        df_meals["meal_taken"] = 0

        meal_position = self.config.get("analysis.meal_time_position", 6)

        def mark_meal_time(group):
            if len(group) >= meal_position:
                group.iloc[meal_position - 1, group.columns.get_loc("meal_taken")] = 1
            return group

        grouped = df_meals.groupby(["Meal", "userID"])
        df_meals = grouped.apply(mark_meal_time).reset_index(drop=True)

        meal_count = df_meals["meal_taken"].sum()
        logger.info(f"Marked {meal_count} meal times")

        return df_meals

    def preprocess_all(
        self, df_hall: pd.DataFrame, df_meals: pd.DataFrame
    ) -> pd.DataFrame:
        """Run all preprocessing steps."""
        logger.info("Starting data preprocessing...")

        df_meals_filtered = self.filter_subjects(df_hall, df_meals)
        df_meals_final = self.mark_meal_times(df_meals_filtered)

        logger.info("Data preprocessing completed")
        return df_meals_final
