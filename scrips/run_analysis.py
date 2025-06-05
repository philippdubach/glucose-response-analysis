"""Main script to run the complete glucose response analysis."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.logging import setup_logging, get_logger
from src.utils.config import config
from src.data.data_loader import HallDataLoader
from src.data.data_preprocessor import HallDataPreprocessor
from src.models.glucose_response_analyzer import GlucoseResponseAnalyzer
from src.features.feature_engineering import FeatureEngineer
from src.visualization.plotting import GlucosePlotter


def main():
    """Run complete analysis pipeline."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("Starting glucose response analysis...")

    try:
        # Load data
        loader = HallDataLoader()
        df_hall, df_meals = loader.load_all_data()

        # Preprocess data
        preprocessor = HallDataPreprocessor()
        df_meals_processed = preprocessor.preprocess_all(df_hall, df_meals)

        # Fit glucose response curves
        analyzer = GlucoseResponseAnalyzer()
        df_fitted_params = analyzer.fit_all_meals(df_meals_processed)

        # Feature engineering
        feature_engineer = FeatureEngineer()
        df_ml = feature_engineer.prepare_ml_dataset(df_fitted_params, df_hall)

        # Save results
        results_path = config.get_results_path()
        results_path.mkdir(parents=True, exist_ok=True)

        df_fitted_params.to_csv(results_path / "fitted_parameters.csv", index=False)
        df_ml.to_csv(results_path / "ml_dataset.csv", index=False)

        logger.info(f"Analysis completed. Results saved to {results_path}")
        logger.info(f"Fitted parameters: {df_fitted_params.shape}")
        logger.info(f"ML dataset: {df_ml.shape}")

        return df_fitted_params, df_ml

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    fitted_params, ml_dataset = main()
