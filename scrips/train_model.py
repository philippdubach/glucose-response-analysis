"""Enhanced script for complete model training pipeline."""

import sys
from pathlib import Path
import argparse
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.logging import setup_logging, get_logger
from src.utils.config import config
from src.data.data_loader import HallDataLoader
from src.data.data_preprocessor import HallDataPreprocessor
from src.models.glucose_response_analyzer import GlucoseResponseAnalyzer
from src.features.feature_engineering import EnhancedFeatureEngineer
from src.models.xgboost_regressor import GlucoseXGBoostRegressor, MultiLinearRegressor
from src.visualization.plotting import GlucosePlotter

def main():
    """Run complete model training pipeline."""
    parser = argparse.ArgumentParser(description='Train glucose response prediction models')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit dataset size for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting complete model training pipeline...")
    
    try:
        # 1. Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data...")
        loader = HallDataLoader()
        df_hall, df_meals = loader.load_all_data()
        
        preprocessor = HallDataPreprocessor()
        df_meals_processed = preprocessor.preprocess_all(df_hall, df_meals)
        
        # 2. Fit glucose response curves
        logger.info("Step 2: Fitting glucose response curves...")
        analyzer = GlucoseResponseAnalyzer()
        df_fitted_params = analyzer.fit_all_meals(df_meals_processed)
        
        # Apply sample size limit if specified
        if args.sample_size and len(df_fitted_params) > args.sample_size:
            df_fitted_params = df_fitted_params.sample(args.sample_size, random_state=42)
            logger.info(f"Limited dataset to {args.sample_size} samples for testing")
        
        # 3. Feature engineering
        logger.info("Step 3: Engineering features...")
        feature_engineer = EnhancedFeatureEngineer()
        df_ml = feature_engineer.prepare_complete_ml_dataset(df_fitted_params, df_hall)
        
        # 4. Train XGBoost models
        logger.info("Step 4: Training XGBoost models...")
        xgboost_regressor = GlucoseXGBoostRegressor()
        model_results = xgboost_regressor.train_all_targets(
            df_ml, hyperparameter_tuning=args.hyperparameter_tuning
        )
        
        # 5. Train multi-linear regression model
        logger.info("Step 5: Training multi-linear regression model...")
        mlr = MultiLinearRegressor()
        mlr_results = mlr.train(df_ml)
        
        # 6. Generate visualizations
        if not args.skip_visualization:
            logger.info("Step 6: Generating visualizations...")
            plotter = GlucosePlotter()
            plotter.create_comprehensive_report(df_fitted_params, model_results, df_ml, xgboost_regressor)
        
        # 7. Save results
        logger.info("Step 7: Saving results...")
        results_path = config.get_results_path()
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        df_fitted_params.to_csv(results_path / 'fitted_parameters.csv', index=False)
        df_ml.to_csv(results_path / 'ml_dataset.csv', index=False)
        
        # Save models
        xgboost_regressor.save_models(results_path / 'models')
        
        # Save results summary
        summary = {
            'dataset_info': {
                'total_meals': len(df_fitted_params),
                'total_features': df_ml.shape[1],
                'subjects': df_fitted_params['userID'].nunique()
            },
            'xgboost_results': {
                target: {
                    'test_r2': results['test_metrics']['r_squared'],
                    'test_rmse': results['test_metrics']['rmse'],
                    'test_mae': results['test_metrics']['mae'],
                    'correlation': results['test_metrics']['correlation'],
                    'p_value': results['test_metrics']['p_value']
                }
                for target, results in model_results.items()
            },
            'mlr_results': mlr_results
        }
        
        with open(results_path / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Dataset: {len(df_fitted_params)} meals from {df_fitted_params['userID'].nunique()} subjects")
        logger.info(f"Features: {df_ml.shape[1]} total features")
        
        logger.info("\nXGBoost Model Results:")
        for target_name, results in model_results.items():
            metrics = results['test_metrics']
            logger.info(f"  {target_name}: R²={metrics['r_squared']:.3f}, "
                       f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
        
        logger.info(f"\nMulti-linear Regression (Amplitude): R²={mlr_results['r2']:.3f}, "
                   f"RSE={mlr_results['rse']:.2f} mg/dL")
        
        logger.info(f"\nAll results saved to: {results_path}")
        
        return df_fitted_params, df_ml, model_results, mlr_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()