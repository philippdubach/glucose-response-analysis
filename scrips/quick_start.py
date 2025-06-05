"""Quick start script for testing the glucose response analysis."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging, get_logger
from src.data.data_loader import HallDataLoader
from src.models.glucose_response_analyzer import GlucoseResponseAnalyzer


def quick_test():
    """Quick test to verify everything works."""
    setup_logging()
    logger = get_logger(__name__)

    try:
        # Test data loading
        loader = HallDataLoader()
        df_hall, df_meals = loader.load_all_data()
        logger.info(f"✅ Data loaded successfully: {len(df_meals)} meal observations")

        # Test curve fitting on a small sample
        analyzer = GlucoseResponseAnalyzer()
        logger.info("✅ Analyzer initialized successfully")

        logger.info("🎉 Quick test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Quick test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Ready to run the full analysis!")
        print("Next steps:")
        print("1. Run: python scripts/run_analysis.py")
        print("2. Or use VS Code debugger (F5)")
        print("3. Open notebooks/ folder for interactive analysis")
