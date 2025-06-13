# Glucose Response Analysis and Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning pipeline for analyzing glucose response patterns from continuous glucose monitoring (CGM) data and predicting glycemic response parameters using XGBoost regression models.

## 📄 Working Paper
Read the [Working Paper](https://github.com/philippdubach/glucose-response-analysis/blob/1d032b9d7a30df893b4b4e1024b44fdf9451f9d0/Modeling_Postprandial_Glycemic_Response_in_Non_Diabetic_Adults_Using_XGBRegressor.pdf) 

![Working Paper Summary](/paper.png)

## 🎯 Overview

This project implements a comprehensive analysis pipeline for glucose response data, featuring:

- **Gaussian Curve Fitting**: Automated fitting of normalized Gaussian curves to meal-induced glucose responses
- **Machine Learning Prediction**: XGBoost models to predict glucose response parameters (amplitude, time-to-peak, curve width)
- **Feature Engineering**: Comprehensive feature extraction including demographics, meal composition, CGM statistics, and temporal patterns
- **Model Interpretation**: SHAP values for feature importance and Bland-Altman analysis for model evaluation
- **Reproducible Pipeline**: Well-structured codebase following data science best practices

## 📊 Key Features

### Glucose Response Analysis
- Automated detection and fitting of meal-induced glucose responses
- Normalized Gaussian curve fitting with parameters:
  - **A**: Amplitude (peak glucose increase above baseline)
  - **δ**: Time to peak (minutes)
  - **σ**: Curve width (standard deviation)
  - **b**: Baseline glucose level

### Machine Learning Models
- **XGBoost Regressors**: Separate models for predicting A, δ, and σ parameters
- **Multi-linear Regression**: Statistical model for amplitude prediction using macronutrients
- **Hyperparameter Optimization**: Grid search for optimal model parameters
- **Cross-validation**: Robust model evaluation with multiple metrics

### Advanced Analytics
- **SHAP Analysis**: Feature importance and model interpretability
- **Bland-Altman Plots**: Systematic bias detection and agreement analysis
- **Glucotype Classification**: Dynamic time warping-based glucose pattern clustering
- **Autocorrelation Features**: Temporal dependencies in glucose time series

## 🏗️ Project Structure

```
glucose-response-analysis/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── config.yaml              # Main configuration file
│   └── logging.yaml             # Logging configuration
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed datasets
│   └── external/                # External reference data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_glucose_curve_fitting.ipynb
│   └── 03_model_development.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Data loading utilities
│   │   └── data_preprocessor.py # Data preprocessing
│   ├── features/
│   │   └── feature_engineering.py # Feature engineering
│   ├── models/
│   │   ├── glucose_response_analyzer.py # Curve fitting
│   │   └── xgboost_regressor.py # ML models
│   ├── visualization/
│   │   └── plotting.py          # Visualization utilities
│   └── utils/
│       ├── config.py            # Configuration management
│       └── logging.py           # Logging utilities
├── tests/
│   ├── test_data_loader.py
│   └── test_glucose_analyzer.py
├── scripts/
│   ├── run_analysis.py          # Main analysis pipeline
│   └── train_model.py          # Model training script
├── results/
│   ├── figures/                 # Generated plots
│   ├── models/                  # Trained models
│   └── reports/                 # Analysis reports
└── docs/
    └── methodology.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/philippdubach/glucose-response-analysis.git
cd glucose-response-analysis
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv glucose_env

# Activate virtual environment
# On Windows:
glucose_env\Scripts\activate
# On macOS/Linux:
source glucose_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare your data**
   - Place your Hall 2018 dataset in `data/raw/hall2018.csv`
   - Place your meals data in `data/raw/hall_meals.csv`
   - Or use the provided sample data for testing

### Basic Usage

**Run the complete analysis pipeline:**
```bash
python scripts/run_analysis.py
```

**Train machine learning models with options:**
```bash
# Basic training
python scripts/train_model.py

# With hyperparameter tuning (slower but better results)
python scripts/train_model.py --hyperparameter-tuning

# Quick test with limited data
python scripts/train_model.py --sample-size 100 --skip-visualization
```

**Run tests:**
```bash
pytest tests/ -v
```

**Explore with Jupyter notebooks:**
```bash
jupyter notebook notebooks/
```

## 🔧 Configuration

Modify `config/config.yaml` to customize:

```yaml
# Model parameters
model:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.01
  
# Analysis parameters  
analysis:
  observation_window_minutes: 150
  sampling_interval_minutes: 5
  
# Visualization settings
visualization:
  save_plots: true
  figure_size: [10, 6]
```

## 📊 Data Format

### Input Data Requirements

**Hall Dataset (`hall2018.csv`)**:
```csv
id,type,age,gender,BMI,height,weight
user_1,non-diabetic,45,M,24.5,175,75
```

**Meals Dataset (`hall_meals.csv`)**:
```csv
userID,Meal,time,GlucoseValue
user_1,breakfast,01/01/2023 08:00,85
user_1,breakfast,01/01/2023 08:05,87
```

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Demographics** | Age, Gender, BMI, Height, Weight | User characteristics |
| **Meal Composition** | CHO, PRO, FAT | Macronutrient content (grams) |
| **CGM Statistics** | Mean, SD, TIR, IQR | 24h and 4h glucose statistics |
| **Temporal** | Hour, Day of week, Meal type | Time-based features |
| **Glucotypes** | GRT1, GRT2, GRT3 percentages | Glucose pattern clusters |
| **Autocorrelation** | 8h, 4h, 1h, 40min, 20min, 10min | Temporal dependencies |

## 🧪 Methodology

### Curve Fitting Process

1. **Meal Detection**: Identify meal events in CGM data
2. **Data Extraction**: Extract 2.5-hour glucose windows post-meal
3. **Baseline Calculation**: Average glucose in 10 minutes pre-meal
4. **Gaussian Fitting**: Fit normalized Gaussian curve using least squares optimization

$$G(t) = A \cdot \frac{e^{-\frac{(t-\delta)^2}{2\sigma^2}}}{\max(e^{-\frac{(t-\delta)^2}{2\sigma^2}})} + b$$

Where:
- **A**: Amplitude (mg/dL)
- **δ**: Time to peak (minutes)  
- **σ**: Curve width (minutes)
- **b**: Baseline glucose (mg/dL)

### Machine Learning Pipeline

1. **Feature Engineering**: Create comprehensive feature set
2. **Model Training**: Train separate XGBoost models for A, δ, σ
3. **Hyperparameter Tuning**: Grid search optimization
4. **Model Evaluation**: Cross-validation with multiple metrics
5. **Interpretation**: SHAP analysis and Bland-Altman plots

## 📋 Requirements

See `requirements.txt` for complete list. Key dependencies:

- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.1.0` - Machine learning
- `xgboost>=1.6.0` - Gradient boosting
- `shap>=0.41.0` - Model interpretability
- `matplotlib>=3.5.0` - Plotting
- `scipy>=1.9.0` - Scientific computing

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_glucose_analyzer.py -v
```

## 📝 API Documentation

### Core Classes

**`GlucoseResponseAnalyzer`**
```python
from src.models.glucose_response_analyzer import GlucoseResponseAnalyzer

analyzer = GlucoseResponseAnalyzer()
fitted_params = analyzer.fit_all_meals(df_meals)
```

**`GlucoseXGBoostRegressor`**
```python
from src.models.xgboost_regressor import GlucoseXGBoostRegressor

regressor = GlucoseXGBoostRegressor()
model_results = regressor.train_all_targets(df_ml)
predictions = regressor.predict(X_new)
```

**`EnhancedFeatureEngineer`**
```python
from src.features.feature_engineering import EnhancedFeatureEngineer

engineer = EnhancedFeatureEngineer()
df_ml = engineer.prepare_complete_ml_dataset(df_fitted_params)
```

## 🎨 Visualization Examples

The pipeline generates comprehensive visualizations:

- **Glucose Curve Fits**: Individual meal response curves with fitted parameters
- **Parameter Distributions**: Histograms of A, δ, σ across population
- **Model Performance**: Prediction vs truth plots, residual analysis
- **Feature Importance**: SHAP values and feature rankings
- **Bland-Altman Plots**: Agreement analysis and systematic bias detection

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest-cov

# Format code
black src/ tests/ scripts/

# Check linting
flake8 src/ tests/ scripts/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 References

1. Hall, H., et al. (2018). "Glucotypes reveal new patterns of glucose dysregulation." *PLOS Biology*, 16(7), e2005143.

2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *Advances in Neural Information Processing Systems*, 30.

3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🚨 Disclaimer

This project is for research purposes only. The models and analysis are not intended for clinical decision-making or medical diagnosis. Always consult healthcare professionals for medical advice.

## 📊 Citing This Work

If you use this code in your research, please cite:

```bibtex
@software{glucose_response_analysis,
  author = {Philipp Dubach},
  title = {Glucose Response Analysis and Prediction Pipeline},
  url = {https://github.com/philippdubach/glucose-response-analysis},
  version = {1.0.0},
  year = {2024}
}
```

## 🎯 Roadmap

- [ ] Add real-time glucose monitoring integration
- [ ] Implement additional curve fitting models (bi-exponential, gamma distribution)
- [ ] Add deep learning models (LSTM, Transformer)
- [ ] Create web interface for interactive analysis
- [ ] Add support for additional CGM devices
- [ ] Implement population-level analysis tools
