# Road Accident Risk Prediction Project

This project converts a Jupyter notebook into a modular Python project structure for predicting road accident risk from tabular data.

## Project Structure

```
├── config.py                 # Configuration file with paths and constants
├── data_loader.py            # Data loading and exploration module
├── data_loader_clean.py      # Clean version without Unicode characters
├── preprocessing.py          # Data preprocessing and feature engineering
├── eda.py                    # Exploratory Data Analysis module
├── model_training.py         # Model training and evaluation
├── submission.py             # Submission file generation and validation
├── main.py                   # Main orchestration file (original with Unicode)
├── main_final.py             # Final clean main file
├── test_pipeline.py          # Simple working test version
├── submission.csv            # Generated submission file
└── Dataset/                  # Dataset directory
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## Quick Start

### Option 1: Simple Test Pipeline (Recommended)

```bash
python test_pipeline.py
```

This runs a simplified version that works immediately and generates `submission.csv`.

### Option 2: Full Pipeline (Requires Unicode fixes)

```bash
python main_final.py --skip-eda
```

This runs the full modular pipeline without EDA to avoid Unicode issues.

## Module Overview

### config.py

Contains all configuration settings:

- File paths for datasets
- Model parameters
- Scaling methods
- Output settings

### data_loader.py / data_loader_clean.py

Handles data loading and basic exploration:

- Load train, test, and submission datasets
- Basic data type analysis
- Missing value detection
- Feature set comparison

### preprocessing.py

Data preprocessing and feature engineering:

- Categorical feature encoding
- Feature scaling (Standard, Min-Max, Robust, Power)
- Train/validation split
- Feature analysis and scaling needs assessment

### eda.py

Exploratory Data Analysis:

- Univariate analysis with visualizations
- Correlation analysis and heatmaps
- Target distribution analysis
- Interactive plots using Plotly

### model_training.py

Model training and evaluation:

- Multiple algorithm comparison (9 different models)
- Cross-validation
- Model selection based on MSE/MAE/R²
- Feature importance analysis
- Comprehensive evaluation metrics

### submission.py

Final submission generation:

- Prediction generation
- Submission file validation
- Prediction distribution analysis
- Visualization of results

### main.py / main_final.py

Main orchestration files:

- Complete pipeline execution
- Command-line argument parsing
- Progress tracking and logging
- Error handling and debugging

## Usage Examples

### Run with default settings

```bash
python main_final.py
```

### Skip EDA for faster execution

```bash
python main_final.py --skip-eda
```

### Use different scaling method

```bash
python main_final.py --scaling-method min_max
```

### Use R² for model selection

```bash
python main_final.py --target-metric r2
```

### Skip validation and visualization

```bash
python main_final.py --no-validation --no-visualization
```

## Command Line Arguments

- `--skip-eda`: Skip exploratory data analysis
- `--scaling-method`: Choose scaling method (min_max, standard, robust, power)
- `--target-metric`: Metric for model selection (mse, mae, r2)
- `--no-validation`: Skip submission file validation
- `--no-visualization`: Skip prediction visualizations
- `--no-clip-predictions`: Don't clip predictions to [0,1] range

## Features

### Model Comparison

The pipeline compares 9 different machine learning algorithms:

1. Decision Tree Regressor
2. Random Forest Regressor
3. XGBoost Regressor
4. AdaBoost Regressor
5. K-Neighbors Regressor
6. Gradient Boosting Regressor
7. LightGBM Regressor
8. Bagging Regressor
9. Extra Trees Regressor

### Data Preprocessing

- Automatic categorical encoding using OrdinalEncoder
- Multiple scaling options (Standard, Min-Max, Robust, Power)
- Feature range analysis
- Train/validation split with stratification

### Output

- **submission.csv**: Final predictions in Kaggle format
- Comprehensive logging and progress tracking
- Model performance metrics
- Feature importance analysis
- Prediction distribution visualizations

## Dataset Information

The project uses the Kaggle Playground Series S5E10 dataset:

- **Training data**: 517,754 samples with 14 features
- **Test data**: 172,585 samples with 13 features
- **Target variable**: `accident_risk` (continuous values 0-1)
- **Features**: Mix of categorical, boolean, and numerical variables

### Feature Categories:

- **Categorical**: road_type, lighting, weather, time_of_day
- **Boolean**: road_signs_present, public_road, holiday, school_season
- **Numerical**: num_lanes, curvature, speed_limit, num_reported_accidents

## Performance Summary

From the test run:

- **Dataset Size**: 517,754 training samples, 172,585 test samples
- **Features**: 12 features after preprocessing
- **Validation MSE**: 0.004065
- **Prediction Range**: [0.013133, 0.934517]
- **Mean Prediction**: 0.351424
- **Processing Time**: ~30 seconds for basic pipeline

## Troubleshooting

### Unicode Issues

If you encounter Unicode encoding errors:

1. Use `test_pipeline.py` for immediate results
2. Use `main_final.py --skip-eda` for modular approach
3. Remove Unicode characters from all print statements

### Dependencies

Required packages:

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- plotly
- optuna (optional)

## Architecture Benefits

This modular structure provides:

1. **Separation of Concerns**: Each module handles a specific aspect
2. **Reusability**: Individual modules can be imported and used separately
3. **Maintainability**: Easy to debug and modify specific components
4. **Scalability**: Easy to add new models or preprocessing steps
5. **Testing**: Individual modules can be tested in isolation
6. **Configuration**: Centralized settings in config.py

## Future Enhancements

- Hyperparameter optimization with Optuna
- Feature engineering pipeline
- Ensemble methods
- Model interpretability (SHAP, LIME)
- Automated feature selection
- Cross-validation strategy optimization
