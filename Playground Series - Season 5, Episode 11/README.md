# Loan Payback Prediction Project

A comprehensive machine learning pipeline for predicting loan payback outcomes, converted from a Jupyter notebook into a modular Python architecture.

## 📁 Project Structure

```
├── config.py                    # Configuration settings and constants
├── data_loader.py              # Data loading and preprocessing
├── eda.py                      # Exploratory Data Analysis
├── feature_engineering.py      # Feature engineering and transformation
├── model_training.py           # Model training and evaluation
├── predict.py                  # Prediction and submission generation
├── main.py                     # Main pipeline orchestration
├── requirements.txt            # Python dependencies
├── loan-payback.ipynb         # Original Jupyter notebook
└── Dataset/                   # Data directory
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python main.py
```

This will execute the full pipeline:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (Multiple algorithms)
- Prediction and Submission Generation

### 3. Run Quick Pipeline (Faster)

```bash
python main.py --mode quick
```

Runs a faster version with basic feature engineering and Random Forest only.

## 📋 Pipeline Modes

### Full Pipeline

```bash
python main.py --mode full
```

- Complete EDA with visualizations
- Advanced feature engineering with selection
- Multiple ML models with cross-validation
- Hyperparameter tuning
- Detailed model evaluation
- Submission file generation

### Quick Pipeline

```bash
python main.py --mode quick
```

- Basic preprocessing
- Minimal feature engineering
- Single Random Forest model
- Fast prediction generation

### EDA Only

```bash
python main.py --mode eda
```

- Perform only exploratory data analysis
- Generate visualizations and reports
- No model training

### Training Only

```bash
python main.py --mode train
```

- Requires preprocessed data from full pipeline
- Train models only (for experimentation)

## ⚙️ Command Line Options

### Pipeline Control

- `--mode {full,quick,eda,train}`: Choose pipeline mode (default: full)
- `--skip-eda`: Skip exploratory data analysis
- `--no-tuning`: Skip hyperparameter tuning
- `--skip-validation`: Skip detailed validation steps

### Feature Engineering

- `--target-features N`: Number of features to select (default: 20)
- `--use-pca`: Use PCA for dimensionality reduction

### Model Selection

- `--model MODEL_NAME`: Specify model for predictions
- `--ensemble`: Use ensemble predictions from multiple models

### Examples

```bash
# Run full pipeline with specific features
python main.py --mode full --target-features 30

# Run with ensemble predictions
python main.py --mode full --ensemble

# Skip hyperparameter tuning for speed
python main.py --mode full --no-tuning

# Use XGBoost for final predictions
python main.py --model xgboost

# Run with PCA reduction to 10 components
python main.py --mode full --use-pca --target-features 10
```

## 📊 Output Files

After running the pipeline, check the `output/` directory for:

- **submission.csv**: Final prediction file for submission
- **Model files**: Trained models saved as `.joblib` files
- **Feature importance**: CSV files with feature importance scores
- **Visualizations**: EDA plots (correlation heatmaps, distributions, etc.)
- **Model comparison**: Performance metrics for all trained models

## 🔧 Configuration

Edit `config.py` to customize:

- **Data paths**: Change dataset locations
- **Model parameters**: Adjust hyperparameters for different algorithms
- **Cross-validation settings**: Modify CV folds and scoring
- **Feature engineering options**: Configure preprocessing steps
- **Random seed**: Set for reproducible results

## 🧪 Available Models

The pipeline includes multiple machine learning algorithms:

1. **Random Forest** - Ensemble tree-based method
2. **XGBoost** - Gradient boosting framework
3. **LightGBM** - Fast gradient boosting
4. **Logistic Regression** - Linear classification
5. **Gradient Boosting** - Scikit-learn's gradient boosting
6. **Support Vector Machine** - SVM classifier
7. **K-Nearest Neighbors** - KNN classifier

## 📈 Pipeline Workflow

```
┌─────────────────┐
│   Data Loading  │
│   (data_loader) │
└────────┬────────┘
         │
┌────────▼────────┐
│      EDA        │
│     (eda.py)    │
└────────┬────────┘
         │
┌────────▼────────┐
│ Feature Eng.    │
│(feature_eng.)   │
└────────┬────────┘
         │
┌────────▼────────┐
│ Model Training  │
│ (model_train)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Predictions    │
│   (predict)     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Submission     │
│   File          │
└─────────────────┘
```

## 🎯 Customization

### Adding New Models

1. Edit `config.py` to add model parameters
2. Update `model_training.py` to include your model
3. Run the pipeline

### Modifying Feature Engineering

1. Edit `feature_engineering.py`
2. Adjust preprocessing steps in the pipeline
3. Customize feature selection methods

### Changing Data Sources

1. Update paths in `config.py`
2. Modify data loading in `data_loader.py`
3. Adjust column names and target variables

## 🔍 Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **File not found**: Check data file paths in `config.py`
3. **Memory issues**: Reduce `--target-features` or use PCA
4. **Slow execution**: Use `--mode quick` or `--no-tuning`

### Debug Mode

Add prints and check the console output for detailed error messages. All modules include comprehensive logging.

## 📧 Support

For questions or issues:

1. Check the console output for error messages
2. Verify data file paths and formats
3. Ensure all dependencies are installed
4. Review the configuration settings

## 📝 Notes

- **Original Notebook**: The `loan-payback.ipynb` file contains the original notebook that was converted to this modular structure
- **Modular Design**: Each component can be run independently for testing and development
- **Scalability**: Easy to add new models, features, or preprocessing steps
- **Reproducibility**: Fixed random seeds ensure consistent results

---

**Created**: Converted from Jupyter notebook to modular Python architecture  
**Purpose**: Loan payback prediction for machine learning competitions  
**Framework**: Scikit-learn, XGBoost, LightGBM
