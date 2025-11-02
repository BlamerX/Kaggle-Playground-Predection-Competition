"""
Configuration file for Loan Payback Prediction Project
Contains all settings and constants used across the project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "Dataset"
OUTPUT_PATH = PROJECT_ROOT / "output"

# File paths
TRAIN_FILE = DATA_PATH / "train.csv"
TEST_FILE = DATA_PATH / "test.csv"
SAMPLE_SUBMISSION = DATA_PATH / "sample_submission.csv"
SUBMISSION_FILE = OUTPUT_PATH / "submission.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# Feature engineering settings
FEATURE_ENGINEERING_CONFIG = {
    'numerical_features': [],
    'categorical_features': [],
    'target_column': 'target',  # Update based on actual target column
    'id_column': 'id',  # Update based on actual ID column
    'missing_threshold': 0.5,
    'cardinality_threshold': 10
}

# Cross-validation settings
CV_FOLDS = 5
CV_SCORING = 'accuracy'  # Update based on competition metric

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(exist_ok=True)