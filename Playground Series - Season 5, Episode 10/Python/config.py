"""
Configuration file for Road Accident Risk Prediction project
"""

import pandas as pd
import warnings

# Data file paths
TRAIN_PATH = "Dataset/train.csv"
TEST_PATH = "Dataset/test.csv"
SUBMISSION_PATH = "Dataset/sample_submission.csv"

# Output paths
OUTPUT_DIR = "output/"
SUBMISSION_FILE = "submission.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Scaling methods
SCALING_METHODS = {
    'min_max': 'Min-Max Scaling',
    'standard': 'Standard Scaling',
    'robust': 'Robust Scaling',
    'power': 'Power Transformation'
}

# Selected scaling method
SELECTED_SCALING_METHOD = 'standard'

# Display settings
pd.set_option('display.max_columns', None)

# Warnings filter
warnings.filterwarnings('ignore')