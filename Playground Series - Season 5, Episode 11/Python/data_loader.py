"""
Data Loading and Preprocessing Module
Handles data loading, basic preprocessing, and initial data inspection
"""

import pandas as pd
import numpy as np
from config import *
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing loan data"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        
    def load_data(self):
        """Load train, test, and sample submission data"""
        try:
            logger.info("Loading data files...")
            
            # Load data files
            self.train_data = pd.read_csv(TRAIN_FILE)
            self.test_data = pd.read_csv(TEST_FILE)
            self.sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
            
            logger.info(f"Train data shape: {self.train_data.shape}")
            logger.info(f"Test data shape: {self.test_data.shape}")
            logger.info(f"Sample submission shape: {self.sample_submission.shape}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_data_info(self):
        """Get basic information about the loaded data"""
        if self.train_data is None or self.test_data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
            
        info = {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape,
            'train_columns': list(self.train_data.columns),
            'test_columns': list(self.test_data.columns),
            'train_dtypes': self.train_data.dtypes.to_dict(),
            'test_dtypes': self.test_data.dtypes.to_dict(),
            'missing_values_train': self.train_data.isnull().sum().to_dict(),
            'missing_values_test': self.test_data.isnull().sum().to_dict()
        }
        
        return info
    
    def identify_feature_types(self, missing_threshold=0.5, cardinality_threshold=10):
        """Automatically identify numerical and categorical features"""
        if self.train_data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None, None
        
        numerical_features = []
        categorical_features = []
        
        for column in self.train_data.columns:
            if column in [FEATURE_ENGINEERING_CONFIG['target_column'], 
                         FEATURE_ENGINEERING_CONFIG['id_column']]:
                continue
                
            # Check if column is numerical
            if self.train_data[column].dtype in ['int64', 'float64']:
                # Check for high cardinality (likely categorical)
                unique_count = self.train_data[column].nunique()
                if unique_count > cardinality_threshold:
                    numerical_features.append(column)
                else:
                    categorical_features.append(column)
            else:
                categorical_features.append(column)
        
        logger.info(f"Identified {len(numerical_features)} numerical features")
        logger.info(f"Identified {len(categorical_features)} categorical features")
        
        return numerical_features, categorical_features
    
    def basic_preprocessing(self):
        """Perform basic preprocessing on the data"""
        if self.train_data is None or self.test_data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        try:
            logger.info("Performing basic preprocessing...")
            
            # Remove duplicate rows
            train_before = self.train_data.shape[0]
            self.train_data = self.train_data.drop_duplicates()
            train_after = self.train_data.shape[0]
            logger.info(f"Removed {train_before - train_after} duplicate rows from train data")
            
            # Handle missing values - simple imputation
            for column in self.train_data.columns:
                if self.train_data[column].isnull().sum() > 0:
                    if self.train_data[column].dtype in ['int64', 'float64']:
                        # Numerical: fill with median
                        fill_value = self.train_data[column].median()
                        self.train_data[column].fillna(fill_value, inplace=True)
                        self.test_data[column].fillna(fill_value, inplace=True)
                    else:
                        # Categorical: fill with mode
                        fill_value = self.train_data[column].mode()[0]
                        self.train_data[column].fillna(fill_value, inplace=True)
                        self.test_data[column].fillna(fill_value, inplace=True)
            
            logger.info("Basic preprocessing completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in basic preprocessing: {e}")
            return False
    
    def get_train_test_split(self, target_column=None):
        """Get train/test split for modeling"""
        if target_column is None:
            target_column = FEATURE_ENGINEERING_CONFIG['target_column']
        
        if self.train_data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None, None
        
        # Remove ID column if present
        if FEATURE_ENGINEERING_CONFIG['id_column'] in self.train_data.columns:
            X_train = self.train_data.drop([target_column, FEATURE_ENGINEERING_CONFIG['id_column']], axis=1)
        else:
            X_train = self.train_data.drop(target_column, axis=1)
            
        y_train = self.train_data[target_column]
        
        # Prepare test data
        if FEATURE_ENGINEERING_CONFIG['id_column'] in self.test_data.columns:
            X_test = self.test_data.drop(FEATURE_ENGINEERING_CONFIG['id_column'], axis=1)
        else:
            X_test = self.test_data
            
        logger.info(f"Train features shape: {X_train.shape}")
        logger.info(f"Train target shape: {y_train.shape}")
        logger.info(f"Test features shape: {X_test.shape}")
        
        return X_train, X_test, y_train

def load_and_preprocess_data():
    """Convenience function to load and preprocess data"""
    loader = DataLoader()
    
    if not loader.load_data():
        return None, None, None
    
    loader.basic_preprocessing()
    X_train, X_test, y_train = loader.get_train_test_split()
    
    return X_train, X_test, y_train

if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    loader = DataLoader()
    
    if loader.load_data():
        info = loader.get_data_info()
        print("\nData Info:")
        print(f"Train shape: {info['train_shape']}")
        print(f"Test shape: {info['test_shape']}")
        print(f"Missing values in train: {sum(info['missing_values_train'].values())}")
        print(f"Missing values in test: {sum(info['missing_values_test'].values())}")
        
        numerical, categorical = loader.identify_feature_types()
        print(f"\nNumerical features: {numerical[:5]}..." if len(numerical) > 5 else f"\nNumerical features: {numerical}")
        print(f"Categorical features: {categorical[:5]}..." if len(categorical) > 5 else f"Categorical features: {categorical}")