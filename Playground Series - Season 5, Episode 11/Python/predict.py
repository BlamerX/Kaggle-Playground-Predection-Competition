"""
Prediction and Submission Module
Handles model predictions and submission file generation
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from config import *
import warnings
warnings.filterwarnings('ignore')

class PredictionHandler:
    """Class for making predictions and generating submissions"""
    
    def __init__(self, output_path=None):
        if output_path is None:
            output_path = OUTPUT_PATH
        self.output_path = Path(output_path)
        self.best_model = None
        self.model_name = None
        self.feature_columns = None
        
    def load_trained_model(self, model_name='best_model'):
        """Load the best trained model"""
        possible_names = [
            f"{model_name}_model.joblib",
            f"{model_name}_tuned_model.joblib",
            "best_model.joblib",
            "random_forest_model.joblib",
            "xgboost_model.joblib",
            "lightgbm_model.joblib"
        ]
        
        for model_file in possible_names:
            model_path = self.output_path / model_file
            if model_path.exists():
                try:
                    self.best_model = joblib.load(model_path)
                    self.model_name = model_file.replace('_model.joblib', '').replace('_tuned_model.joblib', '')
                    print(f"Loaded model: {self.model_name} from {model_path}")
                    return True
                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
                    continue
        
        print("No trained model found in output directory.")
        return False
    
    def load_feature_columns(self):
        """Load feature columns used during training"""
        # Check for feature importance file
        importance_files = list(self.output_path.glob("feature_importance_*.csv"))
        if importance_files:
            try:
                importance_df = pd.read_csv(importance_files[0])
                self.feature_columns = importance_df['feature'].tolist()
                print(f"Loaded feature columns from {importance_files[0]}")
                return True
            except Exception as e:
                print(f"Error loading feature columns: {e}")
        
        # If no feature columns file, we need to rely on model features
        if self.best_model is not None:
            # For some models, we can get feature names
            if hasattr(self.best_model, 'feature_names_in_'):
                self.feature_columns = self.best_model.feature_names_in_.tolist()
                print("Loaded feature columns from model")
                return True
        
        print("Could not determine feature columns. Using default approach.")
        return False
    
    def prepare_test_data(self, X_test):
        """Prepare test data for prediction"""
        if self.feature_columns is not None:
            # Ensure test data has the same features as training
            available_features = [col for col in self.feature_columns if col in X_test.columns]
            
            if len(available_features) != len(self.feature_columns):
                missing_features = set(self.feature_columns) - set(available_features)
                print(f"Warning: Missing features in test data: {missing_features}")
                
                # Add missing features with default values
                for feature in missing_features:
                    X_test[feature] = 0  # or appropriate default
            
            # Select and order features as in training
            X_test_prepared = X_test[self.feature_columns]
        else:
            # Use all available features
            X_test_prepared = X_test
        
        return X_test_prepared
    
    def make_predictions(self, X_test):
        """Make predictions using the loaded model"""
        if self.best_model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Prepare test data
            X_test_prepared = self.prepare_test_data(X_test)
            
            print(f"Making predictions with {self.model_name}...")
            print(f"Test data shape: {X_test_prepared.shape}")
            
            # Make predictions
            predictions = self.best_model.predict(X_test_prepared)
            prediction_probs = None
            
            # Try to get prediction probabilities if available
            if hasattr(self.best_model, 'predict_proba'):
                try:
                    prediction_probs = self.best_model.predict_proba(X_test_prepared)
                    print(f"Prediction probabilities shape: {prediction_probs.shape}")
                except Exception as e:
                    print(f"Could not get prediction probabilities: {e}")
            
            return predictions, prediction_probs
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None, None
    
    def create_submission_file(self, predictions, submission_format='default'):
        """Create submission file in the required format"""
        if predictions is None:
            print("No predictions available. Cannot create submission.")
            return False
        
        try:
            # Load sample submission to get the format
            sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
            print(f"Sample submission shape: {sample_submission.shape}")
            print(f"Sample submission columns: {sample_submission.columns.tolist()}")
            
            # Determine ID column
            id_column = None
            for col in sample_submission.columns:
                if 'id' in col.lower():
                    id_column = col
                    break
            
            if id_column is None:
                id_column = sample_submission.columns[0]  # Assume first column is ID
            
            # Determine target column
            target_column = None
            for col in sample_submission.columns:
                if col != id_column:
                    target_column = col
                    break
            
            if target_column is None:
                target_column = 'target'
            
            print(f"Using ID column: {id_column}")
            print(f"Using target column: {target_column}")
            
            # Create submission DataFrame
            submission = pd.DataFrame()
            submission[id_column] = sample_submission[id_column]
            
            # Handle different prediction formats
            if submission_format == 'probabilities' and hasattr(self.best_model, 'predict_proba'):
                # Use probabilities for the positive class
                if predictions.ndim > 1:
                    positive_class_probs = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
                else:
                    positive_class_probs = predictions
                submission[target_column] = positive_class_probs
            else:
                # Use class predictions
                submission[target_column] = predictions
            
            # Ensure proper data types
            if submission[target_column].dtype == 'object':
                # Convert categorical predictions to numeric if needed
                unique_values = submission[target_column].unique()
                if len(unique_values) == 2 and set(unique_values) == {'0', '1'}:
                    submission[target_column] = submission[target_column].astype(int)
            
            # Save submission file
            submission_path = SUBMISSION_FILE
            submission.to_csv(submission_path, index=False)
            
            print(f"Submission file created: {submission_path}")
            print(f"Submission shape: {submission.shape}")
            print("\nFirst few rows of submission:")
            print(submission.head())
            
            # Display prediction distribution
            print(f"\nPrediction distribution:")
            if submission[target_column].dtype in ['int64', 'float64']:
                print(submission[target_column].value_counts().sort_index())
            else:
                print(submission[target_column].value_counts())
            
            return True
            
        except Exception as e:
            print(f"Error creating submission file: {e}")
            return False
    
    def validate_predictions(self, X_test, y_test=None):
        """Validate predictions if ground truth is available"""
        if y_test is None:
            print("No ground truth data available for validation.")
            return
        
        try:
            predictions, _ = self.make_predictions(X_test)
            
            if predictions is not None:
                accuracy = accuracy_score(y_test, predictions)
                print(f"Test accuracy: {accuracy:.4f}")
                
                print("\nClassification Report:")
                print(classification_report(y_test, predictions))
                
                return accuracy
            else:
                print("Could not make predictions for validation.")
                return None
                
        except Exception as e:
            print(f"Error in validation: {e}")
            return None
    
    def create_ensemble_predictions(self, X_test, model_names=None):
        """Create ensemble predictions from multiple models"""
        if model_names is None:
            model_names = ['random_forest', 'xgboost', 'lightgbm']
        
        predictions_list = []
        model_scores = []
        
        for model_name in model_names:
            model_path = self.output_path / f"{model_name}_model.joblib"
            if not model_path.exists():
                print(f"Model {model_name} not found.")
                continue
            
            try:
                model = joblib.load(model_path)
                X_test_prepared = self.prepare_test_data(X_test)
                predictions = model.predict(X_test_prepared)
                predictions_list.append(predictions)
                
                # Get model score if available
                score_file = self.output_path / f"{model_name}_score.txt"
                if score_file.exists():
                    with open(score_file, 'r') as f:
                        score = float(f.read().strip())
                    model_scores.append(score)
                else:
                    model_scores.append(1.0)  # Equal weight if no score
                
                print(f"Loaded predictions from {model_name}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        
        if not predictions_list:
            print("No models found for ensemble.")
            return None
        
        # Create ensemble prediction (weighted average based on model scores)
        predictions_array = np.array(predictions_list)
        
        if len(set(model_scores)) == 1:
            # Simple majority voting if all scores are equal
            ensemble_predictions = np.round(np.mean(predictions_array, axis=0)).astype(int)
        else:
            # Weighted voting based on model scores
            weights = np.array(model_scores) / sum(model_scores)
            ensemble_predictions = np.round(np.average(predictions_array, axis=0, weights=weights)).astype(int)
        
        print(f"Ensemble predictions created from {len(predictions_list)} models")
        return ensemble_predictions

def run_predictions(model_name='best_model', use_ensemble=False, X_test=None, y_test=None):
    """Convenience function to run predictions"""
    predictor = PredictionHandler()
    
    # Load model
    if not predictor.load_trained_model(model_name):
        return False
    
    # Load feature columns
    predictor.load_feature_columns()
    
    if use_ensemble:
        predictions = predictor.create_ensemble_predictions(X_test)
        if predictions is None:
            return False
    else:
        if X_test is None:
            print("X_test data is required for predictions.")
            return False
        
        predictions, prediction_probs = predictor.make_predictions(X_test)
        if predictions is None:
            return False
    
    # Create submission file
    success = predictor.create_submission_file(predictions)
    
    # Validate if ground truth is available
    if y_test is not None:
        predictor.validate_predictions(X_test, y_test)
    
    return success

if __name__ == "__main__":
    # Test prediction module
    print("Prediction module loaded. Run main.py to make predictions with trained models.")