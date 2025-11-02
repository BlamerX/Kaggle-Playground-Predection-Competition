"""
Model Training Module
Handles training of multiple machine learning models with cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from config import *
import warnings
import joblib
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class for training and evaluating multiple ML models"""
    
    def __init__(self, output_path=None):
        if output_path is None:
            output_path = OUTPUT_PATH
        self.output_path = Path(output_path)
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(**MODEL_PARAMS['random_forest']),
            'xgboost': xgb.XGBClassifier(**MODEL_PARAMS['xgboost'], eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(**MODEL_PARAMS['lightgbm'], verbose=-1),
            'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'svm': SVC(random_state=RANDOM_STATE, probability=True),
            'knn': KNeighborsClassifier()
        }
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    def evaluate_model(self, model, X_train, y_train, cv_folds=CV_FOLDS):
        """Evaluate a single model using cross-validation"""
        # Use stratified k-fold for classification
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=CV_SCORING)
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def train_and_evaluate_all(self, X_train, y_train, save_models=True):
        """Train and evaluate all models"""
        print("="*60)
        print("MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        if not self.models:
            self.initialize_models()
        
        self.model_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Evaluate model
                scores = self.evaluate_model(model, X_train, y_train)
                self.model_scores[name] = scores
                
                print(f"{name} - CV Score: {scores['mean_score']:.4f} (+/- {scores['std_score']*2:.4f})")
                
                # Train on full dataset
                model.fit(X_train, y_train)
                
                # Save model if requested
                if save_models:
                    model_path = self.output_path / f"{name}_model.joblib"
                    joblib.dump(model, model_path)
                    print(f"Model saved to {model_path}")
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model
        if self.model_scores:
            self.best_model_name = max(self.model_scores.keys(), 
                                     key=lambda k: self.model_scores[k]['mean_score'])
            self.best_model = self.models[self.best_model_name]
            
            print(f"\n{'='*60}")
            print("MODEL COMPARISON RESULTS")
            print(f"{'='*60}")
            
            for name, scores in sorted(self.model_scores.items(), 
                                     key=lambda x: x[1]['mean_score'], reverse=True):
                print(f"{name:20s}: {scores['mean_score']:.4f} (+/- {scores['std_score']*2:.4f})")
            
            print(f"\nBest Model: {self.best_model_name}")
            print(f"Best Score: {self.model_scores[self.best_model_name]['mean_score']:.4f}")
        
        return self.model_scores
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest', 
                            param_grid=None):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\nHyperparameter tuning for {model_name}...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        if param_grid is None:
            # Default parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            print("No parameter grid provided for this model.")
            return None
        
        model = self.models[model_name]
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=CV_SCORING, 
            n_jobs=-1, verbose=1
        )
        
        try:
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update the best model
            self.best_model = grid_search.best_estimator_
            self.best_model_name = model_name
            
            # Save tuned model
            tuned_model_path = self.output_path / f"{model_name}_tuned_model.joblib"
            joblib.dump(grid_search.best_estimator_, tuned_model_path)
            print(f"Tuned model saved to {tuned_model_path}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")
            return None
    
    def detailed_evaluation(self, X_train, y_train):
        """Perform detailed evaluation of the best model"""
        if self.best_model is None:
            print("No best model found. Train models first.")
            return
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION - {self.best_model_name.upper()}")
        print(f"{'='*60}")
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_train
        )
        
        # Train on training split
        self.best_model.fit(X_train_split, y_train_split)
        
        # Make predictions
        y_pred = self.best_model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self, X_train, feature_names=None):
        """Get feature importance from the best model"""
        if self.best_model is None:
            print("No best model found. Train models first.")
            return None
        
        if not hasattr(self.best_model, 'feature_importances_'):
            print("This model doesn't support feature importance.")
            return None
        
        importance_scores = self.best_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Feature Importances ({self.best_model_name}):")
        print(importance_df.head(20))
        
        # Save feature importance
        importance_path = self.output_path / f"feature_importance_{self.best_model_name}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")
        
        return importance_df
    
    def load_trained_model(self, model_name):
        """Load a previously trained model"""
        model_path = self.output_path / f"{model_name}_model.joblib"
        
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Loaded {model_name} model from {model_path}")
            return model
        else:
            print(f"Model file not found: {model_path}")
            return None

def train_models(X_train, y_train, perform_tuning=True, target_model=None):
    """Convenience function to train all models"""
    trainer = ModelTrainer()
    scores = trainer.train_and_evaluate_all(X_train, y_train)
    
    if perform_tuning and target_model and target_model in trainer.models:
        trainer.hyperparameter_tuning(X_train, y_train, target_model)
    
    return trainer

if __name__ == "__main__":
    # Test model training (requires processed data)
    print("Model training module loaded. Run main.py to train models with data.")