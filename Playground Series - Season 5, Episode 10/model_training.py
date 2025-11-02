"""
Model training and evaluation module for Road Accident Risk Prediction
Handles model training, evaluation, and best model selection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def initialize_models():
    """
    Initialize all models for comparison
    
    Returns:
        list: List of model instances
    """
    models = [
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(random_state=42, verbosity=0),
        AdaBoostRegressor(random_state=42),
        KNeighborsRegressor(),
        GradientBoostingRegressor(random_state=42),
        LGBMRegressor(random_state=42, verbose=-1),
        BaggingRegressor(random_state=42),
        ExtraTreesRegressor(n_estimators=100, random_state=42)
    ]
    
    print(f"📦 Initialized {len(models)} models for comparison")
    return models

def evaluate_models(models, X_train, X_val, y_train, y_val):
    """
    Evaluate all models and return performance metrics
    
    Args:
        models (list): List of model instances
        X_train (np.array): Training features
        X_val (np.array): Validation features
        y_train (np.array): Training target
        y_val (np.array): Validation target
        
    Returns:
        dict: Evaluation results
    """
    print("\n🔍 Evaluating Models...")
    print("="*60)
    
    results = {
        'model_names': [],
        'mse_scores': [],
        'mae_scores': [],
        'r2_scores': [],
        'fitted_models': [],
        'predictions': []
    }
    
    for model in models:
        try:
            print(f"Training {model.__class__.__name__}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Store results
            results['model_names'].append(model.__class__.__name__)
            results['mse_scores'].append(mse)
            results['mae_scores'].append(mae)
            results['r2_scores'].append(r2)
            results['fitted_models'].append(model)
            results['predictions'].append(y_pred)
            
            # Print results
            print(f"  {model.__class__.__name__:<30} | MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error training {model.__class__.__name__}: {e}")
            continue
    
    return results

def select_best_model(results, metric='mse'):
    """
    Select best model based on specified metric
    
    Args:
        results (dict): Results from evaluate_models
        metric (str): Metric to use for selection ('mse', 'mae', 'r2')
        
    Returns:
        tuple: (best_model, best_score, best_index)
    """
    if metric == 'mse':
        scores = np.array(results['mse_scores'])
        best_idx = np.argmin(scores)
        print(f"\n✅ Best Model Based on {metric.upper()}: {results['model_names'][best_idx]} (MSE: {scores[best_idx]:.6f})")
    elif metric == 'mae':
        scores = np.array(results['mae_scores'])
        best_idx = np.argmin(scores)
        print(f"\n✅ Best Model Based on {metric.upper()}: {results['model_names'][best_idx]} (MAE: {scores[best_idx]:.6f})")
    elif metric == 'r2':
        scores = np.array(results['r2_scores'])
        best_idx = np.argmax(scores)
        print(f"\n✅ Best Model Based on {metric.upper()}: {results['model_names'][best_idx]} (R²: {scores[best_idx]:.6f})")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return results['fitted_models'][best_idx], scores[best_idx], best_idx

def retrain_on_full_data(best_model, X_full, y_full):
    """
    Retrain best model on full training data
    
    Args:
        best_model: Best model instance
        X_full (np.array): Full training features
        y_full (np.array): Full training target
        
    Returns:
        object: Retrained model
    """
    print(f"\n🚀 Retraining the best model on full training data...")
    
    try:
        # Create fresh instance to avoid issues
        model_type = type(best_model)
        if model_type == RandomForestRegressor:
            retrained_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == XGBRegressor:
            retrained_model = XGBRegressor(random_state=42, verbosity=0)
        elif model_type == LGBMRegressor:
            retrained_model = LGBMRegressor(random_state=42, verbose=-1)
        elif model_type == DecisionTreeRegressor:
            retrained_model = DecisionTreeRegressor(random_state=42)
        elif model_type == AdaBoostRegressor:
            retrained_model = AdaBoostRegressor(random_state=42)
        elif model_type == KNeighborsRegressor:
            retrained_model = KNeighborsRegressor()
        elif model_type == GradientBoostingRegressor:
            retrained_model = GradientBoostingRegressor(random_state=42)
        elif model_type == BaggingRegressor:
            retrained_model = BaggingRegressor(random_state=42)
        elif model_type == ExtraTreesRegressor:
            retrained_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        else:
            retrained_model = model_type()
        
        # Train on full data
        retrained_model.fit(X_full, y_full)
        
        print(f"✅ Model retrained successfully: {retrained_model.__class__.__name__}")
        
        return retrained_model
        
    except Exception as e:
        print(f"❌ Error retraining model: {e}")
        print("Using original model...")
        return best_model

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    Perform cross-validation on a model
    
    Args:
        model: Model instance
        X (np.array): Features
        y (np.array): Target
        cv (int): Number of folds
        scoring (str): Scoring metric
        
    Returns:
        dict: Cross-validation results
    """
    try:
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores_array': cv_scores
        }
    except Exception as e:
        print(f"Warning: Cross-validation failed for {model.__class__.__name__}: {e}")
        return None

def comprehensive_model_evaluation(best_model, X_val, y_val, X_full, y_full):
    """
    Perform comprehensive evaluation of the best model
    
    Args:
        best_model: Best model instance
        X_val (np.array): Validation features
        y_val (np.array): Validation target
        X_full (np.array): Full training features
        y_full (np.array): Full training target
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print("\n📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Predictions on validation set
    y_pred_val = best_model.predict(X_val)
    
    # Metrics on validation set
    mse_val = mean_squared_error(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    
    print(f"Validation Set Performance:")
    print(f"  Mean Squared Error : {mse_val:.6f}")
    print(f"  Mean Absolute Error: {mae_val:.6f}")
    print(f"  R² Score           : {r2_val:.6f}")
    
    # Cross-validation on full training data
    print(f"\nCross-Validation on Full Training Data (5-fold):")
    cv_results = cross_validate_model(best_model, X_full, y_full)
    
    if cv_results:
        print(f"  CV MSE: {cv_results['mean_score']:.6f} (+/- {cv_results['std_score'] * 2:.6f})")
        print(f"  Individual fold scores: {cv_results['scores_array']}")
    
    # Feature importance (if available)
    feature_importance = None
    try:
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            print(f"\nFeature Importance (Top 10):")
            feature_names = getattr(best_model, 'feature_names_in_', None)
            if feature_names is not None:
                importance_pairs = list(zip(feature_names, feature_importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(importance_pairs[:10]):
                    print(f"  {i+1:2d}. {feature:<20}: {importance:.4f}")
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
    
    return {
        'validation_mse': mse_val,
        'validation_mae': mae_val,
        'validation_r2': r2_val,
        'cv_results': cv_results,
        'feature_importance': feature_importance,
        'predictions': y_pred_val
    }

def train_and_evaluate_models(prepared_data):
    """
    Complete model training and evaluation pipeline
    
    Args:
        prepared_data (dict): Output from prepare_data_for_training
        
    Returns:
        dict: Training and evaluation results
    """
    print("🤖 Starting Model Training and Evaluation...")
    
    # Initialize models
    models = initialize_models()
    
    # Evaluate models
    eval_results = evaluate_models(
        models, 
        prepared_data['X_train'], 
        prepared_data['X_val'], 
        prepared_data['y_train'], 
        prepared_data['y_val']
    )
    
    # Select best model based on MSE
    best_model, best_score, best_idx = select_best_model(eval_results, 'mse')
    
    # Comprehensive evaluation
    comp_eval = comprehensive_model_evaluation(
        best_model, 
        prepared_data['X_val'], 
        prepared_data['y_val'],
        prepared_data['X_full'],
        prepared_data['y_full']
    )
    
    # Retrain on full data
    final_model = retrain_on_full_data(best_model, prepared_data['X_full'], prepared_data['y_full'])
    
    print(f"\n✅ Model training completed!")
    print(f"Final model: {final_model.__class__.__name__}")
    
    return {
        'evaluation_results': eval_results,
        'best_model': final_model,
        'best_score': best_score,
        'best_index': best_idx,
        'comprehensive_evaluation': comp_eval
    }