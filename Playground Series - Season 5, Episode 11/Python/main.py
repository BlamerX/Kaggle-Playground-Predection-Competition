"""
Main Entry Point for Loan Payback Prediction Pipeline
Orchestrates the entire machine learning workflow
"""

import argparse
import sys
import time
from pathlib import Path

# Import all modules
from config import *
from data_loader import DataLoader, load_and_preprocess_data
from eda import ExploratoryDataAnalysis, run_eda
from feature_engineering import FeatureEngineer, run_feature_engineering
from model_training import ModelTrainer, train_models
from predict import PredictionHandler, run_predictions

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")

def run_full_pipeline(args):
    """Run the complete machine learning pipeline"""
    start_time = time.time()
    
    print_header("LOAN PAYBACK PREDICTION PIPELINE")
    print(f"Starting pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_PATH}")
    
    try:
        # Step 1: Load and explore data
        if not args.skip_eda:
            print_header("STEP 1: EXPLORATORY DATA ANALYSIS")
            
            # Run EDA
            eda = ExploratoryDataAnalysis()
            eda.run_full_eda(save_plots=True)
            
            print("EDA completed successfully!")
        else:
            print("Skipping EDA as requested...")
        
        # Step 2: Feature Engineering
        print_header("STEP 2: FEATURE ENGINEERING")
        
        X_train, X_test, y_train, feature_engineer = run_feature_engineering(
            target_features=args.target_features,
            use_pca=args.use_pca
        )
        
        if X_train is None:
            print("Feature engineering failed. Exiting.")
            return False
        
        print("Feature engineering completed successfully!")
        
        # Step 3: Model Training
        print_header("STEP 3: MODEL TRAINING")
        
        trainer = train_models(X_train, y_train, perform_tuning=args.tuning)
        
        if trainer.best_model is None:
            print("Model training failed. Exiting.")
            return False
        
        # Detailed evaluation
        if not args.skip_validation:
            trainer.detailed_evaluation(X_train, y_train)
            
            # Feature importance analysis
            trainer.get_feature_importance(X_train, X_train.columns.tolist())
        
        print("Model training completed successfully!")
        
        # Step 4: Make Predictions and Create Submission
        print_header("STEP 4: PREDICTIONS AND SUBMISSION")
        
        # Use the best model or specify model
        model_to_use = args.model if args.model else trainer.best_model_name
        
        success = run_predictions(
            model_name=model_to_use,
            use_ensemble=args.ensemble,
            X_test=X_test
        )
        
        if not success:
            print("Prediction or submission creation failed.")
            return False
        
        print("Predictions and submission created successfully!")
        
        # Final summary
        elapsed_time = time.time() - start_time
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best CV score: {trainer.model_scores[trainer.best_model_name]['mean_score']:.4f}")
        print(f"Submission file: {SUBMISSION_FILE}")
        print(f"Output directory: {OUTPUT_PATH}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_pipeline(args):
    """Run a quick version of the pipeline"""
    start_time = time.time()
    
    print_header("QUICK LOAN PREDICTION PIPELINE")
    print(f"Starting quick pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load data
        print("Loading data...")
        X_train, X_test, y_train = load_and_preprocess_data()
        
        if X_train is None:
            print("Data loading failed. Exiting.")
            return False
        
        # Basic feature engineering (no selection to save time)
        print("Running basic feature engineering...")
        fe = FeatureEngineer()
        X_train, X_test, y_train, _ = fe.full_feature_engineering_pipeline(target_k=None)
        
        # Train single model (Random Forest for speed)
        print("Training Random Forest model...")
        trainer = ModelTrainer()
        trainer.initialize_models()
        scores = trainer.train_and_evaluate_all(X_train, y_train, save_models=True)
        
        # Make predictions
        print("Making predictions...")
        success = run_predictions(X_test=X_test)
        
        if success:
            elapsed_time = time.time() - start_time
            print(f"\nQuick pipeline completed in {elapsed_time:.2f} seconds!")
            print(f"Submission file: {SUBMISSION_FILE}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"ERROR: Quick pipeline failed: {str(e)}")
        return False

def run_eda_only():
    """Run only the EDA analysis"""
    print_header("EXPLORATORY DATA ANALYSIS ONLY")
    run_eda()
    return True

def run_training_only(args):
    """Run only model training (requires preprocessed data)"""
    print_header("MODEL TRAINING ONLY")
    
    try:
        # Load preprocessed data (you would need to implement this)
        print("Note: This requires preprocessed feature-engineered data.")
        print("Please run the full pipeline first to generate processed data.")
        return False
    except Exception as e:
        print(f"ERROR: Training failed: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and run pipeline"""
    parser = argparse.ArgumentParser(description="Loan Payback Prediction Pipeline")
    
    # Pipeline options
    parser.add_argument('--mode', choices=['full', 'quick', 'eda', 'train'], 
                       default='full', help='Pipeline mode to run')
    
    # EDA options
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA analysis')
    
    # Feature engineering options
    parser.add_argument('--target-features', type=int, default=20, 
                       help='Number of features to select (0 for all)')
    parser.add_argument('--use-pca', action='store_true', help='Use PCA for dimensionality reduction')
    
    # Model training options
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--model', type=str, help='Specific model to use for predictions')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble predictions')
    parser.add_argument('--skip-validation', action='store_true', help='Skip detailed validation')
    
    # Add the missing 'store_true' action
    parser.add_argument('--skip-validation', action='store_true', help='Skip detailed validation')
    
    args = parser.parse_args()
    
    # Adjust boolean arguments
    args.tuning = not args.no_tuning
    
    print(f"Running pipeline in '{args.mode}' mode")
    print(f"Arguments: {args}")
    
    # Run the appropriate pipeline
    success = False
    
    if args.mode == 'full':
        success = run_full_pipeline(args)
    elif args.mode == 'quick':
        success = run_quick_pipeline(args)
    elif args.mode == 'eda':
        success = run_eda_only()
    elif args.mode == 'train':
        success = run_training_only(args)
    else:
        print(f"Unknown mode: {args.mode}")
        success = False
    
    # Final status
    if success:
        print(f"\n{'='*80}")
        print("SUCCESS: Pipeline completed successfully!")
        print("="*80)
        return 0
    else:
        print(f"\n{'='*80}")
        print("FAILURE: Pipeline failed!")
        print("="*80)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)