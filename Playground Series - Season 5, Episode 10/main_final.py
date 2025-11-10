"""
Main orchestration file for Road Accident Risk Prediction
Executes the complete pipeline from data loading to submission generation
"""

import sys
import time
import argparse
from pathlib import Path

# Import all modules
import config
from data_loader_clean import load_datasets, get_data_summary
from eda import perform_eda
from preprocessing import prepare_data_for_training, get_preprocessing_summary
from model_training import train_and_evaluate_models
from submission import create_submission_pipeline

def print_header(title, char="=", width=80):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def print_step(step_num, total_steps, title):
    """Print step information"""
    print(f"\nSTEP {step_num}/{total_steps}: {title}")
    print("-" * 60)

def run_complete_pipeline(skip_eda=False, scaling_method=None, target_metric='mse', 
                         clip_predictions=True, validate_submission=True, visualize_predictions=True):
    """
    Run the complete ML pipeline
    
    Args:
        skip_eda (bool): Whether to skip EDA analysis
        scaling_method (str): Scaling method to use (None for default from config)
        target_metric (str): Metric for model selection ('mse', 'mae', 'r2')
        clip_predictions (bool): Whether to clip predictions to [0,1]
        validate_submission (bool): Whether to validate submission file
        visualize_predictions (bool): Whether to create prediction visualizations
        
    Returns:
        dict: Complete pipeline results
    """
    
    # Set scaling method
    if scaling_method is None:
        scaling_method = config.SELECTED_SCALING_METHOD
    
    print_header("ROAD ACCIDENT RISK PREDICTION PIPELINE", "=")
    start_time = time.time()
    
    # Initialize results dictionary
    pipeline_results = {
        'datasets': None,
        'data_summary': None,
        'eda_results': None,
        'prepared_data': None,
        'training_results': None,
        'submission_results': None,
        'execution_time': None,
        'success': False
    }
    
    step = 1
    total_steps = 6 if not skip_eda else 5
    
    try:
        # Step 1: Load Datasets
        print_step(step, total_steps, "Loading Datasets")
        step += 1
        
        train_df, test_df, submission_df = load_datasets()
        
        if train_df is None or test_df is None:
            raise Exception("Failed to load datasets")
        
        pipeline_results['datasets'] = {
            'train': train_df,
            'test': test_df,
            'submission': submission_df
        }
        
        print("[OK] Datasets loaded successfully")
        
        # Step 2: Data Summary
        print_step(step, total_steps, "Data Exploration")
        step += 1
        
        data_summary = get_data_summary(train_df, test_df)
        pipeline_results['data_summary'] = data_summary
        
        print("[OK] Data exploration completed")
        
        # Step 3: EDA (Optional)
        if not skip_eda:
            print_step(step, total_steps, "Exploratory Data Analysis")
            step += 1
            
            eda_results = perform_eda(train_df, target_col='accident_risk')
            pipeline_results['eda_results'] = eda_results
            
            print("[OK] EDA completed")
        else:
            print_step(step, total_steps, "Skipping EDA (as requested)")
            step += 1
        
        # Step 4: Data Preprocessing
        print_step(step, total_steps, "Data Preprocessing")
        step += 1
        
        prepared_data = prepare_data_for_training(
            train_df, 
            test_df, 
            target_col='accident_risk', 
            scaling_method=scaling_method
        )
        pipeline_results['prepared_data'] = prepared_data
        
        preprocessing_summary = get_preprocessing_summary(prepared_data)
        print("[OK] Data preprocessing completed")
        print(f"   Features: {preprocessing_summary['num_features']}")
        print(f"   Training samples: {preprocessing_summary['full_train_shape'][0]}")
        print(f"   Scaling method: {preprocessing_summary['scaling_method']}")
        
        # Step 5: Model Training and Evaluation
        print_step(step, total_steps, "Model Training and Evaluation")
        step += 1
        
        training_results = train_and_evaluate_models(prepared_data)
        pipeline_results['training_results'] = training_results
        
        print("[OK] Model training completed")
        print(f"   Best model: {training_results['best_model'].__class__.__name__}")
        print(f"   Best score: {training_results['best_score']:.6f}")
        
        # Step 6: Submission Generation
        print_step(step, total_steps, "Submission Generation")
        
        submission_results = create_submission_pipeline(
            trained_model=training_results['best_model'],
            test_data=prepared_data,
            test_ids=prepared_data['test_ids'],
            clip_predictions=clip_predictions,
            validate=validate_submission,
            visualize=visualize_predictions
        )
        pipeline_results['submission_results'] = submission_results
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        pipeline_results['execution_time'] = execution_time
        
        # Final summary
        print_header("PIPELINE COMPLETION SUMMARY", "=")
        
        print(f"Task: Road Accident Risk Prediction")
        print(f"Total Execution Time: {execution_time:.2f} seconds")
        print(f"Best Model: {training_results['best_model'].__class__.__name__}")
        print(f"Performance: {training_results['best_score']:.6f} ({target_metric.upper()})")
        print(f"Features Used: {preprocessing_summary['num_features']}")
        print(f"Submission File: {config.SUBMISSION_FILE}")
        
        if submission_results['submission_df'] is not None:
            submission_df = submission_results['submission_df']
            print(f"Predictions: {len(submission_df)} samples")
            print(f"Prediction Range: [{submission_df['accident_risk'].min():.6f}, {submission_df['accident_risk'].max():.6f}]")
            print(f"Mean Prediction: {submission_df['accident_risk'].mean():.6f}")
        
        pipeline_results['success'] = True
        
        # Validation summary
        if validate_submission and submission_results['validation_results']:
            validation = submission_results['validation_results']
            print(f"Submission Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY!", "=")
        
    except Exception as e:
        # Calculate execution time even if failed
        end_time = time.time()
        execution_time = end_time - start_time
        pipeline_results['execution_time'] = execution_time
        
        print(f"\n[PIPELINE FAILED]: {str(e)}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        # Print error details for debugging
        import traceback
        print("\nError Details:")
        print(traceback.format_exc())
        
        pipeline_results['success'] = False
    
    return pipeline_results

def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Road Accident Risk Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --skip-eda                        # Skip EDA analysis
  python main.py --scaling-method min_max          # Use Min-Max scaling
  python main.py --target-metric r2                # Use R² for model selection
  python main.py --no-validation                   # Skip submission validation
  python main.py --no-visualization                # Skip prediction visualizations
        """
    )
    
    parser.add_argument(
        '--skip-eda', 
        action='store_true',
        help='Skip exploratory data analysis'
    )
    
    parser.add_argument(
        '--scaling-method',
        choices=['min_max', 'standard', 'robust', 'power'],
        help='Scaling method to use for feature normalization'
    )
    
    parser.add_argument(
        '--target-metric',
        choices=['mse', 'mae', 'r2'],
        default='mse',
        help='Metric for model selection (default: mse)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip submission file validation'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip prediction visualizations'
    )
    
    parser.add_argument(
        '--no-clip-predictions',
        action='store_true',
        help='Do not clip predictions to [0,1] range'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("ROAD ACCIDENT RISK PREDICTION", "=")
    print("Kaggle Playground Series - Season 5, Episode 10")
    print("Predicting accident risk from tabular data")
    print()
    print("Configuration:")
    print(f"   Skip EDA: {args.skip_eda}")
    print(f"   Scaling Method: {args.scaling_method or config.SELECTED_SCALING_METHOD}")
    print(f"   Target Metric: {args.target_metric}")
    print(f"   Validate Submission: {not args.no_validation}")
    print(f"   Create Visualizations: {not args.no_visualization}")
    print(f"   Clip Predictions: {not args.no_clip_predictions}")
    print()
    
    # Run the pipeline
    pipeline_results = run_complete_pipeline(
        skip_eda=args.skip_eda,
        scaling_method=args.scaling_method,
        target_metric=args.target_metric,
        clip_predictions=not args.no_clip_predictions,
        validate_submission=not args.no_validation,
        visualize_predictions=not args.no_visualization
    )
    
    # Return exit code based on success
    sys.exit(0 if pipeline_results['success'] else 1)

if __name__ == "__main__":
    main()