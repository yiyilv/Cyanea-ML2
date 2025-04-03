import argparse
import os
import pandas as pd
import numpy as np
import sys
from typing import List
import glob
import pickle
from sklearn.model_selection import train_test_split

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import GesturePredictor


def train_models(data_dir: str, output_dir: str = 'models',
                model_types: List[str] = None, use_ensemble: bool = True,
                use_sequence: bool = False) -> None:
    """
    Train models
    
    Parameters:
        data_dir: Data directory
        output_dir: Model output directory
        model_types: List of model types to train
        use_ensemble: Whether to use ensemble model
        use_sequence: Whether to train sequence model
    """
    print("Starting data processing...")
    # 加载和处理数据
    processor = DataProcessor(data_dir)
    
    if use_sequence:
        # Use time series features
        print("Extracting time series features...")
        data = processor.prepare_data(use_time_series=True)
        sequences = data['sequences']
        labels = data['labels']
        file_names = data['file_names'] if 'file_names' in data else None
        
        # Train sequence model
        print("Training sequence model (LSTM/GRU hybrid network)...")
        trainer = ModelTrainer(output_dir)
        results = trainer.train_sequence_model(sequences, labels, file_names)
        
        # Visualize results
        trainer.plot_results(results, "sequence_model")
        
        print(f"Sequence model training completed, accuracy: {results['accuracy']:.4f}")
        print(f"Model saved to: {results['model_filename']}")
    else:
        # Use traditional features
        print("Extracting sequence features...")
        features_df = processor.prepare_data(use_time_series=False)
        
        # Reclassify: Only keep categories 1 and 2, others categorized as 3
        print("Reclassifying: Non-1,2 categories are being categorized as category 3...")
        features_df.loc[~features_df['browsing_mode'].isin([1, 2]), 'browsing_mode'] = 3
        
        # Extract labels
        y = features_df['browsing_mode'].values
        
        # Train models
        trainer = ModelTrainer(output_dir)
        
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'svm']
            
        if use_ensemble:
            # Train ensemble model
            print("Training ensemble model...")
            results = trainer.ensemble_models(features_df, y, model_types)
            
            # Visualize results
            trainer.plot_results(results, "ensemble_model")
            
            print(f"Ensemble model training completed, accuracy: {results['accuracy']:.4f}")
            print(f"Model saved to: {results['model_filename']}")
        else:
            # Train individual models
            for model_type in model_types:
                print(f"Training {model_type} model...")
                results = trainer.train_classical_model(features_df, y, model_type)
                
                # Visualize results
                trainer.plot_results(results, model_type)
                
                print(f"{model_type} model training completed, accuracy: {results['accuracy']:.4f}")
                print(f"Model saved to: {results['model_filename']}")


def predict(model_dir: str, file_paths: List[str], model_type: str = 'ensemble',
           use_time_series: bool = False) -> None:
    """
    Make predictions using model
    
    Parameters:
        model_dir: Model directory
        file_paths: List of file paths to predict
        model_type: Model type to use
        use_time_series: Whether to use time series features
    """
    # Check if model type and time series parameters match
    if use_time_series and model_type != 'sequence':
        print(f"Warning: When using time series features, sequence model is recommended. Currently using {model_type} model.")
    
    # Check if files exist
    valid_file_paths = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_file_paths.append(file_path)
        else:
            print(f"Warning: File does not exist: {file_path}")
    
    if not valid_file_paths:
        print("Error: No valid file paths")
        return
    
    # Load predictor
    predictor = GesturePredictor(model_dir)
    
    # Load model
    try:
        predictor.load_model(model_type)
    except Exception as e:
        print(f"Model loading failed: {e}")
        # Try loading backup model
        if model_type == 'ensemble':
            try:
                print("Trying to load random forest model...")
                predictor.load_model('random_forest')
            except Exception as e2:
                print(f"Backup model loading failed: {e2}")
                return
        else:
            return
    
    # Predict files
    results = predictor.predict_files(valid_file_paths, use_time_series)
    
    # Print result summary
    if 'is_correct' in results.columns and not results['is_correct'].isna().all():
        accuracy = results['is_correct'].mean()
        print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Save results
    results.to_csv(os.path.join(model_dir, "prediction_results.csv"), index=False)
    print(f"Prediction results saved to: {os.path.join(model_dir, 'prediction_results.csv')}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Machine Learning Model Training and Prediction Tool")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    train_parser.add_argument("--output_dir", type=str, default="models", help="Model output directory")
    train_parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    train_parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--model_type", type=str, default="all", 
                            choices=["random_forest", "gradient_boosting", "svm", "all"],
                            help="Model type")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="Make predictions using model")
    predict_parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    predict_parser.add_argument("--model_type", type=str, default="ensemble", 
                             choices=["random_forest", "gradient_boosting", "svm", "ensemble", "sequence"],
                             help="Model type")
    predict_parser.add_argument("--files", type=str, nargs='+', required=True, help="CSV file paths to predict")
    predict_parser.add_argument("--output_dir", type=str, default="models", help="Directory for prediction results")
    predict_parser.add_argument("--use_time_series", action="store_true", help="Whether to use time series features (only applicable for sequence model)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_types=[args.model_type] if args.model_type != "all" else None,
            use_ensemble=args.model_type == "all",
            use_sequence=args.model_type == "sequence"
        )
    elif args.command == "predict":
        predict(
            model_dir=args.model_dir,
            file_paths=args.files,
            model_type=args.model_type,
            use_time_series=args.use_time_series
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 