#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for prediction using deep learning model
"""

import os
import numpy as np
import pandas as pd
import argparse
from src.data_processor import DataProcessor
import glob

# Ensure TensorFlow environment is correctly configured
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF warning messages

def predict_files(model_dir, file_paths, output_dir='results'):
    """
    Predict behavior types for files
    
    Parameters:
        model_dir: Model directory
        file_paths: List of file paths
        output_dir: Output directory for results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TensorFlow and model
    import tensorflow as tf
    try:
        # First try loading model with .keras extension
        model_path = os.path.join(model_dir, "sequence_model.keras")
        if os.path.exists(model_path):
            print(f"Loading model: {model_path}")
            model = tf.keras.models.load_model(model_path)
        else:
            # Backward compatibility: try loading model with old filename
            model_path_old = os.path.join(model_dir, "deep_sequence_model.keras")
            if os.path.exists(model_path_old):
                print(f"Loading model: {model_path_old}")
                model = tf.keras.models.load_model(model_path_old)
            else:
                # Try loading model without extension
                model_path_no_ext = os.path.join(model_dir, "sequence_model")
                if os.path.exists(model_path_no_ext):
                    print(f"Loading model: {model_path_no_ext}")
                    model = tf.keras.models.load_model(model_path_no_ext)
                else:
                    raise FileNotFoundError(f"Model file not found, tried paths: {model_path}, {model_path_old}, {model_path}.h5")
        
        print(f"Deep learning model loaded")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Create data processor
    processor = DataProcessor(".")
    
    # Store prediction results
    results = []
    
    for file_path in file_paths:
        try:
            # Read file
            file_name = os.path.basename(file_path)
            df = pd.read_csv(file_path)
            
            # Ensure data has file_name column
            df['file_name'] = file_name
            
            # Extract time series features
            data = processor.extract_time_series_features(df)
            X_sequence = data['sequences']
            
            # Predict
            y_pred_prob = model.predict(X_sequence, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1) + 1  # Add 1 because labels start from 0
            
            # Get final prediction (mode)
            from scipy import stats
            final_pred = stats.mode(y_pred, keepdims=True)[0][0]
            
            # Reclassify non-1,2 categories as category 3
            if final_pred not in [1, 2]:
                final_pred = 3
            
            # Get label text
            mode_labels = {
                1: "Watching Videos",
                2: "Browsing Infinite Feed",
                3: "Other Activities"
            }
            predicted_label = mode_labels.get(final_pred, "Unknown")
            
            # Try to get true label
            try:
                if 'browsing_mode' in df.columns:
                    true_mode = df['browsing_mode'].iloc[0]
                    # Reclassify non-1,2 categories as category 3
                    if true_mode not in [1, 2]:
                        true_mode = 3
                    true_label = mode_labels.get(true_mode, "Unknown")
                    is_correct = final_pred == true_mode
                else:
                    true_mode = None
                    true_label = None
                    is_correct = None
            except:
                true_mode = None
                true_label = None
                is_correct = None
            
            # Add result
            result = {
                "file_name": file_name,
                "predicted_mode": final_pred,
                "predicted_label": predicted_label,
                "true_mode": true_mode,
                "true_label": true_label,
                "is_correct": is_correct
            }
            
            results.append(result)
            print(f"Prediction result: {predicted_label} (True label: {true_label if is_correct is not None else 'Unknown'})")
            
        except Exception as e:
            print(f"Error predicting file {file_path}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "deep_model_predictions.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Calculate overall accuracy
    if 'is_correct' in results_df.columns and not results_df['is_correct'].isna().all():
        accuracy = results_df['is_correct'].mean()
        print(f"\nOverall accuracy: {accuracy:.2%}")
    
    # Return results dataframe
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction using deep learning model")
    parser.add_argument("--files", type=str, nargs='+', required=True, help="CSV file paths to predict")
    parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Expand file path wildcards to actual file list
    file_paths = []
    for file_pattern in args.files:
        matched_files = glob.glob(file_pattern)
        if not matched_files:
            print(f"Warning: No matching files found: {file_pattern}")
        else:
            file_paths.extend(matched_files)
    
    if not file_paths:
        print("Error: No valid CSV files provided")
        exit(1)  # Use exit to terminate program instead of return
    
    predict_files(args.model_dir, file_paths, args.output_dir) 