import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Union

class GesturePredictor:
    """Gesture prediction class for loading models and making predictions"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize predictor
        
        Parameters:
            model_dir: Model save directory
        """
        self.model_dir = model_dir
        self.model = None
        self.model_type = None
        self.feature_columns = None
        
    def load_model(self, model_type: str = 'ensemble') -> None:
        """
        Load model
        
        Parameters:
            model_type: Model type, options are 'random_forest', 'gradient_boosting', 'svm', 'ensemble', 'sequence'
        """
        self.model_type = model_type
        
        # Load feature column names
        feature_filename = os.path.join(self.model_dir, "feature_columns.pkl")
        if os.path.exists(feature_filename):
            with open(feature_filename, 'rb') as f:
                self.feature_columns = pickle.load(f)
        
        if model_type == 'sequence':
            try:
                from tensorflow.keras.models import load_model
            except ImportError:
                raise ImportError("Using sequence model requires TensorFlow, please run: pip install tensorflow")
                
            model_path = os.path.join(self.model_dir, "sequence_model.keras")
            # Try loading .keras extension model
            if os.path.exists(model_path):
                self.model = load_model(model_path)
            else:
                # Backward compatibility: Try loading without extension
                model_path_old = os.path.join(self.model_dir, "sequence_model")
                if os.path.exists(model_path_old):
                    self.model = load_model(model_path_old)
                elif os.path.exists(model_path + ".h5"):
                    # Try loading .h5 extension model
                    self.model = load_model(model_path + ".h5")
                else:
                    raise FileNotFoundError(f"Sequence model file not found, attempted paths: {model_path}, {model_path_old}, {model_path}.h5")
        else:
            # Load traditional model
            model_filename = os.path.join(self.model_dir, f"{model_type}_model.pkl")
            if os.path.exists(model_filename):
                with open(model_filename, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise FileNotFoundError(f"Model file not found: {model_filename}")
        
        print(f"Loaded {model_type} model")
    
    def predict_from_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions based on extracted features
        
        Parameters:
            features: DataFrame containing extracted features
            
        Returns:
            Array of predicted browsing_mode labels
        """
        if self.model is None:
            raise ValueError("Please load a model first")
        
        # Remove non-feature columns
        X = features.copy()
        if 'file_name' in X.columns:
            X = X.drop('file_name', axis=1)
        if 'browsing_mode' in X.columns:
            X = X.drop('browsing_mode', axis=1)
        
        # Check if feature columns match
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                # Fill missing columns with 0
                for col in missing_cols:
                    X[col] = 0
            # Ensure feature order is consistent
            X = X[self.feature_columns]
        
        if self.model_type == 'ensemble':
            # Use ensemble model for prediction
            if isinstance(self.model, dict) and 'models' in self.model and 'model_types' in self.model:
                models = self.model['models']
                model_types = self.model['model_types']
                
                # Use each model for prediction
                preds = {}
                for model_type in model_types:
                    preds[model_type] = models[model_type].predict_proba(X)
                    
                # Ensemble prediction (simple average)
                ensemble_pred_prob = np.zeros_like(preds[model_types[0]])
                for model_type in model_types:
                    ensemble_pred_prob += preds[model_type]
                ensemble_pred_prob /= len(model_types)
                
                # Get final prediction class
                predictions = np.argmax(ensemble_pred_prob, axis=1) + 1  # +1 because label starts from 1
            else:
                # If model structure does not match expectations, downgrade to random forest model
                print("Warning: Ensemble model structure is incorrect, using random forest model for prediction")
                random_forest_model_path = os.path.join(self.model_dir, "random_forest_model.pkl")
                if os.path.exists(random_forest_model_path):
                    with open(random_forest_model_path, 'rb') as f:
                        rf_model = pickle.load(f)
                    predictions = rf_model.predict(X)
                else:
                    raise ValueError("Ensemble model structure is incorrect and random forest backup model not found")
        elif self.model_type == 'sequence':
            # Use sequence model for prediction
            predictions_proba = self.model.predict(X)
            predictions = np.argmax(predictions_proba, axis=1) + 1  # +1 because label starts from 1
        else:
            # Use traditional model for prediction
            predictions = self.model.predict(X)
        
        return predictions
    
    def predict_from_raw_data(self, df: pd.DataFrame, use_time_series: bool = False) -> np.ndarray:
        """
        Predict browsing_mode from raw data
        
        Parameters:
            df: DataFrame containing raw gesture data
            use_time_series: Whether to use time series features
            
        Returns:
            Array of prediction results
        """
        from src.data_processor import DataProcessor
        
        # Data processing
        processor = DataProcessor(".")
        
        if use_time_series:
            # Use time series features
            if self.model_type != "sequence":
                raise ValueError("When using time series features, model_type must be 'sequence'")
            
            try:
                # Process time series features for a single file
                # Note: Ensure df has the same file_name, i.e., data from the same file
                data = processor.extract_time_series_features(df)
                X_sequences = data['sequences']
                
                # Check if sequence data is valid
                if X_sequences.size == 0 or len(X_sequences.shape) != 3:
                    raise ValueError(f"Invalid sequence data shape: {X_sequences.shape}")
                
                # Load Tensorflow
                try:
                    import tensorflow as tf
                except ImportError:
                    raise ImportError("Using sequence model requires TensorFlow, please run: pip install tensorflow")
                
                # Use loaded model for prediction
                y_pred_proba = self.model.predict(X_sequences, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1) + 1  # Add 1 because label starts from 0
                
                # Convert prediction results to numpy array
                if not isinstance(y_pred, np.ndarray):
                    y_pred = np.array(y_pred)
                
                # Non-1,2 classes are categorized as class 3
                y_pred[~np.isin(y_pred, [1, 2])] = 3
                
                return y_pred
            except Exception as e:
                print(f"Sequence model prediction error: {e}")
                # When error occurs, downgrade to traditional features
                print("Downgrading to traditional features for prediction")
                use_time_series = False
        else:
            # Use traditional features
            # Extract sequence features
            features = processor.extract_sequence_features(df)
            
            # Remove unnecessary columns
            if 'file_name' in features.columns:
                features = features.drop('file_name', axis=1)
            if 'browsing_mode' in features.columns:
                features = features.drop('browsing_mode', axis=1)
            
            # Handle missing values
            features = features.fillna(0)
            
            # Ensure feature columns match training
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            
            # Re-arrange feature columns to match training order
            features = features[self.feature_columns]
            
            # Make prediction
            y_pred = self.model.predict(features)
        
        return y_pred
    
    def predict_file(self, file_path: str, use_time_series: bool = False) -> Dict[str, Any]:
        """
        Predict browsing_mode for a single file
        
        Parameters:
            file_path: File path
            use_time_series: Whether to use time series features
            
        Returns:
            Dictionary containing prediction results
        """
        import pandas as pd
        
        # Extract true label (if any) from file name
        file_name = os.path.basename(file_path)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure data has file_name column
        df['file_name'] = file_name
        
        # Make prediction
        predictions = self.predict_from_raw_data(df, use_time_series)
        
        # If single file, take mode as final prediction
        if isinstance(predictions, np.ndarray) and predictions.ndim > 0:
            from scipy import stats
            most_common_prediction = stats.mode(predictions, keepdims=True)[0][0]
            prediction = most_common_prediction
        else:
            prediction = predictions[0]
        
        # Non-1,2 classes are uniformly predicted as class 3
        if prediction not in [1, 2]:
            prediction = 3
        
        # Get mode mapping labels
        mode_labels = {
            1: "Watching Videos",
            2: "Browsing Infinite Feed",
            3: "Other Activities"
        }
        predicted_label = mode_labels.get(prediction, "Unknown")
        
        # Try getting true label from file
        try:
            if 'browsing_mode' in df.columns:
                true_mode = df['browsing_mode'].iloc[0]
                # Non-1,2 classes are uniformly categorized as class 3
                if true_mode not in [1, 2]:
                    true_mode = 3
                true_label = mode_labels.get(true_mode, "Unknown")
                is_correct = prediction == true_mode
            else:
                true_mode = None
                true_label = None
                is_correct = None
        except:
            true_mode = None
            true_label = None
            is_correct = None
        
        result = {
            "file_name": file_name,
            "predicted_mode": prediction,
            "predicted_label": predicted_label,
            "true_mode": true_mode,
            "true_label": true_label,
            "is_correct": is_correct
        }
        
        print(f"File {file_name} prediction result: {predicted_label}")
        
        return result
    
    def predict_files(self, file_paths: List[str], use_time_series: bool = False) -> pd.DataFrame:
        """
        Batch predict browsing_mode for multiple CSV files
        
        Parameters:
            file_paths: List of CSV file paths
            use_time_series: Whether to use time series features
            
        Returns:
            DataFrame containing prediction results
        """
        results = []
        
        # Special handling for ensemble model
        if self.model_type == 'ensemble' and isinstance(self.model, dict):
            # Extract models and model_types from ensemble model
            try:
                models = self.model.get('models', {})
                model_types = self.model.get('model_types', [])
                
                if not models or not model_types:
                    raise ValueError("Ensemble model structure is incomplete")
                
                # Temporary save original model data
                original_model = self.model
                original_model_type = self.model_type
                
                # Use random forest model for prediction
                try:
                    self.model = models.get('random_forest')
                    self.model_type = 'random_forest'
                    
                    if self.model is None:
                        raise ValueError("Random forest model not found in ensemble model")
                    
                    for file_path in file_paths:
                        try:
                            result = self.predict_file(file_path, use_time_series)
                            results.append(result)
                            print(f"File {result['file_name']} prediction result: {result['predicted_label']}")
                        except Exception as e:
                            print(f"Error predicting file {file_path}: {e}")
                finally:
                    # Restore original model data
                    self.model = original_model
                    self.model_type = original_model_type
            except Exception as e:
                print(f"Error using ensemble model for prediction: {e}")
                # Try falling back to a single model
                self.model_type = 'random_forest'
                model_filename = os.path.join(self.model_dir, f"{self.model_type}_model.pkl")
                try:
                    with open(model_filename, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Downgraded to random forest model")
                    
                    for file_path in file_paths:
                        try:
                            result = self.predict_file(file_path, use_time_series)
                            results.append(result)
                            print(f"File {result['file_name']} prediction result: {result['predicted_label']}")
                        except Exception as e:
                            print(f"Error predicting file {file_path}: {e}")
                except Exception as e2:
                    print(f"Failed to load downgraded random forest model: {e2}")
        else:
            # Normal processing for non-ensemble models
            for file_path in file_paths:
                try:
                    result = self.predict_file(file_path, use_time_series)
                    results.append(result)
                    print(f"File {result['file_name']} prediction result: {result['predicted_label']}")
                except Exception as e:
                    print(f"Error predicting file {file_path}: {e}")
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def _get_mode_label(self, mode: Union[int, None]) -> Union[str, None]:
        """Get text label corresponding to browsing_mode"""
        if mode is None:
            return None
            
        labels = {
            1: "Watching Videos",
            2: "Browsing Infinite Feed",
            3: "Other Activities"
        }
        
        return labels.get(mode, f"Unknown mode ({mode})") 