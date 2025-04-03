import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from typing import Dict, Any, List, Tuple

class ModelTrainer:
    """Model training class for training, evaluating and saving models"""
    
    def __init__(self, output_dir: str):
        """
        Initialize model trainer
        
        Parameters:
            output_dir: Model save directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}  # Store trained models
        
    def train_classical_model(self, X: pd.DataFrame, y: np.ndarray, 
                              model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train traditional machine learning model
        
        Parameters:
            X: Feature dataframe
            y: Label array
            model_type: Model type, options are 'random_forest', 'gradient_boosting', 'svm'
            
        Returns:
            Dictionary containing training results
        """
        # Remove non-feature columns
        if 'file_name' in X.columns:
            X = X.drop('file_name', axis=1)
        if 'browsing_mode' in X.columns:
            X = X.drop('browsing_mode', axis=1)
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create processing pipeline
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'clf__n_estimators': [50, 100, 200],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [3, 5, 10]
            }
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
            param_grid = {
                'clf__C': [0.1, 1, 10, 100],
                'clf__gamma': ['scale', 'auto', 0.1, 0.01],
                'clf__kernel': ['rbf', 'linear']
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create processing pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        
        # Check if SMOTE oversampling can be used
        # SMOTE requires at least k_neighbors+1 samples per class, k_neighbors defaults to 5
        use_smote = True
        min_samples_per_class = 6  # Default k_neighbors=5, so need at least 6 samples
        
        # Check sample count for each class
        class_counts = Counter(y_train)
        if min(class_counts.values()) < min_samples_per_class:
            print(f"Warning: Some classes have fewer than {min_samples_per_class} samples, cannot use SMOTE oversampling")
            print(f"Class distribution: {class_counts}")
            use_smote = False
        
        # Use SMOTE to handle imbalanced data (if feasible)
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Class distribution after SMOTE oversampling: {Counter(y_train_resampled)}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
            print(f"Skipping SMOTE oversampling, using original training data")
        
        # Grid search for best parameters
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance (if model supports it)
        feature_importance = None
        if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.named_steps['clf'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Save model
        model_filename = os.path.join(self.output_dir, f"{model_type}_model.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save feature column names for prediction
        feature_filename = os.path.join(self.output_dir, "feature_columns.pkl")
        with open(feature_filename, 'wb') as f:
            pickle.dump(X.columns.tolist(), f)
        
        # Print classification report
        print(f"\n{model_type} model classification report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Return results
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'model_filename': model_filename
        }
        
        return results
    
    def train_sequence_model(self, X_sequences: np.ndarray, y: np.ndarray, file_names: np.ndarray = None) -> Dict[str, Any]:
        """
        Train sequence model (LSTM or GRU)
        
        Parameters:
            X_sequences: Sequence feature array [n_samples, max_seq_length, n_features]
            y: Label array
            file_names: File name array (optional)
            
        Returns:
            Dictionary containing training results
        """
        # Import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model, Sequential
            from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
            from tensorflow.keras.layers import Attention, LayerNormalization, Conv1D, MaxPooling1D, Flatten
            from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.regularizers import l1_l2
        except ImportError:
            raise ImportError("Using sequence model requires TensorFlow, please run: pip install tensorflow")
        
        # Output class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nOriginal class distribution:", dict(zip(unique, counts)))
        
        # Split training and test sets
        if file_names is not None:
            # Use stratified sampling by file name, ensuring data from the same file is all in training or test set
            unique_files = np.unique(file_names)
            file_labels = np.array([y[np.where(file_names == f)[0][0]] for f in unique_files])
            
            train_files, test_files, _, _ = train_test_split(
                unique_files, file_labels, test_size=0.2, random_state=42, stratify=file_labels
            )
            
            train_indices = np.isin(file_names, train_files)
            test_indices = np.isin(file_names, test_files)
            
            X_train, X_test = X_sequences[train_indices], X_sequences[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
        else:
            # Regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Calculate class weights to handle class imbalance
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        class_weights = {cls: n_samples / (n_classes * count) for cls, count in zip(unique_classes, class_counts)}
        
        print("\nTraining set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
        print("Test set class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))
        print("Class weights:", class_weights)
        
        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train - 1)  # Subtract 1 because labels start from 1
        y_test_cat = to_categorical(y_test - 1)
        
        # Get sequence dimensions
        n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
        n_outputs = y_train_cat.shape[1]
        
        # Create advanced LSTM+CNN hybrid model
        # Input layer
        input_layer = Input(shape=(n_timesteps, n_features))
        
        # LSTM branch
        lstm_branch = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(input_layer)
        lstm_branch = LayerNormalization()(lstm_branch)
        lstm_branch = Dropout(0.4)(lstm_branch)
        lstm_branch = Bidirectional(LSTM(32, return_sequences=True))(lstm_branch)
        lstm_branch = Dropout(0.4)(lstm_branch)
        lstm_branch = GlobalAveragePooling1D()(lstm_branch)
        
        # GRU branch (to handle different time scales)
        gru_branch = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(input_layer)
        gru_branch = LayerNormalization()(gru_branch)
        gru_branch = Dropout(0.4)(gru_branch)
        gru_branch = GlobalAveragePooling1D()(gru_branch)
        
        # CNN branch (to capture local patterns)
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        
        # Merge branches
        combined = Concatenate()([lstm_branch, gru_branch, cnn_branch])
        
        # Dense layer
        dense = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
        dense = Dropout(0.5)(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.3)(dense)
        
        # Output layer
        output_layer = Dense(n_outputs, activation='softmax')(dense)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model - Use Adam optimizer with smaller learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Callback functions
        callbacks = [
            # Early stopping strategy
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate dynamic adjustment
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Output model structure summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            epochs=100,
            batch_size=16,  # Use smaller batch_size
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        _, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Predict class
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1) + 1  # Add 1 because labels start from 1
        
        # Save model
        model_filename = os.path.join(self.output_dir, "sequence_model.keras")
        model.save(model_filename)
        
        # Print classification report
        print("\nLSTM model classification report:")
        print(classification_report(y_test, y_pred, target_names=['Watching Videos', 'Browsing Infinite Feed', 'Other Activities']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Return results
        results = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'history': history.history,
            'model_filename': model_filename
        }
        
        return results
    
    def plot_results(self, results: Dict[str, Any], model_name: str) -> None:
        """
        Plot model results
        
        Parameters:
            results: Dictionary containing model results
            model_name: Model name
        """
        # Plot confusion matrix
        cm = results['confusion_matrix']
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.colorbar()
        
        classes = ['Watching Videos', 'Browsing Infinite Feed', 'Other Activities']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save image
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
        
        # Plot feature importance (if available)
        if 'feature_importance' in results and results['feature_importance'] is not None:
            feature_importance = results['feature_importance'].head(15)  # Only show top 15 features
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title(f'{model_name} Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_feature_importance.png"))
            plt.close()
        
        # Plot learning curve (for sequence models)
        if 'history' in results:
            history = results['history']
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='lower right')
            
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_learning_curve.png"))
            plt.close()
    
    def ensemble_models(self, X: pd.DataFrame, y: np.ndarray, 
                        model_types: List[str] = ['random_forest', 'gradient_boosting', 'svm']) -> Dict[str, Any]:
        """
        Train ensemble model
        
        Parameters:
            X: Feature dataframe
            y: Label array
            model_types: List of model types to ensemble
            
        Returns:
            Dictionary containing training results
        """
        # Train individual models
        models = {}
        for model_type in model_types:
            print(f"\nTraining {model_type} model...")
            results = self.train_classical_model(X, y, model_type)
            models[model_type] = results['model']
            
        # Split training and test sets
        X_feat = X.copy()
        if 'file_name' in X_feat.columns:
            X_feat = X_feat.drop('file_name', axis=1)
        if 'browsing_mode' in X_feat.columns:
            X_feat = X_feat.drop('browsing_mode', axis=1)
            
        X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)
        
        # Use individual models for prediction
        preds = {}
        for model_type, model in models.items():
            preds[model_type] = model.predict_proba(X_test)
            
        # Ensemble prediction (simple average)
        ensemble_pred_prob = np.zeros_like(preds[model_types[0]])
        for model_type in model_types:
            ensemble_pred_prob += preds[model_type]
        ensemble_pred_prob /= len(model_types)
        
        # Get final predicted class
        ensemble_pred = np.argmax(ensemble_pred_prob, axis=1) + 1  # +1 because label starts from 1
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Print classification report
        print("\nEnsemble model classification report:")
        print(classification_report(y_test, ensemble_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        
        # Save ensemble model
        ensemble_model = {
            'models': models,
            'model_types': model_types
        }
        
        model_filename = os.path.join(self.output_dir, "ensemble_model.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(ensemble_model, f)
            
        # Return results
        results = {
            'model': ensemble_model,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': ensemble_pred,
            'model_filename': model_filename
        }
        
        return results 