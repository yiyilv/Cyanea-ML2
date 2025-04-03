#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train deep learning model to recognize user behavior
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.data_processor import DataProcessor

# Ensure TensorFlow environment is correctly configured
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF warning messages

def create_deep_model(input_shape, num_classes):
    """Create LSTM/GRU hybrid deep learning model"""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.layers import LayerNormalization, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.layers import GlobalAveragePooling1D, Concatenate
    from tensorflow.keras.regularizers import l1_l2
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # LSTM branch
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(input_layer)
    lstm_branch = LayerNormalization()(lstm_branch)
    lstm_branch = Dropout(0.4)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(32, return_sequences=True))(lstm_branch)
    lstm_branch = Dropout(0.4)(lstm_branch)
    lstm_branch = GlobalAveragePooling1D()(lstm_branch)
    
    # GRU branch
    gru_branch = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(input_layer)
    gru_branch = LayerNormalization()(gru_branch)
    gru_branch = Dropout(0.4)(gru_branch)
    gru_branch = GlobalAveragePooling1D()(gru_branch)
    
    # CNN branch
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
    output_layer = Dense(num_classes, activation='softmax')(dense)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def train_and_save_model(data_dir, output_dir='models'):
    """Train and save deep learning model"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting data processing...")
    # Load and process data
    processor = DataProcessor(data_dir)
    
    # Extract time series features
    print("Extracting time series features...")
    data = processor.prepare_data(use_time_series=True)
    sequences = data['sequences']
    labels = data['labels']
    file_names = data['file_names']
    
    # Reclassify: Only keep categories 1 and 2, others categorized as 3
    print("Reclassifying: Non-1,2 categories are being categorized as category 3...")
    for i in range(len(labels)):
        if labels[i] not in [1, 2]:
            labels[i] = 3
    
    # Output class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:", dict(zip(unique, counts)))
    
    # Import TensorFlow
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Split training and test sets (by file name)
    unique_files = np.unique(file_names)
    file_labels = np.array([labels[np.where(file_names == f)[0][0]] for f in unique_files])
    
    train_files, test_files, _, _ = train_test_split(
        unique_files, file_labels, test_size=0.2, random_state=42, stratify=file_labels
    )
    
    train_indices = np.isin(file_names, train_files)
    test_indices = np.isin(file_names, test_files)
    
    X_train, X_test = sequences[train_indices], sequences[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    # Calculate class weights to handle class imbalance
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique_classes)
    class_weights = {cls-1: n_samples / (n_classes * count) for cls, count in zip(unique_classes, class_counts)}
    
    print("\nTraining set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test set class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))
    print("Class weights:", class_weights)
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train - 1)  # Subtract 1 because labels start from 1
    y_test_cat = to_categorical(y_test - 1)
    
    # Create model
    print("\nCreating LSTM/GRU hybrid network model...")
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = create_deep_model((n_timesteps, n_features), y_train_cat.shape[1])
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Output model structure
    model.summary()
    
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
    
    # Train model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    _, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test set accuracy: {accuracy:.4f}")
    
    # Predict class
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1) + 1  # Add 1 because labels start from 0
    
    # Print classification report
    print("\nLSTM/GRU Deep Learning Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Watching Videos', 'Browsing Infinite Feed', 'Other Activities']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model
    model_filename = os.path.join(output_dir, "sequence_model.keras")
    model.save(model_filename)
    print(f"Model saved to: {model_filename}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('LSTM/GRU Deep Learning Model Confusion Matrix')
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
    plt.savefig(os.path.join(output_dir, "deep_sequence_model_confusion_matrix.png"))
    plt.close()
    
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deep_sequence_model_learning_curve.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Deep Learning Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    train_and_save_model(args.data_dir, args.output_dir) 