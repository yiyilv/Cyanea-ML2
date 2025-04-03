import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_feature_importance(model_path='models/random_forest_model.pkl', 
                               feature_columns_path='models/feature_columns.pkl',
                               output_path='models/feature_importance.png'):
    # Load model
    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature columns
    print(f"Loading feature columns: {feature_columns_path}")
    with open(feature_columns_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Check model type to determine how to get feature importance
    from sklearn.pipeline import Pipeline
    
    if isinstance(model, Pipeline):
        # If it's a Pipeline, get feature importance through named_steps
        if hasattr(model.named_steps['clf'], 'feature_importances_'):
            importances = model.named_steps['clf'].feature_importances_
        else:
            print("Pipeline model doesn't support feature_importances_ attribute")
            return None
    else:
        # Get feature importance directly from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("Model doesn't support feature_importances_ attribute")
            return None
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Print top 15 most important features
    print("\nTop 15 most important features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(15), 
                                               feature_importance['importance'].head(15)), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Only show top 15 features, or all if less than 15
    num_features = min(15, len(feature_importance))
    top_features = feature_importance.head(num_features)
    
    # Reverse order to put most important feature at the top
    plt.barh(np.arange(num_features), top_features['importance'], align='center')
    plt.yticks(np.arange(num_features), top_features['feature'])
    plt.gca().invert_yaxis()  # Invert y-axis to show most important features at the top
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path)
    print(f"Feature importance chart saved to: {output_path}")
    
    return feature_importance

if __name__ == "__main__":
    # Visualize Random Forest model feature importance
    visualize_feature_importance(
        model_path='models/random_forest_model.pkl',
        feature_columns_path='models/feature_columns.pkl',
        output_path='models/random_forest_feature_importance.png'
    )
    
    # Visualize Gradient Boosting model feature importance
    visualize_feature_importance(
        model_path='models/gradient_boosting_model.pkl',
        feature_columns_path='models/feature_columns.pkl',
        output_path='models/gradient_boosting_feature_importance.png'
    ) 