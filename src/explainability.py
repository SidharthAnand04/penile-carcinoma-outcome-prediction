"""
Model explainability with feature importance and SHAP.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def get_feature_importance_logistic(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from Logistic Regression coefficients.
    
    Args:
        model: Trained LogisticRegression pipeline
        feature_names: List of feature names after transformation
        
    Returns:
        DataFrame with feature importance
    """
    # Get coefficients
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    return importance_df


def get_feature_importance_tree(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained tree-based pipeline (RF or XGBoost)
        feature_names: List of feature names after transformation
        
    Returns:
        DataFrame with feature importance
    """
    # Get built-in feature importance
    importance = model.named_steps['classifier'].feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def compute_permutation_importance(model, X_test, y_test, 
                                   feature_names: List[str],
                                   n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation importance.
    
    Args:
        model: Trained pipeline
        X_test: Test features (original, not transformed)
        y_test: Test labels
        feature_names: Original feature names (before one-hot encoding)
        n_repeats: Number of permutations
        
    Returns:
        DataFrame with permutation importance
    """
    # Compute permutation importance
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, 
                           title: str,
                           save_path: str,
                           top_n: int = 20):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
        save_path: Path to save figure
        top_n: Number of top features to show
    """
    # Get top N features
    plot_df = importance_df.head(top_n).copy()
    
    # Determine importance column name
    if 'abs_coefficient' in plot_df.columns:
        importance_col = 'abs_coefficient'
        xlabel = 'Absolute Coefficient Value'
    elif 'importance' in plot_df.columns:
        importance_col = 'importance'
        xlabel = 'Importance Score'
    elif 'importance_mean' in plot_df.columns:
        importance_col = 'importance_mean'
        xlabel = 'Permutation Importance'
    else:
        raise ValueError("Unknown importance column")
    
    # Sort for plotting (ascending for horizontal bar)
    plot_df = plot_df.sort_values(importance_col, ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    bars = ax.barh(range(len(plot_df)), plot_df[importance_col], color=colors)
    
    # Add error bars for permutation importance
    if 'importance_std' in plot_df.columns:
        ax.errorbar(plot_df[importance_col], range(len(plot_df)),
                   xerr=plot_df['importance_std'], fmt='none',
                   ecolor='black', capsize=3)
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved to: {save_path}")


def compute_shap_values(model, X_sample, feature_names: List[str],
                       model_type: str = 'tree') -> shap.Explanation:
    """
    Compute SHAP values for model interpretation.
    
    Args:
        model: Trained pipeline
        X_sample: Sample of data (transformed features)
        feature_names: List of feature names after transformation
        model_type: 'tree' for tree models, 'linear' for linear models
        
    Returns:
        SHAP Explanation object
    """
    print(f"\nComputing SHAP values ({model_type})...")
    
    # Get the classifier from pipeline
    classifier = model.named_steps['classifier']
    
    # Create appropriate explainer
    if model_type == 'tree':
        explainer = shap.TreeExplainer(classifier)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(classifier, X_sample)
    else:
        # Use KernelExplainer as fallback (slower)
        explainer = shap.KernelExplainer(classifier.predict_proba, X_sample)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values


def plot_shap_summary(shap_values, X_sample, feature_names: List[str], save_path: str):
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample of transformed features
        feature_names: List of feature names
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    
    plt.title('SHAP Summary Plot - Feature Impact on LN Metastasis Prediction',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP summary plot saved to: {save_path}")


def generate_explainability_report(models: Dict, 
                                  X_train, X_test, y_test,
                                  num_feats: List[str],
                                  cat_feats: List[str],
                                  output_dir: str):
    """
    Generate comprehensive explainability analysis.
    
    Args:
        models: Dict of trained models
        X_train: Training features (original)
        X_test: Test features (original)
        y_test: Test labels
        num_feats: Numeric feature names
        cat_feats: Categorical feature names
        output_dir: Directory to save outputs
    """
    print("\n" + "=" * 80)
    print("MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    
    # Get transformed feature names from best model (assuming all use same preprocessor)
    best_model = list(models.values())[0]
    preprocessor = best_model.named_steps['preprocessor']
    
    # Get feature names after transformation
    from preprocessing import get_feature_names
    feature_names = get_feature_names(preprocessor, num_feats, cat_feats)
    
    # Transform test data for SHAP (sample for efficiency)
    X_test_transformed = preprocessor.transform(X_test)
    sample_size = min(500, len(X_test_transformed))
    sample_indices = np.random.choice(len(X_test_transformed), sample_size, replace=False)
    X_sample = X_test_transformed[sample_indices]
    
    for model_name, model in models.items():
        print(f"\n{'-' * 80}")
        print(f"Analyzing: {model_name}")
        print(f"{'-' * 80}")
        
        # 1. Built-in feature importance
        if model_name == 'Logistic Regression':
            importance_df = get_feature_importance_logistic(model, feature_names)
            plot_feature_importance(
                importance_df,
                f'Feature Importance - {model_name} (Coefficients)',
                f'{output_dir}/{model_name.lower().replace(" ", "_")}_coefficients.png'
            )
        else:
            importance_df = get_feature_importance_tree(model, feature_names)
            plot_feature_importance(
                importance_df,
                f'Feature Importance - {model_name}',
                f'{output_dir}/{model_name.lower().replace(" ", "_")}_importance.png'
            )
        
        print(f"\nTop 10 important features for {model_name}:")
        print(importance_df.head(10).to_string(index=False))
        
        # 2. Permutation importance (on original features)
        print(f"\nComputing permutation importance for {model_name}...")
        perm_importance_df = compute_permutation_importance(
            model, X_test, y_test, 
            X_test.columns.tolist(),
            n_repeats=5  # Reduced for speed
        )
        
        plot_feature_importance(
            perm_importance_df,
            f'Permutation Importance - {model_name}',
            f'{output_dir}/{model_name.lower().replace(" ", "_")}_permutation.png'
        )
        
        print(f"\nTop 10 features by permutation importance:")
        print(perm_importance_df.head(10).to_string(index=False))
        
        # 3. SHAP analysis (only for best model to save time)
        if model_name == list(models.keys())[0]:  # First model (best by ROC-AUC)
            try:
                if 'XGBoost' in model_name or 'Random Forest' in model_name:
                    model_type = 'tree'
                else:
                    model_type = 'linear'
                
                shap_values = compute_shap_values(
                    model, X_sample, feature_names, model_type
                )
                
                plot_shap_summary(
                    shap_values, X_sample, feature_names,
                    f'{output_dir}/shap_summary_{model_name.lower().replace(" ", "_")}.png'
                )
            except Exception as e:
                print(f"Warning: SHAP analysis failed for {model_name}: {e}")
                print("Skipping SHAP plots.")


if __name__ == '__main__':
    # Test explainability
    from data_loader import load_data
    from preprocessing import split_data, create_preprocessing_pipeline
    from models import train_all_models
    import os
    
    df, num_feats, cat_feats = load_data()
    X_train, X_test, y_train, y_test = split_data(df, num_feats, cat_feats)
    
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    models = train_all_models(preprocessor, X_train, y_train, tune=False)
    
    os.makedirs('./reports/figures', exist_ok=True)
    generate_explainability_report(
        models, X_train, X_test, y_test,
        num_feats, cat_feats,
        './reports/figures'
    )
