"""
Model evaluation metrics, plots, and threshold selection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    brier_score_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from typing import Dict, Tuple
import os


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """
    Comprehensive evaluation of a single model.
    
    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dict with all metrics
    """
    # Predict probabilities and classes
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_test, y_prob),
    }
    
    return metrics


def evaluate_all_models(models: Dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return comparison table.
    
    Args:
        models: Dict mapping model name to trained pipeline
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 80)
    
    results = []
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return results_df


def find_optimal_threshold(model, X_test, y_test, 
                          strategy: str = 'f1_max',
                          min_recall: float = 0.85) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    Strategies:
    - 'f1_max': Maximize F1 score
    - 'high_recall': Ensure recall >= min_recall, maximize precision
    
    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test labels
        strategy: Threshold selection strategy
        min_recall: Minimum recall for 'high_recall' strategy
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    
    if strategy == 'f1_max':
        # Find threshold that maximizes F1
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
    elif strategy == 'high_recall':
        # Find threshold that gives recall >= min_recall with max precision
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        valid_idx = recalls >= min_recall
        if not valid_idx.any():
            print(f"Warning: Cannot achieve recall >= {min_recall}, using F1 max instead")
            return find_optimal_threshold(model, X_test, y_test, strategy='f1_max')
        
        valid_precisions = precisions[valid_idx]
        valid_thresholds = np.append(thresholds, 0)[valid_idx]
        optimal_idx = np.argmax(valid_precisions)
        optimal_threshold = valid_thresholds[optimal_idx]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_test, y_pred_optimal),
        'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
        'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
        'f1': f1_score(y_test, y_pred_optimal, zero_division=0)
    }
    
    return optimal_threshold, metrics


def plot_roc_curves(models: Dict, X_test, y_test, save_path: str):
    """
    Plot ROC curves for all models.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Lymph Node Metastasis Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to: {save_path}")


def plot_pr_curves(models: Dict, X_test, y_test, save_path: str):
    """
    Plot Precision-Recall curves for all models.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    baseline = (y_test == 1).sum() / len(y_test)
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
    
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})', linewidth=1)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Lymph Node Metastasis Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ PR curve saved to: {save_path}")


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of model
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['N0 (Negative)', 'N+ (Positive)'], fontsize=10)
    plt.yticks(tick_marks, ['N0 (Negative)', 'N+ (Positive)'], fontsize=10)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_calibration_curve(models: Dict, X_test, y_test, save_path: str):
    """
    Plot calibration curves for all models.
    
    Args:
        models: Dict of trained models
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10, strategy='uniform'
        )
        brier = brier_score_loss(y_test, y_prob)
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o',
                label=f'{name} (Brier = {brier:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curves - Lymph Node Metastasis Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Calibration curve saved to: {save_path}")


def generate_classification_report(y_test, y_pred, model_name: str) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of model
        
    Returns:
        Classification report as string
    """
    report = f"\n{'=' * 80}\n"
    report += f"CLASSIFICATION REPORT - {model_name}\n"
    report += f"{'=' * 80}\n"
    report += classification_report(y_test, y_pred, 
                                   target_names=['N0 (No LN)', 'N+ (LN metastasis)'])
    
    cm = confusion_matrix(y_test, y_pred)
    report += f"\nConfusion Matrix:\n"
    report += f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}\n"
    report += f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}\n"
    
    return report


def save_predictions(model, X_test, y_test, test_indices, save_path: str):
    """
    Save test predictions to CSV.
    
    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test labels
        test_indices: Original indices from dataframe
        save_path: Path to save CSV
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    predictions_df = pd.DataFrame({
        'index': test_indices,
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability_ln_metastasis': y_prob
    })
    
    predictions_df.to_csv(save_path, index=False)
    print(f"✓ Predictions saved to: {save_path}")


if __name__ == '__main__':
    # Test evaluation
    from data_loader import load_data
    from preprocessing import split_data, create_preprocessing_pipeline
    from models import train_all_models
    
    df, num_feats, cat_feats = load_data()
    X_train, X_test, y_train, y_test = split_data(df, num_feats, cat_feats)
    
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    models = train_all_models(preprocessor, X_train, y_train, tune=False)
    
    # Evaluate
    results = evaluate_all_models(models, X_test, y_test)
    
    # Create reports directory if needed
    os.makedirs('./reports/figures', exist_ok=True)
    
    # Generate plots
    plot_roc_curves(models, X_test, y_test, './reports/figures/roc_curves.png')
    plot_pr_curves(models, X_test, y_test, './reports/figures/pr_curves.png')
