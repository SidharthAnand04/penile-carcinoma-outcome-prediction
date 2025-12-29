"""
Enhanced visualizations for the ML pipeline.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_model_comparison_bar(results_df: pd.DataFrame, save_path: str):
    """
    Create a bar chart comparing models across multiple metrics.
    
    Args:
        results_df: DataFrame with model comparison results
        save_path: Path to save figure
    """
    metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision']
    metric_labels = ['ROC-AUC', 'PR-AUC', 'F1 Score', 'Recall', 'Precision']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Sort by metric value
        plot_df = results_df.sort_values(metric, ascending=False)
        
        # Create bar plot
        colors = sns.color_palette("viridis", len(plot_df))
        bars = ax.barh(plot_df['model'], plot_df[metric], color=colors)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, plot_df[metric])):
            ax.text(val + 0.01, i, f'{val:.3f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
    
    plt.suptitle('Model Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model comparison bar chart saved to: {save_path}")


def plot_threshold_analysis(model, X_test, y_test, save_path: str):
    """
    Plot threshold analysis showing how metrics change with threshold.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(thresholds, precisions[:-1], 'b-', linewidth=2, label='Precision', alpha=0.8)
    ax1.plot(thresholds, recalls[:-1], 'g-', linewidth=2, label='Recall', alpha=0.8)
    ax1.plot(thresholds, f1_scores, 'r-', linewidth=3, label='F1 Score')
    
    # Find optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    ax1.axvline(optimal_threshold, color='orange', linestyle='--', 
               linewidth=2, label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax1.set_xlabel('Threshold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('Threshold Analysis: Precision, Recall, and F1 Score',
                 fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Threshold analysis saved to: {save_path}")


def plot_class_distribution(y_train, y_test, save_path: str):
    """
    Plot class distribution in train and test sets.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    train_counts = y_train.value_counts()
    colors = ['#3498db', '#e74c3c']
    
    axes[0].pie(train_counts.values, labels=['N0 (Negative)', 'N+ (Positive)'],
               autopct='%1.1f%%', colors=colors, startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title(f'Training Set (n={len(y_train)})', 
                     fontsize=14, fontweight='bold')
    
    # Test set
    test_counts = y_test.value_counts()
    axes[1].pie(test_counts.values, labels=['N0 (Negative)', 'N+ (Positive)'],
               autopct='%1.1f%%', colors=colors, startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title(f'Test Set (n={len(y_test)})', 
                     fontsize=14, fontweight='bold')
    
    plt.suptitle('Class Distribution: Lymph Node Metastasis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class distribution plot saved to: {save_path}")


def plot_feature_correlation(X, numeric_features: list, save_path: str):
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        X: Feature DataFrame
        numeric_features: List of numeric feature names
        save_path: Path to save figure
    """
    if len(numeric_features) < 2:
        print("Not enough numeric features for correlation plot")
        return
    
    # Calculate correlation
    corr = X[numeric_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1,
               cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation heatmap saved to: {save_path}")


def plot_probability_distribution(model, X_test, y_test, save_path: str):
    """
    Plot distribution of predicted probabilities by true class.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram for each class
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label='N0 (Negative)', 
           color='#3498db', edgecolor='black')
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label='N+ (Positive)', 
           color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Predicted Probability of LN Metastasis', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Predicted Probabilities by True Class',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Probability distribution plot saved to: {save_path}")


def plot_learning_curves(model_scores: Dict, save_path: str):
    """
    Plot cross-validation scores for each model.
    
    Args:
        model_scores: Dict mapping model name to CV scores dict
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['test_roc_auc', 'test_f1', 'test_recall']
    titles = ['ROC-AUC', 'F1 Score', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for model_name, scores in model_scores.items():
            if metric in scores:
                cv_scores = scores[metric]
                folds = range(1, len(cv_scores) + 1)
                ax.plot(folds, cv_scores, marker='o', linewidth=2, 
                       label=model_name, alpha=0.7)
        
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} across CV Folds', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Cross-Validation Performance',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Learning curves saved to: {save_path}")


if __name__ == '__main__':
    print("Visualization module loaded successfully")
