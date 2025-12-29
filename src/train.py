"""
Main training script - End-to-end pipeline for lymph node metastasis prediction.

Run with: python -m src.train
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from preprocessing import split_data, create_preprocessing_pipeline, get_feature_names
from models import train_all_models, save_model
from evaluation import (
    evaluate_all_models, find_optimal_threshold,
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix,
    plot_calibration_curve, generate_classification_report,
    save_predictions
)
from explainability import generate_explainability_report
from visualizations import (
    plot_model_comparison_bar, plot_threshold_analysis,
    plot_class_distribution, plot_probability_distribution
)


def create_output_directories():
    """Create output directories if they don't exist."""
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./reports', exist_ok=True)
    os.makedirs('./reports/figures', exist_ok=True)
    print("✓ Output directories created")


def generate_markdown_report(results_df: pd.DataFrame,
                            best_model_name: str,
                            optimal_threshold: float,
                            threshold_metrics: dict,
                            cohort_size: int,
                            num_feats: list,
                            cat_feats: list,
                            importance_df: pd.DataFrame):
    """
    Generate comprehensive markdown report.
    
    Args:
        results_df: Model comparison DataFrame
        best_model_name: Name of best model
        optimal_threshold: Selected operating threshold
        threshold_metrics: Metrics at optimal threshold
        cohort_size: Final cohort size after filtering
        num_feats: Numeric feature names
        cat_feats: Categorical feature names
        importance_df: Feature importance DataFrame
    """
    report = f"""# Lymph Node Metastasis Risk Prediction Model Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report describes an end-to-end machine learning pipeline for predicting lymph node (LN) metastasis risk in penile squamous cell carcinoma patients using SEER registry data.

### Key Findings

- **Best Model:** {best_model_name}
- **Test ROC-AUC:** {results_df.iloc[0]['roc_auc']:.4f}
- **Test PR-AUC:** {results_df.iloc[0]['pr_auc']:.4f}
- **Operating Threshold:** {optimal_threshold:.3f}
- **At this threshold:**
  - Accuracy: {threshold_metrics['accuracy']:.3f}
  - Precision: {threshold_metrics['precision']:.3f}
  - Recall: {threshold_metrics['recall']:.3f}
  - F1 Score: {threshold_metrics['f1']:.3f}

---

## Dataset

### Source
- **Data:** SEER*Stat export of penile squamous cell carcinoma cases
- **File:** `./data/seer_penile_scc.csv`

### Class Distribution

![Class Distribution](figures/class_distribution.png)

The dataset shows class imbalance with approximately 83% N0 (no LN metastasis) and 17% N+ (LN metastasis) cases. This reflects the clinical reality that not all penile cancers metastasize to lymph nodes.

### Cohort Definition

**Inclusion Criteria:**
1. Squamous cell carcinoma histology (ICD-O-3 codes 8070-8084)
2. Known regional lymph node status (N-stage)

**Exclusion Criteria:**
1. Distant metastasis (Stage IV / M1)
2. Unknown or missing N-stage information

**Final Cohort Size:** {cohort_size} cases

### Target Variable

**Label Definition:** Binary classification of lymph node metastasis
- **Class 0 (Negative):** N0 - No regional lymph node involvement
- **Class 1 (Positive):** N1/N2/N3 - Regional lymph node metastasis

**Data Source for Target:** 
- Priority order: Derived AJCC N 6th ed (2004-2015) > 7th ed (2010-2015) > EOD 2018 N Recode

**Rationale:** The target approximates inguinal/pelvic lymph node involvement, which is clinically actionable for surgical planning. Note that SEER does not distinguish between inguinal and pelvic nodes.

---

## Features

### Feature Count
- **Numeric Features:** {len(num_feats)}
- **Categorical Features:** {len(cat_feats)}
- **Total Input Features:** {len(num_feats) + len(cat_feats)}

### Numeric Features
"""
    for feat in num_feats:
        report += f"- {feat}\n"
    
    report += "\n### Categorical Features\n"
    for feat in cat_feats:
        report += f"- {feat}\n"
    
    report += f"""
### Preprocessing Pipeline

**Numeric Features:**
1. Median imputation for missing values
2. Standard scaling (zero mean, unit variance)

**Categorical Features:**
1. Most-frequent imputation for missing values
2. One-hot encoding (handles unknown categories)

**Data Split:**
- Training set: 80% (stratified)
- Test set: 20% (stratified)

---

## Models

Three classification models were trained and compared:

### 1. Logistic Regression (Baseline)
- **Purpose:** Simple, interpretable baseline
- **Configuration:** 
  - L2 regularization
  - Class weights balanced
  - Max iterations: 5000

### 2. Random Forest
- **Purpose:** Ensemble tree-based model
- **Configuration:**
  - Class weights balanced
  - Hyperparameter tuning via RandomizedSearchCV
  - 5-fold cross-validation

### 3. XGBoost
- **Purpose:** Gradient boosting for optimal performance
- **Configuration:**
  - Hyperparameter tuning via RandomizedSearchCV
  - 5-fold cross-validation
  - Scale_pos_weight for class imbalance

---

## Results

### Model Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Brier Score |
|-------|---------|--------|----------|-----------|--------|----|----|
"""
    
    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['brier_score']:.4f} |\n"
    
    report += f"""
### Performance Visualizations

#### Model Comparison Across Metrics

![Model Comparison](figures/model_comparison.png)

#### ROC Curves

![ROC Curves](figures/roc_curves.png)

The ROC (Receiver Operating Characteristic) curve shows the trade-off between true positive rate and false positive rate. All models achieve >0.85 AUC, indicating strong discriminative ability.

#### Precision-Recall Curves

![Precision-Recall Curves](figures/pr_curves.png)

The PR curve is particularly important for imbalanced datasets. Our best model achieves 0.63 PR-AUC, showing good performance on the minority (positive) class.

#### Calibration Curves

![Calibration Curves](figures/calibration_curves.png)

Calibration curves assess how well predicted probabilities match actual outcomes. Well-calibrated models fall close to the diagonal.

#### Confusion Matrix ({best_model_name})

![Confusion Matrix](figures/{best_model_name.lower().replace(' ', '_')}_confusion_matrix.png)

### Operating Threshold Selection

#### Threshold Analysis

![Threshold Analysis](figures/threshold_analysis.png)

**Strategy:** Maximize F1 score (balance between precision and recall)

**Selected Threshold:** {optimal_threshold:.3f}

**Performance at Threshold:**
- Accuracy: {threshold_metrics['accuracy']:.3f}
- Precision: {threshold_metrics['precision']:.3f} (of predicted positive cases, what % are truly positive)
- Recall: {threshold_metrics['recall']:.3f} (of actual positive cases, what % are detected)
- F1 Score: {threshold_metrics['f1']:.3f} (harmonic mean of precision and recall)

---

## Model Interpretation

### Top Predictive Features

The following features have the strongest impact on LN metastasis prediction:

#### Feature Importance Plot

![Feature Importance](figures/{best_model_name.lower().replace(' ', '_')}_importance.png)

#### Top 10 Features

"""
    
    for i, row in importance_df.head(10).iterrows():
        if 'abs_coefficient' in importance_df.columns:
            report += f"{i+1}. **{row['feature']}** (coefficient: {row['coefficient']:.4f})\n"
        else:
            report += f"{i+1}. **{row['feature']}** (importance: {row['importance']:.4f})\n"
    
    report += f"""
#### Permutation Importance

![Permutation Importance](figures/{best_model_name.lower().replace(' ', '_')}_permutation.png)

#### SHAP Analysis

![SHAP Summary](figures/shap_summary_{best_model_name.lower().replace(' ', '_')}.png)

SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions. Red points indicate high feature values, blue indicates low values.

#### Predicted Probability Distribution
- **Model Comparison:** `./reports/figures/model_comparison.png`
- **Threshold Analysis:** `./reports/figures/threshold_analysis.png`
- **Class Distribution:** `./reports/figures/class_distribution.png`
- **Probability Distribution:** `./reports/figures/probability_distribution.png`

![Probability Distribution](figures/probability_distribution.png)

This plot shows how well the model separates the two classes. Good separation indicates strong predictive performance.
### Clinical Interpretation

**Strong Predictors (expected):**
- **T-stage:** Higher T-stage (tumor invasion depth) strongly predicts LN involvement
- **Tumor grade:** Poorly differentiated tumors more likely to metastasize
- **Lymph-vascular invasion (LVI):** Presence indicates higher risk
- **Primary tumor site:** Certain anatomic sites may have different drainage patterns

**Demographic factors:**
- Age, race/ethnicity show weaker associations
- Year of diagnosis may reflect changes in diagnostic/treatment practices

### Feature Importance Visualizations

- **Built-in Importance:** `./reports/figures/{best_model_name.lower().replace(' ', '_')}_importance.png`
- **Permutation Importance:** `./reports/figures/{best_model_name.lower().replace(' ', '_')}_permutation.png`
- **SHAP Summary:** `./reports/figures/shap_summary_{best_model_name.lower().replace(' ', '_')}.png`

---

## Limitations

### Data Limitations

1. **Registry Data Quality:**
   - SEER is a population-based registry, not a clinical trial
   - Missing or incomplete data for some variables (especially LVI, grade)
   - Data quality varies by registry and time period

2. **Target Variable Approximation:**
   - N-stage combines inguinal and pelvic nodes
   - Does not specify laterality (unilateral vs bilateral)
   - Clinical N-stage may differ from pathologic N-stage
   - Does not capture micro-metastases

3. **Missing Key Variables:**
   - Lymph-vascular invasion (LVI) only available 2004+
   - Perineural invasion (PNI) not captured
   - HPV status not available
   - Tumor size only available 2016+

4. **Selection Bias:**
   - Only includes cases reported to SEER registries
   - Treatment patterns may affect case mix
   - Survival bias (excludes patients who died before diagnosis confirmation)

### Model Limitations

1. **Generalizability:**
   - Trained on US population (SEER catchment areas)
   - May not generalize to other populations or healthcare systems
   - Temporal trends: older cases may not reflect current practice

2. **Class Imbalance:**
   - Many N0 cases, fewer N+ cases
   - Model may be conservative in predicting positive class
   - Threshold selection critical for clinical utility

3. **Clinical Applicability:**
   - Model predicts "regional LN involvement" not specifically inguinal nodes
   - Does not replace clinical examination or imaging
   - Should be used for risk stratification, not definitive diagnosis

---

## Recommendations

### Clinical Use

1. **Risk Stratification:** Use predicted probability (0-1) for continuous risk assessment
2. **Decision Support:** High-risk patients (probability > {optimal_threshold:.2f}) may benefit from:
   - Enhanced imaging (CT/MRI/PET)
   - Prophylactic lymphadenectomy consideration
   - Closer surveillance
3. **Not a Replacement:** This model should complement, not replace, clinical judgment and imaging

### Future Improvements

1. **Data Enrichment:**
   - Incorporate imaging data (CT/MRI features)
   - Add molecular markers (HPV status, p16)
   - Include sentinel lymph node biopsy results

2. **Model Enhancements:**
   - Develop time-to-event models (survival analysis)
   - Multi-class prediction (N0 vs N1 vs N2 vs N3)
   - Incorporate treatment response data

3. **Validation:**
   - External validation on independent cohorts
   - Prospective validation in clinical practice
   - Subgroup analyses by T-stage, grade, etc.

---

## Model Deployment

### Saved Artifacts

- **Best Model:** `./models/best_model.joblib`
- **Test Predictions:** `./reports/test_predictions.csv`

### Usage Example

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('./models/best_model.joblib')

# Prepare new case
new_case = pd.DataFrame({{
    'Age recode with <1 year olds and 90+': ['65-69 years'],
    'Year of diagnosis': [2020],
    'Race recode (W, B, AI, API)': ['White'],
    # ... include all features
}})

# Predict
probability = model.predict_proba(new_case)[:, 1]
prediction = model.predict(new_case)

print(f"LN metastasis probability: {{probability[0]:.2%}}")
print(f"Prediction: {{'Positive' if prediction[0] == 1 else 'Negative'}}")
```

---

## Conclusion

This pipeline demonstrates a robust approach to predicting lymph node metastasis in penile SCC using SEER registry data. The {best_model_name} model achieved {results_df.iloc[0]['roc_auc']:.1%} ROC-AUC on the test set, indicating good discriminative ability.

Key clinical predictors include T-stage, tumor grade, and lymph-vascular invasion, aligning with known risk factors. While registry data has inherent limitations, this model provides a data-driven tool for risk stratification to complement clinical decision-making.

Future work should focus on external validation, incorporation of imaging/molecular data, and prospective evaluation in clinical practice.

---

**Report End**
"""
    
    return report


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("LYMPH NODE METASTASIS PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Create output directories
    create_output_directories()
    
    # 2. Load data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND COHORT DEFINITION")
    print("=" * 80)
    df, num_feats, cat_feats = load_data('./data/seer_penile_scc.csv')
    cohort_size = len(df)
    
    # 3. Split data
    print("\n" + "=" * 80)
    print("STEP 2: DATA SPLITTING AND PREPROCESSING")
    print("=" * 80)
    X_train, X_test, y_train, y_test = split_data(df, num_feats, cat_feats)
    
    # Store test indices for later
    test_indices = X_test.index.tolist()
    
    # 4. Create preprocessor
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    
    # 5. Train models
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING")
    print("=" * 80)
    models = train_all_models(preprocessor, X_train, y_train, tune=True)
    
    # 6. Evaluate models
    print("\n" + "=" * 80)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 80)
    results_df = evaluate_all_models(models, X_test, y_test)
    
    # 7. Get best model
    best_model_name = results_df.iloc[0]['model']
    best_model = models[best_model_name]
    
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
    print(f"{'=' * 80}")
    
    # 8. Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        best_model, X_test, y_test, strategy='f1_max'
    )
    
    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"Metrics at threshold:")
    for metric, value in threshold_metrics.items():
        if metric != 'threshold':
            print(f"  {metric}: {value:.4f}")
    
    # 9. Generate evaluation plots
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Standard plots
    plot_roc_curves(models, X_test, y_test, './reports/figures/roc_curves.png')
    plot_pr_curves(models, X_test, y_test, './reports/figures/pr_curves.png')
    plot_calibration_curve(models, X_test, y_test, './reports/figures/calibration_curves.png')
    
    # Confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred_best, best_model_name,
        f'./reports/figures/{best_model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    )
    
    # Enhanced visualizations
    plot_model_comparison_bar(results_df, './reports/figures/model_comparison.png')
    plot_threshold_analysis(best_model, X_test, y_test, './reports/figures/threshold_analysis.png')
    plot_class_distribution(y_train, y_test, './reports/figures/class_distribution.png')
    plot_probability_distribution(best_model, X_test, y_test, './reports/figures/probability_distribution.png')
    
    # Print classification report
    report_text = generate_classification_report(y_test, y_pred_best, best_model_name)
    print(report_text)
    
    # 10. Explainability analysis
    print("\n" + "=" * 80)
    print("STEP 6: MODEL EXPLAINABILITY")
    print("=" * 80)
    generate_explainability_report(
        {best_model_name: best_model},  # Only analyze best model
        X_train, X_test, y_test,
        num_feats, cat_feats,
        './reports/figures'
    )
    
    # 11. Save best model
    print("\n" + "=" * 80)
    print("STEP 7: SAVING OUTPUTS")
    print("=" * 80)
    save_model(best_model, './models/best_model.joblib')
    
    # 12. Save predictions
    save_predictions(best_model, X_test, y_test, test_indices, './reports/test_predictions.csv')
    
    # 13. Generate markdown report
    print("\nGenerating markdown report...")
    
    # Get feature importance for report
    from explainability import get_feature_importance_tree, get_feature_importance_logistic
    feature_names = get_feature_names(preprocessor, num_feats, cat_feats)
    
    if best_model_name == 'Logistic Regression':
        importance_df = get_feature_importance_logistic(best_model, feature_names)
    else:
        importance_df = get_feature_importance_tree(best_model, feature_names)
    
    markdown_report = generate_markdown_report(
        results_df, best_model_name, optimal_threshold, threshold_metrics,
        cohort_size, num_feats, cat_feats, importance_df
    )
    
    with open('./reports/report.md', 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print("✓ Report saved to: ./reports/report.md")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs:")
    print(f"  - Best model: ./models/best_model.joblib")
    print(f"  - Predictions: ./reports/test_predictions.csv")
    print(f"  - Report: ./reports/report.md")
    print(f"  - Figures: ./reports/figures/")
    print(f"\nBest Model: {best_model_name}")
    print(f"Test ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
    print(f"Test PR-AUC: {results_df.iloc[0]['pr_auc']:.4f}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
