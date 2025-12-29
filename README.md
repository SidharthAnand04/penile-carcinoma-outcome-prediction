# Penile Carcinoma Lymph Node Metastasis Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive machine learning pipeline for predicting lymph node metastasis risk in penile squamous cell carcinoma patients using SEER registry data.

## 📊 Project Overview

This project implements an end-to-end ML pipeline to predict regional lymph node involvement in penile cancer patients. The models achieve **86.4% ROC-AUC** using clinical and demographic features from the SEER database.

### Key Features

- ✅ **Multiple Models**: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning
- ✅ **Comprehensive Evaluation**: ROC-AUC, PR-AUC, calibration curves, confusion matrices
- ✅ **Model Explainability**: Feature importance, permutation importance, SHAP values
- ✅ **Automated Reporting**: Generates markdown reports with visualizations
- ✅ **Production Ready**: Modular code, proper preprocessing pipelines, saved models

### Performance Highlights

| Model | ROC-AUC | PR-AUC | F1 Score | Accuracy |
|-------|---------|--------|----------|----------|
| **Random Forest** | **0.864** | **0.634** | **0.548** | **0.832** |
| Logistic Regression | 0.861 | 0.614 | 0.576 | 0.820 |
| XGBoost | 0.858 | 0.594 | 0.567 | 0.818 |

## 🗂️ Project Structure

```
penile-carcinoma-outcome-prediction/
├── data/
│   ├── seer_penile_scc.csv          # SEER dataset (not included)
│   └── export.dic                    # Data dictionary
├── src/
│   ├── __init__.py
│   ├── train.py                      # Main training script
│   ├── data_loader.py                # Data loading and filtering
│   ├── preprocessing.py              # Feature engineering
│   ├── models.py                     # Model training
│   ├── evaluation.py                 # Evaluation metrics
│   ├── explainability.py             # Feature importance & SHAP
│   └── visualizations.py             # Enhanced visualizations
├── models/
│   └── best_model.joblib             # Trained model
├── reports/
│   ├── report.md                     # Full analysis report
│   ├── test_predictions.csv          # Test set predictions
│   └── figures/                      # All visualizations
├── notebooks/                         # Jupyter notebooks (optional)
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/SidharthAnand04/penile-carcinoma-outcome-prediction.git
cd penile-carcinoma-outcome-prediction
```

2. **Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Pipeline

**Full training pipeline** (trains all models, generates reports):

```bash
python -m src.train
```

This will:
- Load and preprocess the SEER data
- Train 3 models with cross-validation
- Tune hyperparameters for Random Forest and XGBoost
- Evaluate on held-out test set
- Generate visualizations
- Perform explainability analysis
- Save the best model to `./models/best_model.joblib`
- Generate a comprehensive report in `./reports/report.md`

**Expected runtime**: ~5-10 minutes on a standard laptop

## 📈 Using the Trained Model

### Load and Predict

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('./models/best_model.joblib')

# Prepare a new case (example)
new_case = pd.DataFrame({
    'Age recode with <1 year olds and 90+': ['65-69 years'],
    'Year of diagnosis': [2020],
    'Race recode (W, B, AI, API)': ['White'],
    'Origin recode NHIA (Hispanic, Non-Hisp)': ['Non-Spanish-Hispanic-Latino'],
    'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': ['Non-Hispanic White'],
    'Primary Site - labeled': ['C60.2-Body of penis'],
    'Derived AJCC T, 7th ed (2010-2015)': ['T2'],
    'Derived AJCC T, 6th ed (2004-2015)': ['T2'],
    'Grade Recode (thru 2017)': ['Moderately differentiated; Grade II'],
    'Tumor Size Summary (2016+)': [25],
    'Lymph-vascular Invasion (2004+ varying by schema)': ['Not Present (absent)/Not Identified'],
    'Radiation recode': ['None/Unknown'],
    'Chemotherapy recode (yes, no/unk)': ['No/Unknown'],
    'T value - based on AJCC 3rd (1988-2003)': ['T2'],
    'Derived EOD 2018 T Recode (2018+)': ['T2']
})

# Predict probability and class
probability = model.predict_proba(new_case)[:, 1]
prediction = model.predict(new_case)

print(f"Lymph Node Metastasis Probability: {probability[0]:.2%}")
print(f"Prediction: {'Positive (N+)' if prediction[0] == 1 else 'Negative (N0)'}")
```

## 📊 Dataset

### Source

- **Database**: SEER (Surveillance, Epidemiology, and End Results) Program
- **Cancer Site**: Penile squamous cell carcinoma (ICD-O-3 codes 8070-8084)
- **Time Period**: 1975-2022
- **Cohort Size**: 4,325 cases after filtering

### Target Variable

**Binary Classification:**
- **Class 0 (N0)**: No regional lymph node metastasis
- **Class 1 (N+)**: Regional lymph node metastasis (N1/N2/N3)

### Features

**Numeric (2):**
- Year of diagnosis
- Tumor size (when available, 2016+)

**Categorical (13):**
- Age group
- Race/ethnicity (multiple encodings)
- Primary tumor site
- T-stage (multiple editions: AJCC 3rd/6th/7th, EOD 2018)
- Tumor grade
- Lymph-vascular invasion (LVI)
- Radiation treatment
- Chemotherapy treatment

### Inclusion/Exclusion Criteria

**Included:**
- Squamous cell carcinoma histology
- Known regional lymph node status

**Excluded:**
- Distant metastasis (Stage IV/M1)
- Unknown N-stage

## 🔬 Methodology

### Data Preprocessing

1. **Cohort Filtering**: SCC histology, exclude M1, drop unknown N-stage
2. **Train/Test Split**: 80/20 stratified split
3. **Feature Engineering**:
   - Numeric: Median imputation + Standard scaling
   - Categorical: Most-frequent imputation + One-hot encoding
4. **Pipeline Integration**: All preprocessing in sklearn Pipeline to prevent leakage

### Models

1. **Logistic Regression** (Baseline)
   - L2 regularization, balanced class weights
   - Simple, interpretable

2. **Random Forest** (Best Model)
   - Hyperparameter tuning via RandomizedSearchCV
   - 5-fold cross-validation
   - Balanced class weights

3. **XGBoost**
   - Gradient boosting with hyperparameter tuning
   - Scale_pos_weight for class imbalance
   - Early stopping

### Evaluation Metrics

- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Performance on minority class (important for imbalanced data)
- **F1 Score**: Balance of precision and recall
- **Calibration**: Brier score, calibration curves
- **Confusion Matrix**: Detailed classification performance

### Model Explainability

- **Feature Importance**: Built-in (trees) or coefficients (linear)
- **Permutation Importance**: Model-agnostic feature importance
- **SHAP Values**: Individual prediction explanations

## 📊 Results

### Model Performance

The **Random Forest** model achieved the best performance:

- **ROC-AUC**: 0.864 (86.4% discrimination)
- **PR-AUC**: 0.634 (good performance on positive class)
- **F1 Score**: 0.548 (optimal threshold = 0.460)
- **Accuracy**: 83.2%
- **Recall**: 59.1% (detects 59% of LN+ cases)
- **Precision**: 51.2% (of predicted positive, 51% are true positive)

### Top Predictive Features

1. **Chemotherapy status** - Strongest predictor
2. **Lymph-vascular invasion (LVI)** - Known biological risk factor
3. **Radiation treatment** - May indicate advanced disease
4. **T-stage** - Tumor size/invasion depth
5. **Tumor grade** - Differentiation level

### Clinical Interpretation

- **High-risk features**: Chemotherapy use, LVI present, higher T-stage, poor differentiation
- **Model utility**: Risk stratification for imaging/biopsy decisions
- **Operating point**: Threshold of 0.46 balances sensitivity and specificity

## 📁 Outputs

After running the pipeline, the following files are generated:

### Models

- `./models/best_model.joblib` - Trained Random Forest pipeline (ready for deployment)

### Reports

- `./reports/report.md` - Comprehensive markdown report with:
  - Dataset description
  - Model comparison table
  - Feature importance analysis
  - Clinical interpretation
  - Limitations and future work
  
- `./reports/test_predictions.csv` - Test set predictions with probabilities

### Visualizations

Located in `./reports/figures/`:

- `model_comparison.png` - Bar chart comparing all models
- `roc_curves.png` - ROC curves for all models
- `pr_curves.png` - Precision-Recall curves
- `calibration_curves.png` - Calibration assessment
- `confusion_matrix.png` - Confusion matrix for best model
- `threshold_analysis.png` - How metrics change with threshold
- `class_distribution.png` - Train/test class balance
- `probability_distribution.png` - Predicted probabilities by true class
- `random_forest_importance.png` - Feature importance
- `random_forest_permutation.png` - Permutation importance
- `shap_summary_random_forest.png` - SHAP value summary

## ⚠️ Limitations

### Data Limitations

1. **Registry Data Quality**
   - SEER is population-based, not controlled clinical trial
   - Missing data for some variables (especially LVI, grade)
   - Data quality varies by registry and time period

2. **Target Variable**
   - N-stage combines inguinal and pelvic nodes
   - Clinical N-stage may differ from pathologic
   - Does not capture micro-metastases

3. **Missing Variables**
   - HPV status not available
   - Perineural invasion (PNI) not captured
   - Tumor size only available 2016+

### Model Limitations

1. **Generalizability**
   - Trained on US population (SEER catchment areas)
   - May not generalize to other populations

2. **Class Imbalance**
   - Minority class (LN+) is only 17% of data
   - Model may be conservative

3. **Clinical Applicability**
   - Should complement, not replace, clinical judgment
   - Does not replace imaging or biopsy

## 🔮 Future Work

### Data Enhancement

- [ ] Incorporate imaging features (CT/MRI/PET)
- [ ] Add molecular markers (HPV, p16, biomarkers)
- [ ] Include sentinel lymph node biopsy results
- [ ] Expand to international registries

### Model Improvements

- [ ] Time-to-event models (survival analysis)
- [ ] Multi-class prediction (N0 vs N1 vs N2 vs N3)
- [ ] Deep learning approaches (if more data available)
- [ ] Ensemble methods combining multiple data types

### Validation

- [ ] External validation on independent cohorts
- [ ] Prospective validation in clinical practice
- [ ] Subgroup analyses (by T-stage, grade, etc.)
- [ ] Cost-effectiveness analysis

## 📚 References

### SEER Data

- [SEER Program](https://seer.cancer.gov/)
- [SEER*Stat Software](https://seer.cancer.gov/seerstat/)

### Clinical Guidelines

- NCCN Guidelines for Penile Cancer
- EAU Guidelines on Penile Cancer

### Machine Learning

- Scikit-learn Documentation
- XGBoost Documentation
- SHAP Documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sidharth Anand**

- GitHub: [@SidharthAnand04](https://github.com/SidharthAnand04)
- Email: [your.email@example.com]

## 🙏 Acknowledgments

- **SEER Program**: For providing open-access cancer registry data
- **Open Source Community**: scikit-learn, XGBoost, SHAP, pandas, matplotlib
- **Clinical Experts**: For domain knowledge and validation

## 📞 Contact

For questions, issues, or collaborations:

- Open an issue on [GitHub](https://github.com/SidharthAnand04/penile-carcinoma-outcome-prediction/issues)
- Email: [your.email@example.com]

---

**Disclaimer**: This model is for research purposes only and should not be used as the sole basis for clinical decision-making. Always consult with qualified healthcare professionals for medical advice.

---

*Last Updated: December 2025*
