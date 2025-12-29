# Penile Carcinoma Lymph Node Metastasis Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive machine learning pipeline for predicting lymph node metastasis risk in penile squamous cell carcinoma patients using SEER registry data.

## � Personal Motivation

This project was born from a deeply personal place. My grandfather was diagnosed with penile carcinoma, and watching him navigate the uncertainty of treatment decisions was incredibly difficult for our family. The question that haunted us was: **Has the cancer spread to his lymph nodes?**

This single question determines everything - whether he needs aggressive surgery, what his prognosis looks like, and what quality of life he can expect. Yet the traditional approach involves invasive biopsies or waiting to see if nodes become clinically apparent, both with significant drawbacks.

As someone fascinated by the intersection of **biology, health technology, and machine learning**, I saw an opportunity to contribute something meaningful. This project represents my attempt to use data science to help families like mine make more informed decisions earlier. By analyzing patterns from thousands of penile cancer cases in the SEER database, this model provides a probability-based risk assessment using routinely collected pathology data - no additional procedures required.

While this won't replace clinical judgment, my hope is that it can serve as an additional tool to help oncologists and urologists guide surveillance and treatment decisions, potentially sparing some patients from unnecessary procedures while ensuring others get the aggressive treatment they need.

This is more than just a machine learning project - it's my contribution to improving outcomes for a rare but devastating disease that affects real people and families.

## �📊 Project Overview

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

## 🏥 Clinical Use Cases & Applications

### Current Clinical Practice Challenges

In current penile cancer management, determining lymph node involvement is critical but challenging:

1. **Physical Examination**: Only 58% accurate - palpable nodes may be reactive, non-palpable nodes may harbor metastases
2. **Imaging (CT/MRI)**: Sensitivity ~40-60% - misses micrometastases, false positives from inflammation
3. **Sentinel Lymph Node Biopsy**: Invasive, requires expertise, 5-10% false negative rate
4. **Wait-and-See**: Delays treatment, may miss window for cure
5. **Prophylactic Lymphadenectomy**: Overtreatment in 80% of low-risk cases, significant morbidity (lymphedema, infections)

### How This Model Helps Clinicians

#### 1. **Risk Stratification for Imaging Decisions**

**Scenario**: 62-year-old man with T1b penile cancer, Grade 2, LVI present

**Traditional Approach**: 
- Guidelines recommend imaging (CT/PET) or sentinel node biopsy
- Expensive ($3,000-8,000), radiation exposure, may miss micrometastases

**ML-Enhanced Approach**:
```
Model Prediction: 42% probability of lymph node metastasis
Interpretation: HIGH RISK → Proceed with PET/CT and plan for biopsy
```

**For LOW risk** (probability <20%): Consider surveillance with clinical follow-up, spare patient from imaging

#### 2. **Personalized Surveillance Protocols**

**Use Case**: Tailoring follow-up intensity based on risk

- **Very Low Risk (<15%)**: Clinical exam every 6 months
- **Low Risk (15-30%)**: Exam every 3-4 months + annual imaging
- **Moderate Risk (30-50%)**: Exam every 3 months + CT at 6 months, consider sentinel node biopsy
- **High Risk (>50%)**: Immediate staging workup + consider prophylactic lymphadenectomy

#### 3. **Shared Decision-Making Tool**

**Conversation with Patient**:

> "Your tumor has several high-risk features. Based on analysis of 4,300 similar cases, patients with your tumor characteristics have a 38% chance of lymph node involvement. This means:
> - Option A (Surveillance): 62% chance nodes are clear, avoid surgery complications
> - Option B (Immediate biopsy): Catch metastases early if present, but 62% chance of unnecessary procedure
> Let's discuss what matters most to you..."

Provides quantitative risk to facilitate informed consent.

#### 4. **Pre-Surgical Planning**

**For Urologic Surgeons**:

If model predicts **high risk (>40%)**:
- Schedule staging imaging before penile surgery
- Coordinate with surgical oncology for potential lymphadenectomy
- Counsel patient on combined procedure
- Arrange intraoperative frozen section

If model predicts **low risk (<20%)**:
- Proceed with organ-sparing surgery
- Plan close surveillance post-op
- Reserve lymph node surgery for clinical progression

### Comparison to Current Methodologies

| Method | Sensitivity | Specificity | Invasive? | Cost | Our Model Advantage |
|--------|-------------|-------------|-----------|------|---------------------|
| **Physical Exam** | 58% | 42% | No | Free | More accurate, objective |
| **CT Scan** | 40-55% | 60-75% | No (radiation) | $500-1,500 | No additional cost/radiation |
| **PET/CT** | 60-80% | 88-96% | No (radiation) | $3,000-5,000 | Pre-test probability guides need |
| **MRI** | 50-70% | 70-85% | No | $1,500-3,000 | Can defer in low-risk cases |
| **Sentinel Node Biopsy** | 88-95% | 100% | **Yes** | $5,000-10,000 | Selects who truly needs it |
| **Prophylactic Lymphadenectomy** | 100% | 100% | **Yes** | $15,000-30,000 | Avoids overtreatment |
| **Our ML Model** | 59% (recall) | 86% (specificity) | **No** | **Routine pathology only** | Uses existing data, no extra tests |

### Key Advantages

✅ **Non-invasive**: Uses routinely collected pathology data, no additional procedures  
✅ **Objective**: Removes inter-observer variability in clinical assessment  
✅ **Quantitative**: Provides probability, not just binary high/low risk  
✅ **Population-derived**: Trained on 4,325 real-world cases  
✅ **Explainable**: SHAP values show which features drive each prediction  
✅ **Cost-effective**: No marginal cost beyond pathology already done  
✅ **Immediate**: Results available as soon as pathology report finalized  

### Integration into Clinical Workflow

```
[Patient Diagnosed with Penile Cancer]
           ↓
[Primary Tumor Pathology Review]
    (T-stage, Grade, LVI, Size)
           ↓
[Input to ML Model] ← Also consider: Age, Race, Treatment
           ↓
[Risk Probability Generated]
           ↓
    ┌──────┴──────┐
    ↓             ↓
[<20% Risk]   [>40% Risk]
    ↓             ↓
Surveillance   Imaging + Biopsy
```

## 📈 Using the Trained Model

### Quick Start Example

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('./models/best_model.joblib')

# Example Case: 67-year-old man with T2 tumor, Grade 2, no LVI
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
    'Tumor Size Summary (2016+)': [25],  # 25mm tumor
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

# Clinical Interpretation
if probability[0] < 0.20:
    print("Risk Category: LOW - Consider surveillance")
elif probability[0] < 0.40:
    print("Risk Category: MODERATE - Recommend imaging")
else:
    print("Risk Category: HIGH - Imaging + consider sentinel node biopsy")
```

### Step-by-Step Guide for Your Own Data

#### 1. Prepare Your Data

Your data must include these columns (match SEER format exactly):

**Required Categorical Variables**:
- `Age recode with <1 year olds and 90+`: e.g., "65-69 years"
- `Race recode (W, B, AI, API)`: "White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"
- `Origin recode NHIA (Hispanic, Non-Hisp)`: "Spanish-Hispanic-Latino" or "Non-Spanish-Hispanic-Latino"
- `Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)`: Combined race/ethnicity
- `Primary Site - labeled`: e.g., "C60.1-Glans penis", "C60.2-Body of penis"
- `Derived AJCC T, 7th ed (2010-2015)`: "T1a", "T1b", "T2", "T3", "T4", "TX"
- `Derived AJCC T, 6th ed (2004-2015)`: Same as above (use most recent edition available)
- `Grade Recode (thru 2017)`: "Well differentiated; Grade I", "Moderately differentiated; Grade II", "Poorly differentiated; Grade III"
- `Lymph-vascular Invasion (2004+ varying by schema)`: "Not Present (absent)/Not Identified" or "Lymph-vascular Invasion Present/Identified"
- `Radiation recode`: "None/Unknown", "Beam radiation", etc.
- `Chemotherapy recode (yes, no/unk)`: "Yes" or "No/Unknown"
- `T value - based on AJCC 3rd (1988-2003)`: Legacy T-stage
- `Derived EOD 2018 T Recode (2018+)`: Most recent T-stage

**Required Numeric Variables**:
- `Year of diagnosis`: e.g., 2020
- `Tumor Size Summary (2016+)`: Size in millimeters (999 if unknown)

#### 2. Load and Transform

```python
import pandas as pd
import joblib

# Load your data
your_data = pd.read_csv('your_penile_cases.csv')

# Load model
model = joblib.load('./models/best_model.joblib')

# Predict
probabilities = model.predict_proba(your_data)[:, 1]
predictions = model.predict(your_data)

# Add to dataframe
your_data['LN_metastasis_probability'] = probabilities
your_data['LN_metastasis_prediction'] = predictions

# Save results
your_data.to_csv('predictions_output.csv', index=False)
```

#### 3. Interpret Results

**For Each Patient**:

```python
for idx, row in your_data.iterrows():
    prob = row['LN_metastasis_probability']
    print(f"Patient {idx+1}:")
    print(f"  Probability: {prob:.1%}")
    
    if prob < 0.15:
        risk = "VERY LOW"
        action = "Surveillance every 6 months"
    elif prob < 0.30:
        risk = "LOW"
        action = "Close surveillance every 3-4 months"
    elif prob < 0.50:
        risk = "MODERATE"
        action = "Consider CT imaging + sentinel node biopsy"
    else:
        risk = "HIGH"
        action = "Recommend immediate staging workup"
    
    print(f"  Risk Category: {risk}")
    print(f"  Suggested Action: {action}\n")
```

### Understanding Model Predictions

**What the probability means**:
- **28% probability** = Among 100 patients with similar features, 28 would have lymph node metastasis
- **NOT**: This patient has a 28% tumor burden in nodes
- **IS**: Based on historical patterns, this risk profile

**Important Caveats**:
1. Model trained on US population - may not generalize to other countries
2. Does not replace clinical judgment or imaging findings
3. Pathologic N-stage may differ from clinical assessment
4. Model cannot see inside the patient - only learns from historical patterns

### Variable Importance Reference

See [VARIABLE_GUIDE.md](VARIABLE_GUIDE.md) for detailed explanations of what each variable means, how it's measured, and its clinical significance

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
- Email: sid@sidharthanand.com

## 🙏 Acknowledgments

- **SEER Program**: For providing open-access cancer registry data
- **Open Source Community**: scikit-learn, XGBoost, SHAP, pandas, matplotlib
- **Clinical Experts**: For domain knowledge and validation

## 📞 Contact

For questions, issues, or collaborations:

- Open an issue on [GitHub](https://github.com/SidharthAnand04/penile-carcinoma-outcome-prediction/issues)
- Email: sid@sidharthanand.com

---

**Disclaimer**: This model is for research purposes only and should not be used as the sole basis for clinical decision-making. Always consult with qualified healthcare professionals for medical advice.

---

*Last Updated: December 2025*
