# Data Dictionary: Variable Descriptions and Clinical Significance

This document provides detailed explanations of all variables used in the lymph node metastasis prediction model, including their clinical significance and how they contribute to prediction.

---

## Target Variable

### Lymph Node Metastasis Status
- **Variable Name**: `lymph_node_metastasis`
- **Type**: Binary (0/1)
- **Values**:
  - `0`: N0 - No regional lymph node involvement
  - `1`: N+ - Regional lymph node metastasis (N1, N2, or N3)
- **Source**: Derived from AJCC N-stage (6th/7th edition) or EOD 2018 N Recode
- **Clinical Significance**: 
  - Most critical prognostic factor in penile cancer
  - Determines treatment strategy (surveillance vs. lymphadenectomy)
  - 5-year survival: ~85% for N0 vs. ~30% for N+ patients
  - Inguinal lymph node status is the single most important predictor of survival

---

## Predictor Variables

### 1. Age at Diagnosis

**Variable**: `Age recode with <1 year olds and 90+`

- **Type**: Categorical (age groups)
- **Possible Values**: 
  - `<1 year`, `1-4 years`, `5-9 years`, ..., `85-89 years`, `90+ years`
- **Clinical Significance**:
  - Penile cancer typically affects older men (median age 60-70)
  - Older age may be associated with delayed presentation
  - Age influences treatment decisions and surgical tolerance
  - Younger patients may have different tumor biology
- **Model Impact**: Moderate predictor
- **Usage**: Helps stratify risk across age groups

---

### 2. Year of Diagnosis

**Variable**: `Year of diagnosis`

- **Type**: Numeric (year)
- **Range**: 1975-2022 (SEER data span)
- **Clinical Significance**:
  - Captures temporal trends in diagnosis and treatment
  - Earlier years: less advanced imaging, different surgical techniques
  - Recent years: improved staging with PET/CT, sentinel node biopsy
  - HPV awareness and testing evolved over time
- **Model Impact**: Weak predictor
- **Usage**: Accounts for changes in medical practice and diagnostic capabilities

---

### 3. Race and Ethnicity

**Variables** (multiple encodings of same information):
- `Race recode (W, B, AI, API)` - Simple racial categories
- `Origin recode NHIA (Hispanic, Non-Hisp)` - Hispanic ethnicity
- `Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)` - Combined

- **Type**: Categorical
- **Possible Values**: 
  - White, Black, American Indian/Alaska Native, Asian/Pacific Islander
  - Hispanic vs Non-Hispanic
- **Clinical Significance**:
  - Incidence varies by race: Higher in Hispanic men
  - May reflect differences in HPV prevalence
  - Socioeconomic factors affecting access to care
  - Circumcision rates vary by culture/religion
- **Model Impact**: Weak to moderate predictor
- **Usage**: Helps account for population-level risk differences

---

### 4. Primary Tumor Site

**Variable**: `Primary Site - labeled`

- **Type**: Categorical
- **Possible Values**:
  - `C60.0` - Prepuce (foreskin)
  - `C60.1` - Glans penis (head)
  - `C60.2` - Body/shaft of penis
  - `C60.8` - Overlapping lesion
  - `C60.9` - Penis, NOS (not otherwise specified)
- **Clinical Significance**:
  - **Glans** (most common): Lymphatic drainage to superficial inguinal nodes
  - **Prepuce**: May present later, often in uncircumcised men
  - **Shaft**: Different lymphatic drainage patterns
  - Location affects surgical approach and node involvement risk
- **Model Impact**: Moderate predictor
- **Usage**: Anatomic location influences metastatic spread patterns

---

### 5. T-Stage (Tumor Size/Invasion)

**Variables** (multiple editions, model uses all available):
- `Derived AJCC T, 6th ed (2004-2015)`
- `Derived AJCC T, 7th ed (2010-2015)`
- `T value - based on AJCC 3rd (1988-2003)`
- `Derived EOD 2018 T Recode (2018+)`

- **Type**: Categorical
- **Possible Values**: T0, Tis, Ta, T1a, T1b, T2, T3, T4, TX
- **Definitions**:
  - **Ta**: Non-invasive verrucous carcinoma
  - **Tis**: Carcinoma in situ
  - **T1a**: Invades subepithelial connective tissue, no LVI, not poorly differentiated
  - **T1b**: Invades subepithelial connective tissue WITH LVI or poor differentiation
  - **T2**: Invades corpus spongiosum/cavernosum
  - **T3**: Invades urethra or prostate
  - **T4**: Invades other adjacent structures
- **Clinical Significance**:
  - **STRONGEST predictor of lymph node metastasis**
  - Higher T-stage = deeper invasion = higher risk
  - T1a: ~5% risk of node metastasis
  - T1b: ~15-20% risk
  - T2: ~25-40% risk
  - T3-T4: >50% risk
- **Model Impact**: **HIGHEST importance**
- **Usage**: Primary determinant of lymphadenectomy recommendation

---

### 6. Tumor Grade (Differentiation)

**Variable**: `Grade Recode (thru 2017)`

- **Type**: Categorical
- **Possible Values**:
  - `Well differentiated; Grade I` - Cells look almost normal
  - `Moderately differentiated; Grade II` - Intermediate appearance
  - `Poorly differentiated; Grade III` - Abnormal cells
  - `Undifferentiated; anaplastic; Grade IV` - Highly abnormal
  - `Unknown`
- **Clinical Significance**:
  - Reflects tumor aggressiveness
  - **Grade 3-4**: Higher mitotic activity, more likely to metastasize
  - **Grade 1**: Slow-growing, lower metastatic potential
  - Independent predictor of lymph node involvement
  - Grade 3 found in ~15-20% of penile cancers
- **Model Impact**: High importance
- **Usage**: Combined with T-stage for risk stratification
- **Histology Note**: Most penile cancers are squamous cell carcinoma with varying grades

---

### 7. Tumor Size

**Variable**: `Tumor Size Summary (2016+)`

- **Type**: Numeric (millimeters)
- **Range**: 1-999 mm
- **Special Codes**: 999 = unknown
- **Clinical Significance**:
  - Larger tumors more likely to invade deeper
  - Size correlates with T-stage but adds independent information
  - Tumors >3 cm have higher metastatic risk
  - Limited availability (only recorded 2016+)
- **Model Impact**: Moderate (when available)
- **Usage**: Provides quantitative measure of tumor burden

---

### 8. Lymph-Vascular Invasion (LVI)

**Variable**: `Lymph-vascular Invasion (2004+ varying by schema)`

- **Type**: Categorical
- **Possible Values**:
  - `Not Present (absent)/Not Identified` - No LVI seen
  - `Lymph-vascular Invasion Present/Identified` - LVI present
  - `Unknown/Indeterminate`
  - `Blank(s)` - Not recorded (pre-2004)
- **Clinical Significance**:
  - **CRITICAL prognostic factor**
  - Presence of tumor in lymphatic or blood vessels
  - **LVI present**: 3-5x increased risk of node metastasis
  - Defines T1b (vs T1a without LVI)
  - Strong indicator for prophylactic lymphadenectomy
  - Seen in ~20-30% of T1 tumors
- **Model Impact**: **VERY HIGH importance** (when available)
- **Usage**: Major factor in treatment planning
- **Limitation**: Only available 2004+, variable recording quality

---

### 9. Radiation Treatment

**Variable**: `Radiation recode`

- **Type**: Categorical
- **Possible Values**:
  - `None/Unknown` - No radiation or status unknown
  - `Beam radiation` - External beam radiotherapy
  - `Combination` - Multiple radiation modalities
  - `Other` - Brachytherapy, etc.
- **Clinical Significance**:
  - May indicate advanced disease
  - Used for organ preservation in select T1-T2 tumors
  - Adjuvant radiation for positive margins or nodes
  - Presence suggests more aggressive tumor
- **Model Impact**: Moderate predictor
- **Usage**: Reflects disease severity and treatment intensity
- **Note**: This is treatment received, which may indicate clinician's assessment of risk

---

### 10. Chemotherapy Treatment

**Variable**: `Chemotherapy recode (yes, no/unk)`

- **Type**: Binary categorical
- **Possible Values**:
  - `Yes` - Chemotherapy administered
  - `No/Unknown` - No chemo or status unknown
- **Clinical Significance**:
  - **STRONGEST predictor in this model** (somewhat paradoxical)
  - Chemotherapy given for:
    - Node-positive disease (adjuvant)
    - Unresectable tumors
    - Recurrent/metastatic disease
  - Presence indicates high-risk or advanced disease
  - Often neoadjuvant before lymphadenectomy
- **Model Impact**: **HIGHEST feature importance**
- **Usage**: Strong proxy for clinically assessed high-risk disease
- **Interpretation**: Chemotherapy is a marker of disease severity, not a causal risk factor
- **Limitation**: May introduce circular reasoning (treatment based on suspected nodes)

---

## Feature Interpretation Notes

### High-Importance Features (Top Predictors)

1. **Chemotherapy status** - Marker of clinician-assessed high risk
2. **Lymph-vascular invasion** - Direct biological pathway for metastasis
3. **T-stage** - Depth of invasion correlates with node risk
4. **Tumor grade** - Cellular aggressiveness
5. **Radiation treatment** - Another marker of advanced disease

### Interactions

- **T1b = T1a + LVI or Grade 3**: Model captures this interaction
- **T-stage × Grade**: Both work together to predict risk
- **Treatment variables**: May be downstream consequences of true risk factors

### Clinical Utility

The model learns patterns like:
- "T2 tumor + LVI present + Grade 3 → Very high risk"
- "T1a + No LVI + Grade 1 → Low risk, surveillance OK"
- "Chemotherapy given → Clinician suspected nodes"

### Limitations

1. **Treatment variables**: May reflect clinical suspicion rather than true biology
2. **Missing data**: LVI only available 2004+; tumor size 2016+
3. **Retrospective**: Can't capture imaging findings or clinical exam
4. **N-stage combination**: Model predicts regional nodes (inguinal + pelvic) not separately

---

## Using This Information

### For Clinicians

- **High-risk profile**: T2+, Grade 3, LVI+, large tumor → Consider immediate bilateral inguinal lymphadenectomy
- **Low-risk profile**: T1a, Grade 1-2, no LVI, small tumor → Surveillance or sentinel node biopsy
- **Intermediate risk**: Use model probability to guide imaging and biopsy decisions

### For Patients

- **Understanding risk factors**: T-stage and LVI are most important biological factors
- **Treatment implications**: The model helps decide between surveillance vs. surgery
- **Prognosis**: Higher predicted probability indicates need for aggressive treatment

### For Researchers

- **Key targets**: LVI mechanisms, Grade 3 biology, improving early T-staging
- **Model limitations**: Need prospective imaging data, HPV status, molecular markers
- **Future directions**: Multi-class prediction (N1 vs N2 vs N3), survival prediction

---

## Data Quality Notes

| Variable | Completeness | Reliability | Time Period |
|----------|--------------|-------------|-------------|
| Age | 100% | High | All years |
| Race/Ethnicity | 100% | High | All years |
| T-stage | ~90% | Moderate | Varies by edition |
| Grade | ~75% | Moderate | All years |
| Tumor Size | ~10% | High | 2016+ only |
| LVI | ~60% | Moderate | 2004+ only |
| Treatment | ~95% | High | All years |

---

## References

1. AJCC Cancer Staging Manual (8th Edition)
2. EAU Guidelines on Penile Cancer
3. NCCN Guidelines for Penile Cancer
4. Seer.cancer.gov - Data Dictionary

---

*Last Updated: December 2025*
