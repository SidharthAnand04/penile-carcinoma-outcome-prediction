"""
Data loading and cohort filtering for SEER penile SCC dataset.
"""
import pandas as pd
import numpy as np
from typing import Tuple


# ===========================
# CONFIGURATION
# ===========================
# Target column: We'll create a binary label from N-stage columns
# Priority: Use 6th ed first (most populated), then 7th ed, then EOD 2018
TARGET_N_COLUMNS = [
    'Derived AJCC N, 6th ed (2004-2015)',
    'Derived AJCC N, 7th ed (2010-2015)',
    'Derived EOD 2018 N Recode (2018+)'
]

# Feature columns to use (will be determined after loading)
# Numeric features: Year of diagnosis, Tumor Size (when available)
# Categorical features: Age group, Race/ethnicity, Primary site, Grade, T-stage, LVI, treatments

# Histology codes for squamous cell carcinoma (8070-8084 range)
SCC_HISTOLOGY_CODES = list(range(8070, 8085))


def load_and_inspect_data(csv_path: str) -> pd.DataFrame:
    """
    Load the SEER CSV and print inspection summary.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Raw DataFrame
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("DATA INSPECTION")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values (NaN). Note: SEER uses 'Blank(s)' for missing data.")
    else:
        print(missing[missing > 0])
    
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    return df


def create_target_label(df: pd.DataFrame) -> pd.Series:
    """
    Create binary target label from N-stage columns.
    
    Label = 1 if N-stage indicates N1, N2, or N3 (lymph node involvement)
    Label = 0 if N-stage is N0 (no lymph node involvement)
    Missing/Unknown (NX, Blank(s), etc.) will be dropped later
    
    Priority: Use 6th ed > 7th ed > EOD 2018
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Series with target labels (0/1) or NaN for unknown
    """
    # Initialize with NaN
    target = pd.Series(np.nan, index=df.index, name='lymph_node_metastasis')
    
    # Combine N-stage columns (priority order)
    n_stage = pd.Series('Unknown', index=df.index)
    for col in reversed(TARGET_N_COLUMNS):  # Reverse so first has priority
        if col in df.columns:
            mask = ~df[col].isin(['Blank(s)', 'Unknown', 'UNK Staging'])
            n_stage[mask] = df.loc[mask, col]
    
    # Create binary label
    # Positive: N1, N2, N3
    positive_mask = n_stage.str.contains('N1|N2|N3', case=False, na=False)
    target[positive_mask] = 1
    
    # Negative: N0
    negative_mask = n_stage.str.strip() == 'N0'
    target[negative_mask] = 0
    
    # Report label distribution
    print("\n" + "=" * 80)
    print("TARGET LABEL CREATION")
    print("=" * 80)
    print(f"\nN-stage source distribution:")
    print(n_stage.value_counts().head(10))
    print(f"\nTarget label distribution:")
    print(f"  Class 0 (N0, no LN metastasis): {(target == 0).sum()}")
    print(f"  Class 1 (N1/N2/N3, LN metastasis): {(target == 1).sum()}")
    print(f"  Unknown/Missing: {target.isna().sum()}")
    print(f"  Class balance: {(target == 1).sum() / (target == 1).sum() + (target == 0).sum():.1%} positive")
    
    return target


def apply_cohort_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cohort inclusion/exclusion criteria.
    
    Filters:
    1. Squamous cell carcinoma histology (ICD-O-3 codes 8070-8084)
    2. Exclude stage IV (distant metastasis) - want to predict regional LN, not M1
    3. Drop rows with unknown target label
    
    Args:
        df: DataFrame with target column
        
    Returns:
        Filtered DataFrame
    """
    initial_n = len(df)
    
    print("\n" + "=" * 80)
    print("COHORT FILTERING")
    print("=" * 80)
    
    # Filter 1: SCC histology
    mask_scc = df['Histologic Type ICD-O-3'].isin(SCC_HISTOLOGY_CODES)
    df = df[mask_scc].copy()
    print(f"\n1. Squamous cell carcinoma filter:")
    print(f"   Retained: {len(df)} / {initial_n} ({len(df)/initial_n:.1%})")
    
    # Filter 2: Exclude Stage IV (distant metastasis)
    # Check if Stage Group column exists and contains IV
    stage_col = 'Derived AJCC Stage Group, 6th ed (2004-2015)'
    if stage_col in df.columns:
        mask_not_m1 = ~df[stage_col].str.contains('IV', case=False, na=False)
        n_before = len(df)
        df = df[mask_not_m1].copy()
        print(f"\n2. Exclude Stage IV (M1 metastasis):")
        print(f"   Retained: {len(df)} / {n_before} ({len(df)/n_before:.1%})")
    else:
        print(f"\n2. Stage Group column not found, skipping M1 filter")
    
    # Filter 3: Drop rows with missing target
    mask_target_known = df['lymph_node_metastasis'].notna()
    n_before = len(df)
    df = df[mask_target_known].copy()
    print(f"\n3. Drop cases with unknown N-stage:")
    print(f"   Retained: {len(df)} / {n_before} ({len(df)/n_before:.1%})")
    
    print(f"\n✓ Final cohort size: {len(df)} cases")
    print(f"  Excluded: {initial_n - len(df)} ({(initial_n - len(df))/initial_n:.1%})")
    
    return df


def identify_features(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical feature columns.
    
    Args:
        df: DataFrame with target
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    # Exclude target and identifier-like columns
    exclude_cols = [
        'lymph_node_metastasis',
        'Histologic Type ICD-O-3',  # Already filtered
    ]
    
    # Identify numeric columns
    numeric_features = []
    if 'Year of diagnosis' in df.columns:
        numeric_features.append('Year of diagnosis')
    
    # Try to parse tumor size if available
    if 'Tumor Size Summary (2016+)' in df.columns:
        # Convert tumor size to numeric (many are coded values like 999 for unknown)
        # We'll handle this in preprocessing
        numeric_features.append('Tumor Size Summary (2016+)')
    
    # Categorical features
    categorical_features = []
    potential_categorical = [
        'Age recode with <1 year olds and 90+',
        'Race recode (W, B, AI, API)',
        'Origin recode NHIA (Hispanic, Non-Hisp)',
        'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)',
        'Primary Site - labeled',
        'Derived AJCC T, 7th ed (2010-2015)',
        'Derived AJCC T, 6th ed (2004-2015)',
        'Grade Recode (thru 2017)',
        'Lymph-vascular Invasion (2004+ varying by schema)',
        'Radiation recode',
        'Chemotherapy recode (yes, no/unk)',
        'T value - based on AJCC 3rd (1988-2003)',
        'Derived EOD 2018 T Recode (2018+)',
    ]
    
    for col in potential_categorical:
        if col in df.columns and col not in exclude_cols:
            categorical_features.append(col)
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION")
    print("=" * 80)
    print(f"\nNumeric features ({len(numeric_features)}):")
    for feat in numeric_features:
        print(f"  - {feat}")
    
    print(f"\nCategorical features ({len(categorical_features)}):")
    for feat in categorical_features:
        print(f"  - {feat}")
    
    return numeric_features, categorical_features


def load_data(csv_path: str = './data/seer_penile_scc.csv') -> Tuple[pd.DataFrame, list, list]:
    """
    Main function to load and prepare data.
    
    Args:
        csv_path: Path to SEER CSV file
        
    Returns:
        Tuple of (filtered_df, numeric_features, categorical_features)
    """
    # Load and inspect
    df = load_and_inspect_data(csv_path)
    
    # Create target label
    df['lymph_node_metastasis'] = create_target_label(df)
    
    # Apply cohort filters
    df = apply_cohort_filters(df)
    
    # Identify features
    numeric_features, categorical_features = identify_features(df)
    
    return df, numeric_features, categorical_features


if __name__ == '__main__':
    # Test the data loader
    df, num_feats, cat_feats = load_data()
    print("\n" + "=" * 80)
    print("DATA LOADING COMPLETE")
    print("=" * 80)
    print(f"Final dataset: {df.shape}")
    print(f"Features: {len(num_feats)} numeric + {len(cat_feats)} categorical = {len(num_feats) + len(cat_feats)} total")
