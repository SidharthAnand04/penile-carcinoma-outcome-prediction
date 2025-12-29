"""
Preprocessing pipeline for SEER data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing.
    
    Numeric pipeline:
        - Impute with median
        - StandardScaler
    
    Categorical pipeline:
        - Impute with most frequent
        - OneHotEncoder with unknown handling
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        
    Returns:
        ColumnTransformer
    """
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def preprocess_tumor_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess tumor size column - convert to numeric and handle special codes.
    
    SEER codes like 999 = unknown, we'll convert to NaN for imputation.
    
    Args:
        df: DataFrame with 'Tumor Size Summary (2016+)' column
        
    Returns:
        DataFrame with processed tumor size
    """
    df = df.copy()
    
    if 'Tumor Size Summary (2016+)' in df.columns:
        # Convert to string first, then numeric
        size_col = df['Tumor Size Summary (2016+)'].astype(str)
        
        # Replace special codes with NaN
        size_col = size_col.replace(['999', 'Blank(s)', 'Unknown', ''], np.nan)
        
        # Convert to numeric
        df['Tumor Size Summary (2016+)'] = pd.to_numeric(size_col, errors='coerce')
    
    return df


def split_data(df: pd.DataFrame, 
               numeric_features: list, 
               categorical_features: list,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple:
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: DataFrame with features and target
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Preprocess special columns
    df = preprocess_tumor_size(df)
    
    # Separate features and target
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df['lymph_node_metastasis'].copy()
    
    # Replace 'Blank(s)' strings with NaN for proper imputation
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].replace('Blank(s)', np.nan)
            X[col] = X[col].replace('Unknown', np.nan)
            X[col] = X[col].replace('UNK Staging', np.nan)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print("\n" + "=" * 80)
    print("DATA SPLITTING")
    print("=" * 80)
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"  Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train):.1%})")
    print(f"  Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train):.1%})")
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"  Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test):.1%})")
    print(f"  Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test):.1%})")
    
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer, 
                     numeric_features: list, 
                     categorical_features: list) -> list:
    """
    Get feature names after one-hot encoding.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numeric_features: Original numeric feature names
        categorical_features: Original categorical feature names
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Numeric features (no change)
    feature_names.extend(numeric_features)
    
    # Categorical features (get from encoder)
    try:
        # Try to get from fitted encoder
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out()
        feature_names.extend(cat_feature_names)
    except:
        # Fallback: manually construct based on what was actually fit
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        for i, cat in enumerate(cat_encoder.categories_):
            col_name = f"cat{i}" if not hasattr(cat_encoder, 'feature_names_in_') else cat_encoder.feature_names_in_[i]
            for val in cat:
                feature_names.append(f"{col_name}_{val}")
    
    return feature_names


if __name__ == '__main__':
    # Test preprocessing
    from data_loader import load_data
    
    df, num_feats, cat_feats = load_data()
    X_train, X_test, y_train, y_test = split_data(df, num_feats, cat_feats)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTransformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, num_feats, cat_feats)
    print(f"\nTotal features after encoding: {len(feature_names)}")
