"""
Model training with Logistic Regression, Random Forest, and XGBoost.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from typing import Dict, Tuple
import joblib


def create_logistic_regression_pipeline(preprocessor) -> Pipeline:
    """
    Create Logistic Regression baseline pipeline.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        
    Returns:
        Pipeline with preprocessor + LogisticRegression
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=5000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        ))
    ])
    
    return pipeline


def create_random_forest_pipeline(preprocessor) -> Pipeline:
    """
    Create Random Forest pipeline.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        
    Returns:
        Pipeline with preprocessor + RandomForestClassifier
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def create_xgboost_pipeline(preprocessor) -> Pipeline:
    """
    Create XGBoost pipeline.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        
    Returns:
        Pipeline with preprocessor + XGBClassifier
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def cross_validate_model(pipeline: Pipeline, 
                         X_train, 
                         y_train, 
                         cv: int = 5) -> Dict:
    """
    Perform cross-validation on a pipeline.
    
    Args:
        pipeline: sklearn Pipeline
        X_train: Training features
        y_train: Training labels
        cv: Number of CV folds
        
    Returns:
        Dict with cross-validation scores
    """
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    return scores


def tune_random_forest(preprocessor, X_train, y_train, n_iter: int = 20) -> Tuple[Pipeline, Dict]:
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.
    
    Args:
        preprocessor: ColumnTransformer
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter settings sampled
        
    Returns:
        Tuple of (best_pipeline, best_params)
    """
    pipeline = create_random_forest_pipeline(preprocessor)
    
    param_distributions = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [5, 10, 15, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nTuning Random Forest...")
    search.fit(X_train, y_train)
    
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def tune_xgboost(preprocessor, X_train, y_train, n_iter: int = 20) -> Tuple[Pipeline, Dict]:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.
    
    Args:
        preprocessor: ColumnTransformer
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter settings sampled
        
    Returns:
        Tuple of (best_pipeline, best_params)
    """
    pipeline = create_xgboost_pipeline(preprocessor)
    
    param_distributions = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__scale_pos_weight': [1, (y_train == 0).sum() / (y_train == 1).sum()]
    }
    
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nTuning XGBoost...")
    search.fit(X_train, y_train)
    
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def train_all_models(preprocessor, X_train, y_train, tune: bool = True) -> Dict[str, Pipeline]:
    """
    Train all models: Logistic Regression, Random Forest, XGBoost.
    
    Args:
        preprocessor: ColumnTransformer
        X_train: Training features
        y_train: Training labels
        tune: Whether to tune RF and XGBoost
        
    Returns:
        Dict mapping model name to trained pipeline
    """
    models = {}
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    # 1. Logistic Regression (baseline, no tuning)
    print("\n1. Training Logistic Regression (baseline)...")
    lr_pipeline = create_logistic_regression_pipeline(preprocessor)
    lr_pipeline.fit(X_train, y_train)
    models['Logistic Regression'] = lr_pipeline
    print("✓ Logistic Regression trained")
    
    # Cross-validate
    lr_scores = cross_validate_model(lr_pipeline, X_train, y_train)
    print(f"  CV ROC-AUC: {lr_scores['test_roc_auc'].mean():.4f} ± {lr_scores['test_roc_auc'].std():.4f}")
    print(f"  CV F1: {lr_scores['test_f1'].mean():.4f} ± {lr_scores['test_f1'].std():.4f}")
    
    # 2. Random Forest
    if tune:
        print("\n2. Training and tuning Random Forest...")
        rf_pipeline, rf_params = tune_random_forest(preprocessor, X_train, y_train, n_iter=20)
        models['Random Forest'] = rf_pipeline
    else:
        print("\n2. Training Random Forest (no tuning)...")
        rf_pipeline = create_random_forest_pipeline(preprocessor)
        rf_pipeline.fit(X_train, y_train)
        models['Random Forest'] = rf_pipeline
        rf_scores = cross_validate_model(rf_pipeline, X_train, y_train)
        print(f"  CV ROC-AUC: {rf_scores['test_roc_auc'].mean():.4f} ± {rf_scores['test_roc_auc'].std():.4f}")
    
    print("✓ Random Forest trained")
    
    # 3. XGBoost
    if tune:
        print("\n3. Training and tuning XGBoost...")
        xgb_pipeline, xgb_params = tune_xgboost(preprocessor, X_train, y_train, n_iter=20)
        models['XGBoost'] = xgb_pipeline
    else:
        print("\n3. Training XGBoost (no tuning)...")
        xgb_pipeline = create_xgboost_pipeline(preprocessor)
        xgb_pipeline.fit(X_train, y_train)
        models['XGBoost'] = xgb_pipeline
        xgb_scores = cross_validate_model(xgb_pipeline, X_train, y_train)
        print(f"  CV ROC-AUC: {xgb_scores['test_roc_auc'].mean():.4f} ± {xgb_scores['test_roc_auc'].std():.4f}")
    
    print("✓ XGBoost trained")
    
    return models


def save_model(model: Pipeline, filepath: str):
    """
    Save a trained model pipeline.
    
    Args:
        model: Trained Pipeline
        filepath: Path to save .joblib file
    """
    joblib.dump(model, filepath)
    print(f"\n✓ Model saved to: {filepath}")


if __name__ == '__main__':
    # Test model training
    from data_loader import load_data
    from preprocessing import split_data, create_preprocessing_pipeline
    
    df, num_feats, cat_feats = load_data()
    X_train, X_test, y_train, y_test = split_data(df, num_feats, cat_feats)
    
    preprocessor = create_preprocessing_pipeline(num_feats, cat_feats)
    models = train_all_models(preprocessor, X_train, y_train, tune=False)
    
    print("\n" + "=" * 80)
    print(f"Trained {len(models)} models successfully")
