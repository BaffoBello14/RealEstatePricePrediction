#!/usr/bin/env python3
"""
Script di test per verificare il supporto delle feature categoriche
nei modelli CatBoost, LightGBM e TabM.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.models import TabMWrapper
import catboost as cb
import lightgbm as lgb

def create_test_dataset():
    """Crea un dataset di test con feature numeriche e categoriche."""
    
    # Crea feature numeriche
    X_num, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    
    # Aggiungi feature categoriche
    np.random.seed(42)
    cat1 = np.random.choice(['A', 'B', 'C'], size=1000)
    cat2 = np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], size=1000)
    cat3 = np.random.choice(['Low', 'Medium', 'High'], size=1000)
    
    # Crea DataFrame
    df = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(5)])
    df['cat1'] = cat1
    df['cat2'] = cat2  
    df['cat3'] = cat3
    df['target'] = y
    
    return df

def test_catboost_with_categorical():
    """Test CatBoost con feature categoriche native."""
    print("🧪 Test CatBoost con feature categoriche...")
    
    df = create_test_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identifica feature categoriche
    categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
    cat_indices = [X.columns.get_loc(col) for col in categorical_features]
    
    # Addestra CatBoost
    model = cb.CatBoostRegressor(
        cat_features=cat_indices,
        iterations=100,
        learning_rate=0.1,
        depth=4,
        logging_level='Silent',
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"   ✅ CatBoost score: {score:.4f}")
    print(f"   📊 Feature categoriche utilizzate: {categorical_features}")
    return score

def test_lightgbm_with_categorical():
    """Test LightGBM con feature categoriche native."""
    print("🧪 Test LightGBM con feature categoriche...")
    
    df = create_test_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identifica feature categoriche
    categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
    
    # Converti le colonne categoriche in dtype category per LightGBM
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    for col in categorical_features:
        X_train_processed[col] = X_train_processed[col].astype('category')
        X_test_processed[col] = X_test_processed[col].astype('category')
    
    # Addestra LightGBM
    model = lgb.LGBMRegressor(
        categorical_feature=categorical_features,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_processed, y_train)
    score = model.score(X_test_processed, y_test)
    
    print(f"   ✅ LightGBM score: {score:.4f}")
    print(f"   📊 Feature categoriche utilizzate: {categorical_features}")
    return score

def test_tabm_wrapper():
    """Test TabM wrapper con feature categoriche."""
    print("🧪 Test TabM wrapper con feature categoriche...")
    
    df = create_test_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identifica feature categoriche
    categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
    
    # Addestra TabM con wrapper
    model = TabMWrapper(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        d_out=1,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        n_num_features=X.shape[1],  # Sarà aggiornato dal wrapper
        cat_cardinalities=[],
        k=2,
        start_scaling_init=1.0  # Aggiunto parametro mancante
    )
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"   ✅ TabM wrapper score: {score:.4f}")
    print(f"   📊 Feature categoriche convertite: {categorical_features}")
    print(f"   🔧 Label encoders creati: {list(model.label_encoders.keys())}")
    return score

def test_comparison():
    """Confronta le prestazioni dei modelli con feature categoriche."""
    print("\n" + "="*60)
    print("📊 CONFRONTO PRESTAZIONI CON FEATURE CATEGORICHE")
    print("="*60)
    
    scores = {}
    
    try:
        scores['CatBoost'] = test_catboost_with_categorical()
    except Exception as e:
        print(f"   ❌ CatBoost fallito: {e}")
        scores['CatBoost'] = None
    
    print()
    
    try:
        scores['LightGBM'] = test_lightgbm_with_categorical()
    except Exception as e:
        print(f"   ❌ LightGBM fallito: {e}")
        scores['LightGBM'] = None
    
    print()
    
    try:
        scores['TabM_Wrapper'] = test_tabm_wrapper()
    except Exception as e:
        print(f"   ❌ TabM Wrapper fallito: {e}")
        scores['TabM_Wrapper'] = None
    
    print("\n" + "="*60)
    print("📈 RISULTATI FINALI")
    print("="*60)
    
    for model_name, score in scores.items():
        if score is not None:
            print(f"🏆 {model_name}: {score:.4f}")
        else:
            print(f"❌ {model_name}: Fallito")
    
    # Trova il migliore
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    if valid_scores:
        best_model = max(valid_scores.items(), key=lambda x: x[1])
        print(f"\n🥇 Miglior modello: {best_model[0]} (score: {best_model[1]:.4f})")

if __name__ == "__main__":
    print("🚀 Test supporto feature categoriche per CatBoost, LightGBM e TabM")
    print("="*70)
    
    test_comparison()
    
    print("\n✅ Test completati!")