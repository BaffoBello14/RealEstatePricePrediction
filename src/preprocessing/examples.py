"""
Esempi di utilizzo delle funzioni riorganizzate.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def example_separate_log_outlier_detection():
    """
    Esempio di utilizzo delle funzioni separate per trasformazione log e outlier detection.
    """
    # Simula dati
    np.random.seed(42)
    n_samples = 1000
    
    # Target con distribuzione log-normale
    y_train = np.random.lognormal(mean=3, sigma=1, size=n_samples)
    
    # Features numeriche
    X_train = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(2, 0.5, n_samples),
        'feature_3': np.random.exponential(1, n_samples)
    })
    
    print("=== ESEMPIO FUNZIONI SEPARATE ===")
    
    # 1. Trasformazione logaritmica separata
    from .transformation import transform_target_log
    y_train_log, transform_info = transform_target_log(pd.Series(y_train))
    
    print(f"Skewness originale: {transform_info['original_skew']:.3f}")
    print(f"Skewness dopo log: {transform_info['log_skew']:.3f}")
    print(f"Miglioramento: {transform_info['skew_improvement']:.3f}")
    
    # 2. Outlier detection separata
    from .cleaning import detect_outliers_multimethod
    outliers_mask, outlier_info = detect_outliers_multimethod(
        y_train_log, X_train, 
        z_threshold=3.0, 
        iqr_multiplier=1.5, 
        min_methods=2
    )
    
    print(f"Outliers trovati: {outlier_info['total_outliers']} ({outlier_info['outlier_percentage']:.2f}%)")
    print(f"Metodi usati: {[method[0] for method in outlier_info['methods_used']]}")
    
    return {
        'y_original': y_train,
        'y_log': y_train_log,
        'X_features': X_train,
        'outliers_mask': outliers_mask,
        'transform_info': transform_info,
        'outlier_info': outlier_info
    }

def example_feature_importance_comparison():
    """
    Esempio di confronto tra metodi di feature importance.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # Simula dati
    np.random.seed(42)
    n_samples, n_features = 500, 10
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    # Target dipendente principalmente dalle prime 3 features
    y = 2*X[:, 0] + 1.5*X[:, 1] - X[:, 2] + 0.1*np.sum(X[:, 3:], axis=1) + np.random.normal(0, 0.1, n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)
    
    print("\n=== ESEMPIO FEATURE IMPORTANCE AVANZATA ===")
    
    # Addestra modelli
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ridge_model = Ridge(alpha=1.0)
    
    rf_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    
    # Prepara best_models structure
    best_models = {
        'rf': {'name': 'Random Forest', 'model': rf_model},
        'ridge': {'name': 'Ridge Regression', 'model': ridge_model}
    }
    
    # Analisi feature importance 
    from ..training.feature_importance import compare_importance_methods
    
    print("Confronto metodi per Random Forest:")
    rf_comparison = compare_importance_methods(
        rf_model, X_train, X_test, y_test, 'Random Forest', feature_names
    )
    print(rf_comparison.head())
    
    print("\nConfronto metodi per Ridge:")
    ridge_comparison = compare_importance_methods(
        ridge_model, X_train, X_test, y_test, 'Ridge', feature_names
    )
    print(ridge_comparison.head())
    
    return {
        'models': best_models,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'rf_comparison': rf_comparison,
        'ridge_comparison': ridge_comparison
    }

def example_pipeline_configuration():
    """
    Esempio di configurazione per usare le funzioni separate nella pipeline.
    """
    
    # Configurazione per usare funzioni separate
    config_separate = {
        'steps': {
            'enable_log_transformation': True,
            'enable_outlier_detection': True,
            'use_separate_log_outlier_functions': True,  # NUOVA OPZIONE!
            'enable_feature_filtering': True,
            'enable_encoding': True,
            'enable_scaling': True
        },
        'z_threshold': 2.5,  # Più permissivo
        'iqr_multiplier': 1.5,
        'min_methods_outlier': 2
    }
    
    # Configurazione tradizionale (funzione combinata)
    config_combined = {
        'steps': {
            'enable_log_transformation': True,
            'enable_outlier_detection': True,
            'use_separate_log_outlier_functions': False,  # Usa funzione combinata
            'enable_feature_filtering': True,
            'enable_encoding': True,
            'enable_scaling': True
        },
        'z_threshold': 3.0,
        'iqr_multiplier': 1.5,
        'min_methods_outlier': 2
    }
    
    print("\n=== CONFIGURAZIONI PIPELINE ===")
    print("Configurazione con funzioni separate:")
    print(f"  use_separate_log_outlier_functions: {config_separate['steps']['use_separate_log_outlier_functions']}")
    print(f"  z_threshold: {config_separate['z_threshold']}")
    
    print("\nConfigurazione tradizionale:")
    print(f"  use_separate_log_outlier_functions: {config_combined['steps']['use_separate_log_outlier_functions']}")
    print(f"  z_threshold: {config_combined['z_threshold']}")
    
    return {
        'config_separate': config_separate,
        'config_combined': config_combined
    }

if __name__ == "__main__":
    print("ESEMPI DI RIORGANIZZAZIONE PREPROCESSING E FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Esempio 1: Funzioni separate
    example_separate_log_outlier_detection()
    
    # Esempio 2: Feature importance avanzata
    example_feature_importance_comparison()
    
    # Esempio 3: Configurazioni pipeline
    example_pipeline_configuration()
    
    print("\n✅ Tutti gli esempi completati!")
    print("\nVANTAGGI DELLA RIORGANIZZAZIONE:")
    print("• Maggiore modularità e riusabilità")
    print("• Flessibilità nell'ordine di applicazione")
    print("• Feature importance con SHAP e metodi avanzati")
    print("• Migliore testabilità delle singole componenti")
    print("• Compatibilità con codice esistente")