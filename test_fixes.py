#!/usr/bin/env python3
"""
Script di test per verificare le correzioni dei problemi di data leakage.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logger, get_logger
from src.preprocessing.transformation import split_dataset_with_validation
from src.preprocessing.encoding import encode_features
from src.utils.validation import validate_temporal_split, check_target_leakage

def create_test_data(n_samples=1000):
    """Crea dataset di test con date temporali corrette."""
    
    # Date temporali simulate
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i*30) for i in range(n_samples)]
    
    # Target correlato al tempo (simulazione trend immobiliare)
    base_price = 200000
    time_trend = np.linspace(0, 50000, n_samples)  # Trend crescente
    noise = np.random.normal(0, 10000, n_samples)
    target = base_price + time_trend + noise
    target_log = np.log1p(target)
    
    # Features
    data = {
        'A_DataStipula': dates,
        'AI_Superficie': np.random.normal(100, 30, n_samples),
        'AI_Piano': np.random.choice(['T', '1', '2', '3', '4', '5'], n_samples),
        'AI_IdCategoriaCatastale': np.random.choice(['A2', 'A3', 'A4'], n_samples),
        'PC_Zona': np.random.choice(['Centro', 'Periferia', 'Semicentro'], n_samples, p=[0.3, 0.4, 0.3]),
        'Feature_Numerica': np.random.normal(0, 1, n_samples),
        'Feature_Categorica_High': [f'Cat_{i%50}' for i in range(n_samples)],
        'AI_Prezzo_Ridistribuito': target_log,
        'AI_Prezzo_Original': target,
        # Feature sospetta per test leakage
        'Prezzo_Derived': target_log * 0.95 + np.random.normal(0, 0.1, n_samples)
    }
    
    return pd.DataFrame(data)

def test_temporal_split():
    """Test dello split temporale."""
    print("\n=== TEST SPLIT TEMPORALE ===")
    
    df = create_test_data(1000)
    
    # Test split temporale
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = split_dataset_with_validation(
        df, 
        target_column='AI_Prezzo_Ridistribuito',
        test_size=0.2,
        val_size=0.2,
        use_temporal_split=True,
        date_column='A_DataStipula'
    )
    
    print(f"Train dates: {X_train['A_DataStipula'].min()} to {X_train['A_DataStipula'].max()}")
    print(f"Val dates: {X_val['A_DataStipula'].min()} to {X_val['A_DataStipula'].max()}")
    print(f"Test dates: {X_test['A_DataStipula'].min()} to {X_test['A_DataStipula'].max()}")
    
    # Validazione
    validation_result = validate_temporal_split(X_train, X_val, X_test)
    
    if validation_result['valid']:
        print("✓ Split temporale VALIDO")
    else:
        print("✗ Split temporale INVALIDO")
        for error in validation_result['errors']:
            print(f"  Errore: {error}")
    
    return validation_result['valid']

def test_target_encoding_no_leakage():
    """Test che il target encoding non causi leakage."""
    print("\n=== TEST TARGET ENCODING (NO LEAKAGE) ===")
    
    df = create_test_data(1000)
    
    # Test encoding SENZA target encoding (modalità corretta)
    df_encoded_safe, info_safe = encode_features(
        df.copy(),
        target_column='AI_Prezzo_Ridistribuito',
        apply_target_encoding=False  # SICURO
    )
    
    # Test encoding CON target encoding (modalità pericolosa)
    df_encoded_leak, info_leak = encode_features(
        df.copy(),
        target_column='AI_Prezzo_Ridistribuito',
        apply_target_encoding=True  # PERICOLOSO pre-split
    )
    
    print(f"Encoding sicuro - Features finali: {df_encoded_safe.shape[1]}")
    print(f"Encoding con leakage - Features finali: {df_encoded_leak.shape[1]}")
    
    # Verifica che modalità sicura non abbia target encoding
    safe_has_target_encoding = 'target_encoding' in info_safe
    leak_has_target_encoding = 'target_encoding' in info_leak
    
    print(f"Modalità sicura ha target encoding: {safe_has_target_encoding}")
    print(f"Modalità con leakage ha target encoding: {leak_has_target_encoding}")
    
    success = not safe_has_target_encoding
    if success:
        print("✓ Target encoding correttamente DISABILITATO in modalità sicura")
    else:
        print("✗ Target encoding non correttamente gestito")
    
    return success

def test_leakage_detection():
    """Test del rilevamento di target leakage."""
    print("\n=== TEST RILEVAMENTO TARGET LEAKAGE ===")
    
    df = create_test_data(500)
    
    # Split per avere dati di training
    X_train, _, _, y_train, _, _, _, _, _ = split_dataset_with_validation(
        df, 
        target_column='AI_Prezzo_Ridistribuito',
        use_temporal_split=False  # Per semplicità
    )
    
    # Test rilevamento leakage
    leakage_results = check_target_leakage(
        X_train, y_train, 
        target_column='AI_Prezzo_Ridistribuito'
    )
    
    print(f"Leakage rilevato: {leakage_results['leakage_detected']}")
    print(f"Features sospette: {len(leakage_results['suspicious_features'])}")
    
    for feature_info in leakage_results['suspicious_features']:
        print(f"  - {feature_info['feature']}: {feature_info['reason']}")
    
    # Dovrebbe rilevare 'Prezzo_Derived' come leakage
    expected_leakage = any(
        'Prezzo_Derived' in f['feature'] for f in leakage_results['suspicious_features']
    )
    
    if expected_leakage:
        print("✓ Target leakage correttamente RILEVATO")
    else:
        print("✗ Target leakage NON rilevato (problema nel detection)")
    
    return expected_leakage

def test_pca_variance_handling():
    """Test della gestione corretta della variance threshold in PCA."""
    print("\n=== TEST GESTIONE PCA VARIANCE ===")
    
    from src.preprocessing.transformation import apply_pca
    
    # Crea dati di test
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    X_test = np.random.randn(50, 20)
    
    # Test con variance threshold come float (percentuale)
    try:
        results_float = apply_pca(X_train, X_test, variance_threshold=0.95)
        X_train_pca_float = results_float[0]
        print(f"✓ PCA con variance 0.95: {X_train.shape[1]} -> {X_train_pca_float.shape[1]} componenti")
        success_float = True
    except Exception as e:
        print(f"✗ Errore PCA con variance float: {e}")
        success_float = False
    
    # Test con variance threshold come int (numero componenti)
    try:
        results_int = apply_pca(X_train, X_test, variance_threshold=10)
        X_train_pca_int = results_int[0]
        print(f"✓ PCA con 10 componenti: {X_train.shape[1]} -> {X_train_pca_int.shape[1]} componenti")
        success_int = X_train_pca_int.shape[1] == 10
        if success_int:
            print("✓ Numero componenti corretto")
        else:
            print(f"✗ Numero componenti errato: atteso 10, ottenuto {X_train_pca_int.shape[1]}")
    except Exception as e:
        print(f"✗ Errore PCA con int: {e}")
        success_int = False
    
    return success_float and success_int

def run_all_tests():
    """Esegue tutti i test di verifica."""
    
    # Setup logger
    logger = setup_logger('config/config.yaml')
    
    print("="*60)
    print("VERIFICA CORREZIONI DATA LEAKAGE")
    print("="*60)
    
    results = {}
    
    # Test 1: Split temporale
    results['temporal_split'] = test_temporal_split()
    
    # Test 2: Target encoding sicuro
    results['safe_target_encoding'] = test_target_encoding_no_leakage()
    
    # Test 3: Rilevamento leakage
    results['leakage_detection'] = test_leakage_detection()
    
    # Test 4: PCA variance handling
    results['pca_handling'] = test_pca_variance_handling()
    
    # Risultati finali
    print("\n" + "="*60)
    print("RISULTATI FINALI")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print("-"*60)
    if all_passed:
        print("🎉 TUTTI I TEST PASSATI - Correzioni implementate con successo!")
    else:
        print("⚠️  ALCUNI TEST FALLITI - Verificare le implementazioni")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)