#!/usr/bin/env python3
"""
Script di test per verificare le nuove funzionalità di configurazione:
1. Optuna pruning percentile
2. Outlier detection strategy 
3. Stratified split
"""

import yaml
import sys
from pathlib import Path

def test_config_parameters():
    """Test che verifica che i parametri di configurazione siano presenti e coerenti."""
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ File config.yaml non trovato")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Optuna pruning percentile
    print("🧪 Test 1: Parametro optuna_pruning_percentile")
    pruning_percentile = config.get('training', {}).get('optuna_pruning_percentile')
    if pruning_percentile is not None:
        print(f"✅ optuna_pruning_percentile trovato: {pruning_percentile}")
        if 0 <= pruning_percentile <= 100:
            print(f"✅ Valore valido: {pruning_percentile}")
        else:
            print(f"❌ Valore non valido: {pruning_percentile} (deve essere 0-100)")
            return False
    else:
        print("❌ optuna_pruning_percentile non trovato nel config")
        return False
    
    # Test 2: Outlier detection strategy
    print("\n🧪 Test 2: Parametri outlier detection strategy")
    preprocessing = config.get('preprocessing', {})
    
    outlier_strategy = preprocessing.get('outlier_strategy')
    category_column = preprocessing.get('category_column')
    alternative_category = preprocessing.get('alternative_category_column')
    
    if outlier_strategy:
        print(f"✅ outlier_strategy trovato: {outlier_strategy}")
        if outlier_strategy in ['global', 'category_stratified']:
            print(f"✅ Strategia valida: {outlier_strategy}")
        else:
            print(f"❌ Strategia non valida: {outlier_strategy}")
            return False
    else:
        print("❌ outlier_strategy non trovato")
        return False
    
    if category_column:
        print(f"✅ category_column trovato: {category_column}")
    else:
        print("❌ category_column non trovato")
        return False
    
    if alternative_category:
        print(f"✅ alternative_category_column trovato: {alternative_category}")
    else:
        print("❌ alternative_category_column non trovato")
        return False
    
    # Test 3: Stratified split
    print("\n🧪 Test 3: Parametri stratified split")
    use_stratified = preprocessing.get('use_stratified_split')
    quantiles = preprocessing.get('stratification_quantiles')
    
    if use_stratified is not None:
        print(f"✅ use_stratified_split trovato: {use_stratified}")
    else:
        print("❌ use_stratified_split non trovato")
        return False
    
    if quantiles:
        print(f"✅ stratification_quantiles trovato: {quantiles}")
        if quantiles > 1:
            print(f"✅ Valore valido: {quantiles}")
        else:
            print(f"❌ Valore non valido: {quantiles} (deve essere > 1)")
            return False
    else:
        print("❌ stratification_quantiles non trovato")
        return False
    
    return True

def test_import_functions():
    """Test che verifica che le funzioni possano essere importate."""
    
    print("\n🧪 Test 4: Import delle funzioni")
    
    try:
        # Test import pruner
        from optuna.pruners import PercentilePruner, MedianPruner
        print("✅ Pruner imports: OK")
    except ImportError as e:
        print(f"❌ Errore import pruner: {e}")
        return False
    
    try:
        # Test import outlier detection
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.preprocessing.cleaning import transform_target_and_detect_outliers_by_category
        print("✅ Outlier detection import: OK")
    except ImportError as e:
        print(f"❌ Errore import outlier detection: {e}")
        return False
    
    try:
        # Test import transformation
        from src.preprocessing.transformation import split_dataset_with_validation
        print("✅ Transformation import: OK")
    except ImportError as e:
        print(f"❌ Errore import transformation: {e}")
        return False
    
    return True

def test_config_consistency():
    """Test di coerenza della configurazione."""
    
    print("\n🧪 Test 5: Coerenza configurazione")
    
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessing = config.get('preprocessing', {})
    
    # Test coerenza temporal vs stratified
    use_temporal = preprocessing.get('use_temporal_split', True)
    use_stratified = preprocessing.get('use_stratified_split', False)
    
    if use_temporal and use_stratified:
        print("⚠️ ATTENZIONE: use_temporal_split=True e use_stratified_split=True")
        print("   Stratified split sarà ignorato (incompatibile con temporal)")
    else:
        print("✅ Configurazione split coerente")
    
    # Test coerenza outlier strategy
    outlier_strategy = preprocessing.get('outlier_strategy', 'global')
    if outlier_strategy == 'category_stratified':
        enable_outliers = preprocessing.get('steps', {}).get('enable_outlier_detection', True)
        if not enable_outliers:
            print("⚠️ ATTENZIONE: outlier_strategy=category_stratified ma outlier detection disabilitato")
        else:
            print("✅ Configurazione outlier detection coerente")
    
    return True

if __name__ == "__main__":
    print("🚀 Test nuove funzionalità di configurazione")
    print("="*60)
    
    results = []
    
    # Esegui tutti i test
    results.append(test_config_parameters())
    results.append(test_import_functions())
    results.append(test_config_consistency())
    
    print("\n" + "="*60)
    print("📈 RISULTATI FINALI")
    print("="*60)
    
    if all(results):
        print("🏆 Tutti i test sono passati!")
        print("\n📋 Funzionalità verificate:")
        print("   ✅ Optuna pruning percentile configurabile")
        print("   ✅ Outlier detection strategy (global/category_stratified)")
        print("   ✅ Stratified split per target continuo")
        print("   ✅ Import delle funzioni necessarie")
        print("   ✅ Coerenza della configurazione")
    else:
        print("❌ Alcuni test sono falliti")
    
    print(f"\nRisultato: {sum(results)}/{len(results)} test passati")