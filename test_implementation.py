#!/usr/bin/env python3
"""
Script di test per verificare l'implementazione dei moduli di training e evaluation.
"""

import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Testa gli import dei nuovi moduli."""
    print("=== TEST IMPORTS ===")
    
    try:
        from src.training.models import get_baseline_models, create_ensemble_models
        print("✓ Modulo models importato con successo")
        
        from src.training.tuning import run_full_optimization
        print("✓ Modulo tuning importato con successo")
        
        from src.training.train import run_training_pipeline
        print("✓ Modulo train importato con successo")
        
        from src.training.evaluation import run_evaluation_pipeline
        print("✓ Modulo evaluation importato con successo")
        
        from src.preprocessing.transformation import split_dataset_with_validation
        print("✓ Funzione split_dataset_with_validation importata con successo")
        
        print("✅ Tutti gli import sono riusciti!")
        return True
        
    except Exception as e:
        print(f"❌ Errore negli import: {e}")
        return False

def test_baseline_models():
    """Testa la creazione di modelli baseline."""
    print("\n=== TEST BASELINE MODELS ===")
    
    try:
        from src.training.models import get_baseline_models
        
        models = get_baseline_models(random_state=42)
        print(f"✓ Creati {len(models)} modelli baseline:")
        for name in models.keys():
            print(f"  - {name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Errore nella creazione modelli baseline: {e}")
        return False

def test_config_loading():
    """Testa il caricamento della configurazione aggiornata."""
    print("\n=== TEST CONFIG LOADING ===")
    
    try:
        from src.utils.io import load_config
        
        config = load_config('config/config.yaml')
        
        # Verifica che ci siano i nuovi parametri
        assert 'training' in config, "Sezione training mancante"
        assert 'val_size' in config['preprocessing'], "Parametro val_size mancante"
        
        training_config = config['training']
        required_params = ['cv_folds', 'n_trials', 'optuna_timeout', 'n_jobs', 'random_state']
        
        for param in required_params:
            assert param in training_config, f"Parametro {param} mancante in training config"
            
        print("✓ Configurazione caricata e validata con successo")
        print(f"  - CV folds: {training_config['cv_folds']}")
        print(f"  - N trials: {training_config['n_trials']}")
        print(f"  - Val size: {config['preprocessing']['val_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel caricamento config: {e}")
        return False

def main():
    """Funzione principale di test."""
    print("🧪 TESTING IMPLEMENTAZIONE TRAINING & EVALUATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_baseline_models,
        test_config_loading
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 TUTTI I TEST PASSATI! ({passed}/{total})")
        print("\n✅ L'implementazione sembra funzionare correttamente!")
        print("\nPer testare la pipeline completa, esegui:")
        print("  python main.py --steps preprocessing training evaluation")
    else:
        print(f"⚠️  ALCUNI TEST FALLITI: {passed}/{total}")
        print("\n❌ Ci sono problemi nell'implementazione da risolvere.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)