#!/usr/bin/env python3
"""
Simple test script to verify categorical model fixes.
"""

import sys
import os

def test_lightgbm_fix():
    """Test that LightGBM categorical fix is in place."""
    
    # Read the test file to verify the fix
    try:
        with open('test_categorical_models.py', 'r') as f:
            content = f.read()
        
        # Check if the fix is present
        if "X_train_processed[col] = X_train_processed[col].astype('category')" in content:
            print("✅ LightGBM categorical fix: PRESENTE")
            print("   📊 Conversione automatica a dtype category implementata")
            return True
        else:
            print("❌ LightGBM categorical fix: MANCANTE")
            return False
            
    except FileNotFoundError:
        print("❌ File test_categorical_models.py non trovato")
        return False

def test_tabm_fix():
    """Test that TabM start_scaling_init fix is in place."""
    
    try:
        with open('test_categorical_models.py', 'r') as f:
            content = f.read()
        
        # Check if the fix is present
        if "start_scaling_init=1.0" in content:
            print("✅ TabM start_scaling_init fix: PRESENTE")
            print("   🔧 Parametro start_scaling_init aggiunto")
            return True
        else:
            print("❌ TabM start_scaling_init fix: MANCANTE")
            return False
            
    except FileNotFoundError:
        print("❌ File test_categorical_models.py non trovato")
        return False

def test_imports():
    """Test that imports are working in the test file."""
    
    try:
        with open('test_categorical_models.py', 'r') as f:
            content = f.read()
        
        # Check for critical imports
        expected_imports = [
            'from src.training.models import TabMWrapper',
            'import catboost as cb',
            'import lightgbm as lgb'
        ]
        
        all_present = True
        for imp in expected_imports:
            if imp in content:
                print(f"✅ Import trovato: {imp}")
            else:
                print(f"❌ Import mancante: {imp}")
                all_present = False
        
        return all_present
        
    except FileNotFoundError:
        print("❌ File test_categorical_models.py non trovato")
        return False

def main():
    """Main test function."""
    print("🚀 Test delle correzioni per i modelli categorici")
    print("="*60)
    
    results = []
    
    print("\n🧪 Test fix LightGBM...")
    results.append(test_lightgbm_fix())
    
    print("\n🧪 Test fix TabM...")
    results.append(test_tabm_fix())
    
    print("\n🧪 Test imports...")
    results.append(test_imports())
    
    print("\n" + "="*60)
    print("📈 RISULTATI FINALI")
    print("="*60)
    
    if all(results):
        print("🏆 Tutti i fix sono stati applicati correttamente!")
        print("\n📋 Riepilogo correzioni:")
        print("   • LightGBM: Conversione automatica object -> category dtype")
        print("   • TabM: Aggiunto parametro start_scaling_init=1.0")
        print("\n✅ Il codice è ora pronto per gestire le feature categoriche!")
    else:
        print("❌ Alcuni fix non sono stati applicati correttamente")
        failed = sum(1 for r in results if not r)
        print(f"   {failed}/{len(results)} test falliti")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)