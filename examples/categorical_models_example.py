"""
Esempio di utilizzo delle pipeline di preprocessing per modelli diversi.

Questo script dimostra come:
1. Usare la pipeline tradizionale con encoding per modelli classici
2. Usare la pipeline categorical-aware per modelli che supportano dati categorici
3. Confrontare i risultati

Modelli supportati dalla pipeline categorical-aware:
- CatBoost
- TabNet (TabM) 
- LightGBM con parametri categorici
- XGBoost con parametri categorici (limitato)
"""

import pandas as pd
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import (
    run_preprocessing_pipeline,
    run_categorical_preprocessing_pipeline
)
from src.utils import load_config, setup_logger

def compare_preprocessing_pipelines():
    """
    Confronta le due pipeline di preprocessing disponibili.
    """
    # Setup
    config = load_config("config/config.yaml")
    logger = setup_logger("config/config.yaml")
    
    dataset_path = "data/raw/dataset.csv"  # Modifica con il tuo dataset
    target_column = "AI_Prezzo_Ridistribuito"  # Modifica con la tua target
    
    logger.info("🔄 Confronto delle pipeline di preprocessing...")
    
    # ===============================================
    # 1. PIPELINE TRADIZIONALE (con encoding)
    # ===============================================
    logger.info("\n" + "="*50)
    logger.info("🛠️  PIPELINE TRADIZIONALE")
    logger.info("="*50)
    logger.info("Adatta per modelli classici (LinearRegression, RandomForest, SVM, ecc.)")
    
    try:
        traditional_info = run_preprocessing_pipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            config_path="config/config.yaml",
            output_dir="data/processed_traditional"
        )
        
        logger.info("✅ Pipeline tradizionale completata!")
        logger.info(f"📊 Shape finale: {traditional_info['final_shape']}")
        logger.info(f"📂 Files salvati in: data/processed_traditional")
        
        # Mostra le trasformazioni applicate
        if 'encoding' in traditional_info['steps_info']:
            encoding_info = traditional_info['steps_info']['encoding']
            logger.info(f"🔢 Colonne encode: {encoding_info.get('encoded_columns', [])}")
            
    except Exception as e:
        logger.error(f"❌ Errore nella pipeline tradizionale: {e}")
    
    # ===============================================
    # 2. PIPELINE CATEGORICAL-AWARE
    # ===============================================
    logger.info("\n" + "="*50)
    logger.info("🎯 PIPELINE CATEGORICAL-AWARE")
    logger.info("="*50)
    logger.info("Adatta per modelli che supportano dati categorici nativamente")
    
    try:
        categorical_info = run_categorical_preprocessing_pipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            config_path="config/config.yaml",
            output_dir="data/processed_categorical"
        )
        
        logger.info("✅ Pipeline categorical-aware completata!")
        logger.info(f"📊 Shape finale: {categorical_info['final_shape']}")
        logger.info(f"📂 Files salvati in: data/processed_categorical")
        
        # Mostra i tipi di colonne
        logger.info(f"📋 Colonne categoriche: {categorical_info['categorical_columns']}")
        logger.info(f"📊 Colonne numeriche: {categorical_info['numeric_columns']}")
        
    except Exception as e:
        logger.error(f"❌ Errore nella pipeline categorical-aware: {e}")
    
    # ===============================================
    # 3. CONFRONTO RISULTATI
    # ===============================================
    logger.info("\n" + "="*50)
    logger.info("📈 CONFRONTO RISULTATI")
    logger.info("="*50)
    
    try:
        # Carica i dataset processati
        df_traditional = pd.read_parquet("data/processed_traditional/dataset_processed.parquet")
        df_categorical = pd.read_parquet("data/processed_categorical/dataset_categorical.parquet")
        
        logger.info(f"📊 Dataset tradizionale: {df_traditional.shape}")
        logger.info(f"📊 Dataset categorical: {df_categorical.shape}")
        
        # Analizza differenze nelle colonne
        trad_cols = set(df_traditional.columns)
        cat_cols = set(df_categorical.columns)
        
        logger.info(f"🔢 Colonne solo in tradizionale: {trad_cols - cat_cols}")
        logger.info(f"📋 Colonne solo in categorical: {cat_cols - trad_cols}")
        logger.info(f"🤝 Colonne in comune: {len(trad_cols & cat_cols)}")
        
        # Mostra tipi di dati
        logger.info("\n📊 Tipi di dati nel dataset categorical:")
        for col in df_categorical.columns:
            if col != target_column:
                dtype = df_categorical[col].dtype
                if dtype.name == 'category':
                    logger.info(f"  📋 {col}: {dtype} ({df_categorical[col].nunique()} categorie)")
                else:
                    logger.info(f"  🔢 {col}: {dtype}")
                    
    except Exception as e:
        logger.error(f"❌ Errore nel confronto: {e}")

def demonstrate_model_usage():
    """
    Dimostra come usare i dati processati con diversi modelli.
    """
    logger = setup_logger("config/config.yaml").getChild("model_demo")
    
    logger.info("\n" + "="*50)
    logger.info("🤖 ESEMPI DI USO CON MODELLI")
    logger.info("="*50)
    
    # Esempio con CatBoost (usa dati categorici)
    logger.info("\n🎯 Esempio CatBoost (dati categorici):")
    logger.info("""
from catboost import CatBoostRegressor
import pandas as pd

# Carica dati categorici
df = pd.read_parquet("data/processed_categorical/dataset_categorical.parquet")
X = df.drop(columns=['target'])
y = df['target']

# Identifica colonne categoriche
cat_features = [col for col in X.columns if X[col].dtype.name == 'category']

# Addestra CatBoost
model = CatBoostRegressor(
    cat_features=cat_features,
    iterations=1000,
    verbose=False
)
model.fit(X, y)
""")
    
    # Esempio con RandomForest (usa dati encoded)
    logger.info("\n🌲 Esempio RandomForest (dati encoded):")
    logger.info("""
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Carica dati tradizionali (già encoded)
df = pd.read_parquet("data/processed_traditional/dataset_processed.parquet")
X = df.drop(columns=['target'])
y = df['target']

# Addestra RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
""")
    
    # Esempio con LightGBM categorico
    logger.info("\n💡 Esempio LightGBM (dati categorici):")
    logger.info("""
import lightgbm as lgb
import pandas as pd

# Carica dati categorici
df = pd.read_parquet("data/processed_categorical/dataset_categorical.parquet")
X = df.drop(columns=['target'])
y = df['target']

# Identifica colonne categoriche
cat_features = [col for col in X.columns if X[col].dtype.name == 'category']

# Addestra LightGBM
model = lgb.LGBMRegressor(
    categorical_feature=cat_features,
    n_estimators=1000,
    verbosity=-1
)
model.fit(X, y)
""")

if __name__ == "__main__":
    print("🚀 Avvio confronto pipeline di preprocessing...")
    
    try:
        compare_preprocessing_pipelines()
        demonstrate_model_usage()
        
        print("\n" + "="*50)
        print("✅ CONFRONTO COMPLETATO!")
        print("="*50)
        print("""
📝 RACCOMANDAZIONI:

1. 🎯 USA PIPELINE CATEGORICAL-AWARE per:
   - CatBoost
   - TabNet/TabM
   - LightGBM con categorical_feature
   - XGBoost con enable_categorical=True

2. 🛠️  USA PIPELINE TRADIZIONALE per:
   - LinearRegression, Lasso, Ridge
   - RandomForest, ExtraTrees
   - SVM, Neural Networks classici
   - XGBoost standard

3. 📊 VANTAGGI CATEGORICAL-AWARE:
   - Mantiene informazione semantica
   - Riduce dimensionalità
   - Migliori performance su dati categorici
   - Gestione automatica di nuove categorie

4. 📈 VANTAGGI TRADIZIONALE:
   - Compatibilità universale
   - Controllo preciso su encoding
   - Supporto per tutti gli algoritmi sklearn
        """)
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()