# Guida Completa alle Pipeline di Preprocessing

## Overview

Il sistema di preprocessing ora supporta **due approcci distinti** per adattarsi a diversi tipi di modelli di machine learning:

1. **Pipeline Tradizionale**: Converte dati categorici tramite encoding per modelli classici
2. **Pipeline Categorical-Aware**: Mantiene dati categorici nativi per modelli che li supportano

## Problemi Risolti

### ✅ Funzioni Deprecate Rimosse
- ❌ `convert_to_numeric()` → ✅ `convert_to_numeric_unified()`
- ❌ `clean_data()` → ✅ `clean_dataframe_unified()`
- ❌ `auto_convert_to_numeric()` → ✅ `convert_to_numeric_unified()`

### ✅ Logger Duplicato Corretto
- Modificata configurazione del logger per evitare stampe duplicate
- Aggiunta gestione degli handler per pulire la configurazione

### ✅ Import Error Risolto
- Aggiunta funzione `save_config()` mancante in `utils/io.py`
- Corretti tutti gli import per utilizzare le nuove funzioni unificate

## Pipeline Disponibili

### 1. Pipeline Tradizionale

```python
from src.preprocessing import run_preprocessing_pipeline

# Per modelli classici (sklearn, ecc.)
info = run_preprocessing_pipeline(
    dataset_path="data/raw/dataset.csv",
    target_column="target",
    config_path="config/config.yaml",
    output_dir="data/processed_traditional"
)
```

**Adatta per:**
- LinearRegression, Lasso, Ridge
- RandomForest, ExtraTrees 
- SVM, Neural Networks classici
- XGBoost standard
- Tutti i modelli sklearn

**Caratteristiche:**
- ✅ Encoding automatico delle categoriche (OneHot, Target, etc.)
- ✅ Scaling delle feature numeriche
- ✅ Compatibilità universale
- ✅ Controllo preciso delle trasformazioni

### 2. Pipeline Categorical-Aware

```python
from src.preprocessing import run_categorical_preprocessing_pipeline

# Per modelli che supportano categoriche native
info = run_categorical_preprocessing_pipeline(
    dataset_path="data/raw/dataset.csv",
    target_column="target",
    config_path="config/config.yaml",
    output_dir="data/processed_categorical"
)
```

**Adatta per:**
- **CatBoost** ⭐
- **TabNet/TabM** ⭐
- **LightGBM** con `categorical_feature`
- **XGBoost** con `enable_categorical=True`

**Caratteristiche:**
- ✅ Mantiene informazione semantica delle categoriche
- ✅ Riduce dimensionalità (no encoding esplosivo)
- ✅ Migliori performance su dati categorici
- ✅ Gestione automatica di nuove categorie
- ✅ Identificazione intelligente tipo-colonna

## Test di Cramér's V

### Come Funziona

Il test di Cramér's V analizza le correlazioni tra variabili categoriche:

```python
# Soglia default: 0.95
cramer_threshold = 0.95

# Se Cramér's V > soglia → rimuove colonna
if cramers_v(col1, col2) > cramer_threshold:
    remove_column(col2)  # Rimuove la seconda colonna
```

### Esempio Pratico

```python
# Colonne: ['city', 'postal_code', 'target']
# Se city e postal_code hanno Cramér's V = 0.97
# → postal_code viene rimossa (ridondante con city)
```

**Risposta alla domanda**: **Sì, le colonne categoriche con correlazione più alta della soglia vengono rimosse** per evitare ridondanza.

## Esempi di Utilizzo

### CatBoost con Dati Categorici

```python
from catboost import CatBoostRegressor
import pandas as pd

# 1. Preprocess con pipeline categorical-aware
from src.preprocessing import run_categorical_preprocessing_pipeline

info = run_categorical_preprocessing_pipeline(
    dataset_path="data/raw/dataset.csv",
    target_column="price",
    output_dir="data/processed_categorical"
)

# 2. Carica dati processati
df = pd.read_parquet("data/processed_categorical/dataset_categorical.parquet")
X = df.drop(columns=['price'])
y = df['price']

# 3. Identifica features categoriche
cat_features = [col for col in X.columns if X[col].dtype.name == 'category']
print(f"Categorical features: {cat_features}")

# 4. Addestra CatBoost
model = CatBoostRegressor(
    cat_features=cat_features,
    iterations=1000,
    learning_rate=0.1,
    verbose=False
)

model.fit(X, y)
```

### RandomForest con Dati Encoded

```python
from sklearn.ensemble import RandomForestRegressor

# 1. Preprocess con pipeline tradizionale
from src.preprocessing import run_preprocessing_pipeline

info = run_preprocessing_pipeline(
    dataset_path="data/raw/dataset.csv",
    target_column="price",
    output_dir="data/processed_traditional"
)

# 2. Carica dati processati (già encoded)
df = pd.read_parquet("data/processed_traditional/dataset_processed.parquet")
X = df.drop(columns=['price'])
y = df['price']

# 3. Addestra RandomForest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)
```

### LightGBM con Supporto Categorico

```python
import lightgbm as lgb

# 1. Usa pipeline categorical-aware
df = pd.read_parquet("data/processed_categorical/dataset_categorical.parquet")
X = df.drop(columns=['price'])
y = df['price']

# 2. Identifica categoriche
cat_features = [col for col in X.columns if X[col].dtype.name == 'category']

# 3. Addestra LightGBM
model = lgb.LGBMRegressor(
    categorical_feature=cat_features,
    n_estimators=1000,
    verbosity=-1
)

model.fit(X, y)
```

## Configurazione

### Config YAML Esempio

```yaml
preprocessing:
  steps:
    enable_cramers_analysis: true
    enable_auto_numeric_conversion: true
    enable_specific_columns_removal: true
    enable_outlier_detection: true
    
  # Soglie
  cramer_threshold: 0.95
  auto_numeric_threshold: 0.8
  constant_column_threshold: 0.95
  
  # Colonne da rimuovere
  columns_to_remove:
    - "id"
    - "timestamp"
    
  # Configurazione encoding (solo pipeline tradizionale)
  encoding:
    low_cardinality_threshold: 10
    high_cardinality_max: 100
    apply_target_encoding: true

logging:
  level: INFO
  file: logs/preprocessing.log
```

## Confronto delle Pipeline

| Aspetto | Pipeline Tradizionale | Pipeline Categorical-Aware |
|---------|----------------------|---------------------------|
| **Compatibilità** | ✅ Universale | ⚠️ Solo modelli specifici |
| **Performance** | ⚠️ Buona | ✅ Ottima su categoriche |
| **Dimensionalità** | ⚠️ Può esplodere | ✅ Mantiene compatta |
| **Interpretabilità** | ⚠️ Difficile post-encoding | ✅ Mantiene significato |
| **Nuove categorie** | ❌ Problemi in produzione | ✅ Gestione automatica |
| **Controllo** | ✅ Controllo fine | ⚠️ Limitato |

## Raccomandazioni

### ✅ Usa Pipeline Categorical-Aware quando:
- Hai molte features categoriche ad alta cardinalità
- Usi CatBoost, TabNet, LightGBM
- Vuoi mantenere interpretabilità
- Prevedi nuove categorie in produzione

### ✅ Usa Pipeline Tradizionale quando:
- Usi modelli sklearn classici
- Hai poche features categoriche
- Vuoi controllo fine sull'encoding
- Hai pipeline esistenti da mantenere

## Script di Confronto

Per confrontare entrambe le pipeline sul tuo dataset:

```bash
python examples/categorical_models_example.py
```

Questo script:
1. Esegue entrambe le pipeline
2. Compara i risultati
3. Mostra esempi di utilizzo con diversi modelli
4. Fornisce raccomandazioni specifiche

## Troubleshooting

### Import Error
```python
# ❌ Vecchio (deprecato)
from src.preprocessing import convert_to_numeric, clean_data

# ✅ Nuovo (corretto)
from src.preprocessing.data_cleaning_core import convert_to_numeric_unified, clean_dataframe_unified
```

### Logger Duplicato
Se vedi messaggi duplicati, assicurati di chiamare `setup_logger()` solo una volta all'inizio dell'applicazione.

### Modelli Non Supportati
Se un modello non supporta dati categorici nativamente, usa sempre la pipeline tradizionale.

## Contribuire

Per aggiungere supporto a nuovi modelli categorici:

1. Testa il modello con la pipeline categorical-aware
2. Aggiorna la documentazione in questo file
3. Aggiungi esempi in `examples/categorical_models_example.py`
4. Aggiorna i test in `tests/`