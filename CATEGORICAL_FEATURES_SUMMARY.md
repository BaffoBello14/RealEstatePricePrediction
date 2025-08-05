# Supporto Feature Categoriche per CatBoost e LightGBM

## 📋 Riepilogo delle Modifiche

Questo documento descrive le modifiche implementate per permettere a **CatBoost** e **LightGBM** di utilizzare feature categoriche native invece del dataset completamente encoded.

## 🎯 Obiettivo

Migliorare le prestazioni dei modelli di machine learning permettendo loro di sfruttare le feature categoriche nella loro forma nativa, senza dover ricorrere esclusivamente all'encoding pre-processamento.

## 🔧 Modifiche Implementate

### 1. Pipeline di Preprocessing (`src/preprocessing/pipeline.py`)

#### Nuove Funzionalità:
- **Identificazione automatica feature categoriche**: Nuova funzione `identify_categorical_columns()` che classifica le colonne per tipo e cardinalità
- **Salvataggio dataset paralleli**: Il preprocessing ora salva due versioni dei dati:
  - Dataset encoded (per modelli tradizionali)
  - Dataset con feature categoriche native (per CatBoost/LightGBM/TabM)
- **Split sincrono**: Entrambe le versioni subiscono lo stesso split temporale per garantire consistenza

#### File Aggiunti:
```
data/processed/
├── X_train.parquet              # Dataset encoded
├── X_val.parquet 
├── X_test.parquet
├── X_train_categorical.parquet  # Dataset con categoriche native
├── X_val_categorical.parquet
├── X_test_categorical.parquet
└── ... (altri file target)
```

### 2. Gestione Path (`src/utils/pipeline_utils.py`)

#### Aggiornamenti:
- **Nuovi path configurati**: Aggiunti path per i file categorici nella funzione `get_preprocessing_paths()`
- **Compatibilità backward**: Mantenuti tutti i path esistenti per evitare breaking changes

### 3. Pipeline di Training (`src/training/train.py`)

#### Nuove Funzionalità:
- **Caricamento intelligente**: Nuova funzione `load_data_for_model_type()` che carica il dataset appropriato per ogni modello
- **Training differenziato**: Modificata `run_training_pipeline()` per:
  - Usare dati encoded per modelli baseline e standard
  - Usare dati categorici per CatBoost, LightGBM
  - Ottimizzazione separata per modelli categorici
- **Valutazione appropriata**: Nuova funzione `evaluate_all_models_with_appropriate_data()` che usa i dati corretti per ogni modello

### 4. Modelli ML (`src/training/models.py`)

#### CatBoost:
- **Feature categoriche automatiche**: Identificazione e configurazione automatica delle `cat_features`
- **Logging migliorato**: Report delle feature categoriche utilizzate

#### LightGBM:
- **Gestione categoriche native**: Configurazione automatica del parametro `categorical_feature`
- **Cross-validation manuale**: Implementata CV personalizzata per gestire le feature categoriche



## 📊 Vantaggi del Nuovo Approccio

### 🎯 **CatBoost**
- ✅ Sfrutta il supporto nativo per categoriche ad alta cardinalità
- ✅ Evita overfitting da one-hot encoding su categoriche rare
- ✅ Gestione automatica di categorie mancanti

### 🎯 **LightGBM**  
- ✅ Utilizza algoritmi ottimizzati per feature categoriche
- ✅ Migliore gestione della memoria con categoriche
- ✅ Split ottimali per variabili categoriche



## 🔄 Backward Compatibility

- ✅ **Modelli esistenti**: Continuano a funzionare con dataset encoded
- ✅ **API inalterata**: Nessuna modifica alle interfacce pubbliche
- ✅ **Configurazione**: Stesso file `config.yaml` senza modifiche

## 🧪 Testing

Creato script di test `test_categorical_models.py` che:
- Genera dataset sintetico con feature numeriche e categoriche
- Testa entrambi i modelli con le nuove funzionalità
- Confronta le prestazioni e verifica il corretto funzionamento

### Esecuzione Test:
```bash
python test_categorical_models.py
```

## 📈 Risultati Attesi

### Prima (Solo Dataset Encoded):
- Tutte le categoriche convertite in numerico prima del training
- Perdita di informazioni semantiche delle categoriche
- Possibile overfitting su categoriche rare

### Dopo (Dataset Appropriati):
- **CatBoost**: Migliori prestazioni su categoriche ad alta cardinalità
- **LightGBM**: Split più informativi per categoriche


## 🚀 Come Usare

1. **Eseguire preprocessing**: Il sistema salva automaticamente entrambe le versioni
2. **Configurare modelli**: Abilitare CatBoost/LightGBM nel `config.yaml`
3. **Eseguire training**: La pipeline sceglie automaticamente i dati appropriati

### Esempio Configurazione:
```yaml
models:
  advanced:
catboost: true     # Userà dataset categorico
lightgbm: true     # Userà dataset categorico
    xgboost: true       # Userà dataset encoded
    random_forest: true # Userà dataset encoded
```

## ⚠️ Note Importanti

1. **Spazio disco**: Il sistema ora salva dataset duplicati (+~50% spazio)
2. **Categoriche support**: Solo CatBoost e LightGBM usano categoriche native
3. **Ensemble models**: Continuano a usare dataset encoded per compatibilità
4. **Preprocessing time**: Leggero aumento per generazione dataset paralleli

## 🔮 Prossimi Sviluppi

- [ ] Ottimizzazione memoria per dataset grandi
- [ ] Support per altri modelli con categoriche native (FT-Transformer, TabNet)
- [ ] Analisi feature importance specifiche per categoriche
- [ ] Benchmark prestazioni dettagliati