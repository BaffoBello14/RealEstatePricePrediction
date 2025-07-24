# ML Pipeline Repository

Una repository completa per il preprocessing e training di modelli di machine learning con supporto per outlier detection raggruppata per categoria catastale e creazione di dataset sia con che senza PCA.

## Struttura del Progetto

```
.
├── config/                          # File di configurazione
│   ├── preprocessing_config.py      # Configurazione preprocessing
│   └── training_config.py          # Configurazione training
├── data_loader/                     # Moduli per caricamento dati
│   └── load_data_2.py              # Caricamento dati con features piano
├── db/                             # Connessioni database
│   └── connection.py               # Connessione al database
├── preprocessing/                   # Moduli preprocessing
│   ├── outlier_detection.py       # Outlier detection con grouping
│   ├── feature_engineering.py     # Feature engineering e selezione
│   ├── dimensionality_reduction.py # PCA e riduzione dimensionalità
│   └── pipeline.py                # Pipeline principale preprocessing
├── training/                       # Moduli training
│   └── models.py                   # Definizioni modelli ML
├── run_preprocessing.py            # Script principale preprocessing
├── requirements.txt               # Dipendenze Python
└── README.md                      # Questo file
```

## Caratteristiche Principali

### 🔍 Outlier Detection Avanzata
- **Outlier detection raggruppata**: Gli outlier vengono rilevati separatamente per ogni categoria catastale (AI_IdCategoriaCatastale) invece che sull'intera distribuzione
- **Metodi multipli**: Z-score, IQR e Isolation Forest
- **Soglie configurabili**: Personalizza i parametri per ogni metodo
- **Visualizzazioni**: Grafici per ogni gruppo e metodo

### 🛠️ Feature Engineering Completo
- **Rimozione feature duplicate**: Configurable
- **Rimozione feature costanti**: Configurable
- **Correlazione alta**: Rimozione automatica di feature altamente correlate (numeriche e categoriche)
- **Varianza bassa**: Rimozione feature con varianza sotto soglia
- **Encoding categorico**: Label encoding e target encoding automatici

### 📊 Dataset Multipli
- **Con PCA**: Dataset ridotto con componenti principali
- **Senza PCA**: Dataset completo scalato
- **Entrambi salvati**: Possibilità di scegliere quale usare per il training

### ⚙️ Configurazione Flessibile
Tutti i parametri sono configurabili tramite file di configurazione:

```python
# Esempio configurazione outlier detection
OUTLIER_CONFIG = {
    'apply_outlier_detection': True,
    'group_by_categorical': True,          # Raggruppa per categoria catastale
    'z_threshold': 3.0,
    'iqr_multiplier': 1.5,
    'isolation_contamination': 0.05,
    'min_methods_outlier': 2,              # Minimo metodi che devono identificare un outlier
    'min_group_size': 50                   # Dimensione minima gruppo per outlier detection
}

# Esempio configurazione rimozione feature
FEATURE_REMOVAL_CONFIG = {
    'remove_duplicates': True,              # Rimuovi colonne duplicate
    'remove_constants': True,               # Rimuovi colonne costanti
    'remove_high_correlation': True,        # Rimuovi feature altamente correlate
    'remove_low_variance': True,           # Rimuovi feature con bassa varianza
    'cramer_threshold': 0.85,              # Soglia correlazione categoriche (Cramer's V)
    'corr_threshold': 0.95,                # Soglia correlazione numeriche
    'variance_threshold': 0.01             # Soglia varianza minima
}
```

## Installation

1. Clona la repository:
```bash
git clone <repository-url>
cd ml-pipeline
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Configura la connessione al database in `db/connection.py`

## Utilizzo

### 1. Preprocessing

Modifica la configurazione in `config/preprocessing_config.py` secondo le tue esigenze, poi esegui:

```bash
python run_preprocessing.py
```

Questo genererà:
- `data/{timestamp}/train.csv` - Dataset training con PCA
- `data/{timestamp}/test.csv` - Dataset test con PCA  
- `data/{timestamp}/train_no_pca.csv` - Dataset training senza PCA
- `data/{timestamp}/test_no_pca.csv` - Dataset test senza PCA
- `transformers/{timestamp}.pkl` - Trasformatori fitted
- `data/{timestamp}/preprocessing_report.json` - Report dettagliato

### 2. Training (TODO)

```bash
python run_training.py
```

## Configurazioni Dettagliate

### Outlier Detection

La configurazione outlier detection supporta:

- **Metodo raggruppato**: `group_by_categorical = True` per raggruppare per `AI_IdCategoriaCatastale`
- **Metodo globale**: `group_by_categorical = False` per outlier detection su tutto il dataset
- **Soglie personalizzabili**: Z-score, IQR multiplier, contamination Isolation Forest
- **Dimensione minima gruppo**: I gruppi troppo piccoli vengono saltati

### Feature Removal

Puoi abilitare/disabilitare ogni tipo di rimozione:

- **Duplicate**: Colonne identiche
- **Constants**: Colonne con >99% valori uguali
- **High Correlation**: Correlazione numerica (Pearson) e categorica (Cramer's V)
- **Low Variance**: Varianza sotto soglia configurabile

### PCA

- **Soglia varianza**: Quanta varianza mantenere (default 95%)
- **Creazione entrambe versioni**: Con e senza PCA
- **Analisi componenti**: Report dettagliato delle componenti principali
- **Visualizzazioni**: Grafici varianza e loadings

## Logging

Il sistema produce log dettagliati in:
- `log/preprocessing/{timestamp}/{timestamp}.log`
- Console output con progress

## File di Output

### Dataset
- Training e test sets sia con che senza PCA
- Target sia log-trasformato che originale
- Colonne chiaramente etichettate

### Transformers
File pickle con tutti i trasformatori fitted:
- Scaler
- Imputers (numerici e categorici)  
- Encoders (label e target)
- Modello PCA
- Outlier detector

### Report
JSON dettagliato con:
- Configurazione utilizzata
- Statistiche dataset
- Feature rimosse per categoria
- Informazioni PCA
- Outliers rimossi per gruppo

## Personalizzazione

Per modificare il comportamento:

1. **Aggiungi nuovi metodi outlier detection**: Estendi `preprocessing/outlier_detection.py`
2. **Cambia logica feature removal**: Modifica `preprocessing/feature_engineering.py`
3. **Nuove tecniche dimensionality reduction**: Estendi `preprocessing/dimensionality_reduction.py`
4. **Diversi data loaders**: Crea nuovi moduli in `data_loader/`

## Note

- La colonna di raggruppamento per outlier detection è configurabile (`CATEGORICAL_GROUP_COLUMN`)
- Il sistema gestisce automaticamente categorie mancanti nei test set
- Tutti i trasformatori sono fitted solo sui dati di training
- Le visualizzazioni vengono generate automaticamente durante il preprocessing

## TODO

- [ ] Completare modulo training con hyperparameter optimization
- [ ] Aggiungere feature selection avanzata
- [ ] Implementare pipeline inference
- [ ] Aggiungere test automatizzati
- [ ] Documentazione API dettagliata