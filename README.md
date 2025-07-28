# ML Pipeline per Analisi Immobiliare

Pipeline di Machine Learning completa per l'analisi e la predizione dei prezzi immobiliari basata su dati OMI (Osservatorio del Mercato Immobiliare) e caratteristiche degli immobili.

## Struttura del Progetto

```
RealEstatePricePrediction/
│
├── README.md
├── requirements.txt
├── .env                    # Variabili di ambiente (DB credentials)
├── main.py                 # Entry point principale
│
├── config/
│   └── config.yaml         # Configurazione parametri globali
│
├── data/
│   ├── raw/               # Dati grezzi estratti dal DB
│   ├── processed/         # Dataset dopo il preprocessing
│   └── interim/           # Dati parziali (opzionale)
│
├── logs/                  # File di log
│
├── models/                # Modelli salvati
│
├── notebooks/             # Jupyter notebooks per analisi esplorative
│
└── src/
    ├── __init__.py
    │
    ├── db/
    │   ├── __init__.py
    │   ├── connect.py      # Connessione al database
    │   └── retrieve.py     # Query e recupero dati
    │
    ├── dataset/
    │   ├── __init__.py
    │   └── build_dataset.py # Costruzione dataset e feature engineering
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── cleaning.py     # Pulizia dati e outlier detection
    │   ├── filtering.py    # Filtri su correlazioni
    │   ├── encoding.py     # Encoding variabili categoriche
    │   ├── imputation.py   # Gestione valori mancanti
    │   ├── transformation.py # Scaling, PCA, split
    │   └── pipeline.py     # Orchestrazione preprocessing
    │
    ├── training/           # TODO: Moduli di training
    │   ├── __init__.py
    │   ├── models.py       # Wrapper per modelli
    │   ├── train.py        # Training
    │   ├── evaluation.py   # Metriche e valutazione
    │   └── tuning.py       # Ottimizzazione iperparametri
    │
    └── utils/
        ├── __init__.py
        ├── io.py           # Salvataggio/caricamento file
        └── logger.py       # Sistema di logging
```

## Installazione

1. **Clona il repository:**
```bash
git clone <repository-url>
cd ml_project
```

2. **Crea ambiente virtuale:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows
```

3. **Installa dipendenze:**
```bash
pip install -r requirements.txt
```

4. **Configura database:**
Crea file `.env` nella root con le credenziali del database:
```env
SERVER=your_server
DATABASE=your_database
USER=your_username
PASSWORD=your_password
```

5. **Configura parametri:**
Modifica `config/config.yaml` secondo le tue esigenze.

## Utilizzo

### Esecuzione Completa
```bash
python main.py
```

### Esecuzione Step Specifici
```bash
# Solo recupero dati
python main.py --steps retrieve_data

# Preprocessing + Training
python main.py --steps preprocessing training

# Forza ricaricamento di tutti i dati
python main.py --force-reload
```

### Configurazione Custom
```bash
python main.py --config path/to/custom/config.yaml
```

## Pipeline di Preprocessing

La pipeline di preprocessing include i seguenti step:

1. **Pulizia Dati**
   - Rimozione colonne non predittive
   - Rimozione colonne costanti
   - Pulizia valori nulli/vuoti

2. **Feature Engineering**
   - Estrazione features dai piani (`AI_Piano`)
   - Conversione automatica a numerico
   - Encoding variabili categoriche (One-hot, Target, Label)

3. **Gestione Valori Mancanti**
   - Imputazione mediana per variabili numeriche
   - Imputazione "MISSING" per variabili categoriche

4. **Filtri Correlazione**
   - Rimozione variabili categoriche correlate (Cramér's V)
   - Rimozione variabili numeriche correlate (Pearson)

5. **Train/Test Split**
   - Divisione stratificata del dataset

6. **Trasformazioni**
   - Standardizzazione features (StandardScaler)
   - Riduzione dimensionalità (PCA)

7. **Outlier Detection**
   - Z-Score, IQR, Isolation Forest
   - Trasformazione logaritmica del target

## Configurazione

Il file `config/config.yaml` contiene tutti i parametri configurabili:

- **Database**: Schema e alias tabelle
- **Preprocessing**: Soglie, parametri algoritmi
- **Modelli**: Configurazione modelli ML
- **Paths**: Directory di input/output
- **Logging**: Livello e formato log

## Feature Engineering

### Piani (`AI_Piano`)
Il sistema estrae automaticamente le seguenti features dai dati del piano:
- `min_floor`, `max_floor`: Piano minimo e massimo
- `n_floors`: Numero di piani
- `has_basement`, `has_ground`, `has_upper`: Flags booleani
- `floor_span`: Ampiezza piani
- `floor_numeric_weighted`: Media pesata piani

### Prezzi OMI
- Gestione stati multipli (Normale, Ottimo, Scadente)
- Calcolo coefficienti di ridistribuzione
- Target: `AI_Prezzo_Ridistribuito`

## Output

La pipeline genera i seguenti file:

### Dati Processati
- `X_train.parquet`: Features di training (PCA)
- `X_test.parquet`: Features di test (PCA)
- `y_train.parquet`: Target di training (log-trasformato)
- `y_test.parquet`: Target di test (scala originale)

### Metadati
- `preprocessing_info.json`: Informazioni complete sul preprocessing
- Log dettagliati in `logs/ml_pipeline.log`

## Logging

Il sistema di logging è configurabile e include:
- Livelli: DEBUG, INFO, WARNING, ERROR
- Output: Console + File
- Formato personalizzabile
- Logger separati per modulo

## Sviluppo

### Aggiungere Nuovi Moduli
1. Crea il file nella directory appropriata
2. Aggiungi import in `__init__.py`
3. Aggiorna la documentazione

### Testing
```bash
# Test connessione database
python -c "from src.db.connect import test_connection; test_connection()"

# Test caricamento configurazione
python -c "from src.utils.io import load_config; print(load_config())"
```

## TODO

- [ ] Implementare moduli di training (`src/training/`)
- [ ] Aggiungere cross-validation
- [ ] Implementare ensemble methods
- [ ] Aggiungere metriche di valutazione
- [ ] Creare dashboard di monitoraggio
- [ ] Aggiungere unit tests
- [ ] Ottimizzazione performance

## Requisiti Sistema

- Python 3.8+
- SQL Server con driver ODBC 18
- Memoria RAM: 8GB+ raccomandati
- Spazio disco: 2GB+ per dati e modelli

## Troubleshooting

### Errori Comuni

1. **Errore connessione database**
   - Verifica file `.env`
   - Controlla driver ODBC installato

2. **Errori memory**
   - Riduci `pca_variance_threshold`
   - Aumenta RAM virtuale

3. **Errori encoding**
   - Verifica encoding UTF-8 nei file
   - Controlla caratteri speciali nei nomi colonne

### Performance

- Usa formato Parquet per I/O veloce
- Abilita caching per sviluppo
- Monitora uso memoria durante PCA

## Licenza

[Specifica la licenza del progetto]

## Contatti

[Informazioni di contatto del team]