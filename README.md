# 🏠 Real Estate Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![pytest](https://img.shields.io/badge/pytest-7.0+-green.svg)](https://pytest.org)
[![Coverage](https://img.shields.io/badge/coverage-70%25+-brightgreen.svg)](https://pytest-cov.readthedocs.io)

Pipeline completa di Machine Learning per l'analisi e la predizione dei prezzi immobiliari basata su dati OMI (Osservatorio del Mercato Immobiliare) e caratteristiche degli immobili. Il sistema implementa un workflow end-to-end che va dal recupero dati fino alla valutazione dei modelli, con particolare attenzione alla qualità dei dati e all'ingegneria delle features.

## 📋 Indice

- [🏗️ Architettura del Sistema](#️-architettura-del-sistema)
- [⚡ Quick Start](#-quick-start)
- [🔧 Installazione Dettagliata](#-installazione-dettagliata)
- [📊 Configurazione](#-configurazione)
- [🚀 Utilizzo](#-utilizzo)
- [🔄 Pipeline di Preprocessing](#-pipeline-di-preprocessing)
- [🤖 Training e Modelli](#-training-e-modelli)
- [🧪 Testing](#-testing)
- [📈 Monitoraggio e Logging](#-monitoraggio-e-logging)
- [🛠️ Sviluppo](#️-sviluppo)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 API Reference](#-api-reference)

## 🏗️ Architettura del Sistema

Il progetto è strutturato seguendo le best practices per progetti ML:

```
RealEstatePricePrediction/
│
├── 📁 Root Files
│   ├── main.py                 # Entry point principale
│   ├── run_tests.py            # Test runner con interfaccia CLI
│   ├── requirements.txt        # Dipendenze Python
│   ├── pytest.ini            # Configurazione testing
│   ├── Makefile               # Automazione comandi
│   ├── .gitignore             # Esclusioni Git
│   └── .env                   # Variabili ambiente (da creare)
│
├── 📁 config/
│   └── config.yaml            # Configurazione parametri globali
│
├── 📁 data/
│   ├── raw/                   # Dati grezzi estratti dal DB
│   ├── processed/             # Dataset dopo preprocessing
│   └── db_schema.json         # Schema database OMI
│
├── 📁 logs/                   # File di log generati automaticamente
├── 📁 models/                 # Modelli salvati e artifacts
├── 📁 notebooks/              # Jupyter notebooks per analisi esplorative
│
├── 📁 src/                    # Codice sorgente principale
│   ├── 📁 db/                 # Gestione database
│   │   ├── connect.py         # Connessione SQL Server
│   │   └── retrieve.py        # Query e recupero dati
│   │
│   ├── 📁 dataset/            # Costruzione dataset
│   │   └── build_dataset.py   # Feature engineering e costruzione
│   │
│   ├── 📁 preprocessing/      # Pipeline di preprocessing
│   │   ├── cleaning.py        # Pulizia dati e rimozione outlier
│   │   ├── filtering.py       # Filtri correlazioni
│   │   ├── encoding.py        # Encoding variabili categoriche
│   │   ├── imputation.py      # Gestione valori mancanti
│   │   ├── transformation.py  # Scaling, PCA, split
│   │   └── pipeline.py        # Orchestrazione completa
│   │
│   ├── 📁 training/           # Training e ottimizzazione
│   │   ├── models.py          # Wrapper modelli ML
│   │   ├── train.py           # Training pipeline
│   │   ├── evaluation.py      # Metriche e valutazione
│   │   └── tuning.py          # Ottimizzazione iperparametri
│   │
│   └── 📁 utils/              # Utilità trasversali
│       ├── io.py              # I/O file e serializzazione
│       └── logger.py          # Sistema logging configurabile
│
└── 📁 tests/                  # Suite di test completa
    ├── conftest.py            # Configurazione pytest e fixtures
    ├── test_database.py       # Test connessione e query DB
    ├── test_dataset.py        # Test costruzione dataset
    ├── test_preprocessing.py  # Test pipeline preprocessing
    ├── test_training.py       # Test training e modelli
    ├── test_evaluation.py     # Test metriche e valutazione
    ├── test_integration.py    # Test integrazione end-to-end
    └── test_utils.py          # Test utilità
```

### 🎯 Caratteristiche Principali

- **Modulare**: Architettura a componenti indipendenti e riutilizzabili
- **Configurabile**: Tutti i parametri centralizzati in `config.yaml`
- **Testabile**: Suite di test completa con copertura >70%
- **Scalabile**: Supporto per dataset grandi con ottimizzazioni memoria
- **Robusto**: Gestione errori completa e logging dettagliato
- **MLOps Ready**: Struttura compatibile con CI/CD e deployment

## ⚡ Quick Start

```bash
# 1. Clona e setup
git clone <https://github.com/BaffoBello14/RealEstatePricePrediction>
cd ml_project
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Configura ambiente
cp .env.example .env  # Modifica con le tue credenziali DB
# Modifica config/config.yaml se necessario

# 3. Esegui pipeline completa
python main.py

# 4. Oppure step specifici
python main.py --steps retrieve_data preprocessing
python main.py --steps training evaluation
```

## 🔧 Installazione Dettagliata

### Prerequisiti di Sistema

- **Python 3.8+** (testato su 3.8, 3.9, 3.10, 3.11)
- **SQL Server** con driver ODBC 18+ 
- **Memoria RAM**: 8GB+ raccomandati (4GB minimo)
- **Spazio disco**: 2GB+ per dati e modelli
- **Sistema operativo**: Linux, macOS, Windows

### 1. Ambiente Virtuale

```bash
# Creazione ambiente virtuale
python -m venv venv

# Attivazione
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Verifica versione Python
python --version  # Deve essere 3.8+
```

### 2. Installazione Dipendenze

```bash
# Installazione base
pip install --upgrade pip
pip install -r requirements.txt

# Verifica installazione
pip list | grep -E "(pandas|scikit-learn|xgboost)"

# Per sviluppo (opzionale)
make install-dev
```

### 3. Configurazione Database

#### SQL Server Setup

```bash
# Ubuntu/Debian - Installa driver ODBC
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Verifica driver installati
odbcinst -q -d
```

#### File .env

```env
# Database Configuration
SERVER=your_server_name
DATABASE=your_database_name
USER=your_username
PASSWORD=your_secure_password
```

### 4. Verifica Installazione

```bash
# Test connessione database
python -c "from src.db.connect import test_connection; test_connection()"

# Test caricamento configurazione
python -c "from src.utils.io import load_config; print('Config OK')"

# Test moduli principali
python -c "import pandas, sklearn, xgboost, lightgbm; print('Dependencies OK')"
```

## 📊 Configurazione

Il file `config/config.yaml` è il centro di controllo di tutto il sistema. È organizzato in sezioni logiche:

### 🗄️ Database

```yaml
database:
  schema_path: "data/db_schema.json"
  selected_aliases: ["A", "AI", "PC", "ISC", "II", "PC_OZ", "OZ", "OV"]
```

- `schema_path`: Schema tabelle database OMI
- `selected_aliases`: Tabelle da includere nell'analisi

### 📂 Paths

```yaml
paths:
  data_raw: "data/raw/"
  data_processed: "data/processed/"
  models: "models/"
  logs: "logs/"
```

### 🎯 Target

```yaml
target:
  column: "AI_Prezzo_Ridistribuito"
```

Definisce la variabile target per la predizione.

### 🔄 Preprocessing

Configurazione completa della pipeline di preprocessing:

```yaml
preprocessing:
  # Controllo step (enable/disable granulare)
  steps:
    enable_specific_columns_removal: true
    enable_constant_columns_removal: true
    enable_cramers_analysis: true
    enable_auto_numeric_conversion: true
    enable_correlation_removal: true
    enable_advanced_encoding: true
    enable_feature_scaling: true
    enable_log_transformation: true
    enable_outlier_detection: true
    enable_pca: false

  # Parametri pulizia dati
  columns_to_remove: ['A_Id', 'A_Codice', 'A_Prezzo', 'AI_Id']
  constant_column_threshold: 0.95
  auto_numeric_threshold: 0.8

  # Correlazioni
  cramer_threshold: 0.95
  corr_threshold: 0.95

  # Split temporale per evitare data leakage
  use_temporal_split: true
  year_column: "A_AnnoStipula"
  month_column: "A_MeseStipula"
```

### 🤖 Training

```yaml
training:
  # Configurazione modelli
  models:
    enable_linear: true
    enable_ridge: true
    enable_elastic_net: true
    enable_random_forest: true
    enable_gradient_boosting: true
    enable_xgboost: true
    enable_lightgbm: true
    enable_catboost: true

  # Cross-validation
  cv_folds: 5
  cv_scoring: 'neg_mean_absolute_error'

  # Ottimizzazione iperparametri
  tuning:
    enable_optuna: true
    n_trials: 100
    timeout: 3600  # 1 ora
```

### 📊 Evaluation

```yaml
evaluation:
  # Metriche da calcolare
  metrics:
    - 'mae'
    - 'mse' 
    - 'rmse'
    - 'r2'
    - 'mape'

  # Analisi feature importance
  feature_importance:
    enable_shap: true
    enable_permutation: true
    max_features_plot: 20
```

## 🚀 Utilizzo

### Interfaccia Command Line

Il sistema offre una CLI flessibile attraverso `main.py`:

```bash
# Esecuzione completa (tutti gli step)
python main.py

# Step specifici
python main.py --steps retrieve_data
python main.py --steps retrieve_data preprocessing
python main.py --steps training evaluation

# Con configurazione custom
python main.py --config path/to/custom/config.yaml

# Forza ricaricamento dati (ignora cache)
python main.py --force-reload

# Debug mode (logging verboso)
python main.py --debug

# Help completo
python main.py --help
```

### Step della Pipeline

#### 1. 📥 Retrieve Data
```bash
python main.py --steps retrieve_data
```
- Connessione al database SQL Server
- Esecuzione query sui dati OMI
- Salvataggio dati grezzi in formato Parquet
- Validazione integrità dati

#### 2. 🏗️ Build Dataset
```bash
python main.py --steps build_dataset
```
- Feature engineering automatico
- Estrazione features dai piani (`AI_Piano`)
- Gestione prezzi OMI multipli
- Calcolo target ridistribuito

#### 3. 🔄 Preprocessing
```bash
python main.py --steps preprocessing
```
- Pulizia dati (removal outlier, colonne costanti)
- Conversione automatica a numerico
- Encoding variabili categoriche
- Gestione valori mancanti
- Filtri correlazione
- Scaling e trasformazioni
- Train/test split

#### 4. 🤖 Training
```bash
python main.py --steps training
```
- Training modelli multipli
- Cross-validation
- Ottimizzazione iperparametri con Optuna
- Salvataggio modelli e artifacts

#### 5. 📈 Evaluation
```bash
python main.py --steps evaluation
```
- Calcolo metriche multiple
- Feature importance analysis
- SHAP values
- Generazione report

### Utilizzo Programmativo

```python
# Import moduli principali
from src.utils.io import load_config
from src.db.retrieve import retrieve_data
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.training.train import run_training_pipeline

# Carica configurazione
config = load_config("config/config.yaml")

# Esegui step specifici
raw_data_path = retrieve_data(config)
preprocessing_info = run_preprocessing_pipeline(config)
training_results = run_training_pipeline(config)
```

## 🔄 Pipeline di Preprocessing

La pipeline di preprocessing è il cuore del sistema e include diversi step sofisticati:

### 1. 🧹 Data Cleaning

```python
# Configurazione in config.yaml
preprocessing:
  columns_to_remove: ['A_Id', 'A_Codice', 'A_Prezzo', 'AI_Id']
  constant_column_threshold: 0.95
```

**Operazioni:**
- Rimozione colonne specifiche non predittive
- Rimozione colonne costanti (>95% valori uguali)
- Pulizia valori nulli/vuoti
- Validazione tipi di dati

### 2. 🔧 Feature Engineering

**Estrazione Features dai Piani (`AI_Piano`):**
```python
# Esempio: "1;2;3" → multiple features
{
    'min_floor': 1,
    'max_floor': 3, 
    'n_floors': 3,
    'has_basement': True,
    'has_ground': True,
    'has_upper': True,
    'floor_span': 2,
    'floor_numeric_weighted': 2.0
}
```

**Gestione Prezzi OMI:**
- Stati multipli: Normale, Ottimo, Scadente
- Calcolo coefficienti di ridistribuzione
- Target finale: `AI_Prezzo_Ridistribuito`

### 3. 🔄 Auto-Conversion

```python
# Conversione automatica a numerico
auto_numeric_threshold: 0.8  # 80% conversioni valide
```

Il sistema tenta di convertire automaticamente colonne object a numeriche quando il tasso di successo supera la soglia.

### 4. 🏷️ Advanced Encoding

**Strategia di Encoding:**
- **One-Hot**: Variabili con poche categorie (<10)
- **Target Encoding**: Variabili con molte categorie
- **Label Encoding**: Variabili ordinali
- **Binary Encoding**: Fallback per categorie intermedie

```python
# Configurazione encoding
encoding:
  target_encoding_smoothing: 10.0
  onehot_max_categories: 10
  rare_category_threshold: 0.01
```

### 5. 🎯 Correlation Filtering

**Cramér's V** per variabili categoriche:
```python
cramer_threshold: 0.95
```

**Pearson** per variabili numeriche:
```python
corr_threshold: 0.95
```

### 6. ⚖️ Train/Test Split

**Split Temporale** (raccomandato per dati finanziari):
```python
use_temporal_split: true
year_column: "A_AnnoStipula"
month_column: "A_MeseStipula"
```

**Split Stratificato** (alternativo):
```python
use_temporal_split: false
test_size: 0.2
stratify_column: "target_binned"
```

### 7. 📊 Scaling e Transformations

**StandardScaler** per features numeriche:
```python
feature_scaling:
  method: "standard"  # o "minmax", "robust"
```

**PCA** per riduzione dimensionalità:
```python
pca:
  variance_threshold: 0.95
  max_components: 100
```

**Log Transformation** del target:
```python
log_transformation:
  method: "log1p"  # evita log(0)
```

### 8. 🚨 Outlier Detection

Metodi multipli per robustezza:

**Z-Score:**
```python
zscore_threshold: 3.0
```

**IQR (Interquartile Range):**
```python
iqr_factor: 1.5
```

**Isolation Forest:**
```python
isolation_forest:
  contamination: 0.1
  n_estimators: 100
```

## 🤖 Training e Modelli

### Modelli Supportati

Il sistema supporta un'ampia gamma di algoritmi ML:

#### 📈 Linear Models
- **Linear Regression**: Baseline semplice
- **Ridge Regression**: Regolarizzazione L2
- **Elastic Net**: Combinazione L1+L2

#### 🌳 Tree-Based Models
- **Random Forest**: Ensemble di alberi
- **Gradient Boosting**: Scikit-learn implementation
- **XGBoost**: Extreme Gradient Boosting
- **LightGBM**: Microsoft's fast gradient boosting
- **CatBoost**: Yandex's categorical boosting

### Cross-Validation

```python
# Configurazione CV
cv_folds: 5
cv_scoring: 'neg_mean_absolute_error'
cv_stratify: true
```

**Strategie CV:**
- **KFold**: Standard per dati non temporali
- **TimeSeriesSplit**: Per dati temporali
- **StratifiedKFold**: Per target sbilanciati

### Ottimizzazione Iperparametri

**Optuna Integration:**
```python
tuning:
  enable_optuna: true
  n_trials: 100
  timeout: 3600  # 1 ora max
  pruning: true
  sampler: "TPE"  # Tree-structured Parzen Estimator
```

**Search Spaces Personalizzati:**
```python
# Esempio per XGBoost
xgboost_params:
  n_estimators: [100, 200, 500, 1000]
  max_depth: [3, 5, 7, 9]
  learning_rate: [0.01, 0.1, 0.2]
  subsample: [0.8, 0.9, 1.0]
```

### Model Persistence

**Salvataggio Automatico:**
```python
# Modelli salvati in models/
- best_model.joblib          # Miglior modello
- model_pipeline.joblib      # Pipeline completa
- preprocessing_objects.pkl  # Oggetti preprocessing
- feature_names.json         # Nomi features
- training_metadata.json     # Metadati training
```

## 🧪 Testing

Il progetto implementa una suite di test completa con diverse tipologie:

### Struttura Testing

```bash
tests/
├── conftest.py              # Fixtures e configurazione pytest
├── test_database.py         # Test connessione e query DB
├── test_dataset.py          # Test costruzione dataset  
├── test_preprocessing.py    # Test pipeline preprocessing
├── test_training.py         # Test training e modelli
├── test_evaluation.py       # Test metriche e valutazione
├── test_integration.py      # Test end-to-end
└── test_utils.py           # Test utilità
```

### Tipologie di Test

#### 🔧 Unit Tests
```bash
# Test singoli moduli
pytest -m "unit"
python run_tests.py --unit

# Test modulo specifico
pytest -m "unit" tests/test_preprocessing.py
python run_tests.py --preprocessing
```

#### 🔗 Integration Tests
```bash
# Test integrazione tra moduli
pytest -m "integration"
python run_tests.py --integration

# Test end-to-end completo
python run_tests.py --e2e
```

#### ⚡ Fast/Slow Tests
```bash
# Solo test veloci (<10s)
pytest -m "not slow"
python run_tests.py --fast

# Solo test lenti (>10s) 
pytest -m "slow"
python run_tests.py --slow
```

### Configurazione Pytest

```ini
# pytest.ini
[tool:pytest]
addopts = 
    --cov=src 
    --cov-report=html:htmlcov 
    --cov-fail-under=70
    -v
markers:
    slow: marks tests as slow
    integration: integration tests
    unit: unit tests
    preprocessing: preprocessing tests
    training: training tests
    database: database tests
```

### Coverage Report

```bash
# Genera report copertura
pytest --cov=src --cov-report=html
python run_tests.py --coverage

# Visualizza report
open htmlcov/index.html
```

### Test Runner Avanzato

Il file `run_tests.py` fornisce un'interfaccia user-friendly:

```bash
# Help completo
python run_tests.py --help

# Test specifici per modulo
python run_tests.py --utils
python run_tests.py --db
python run_tests.py --preprocessing
python run_tests.py --training

# Combinazioni
python run_tests.py --unit --fast
python run_tests.py --integration --coverage

# Con opzioni pytest
python run_tests.py --unit --verbose --fail-fast
```

### Fixtures e Mock Data

```python
# conftest.py - Fixtures principali
@pytest.fixture
def sample_config():
    """Configurazione di test."""
    return load_test_config()

@pytest.fixture  
def mock_database():
    """Database mock per test."""
    return create_mock_db()

@pytest.fixture
def sample_dataframe():
    """DataFrame di esempio per test."""
    return create_sample_data()
```

## 📈 Monitoraggio e Logging

### Sistema di Logging

**Configurazione Logging:**
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_output: true
  console_output: true
  max_file_size: "10MB"
  backup_count: 5
```

**Logger Specializzati:**
```python
# Logger per modulo
from src.utils.logger import get_logger
logger = get_logger(__name__)

# Logger per step pipeline
logger = get_logger("preprocessing.cleaning")
logger = get_logger("training.xgboost")
logger = get_logger("evaluation.metrics")
```

### Monitoring Pipeline

**Metriche di Performance:**
- Tempo esecuzione per step
- Utilizzo memoria RAM
- Dimensioni dataset
- Numero features prima/dopo preprocessing
- Accuracy e loss dei modelli

**Dashboard Info (logs/ml_pipeline.log):**
```
2024-01-15 10:30:15 - preprocessing.cleaning - INFO - Dataset shape: (50000, 120)
2024-01-15 10:30:16 - preprocessing.cleaning - INFO - Removed 15 constant columns
2024-01-15 10:30:18 - preprocessing.encoding - INFO - Applied target encoding to 8 features
2024-01-15 10:30:20 - preprocessing.correlation - INFO - Removed 12 highly correlated features
2024-01-15 10:30:25 - preprocessing.pipeline - INFO - Final dataset shape: (50000, 85)
2024-01-15 10:35:12 - training.xgboost - INFO - Best CV MAE: 15420.35
2024-01-15 10:35:15 - training.pipeline - INFO - Training completed in 4m 50s
```

### Output Files e Artifacts

**Dati Processati:**
```
data/processed/
├── X_train.parquet              # Features training
├── X_test.parquet               # Features test  
├── y_train.parquet              # Target training (log-transformed)
├── y_test.parquet               # Target test (scala originale)
├── feature_names.json           # Nomi features finali
└── preprocessing_info.json      # Metadati preprocessing completi
```

**Modelli e Artifacts:**
```
models/
├── best_model.joblib            # Miglior modello addestrato
├── model_pipeline.joblib        # Pipeline completa (preprocessing + model)
├── scaler.joblib               # StandardScaler fitted
├── encoder_objects.pkl          # Oggetti encoding fitted
├── pca_model.joblib            # PCA transformer (se abilitato)
└── training_results.json       # Risultati completi training
```

**Report di Valutazione:**
```
models/evaluation/
├── metrics_report.json          # Metriche numeriche
├── feature_importance.png       # Plot importance
├── shap_summary.png            # SHAP values summary
├── residuals_plot.png          # Analisi residui
└── predictions_vs_actual.png   # Scatter plot predizioni
```

## 🛠️ Sviluppo

### Setup Ambiente di Sviluppo

```bash
# Installa dipendenze sviluppo
make install-dev

# Oppure manualmente
pip install black flake8 mypy bandit safety pre-commit
```

### Code Quality Tools

**Formattazione Codice:**
```bash
# Formatta tutto il codice
make format
black src/ tests/

# Check formattazione
black --check src/ tests/
```

**Linting:**
```bash
# Lint completo
make lint
flake8 src/ tests/

# Type checking
mypy src/
```

**Security Scan:**
```bash
# Scan vulnerabilità
bandit -r src/
safety check
```

### Pre-commit Hooks

```bash
# Setup pre-commit
pre-commit install

# Run manualmente
pre-commit run --all-files
```

### Aggiungere Nuovi Moduli

1. **Crea il modulo** nella directory appropriata:
```python
# src/preprocessing/new_module.py
def new_function():
    """Documentazione dettagliata."""
    pass
```

2. **Aggiungi import** in `__init__.py`:
```python
# src/preprocessing/__init__.py
from .new_module import new_function
```

3. **Scrivi test**:
```python
# tests/test_preprocessing.py
def test_new_function():
    """Test per new_function."""
    result = new_function()
    assert result is not None
```

4. **Aggiorna documentazione** e configurazione se necessario.

### Makefile Commands

Il `Makefile` fornisce comandi di automazione:

```bash
# Visualizza tutti i comandi
make help

# Installazione e setup
make install          # Installa dipendenze base
make install-dev      # Installa dipendenze sviluppo

# Testing
make test            # Tutti i test
make test-unit       # Solo unit test
make test-integration # Solo integration test
make test-fast       # Solo test veloci
make test-slow       # Solo test lenti
make test-cov        # Test con coverage
make test-parallel   # Test paralleli

# Code quality
make lint            # Linting completo
make format          # Formattazione codice
make type-check      # Type checking
make security-check  # Security scan

# Pulizia
make clean           # Rimuovi file temporanei
make clean-pyc       # Rimuovi file .pyc
make clean-build     # Rimuovi build artifacts
make clean-test      # Rimuovi file test
make clean-all       # Pulizia completa
```

## 🐛 Troubleshooting

### Problemi Comuni e Soluzioni

#### 🔌 Errori di Connessione Database

**Problema:** `pyodbc.Error: Data source name not found`
```bash
# Soluzione: Verifica driver ODBC
odbcinst -q -d

# Installa driver mancanti (Ubuntu)
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
```

**Problema:** `Login failed for user`
```bash
# Verifica credenziali in .env
cat .env

# Test connessione
python -c "from src.db.connect import test_connection; test_connection()"
```

#### 💾 Errori di Memoria

**Problema:** `MemoryError during PCA`
```yaml
# Riduci soglia PCA in config.yaml
pca:
  variance_threshold: 0.90  # Era 0.95
  max_components: 50        # Era 100
```

**Problema:** `Out of memory during training`
```yaml
# Riduci parametri modelli
training:
  batch_size: 1000         # Riduci batch size
  max_features: 0.5        # Usa subset features
```

#### 🔧 Errori di Encoding

**Problema:** `UnicodeDecodeError`
```python
# Forza encoding UTF-8
import pandas as pd
df = pd.read_csv(file, encoding='utf-8')
```

**Problema:** `KeyError during encoding`
```yaml
# Abilita gestione categorie rare
encoding:
  handle_unknown: "ignore"
  rare_category_threshold: 0.01
```

#### ⚡ Problemi di Performance

**Problema:** Pipeline troppo lenta
```yaml
# Ottimizzazioni in config.yaml
preprocessing:
  steps:
    enable_pca: false              # Disabilita PCA
    enable_cramers_analysis: false # Disabilita Cramér's V
    
training:
  models:
    enable_catboost: false        # Disabilita modelli lenti
    enable_xgboost: false
```

**Problema:** Optuna troppo lento
```yaml
tuning:
  n_trials: 20          # Riduci da 100
  timeout: 600          # 10 minuti invece di 1 ora
  pruning: true         # Abilita pruning
```

### Debug Mode

```bash
# Esegui con debug completo
python main.py --debug

# Log solo errori
python main.py --log-level ERROR

# Profiling performance
python -m cProfile -o profile.stats main.py
```

### Diagnostica Sistema

```bash
# Check dipendenze
pip check

# Info sistema
python -c "import sys; print(sys.version)"
python -c "import platform; print(platform.platform())"

# Memory disponibile
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total/1e9:.1f}GB')"

# Spazio disco
df -h .
```

## 📚 API Reference

### Core Modules

#### Database (`src.db`)

```python
from src.db.connect import DatabaseConnection, test_connection
from src.db.retrieve import retrieve_data

# Test connessione
test_connection()

# Retrieve data
config = load_config()
data_path = retrieve_data(config)
```

#### Preprocessing (`src.preprocessing`)

```python
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.preprocessing.cleaning import remove_constant_columns
from src.preprocessing.encoding import apply_target_encoding

# Pipeline completa
result = run_preprocessing_pipeline(config)

# Step individuali
df_clean = remove_constant_columns(df, threshold=0.95)
df_encoded = apply_target_encoding(df, target_col, cat_cols)
```

#### Training (`src.training`)

```python
from src.training.train import run_training_pipeline
from src.training.models import MLModelWrapper
from src.training.evaluation import calculate_metrics

# Training completo
results = run_training_pipeline(config)

# Modello singolo
model = MLModelWrapper('xgboost', **params)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Valutazione
metrics = calculate_metrics(y_true, y_pred)
```

#### Utils (`src.utils`)

```python
from src.utils.io import load_config, save_parquet, load_parquet
from src.utils.logger import setup_logger, get_logger

# I/O
config = load_config('config/config.yaml')
save_parquet(df, 'data/output.parquet')
df = load_parquet('data/input.parquet')

# Logging
logger = get_logger(__name__)
logger.info("Messaggio di log")
```

### Configuration Schema

Il file `config.yaml` segue questo schema:

```yaml
# Schema completo configurazione
database:
  schema_path: str
  selected_aliases: list[str]

paths:
  data_raw: str
  data_processed: str  
  models: str
  logs: str

target:
  column: str

preprocessing:
  steps:
    enable_*: bool
  *_threshold: float
  *_params: dict

training:
  models:
    enable_*: bool
  cv_folds: int
  cv_scoring: str
  tuning:
    enable_optuna: bool
    n_trials: int
    timeout: int

evaluation:
  metrics: list[str]
  feature_importance:
    enable_shap: bool
    enable_permutation: bool

logging:
  level: str
  format: str
  file_output: bool
  console_output: bool
```

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli completi.

## 👥 Contribuzioni

Le contribuzioni sono benvenute! Per contribuire:

1. Fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

### Linee Guida Contribuzioni

- Scrivi test per ogni nuova funzionalità
- Mantieni copertura codice >70%
- Segui le convenzioni di naming del progetto
- Documenta le API pubbliche
- Esegui `make lint` prima del commit

---

<div align="center">

**⭐ Se questo progetto ti è stato utile, lascia una stella su GitHub! ⭐**

![Built with ❤️](https://img.shields.io/badge/Built%20with-%E2%9D%A4%EF%B8%8F-red)
![Python](https://img.shields.io/badge/Made%20with-Python-blue)

</div>