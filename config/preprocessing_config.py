"""
Configuration file for preprocessing pipeline
"""
from pathlib import Path
from datetime import datetime

# Base paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
SCHEMA_PATH = DATA_DIR / "tabelle_alias_colonne.json"

# Versioning with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output directories
OUTPUT_DIR = BASE_DIR / "data" / TIMESTAMP
TRANSFORMERS_DIR = BASE_DIR / "transformers"
LOG_DIR = BASE_DIR / "log" / "preprocessing" / TIMESTAMP

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# File paths
TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
TRAIN_NO_PCA_PATH = OUTPUT_DIR / "train_no_pca.csv"
TEST_NO_PCA_PATH = OUTPUT_DIR / "test_no_pca.csv"
PREPROCESSING_REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"
TRANSFORMERS_PATH = TRANSFORMERS_DIR / f"{TIMESTAMP}.pkl"
LOG_FILE_PATH = LOG_DIR / f"{TIMESTAMP}.log"

# Model parameters
TARGET_COLUMN = 'AI_Prezzo_Ridistribuito'
CATEGORICAL_GROUP_COLUMN = 'AI_IdCategoriaCatastale'  # Column for grouped outlier detection
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Feature removal settings
FEATURE_REMOVAL_CONFIG = {
    'remove_duplicates': True,              # Remove duplicate columns
    'remove_constants': True,               # Remove constant columns
    'remove_high_correlation': True,        # Remove highly correlated features
    'remove_low_variance': True,           # Remove low variance features
    'cramer_threshold': 0.85,              # Threshold for categorical correlation (Cramer's V)
    'corr_threshold': 0.95,                # Threshold for numerical correlation
    'variance_threshold': 0.01             # Minimum variance threshold
}

# PCA settings
PCA_CONFIG = {
    'apply_pca': True,                     # Whether to apply PCA
    'variance_threshold': 0.95,            # Variance to retain in PCA
    'create_no_pca_version': True          # Create version without PCA
}

# Outlier detection settings
OUTLIER_CONFIG = {
    'apply_outlier_detection': True,
    'group_by_categorical': True,          # Group outlier detection by categorical variable
    'z_threshold': 3.0,
    'iqr_multiplier': 1.5,
    'isolation_contamination': 0.05,
    'min_methods_outlier': 2,              # Minimum methods that must identify an outlier
    'min_group_size': 50                   # Minimum group size for grouped outlier detection
}

# Encoding settings
ENCODING_CONFIG = {
    'low_cardinality_threshold': 20,       # Threshold for one-hot encoding
    'high_cardinality_max': 100,           # Maximum cardinality for target encoding
    'target_smoothing': 10                 # Smoothing factor for target encoding
}

# Imputation settings
IMPUTATION_CONFIG = {
    'numerical_strategy': 'median',        # 'mean', 'median', 'most_frequent'
    'categorical_strategy': 'most_frequent' # 'most_frequent', 'constant'
}

# Scaling settings
SCALING_CONFIG = {
    'method': 'standard'                   # 'standard', 'minmax', 'robust'
}

# Data loading settings
DATA_LOADING_CONFIG = {
    'selected_aliases': ["A", "AI", "PC", "ISC", "II", "PCOZ", "OZ", "OV"]
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}