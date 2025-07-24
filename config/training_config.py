"""
Configuration file for training pipeline
"""
import re
from pathlib import Path
from datetime import datetime

# Timestamp for this training session
TRAINING_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Find most recent preprocessing timestamp
DATA_BASE_DIR = Path("data")
if DATA_BASE_DIR.exists():
    timestamps = [
        p.name for p in DATA_BASE_DIR.iterdir()
        if p.is_dir() and re.match(r"\d{8}_\d{6}$", p.name)
    ]
    if timestamps:
        PREPROCESSING_TIMESTAMP = max(timestamps)
    else:
        PREPROCESSING_TIMESTAMP = "latest"  # Fallback
else:
    PREPROCESSING_TIMESTAMP = "latest"

# Input paths (from preprocessing)
DATA_DIR = DATA_BASE_DIR / PREPROCESSING_TIMESTAMP
TRANSFORMERS_DIR = Path("transformers")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TRAIN_NO_PCA_PATH = DATA_DIR / "train_no_pca.csv"
TEST_NO_PCA_PATH = DATA_DIR / "test_no_pca.csv"
PREPROCESSING_REPORT_PATH = DATA_DIR / "preprocessing_report.json"
TRANSFORMERS_PATH = TRANSFORMERS_DIR / f"{PREPROCESSING_TIMESTAMP}.pkl"

# Output paths (training)
MODELS_DIR = Path("models") / TRAINING_TIMESTAMP
RESULTS_DIR = Path("results") / TRAINING_TIMESTAMP
LOG_DIR = Path("log") / "training" / TRAINING_TIMESTAMP

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Output files
BEST_MODELS_PATH = MODELS_DIR / "best_models.pkl"
ENSEMBLE_MODELS_PATH = MODELS_DIR / "ensemble_models.pkl"
TRAINING_RESULTS_PATH = RESULTS_DIR / "training_results.json"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
LOG_FILE_PATH = LOG_DIR / f"{TRAINING_TIMESTAMP}.log"

# Training parameters
DATASET_CONFIG = {
    'use_pca': True,                    # Whether to use PCA version of dataset
    'target_column': 'target_log',      # Target column name in dataset
    'validation_size': 0.18,            # Validation split from training data
    'random_state': 42
}

# Cross-validation settings
CV_CONFIG = {
    'n_folds': 5,
    'scoring': 'neg_root_mean_squared_error',
    'shuffle': True,
    'random_state': 42
}

# Hyperparameter optimization settings
OPTUNA_CONFIG = {
    'n_trials': 100,
    'timeout': 7200,  # 2 hours
    'pruner': 'median',
    'pruning_percentile': 25,
    'random_state': 42,
    'n_jobs': -1
}

# Model configurations
MODEL_CONFIGS = {
    'linear_models': {
        'LinearRegression': {
            'enabled': True,
            'params': {},
            'param_distributions': {}
        },
        'Ridge': {
            'enabled': True,
            'params': {},
            'param_distributions': {
                'alpha': ['float', 0.1, 100.0, 'log']
            }
        },
        'Lasso': {
            'enabled': True,
            'params': {},
            'param_distributions': {
                'alpha': ['float', 0.001, 10.0, 'log']
            }
        },
        'ElasticNet': {
            'enabled': True,
            'params': {},
            'param_distributions': {
                'alpha': ['float', 0.001, 10.0, 'log'],
                'l1_ratio': ['float', 0.1, 0.9]
            }
        }
    },
    'tree_models': {
        'RandomForest': {
            'enabled': True,
            'params': {'random_state': 42},
            'param_distributions': {
                'n_estimators': ['int', 50, 300],
                'max_depth': ['int', 3, 20],
                'min_samples_split': ['int', 2, 20],
                'min_samples_leaf': ['int', 1, 10],
                'max_features': ['categorical', ['auto', 'sqrt', 'log2']]
            }
        },
        'GradientBoosting': {
            'enabled': True,
            'params': {'random_state': 42},
            'param_distributions': {
                'n_estimators': ['int', 50, 300],
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'max_depth': ['int', 3, 10],
                'min_samples_split': ['int', 2, 20],
                'min_samples_leaf': ['int', 1, 10]
            }
        },
        'HistGradientBoosting': {
            'enabled': True,
            'params': {'random_state': 42},
            'param_distributions': {
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'max_iter': ['int', 50, 300],
                'max_depth': ['int', 3, 15],
                'min_samples_leaf': ['int', 10, 100]
            }
        }
    },
    'boosting_models': {
        'XGBoost': {
            'enabled': True,
            'params': {'random_state': 42, 'verbosity': 0},
            'param_distributions': {
                'n_estimators': ['int', 50, 300],
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'max_depth': ['int', 3, 10],
                'min_child_weight': ['int', 1, 10],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.6, 1.0],
                'reg_alpha': ['float', 0.0, 10.0],
                'reg_lambda': ['float', 1.0, 10.0]
            }
        },
        'LightGBM': {
            'enabled': True,
            'params': {'random_state': 42, 'verbose': -1},
            'param_distributions': {
                'n_estimators': ['int', 50, 300],
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'max_depth': ['int', 3, 15],
                'num_leaves': ['int', 10, 100],
                'min_child_samples': ['int', 5, 100],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.6, 1.0],
                'reg_alpha': ['float', 0.0, 10.0],
                'reg_lambda': ['float', 1.0, 10.0]
            }
        },
        'CatBoost': {
            'enabled': True,
            'params': {'random_state': 42, 'verbose': False},
            'param_distributions': {
                'iterations': ['int', 50, 300],
                'learning_rate': ['float', 0.01, 0.3, 'log'],
                'depth': ['int', 3, 10],
                'l2_leaf_reg': ['float', 1.0, 10.0],
                'border_count': ['int', 32, 255]
            }
        }
    },
    'other_models': {
        'SVR': {
            'enabled': False,  # Disabled by default (slow)
            'params': {},
            'param_distributions': {
                'C': ['float', 0.1, 100.0, 'log'],
                'epsilon': ['float', 0.01, 1.0],
                'gamma': ['categorical', ['scale', 'auto']]
            }
        },
        'KNeighbors': {
            'enabled': False,  # Disabled by default (not great for regression)
            'params': {},
            'param_distributions': {
                'n_neighbors': ['int', 3, 20],
                'weights': ['categorical', ['uniform', 'distance']],
                'algorithm': ['categorical', ['auto', 'ball_tree', 'kd_tree']]
            }
        }
    }
}

# Ensemble configurations
ENSEMBLE_CONFIG = {
    'voting_regressor': {
        'enabled': True,
        'n_best_models': 3,  # Use top 3 models
        'weights': None      # Equal weights, or 'performance' for weighted
    },
    'stacking_regressor': {
        'enabled': True,
        'n_base_models': 5,  # Use top 5 as base models
        'meta_learner': 'Ridge',
        'cv_folds': 3
    }
}

# Feature importance analysis
FEATURE_IMPORTANCE_CONFIG = {
    'analyze_importance': True,
    'importance_methods': ['permutation', 'built_in'],
    'n_top_features': 20,
    'create_plots': True
}

# Early stopping and performance tracking
PERFORMANCE_CONFIG = {
    'early_stopping_patience': 20,  # For models that support it
    'metric_direction': 'minimize',  # For RMSE
    'save_best_only': True,
    'track_training_history': True
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_to_console': True
}