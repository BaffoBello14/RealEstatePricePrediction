"""
Configurazione pytest e fixtures per i test della ML Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any
import sys
import os

# Aggiungi src al path per import
sys.path.append(str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_config():
    """Configurazione di test standard."""
    return {
        'database': {
            'schema_path': 'data/db_schema.json',
            'selected_aliases': ['A', 'AI', 'PC']
        },
        'paths': {
            'data_raw': 'tests/data/raw/',
            'data_processed': 'tests/data/processed/',
            'models': 'tests/models/',
            'logs': 'tests/logs/'
        },
        'target': {
            'column': 'AI_Prezzo_Ridistribuito'
        },
        'preprocessing': {
            'steps': {
                'enable_specific_columns_removal': True,
                'enable_constant_columns_removal': True,
                'enable_cramers_analysis': True,
                'enable_auto_numeric_conversion': True,
                'enable_correlation_removal': True,
                'enable_advanced_encoding': True,
                'enable_feature_scaling': True,
                'enable_log_transformation': True,
                'enable_outlier_detection': True,
                'enable_pca': False
            },
            'columns_to_remove': ['A_Id', 'A_Codice'],
            'constant_column_threshold': 0.95,
            'auto_numeric_threshold': 0.8,
            'use_temporal_split': True,
            'year_column': 'A_AnnoStipula',
            'month_column': 'A_MeseStipula',
            'cramer_threshold': 0.95,
            'corr_threshold': 0.95,
            'low_cardinality_threshold': 10,
            'high_cardinality_max': 100,
            'outlier_strategy': 'category_stratified',
            'category_column': 'AI_IdCategoriaCatastale',
            'z_threshold': 2.5,
            'iqr_multiplier': 1.5,
            'isolation_contamination': 0.05,
            'min_methods_outlier': 2,
            'min_samples_per_category': 30,
            'use_pca': False,
            'pca_variance_threshold': 0.95,
            'val_size': 0.22,
            'test_size': 0.1,
            'random_state': 42,
            'use_stratified_split': False,
            'stratification_quantiles': 5
        },
        'training': {
            'cv_folds': 3,  # Ridotto per test
            'use_time_series_cv': False,
            'optimization_metric': 'neg_root_mean_squared_error',
            'optimization_direction': 'maximize',
            'ranking_metric': 'Val_RMSE',
            'ranking_ascending': True,
            'n_trials': 5,  # Ridotto per test
            'optuna_timeout': 60,  # Ridotto per test
            'n_jobs': 1,  # Single core per test
            'random_state': 42
        },
        'models': {
            'baseline': {
                'linear_regression': True,
                'ridge': True,
                'lasso': False,  # Semplificato per test
                'elastic_net': False,
                'decision_tree': True,
                'knn': False,
                'svr': False
            },
            'advanced': {
                'random_forest': True,
                'gradient_boosting': False,  # Semplificato per test
                'xgboost': False,
                'catboost': False,
                'lightgbm': False,
                'hist_gradient_boosting': False
            },
            'ensemble': {
                'voting_regressor': False,  # Semplificato per test
                'stacking_regressor': False
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'tests/logs/test_pipeline.log'
        },
        'execution': {
            'steps': ['retrieve_data', 'build_dataset', 'preprocessing', 'training', 'evaluation'],
            'force_reload': True
        }
    }

@pytest.fixture
def sample_dataframe():
    """DataFrame di test con dati realistici."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        # Colonne di identità
        'A_Id': range(1, n_samples + 1),
        'A_Codice': [f'ACT{i:04d}' for i in range(1, n_samples + 1)],
        'AI_Id': range(1, n_samples + 1),
        
        # Colonne temporali
        'A_AnnoStipula': np.random.choice([2020, 2021, 2022, 2023], n_samples),
        'A_MeseStipula': np.random.choice(range(1, 13), n_samples),
        
        # Colonne categoriche
        'AI_IdCategoriaCatastale': np.random.choice(['A/2', 'A/3', 'A/4', 'C/1', 'C/2'], n_samples),
        'PC_Comune': np.random.choice(['Milano', 'Roma', 'Napoli', 'Torino'], n_samples),
        'AI_Piano': np.random.choice(['PT', 'P1', 'P2', 'P1-2', 'S1', 'PT-P1'], n_samples),
        
        # Colonne numeriche
        'AI_Superficie': np.random.normal(80, 20, n_samples),
        'AI_Vani': np.random.choice([2, 3, 4, 5], n_samples),
        'AI_Bagni': np.random.choice([1, 2], n_samples),
        
        # Colonne costanti (per test rimozione)
        'Constant_Col': ['SAME_VALUE'] * n_samples,
        
        # Colonne correlate (per test correlazioni)
        'Corr_Col_1': np.random.normal(100, 10, n_samples),
        
        # Target
        'AI_Prezzo_Ridistribuito': np.random.lognormal(11, 0.5, n_samples)  # Prezzi realistici
    }
    
    # Crea colonna correlata a Corr_Col_1
    data['Corr_Col_2'] = data['Corr_Col_1'] * 1.02 + np.random.normal(0, 0.1, n_samples)
    
    # Aggiungi valori mancanti
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.1).index, 'AI_Superficie'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'PC_Comune'] = np.nan
    
    return df

@pytest.fixture
def sample_schema():
    """Schema database di test."""
    return {
        "Atti": {
            "alias": "A",
            "columns": [
                {"name": "Id", "type": "int", "retrieve": True},
                {"name": "Codice", "type": "varchar", "retrieve": True},
                {"name": "AnnoStipula", "type": "int", "retrieve": True},
                {"name": "MeseStipula", "type": "int", "retrieve": True}
            ]
        },
        "AttiImmobili": {
            "alias": "AI", 
            "columns": [
                {"name": "Id", "type": "int", "retrieve": True},
                {"name": "IdCategoriaCatastale", "type": "varchar", "retrieve": True},
                {"name": "Piano", "type": "varchar", "retrieve": True},
                {"name": "Superficie", "type": "float", "retrieve": True},
                {"name": "Vani", "type": "int", "retrieve": True},
                {"name": "Prezzo_Ridistribuito", "type": "float", "retrieve": True}
            ]
        }
    }

@pytest.fixture
def temp_dir():
    """Directory temporanea per test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_config_file(temp_dir, sample_config):
    """File di configurazione temporaneo."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return str(config_path)

@pytest.fixture
def test_schema_file(temp_dir, sample_schema):
    """File schema temporaneo."""
    schema_path = temp_dir / "test_schema.json"
    with open(schema_path, 'w') as f:
        json.dump(sample_schema, f)
    return str(schema_path)

@pytest.fixture
def mock_engine():
    """Mock engine per database."""
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    engine.connect.return_value.__exit__.return_value = None
    return engine

@pytest.fixture
def sample_preprocessing_paths(temp_dir):
    """Path di preprocessing di test."""
    base_path = temp_dir / "processed"
    base_path.mkdir(exist_ok=True)
    
    return {
        'train_features': str(base_path / 'X_train.parquet'),
        'val_features': str(base_path / 'X_val.parquet'),
        'test_features': str(base_path / 'X_test.parquet'),
        'train_target': str(base_path / 'y_train.parquet'),
        'val_target': str(base_path / 'y_val.parquet'),
        'test_target': str(base_path / 'y_test.parquet'),
        'val_target_orig': str(base_path / 'y_val_orig.parquet'),
        'test_target_orig': str(base_path / 'y_test_orig.parquet'),
        'preprocessing_info': str(base_path / 'preprocessing_info.json')
    }

@pytest.fixture
def sample_features_and_target():
    """Features e target di test per ML."""
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples) + X.iloc[:, 0], name='target')
    
    return X, y

@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir):
    """Setup automatico per ogni test."""
    # Crea directory necessarie
    for subdir in ['data/raw', 'data/processed', 'models', 'logs']:
        (temp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Patch delle variabili d'ambiente per evitare accesso DB reale
    with patch.dict(os.environ, {
        'SERVER': 'test_server',
        'DATABASE': 'test_db', 
        'USER': 'test_user',
        'PASSWORD': 'test_password'
    }):
        yield

@pytest.fixture
def sample_training_results():
    """Risultati di training simulati."""
    return {
        'baseline_results': {
            'LinearRegression': {
                'cv_score_mean': -0.5,
                'cv_score_std': 0.1,
                'model': MagicMock()
            },
            'Ridge': {
                'cv_score_mean': -0.48,
                'cv_score_std': 0.09,
                'model': MagicMock()
            }
        },
        'optimized_results': {
            'RandomForest': {
                'best_score': -0.45,
                'best_params': {'n_estimators': 100, 'max_depth': 10},
                'model': MagicMock()
            }
        },
        'ensemble_results': {},
        'best_models': {
            'best_baseline': {
                'name': 'Ridge',
                'score': -0.48,
                'model': MagicMock()
            },
            'best_advanced': {
                'name': 'RandomForest', 
                'score': -0.45,
                'model': MagicMock()
            }
        },
        'training_summary': {
            'total_models_trained': 3,
            'best_overall_score': -0.45,
            'training_time_seconds': 120
        }
    }

# Utility functions per test
def create_sample_parquet_files(paths_dict, X_train, X_val, X_test, y_train, y_val, y_test):
    """Crea file parquet di esempio."""
    X_train.to_parquet(paths_dict['train_features'], index=False)
    X_val.to_parquet(paths_dict['val_features'], index=False)
    X_test.to_parquet(paths_dict['test_features'], index=False)
    y_train.to_frame().to_parquet(paths_dict['train_target'], index=False)
    y_val.to_frame().to_parquet(paths_dict['val_target'], index=False)
    y_test.to_frame().to_parquet(paths_dict['test_target'], index=False)
    y_val.to_frame().to_parquet(paths_dict['val_target_orig'], index=False)
    y_test.to_frame().to_parquet(paths_dict['test_target_orig'], index=False)

def assert_dataframe_properties(df, expected_shape=None, expected_columns=None, no_nulls=None):
    """Utility per validare proprietà DataFrame."""
    if expected_shape:
        assert df.shape == expected_shape, f"Expected shape {expected_shape}, got {df.shape}"
    
    if expected_columns:
        assert list(df.columns) == expected_columns, f"Column mismatch"
    
    if no_nulls:
        null_cols = df.columns[df.isnull().any()].tolist()
        assert len(null_cols) == 0, f"Found nulls in columns: {null_cols}"