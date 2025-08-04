"""
Package principale src per la pipeline di Machine Learning.

Questo package contiene tutti i moduli principali della pipeline:
- db: Connessione e recupero dati dal database
- dataset: Costruzione e gestione dataset
- preprocessing: Pipeline di preprocessing modulare
- training: Training e valutazione modelli
- utils: Utilità condivise (logger, I/O, path management, error handling)
"""

# Import di utilità comuni per facilitare l'accesso
from .utils.logger import setup_logger, get_logger
from .utils.io import load_config, save_config, check_file_exists
from .utils.path_manager import PathManager, create_path_manager
from .utils.error_handling import (
    PreprocessingError, 
    DataValidationError, 
    ConfigurationError, 
    PipelineExecutionError,
    safe_execution,
    validate_dataframe,
    validate_target_column,
    validate_config
)
from .utils.pipeline_orchestrator import PipelineOrchestrator

# Import delle pipeline principali
from .preprocessing.pipeline_modular import run_modular_preprocessing_pipeline
from .preprocessing.pipeline import run_preprocessing_pipeline

# Version info
__version__ = "2.0.0"
__author__ = "ML Pipeline Team"
__description__ = "Pipeline modulare e ristrutturata per Machine Learning"

# Esporta principali funzionalità
__all__ = [
    # Logger
    'setup_logger', 'get_logger',
    
    # I/O
    'load_config', 'save_config', 'check_file_exists',
    
    # Path management
    'PathManager', 'create_path_manager',
    
    # Error handling
    'PreprocessingError', 'DataValidationError', 'ConfigurationError', 
    'PipelineExecutionError', 'safe_execution',
    'validate_dataframe', 'validate_target_column', 'validate_config',
    
    # Pipeline orchestration
    'PipelineOrchestrator',
    
    # Preprocessing pipelines
    'run_modular_preprocessing_pipeline', 'run_preprocessing_pipeline',
    
    # Meta info
    '__version__', '__author__', '__description__'
]