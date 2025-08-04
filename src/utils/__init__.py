"""
Package utils per utilità condivise nella pipeline di Machine Learning.

Contiene:
- logger: Setup e gestione del logging
- io: Funzioni di input/output (caricamento config, salvataggio file)
- path_manager: Gestione centralizzata dei path
- error_handling: Gestione degli errori e validazione
- pipeline_orchestrator: Orchestrazione centrale della pipeline
- validation: Validazione dati e configurazioni
"""

# Import delle utilità principali
from .logger import setup_logger, get_logger
from .io import (
    load_config, save_config, 
    load_dataframe, save_dataframe,
    load_model, save_model,
    load_json, save_json,
    ensure_dir, check_file_exists
)
from .path_manager import PathManager, create_path_manager
from .error_handling import (
    PreprocessingError,
    DataValidationError, 
    ConfigurationError,
    PipelineExecutionError,
    safe_execution,
    validate_dataframe,
    validate_target_column,
    validate_config,
    ValidationContext
)
from .pipeline_orchestrator import PipelineOrchestrator
from .validation import check_target_leakage, validate_category_distribution

__version__ = "2.0.0"

__all__ = [
    # Logger
    'setup_logger', 'get_logger',
    
    # I/O operations
    'load_config', 'save_config',
    'load_dataframe', 'save_dataframe', 
    'load_model', 'save_model',
    'load_json', 'save_json',
    'ensure_dir', 'check_file_exists',
    
    # Path management
    'PathManager', 'create_path_manager',
    
    # Error handling e validation
    'PreprocessingError', 'DataValidationError', 'ConfigurationError', 'PipelineExecutionError',
    'safe_execution',
    'validate_dataframe', 'validate_target_column', 'validate_config',
    'ValidationContext',
    
    # Pipeline orchestration
    'PipelineOrchestrator',
    
    # Validation utilities
    'check_target_leakage', 'validate_category_distribution'
]