"""
Package preprocessing per la pipeline di Machine Learning.

Contiene tutti i moduli per il preprocessing dei dati:
- pipeline: Pipeline legacy (retrocompatibilità)
- pipeline_modular: Nuova pipeline modulare
- steps: Moduli per i singoli step del preprocessing
- Core modules: cleaning, encoding, transformation, etc.
"""

# Import delle pipeline principali
from .pipeline import run_preprocessing_pipeline
from .pipeline_modular import run_modular_preprocessing_pipeline, PreprocessingPipeline

# Import dei core modules
from .data_cleaning_core import (
    convert_to_numeric_unified,
    clean_dataframe_unified, 
    remove_constant_columns_unified
)
from .target_processing_core import (
    transform_target_log,
    detect_outliers_univariate,
    detect_outliers_multivariate
)
from .target_utils import (
    determine_target_scale_and_get_original,
    create_target_scale_metadata
)

# Import dei moduli step
from .steps import (
    load_and_validate_dataset,
    execute_data_cleaning_step,
    execute_feature_analysis_step
)

# Import delle funzioni legacy per retrocompatibilità
from .cleaning import convert_to_numeric, clean_data, transform_target_and_detect_outliers
from .encoding import auto_convert_to_numeric
from .imputation import impute_missing_values, handle_missing_values
from .transformation import split_dataset_with_validation, apply_feature_scaling, apply_pca_transformation
from .filtering import cramers_v, analyze_cramers_correlations, remove_highly_correlated_numeric_pre_split

__version__ = "2.0.0"

__all__ = [
    # Pipeline principali
    'run_preprocessing_pipeline',
    'run_modular_preprocessing_pipeline', 
    'PreprocessingPipeline',
    
    # Core unified functions
    'convert_to_numeric_unified',
    'clean_dataframe_unified',
    'remove_constant_columns_unified',
    'transform_target_log',
    'detect_outliers_univariate', 
    'detect_outliers_multivariate',
    'determine_target_scale_and_get_original',
    'create_target_scale_metadata',
    
    # Modular steps
    'load_and_validate_dataset',
    'execute_data_cleaning_step',
    'execute_feature_analysis_step',
    
    # Legacy functions (retrocompatibilità)
    'convert_to_numeric', 'clean_data', 'transform_target_and_detect_outliers',
    'auto_convert_to_numeric',
    'impute_missing_values', 'handle_missing_values',
    'split_dataset_with_validation', 'apply_feature_scaling', 'apply_pca_transformation',
    'cramers_v', 'analyze_cramers_correlations', 'remove_highly_correlated_numeric_pre_split'
]