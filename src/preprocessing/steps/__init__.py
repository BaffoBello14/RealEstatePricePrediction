"""
Moduli step per la pipeline di preprocessing modularizzata.
Ogni step è una funzione specializzata che fa una cosa specifica.
"""

from .data_loading import load_and_validate_dataset
from .data_cleaning import execute_data_cleaning_step
from .feature_analysis import execute_feature_analysis_step
from .data_encoding import execute_encoding_step
from .data_imputation import execute_imputation_step
from .feature_filtering import execute_feature_filtering_step
from .data_splitting import execute_data_splitting_step
from .target_processing import execute_target_processing_step
from .feature_scaling import execute_feature_scaling_step
from .dimensionality_reduction import execute_pca_step

__all__ = [
    'load_and_validate_dataset',
    'execute_data_cleaning_step',
    'execute_feature_analysis_step', 
    'execute_encoding_step',
    'execute_imputation_step',
    'execute_feature_filtering_step',
    'execute_data_splitting_step',
    'execute_target_processing_step',
    'execute_feature_scaling_step',
    'execute_pca_step'
]