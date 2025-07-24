#!/usr/bin/env python3
"""
Main script to run the preprocessing pipeline
"""
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.preprocessing_config import *
from preprocessing.pipeline import run_preprocessing_pipeline, setup_logging

def main():
    """Run the preprocessing pipeline"""
    
    # Create configuration dictionary
    config = {
        # Paths and versioning
        'TIMESTAMP': TIMESTAMP,
        'SCHEMA_PATH': SCHEMA_PATH,
        'TRAIN_PATH': TRAIN_PATH,
        'TEST_PATH': TEST_PATH,
        'TRAIN_NO_PCA_PATH': TRAIN_NO_PCA_PATH,
        'TEST_NO_PCA_PATH': TEST_NO_PCA_PATH,
        'PREPROCESSING_REPORT_PATH': PREPROCESSING_REPORT_PATH,
        'TRANSFORMERS_PATH': TRANSFORMERS_PATH,
        'LOG_FILE_PATH': LOG_FILE_PATH,
        
        # Model parameters
        'TARGET_COLUMN': TARGET_COLUMN,
        'CATEGORICAL_GROUP_COLUMN': CATEGORICAL_GROUP_COLUMN,
        'TEST_SIZE': TEST_SIZE,
        'RANDOM_STATE': RANDOM_STATE,
        
        # Configuration sections
        'FEATURE_REMOVAL_CONFIG': FEATURE_REMOVAL_CONFIG,
        'PCA_CONFIG': PCA_CONFIG,
        'OUTLIER_CONFIG': OUTLIER_CONFIG,
        'ENCODING_CONFIG': ENCODING_CONFIG,
        'IMPUTATION_CONFIG': IMPUTATION_CONFIG,
        'SCALING_CONFIG': SCALING_CONFIG,
        'DATA_LOADING_CONFIG': DATA_LOADING_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG
    }
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['LOGGING_CONFIG']['level']),
        format=config['LOGGING_CONFIG']['format'],
        handlers=[
            logging.FileHandler(config['LOG_FILE_PATH']),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {config['TIMESTAMP']}")
    logger.info(f"Target column: {config['TARGET_COLUMN']}")
    logger.info(f"Categorical grouping column: {config['CATEGORICAL_GROUP_COLUMN']}")
    logger.info(f"Test size: {config['TEST_SIZE']}")
    logger.info(f"Random state: {config['RANDOM_STATE']}")
    logger.info("")
    
    logger.info("Feature Removal Settings:")
    for key, value in config['FEATURE_REMOVAL_CONFIG'].items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    logger.info("PCA Settings:")
    for key, value in config['PCA_CONFIG'].items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    logger.info("Outlier Detection Settings:")
    for key, value in config['OUTLIER_CONFIG'].items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    logger.info("Encoding Settings:")
    for key, value in config['ENCODING_CONFIG'].items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    logger.info("=" * 80)
    
    try:
        # Run the preprocessing pipeline
        results = run_preprocessing_pipeline(config)
        
        # Print final summary
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        report = results['report']
        logger.info(f"Original dataset shape: {report['data_info'].get('original_shape', 'unknown')}")
        logger.info(f"Final training set: {report['data_info']['final_train_shape']}")
        logger.info(f"Final test set: {report['data_info']['final_test_shape']}")
        logger.info(f"Outliers removed: {report['data_info']['outliers_removed']} ({report['data_info']['outliers_percentage']:.2f}%)")
        
        if report['transformations']['pca_applied']:
            logger.info(f"PCA components: {report['transformations']['pca_components']}")
            logger.info(f"PCA variance explained: {report['transformations']['pca_variance_explained']:.3f}")
        
        logger.info(f"Feature removal summary:")
        removal = report['feature_removal']
        total_removed = sum(len(v) if isinstance(v, list) else 0 for v in removal.values())
        logger.info(f"  Total features removed: {total_removed}")
        logger.info(f"  Duplicate columns: {len(removal.get('duplicate_columns', []))}")
        logger.info(f"  Constant columns: {len(removal.get('constant_columns', []))}")
        logger.info(f"  Low variance columns: {len(removal.get('low_variance_columns', []))}")
        logger.info(f"  High correlation numerical: {len(removal.get('high_corr_numerical', []))}")
        logger.info(f"  High correlation categorical: {len(removal.get('high_corr_categorical', []))}")
        
        logger.info("")
        logger.info("Output files:")
        if config['PCA_CONFIG']['apply_pca']:
            logger.info(f"  Training set (PCA): {config['TRAIN_PATH']}")
            logger.info(f"  Test set (PCA): {config['TEST_PATH']}")
        if config['PCA_CONFIG']['create_no_pca_version']:
            logger.info(f"  Training set (No PCA): {config['TRAIN_NO_PCA_PATH']}")
            logger.info(f"  Test set (No PCA): {config['TEST_NO_PCA_PATH']}")
        logger.info(f"  Transformers: {config['TRANSFORMERS_PATH']}")
        logger.info(f"  Preprocessing report: {config['PREPROCESSING_REPORT_PATH']}")
        logger.info(f"  Log file: {config['LOG_FILE_PATH']}")
        
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    results = main()