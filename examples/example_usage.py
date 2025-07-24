#!/usr/bin/env python3
"""
Example usage of the preprocessing pipeline with different configurations
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.preprocessing_config import *
from preprocessing.pipeline import run_preprocessing_pipeline

def example_basic_preprocessing():
    """Example: Basic preprocessing with all defaults"""
    print("=" * 60)
    print("EXAMPLE 1: Basic preprocessing with defaults")
    print("=" * 60)
    
    config = {
        'TIMESTAMP': TIMESTAMP,
        'SCHEMA_PATH': SCHEMA_PATH,
        'TRAIN_PATH': TRAIN_PATH,
        'TEST_PATH': TEST_PATH,
        'TRAIN_NO_PCA_PATH': TRAIN_NO_PCA_PATH,
        'TEST_NO_PCA_PATH': TEST_NO_PCA_PATH,
        'PREPROCESSING_REPORT_PATH': PREPROCESSING_REPORT_PATH,
        'TRANSFORMERS_PATH': TRANSFORMERS_PATH,
        'LOG_FILE_PATH': LOG_FILE_PATH,
        'TARGET_COLUMN': TARGET_COLUMN,
        'CATEGORICAL_GROUP_COLUMN': CATEGORICAL_GROUP_COLUMN,
        'TEST_SIZE': TEST_SIZE,
        'RANDOM_STATE': RANDOM_STATE,
        'FEATURE_REMOVAL_CONFIG': FEATURE_REMOVAL_CONFIG,
        'PCA_CONFIG': PCA_CONFIG,
        'OUTLIER_CONFIG': OUTLIER_CONFIG,
        'ENCODING_CONFIG': ENCODING_CONFIG,
        'IMPUTATION_CONFIG': IMPUTATION_CONFIG,
        'SCALING_CONFIG': SCALING_CONFIG,
        'DATA_LOADING_CONFIG': DATA_LOADING_CONFIG,
        'LOGGING_CONFIG': LOGGING_CONFIG
    }
    
    print("Configuration: Default settings with grouped outlier detection")
    print(f"- Outlier detection grouped by: {config['CATEGORICAL_GROUP_COLUMN']}")
    print(f"- PCA enabled: {config['PCA_CONFIG']['apply_pca']}")
    print(f"- Create no-PCA version: {config['PCA_CONFIG']['create_no_pca_version']}")
    
    return config


def example_no_feature_removal():
    """Example: Disable all feature removal"""
    print("=" * 60)
    print("EXAMPLE 2: No feature removal")
    print("=" * 60)
    
    # Start with basic config
    config = example_basic_preprocessing()
    
    # Disable all feature removal
    config['FEATURE_REMOVAL_CONFIG'] = {
        'remove_duplicates': False,
        'remove_constants': False,
        'remove_high_correlation': False,
        'remove_low_variance': False,
        'cramer_threshold': 0.85,
        'corr_threshold': 0.95,
        'variance_threshold': 0.01
    }
    
    print("Configuration: All feature removal disabled")
    print("- Keep all original features")
    
    return config


def example_global_outlier_detection():
    """Example: Global outlier detection (not grouped)"""
    print("=" * 60)
    print("EXAMPLE 3: Global outlier detection")
    print("=" * 60)
    
    # Start with basic config
    config = example_basic_preprocessing()
    
    # Change to global outlier detection
    config['OUTLIER_CONFIG'] = {
        'apply_outlier_detection': True,
        'group_by_categorical': False,  # Changed to False
        'z_threshold': 3.0,
        'iqr_multiplier': 1.5,
        'isolation_contamination': 0.05,
        'min_methods_outlier': 2,
        'min_group_size': 50
    }
    
    print("Configuration: Global outlier detection")
    print("- Outlier detection on entire dataset (not grouped)")
    
    return config


def example_no_pca():
    """Example: Disable PCA completely"""
    print("=" * 60)
    print("EXAMPLE 4: No PCA")
    print("=" * 60)
    
    # Start with basic config
    config = example_basic_preprocessing()
    
    # Disable PCA
    config['PCA_CONFIG'] = {
        'apply_pca': False,
        'variance_threshold': 0.95,
        'create_no_pca_version': True  # Still create the scaled version
    }
    
    print("Configuration: PCA disabled")
    print("- Only scaled features, no dimensionality reduction")
    
    return config


def example_strict_feature_removal():
    """Example: Very strict feature removal"""
    print("=" * 60)
    print("EXAMPLE 5: Strict feature removal")
    print("=" * 60)
    
    # Start with basic config
    config = example_basic_preprocessing()
    
    # More strict feature removal
    config['FEATURE_REMOVAL_CONFIG'] = {
        'remove_duplicates': True,
        'remove_constants': True,
        'remove_high_correlation': True,
        'remove_low_variance': True,
        'cramer_threshold': 0.7,   # Lower threshold (more strict)
        'corr_threshold': 0.8,     # Lower threshold (more strict)
        'variance_threshold': 0.05  # Higher threshold (more strict)
    }
    
    print("Configuration: Strict feature removal")
    print("- Lower correlation thresholds")
    print("- Higher variance threshold")
    
    return config


def example_conservative_outlier_detection():
    """Example: More conservative outlier detection"""
    print("=" * 60)
    print("EXAMPLE 6: Conservative outlier detection")
    print("=" * 60)
    
    # Start with basic config
    config = example_basic_preprocessing()
    
    # More conservative outlier detection
    config['OUTLIER_CONFIG'] = {
        'apply_outlier_detection': True,
        'group_by_categorical': True,
        'z_threshold': 4.0,        # Higher threshold (more conservative)
        'iqr_multiplier': 2.0,     # Higher multiplier (more conservative)
        'isolation_contamination': 0.03,  # Lower contamination (more conservative)
        'min_methods_outlier': 3,  # All methods must agree
        'min_group_size': 100      # Larger minimum group size
    }
    
    print("Configuration: Conservative outlier detection")
    print("- Higher thresholds, all methods must agree")
    print("- Larger minimum group size")
    
    return config


def run_example(example_func, run_pipeline=False):
    """Run an example configuration"""
    config = example_func()
    
    if run_pipeline:
        print("\nRunning preprocessing pipeline...")
        try:
            results = run_preprocessing_pipeline(config)
            print("✓ Preprocessing completed successfully!")
            print(f"✓ Results saved to: {config['TRAIN_PATH'].parent}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("\nConfiguration created (not executed)")
        print("To run this configuration, call:")
        print("run_example(example_func, run_pipeline=True)")
    
    print("\n")


if __name__ == "__main__":
    print("ML Pipeline - Configuration Examples")
    print("=" * 60)
    print()
    
    # Show all examples (without running)
    examples = [
        example_basic_preprocessing,
        example_no_feature_removal,
        example_global_outlier_detection,
        example_no_pca,
        example_strict_feature_removal,
        example_conservative_outlier_detection
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Running example {i}:")
        run_example(example, run_pipeline=False)
    
    print("=" * 60)
    print("To execute any of these examples, modify this script and set")
    print("run_pipeline=True for the desired example.")
    print("=" * 60)