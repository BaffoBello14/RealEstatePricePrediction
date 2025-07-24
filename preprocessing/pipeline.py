"""
Main preprocessing pipeline that orchestrates all preprocessing steps
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_loader.load_data_2 import load_data_dual_omi
from preprocessing.feature_engineering import (
    remove_features, encode_categorical_features, 
    impute_missing_values, scale_features
)
from preprocessing.outlier_detection import (
    detect_outliers_grouped, detect_outliers_global,
    visualize_outliers, visualize_outliers_by_group
)
from preprocessing.dimensionality_reduction import (
    apply_pca, visualize_pca_variance, analyze_pca_components
)

logger = logging.getLogger(__name__)


def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['level']),
        format=config['format'],
        handlers=[
            logging.FileHandler(config.get('log_file', 'preprocessing.log')),
            logging.StreamHandler()
        ]
    )


def load_and_split_data(config):
    """
    Load data and split into train/test sets
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df_full)
    """
    logger.info("Loading data...")
    
    # Load data
    df = load_data_dual_omi(
        config['SCHEMA_PATH'], 
        config['DATA_LOADING_CONFIG']['selected_aliases']
    )
    
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Check if target column exists
    target_col = config['TARGET_COLUMN']
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['TEST_SIZE'],
        random_state=config['RANDOM_STATE'],
        stratify=None  # For regression, no stratification
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, df


def preprocess_features(X_train, X_test, y_train, config):
    """
    Apply feature preprocessing steps
    
    Args:
        X_train, X_test: Training and test features
        y_train: Training target
        config: Configuration dictionary
        
    Returns:
        tuple: (X_train_processed, X_test_processed, preprocessing_artifacts)
    """
    logger.info("Starting feature preprocessing...")
    
    artifacts = {}
    
    # 1. Feature removal
    if any(config['FEATURE_REMOVAL_CONFIG'].values()):
        logger.info("Applying feature removal...")
        X_train_clean, removal_report = remove_features(X_train, config['FEATURE_REMOVAL_CONFIG'])
        # Apply same removals to test set
        cols_to_remove = []
        for removal_type, removed_cols in removal_report.items():
            if isinstance(removed_cols, list):
                cols_to_remove.extend(removed_cols)
        
        X_test_clean = X_test.drop(columns=cols_to_remove, errors='ignore')
        artifacts['feature_removal'] = removal_report
        
        logger.info(f"Features after removal: {X_train_clean.shape[1]}")
    else:
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        artifacts['feature_removal'] = {}
    
    # 2. Imputation
    logger.info("Applying imputation...")
    X_train_imputed, imputers = impute_missing_values(X_train_clean, config['IMPUTATION_CONFIG'])
    
    # Apply same imputers to test set
    X_test_imputed = X_test_clean.copy()
    
    # Apply numerical imputer
    if 'numerical' in imputers:
        num_cols = X_test_imputed.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            X_test_imputed[num_cols] = imputers['numerical'].transform(X_test_imputed[num_cols])
    
    # Apply categorical imputer
    if 'categorical' in imputers:
        cat_cols = X_test_imputed.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            X_test_imputed[cat_cols] = imputers['categorical'].transform(X_test_imputed[cat_cols].astype(str))
    
    artifacts['imputers'] = imputers
    
    # 3. Categorical encoding
    logger.info("Applying categorical encoding...")
    X_train_encoded, encoding_info = encode_categorical_features(
        X_train_imputed, y_train, config['ENCODING_CONFIG']
    )
    
    # Apply same encoders to test set
    X_test_encoded = X_test_imputed.copy()
    for col, encoder in encoding_info['encoders'].items():
        if col in X_test_encoded.columns:
            if hasattr(encoder, 'transform'):
                if col in encoding_info['target_encoded']:
                    # Target encoder
                    X_test_encoded[col] = encoder.transform(
                        X_test_encoded[col].fillna('Missing').values.reshape(-1, 1)
                    ).ravel()
                else:
                    # Label encoder
                    # Handle unseen categories
                    test_values = X_test_encoded[col].fillna('Missing').astype(str)
                    encoded_values = []
                    for val in test_values:
                        if val in encoder.classes_:
                            encoded_values.append(encoder.transform([val])[0])
                        else:
                            # Assign to most frequent class
                            encoded_values.append(0)
                    X_test_encoded[col] = encoded_values
    
    artifacts['encoding_info'] = encoding_info
    
    logger.info("Feature preprocessing completed")
    return X_train_encoded, X_test_encoded, artifacts


def apply_outlier_detection(X_train, y_train, df_train, config):
    """
    Apply outlier detection
    
    Args:
        X_train: Training features (scaled)
        y_train: Training target
        df_train: Training DataFrame (for grouping column)
        config: Configuration dictionary
        
    Returns:
        tuple: (y_train_log, outliers_mask, outlier_detector, outlier_stats)
    """
    if not config['OUTLIER_CONFIG']['apply_outlier_detection']:
        logger.info("Outlier detection disabled")
        y_train_log = np.log1p(y_train)
        outliers_mask = np.zeros(len(y_train), dtype=bool)
        return y_train_log, outliers_mask, None, {}
    
    # Prepare data for outlier detection
    temp_df = df_train.copy()
    temp_df[config['TARGET_COLUMN']] = y_train
    
    # Apply grouped or global outlier detection
    if (config['OUTLIER_CONFIG']['group_by_categorical'] and 
        config['CATEGORICAL_GROUP_COLUMN'] in temp_df.columns):
        
        logger.info(f"Applying grouped outlier detection by {config['CATEGORICAL_GROUP_COLUMN']}")
        y_train_log, outliers_mask, outlier_detector, outlier_stats = detect_outliers_grouped(
            temp_df, 
            config['TARGET_COLUMN'],
            config['CATEGORICAL_GROUP_COLUMN'],
            X_train,
            config['OUTLIER_CONFIG']
        )
        
        # Visualize outliers by group
        if outliers_mask.sum() > 0:
            visualize_outliers_by_group(
                temp_df, y_train_log, outliers_mask, 
                config['CATEGORICAL_GROUP_COLUMN']
            )
        
    else:
        logger.info("Applying global outlier detection")
        y_train_log, outliers_mask, outlier_detector = detect_outliers_global(
            temp_df,
            config['TARGET_COLUMN'],
            X_train,
            config['OUTLIER_CONFIG']
        )
        outlier_stats = {}
    
    # Visualize outliers
    visualize_outliers(
        temp_df, y_train, y_train_log, outliers_mask, 
        config['TARGET_COLUMN'], config['CATEGORICAL_GROUP_COLUMN']
    )
    
    return y_train_log, outliers_mask, outlier_detector, outlier_stats


def apply_scaling_and_pca(X_train_clean, X_test, y_train_clean, config):
    """
    Apply scaling and optionally PCA
    
    Args:
        X_train_clean: Clean training features
        X_test: Test features
        y_train_clean: Clean training target
        config: Configuration dictionary
        
    Returns:
        dict: Results with both PCA and no-PCA versions
    """
    logger.info("Applying scaling...")
    
    # Scale features
    X_train_scaled, scaler = scale_features(X_train_clean, config['SCALING_CONFIG'])
    X_test_scaled, _ = scale_features(X_test, config['SCALING_CONFIG'], scaler)
    
    results = {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler
    }
    
    # Apply PCA if configured
    if config['PCA_CONFIG']['apply_pca']:
        logger.info("Applying PCA...")
        
        X_train_pca, X_test_pca, pca_model, loadings_df = apply_pca(
            X_train_scaled, X_test_scaled, 
            {**config['PCA_CONFIG'], 'random_state': config['RANDOM_STATE']}
        )
        
        # Analyze PCA components
        pca_analysis = analyze_pca_components(pca_model, loadings_df)
        
        # Visualize PCA
        visualize_pca_variance(pca_model)
        
        results.update({
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca,
            'pca_model': pca_model,
            'loadings_df': loadings_df,
            'pca_analysis': pca_analysis
        })
    
    return results


def save_datasets_and_artifacts(results, y_train_clean, y_test, y_train_original, 
                               outliers_mask, artifacts, config):
    """
    Save processed datasets and all artifacts
    
    Args:
        results: Processing results
        y_train_clean: Clean training target
        y_test: Test target
        y_train_original: Original training target
        outliers_mask: Outlier mask
        artifacts: Preprocessing artifacts
        config: Configuration dictionary
    """
    logger.info("Saving datasets and artifacts...")
    
    # Create datasets
    datasets = {}
    
    # PCA version (if available)
    if 'X_train_pca' in results:
        pca_cols = [f"PC{i+1}" for i in range(results['X_train_pca'].shape[1])]
        
        # Training set
        train_pca_df = pd.DataFrame(results['X_train_pca'], columns=pca_cols)
        train_pca_df['target_log'] = y_train_clean.values
        train_pca_df['target_original'] = y_train_original.loc[~outliers_mask].values
        
        # Test set
        test_pca_df = pd.DataFrame(results['X_test_pca'], columns=pca_cols)
        test_pca_df['target_log'] = np.log1p(y_test).values
        test_pca_df['target_original'] = y_test.values
        
        datasets['train_pca'] = train_pca_df
        datasets['test_pca'] = test_pca_df
        
        # Save to CSV
        train_pca_df.to_csv(config['TRAIN_PATH'], index=False)
        test_pca_df.to_csv(config['TEST_PATH'], index=False)
        
        logger.info(f"PCA datasets saved: train {train_pca_df.shape}, test {test_pca_df.shape}")
    
    # No-PCA version (if configured)
    if config['PCA_CONFIG']['create_no_pca_version']:
        # Training set
        train_no_pca_df = pd.DataFrame(results['X_train_scaled'])
        train_no_pca_df['target_log'] = y_train_clean.values
        train_no_pca_df['target_original'] = y_train_original.loc[~outliers_mask].values
        
        # Test set
        test_no_pca_df = pd.DataFrame(results['X_test_scaled'])
        test_no_pca_df['target_log'] = np.log1p(y_test).values
        test_no_pca_df['target_original'] = y_test.values
        
        datasets['train_no_pca'] = train_no_pca_df
        datasets['test_no_pca'] = test_no_pca_df
        
        # Save to CSV
        train_no_pca_df.to_csv(config['TRAIN_NO_PCA_PATH'], index=False)
        test_no_pca_df.to_csv(config['TEST_NO_PCA_PATH'], index=False)
        
        logger.info(f"No-PCA datasets saved: train {train_no_pca_df.shape}, test {test_no_pca_df.shape}")
    
    # Save transformers
    transformers = {
        'scaler': results['scaler'],
        'imputers': artifacts['imputers'],
        'encoding_info': artifacts['encoding_info'],
        'timestamp': config['TIMESTAMP']
    }
    
    if 'pca_model' in results:
        transformers['pca_model'] = results['pca_model']
    
    if 'outlier_detector' in results:
        transformers['outlier_detector'] = results['outlier_detector']
    
    with open(config['TRANSFORMERS_PATH'], 'wb') as f:
        pickle.dump(transformers, f)
    
    logger.info(f"Transformers saved to {config['TRANSFORMERS_PATH']}")
    
    # Create preprocessing report
    report = {
        'timestamp': config['TIMESTAMP'],
        'configuration': {
            'target_column': config['TARGET_COLUMN'],
            'categorical_group_column': config['CATEGORICAL_GROUP_COLUMN'],
            'test_size': config['TEST_SIZE'],
            'random_state': config['RANDOM_STATE'],
            'feature_removal': config['FEATURE_REMOVAL_CONFIG'],
            'pca_config': config['PCA_CONFIG'],
            'outlier_config': config['OUTLIER_CONFIG'],
            'encoding_config': config['ENCODING_CONFIG'],
            'imputation_config': config['IMPUTATION_CONFIG'],
            'scaling_config': config['SCALING_CONFIG']
        },
        'data_info': {
            'original_shape': artifacts.get('original_shape', 'unknown'),
            'final_train_shape': [len(y_train_clean), results['X_train_scaled'].shape[1]],
            'final_test_shape': [len(y_test), results['X_test_scaled'].shape[1]],
            'outliers_removed': int(outliers_mask.sum()),
            'outliers_percentage': float(outliers_mask.sum() / len(outliers_mask) * 100)
        },
        'feature_removal': artifacts['feature_removal'],
        'encoding_info': {
            'label_encoded': artifacts['encoding_info'].get('label_encoded', []),
            'target_encoded': artifacts['encoding_info'].get('target_encoded', []),
            'onehot_encoded': artifacts['encoding_info'].get('onehot_encoded', [])
        },
        'transformations': {}
    }
    
    # Add PCA info if available
    if 'pca_analysis' in results:
        report['transformations'].update({
            'pca_applied': True,
            'pca_components': int(results['pca_model'].n_components_),
            'pca_variance_explained': float(results['pca_analysis']['total_variance_explained']),
            'pca_individual_variance': results['pca_analysis']['individual_variance']
        })
    else:
        report['transformations']['pca_applied'] = False
    
    # Save report
    with open(config['PREPROCESSING_REPORT_PATH'], 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Preprocessing report saved to {config['PREPROCESSING_REPORT_PATH']}")
    
    return datasets, report


def run_preprocessing_pipeline(config):
    """
    Run the complete preprocessing pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: All results and artifacts
    """
    logger.info(f"Starting preprocessing pipeline - Timestamp: {config['TIMESTAMP']}")
    
    # 1. Load and split data
    X_train, X_test, y_train, y_test, df_full = load_and_split_data(config)
    
    # 2. Preprocess features
    X_train_processed, X_test_processed, artifacts = preprocess_features(
        X_train, X_test, y_train, config
    )
    
    # Store original target for later use
    y_train_original = y_train.copy()
    
    # 3. Scale features (needed for outlier detection)
    X_train_scaled_temp, scaler_temp = scale_features(
        X_train_processed, config['SCALING_CONFIG']
    )
    
    # 4. Outlier detection
    df_train = df_full.loc[X_train.index].copy()
    y_train_log, outliers_mask, outlier_detector, outlier_stats = apply_outlier_detection(
        X_train_scaled_temp, y_train, df_train, config
    )
    
    # 5. Remove outliers from training data
    X_train_clean = X_train_processed[~outliers_mask]
    y_train_clean = y_train_log[~outliers_mask]
    
    logger.info(f"Training set after outlier removal: {X_train_clean.shape[0]} samples")
    
    # 6. Apply final scaling and PCA
    results = apply_scaling_and_pca(X_train_clean, X_test_processed, y_train_clean, config)
    
    # Add outlier detector to results
    if outlier_detector is not None:
        results['outlier_detector'] = outlier_detector
    
    # 7. Save everything
    datasets, report = save_datasets_and_artifacts(
        results, y_train_clean, y_test, y_train_original, 
        outliers_mask, artifacts, config
    )
    
    # Final summary
    logger.info("Preprocessing pipeline completed successfully!")
    logger.info(f"Final training set: {results['X_train_scaled'].shape}")
    logger.info(f"Final test set: {results['X_test_scaled'].shape}")
    if 'X_train_pca' in results:
        logger.info(f"PCA components: {results['X_train_pca'].shape[1]}")
        logger.info(f"PCA variance explained: {results['pca_analysis']['total_variance_explained']:.3f}")
    logger.info(f"Outliers removed: {outliers_mask.sum()} ({outliers_mask.sum()/len(outliers_mask)*100:.2f}%)")
    
    return {
        'datasets': datasets,
        'results': results,
        'artifacts': artifacts,
        'report': report,
        'outliers_mask': outliers_mask,
        'outlier_stats': outlier_stats
    }