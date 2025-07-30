"""
Pipeline di preprocessing orchestrata.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_dataframe, save_json

from .cleaning import clean_data, transform_target_and_detect_outliers
from .filtering import filter_features
from .encoding import encode_features
from .imputation import handle_missing_values
from .transformation import split_dataset_with_validation, apply_transformations

logger = get_logger(__name__)

def run_preprocessing_pipeline(
    dataset_path: str,
    target_column: str,
    config: Dict[str, Any],
    output_paths: Dict[str, str]
) -> None:
    """
    Esegue la pipeline completa di preprocessing.
    
    Args:
        dataset_path: Path al dataset da processare
        target_column: Nome della colonna target
        config: Configurazione del preprocessing
        output_paths: Dictionary con i path di output
    """
    logger.info("=== AVVIO PIPELINE PREPROCESSING ===")
    
    try:
        # 1. Caricamento dati
        logger.info("Step 1: Caricamento dataset...")
        df = load_dataframe(dataset_path)
        logger.info(f"Dataset caricato: {df.shape}")
        
        # 2. Pulizia dati
        logger.info("Step 2: Pulizia dati...")
        df, cleaning_info = clean_data(df, target_column, config)
        
        # 3. Conversione automatica a numerico e encoding BASE (senza target encoding)
        logger.info("Step 3: Encoding features di base...")
        df, encoding_info = encode_features(
            df, 
            target_column,
            low_card_threshold=config.get('low_cardinality_threshold', 10),
            high_card_max=config.get('high_cardinality_max', 100),
            random_state=config.get('random_state', 42),
            apply_target_encoding=False  # EVITA TARGET ENCODING PRE-SPLIT
        )
        
        # 4. Imputazione valori mancanti
        logger.info("Step 4: Imputazione valori mancanti...")
        df, imputation_info = handle_missing_values(df, target_column)
        
        # 5. Train/Validation/Test Split PRIMA dei filtri di correlazione
        logger.info("Step 5: Train/Validation/Test Split...")
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = split_dataset_with_validation(
            df, 
            target_column,
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.2),
            random_state=config.get('random_state', 42),
            use_temporal_split=config.get('use_temporal_split', True),
            date_column=config.get('date_column', 'A_DataStipula')
        )
        
        # 6. Target encoding SOLO sui dati di training (post-split)
        logger.info("Step 6: Target encoding post-split...")
        X_train, X_val, X_test, target_encoding_info = apply_target_encoding_post_split(
            X_train, X_val, X_test, y_train,
            low_card_threshold=config.get('low_cardinality_threshold', 10),
            high_card_max=config.get('high_cardinality_max', 100),
            random_state=config.get('random_state', 42)
        )
        
        # 7. Filtro correlazioni (post-split, solo su training)
        logger.info("Step 7: Filtro correlazioni...")
        filter_results = filter_features(
            df=None,  # Non usato per filtri post-split
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            cramer_threshold=config.get('cramer_threshold', 0.95),
            corr_threshold=config.get('corr_threshold', 0.95)
        )
        
        # Estrai i risultati (_, X_train, X_val, X_test, filter_info)
        _, X_train, X_val, X_test, filter_info = filter_results
        
        # 8. Trasformazioni (scaling + PCA)
        logger.info("Step 8: Trasformazioni (scaling + PCA)...")
        X_train_transformed, X_val_transformed, X_test_transformed, transformation_info = apply_transformations(
            X_train,
            X_val, 
            X_test,
            use_pca=config.get('use_pca', False),
            pca_variance_threshold=config.get('pca_variance_threshold', 0.95),
            random_state=config.get('random_state', 42)
        )
        
        # 9. Trasformazione target e rimozione outliers (solo su train)
        logger.info("Step 9: Trasformazione target e outlier detection...")
        y_train_log, outliers_mask, outlier_detector = transform_target_and_detect_outliers(
            y_train,
            X_train_transformed,
            z_threshold=config.get('z_threshold', 3.0),
            iqr_multiplier=config.get('iqr_multiplier', 1.5),
            contamination=config.get('isolation_contamination', 0.1),
            min_methods=config.get('min_methods_outlier', 2)
        )
        
        # 10. Rimozione outliers dal training set
        logger.info("Step 10: Rimozione outliers dal training set...")
        outliers_count = outliers_mask.sum()
        logger.info(f"Rimozione di {outliers_count} outliers dal training set "
                   f"({outliers_count/len(y_train)*100:.2f}%)")
        
        X_train_clean = X_train_transformed[~outliers_mask]
        y_train_clean = y_train_log[~outliers_mask]
        
        logger.info(f"Training set finale: {X_train_clean.shape[0]} righe")
        
        # 11. Validazioni qualità dati
        logger.info("Step 11: Validazioni qualità dati...")
        from ..utils.validation import run_all_validations
        
        validation_results = run_all_validations(
            X_train_transformed, X_val_transformed, X_test_transformed, 
            y_train, config
        )
        
        # Salva risultati validazioni
        if not validation_results['summary']['all_valid']:
            logger.warning("ATTENZIONE: Alcuni controlli qualità sono falliti. Controllare i log.")
        
        # 12. Salvataggio risultati
        logger.info("Step 12: Salvataggio risultati...")
        
        # Determina i nomi delle colonne in base al tipo di trasformazione
        if transformation_info.get('use_pca', False):
            # Con PCA: usa nomi PC1, PC2, etc.
            feature_columns = [f'PC{i+1}' for i in range(X_train_clean.shape[1])]
        else:
            # Senza PCA: mantieni i nomi originali delle colonne
            feature_columns = X_train.columns.tolist()
        
        # Converte array numpy in DataFrame per il salvataggio
        X_train_df = pd.DataFrame(
            X_train_clean, 
            columns=feature_columns
        )
        X_val_df = pd.DataFrame(
            X_val_transformed,
            columns=feature_columns
        )
        X_test_df = pd.DataFrame(
            X_test_transformed,
            columns=feature_columns
        )
        # Applica la stessa trasformazione logaritmica a validation e test
        y_val_log = np.log1p(y_val)
        y_test_log = np.log1p(y_test)
        
        y_train_df = pd.DataFrame({'target_log': y_train_clean})
        y_val_df = pd.DataFrame({'target_log': y_val_log})
        y_test_df = pd.DataFrame({'target_log': y_test_log})
        
        # Target in scala originale per evaluation
        y_val_orig_df = pd.DataFrame({'target_original': y_val_orig})
        y_test_orig_df = pd.DataFrame({'target_original': y_test_orig})
        
        # Salva i file
        save_dataframe(X_train_df, output_paths['train_features'])
        save_dataframe(X_val_df, output_paths['val_features'])
        save_dataframe(X_test_df, output_paths['test_features'])
        save_dataframe(y_train_df, output_paths['train_target'])
        save_dataframe(y_val_df, output_paths['val_target'])
        save_dataframe(y_test_df, output_paths['test_target'])
        save_dataframe(y_val_orig_df, output_paths['val_target_orig'])
        save_dataframe(y_test_orig_df, output_paths['test_target_orig'])
        
        # 13. Salva informazioni preprocessing
        logger.info("Step 13: Salvataggio informazioni preprocessing...")
        
        preprocessing_info = {
            'dataset_original_shape': list(df.shape),
            'train_shape': [X_train_clean.shape[0], X_train_clean.shape[1]],
            'val_shape': list(X_val_transformed.shape),
            'test_shape': list(X_test_transformed.shape),
            'target_column': target_column,
            'outliers_removed': int(outliers_count),
            'outliers_percentage': float(outliers_count/len(y_train)*100),
            'cleaning_info': cleaning_info,
            'encoding_info': {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                            for k, v in encoding_info.items()},
            'target_encoding_info': {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                                   for k, v in target_encoding_info.items()},
            'imputation_info': {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                              for k, v in imputation_info.items()},
            'filter_info': filter_info,
            'transformation_info': {
                'original_features': transformation_info['original_features'],
                'transformed_features': transformation_info['transformed_features'],
                'variance_explained': float(transformation_info['variance_explained']),
                'use_pca': transformation_info.get('use_pca', False)
            },
            'validation_results': validation_results,
            'config_used': config
        }
        
        save_json(preprocessing_info, output_paths['preprocessing_info'])
        
        logger.info("=== PREPROCESSING COMPLETATO CON SUCCESSO ===")
        logger.info(f"Dataset finale: {X_train_clean.shape[0]} train + {X_val_transformed.shape[0]} val + {X_test_transformed.shape[0]} test")
        
        if transformation_info.get('use_pca', False):
            logger.info(f"Features finali: {X_train_clean.shape[1]} (PCA)")
            logger.info(f"Varianza preservata: {transformation_info['variance_explained']:.3f}")
        else:
            logger.info(f"Features finali: {X_train_clean.shape[1]} (solo scaling)")
            logger.info("Nessuna perdita di varianza (solo scaling)")
        
    except Exception as e:
        logger.error(f"Errore nella pipeline di preprocessing: {e}")
        raise

def apply_target_encoding_post_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    low_card_threshold: int = 10,
    high_card_max: int = 100,
    random_state: int = 42
) -> tuple:
    """
    Applica target encoding DOPO il split per evitare data leakage.
    
    Args:
        X_train: Features di training
        X_val: Features di validation
        X_test: Features di test
        y_train: Target di training
        low_card_threshold: Soglia per bassa cardinalità
        high_card_max: Soglia massima per alta cardinalità
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train, X_val, X_test encodati e info
    """
    from sklearn.preprocessing import TargetEncoder
    
    logger.info("Applicazione target encoding post-split...")
    
    # Identifica colonne categoriche per target encoding
    cat_cols = X_train.select_dtypes(include='object').columns
    high_card_cols = [
        col for col in cat_cols 
        if X_train[col].nunique() >= low_card_threshold and X_train[col].nunique() < high_card_max
    ]
    
    target_encoders = {}
    encoding_info = {'target_encoded_columns': high_card_cols}
    
    if high_card_cols:
        logger.info(f"Target encoding per {len(high_card_cols)} colonne: {high_card_cols}")
        
        for col in high_card_cols:
            # Fit encoder SOLO sui dati di training
            encoder = TargetEncoder(random_state=random_state)
            X_train[col] = encoder.fit_transform(X_train[[col]], y_train)
            
            # Applica trasformazione a validation e test
            X_val[col] = encoder.transform(X_val[[col]])
            X_test[col] = encoder.transform(X_test[[col]])
            
            target_encoders[col] = encoder
            
        encoding_info['target_encoders'] = target_encoders
        logger.info("Target encoding completato senza data leakage")
    else:
        logger.info("Nessuna colonna per target encoding")
    
    return X_train, X_val, X_test, encoding_info

def load_preprocessed_data(output_paths: Dict[str, str]) -> tuple:
    """
    Carica i dati preprocessati salvati.
    
    Args:
        output_paths: Dictionary con i path dei file
        
    Returns:
        Tuple con X_train, X_test, y_train, y_test, preprocessing_info
    """
    logger.info("Caricamento dati preprocessati...")
    
    try:
        from ..utils.io import load_json
        
        X_train = load_dataframe(output_paths['train_features'])
        X_test = load_dataframe(output_paths['test_features'])
        y_train = load_dataframe(output_paths['train_target'])
        y_test = load_dataframe(output_paths['test_target'])
        preprocessing_info = load_json(output_paths['preprocessing_info'])
        
        logger.info(f"Dati caricati: X_train {X_train.shape}, X_test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, preprocessing_info
        
    except Exception as e:
        logger.error(f"Errore nel caricamento dati preprocessati: {e}")
        raise