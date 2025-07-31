"""
Pipeline di preprocessing orchestrata.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_dataframe, save_json

from .cleaning import clean_data, remove_specific_columns, remove_constant_columns, convert_to_numeric, transform_target_and_detect_outliers, detect_outliers_multimethod
from .filtering import analyze_cramers_correlations, filter_features
from .encoding import encode_features
from .imputation import handle_missing_values
from .transformation import split_dataset_with_validation, apply_feature_scaling, apply_pca_transformation, transform_target_log

logger = get_logger(__name__)

def run_preprocessing_pipeline(
    dataset_path: str,
    target_column: str,
    config: Dict[str, Any],
    output_paths: Dict[str, str]
) -> None:
    """
    Esegue la pipeline completa di preprocessing con la nuova sequenza ottimizzata.
    
    Args:
        dataset_path: Path al dataset da processare
        target_column: Nome della colonna target
        config: Configurazione del preprocessing
        output_paths: Dictionary con i path di output
    """
    logger.info("=== AVVIO PIPELINE PREPROCESSING RISTRUTTURATA ===")
    
    try:
        steps_config = config.get('steps', {})
        preprocessing_info = {'config_used': config, 'steps_info': {}}
        
        # ===== STEP 1: CARICAMENTO DATI =====
        logger.info("Step 1: Caricamento dataset...")
        df = load_dataframe(dataset_path)
        logger.info(f"Dataset caricato: {df.shape}")
        preprocessing_info['dataset_original_shape'] = list(df.shape)
        
        # ===== STEP 2: PULIZIA DATI =====
        logger.info("Step 2: Pulizia dati...")
        df, cleaning_info = clean_data(df, target_column, config)
        preprocessing_info['steps_info']['cleaning'] = cleaning_info
        
        # ===== STEP 3: ANALISI CRAMER (se abilitato) =====
        if steps_config.get('enable_cramers_analysis', True):
            logger.info("Step 3: Analisi correlazioni Cramér's V...")
            cramer_threshold = config.get('cramer_threshold', 0.95)
            cramers_analysis = analyze_cramers_correlations(df, target_column, cramer_threshold)
            preprocessing_info['steps_info']['cramers_analysis'] = cramers_analysis
        else:
            logger.info("Step 3: Analisi Cramér's V DISABILITATA")
            preprocessing_info['steps_info']['cramers_analysis'] = {'skipped': True}
        
        # ===== STEP 4: CONVERSIONE AUTOMATICA A NUMERICO (se abilitato) =====
        if steps_config.get('enable_auto_numeric_conversion', True):
            logger.info("Step 4: Conversione automatica a numerico...")
            auto_numeric_threshold = config.get('auto_numeric_threshold', 0.8)
            df, conversion_info = convert_to_numeric(df, target_column, auto_numeric_threshold)
            preprocessing_info['steps_info']['numeric_conversion'] = conversion_info
        else:
            logger.info("Step 4: Conversione automatica a numerico DISABILITATA")
            preprocessing_info['steps_info']['numeric_conversion'] = {'skipped': True}
        
        # ===== STEP 5: ENCODING CATEGORICHE DI BASE (se abilitato) =====
        if steps_config.get('enable_advanced_encoding', True):
            logger.info("Step 5: Encoding categoriche di base (senza target encoding)...")
            df, encoding_info = encode_features(
                df, 
                target_column,
                low_card_threshold=config.get('low_cardinality_threshold', 10),
                high_card_max=config.get('high_cardinality_max', 100),
                random_state=config.get('random_state', 42),
                apply_target_encoding=False  # EVITA TARGET ENCODING PRE-SPLIT
            )
            preprocessing_info['steps_info']['basic_encoding'] = {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                                    for k, v in encoding_info.items()}
        else:
            logger.info("Step 5: Encoding categoriche di base DISABILITATO")
            preprocessing_info['steps_info']['basic_encoding'] = {'skipped': True}
        
        # ===== STEP 6: IMPUTAZIONE VALORI NULLI =====
        logger.info("Step 6: Imputazione valori mancanti...")
        df, imputation_info = handle_missing_values(df, target_column)
        preprocessing_info['steps_info']['imputation'] = {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                              for k, v in imputation_info.items()}
        
        # ===== STEP 7: RIMOZIONE FEATURE CORRELATE PRE-SPLIT (se abilitato) =====
        if steps_config.get('enable_correlation_removal', True):
            logger.info("Step 7: Rimozione feature altamente correlate...")
            # Usa la funzione di filtering esistente per rimozione correlazioni numeriche
            from .filtering import remove_highly_correlated_numeric
            
            # Separazione temporanea per analisi correlazioni
            X_temp = df.drop(columns=[target_column])
            numeric_cols = X_temp.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                X_numeric = X_temp[numeric_cols]
                # Applica rimozione solo alle colonne numeriche
                _, _, removed_numeric_cols = remove_highly_correlated_numeric(
                    X_numeric, None, config.get('corr_threshold', 0.95)
                )
                
                if removed_numeric_cols:
                    df = df.drop(columns=removed_numeric_cols)
                    logger.info(f"Rimosse {len(removed_numeric_cols)} colonne numeriche correlate")
                
                correlation_removal_info = {
                    'removed_columns': removed_numeric_cols,
                    'threshold_used': config.get('corr_threshold', 0.95)
                }
            else:
                correlation_removal_info = {'removed_columns': [], 'note': 'Meno di 2 colonne numeriche'}
            
            preprocessing_info['steps_info']['correlation_removal'] = correlation_removal_info
        else:
            logger.info("Step 7: Rimozione feature correlate DISABILITATA")
            preprocessing_info['steps_info']['correlation_removal'] = {'skipped': True}
        
        # ===== STEP 8: SPLIT TRAIN/TEST/VAL =====
        logger.info("Step 8: Train/Validation/Test Split...")
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = split_dataset_with_validation(
            df, 
            target_column,
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.2),
            random_state=config.get('random_state', 42),
            use_temporal_split=config.get('use_temporal_split', True),
            date_column=config.get('date_column', 'A_AnnoStipula')
        )
        
        split_info = {
            'train_shape': list(X_train.shape),
            'val_shape': list(X_val.shape),
            'test_shape': list(X_test.shape),
            'use_temporal_split': config.get('use_temporal_split', True)
        }
        preprocessing_info['steps_info']['split'] = split_info
        
        # ===== STEP 9: TARGET ENCODING POST-SPLIT (se abilitato) =====
        if steps_config.get('enable_advanced_encoding', True):
            logger.info("Step 9: Target encoding post-split...")
            X_train, X_val, X_test, target_encoding_info = apply_target_encoding_post_split(
                X_train, X_val, X_test, y_train,
                low_card_threshold=config.get('low_cardinality_threshold', 10),
                high_card_max=config.get('high_cardinality_max', 100),
                random_state=config.get('random_state', 42)
            )
            preprocessing_info['steps_info']['target_encoding'] = {k: str(v) if not isinstance(v, (list, dict, str, int, float)) else v 
                                   for k, v in target_encoding_info.items()}
        else:
            logger.info("Step 9: Target encoding DISABILITATO")
            preprocessing_info['steps_info']['target_encoding'] = {'skipped': True}
        
        # ===== STEP 10: FEATURE SCALING (se abilitato) =====
        if steps_config.get('enable_feature_scaling', True):
            logger.info("Step 10: Feature scaling...")
            scaling_results = apply_feature_scaling(X_train, X_val, X_test)
            
            X_train_scaled = scaling_results[0]
            X_val_scaled = scaling_results[1] if len(scaling_results) > 2 else None
            X_test_scaled = scaling_results[2] if len(scaling_results) > 3 else scaling_results[1]
            scaling_info = scaling_results[-1]
            
            preprocessing_info['steps_info']['scaling'] = {
                'original_features': scaling_info['original_features'],
                'scaled_features': scaling_info['scaled_features']
            }
        else:
            logger.info("Step 10: Feature scaling DISABILITATO")
            X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
            preprocessing_info['steps_info']['scaling'] = {'skipped': True}
        
        # ===== STEP 11: TRASFORMAZIONE LOG + OUTLIER DETECTION (se abilitato) =====
        enable_log = steps_config.get('enable_log_transformation', True)
        enable_outliers = steps_config.get('enable_outlier_detection', True)
        use_separate_functions = steps_config.get('use_separate_log_outlier_functions', False)
        
        if enable_log or enable_outliers:
            if use_separate_functions:
                # Usa funzioni separate per maggiore flessibilità
                logger.info("Step 11: Usando funzioni separate per trasformazione log e outlier detection...")
                
                if enable_log:
                    logger.info("Step 11a: Trasformazione logaritmica target...")
                    y_train_log, transform_info = transform_target_log(y_train)
                    preprocessing_info['steps_info']['log_transformation'] = transform_info
                else:
                    y_train_log = y_train
                    preprocessing_info['steps_info']['log_transformation'] = {'skipped': True}
                
                if enable_outliers:
                    logger.info("Step 11b: Outlier detection...")
                    outliers_mask, outlier_info = detect_outliers_multimethod(
                        y_train_log,
                        X_train_scaled,
                        z_threshold=config.get('z_threshold', 3.0),
                        iqr_multiplier=config.get('iqr_multiplier', 1.5),
                        contamination=config.get('isolation_contamination', 0.1),
                        min_methods=config.get('min_methods_outlier', 2)
                    )
                    preprocessing_info['steps_info']['outlier_detection'] = outlier_info
                else:
                    outliers_mask = np.zeros(len(y_train_log), dtype=bool)
                    preprocessing_info['steps_info']['outlier_detection'] = {'skipped': True}
                
                # Combina info per compatibilità
                outlier_detector = {
                    'method': 'separate_functions',
                    'transform_info': preprocessing_info['steps_info'].get('log_transformation', {}),
                    'outlier_info': preprocessing_info['steps_info'].get('outlier_detection', {})
                }
            else:
                # Usa funzione combinata (comportamento originale)
                logger.info("Step 11: Trasformazione log target + outlier detection (funzione combinata)...")
                y_train_log, outliers_mask, outlier_detector = transform_target_and_detect_outliers(
                    y_train,
                    X_train_scaled,
                    z_threshold=config.get('z_threshold', 3.0),
                    iqr_multiplier=config.get('iqr_multiplier', 1.5),
                    contamination=config.get('isolation_contamination', 0.1),
                    min_methods=config.get('min_methods_outlier', 2)
                )
            
            # Rimozione outliers dal training set
            outliers_count = outliers_mask.sum()
            logger.info(f"Rimozione di {outliers_count} outliers dal training set "
                       f"({outliers_count/len(y_train)*100:.2f}%)")
            
            X_train_clean = X_train_scaled[~outliers_mask]
            y_train_clean = y_train_log[~outliers_mask]
            
            # Applica la stessa trasformazione logaritmica a validation e test
            y_val_log = np.log1p(y_val)
            y_test_log = np.log1p(y_test)
            
            outlier_info = {
                'outliers_removed': int(outliers_count),
                'outliers_percentage': float(outliers_count/len(y_train)*100),
                'final_train_size': X_train_clean.shape[0]
            }
            preprocessing_info['steps_info']['outlier_detection'] = outlier_info
            
        elif steps_config.get('enable_log_transformation', True):
            logger.info("Step 11: Solo trasformazione log target (outlier detection disabilitata)...")
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)
            y_test_log = np.log1p(y_test)
            X_train_clean = X_train_scaled
            y_train_clean = y_train_log
            preprocessing_info['steps_info']['outlier_detection'] = {'outliers_removed': 0, 'log_transform_only': True}
        else:
            logger.info("Step 11: Trasformazione log e outlier detection DISABILITATE")
            y_train_log = y_train
            y_val_log = y_val
            y_test_log = y_test
            X_train_clean = X_train_scaled
            y_train_clean = y_train_log
            preprocessing_info['steps_info']['outlier_detection'] = {'skipped': True}
        
        # ===== STEP 12: PCA (se abilitato) =====
        if steps_config.get('enable_pca', False):
            logger.info("Step 12: Applicazione PCA...")
            pca_results = apply_pca_transformation(
                X_train_clean,
                X_val_scaled,
                X_test_scaled,
                variance_threshold=config.get('pca_variance_threshold', 0.95),
                random_state=config.get('random_state', 42)
            )
            
            X_train_final = pca_results[0]
            X_val_final = pca_results[1] if len(pca_results) > 2 else None
            X_test_final = pca_results[2] if len(pca_results) > 3 else pca_results[1]
            pca_info = pca_results[-1]
            
            preprocessing_info['steps_info']['pca'] = {
                'original_features': pca_info['original_features'],
                'pca_components': pca_info['pca_components'],
                'variance_explained': pca_info['variance_explained']
            }
            
            feature_columns = pca_info['component_names']
        else:
            logger.info("Step 12: PCA DISABILITATA")
            X_train_final = X_train_clean
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
            preprocessing_info['steps_info']['pca'] = {'skipped': True}
            feature_columns = X_train_clean.columns.tolist()
        
        # ===== STEP 13: SALVATAGGIO RISULTATI =====
        logger.info("Step 13: Salvataggio risultati...")
        
        # Converte a DataFrame se necessario
        if not isinstance(X_train_final, pd.DataFrame):
            X_train_final = pd.DataFrame(X_train_final, columns=feature_columns)
        if not isinstance(X_val_final, pd.DataFrame):
            X_val_final = pd.DataFrame(X_val_final, columns=feature_columns)
        if not isinstance(X_test_final, pd.DataFrame):
            X_test_final = pd.DataFrame(X_test_final, columns=feature_columns)
        
        # Prepara target per salvataggio
        y_train_df = pd.DataFrame({'target_log': y_train_clean})
        y_val_df = pd.DataFrame({'target_log': y_val_log})
        y_test_df = pd.DataFrame({'target_log': y_test_log})
        
        # Target in scala originale per evaluation
        y_val_orig_df = pd.DataFrame({'target_original': y_val_orig})
        y_test_orig_df = pd.DataFrame({'target_original': y_test_orig})
        
        # Salva i file
        save_dataframe(X_train_final, output_paths['train_features'])
        save_dataframe(X_val_final, output_paths['val_features'])
        save_dataframe(X_test_final, output_paths['test_features'])
        save_dataframe(y_train_df, output_paths['train_target'])
        save_dataframe(y_val_df, output_paths['val_target'])
        save_dataframe(y_test_df, output_paths['test_target'])
        save_dataframe(y_val_orig_df, output_paths['val_target_orig'])
        save_dataframe(y_test_orig_df, output_paths['test_target_orig'])
        
        # Completa informazioni preprocessing
        preprocessing_info.update({
            'target_column': target_column,
            'final_train_shape': [X_train_final.shape[0], X_train_final.shape[1]],
            'final_val_shape': list(X_val_final.shape),
            'final_test_shape': list(X_test_final.shape),
            'feature_columns': feature_columns
        })
        
        save_json(preprocessing_info, output_paths['preprocessing_info'])
        
        logger.info("=== PREPROCESSING COMPLETATO CON SUCCESSO ===")
        logger.info(f"Dataset finale: {X_train_final.shape[0]} train + {X_val_final.shape[0]} val + {X_test_final.shape[0]} test")
        logger.info(f"Features finali: {X_train_final.shape[1]}")
        
        # Mostra summary degli step eseguiti
        logger.info("=== SUMMARY STEP ESEGUITI ===")
        for step_name, step_info in preprocessing_info['steps_info'].items():
            if 'skipped' in step_info:
                logger.info(f"  {step_name}: DISABILITATO")
            else:
                logger.info(f"  {step_name}: ESEGUITO")
        
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