"""
Pipeline di preprocessing orchestrata.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_dataframe, save_json

from .cleaning import clean_data, convert_to_numeric, detect_outliers_multimethod, transform_target_and_detect_outliers_by_category
from .filtering import analyze_cramers_correlations
from .encoding import encode_features
from .imputation import handle_missing_values
from .transformation import split_dataset_with_validation, apply_feature_scaling, apply_pca_transformation, transform_target_log

logger = get_logger(__name__)

def identify_categorical_columns(df: pd.DataFrame, target_column: str) -> Dict[str, list]:
    """
    Identifica e categorizza le colonne categoriche per tipo.
    
    Args:
        df: DataFrame da analizzare
        target_column: Nome della colonna target da escludere
        
    Returns:
        Dictionary con liste di colonne per tipo
    """
    cat_cols = df.select_dtypes(include='object').columns
    cat_cols = [col for col in cat_cols if col != target_column]  # Escludi target se categorico
    
    if len(cat_cols) == 0:
        return {'all_categorical': [], 'low_card': [], 'high_card': [], 'very_high_card': []}
    
    # Separazione per cardinalità (usa soglie dal config di default)
    low_card = [col for col in cat_cols if df[col].nunique() < 10]
    high_card = [col for col in cat_cols if 10 <= df[col].nunique() < 100]
    very_high_card = [col for col in cat_cols if df[col].nunique() >= 100]
    
    return {
        'all_categorical': cat_cols,
        'low_card': low_card,
        'high_card': high_card,
        'very_high_card': very_high_card
    }

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
        
        # ===== IDENTIFICAZIONE FEATURE CATEGORICHE PRIMA ENCODING =====
        categorical_info = identify_categorical_columns(df, target_column)
        preprocessing_info['categorical_columns'] = categorical_info
        logger.info(f"Feature categoriche identificate: {len(categorical_info['all_categorical'])} totali")
        
        # Salva una copia del dataset pre-encoding per modelli che supportano categoriche
        df_pre_encoding = df.copy()
        
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
            # Usa la funzione dedicata PRE-SPLIT
            from .filtering import remove_highly_correlated_numeric_pre_split
            
            df, removed_numeric_cols = remove_highly_correlated_numeric_pre_split(
                df, 
                target_column=target_column,
                threshold=config.get('corr_threshold', 0.95)
            )
            
            correlation_removal_info = {
                'removed_columns': removed_numeric_cols,
                'threshold_used': config.get('corr_threshold', 0.95)
            }
            
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
            year_column=config.get('year_column', 'A_AnnoStipula'),
            month_column=config.get('month_column', 'A_MeseStipula'),
            use_stratified_split=config.get('use_stratified_split', False),
            stratification_quantiles=config.get('stratification_quantiles', 5)
        )
        
        # ===== SPLIT ANCHE PER DATI CATEGORICI =====
        logger.info("Step 8b: Split per dati con feature categoriche native...")
        X_train_cat, X_val_cat, X_test_cat, _, _, _, _, _, _ = split_dataset_with_validation(
            df_pre_encoding, 
            target_column,
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.2),
            random_state=config.get('random_state', 42),
            use_temporal_split=config.get('use_temporal_split', True),
            year_column=config.get('year_column', 'A_AnnoStipula'),
            month_column=config.get('month_column', 'A_MeseStipula'),
            use_stratified_split=config.get('use_stratified_split', False),
            stratification_quantiles=config.get('stratification_quantiles', 5)
        )
        logger.info(f"Split categorico completato: X_train_cat {X_train_cat.shape}, X_val_cat {X_val_cat.shape}, X_test_cat {X_test_cat.shape}")
        
        split_info = {
            'train_shape': list(X_train.shape),
            'val_shape': list(X_val.shape),
            'test_shape': list(X_test.shape),
            'use_temporal_split': config.get('use_temporal_split', True),
            'use_stratified_split': config.get('use_stratified_split', False),
            'stratification_quantiles': config.get('stratification_quantiles', 5)
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
            
            # Il return di apply_feature_scaling è: (X_train_scaled, [X_val_scaled], [X_test_scaled], scaling_info)
            # dove X_val_scaled e X_test_scaled sono presenti solo se i dati sono stati forniti
            X_train_scaled = scaling_results[0]
            scaling_info = scaling_results[-1]  # L'ultimo elemento è sempre scaling_info
            
            # Gestisci X_val e X_test in base a cosa è stato restituito
            if len(scaling_results) == 4:  # X_train, X_val, X_test, info
                X_val_scaled = scaling_results[1]
                X_test_scaled = scaling_results[2]
            elif len(scaling_results) == 3:  # X_train, X_val, info (senza X_test)
                X_val_scaled = scaling_results[1]
                X_test_scaled = None
            else:  # len == 2: X_train, info (senza X_val e X_test)
                X_val_scaled = None
                X_test_scaled = None
            
            preprocessing_info['steps_info']['scaling'] = {
                'original_features': scaling_info['original_features'],
                'scaled_features': scaling_info['scaled_features']
            }
        else:
            logger.info("Step 10: Feature scaling DISABILITATO")
            X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
            preprocessing_info['steps_info']['scaling'] = {'skipped': True}
        
        # ===== STEP 11: TRASFORMAZIONE LOGARITMICA (se abilitata) =====
        enable_log = steps_config.get('enable_log_transformation', True)
        
        if enable_log:
            logger.info("Step 11: Trasformazione logaritmica target...")
            y_train_processed, transform_info = transform_target_log(y_train)
            target_is_log_transformed = True
            preprocessing_info['steps_info']['log_transformation'] = transform_info
            
            # Applica la stessa trasformazione ai dati di validation e test
            y_val_processed = np.log1p(y_val)
            y_test_processed = np.log1p(y_test)
            
            logger.info("Trasformazione logaritmica applicata a train, validation e test set")
        else:
            logger.info("Step 11: Trasformazione logaritmica DISABILITATA")
            y_train_processed = y_train
            y_val_processed = y_val
            y_test_processed = y_test
            target_is_log_transformed = False
            preprocessing_info['steps_info']['log_transformation'] = {
                'skipped': True,
                'applied': False,
                'target_files_contain_log_values': False
            }
        
        # ===== STEP 12: RILEVAZIONE OUTLIER (se abilitata) =====
        enable_outliers = steps_config.get('enable_outlier_detection', True)
        
        if enable_outliers:
            logger.info("Step 12: Rilevazione outlier...")
            
            # Strategia outlier detection dal config
            outlier_strategy = config.get('outlier_strategy', 'global')
            
            if outlier_strategy == 'category_stratified':
                # Usa detection stratificata per categoria
                category_column = config.get('category_column', 'AI_IdCategoriaCatastale')
                alternative_category_column = config.get('alternative_category_column', 'CC_Id')
                
                                 # Controlla se la colonna principale esiste, altrimenti usa fallback
                 if category_column not in X_train.columns and alternative_category_column in X_train.columns:
                     logger.info(f"Colonna {category_column} non trovata, usando {alternative_category_column}")
                     category_column = alternative_category_column
                 
                 if category_column in X_train.columns:
                     logger.info(f"Usando outlier detection stratificata per categoria: {category_column}")
                     y_train_processed, outliers_mask, outlier_info = transform_target_and_detect_outliers_by_category(
                         y_train_processed,
                         X_train,  # Usa dati pre-scaling per categoria
                        category_column=category_column,
                        z_threshold=config.get('z_threshold', 2.5),
                        iqr_multiplier=config.get('iqr_multiplier', 1.5),
                        contamination=config.get('isolation_contamination', 0.05),
                        min_methods=config.get('min_methods_outlier', 2),
                        min_samples_per_category=config.get('min_samples_per_category', 30)
                    )
                else:
                    logger.warning(f"Nessuna colonna categoria trovata ({category_column}, {alternative_category_column}). Fallback a detection globale.")
                    outliers_mask, outlier_info = detect_outliers_multimethod(
                        y_train_processed,
                        X_train_scaled,
                        z_threshold=config.get('global_z_threshold', 3.0),
                        iqr_multiplier=config.get('global_iqr_multiplier', 1.5),
                        contamination=config.get('global_isolation_contamination', 0.1),
                        min_methods=config.get('min_methods_outlier', 2)
                    )
            else:
                # Usa detection globale standard
                logger.info("Usando outlier detection globale")
                outliers_mask, outlier_info = detect_outliers_multimethod(
                    y_train_processed,
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
            y_train_clean = y_train_processed[~outliers_mask]
            
            preprocessing_info['steps_info']['outlier_detection'] = outlier_info
        else:
            logger.info("Step 12: Rilevazione outlier DISABILITATA")
            # Nessun outlier rimosso
            X_train_clean = X_train_scaled
            y_train_clean = y_train_processed
            preprocessing_info['steps_info']['outlier_detection'] = {'skipped': True}
        
        # ===== STEP 13: PCA (se abilitato) =====
        if steps_config.get('enable_pca', False):
            logger.info("Step 13: Applicazione PCA...")
            pca_results = apply_pca_transformation(
                X_train_clean,
                X_val_scaled,
                X_test_scaled,
                variance_threshold=config.get('pca_variance_threshold', 0.95),
                random_state=config.get('random_state', 42)
            )
            
            # Il return di apply_pca_transformation ha struttura simile al feature scaling
            X_train_final = pca_results[0]
            pca_info = pca_results[-1]  # L'ultimo elemento è sempre pca_info
            
            # Gestisci X_val e X_test in base a cosa è stato restituito
            if len(pca_results) == 4:  # X_train, X_val, X_test, info
                X_val_final = pca_results[1]
                X_test_final = pca_results[2]
            elif len(pca_results) == 3:  # X_train, X_val, info (senza X_test)
                X_val_final = pca_results[1]
                X_test_final = None
            else:  # len == 2: X_train, info (senza X_val e X_test)
                X_val_final = None
                X_test_final = None
            
            preprocessing_info['steps_info']['pca'] = {
                'original_features': pca_info['original_features'],
                'pca_components': pca_info['pca_components'],
                'variance_explained': pca_info['variance_explained']
            }
            
            feature_columns = pca_info['component_names']
        else:
            logger.info("Step 13: PCA DISABILITATA")
            X_train_final = X_train_clean
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
            preprocessing_info['steps_info']['pca'] = {'skipped': True}
            feature_columns = X_train_clean.columns.tolist()
        
        # ===== STEP 14: SALVATAGGIO RISULTATI =====
        logger.info("Step 14: Salvataggio risultati...")
        
        # Converte a DataFrame se necessario
        if not isinstance(X_train_final, pd.DataFrame):
            X_train_final = pd.DataFrame(X_train_final, columns=feature_columns)
        if X_val_final is not None and not isinstance(X_val_final, pd.DataFrame):
            X_val_final = pd.DataFrame(X_val_final, columns=feature_columns)
        if X_test_final is not None and not isinstance(X_test_final, pd.DataFrame):
            X_test_final = pd.DataFrame(X_test_final, columns=feature_columns)
        
        # Prepara target per salvataggio con nomi appropriati
        if target_is_log_transformed:
            target_column_name = 'target_log'
            logger.info("Salvando target con trasformazione logaritmica applicata")
        else:
            target_column_name = 'target_original'
            logger.info("Salvando target senza trasformazione logaritmica (valori originali)")
        
        y_train_df = pd.DataFrame({target_column_name: y_train_clean})
        y_val_df = pd.DataFrame({target_column_name: y_val_processed})
        y_test_df = pd.DataFrame({target_column_name: y_test_processed})
        
        # Target in scala originale per evaluation
        y_val_orig_df = pd.DataFrame({'target_original': y_val_orig})
        y_test_orig_df = pd.DataFrame({'target_original': y_test_orig})
        
        # Prepara anche dati categorici finali (con stesse trasformazioni tranne encoding)
        # Applica le stesse trasformazioni ai dati categorici (scaling, outlier removal, etc.)
        
        # Per ora manteniamo i dati categorici dalla versione pre-encoding
        # ma con gli stessi indici dei dati finali per consistenza
        if X_train_final is not None:
            X_train_cat_final = X_train_cat.loc[X_train_final.index] if hasattr(X_train_final, 'index') else X_train_cat
        else:
            X_train_cat_final = None
            
        if X_val_final is not None:
            X_val_cat_final = X_val_cat.loc[X_val_final.index] if hasattr(X_val_final, 'index') else X_val_cat
        else:
            X_val_cat_final = None
            
        if X_test_final is not None:
            X_test_cat_final = X_test_cat.loc[X_test_final.index] if hasattr(X_test_final, 'index') else X_test_cat
        else:
            X_test_cat_final = None

        # Salva i file usando batch saving
        dataframes_to_save = {
            'train_features': X_train_final,
            'val_features': X_val_final,
            'test_features': X_test_final,
            'train_features_categorical': X_train_cat_final,
            'val_features_categorical': X_val_cat_final,
            'test_features_categorical': X_test_cat_final,
            'train_target': y_train_df,
            'val_target': y_val_df,
            'test_target': y_test_df,
            'val_target_orig': y_val_orig_df,
            'test_target_orig': y_test_orig_df
        }
        
        from ..utils.pipeline_utils import DataLoader
        DataLoader.save_multiple_dataframes(dataframes_to_save, output_paths)
        
        # Completa informazioni preprocessing
        preprocessing_info.update({
            'target_column': target_column,
            'final_train_shape': [X_train_final.shape[0], X_train_final.shape[1]],
            'final_val_shape': list(X_val_final.shape) if X_val_final is not None else None,
            'final_test_shape': list(X_test_final.shape) if X_test_final is not None else None,
            'feature_columns': feature_columns
        })
        
        save_json(preprocessing_info, output_paths['preprocessing_info'])
        
        logger.info("=== PREPROCESSING COMPLETATO CON SUCCESSO ===")
        
        # Prepara statistiche per il log
        train_size = X_train_final.shape[0]
        val_size = X_val_final.shape[0] if X_val_final is not None else 0
        test_size = X_test_final.shape[0] if X_test_final is not None else 0
        
        logger.info(f"Dataset finale: {train_size} train + {val_size} val + {test_size} test")
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