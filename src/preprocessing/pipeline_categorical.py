"""
Pipeline di preprocessing specificamente progettata per modelli che accettano dati categorici.

Modelli supportati:
- CatBoost
- TabNet (TabM)
- LightGBM con supporto categorico
- XGBoost con supporto categorico limitato

Questa pipeline mantiene le variabili categoriche nel loro formato originale invece di
convertirle tramite encoding, permettendo ai modelli di sfruttare la loro gestione
nativa delle features categoriche.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_dataframe, save_json
from ..utils.error_handling import safe_execution, PreprocessingError
from .data_cleaning_core import clean_dataframe_unified, remove_constant_columns_unified
from .filtering import analyze_cramers_correlations
from .imputation import handle_missing_values

logger = get_logger(__name__)

class CategoricalAwarePreprocessor:
    """
    Preprocessore che mantiene le variabili categoriche per modelli che le supportano nativamente.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_info = {
            'pipeline_type': 'categorical_aware',
            'steps_executed': [],
            'categorical_columns': [],
            'numeric_columns': [],
            'target_column': None,
            'steps_info': {}
        }
        
    @safe_execution(reraise=True)
    def preprocess_for_categorical_models(
        self, 
        dataset_path: str, 
        target_column: str,
        output_dir: str = "data/processed_categorical"
    ) -> Dict[str, Any]:
        """
        Esegue preprocessing mantenendo le colonne categoriche per modelli che le supportano.
        
        Args:
            dataset_path: Path del dataset da processare
            target_column: Nome della colonna target
            output_dir: Directory di output
            
        Returns:
            Dictionary con informazioni sul preprocessing e path dei file salvati
        """
        logger.info("🚀 Avvio preprocessing per modelli categorici...")
        
        steps_config = self.config.get('steps', {})
        self.preprocessing_info['target_column'] = target_column
        
        # ===== STEP 1: CARICAMENTO DATI =====
        logger.info("Step 1: Caricamento dataset...")
        df = load_dataframe(dataset_path)
        logger.info(f"Dataset caricato: {df.shape}")
        self.preprocessing_info['dataset_original_shape'] = list(df.shape)
        
        # ===== STEP 2: PULIZIA BASE =====
        logger.info("Step 2: Pulizia base dati...")
        df, cleaning_info = clean_dataframe_unified(
            df=df,
            target_column=target_column,
            remove_empty_strings=True,
            remove_duplicates=True,
            remove_empty_columns=True,
            remove_target_nulls=True
        )
        self.preprocessing_info['steps_info']['cleaning'] = cleaning_info
        self.preprocessing_info['steps_executed'].append('cleaning')
        
        # ===== STEP 3: IDENTIFICAZIONE TIPI =====
        logger.info("Step 3: Identificazione tipi di colonne...")
        categorical_cols, numeric_cols = self._identify_column_types(df, target_column)
        self.preprocessing_info['categorical_columns'] = categorical_cols
        self.preprocessing_info['numeric_columns'] = numeric_cols
        
        logger.info(f"📊 Colonne categoriche identificate ({len(categorical_cols)}): {categorical_cols}")
        logger.info(f"📊 Colonne numeriche identificate ({len(numeric_cols)}): {numeric_cols}")
        
        # ===== STEP 4: ANALISI CRAMÉR (se abilitato) =====
        if steps_config.get('enable_cramers_analysis', True):
            logger.info("Step 4: Analisi correlazioni Cramér's V...")
            cramer_threshold = self.config.get('cramer_threshold', 0.95)
            cramers_analysis = analyze_cramers_correlations(df, target_column, cramer_threshold)
            self.preprocessing_info['steps_info']['cramers_analysis'] = cramers_analysis
            self.preprocessing_info['steps_executed'].append('cramers_analysis')
        else:
            logger.info("Step 4: Analisi Cramér's V DISABILITATA")
        
        # ===== STEP 5: GESTIONE VALORI MANCANTI =====
        logger.info("Step 5: Gestione valori mancanti...")
        df, imputation_info = handle_missing_values(df, target_column)
        self.preprocessing_info['steps_info']['imputation'] = imputation_info
        self.preprocessing_info['steps_executed'].append('imputation')
        
        # ===== STEP 6: RIMOZIONE COLONNE COSTANTI =====
        logger.info("Step 6: Rimozione colonne costanti...")
        constant_threshold = self.config.get('constant_column_threshold', 0.95)
        df, constant_removal_info = remove_constant_columns_unified(
            df, target_column, threshold=constant_threshold
        )
        self.preprocessing_info['steps_info']['constant_removal'] = constant_removal_info
        self.preprocessing_info['steps_executed'].append('constant_removal')
        
        # ===== STEP 7: PREPARAZIONE DATI CATEGORICI =====
        logger.info("Step 7: Preparazione dati categorici...")
        df_categorical = self._prepare_categorical_data(df, categorical_cols, numeric_cols)
        
        # ===== STEP 8: SALVATAGGIO =====
        logger.info("Step 8: Salvataggio dati preprocessati...")
        save_paths = self._save_processed_data(df_categorical, output_dir, target_column)
        self.preprocessing_info['output_paths'] = save_paths
        self.preprocessing_info['final_shape'] = list(df_categorical.shape)
        
        # Salva info preprocessing
        save_json(self.preprocessing_info, f"{output_dir}/categorical_preprocessing_info.json")
        
        logger.info("✅ Preprocessing per modelli categorici completato!")
        logger.info(f"📊 Shape finale: {df_categorical.shape}")
        logger.info(f"📂 Files salvati in: {output_dir}")
        
        return self.preprocessing_info
    
    def _identify_column_types(self, df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
        """
        Identifica colonne categoriche e numeriche.
        """
        feature_columns = [col for col in df.columns if col != target_column]
        
        categorical_cols = []
        numeric_cols = []
        
        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Categorica esplicita
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Potenzialmente numerica, ma controlliamo se è discreta con pochi valori
                unique_values = df[col].nunique()
                total_values = len(df)
                unique_ratio = unique_values / total_values
                
                # Se ha pochi valori unici relativi al totale, potrebbe essere categorica
                if unique_ratio < 0.05 and unique_values < 20:
                    logger.info(f"🔄 Colonna {col} convertita a categorica (ratio: {unique_ratio:.3f}, unique: {unique_values})")
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                # Datetime o altri tipi speciali - trattiamo come categorici per sicurezza
                categorical_cols.append(col)
        
        return categorical_cols, numeric_cols
    
    def _prepare_categorical_data(self, df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> pd.DataFrame:
        """
        Prepara i dati mantenendo le categoriche nel formato corretto per i modelli.
        """
        df_prepared = df.copy()
        
        # Converti esplicitamente le colonne categoriche al tipo category di pandas
        for col in categorical_cols:
            if col in df_prepared.columns:
                # Gestisci valori nulli nelle categoriche
                if df_prepared[col].isnull().any():
                    df_prepared[col] = df_prepared[col].fillna('MISSING')
                
                # Converti a stringa poi a category per assicurare consistenza
                df_prepared[col] = df_prepared[col].astype(str).astype('category')
                logger.info(f"📋 {col}: convertita a categorica ({df_prepared[col].nunique()} categorie)")
        
        # Assicurati che le numeriche siano del tipo corretto
        for col in numeric_cols:
            if col in df_prepared.columns:
                df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
        
        return df_prepared
    
    def _save_processed_data(self, df: pd.DataFrame, output_dir: str, target_column: str) -> Dict[str, str]:
        """
        Salva i dati preprocessati in diversi formati.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        save_paths = {}
        
        # Salva dataset completo
        full_path = f"{output_dir}/dataset_categorical.parquet"
        save_dataframe(df, full_path, format='parquet')
        save_paths['full_dataset'] = full_path
        
        # Salva anche versione CSV per compatibilità
        csv_path = f"{output_dir}/dataset_categorical.csv"
        df.to_csv(csv_path, index=False)
        save_paths['csv_dataset'] = csv_path
        
        # Salva separatamente X e y
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        X_path = f"{output_dir}/X_categorical.parquet"
        y_path = f"{output_dir}/y_categorical.parquet"
        
        save_dataframe(X, X_path, format='parquet')
        save_dataframe(pd.DataFrame({target_column: y}), y_path, format='parquet')
        
        save_paths['features'] = X_path
        save_paths['target'] = y_path
        
        # Salva info sulle colonne categoriche
        column_info = {
            'categorical_columns': [col for col in feature_cols if df[col].dtype.name == 'category'],
            'numeric_columns': [col for col in feature_cols if df[col].dtype.name != 'category'],
            'target_column': target_column,
            'total_features': len(feature_cols)
        }
        
        info_path = f"{output_dir}/column_info.json"
        save_json(column_info, info_path)
        save_paths['column_info'] = info_path
        
        return save_paths

def run_categorical_preprocessing_pipeline(
    dataset_path: str,
    target_column: str,
    config_path: str = "config/config.yaml",
    output_dir: str = "data/processed_categorical"
) -> Dict[str, Any]:
    """
    Funzione di convenienza per eseguire la pipeline categorical-aware.
    
    Args:
        dataset_path: Path del dataset
        target_column: Nome colonna target
        config_path: Path configurazione
        output_dir: Directory output
        
    Returns:
        Informazioni sul preprocessing eseguito
    """
    from ..utils.io import load_config
    
    config = load_config(config_path)
    preprocessor = CategoricalAwarePreprocessor(config.get('preprocessing', {}))
    
    return preprocessor.preprocess_for_categorical_models(
        dataset_path, target_column, output_dir
    )