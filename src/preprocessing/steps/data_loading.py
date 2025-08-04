"""
Step 1: Caricamento e validazione iniziale del dataset.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from ...utils.logger import get_logger
from ...utils.io import load_dataframe

logger = get_logger(__name__)


def load_and_validate_dataset(dataset_path: str, target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carica e valida il dataset iniziale.
    
    Args:
        dataset_path: Path al dataset da caricare
        target_column: Nome della colonna target per validazione
        
    Returns:
        Tuple con DataFrame caricato e informazioni di validazione
        
    Raises:
        FileNotFoundError: Se il dataset non esiste
        ValueError: Se il dataset non è valido
        
    Note:
        - Esegue controlli di integrità base sul dataset
        - Valida la presenza della colonna target
        - Fornisce statistiche descrittive iniziali
    """
    logger.info(f"📂 Step 1: Caricamento dataset da {dataset_path}")
    
    try:
        # Carica dataset
        df = load_dataframe(dataset_path)
        original_shape = df.shape
        
        # Validazioni iniziali
        validation_info = {
            'dataset_path': dataset_path,
            'original_shape': list(original_shape),
            'target_column': target_column,
            'validations': {}
        }
        
        # 1. Controllo dataset non vuoto
        if df.empty:
            raise ValueError("Dataset vuoto")
        validation_info['validations']['non_empty'] = True
        
        # 2. Controllo presenza target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' non trovata. Colonne disponibili: {list(df.columns)}")
        validation_info['validations']['target_exists'] = True
        
        # 3. Controllo target non completamente nullo
        target_null_count = df[target_column].isnull().sum()
        target_null_percentage = target_null_count / len(df) * 100
        
        if target_null_count == len(df):
            raise ValueError(f"Target column '{target_column}' è completamente nulla")
        
        validation_info['validations']['target_has_values'] = True
        validation_info['target_null_count'] = int(target_null_count)
        validation_info['target_null_percentage'] = float(target_null_percentage)
        
        # 4. Statistiche descrittive
        stats = {
            'total_rows': int(original_shape[0]),
            'total_columns': int(original_shape[1]),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'dtypes_count': df.dtypes.value_counts().to_dict(),
            'missing_values_total': int(df.isnull().sum().sum())
        }
        
        # Statistiche target
        target_stats = {
            'dtype': str(df[target_column].dtype),
            'non_null_count': int(df[target_column].notna().sum()),
            'unique_values': int(df[target_column].nunique())
        }
        
        # Se target è numerico, aggiungi statistiche numeriche
        if pd.api.types.is_numeric_dtype(df[target_column]):
            target_desc = df[target_column].describe()
            target_stats.update({
                'min': float(target_desc['min']),
                'max': float(target_desc['max']), 
                'mean': float(target_desc['mean']),
                'std': float(target_desc['std']),
                'median': float(target_desc['50%'])
            })
        
        validation_info.update({
            'dataset_stats': stats,
            'target_stats': target_stats
        })
        
        # Log informazioni
        logger.info(f"✅ Dataset caricato con successo: {original_shape}")
        logger.info(f"🎯 Target '{target_column}': {target_stats['non_null_count']} valori validi ({100 - target_null_percentage:.1f}%)")
        logger.info(f"💾 Memoria utilizzata: {stats['memory_usage_mb']:.2f} MB")
        logger.info(f"📊 Valori mancanti totali: {stats['missing_values_total']}")
        
        # Warning per alta percentuale di valori mancanti nel target
        if target_null_percentage > 10:
            logger.warning(f"⚠️  Target ha {target_null_percentage:.1f}% di valori mancanti")
        
        # Warning per dataset molto grandi
        if stats['memory_usage_mb'] > 1000:  # > 1GB
            logger.warning(f"⚠️  Dataset molto grande ({stats['memory_usage_mb']:.1f} MB)")
        
        return df, validation_info
        
    except Exception as e:
        logger.error(f"❌ Errore nel caricamento dataset: {str(e)}")
        raise


def validate_dataset_integrity(df: pd.DataFrame, validation_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue controlli di integrità aggiuntivi sul dataset.
    
    Args:
        df: DataFrame da validare
        validation_info: Informazioni di validazione esistenti
        
    Returns:
        Dictionary con risultati dei controlli di integrità aggiornati
        
    Note:
        - Controlli aggiuntivi per identificare problemi potenziali
        - Analisi della distribuzione dei tipi di dati
        - Identificazione di colonne problematiche
    """
    logger.info("🔍 Controlli di integrità aggiuntivi...")
    
    integrity_checks = {}
    
    # 1. Controllo colonne completamente duplicate
    duplicate_cols = []
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            if df[col1].equals(df[col2]):
                duplicate_cols.append((col1, col2))
    
    integrity_checks['duplicate_columns'] = duplicate_cols
    if duplicate_cols:
        logger.warning(f"⚠️  Trovate {len(duplicate_cols)} coppie di colonne duplicate")
        for col1, col2 in duplicate_cols:
            logger.warning(f"   - '{col1}' == '{col2}'")
    
    # 2. Controllo colonne con nomi sospetti
    suspicious_names = []
    suspicious_patterns = ['unnamed', 'index', 'level', 'nan', 'null', 'empty']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern in col_lower for pattern in suspicious_patterns):
            suspicious_names.append(col)
    
    integrity_checks['suspicious_column_names'] = suspicious_names
    if suspicious_names:
        logger.warning(f"⚠️  Trovate {len(suspicious_names)} colonne con nomi sospetti: {suspicious_names}")
    
    # 3. Controllo righe completamente duplicate
    duplicate_rows = df.duplicated().sum()
    integrity_checks['duplicate_rows_count'] = int(duplicate_rows)
    if duplicate_rows > 0:
        duplicate_percentage = duplicate_rows / len(df) * 100
        logger.warning(f"⚠️  Trovate {duplicate_rows} righe duplicate ({duplicate_percentage:.1f}%)")
    
    # 4. Controllo per colonne miste (object con valori numerici e stringhe)
    mixed_columns = []
    for col in df.select_dtypes(include='object').columns:
        sample_values = df[col].dropna().head(100)
        numeric_count = 0
        string_count = 0
        
        for val in sample_values:
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                string_count += 1
        
        if numeric_count > 0 and string_count > 0:
            mixed_columns.append({
                'column': col,
                'numeric_sample_count': numeric_count,
                'string_sample_count': string_count
            })
    
    integrity_checks['mixed_type_columns'] = mixed_columns
    if mixed_columns:
        logger.warning(f"⚠️  Trovate {len(mixed_columns)} colonne con tipi misti:")
        for col_info in mixed_columns:
            logger.warning(f"   - {col_info['column']}: {col_info['numeric_sample_count']} numerici, {col_info['string_sample_count']} stringhe")
    
    # Aggiorna validation_info
    validation_info['integrity_checks'] = integrity_checks
    
    logger.info("✅ Controlli di integrità completati")
    return validation_info