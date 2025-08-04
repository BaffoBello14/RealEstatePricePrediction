"""
Modulo core per le operazioni di pulizia e conversione dati.
Centralizza le funzioni comuni per eliminare duplicazioni nel codice.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def convert_to_numeric_unified(
    df: pd.DataFrame, 
    target_column: str, 
    threshold: float = 0.8,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Conversione unificata di colonne a tipo numerico.
    Sostituisce le funzioni duplicate in cleaning.py e encoding.py.
    
    Args:
        df: DataFrame da processare
        target_column: Nome della colonna target (da non toccare)
        threshold: Soglia minima di conversioni valide per applicare la conversione (0.0-1.0)
        exclude_columns: Lista di colonne da escludere dalla conversione
        
    Returns:
        Tuple con DataFrame convertito e informazioni dettagliate sulle conversioni
        
    Note:
        - Solo colonne object vengono considerate per la conversione
        - La conversione è applicata solo se il tasso di successo >= threshold
        - Mantiene traccia dettagliata di tutte le operazioni per debugging
    """
    logger.info(f"🔧 Conversione unificata a numerico (soglia: {threshold:.1%})...")
    
    # Inizializza tracking delle conversioni
    conversion_info = {
        'threshold_used': threshold,
        'target_column_excluded': target_column,
        'exclude_columns': exclude_columns or [],
        'converted_columns': [],
        'failed_conversions': [],
        'skipped_columns': [],
        'conversion_stats': {},
        'total_converted': 0
    }
    
    # Identifica colonne candidte per conversione
    object_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Rimuovi colonne da escludere
    excluded = {target_column} | set(exclude_columns or [])
    candidate_cols = [col for col in object_cols if col not in excluded]
    
    if not candidate_cols:
        logger.info("📊 Nessuna colonna object candidata per conversione numerica")
        return df, conversion_info
    
    logger.info(f"📊 Analizzando {len(candidate_cols)} colonne candidate: {candidate_cols}")
    
    df_copy = df.copy()  # Lavora su una copia per sicurezza
    
    for col in candidate_cols:
        try:
            # Statistiche pre-conversione
            original_count = len(df_copy[col])
            non_null_count = df_copy[col].notna().sum()
            
            if non_null_count == 0:
                conversion_info['skipped_columns'].append(col)
                logger.debug(f"  📋 {col}: saltata (interamente null)")
                continue
            
            # Tentativo di conversione
            original_dtype = str(df_copy[col].dtype)
            numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
            valid_conversions = numeric_series.notna().sum()
            conversion_rate = valid_conversions / non_null_count if non_null_count > 0 else 0.0
            
            # Statistiche dettagliate
            conversion_stats = {
                'original_dtype': original_dtype,
                'original_total': original_count,
                'original_non_null': non_null_count,
                'valid_conversions': int(valid_conversions),
                'conversion_rate': float(conversion_rate),
                'new_dtype': str(numeric_series.dtype)
            }
            
            conversion_info['conversion_stats'][col] = conversion_stats
            
            # Decide se applicare la conversione
            if conversion_rate >= threshold:
                df_copy[col] = numeric_series
                conversion_info['converted_columns'].append(col)
                conversion_info['total_converted'] += 1
                logger.info(f"  ✅ {col}: convertita ({conversion_rate:.1%} successo, {original_dtype} → {numeric_series.dtype})")
            else:
                conversion_info['failed_conversions'].append(col)
                logger.info(f"  ❌ {col}: conversione rifiutata ({conversion_rate:.1%} < {threshold:.1%})")
                
        except Exception as e:
            conversion_info['failed_conversions'].append(col)
            logger.warning(f"  ⚠️  {col}: errore nella conversione - {str(e)}")
            # Aggiungi info dell'errore
            conversion_info['conversion_stats'][col] = {
                'error': str(e),
                'original_dtype': str(df_copy[col].dtype)
            }
    
    # Log riassuntivo
    total_converted = conversion_info['total_converted']
    total_failed = len(conversion_info['failed_conversions'])
    total_skipped = len(conversion_info['skipped_columns'])
    
    logger.info(f"🏁 Conversione completata: {total_converted} convertite, {total_failed} fallite, {total_skipped} saltate")
    
    if total_converted > 0:
        logger.info(f"✨ Colonne convertite con successo: {conversion_info['converted_columns']}")
    
    return df_copy, conversion_info


def clean_dataframe_unified(
    df: pd.DataFrame, 
    target_column: Optional[str] = None,
    remove_empty_strings: bool = True,
    remove_duplicates: bool = True,
    remove_empty_columns: bool = True,
    remove_target_nulls: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pulizia unificata del DataFrame.
    Sostituisce le funzioni duplicate di pulizia base in vari moduli.
    
    Args:
        df: DataFrame da pulire
        target_column: Nome della colonna target (None se non presente)
        remove_empty_strings: Se True, sostituisce stringhe vuote con NaN
        remove_duplicates: Se True, rimuove righe duplicate
        remove_empty_columns: Se True, rimuove colonne completamente vuote
        remove_target_nulls: Se True, rimuove righe con target mancante
        
    Returns:
        Tuple con DataFrame pulito e informazioni dettagliate sulla pulizia
        
    Note:
        - Mantiene traccia di tutte le operazioni di pulizia
        - Operazioni applicate nell'ordine ottimale per massimizzare efficacia
        - Log dettagliato per ogni step di pulizia
    """
    logger.info("🧹 Avvio pulizia unificata DataFrame...")
    
    original_shape = df.shape
    cleaning_info = {
        'original_shape': list(original_shape),
        'operations_applied': [],
        'target_column': target_column
    }
    
    df_clean = df.copy()
    
    # 1. Sostituzione stringhe vuote con NaN
    if remove_empty_strings:
        logger.info("🔄 Sostituzione stringhe vuote con NaN...")
        empty_count_before = (df_clean == '').sum().sum()
        df_clean = df_clean.replace('', np.nan)
        empty_count_after = (df_clean == '').sum().sum()
        
        operation_info = {
            'operation': 'replace_empty_strings',
            'empty_strings_replaced': int(empty_count_before - empty_count_after)
        }
        cleaning_info['operations_applied'].append(operation_info)
        logger.info(f"  📊 Sostituite {operation_info['empty_strings_replaced']} stringhe vuote")
    
    # 2. Rimozione righe con target mancante (deve essere fatto presto)
    if remove_target_nulls and target_column and target_column in df_clean.columns:
        target_null_count = df_clean[target_column].isnull().sum()
        if target_null_count > 0:
            logger.info(f"🎯 Rimozione {target_null_count} righe con target '{target_column}' mancante...")
            df_clean = df_clean.dropna(subset=[target_column])
            
            operation_info = {
                'operation': 'remove_target_nulls',
                'target_column': target_column,
                'rows_removed': int(target_null_count)
            }
            cleaning_info['operations_applied'].append(operation_info)
        else:
            logger.info(f"✅ Target '{target_column}': nessun valore mancante")
    
    # 3. Rimozione colonne completamente vuote
    if remove_empty_columns:
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            logger.info(f"🗑️  Rimozione {len(empty_cols)} colonne completamente vuote...")
            df_clean = df_clean.drop(columns=empty_cols)
            
            operation_info = {
                'operation': 'remove_empty_columns',
                'columns_removed': empty_cols,
                'count': len(empty_cols)
            }
            cleaning_info['operations_applied'].append(operation_info)
            logger.info(f"  📊 Colonne rimosse: {empty_cols}")
        else:
            logger.info("✅ Nessuna colonna completamente vuota trovata")
    
    # 4. Rimozione duplicati completi
    if remove_duplicates:
        duplicates_count = df_clean.duplicated().sum()
        if duplicates_count > 0:
            logger.info(f"🔄 Rimozione {duplicates_count} righe duplicate...")
            df_clean = df_clean.drop_duplicates()
            
            operation_info = {
                'operation': 'remove_duplicates',
                'duplicates_removed': int(duplicates_count)
            }
            cleaning_info['operations_applied'].append(operation_info)
        else:
            logger.info("✅ Nessuna riga duplicata trovata")
    
    # Statistiche finali
    final_shape = df_clean.shape
    cleaning_info.update({
        'final_shape': list(final_shape),
        'rows_removed': original_shape[0] - final_shape[0],
        'columns_removed': original_shape[1] - final_shape[1],
        'data_reduction_percentage': {
            'rows': (original_shape[0] - final_shape[0]) / original_shape[0] * 100 if original_shape[0] > 0 else 0,
            'columns': (original_shape[1] - final_shape[1]) / original_shape[1] * 100 if original_shape[1] > 0 else 0
        }
    })
    
    logger.info(f"🏁 Pulizia completata: {original_shape} → {final_shape}")
    logger.info(f"📊 Riduzione dati: {cleaning_info['rows_removed']} righe (-{cleaning_info['data_reduction_percentage']['rows']:.1f}%), "
               f"{cleaning_info['columns_removed']} colonne (-{cleaning_info['data_reduction_percentage']['columns']:.1f}%)")
    
    return df_clean, cleaning_info


def remove_constant_columns_unified(
    df: pd.DataFrame, 
    target_column: Optional[str] = None,
    threshold: float = 0.95,
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rimozione unificata di colonne quasi-costanti.
    
    Args:
        df: DataFrame da processare
        target_column: Nome della colonna target (da non rimuovere)
        threshold: Soglia per considerare una colonna costante (% valori uguali)
        exclude_columns: Lista di colonne da escludere dalla rimozione
        
    Returns:
        Tuple con DataFrame filtrato e informazioni dettagliate
        
    Note:
        - Considera una colonna costante se un valore appare >= threshold delle volte
        - Esclude automaticamente target_column e exclude_columns
        - Fornisce analisi dettagliata per ogni colonna rimossa
    """
    logger.info(f"📊 Rimozione colonne quasi-costanti (soglia: {threshold:.1%})...")
    
    # Setup tracking
    removal_info = {
        'threshold_used': threshold,
        'target_column_excluded': target_column,
        'exclude_columns': exclude_columns or [],
        'constant_columns': [],
        'columns_removed_count': 0,
        'column_analysis': {}
    }
    
    # Identifica colonne da analizzare
    excluded = set(exclude_columns or [])
    if target_column:
        excluded.add(target_column)
    
    analyzable_cols = [col for col in df.columns if col not in excluded]
    
    if not analyzable_cols:
        logger.info("📋 Nessuna colonna da analizzare per costanza")
        return df, removal_info
    
    logger.info(f"🔍 Analizzando {len(analyzable_cols)} colonne per costanza...")
    
    constant_columns = []
    
    for col in analyzable_cols:
        try:
            # Analisi distribuzione valori
            value_counts = df[col].value_counts(normalize=True, dropna=False)
            
            if len(value_counts) == 0:
                # Colonna completamente vuota (questo caso dovrebbe essere gestito da clean_dataframe_unified)
                continue
            
            max_frequency = value_counts.iloc[0]
            most_common_value = value_counts.index[0]
            unique_values = len(value_counts)
            
            # Statistiche dettagliate
            analysis = {
                'max_frequency': float(max_frequency),
                'unique_values': int(unique_values),
                'most_common_value': str(most_common_value),
                'is_constant': max_frequency >= threshold,
                'total_non_null': int(df[col].notna().sum())
            }
            
            removal_info['column_analysis'][col] = analysis
            
            # Decide se rimuovere
            if max_frequency >= threshold:
                constant_columns.append(col)
                logger.info(f"  🚫 {col}: COSTANTE ({max_frequency:.1%} = '{most_common_value}', {unique_values} valori unici)")
            else:
                logger.debug(f"  ✅ {col}: variabile ({max_frequency:.1%} max freq, {unique_values} valori unici)")
                
        except Exception as e:
            logger.warning(f"  ⚠️  {col}: errore nell'analisi - {str(e)}")
            removal_info['column_analysis'][col] = {'error': str(e)}
    
    # Applica rimozione
    removal_info['constant_columns'] = constant_columns
    removal_info['columns_removed_count'] = len(constant_columns)
    
    if constant_columns:
        df_filtered = df.drop(columns=constant_columns)
        logger.info(f"🗑️  Rimosse {len(constant_columns)} colonne quasi-costanti")
        
        # Log dettagliato per debugging
        for col in constant_columns:
            stats = removal_info['column_analysis'][col]
            logger.info(f"    - {col}: {stats['max_frequency']:.1%} frequenza, {stats['unique_values']} valori, dominante='{stats['most_common_value']}'")
    else:
        df_filtered = df
        logger.info("✅ Nessuna colonna quasi-costante trovata")
    
    return df_filtered, removal_info