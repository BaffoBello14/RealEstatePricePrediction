"""
Step 2: Pulizia dati utilizzando il sistema unificato.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from ...utils import get_logger
from ..data_cleaning_core import (
    clean_dataframe_unified,
    remove_constant_columns_unified,
    convert_to_numeric_unified
)
from ..cleaning import remove_specific_columns

logger = get_logger(__name__)


def execute_data_cleaning_step(
    df: pd.DataFrame, 
    target_column: str, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Esegue il step di pulizia dati in modo modulare e controllato.
    
    Args:
        df: DataFrame da pulire
        target_column: Nome della colonna target
        config: Configurazione per la pulizia
        
    Returns:
        Tuple con DataFrame pulito e informazioni dettagliate
        
    Note:
        - Utilizza il sistema unificato per la pulizia
        - Esegue i step in ordine ottimale
        - Mantiene traccia dettagliata di ogni operazione
    """
    logger.info("🧹 Step 2: Pulizia dati modulare")
    
    steps_config = config.get('steps', {})
    cleaning_info = {
        'step_name': 'data_cleaning',
        'config_used': config,
        'substeps_executed': [],
        'original_shape': list(df.shape)
    }
    
    df_clean = df.copy()
    
    # SUBSTEP 2.1: Pulizia base unificata
    logger.info("🔧 Substep 2.1: Pulizia base (stringhe vuote, duplicati, target null)")
    if steps_config.get('enable_basic_cleaning', True):
        df_clean, base_cleaning_info = clean_dataframe_unified(
            df=df_clean,
            target_column=target_column,
            remove_empty_strings=True,
            remove_duplicates=True,
            remove_empty_columns=True,
            remove_target_nulls=True
        )
        cleaning_info['base_cleaning'] = base_cleaning_info
        cleaning_info['substeps_executed'].append('base_cleaning_unified')
        logger.info(f"   ✅ Base cleaning: {base_cleaning_info['original_shape']} → {base_cleaning_info['final_shape']}")
    else:
        logger.info("   ⏭️  Pulizia base DISABILITATA")
    
    # SUBSTEP 2.2: Rimozione colonne specifiche
    if steps_config.get('enable_specific_columns_removal', True):
        columns_to_remove = config.get('columns_to_remove', [])
        if columns_to_remove:
            logger.info(f"🗑️  Substep 2.2: Rimozione colonne specifiche ({len(columns_to_remove)} colonne)")
            shape_before = df_clean.shape
            df_clean, specific_removal_info = remove_specific_columns(df_clean, columns_to_remove)
            shape_after = df_clean.shape
            
            cleaning_info['specific_columns_removal'] = specific_removal_info
            cleaning_info['substeps_executed'].append('specific_columns_removal')
            logger.info(f"   ✅ Colonne specifiche rimosse: {shape_before} → {shape_after}")
            
            # Log dettagliato delle colonne rimosse
            removed_cols = specific_removal_info.get('existing_columns', [])
            if removed_cols:
                logger.info(f"   📋 Colonne rimosse: {removed_cols}")
        else:
            logger.info("   📋 Substep 2.2: Nessuna colonna specifica da rimuovere")
    else:
        logger.info("   ⏭️  Substep 2.2: Rimozione colonne specifiche DISABILITATA")
    
    # SUBSTEP 2.3: Rimozione colonne quasi-costanti
    if steps_config.get('enable_constant_columns_removal', True):
        constant_threshold = config.get('constant_column_threshold', 0.95)
        logger.info(f"📊 Substep 2.3: Rimozione colonne quasi-costanti (soglia: {constant_threshold:.1%})")
        
        shape_before = df_clean.shape
        df_clean, constant_removal_info = remove_constant_columns_unified(
            df=df_clean,
            target_column=target_column,
            threshold=constant_threshold
        )
        shape_after = df_clean.shape
        
        cleaning_info['constant_columns_removal'] = constant_removal_info
        cleaning_info['substeps_executed'].append('constant_columns_removal')
        
        removed_count = constant_removal_info.get('columns_removed_count', 0)
        logger.info(f"   ✅ Colonne costanti rimosse: {removed_count} colonne, {shape_before} → {shape_after}")
        
        if removed_count > 0:
            removed_cols = constant_removal_info.get('constant_columns', [])
            logger.info(f"   📋 Colonne costanti rimosse: {removed_cols}")
    else:
        logger.info("   ⏭️  Substep 2.3: Rimozione colonne costanti DISABILITATA")
    
    # SUBSTEP 2.4: Conversione automatica a numerico
    if steps_config.get('enable_auto_numeric_conversion', True):
        auto_numeric_threshold = config.get('auto_numeric_threshold', 0.8)
        logger.info(f"🔢 Substep 2.4: Conversione automatica a numerico (soglia: {auto_numeric_threshold:.1%})")
        
        df_clean, conversion_info = convert_to_numeric_unified(
            df=df_clean,
            target_column=target_column,
            threshold=auto_numeric_threshold
        )
        
        cleaning_info['numeric_conversion'] = conversion_info
        cleaning_info['substeps_executed'].append('numeric_conversion')
        
        converted_count = conversion_info.get('total_converted', 0)
        logger.info(f"   ✅ Conversioni numeriche: {converted_count} colonne convertite")
        
        if converted_count > 0:
            converted_cols = conversion_info.get('converted_columns', [])
            logger.info(f"   📋 Colonne convertite: {converted_cols}")
    else:
        logger.info("   ⏭️  Substep 2.4: Conversione automatica a numerico DISABILITATA")
    
    # Statistiche finali
    final_shape = df_clean.shape
    cleaning_info.update({
        'final_shape': list(final_shape),
        'total_rows_removed': cleaning_info['original_shape'][0] - final_shape[0],
        'total_columns_removed': cleaning_info['original_shape'][1] - final_shape[1],
        'data_reduction_percentage': {
            'rows': (cleaning_info['original_shape'][0] - final_shape[0]) / cleaning_info['original_shape'][0] * 100 if cleaning_info['original_shape'][0] > 0 else 0,
            'columns': (cleaning_info['original_shape'][1] - final_shape[1]) / cleaning_info['original_shape'][1] * 100 if cleaning_info['original_shape'][1] > 0 else 0
        }
    })
    
    # Log riassuntivo
    logger.info(f"🏁 Step 2 completato: {cleaning_info['original_shape']} → {final_shape}")
    logger.info(f"📊 Riduzione: {cleaning_info['total_rows_removed']} righe (-{cleaning_info['data_reduction_percentage']['rows']:.1f}%), "
               f"{cleaning_info['total_columns_removed']} colonne (-{cleaning_info['data_reduction_percentage']['columns']:.1f}%)")
    logger.info(f"✨ Substep eseguiti: {cleaning_info['substeps_executed']}")
    
    # Warning per riduzioni drastiche
    if cleaning_info['data_reduction_percentage']['rows'] > 50:
        logger.warning(f"⚠️  Riduzione drastica delle righe: {cleaning_info['data_reduction_percentage']['rows']:.1f}%")
    
    if cleaning_info['data_reduction_percentage']['columns'] > 30:
        logger.warning(f"⚠️  Riduzione significativa delle colonne: {cleaning_info['data_reduction_percentage']['columns']:.1f}%")
    
    return df_clean, cleaning_info


def validate_cleaning_results(
    df_original: pd.DataFrame, 
    df_cleaned: pd.DataFrame, 
    cleaning_info: Dict[str, Any],
    target_column: str
) -> Dict[str, Any]:
    """
    Valida i risultati della pulizia dati.
    
    Args:
        df_original: DataFrame originale (per confronto)
        df_cleaned: DataFrame dopo pulizia
        cleaning_info: Informazioni sulla pulizia
        target_column: Nome colonna target
        
    Returns:
        Dictionary con risultati della validazione
        
    Note:
        - Controlla che la pulizia non abbia introdotto problemi
        - Valida che il target sia ancora presente e valido
        - Fornisce metriche di qualità post-pulizia
    """
    logger.info("🔍 Validazione risultati pulizia...")
    
    validation_results = {
        'validation_passed': True,
        'issues_found': [],
        'quality_metrics': {}
    }
    
    try:
        # 1. Controllo target ancora presente
        if target_column not in df_cleaned.columns:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append(f"Target column '{target_column}' rimossa durante pulizia")
        
        # 2. Controllo target ancora ha valori
        target_valid_count = df_cleaned[target_column].notna().sum()
        if target_valid_count == 0:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append("Target column non ha più valori validi dopo pulizia")
        
        # 3. Controllo dimensioni minime
        if len(df_cleaned) < 10:  # Dataset troppo piccolo
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append(f"Dataset ridotto a soli {len(df_cleaned)} righe")
        
        if df_cleaned.shape[1] < 2:  # Solo target rimasto
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append(f"Rimaste solo {df_cleaned.shape[1]} colonne")
        
        # 4. Metriche di qualità
        quality_metrics = {
            'rows_retained_percentage': len(df_cleaned) / len(df_original) * 100 if len(df_original) > 0 else 0,
            'columns_retained_percentage': df_cleaned.shape[1] / df_original.shape[1] * 100 if df_original.shape[1] > 0 else 0,
            'target_completeness_percentage': target_valid_count / len(df_cleaned) * 100 if len(df_cleaned) > 0 else 0,
            'total_missing_values': int(df_cleaned.isnull().sum().sum()),
            'memory_reduction_percentage': (1 - (df_cleaned.memory_usage(deep=True).sum() / df_original.memory_usage(deep=True).sum())) * 100
        }
        
        validation_results['quality_metrics'] = quality_metrics
        
        # Log risultati
        if validation_results['validation_passed']:
            logger.info("✅ Validazione pulizia: SUCCESSO")
            logger.info(f"📊 Qualità: {quality_metrics['rows_retained_percentage']:.1f}% righe, "
                       f"{quality_metrics['columns_retained_percentage']:.1f}% colonne mantenute")
            logger.info(f"🎯 Target completezza: {quality_metrics['target_completeness_percentage']:.1f}%")
        else:
            logger.error("❌ Validazione pulizia: FALLITA")
            for issue in validation_results['issues_found']:
                logger.error(f"   - {issue}")
        
    except Exception as e:
        logger.error(f"❌ Errore durante validazione: {str(e)}")
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(f"Errore validazione: {str(e)}")
    
    return validation_results