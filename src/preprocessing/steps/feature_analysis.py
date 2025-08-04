"""
Step 3: Analisi delle feature (correlazioni Cramér's V, etc.).
"""

import pandas as pd
from typing import Dict, Any, Tuple
from ...utils import get_logger
from ..filtering import analyze_cramers_correlations

logger = get_logger(__name__)


def execute_feature_analysis_step(
    df: pd.DataFrame, 
    target_column: str, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Esegue l'analisi delle feature per identificare correlazioni e problemi.
    
    Args:
        df: DataFrame da analizzare
        target_column: Nome della colonna target
        config: Configurazione per l'analisi
        
    Returns:
        Tuple con DataFrame (invariato) e informazioni di analisi
        
    Note:
        - Questo step non modifica il DataFrame, solo raccoglie informazioni
        - Identifica correlazioni problematiche tra feature categoriche
        - Prepara informazioni per step successivi di filtering
    """
    logger.info("🔍 Step 3: Analisi feature e correlazioni")
    
    steps_config = config.get('steps', {})
    analysis_info = {
        'step_name': 'feature_analysis',
        'config_used': config,
        'analyses_performed': [],
        'dataframe_shape': list(df.shape)
    }
    
    # SUBSTEP 3.1: Analisi correlazioni Cramér's V
    if steps_config.get('enable_cramers_analysis', True):
        logger.info("📊 Substep 3.1: Analisi correlazioni Cramér's V")
        
        cramer_threshold = config.get('cramer_threshold', 0.95)
        cramers_analysis = analyze_cramers_correlations(df, target_column, cramer_threshold)
        
        analysis_info['cramers_analysis'] = cramers_analysis
        analysis_info['analyses_performed'].append('cramers_correlations')
        
        # Log risultati
        high_corr_count = len(cramers_analysis.get('high_correlations', []))
        cat_cols_count = len(cramers_analysis.get('categorical_columns_analyzed', []))
        
        logger.info(f"   ✅ Analisi Cramér's V completata: {cat_cols_count} colonne categoriche analizzate")
        
        if high_corr_count > 0:
            logger.warning(f"   ⚠️  Trovate {high_corr_count} correlazioni elevate (>{cramer_threshold:.1%})")
            
            # Log dettagli correlazioni problematiche
            for corr in cramers_analysis.get('high_correlations', []):
                col1, col2 = corr.get('column1'), corr.get('column2')
                cramers_v = corr.get('cramers_v', 0)
                logger.warning(f"      - {col1} ↔ {col2}: {cramers_v:.3f}")
        else:
            logger.info(f"   ✅ Nessuna correlazione elevata trovata (soglia: {cramer_threshold:.1%})")
            
    else:
        logger.info("   ⏭️  Substep 3.1: Analisi Cramér's V DISABILITATA")
        analysis_info['cramers_analysis'] = {'skipped': True}
    
    # SUBSTEP 3.2: Analisi distribuzione tipi di dati
    logger.info("📈 Substep 3.2: Analisi distribuzione tipi di dati")
    
    dtype_analysis = analyze_data_types_distribution(df, target_column)
    analysis_info['dtype_analysis'] = dtype_analysis
    analysis_info['analyses_performed'].append('data_types_distribution')
    
    logger.info(f"   ✅ Analisi tipi completata: {dtype_analysis['summary']}")
    
    # SUBSTEP 3.3: Analisi cardinalità colonne categoriche
    logger.info("🔢 Substep 3.3: Analisi cardinalità feature categoriche")
    
    cardinality_analysis = analyze_categorical_cardinality(df, target_column, config)
    analysis_info['cardinality_analysis'] = cardinality_analysis
    analysis_info['analyses_performed'].append('categorical_cardinality')
    
    low_card_count = len(cardinality_analysis['low_cardinality'])
    high_card_count = len(cardinality_analysis['high_cardinality']) 
    very_high_card_count = len(cardinality_analysis['very_high_cardinality'])
    
    logger.info(f"   ✅ Analisi cardinalità: {low_card_count} bassa, {high_card_count} alta, {very_high_card_count} molto alta")
    
    # Log riassuntivo
    logger.info(f"🏁 Step 3 completato: {len(analysis_info['analyses_performed'])} analisi eseguite")
    logger.info(f"✨ Analisi eseguite: {analysis_info['analyses_performed']}")
    
    return df, analysis_info


def analyze_data_types_distribution(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Analizza la distribuzione dei tipi di dati nel DataFrame.
    
    Args:
        df: DataFrame da analizzare
        target_column: Nome della colonna target
        
    Returns:
        Dictionary con analisi dei tipi di dati
    """
    logger.debug("Analisi distribuzione tipi di dati...")
    
    # Conta tipi di dati
    dtype_counts = df.dtypes.value_counts().to_dict()
    
    # Analizza colonne per categoria
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    object_cols = df.select_dtypes(include='object').columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    
    # Rimuovi target dalle liste per analisi pulita
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in object_cols:
        object_cols.remove(target_column)
    if target_column in datetime_cols:
        datetime_cols.remove(target_column)
    if target_column in bool_cols:
        bool_cols.remove(target_column)
    
    # Statistiche per tipo
    analysis = {
        'dtype_counts': {str(k): int(v) for k, v in dtype_counts.items()},
        'columns_by_type': {
            'numeric': numeric_cols,
            'object': object_cols,
            'datetime': datetime_cols,
            'boolean': bool_cols
        },
        'target_column_dtype': str(df[target_column].dtype),
        'summary': f"{len(numeric_cols)} numeriche, {len(object_cols)} object, {len(datetime_cols)} datetime, {len(bool_cols)} boolean"
    }
    
    # Analisi valori mancanti per tipo
    missing_by_type = {}
    for dtype_name, cols in analysis['columns_by_type'].items():
        if cols:
            missing_counts = df[cols].isnull().sum()
            missing_by_type[dtype_name] = {
                'total_missing': int(missing_counts.sum()),
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
                'percentage_missing': float(missing_counts.sum() / (len(df) * len(cols)) * 100) if len(cols) > 0 else 0
            }
    
    analysis['missing_values_by_type'] = missing_by_type
    
    return analysis


def analyze_categorical_cardinality(df: pd.DataFrame, target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizza la cardinalità delle colonne categoriche.
    
    Args:
        df: DataFrame da analizzare
        target_column: Nome della colonna target
        config: Configurazione con soglie di cardinalità
        
    Returns:
        Dictionary con analisi della cardinalità
    """
    logger.debug("Analisi cardinalità colonne categoriche...")
    
    # Soglie dal config
    low_threshold = config.get('low_cardinality_threshold', 10)
    high_threshold = config.get('high_cardinality_max', 100)
    
    # Identifica colonne categoriche (escludi target)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_column in cat_cols:
        cat_cols.remove(target_column)
    
    # Analizza cardinalità
    cardinality_info = {}
    low_cardinality = []
    high_cardinality = []
    very_high_cardinality = []
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        cardinality_ratio = unique_count / total_count if total_count > 0 else 0
        
        col_info = {
            'unique_values': unique_count,
            'total_values': total_count,
            'cardinality_ratio': cardinality_ratio,
            'most_frequent_value': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
        }
        
        cardinality_info[col] = col_info
        
        # Categorizza per cardinalità
        if unique_count < low_threshold:
            low_cardinality.append(col)
        elif unique_count < high_threshold:
            high_cardinality.append(col)
        else:
            very_high_cardinality.append(col)
    
    analysis = {
        'thresholds_used': {
            'low_cardinality_threshold': low_threshold,
            'high_cardinality_max': high_threshold
        },
        'low_cardinality': low_cardinality,
        'high_cardinality': high_cardinality,
        'very_high_cardinality': very_high_cardinality,
        'detailed_cardinality_info': cardinality_info,
        'summary_stats': {
            'total_categorical_columns': len(cat_cols),
            'avg_cardinality': sum(info['unique_values'] for info in cardinality_info.values()) / len(cardinality_info) if cardinality_info else 0,
            'max_cardinality': max(info['unique_values'] for info in cardinality_info.values()) if cardinality_info else 0,
            'min_cardinality': min(info['unique_values'] for info in cardinality_info.values()) if cardinality_info else 0
        }
    }
    
    # Log dettagli per colonne molto alte cardinalità
    if very_high_cardinality:
        logger.warning(f"   ⚠️  Colonne con cardinalità molto alta (>{high_threshold}):")
        for col in very_high_cardinality:
            unique_count = cardinality_info[col]['unique_values']
            ratio = cardinality_info[col]['cardinality_ratio']
            logger.warning(f"      - {col}: {unique_count} valori unici ({ratio:.1%} cardinalità)")
    
    return analysis