"""
Modulo di validazione per controlli di qualità sui dati e prevenzione data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .logger import get_logger

logger = get_logger(__name__)



def check_target_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_column: str,
    max_correlation: float = 0.95
) -> Dict[str, Any]:
    """
    Controlla possibili target leakage nelle features.
    
    Args:
        X_train: Features di training
        y_train: Target di training
        target_column: Nome originale del target
        max_correlation: Soglia massima di correlazione ammessa
        
    Returns:
        Dictionary con risultati del controllo
    """
    logger.info("Controllo target leakage...")
    
    results = {
        'leakage_detected': False,
        'suspicious_features': [],
        'correlations': {}
    }
    
    try:
        # Controlla correlazioni alte con il target
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if target_column.lower() in col.lower():
                # Feature sospetta: contiene il nome del target
                results['suspicious_features'].append({
                    'feature': col,
                    'reason': 'Nome contiene target',
                    'correlation': np.corrcoef(X_train[col].fillna(0), y_train)[0, 1]
                })
                results['leakage_detected'] = True
                continue
            
            # Calcola correlazione
            correlation = np.corrcoef(X_train[col].fillna(0), y_train)[0, 1]
            results['correlations'][col] = correlation
            
            if abs(correlation) > max_correlation:
                results['suspicious_features'].append({
                    'feature': col,
                    'reason': f'Correlazione alta: {correlation:.3f}',
                    'correlation': correlation
                })
                results['leakage_detected'] = True
        
        if results['leakage_detected']:
            logger.warning(f"Possibile target leakage rilevato in {len(results['suspicious_features'])} feature")
            for feature_info in results['suspicious_features']:
                logger.warning(f"  - {feature_info['feature']}: {feature_info['reason']}")
        else:
            logger.info("Nessun target leakage rilevato")
            
    except Exception as e:
        logger.error(f"Errore nel controllo target leakage: {e}")
        results['error'] = str(e)
    
    return results

def validate_category_distribution(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: List[str] = None,
    max_drift: float = 0.05
) -> Dict[str, Any]:
    """
    Valida che le distribuzioni delle categorie siano simili tra train/val/test.
    
    Args:
        X_train: Features di training
        X_val: Features di validation
        X_test: Features di test
        categorical_columns: Lista delle colonne categoriche da controllare
        max_drift: Massima differenza percentuale ammessa
        
    Returns:
        Dictionary con risultati della validazione
    """
    logger.info("Validazione distribuzione categorie...")
    
    results = {
        'valid': True,
        'drift_detected': [],
        'distributions': {}
    }
    
    try:
        if categorical_columns is None:
            categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col not in X_train.columns:
                continue
                
            # Calcola distribuzioni
            train_dist = X_train[col].value_counts(normalize=True)
            val_dist = X_val[col].value_counts(normalize=True)
            test_dist = X_test[col].value_counts(normalize=True)
            
            # Allinea gli indici (categorie)
            all_categories = set(train_dist.index) | set(val_dist.index) | set(test_dist.index)
            
            train_dist = train_dist.reindex(all_categories, fill_value=0)
            val_dist = val_dist.reindex(all_categories, fill_value=0)
            test_dist = test_dist.reindex(all_categories, fill_value=0)
            
            # Calcola drift massimo
            train_val_drift = abs(train_dist - val_dist).max()
            train_test_drift = abs(train_dist - test_dist).max()
            val_test_drift = abs(val_dist - test_dist).max()
            
            max_observed_drift = max(train_val_drift, train_test_drift, val_test_drift)
            
            results['distributions'][col] = {
                'train_val_drift': train_val_drift,
                'train_test_drift': train_test_drift,
                'val_test_drift': val_test_drift,
                'max_drift': max_observed_drift
            }
            
            if max_observed_drift > max_drift:
                results['drift_detected'].append({
                    'column': col,
                    'max_drift': max_observed_drift,
                    'threshold': max_drift
                })
                results['valid'] = False
        
        if results['drift_detected']:
            logger.warning(f"Drift rilevato in {len(results['drift_detected'])} colonne categoriche")
            for drift_info in results['drift_detected']:
                logger.warning(f"  - {drift_info['column']}: drift {drift_info['max_drift']:.3f} > {drift_info['threshold']:.3f}")
        else:
            logger.info("Nessun drift significativo rilevato nelle categorie")
            
    except Exception as e:
        logger.error(f"Errore nella validazione distribuzione categorie: {e}")
        results['error'] = str(e)
        results['valid'] = False
    
    return results

def run_all_validations(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Esegue tutte le validazioni di qualità sui dati.
    
    Args:
        X_train: Features di training
        X_val: Features di validation
        X_test: Features di test
        y_train: Target di training
        config: Configurazione
        
    Returns:
        Dictionary con tutti i risultati delle validazioni
    """
    logger.info("=== AVVIO VALIDAZIONI QUALITÀ DATI ===")
    
    validation_results = {}
    
    # Configurazione validazioni
    quality_config = config.get('quality_checks', {})
    
    # 1. Validazione split temporale
    if quality_config.get('validate_temporal_split', True):
        # Usa la validazione per anno/mese
        preprocessing_config = config.get('preprocessing', {})
        year_col = preprocessing_config.get('year_column', 'A_AnnoStipula')
        month_col = preprocessing_config.get('month_column', 'A_MeseStipula')
        
        from .temporal_utils import validate_temporal_split_year_month
        validation_results['temporal_split'] = validate_temporal_split_year_month(
            X_train, X_val, X_test,
            year_column=year_col,
            month_column=month_col,
            min_gap_months=quality_config.get('min_temporal_gap_months', 1)
        )
    
    # 2. Controllo target leakage
    if quality_config.get('check_target_leakage', True):
        validation_results['target_leakage'] = check_target_leakage(
            X_train, y_train,
            target_column=config.get('target', {}).get('column', 'AI_Prezzo_Ridistribuito')
        )
    
    # 3. Validazione distribuzione categorie
    if quality_config.get('check_category_distribution', True):
        validation_results['category_distribution'] = validate_category_distribution(
            X_train, X_val, X_test,
            max_drift=quality_config.get('max_category_drift', 0.05)
        )
    
    # Riassunto validazioni
    all_valid = all(
        result.get('valid', True) for result in validation_results.values()
        if isinstance(result, dict) and 'valid' in result
    )
    
    leakage_detected = any(
        result.get('leakage_detected', False) for result in validation_results.values()
        if isinstance(result, dict) and 'leakage_detected' in result
    )
    
    logger.info(f"=== RISULTATO VALIDAZIONI ===")
    logger.info(f"Tutte le validazioni passate: {all_valid}")
    logger.info(f"Data leakage rilevato: {leakage_detected}")
    
    if not all_valid or leakage_detected:
        logger.warning("ATTENZIONE: Alcuni controlli di qualità hanno fallito!")
    else:
        logger.info("Tutti i controlli di qualità sono passati ✓")
    
    validation_results['summary'] = {
        'all_valid': all_valid,
        'leakage_detected': leakage_detected
    }
    
    return validation_results