"""
Modulo di validazione per controlli di qualità sui dati e prevenzione data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from .logger import get_logger

logger = get_logger(__name__)

def validate_temporal_split(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame,
    date_column: str = 'A_DataStipula',
    min_gap_days: int = 30
) -> Dict[str, Any]:
    """
    Valida la qualità dello split temporale per prevenire data leakage.
    
    Args:
        X_train: Features di training
        X_val: Features di validation
        X_test: Features di test
        date_column: Nome della colonna data
        min_gap_days: Gap minimo in giorni tra i set
        
    Returns:
        Dictionary con risultati della validazione
    """
    logger.info("Validazione split temporale...")
    
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    try:
        # Controlla se la colonna data esiste
        if date_column not in X_train.columns:
            results['errors'].append(f"Colonna data '{date_column}' non trovata")
            results['valid'] = False
            return results
        
        # Estrai date per ogni set
        train_dates = pd.to_datetime(X_train[date_column], errors='coerce')
        val_dates = pd.to_datetime(X_val[date_column], errors='coerce')
        test_dates = pd.to_datetime(X_test[date_column], errors='coerce')
        
        # Statistiche date
        train_range = (train_dates.min(), train_dates.max())
        val_range = (val_dates.min(), val_dates.max())
        test_range = (test_dates.min(), test_dates.max())
        
        results['stats'] = {
            'train_range': train_range,
            'val_range': val_range,
            'test_range': test_range,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        # Controlla ordine temporale
        if train_range[1] >= val_range[0]:
            results['errors'].append(
                f"Overlap temporale tra train e validation: "
                f"train fino a {train_range[1]}, val da {val_range[0]}"
            )
            results['valid'] = False
        
        if val_range[1] >= test_range[0]:
            results['errors'].append(
                f"Overlap temporale tra validation e test: "
                f"val fino a {val_range[1]}, test da {test_range[0]}"
            )
            results['valid'] = False
        
        # Controlla gap temporale minimo
        train_val_gap = (val_range[0] - train_range[1]).days
        val_test_gap = (test_range[0] - val_range[1]).days
        
        if train_val_gap < min_gap_days:
            results['warnings'].append(
                f"Gap train-val troppo piccolo: {train_val_gap} giorni < {min_gap_days}"
            )
        
        if val_test_gap < min_gap_days:
            results['warnings'].append(
                f"Gap val-test troppo piccolo: {val_test_gap} giorni < {min_gap_days}"
            )
        
        results['stats']['train_val_gap_days'] = train_val_gap
        results['stats']['val_test_gap_days'] = val_test_gap
        
        logger.info(f"Split temporale: Train {train_range}, Val {val_range}, Test {test_range}")
        logger.info(f"Gap temporali: train-val {train_val_gap} giorni, val-test {val_test_gap} giorni")
        
        if results['errors']:
            logger.error(f"Errori split temporale: {results['errors']}")
        if results['warnings']:
            logger.warning(f"Warning split temporale: {results['warnings']}")
        
    except Exception as e:
        results['errors'].append(f"Errore nella validazione: {str(e)}")
        results['valid'] = False
        logger.error(f"Errore validazione split temporale: {e}")
    
    return results

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
        # Controlla se abbiamo colonne anno/mese separate
        preprocessing_config = config.get('preprocessing', {})
        year_col = preprocessing_config.get('year_column', 'A_AnnoStipula')
        month_col = preprocessing_config.get('month_column', 'A_MeseStipula')
        
        if year_col in X_train.columns and month_col in X_train.columns:
            # Usa la nuova validazione per anno/mese
            from .temporal_utils import validate_temporal_split_year_month
            validation_results['temporal_split'] = validate_temporal_split_year_month(
                X_train, X_val, X_test,
                year_column=year_col,
                month_column=month_col,
                min_gap_months=quality_config.get('min_temporal_gap_months', 1)
            )
        else:
            # Fallback alla validazione tradizionale
            validation_results['temporal_split'] = validate_temporal_split(
                X_train, X_val, X_test,
                date_column=preprocessing_config.get('date_column', 'A_DataStipula'),
                min_gap_days=quality_config.get('min_temporal_gap_days', 30)
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