"""
Utility per la gestione robusta dei target originali vs trasformati.
Risolve i problemi di inconsistenza nella gestione dei target log/originali.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def determine_target_scale_and_get_original(
    y_series: pd.Series,
    target_column_name: str,
    dataframe_context: Optional[pd.DataFrame] = None,
    preprocessing_info: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Determina se il target è in scala log o originale e recupera la versione originale.
    
    Args:
        y_series: Serie target da analizzare
        target_column_name: Nome della colonna target
        dataframe_context: DataFrame completo (per cercare colonne originali)
        preprocessing_info: Info preprocessing per capire le trasformazioni applicate
        
    Returns:
        Tuple con target in scala originale e informazioni sulla logica applicata
        
    Note:
        - Risolve l'ambiguità dei file "*_log" che potrebbero contenere valori originali
        - Usa multiple euristiche per determinare la scala corretta
        - Documenta chiaramente la logica applicata per debugging
    """
    logger.info(f"🔍 Determinazione scala target per: {target_column_name}")
    
    analysis_info = {
        'input_column_name': target_column_name,
        'input_series_stats': {
            'min': float(y_series.min()),
            'max': float(y_series.max()),
            'mean': float(y_series.mean()),
            'has_negatives': (y_series < 0).any(),
            'has_zeros': (y_series == 0).any(),
            'range': float(y_series.max() - y_series.min())
        },
        'scale_determination_logic': [],
        'scale_determined': None,
        'transformation_applied': None
    }
    
    # EURISTICA 1: Controlla preprocessing_info se disponibile
    if preprocessing_info:
        log_info = preprocessing_info.get('steps_info', {}).get('log_transformation', {})
        if log_info:
            if log_info.get('applied', False):
                analysis_info['scale_determined'] = 'log'
                analysis_info['scale_determination_logic'].append("preprocessing_info indica trasformazione log applicata")
                logger.info("  📊 Preprocessing info: trasformazione log applicata")
            else:
                analysis_info['scale_determined'] = 'original'
                analysis_info['scale_determination_logic'].append("preprocessing_info indica nessuna trasformazione log")
                logger.info("  📊 Preprocessing info: nessuna trasformazione log")
    
    # EURISTICA 2: Controlla nome colonna
    if analysis_info['scale_determined'] is None:
        if '_log' in target_column_name.lower():
            # Controlla se effettivamente sembra log-trasformato
            if _looks_like_log_transformed(y_series):
                analysis_info['scale_determined'] = 'log'
                analysis_info['scale_determination_logic'].append("nome contiene '_log' e valori sembrano log-trasformati")
                logger.info("  📊 Nome colonna + analisi valori: scala log")
            else:
                analysis_info['scale_determined'] = 'original'
                analysis_info['scale_determination_logic'].append("nome contiene '_log' ma valori NON sembrano log-trasformati")
                logger.warning("  ⚠️  Nome contiene '_log' ma valori sembrano in scala originale!")
        else:
            analysis_info['scale_determined'] = 'original'
            analysis_info['scale_determination_logic'].append("nome non contiene '_log'")
            logger.info("  📊 Nome colonna: scala originale")
    
    # EURISTICA 3: Cerca colonna originale esplicita nel DataFrame
    original_column_name = None
    if dataframe_context is not None:
        # Cerca varianti del nome originale
        possible_original_names = [
            target_column_name.replace('_log', '_original'),
            target_column_name.replace('_log', ''),
            target_column_name + '_original',
            target_column_name.replace('log_', '')
        ]
        
        for possible_name in possible_original_names:
            if possible_name in dataframe_context.columns and possible_name != target_column_name:
                original_column_name = possible_name
                analysis_info['scale_determination_logic'].append(f"trovata colonna originale esplicita: {possible_name}")
                logger.info(f"  📊 Trovata colonna originale: {possible_name}")
                break
    
    # Determina target originale
    if original_column_name and dataframe_context is not None:
        # Usa colonna originale esplicita
        y_original = dataframe_context[original_column_name].copy()
        analysis_info['transformation_applied'] = 'used_explicit_original_column'
        analysis_info['original_column_used'] = original_column_name
        logger.info(f"  ✅ Usando colonna originale esplicita: {original_column_name}")
        
    elif analysis_info['scale_determined'] == 'log':
        # Applica trasformazione inversa
        y_original = np.expm1(y_series)
        analysis_info['transformation_applied'] = 'expm1_inverse_transform'
        logger.info("  🔄 Applicando expm1 per invertire log1p")
        
        # Validazione della trasformazione inversa
        if (y_original < 0).any():
            logger.warning("  ⚠️  expm1 ha prodotto valori negativi, possibile problema nella trasformazione originale")
            analysis_info['transformation_warnings'] = analysis_info.get('transformation_warnings', [])
            analysis_info['transformation_warnings'].append("expm1 ha prodotto valori negativi")
            
    else:
        # Usa valori originali come sono
        y_original = y_series.copy()
        analysis_info['transformation_applied'] = 'no_transformation_needed'
        logger.info("  📋 Usando valori come scala originale")
    
    # Statistiche target originale
    analysis_info['original_target_stats'] = {
        'min': float(y_original.min()),
        'max': float(y_original.max()),
        'mean': float(y_original.mean()),
        'has_negatives': (y_original < 0).any(),
        'has_zeros': (y_original == 0).any(),
        'range': float(y_original.max() - y_original.min())
    }
    
    # Validazione finale
    validation_result = _validate_original_target(y_series, y_original, analysis_info)
    analysis_info.update(validation_result)
    
    logger.info(f"  🏁 Target originale determinato: {analysis_info['transformation_applied']}")
    
    return y_original, analysis_info


def _looks_like_log_transformed(y_series: pd.Series) -> bool:
    """
    Euristica per determinare se una serie sembra log-trasformata.
    
    Args:
        y_series: Serie da analizzare
        
    Returns:
        True se sembra log-trasformata
    """
    # Caratteristiche tipiche di dati log-trasformati:
    # 1. Valori tipicamente positivi e piccoli (< 20 per log di prezzi immobiliari)
    # 2. Distribuzione più simmetrica
    # 3. Range più piccolo del previsto per dati originali
    
    stats = {
        'min': y_series.min(),
        'max': y_series.max(),
        'mean': y_series.mean(),
        'range': y_series.max() - y_series.min()
    }
    
    # Criterio 1: Se massimo è molto grande (>1000), probabilmente non è log
    if stats['max'] > 1000:
        return False
    
    # Criterio 2: Se ci sono valori negativi significativi, probabilmente non è log1p
    if stats['min'] < -10:
        return False
    
    # Criterio 3: Range ragionevole per log di prezzi immobiliari
    if stats['range'] > 0 and stats['range'] < 15:  # log(100) ≈ 4.6, log(1000000) ≈ 13.8
        return True
    
    return False


def _validate_original_target(
    y_input: pd.Series, 
    y_original: pd.Series, 
    analysis_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valida che il target originale sia ragionevole.
    
    Args:
        y_input: Serie target input
        y_original: Serie target originale determinata
        analysis_info: Info dell'analisi
        
    Returns:
        Dictionary con risultati validazione
    """
    validation = {
        'validation_passed': True,
        'validation_warnings': [],
        'validation_errors': []
    }
    
    # Check 1: Target originale dovrebbe avere valori positivi per prezzi immobiliari
    if (y_original <= 0).any():
        negative_count = (y_original <= 0).sum()
        warning = f"Target originale ha {negative_count} valori <= 0 (inaspettato per prezzi)"
        validation['validation_warnings'].append(warning)
        logger.warning(f"  ⚠️  {warning}")
    
    # Check 2: Range ragionevole per prezzi immobiliari
    price_range = y_original.max() - y_original.min()
    if price_range > 10_000_000:  # 10 milioni di range
        warning = f"Range target originale molto grande: {price_range:,.0f}"
        validation['validation_warnings'].append(warning)
        logger.warning(f"  ⚠️  {warning}")
    
    # Check 3: Confronto con input se possibile
    if analysis_info['scale_determined'] == 'original' and not np.allclose(y_input, y_original, rtol=1e-10):
        error = "Target determinato come 'original' ma valori non corrispondono"
        validation['validation_errors'].append(error)
        validation['validation_passed'] = False
        logger.error(f"  ❌ {error}")
    
    # Check 4: Se trasformazione inversa applicata, verifica coerenza
    if analysis_info['transformation_applied'] == 'exmp1_inverse_transform':
        # Riapplica log1p e verifica che sia simile all'input
        y_relogged = np.log1p(y_original)
        if not np.allclose(y_input, y_relogged, rtol=1e-6):
            error = "Trasformazione inversa non coerente: log1p(expm1(input)) != input"
            validation['validation_errors'].append(error)
            validation['validation_passed'] = False
            logger.error(f"  ❌ {error}")
    
    return validation


def create_target_scale_metadata(
    target_column: str,
    is_log_scale: bool,
    transformation_method: Optional[str] = None,
    preprocessing_pipeline_version: str = "v1.0"
) -> Dict[str, Any]:
    """
    Crea metadati chiari sulla scala del target per evitare confusione futura.
    
    Args:
        target_column: Nome della colonna target
        is_log_scale: True se il target è in scala logaritmica
        transformation_method: Metodo di trasformazione usato ('log1p', 'log', etc.)
        preprocessing_pipeline_version: Versione della pipeline
        
    Returns:
        Dictionary con metadati chiari
        
    Note:
        Questi metadati dovrebbero essere salvati insieme ai file per evitare
        ambiguità future sulla scala dei target.
    """
    metadata = {
        'target_scale_info': {
            'column_name': target_column,
            'is_log_scale': is_log_scale,
            'scale_type': 'logarithmic' if is_log_scale else 'original',
            'transformation_method': transformation_method,
            'pipeline_version': preprocessing_pipeline_version,
            'created_timestamp': pd.Timestamp.now().isoformat(),
            'interpretation_guide': {
                'log_scale': "Per ottenere valori originali, applicare np.expm1() se metodo=log1p, o np.exp() se metodo=log",
                'original_scale': "Valori già in scala originale, nessuna trasformazione necessaria"
            }
        }
    }
    
    return metadata