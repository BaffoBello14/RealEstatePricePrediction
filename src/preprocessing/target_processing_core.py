"""
Modulo core per la gestione del target.
Separa la trasformazione logaritmica dall'outlier detection per maggiore chiarezza e modularità.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any, Optional, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


def transform_target_log(
    y_target: pd.Series,
    method: str = 'log1p',
    custom_offset: Optional[float] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Trasformazione logaritmica del target separata dall'outlier detection.
    
    Args:
        y_target: Serie target da trasformare
        method: Metodo di trasformazione ('log1p', 'log', 'custom')
        custom_offset: Offset custom per evitare valori negativi o zero
        
    Returns:
        Tuple con target trasformato e informazioni sulla trasformazione
        
    Note:
        - Funzione SEPARATA dalla detection degli outlier per chiarezza
        - Supporta multiple strategie di trasformazione
        - Mantiene traccia delle statistiche prima/dopo trasformazione
    """
    logger.info(f"🔢 Trasformazione logaritmica target (metodo: {method})")
    
    # Validazione input
    if y_target.empty:
        raise ValueError("Target series vuota")
    
    # Statistiche pre-trasformazione
    original_stats = {
        'min': float(y_target.min()),
        'max': float(y_target.max()),
        'mean': float(y_target.mean()),
        'std': float(y_target.std()),
        'median': float(y_target.median()),
        'skewness': float(stats.skew(y_target)),
        'kurtosis': float(stats.kurtosis(y_target)),
        'has_zeros': (y_target == 0).any(),
        'has_negatives': (y_target < 0).any(),
        'count': len(y_target),
        'non_null_count': int(y_target.notna().sum())
    }
    
    logger.info(f"📊 Target originale: min={original_stats['min']:.3f}, max={original_stats['max']:.3f}, "
               f"μ={original_stats['mean']:.3f}, σ={original_stats['std']:.3f}")
    logger.info(f"📈 Skewness originale: {original_stats['skewness']:.3f}")
    
    # Applica trasformazione
    try:
        if method == 'log1p':
            # log1p gestisce automaticamente i valori zero
            if original_stats['has_negatives']:
                logger.warning("⚠️  Target ha valori negativi, log1p potrebbe non essere appropriato")
            
            y_transformed = np.log1p(y_target)
            transform_applied = 'log1p'
            
        elif method == 'log':
            # log naturale, richiede offset per valori <= 0
            if original_stats['has_zeros'] or original_stats['has_negatives']:
                if custom_offset is None:
                    # Calcola offset automatico
                    min_value = original_stats['min']
                    offset = abs(min_value) + 1 if min_value <= 0 else 0
                else:
                    offset = custom_offset
                
                logger.info(f"🔧 Applicando offset {offset} per valori <= 0")
                y_transformed = np.log(y_target + offset)
                transform_applied = f'log(x + {offset})'
            else:
                y_transformed = np.log(y_target)
                transform_applied = 'log'
                
        elif method == 'custom':
            if custom_offset is None:
                raise ValueError("custom_offset richiesto per metodo 'custom'")
            
            y_transformed = np.log(y_target + custom_offset)
            transform_applied = f'log(x + {custom_offset})'
            
        else:
            raise ValueError(f"Metodo '{method}' non supportato. Usa: 'log1p', 'log', 'custom'")
        
        # Statistiche post-trasformazione
        transformed_stats = {
            'min': float(y_transformed.min()),
            'max': float(y_transformed.max()),
            'mean': float(y_transformed.mean()),
            'std': float(y_transformed.std()),
            'median': float(y_transformed.median()),
            'skewness': float(stats.skew(y_transformed)),
            'kurtosis': float(stats.kurtosis(y_transformed)),
            'has_infinite': np.isinf(y_transformed).any(),
            'has_nan': y_transformed.isna().any()
        }
        
        # Controllo qualità trasformazione
        quality_checks = {
            'skewness_improved': abs(transformed_stats['skewness']) < abs(original_stats['skewness']),
            'no_infinite_values': not transformed_stats['has_infinite'],
            'no_nan_values': not transformed_stats['has_nan'],
            'skewness_reduction': original_stats['skewness'] - transformed_stats['skewness']
        }
        
        # Info trasformazione
        transform_info = {
            'method_used': method,
            'transform_applied': transform_applied,
            'applied': True,
            'target_files_contain_log_values': True,
            'original_stats': original_stats,
            'transformed_stats': transformed_stats,
            'quality_checks': quality_checks,
            'warnings': []
        }
        
        # Log risultati
        logger.info(f"📊 Target trasformato: min={transformed_stats['min']:.3f}, max={transformed_stats['max']:.3f}, "
                   f"μ={transformed_stats['mean']:.3f}, σ={transformed_stats['std']:.3f}")
        logger.info(f"📈 Skewness trasformata: {transformed_stats['skewness']:.3f} "
                   f"(miglioramento: {quality_checks['skewness_reduction']:.3f})")
        
        # Warning per problemi
        if transformed_stats['has_infinite']:
            warning_msg = "Trasformazione ha generato valori infiniti"
            logger.warning(f"⚠️  {warning_msg}")
            transform_info['warnings'].append(warning_msg)
        
        if transformed_stats['has_nan']:
            warning_msg = "Trasformazione ha generato valori NaN"
            logger.warning(f"⚠️  {warning_msg}")
            transform_info['warnings'].append(warning_msg)
        
        if not quality_checks['skewness_improved']:
            warning_msg = f"Trasformazione non ha migliorato la skewness ({original_stats['skewness']:.3f} → {transformed_stats['skewness']:.3f})"
            logger.warning(f"⚠️  {warning_msg}")
            transform_info['warnings'].append(warning_msg)
        
        logger.info("✅ Trasformazione logaritmica completata")
        return y_transformed, transform_info
        
    except Exception as e:
        logger.error(f"❌ Errore nella trasformazione logaritmica: {str(e)}")
        
        # Ritorna target originale con info di errore
        error_info = {
            'method_used': method,
            'applied': False,
            'target_files_contain_log_values': False,
            'error': str(e),
            'original_stats': original_stats
        }
        
        return y_target, error_info


def detect_outliers_univariate(
    y_target: pd.Series,
    methods: List[str] = ['zscore', 'iqr', 'modified_zscore'],
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    modified_z_threshold: float = 3.5,
    min_methods_agreement: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detection outlier univariata separata dalla trasformazione log.
    
    Args:
        y_target: Serie target per outlier detection
        methods: Lista metodi da utilizzare
        z_threshold: Soglia per Z-score standard
        iqr_multiplier: Moltiplicatore per IQR
        modified_z_threshold: Soglia per Modified Z-score (più robusta)
        min_methods_agreement: Numero minimo metodi che devono concordare
        
    Returns:
        Tuple con mask outlier e informazioni dettagliate
        
    Note:
        - Separata dalla trasformazione logaritmica
        - Supporta multipli metodi configurabili
        - Richiede accordo tra metodi per robustezza
    """
    logger.info(f"🔍 Outlier detection univariata (metodi: {methods})")
    
    if y_target.empty:
        raise ValueError("Target series vuota")
    
    n_samples = len(y_target)
    outlier_votes = np.zeros(n_samples, dtype=int)
    method_results = {}
    
    # Statistiche base
    target_stats = {
        'count': n_samples,
        'mean': float(y_target.mean()),
        'std': float(y_target.std()),
        'median': float(y_target.median()),
        'q1': float(y_target.quantile(0.25)),
        'q3': float(y_target.quantile(0.75)),
        'min': float(y_target.min()),
        'max': float(y_target.max())
    }
    
    # METODO 1: Z-Score standard
    if 'zscore' in methods:
        logger.info(f"   🔸 Z-Score (soglia: {z_threshold})")
        
        z_scores = np.abs(stats.zscore(y_target))
        z_outliers = z_scores > z_threshold
        outlier_votes[z_outliers] += 1
        
        z_count = z_outliers.sum()
        method_results['zscore'] = {
            'outliers_count': int(z_count),
            'outliers_percentage': float(z_count / n_samples * 100),
            'threshold_used': z_threshold,
            'outlier_indices': np.where(z_outliers)[0].tolist()
        }
        
        logger.info(f"      → {z_count} outliers ({z_count/n_samples*100:.1f}%)")
    
    # METODO 2: IQR
    if 'iqr' in methods:
        logger.info(f"   🔸 IQR (moltiplicatore: {iqr_multiplier})")
        
        Q1 = target_stats['q1']
        Q3 = target_stats['q3']
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        iqr_outliers = (y_target < lower_bound) | (y_target > upper_bound)
        outlier_votes[iqr_outliers] += 1
        
        iqr_count = iqr_outliers.sum()
        method_results['iqr'] = {
            'outliers_count': int(iqr_count),
            'outliers_percentage': float(iqr_count / n_samples * 100),
            'iqr_value': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'multiplier_used': iqr_multiplier,
            'outlier_indices': np.where(iqr_outliers)[0].tolist()
        }
        
        logger.info(f"      → {iqr_count} outliers ({iqr_count/n_samples*100:.1f}%) bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    # METODO 3: Modified Z-Score (più robusto)
    if 'modified_zscore' in methods:
        logger.info(f"   🔸 Modified Z-Score (soglia: {modified_z_threshold})")
        
        median = target_stats['median']
        mad = np.median(np.abs(y_target - median))  # Median Absolute Deviation
        
        if mad == 0:
            # Fallback se MAD = 0
            logger.warning("      ⚠️  MAD = 0, usando deviazione standard")
            modified_z = np.abs(y_target - median) / target_stats['std']
        else:
            modified_z = 0.6745 * np.abs(y_target - median) / mad
        
        modified_z_outliers = modified_z > modified_z_threshold
        outlier_votes[modified_z_outliers] += 1
        
        mod_z_count = modified_z_outliers.sum()
        method_results['modified_zscore'] = {
            'outliers_count': int(mod_z_count),
            'outliers_percentage': float(mod_z_count / n_samples * 100),
            'threshold_used': modified_z_threshold,
            'mad_value': float(mad),
            'outlier_indices': np.where(modified_z_outliers)[0].tolist()
        }
        
        logger.info(f"      → {mod_z_count} outliers ({mod_z_count/n_samples*100:.1f}%) MAD: {mad:.3f}")
    
    # Combina risultati: outlier se almeno min_methods_agreement metodi concordano
    final_outliers_mask = outlier_votes >= min_methods_agreement
    final_outliers_count = final_outliers_mask.sum()
    
    # Analisi accordo tra metodi
    agreement_analysis = {}
    for votes in range(len(methods) + 1):
        count = (outlier_votes == votes).sum()
        if count > 0:
            agreement_analysis[f'{votes}_methods'] = {
                'count': int(count),
                'percentage': float(count / n_samples * 100)
            }
    
    detection_info = {
        'methods_used': methods,
        'parameters': {
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier,
            'modified_z_threshold': modified_z_threshold,
            'min_methods_agreement': min_methods_agreement
        },
        'target_stats': target_stats,
        'method_results': method_results,
        'final_outliers': {
            'count': int(final_outliers_count),
            'percentage': float(final_outliers_count / n_samples * 100),
            'indices': np.where(final_outliers_mask)[0].tolist()
        },
        'agreement_analysis': agreement_analysis
    }
    
    logger.info(f"🏁 Outlier finali: {final_outliers_count} ({final_outliers_count/n_samples*100:.1f}%) "
               f"con accordo ≥{min_methods_agreement} metodi")
    
    return final_outliers_mask, detection_info


def detect_outliers_multivariate(
    X_features: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detection outlier multivariata usando Isolation Forest.
    
    Args:
        X_features: DataFrame con feature per detection multivariata
        contamination: Frazione attesa di outlier
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con mask outlier e informazioni dettagliate
        
    Note:
        - Separata dalla detection univariata
        - Usa solo feature numeriche
        - Appropriata quando si hanno multiple feature correlate
    """
    logger.info(f"🌐 Outlier detection multivariata (contamination: {contamination})")
    
    # Seleziona solo feature numeriche
    numeric_features = X_features.select_dtypes(include=[np.number])
    
    if numeric_features.empty:
        logger.warning("⚠️  Nessuna feature numerica per detection multivariata")
        return np.zeros(len(X_features), dtype=bool), {
            'skipped': True,
            'reason': 'Nessuna feature numerica disponibile'
        }
    
    n_samples, n_features = numeric_features.shape
    logger.info(f"📊 Usando {n_features} feature numeriche su {n_samples} campioni")
    
    try:
        # Applica Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=1  # Per consistenza
        )
        
        outlier_predictions = iso_forest.fit_predict(numeric_features)
        outliers_mask = outlier_predictions == -1
        outliers_count = outliers_mask.sum()
        
        # Scores (più negativo = più outlier)
        outlier_scores = iso_forest.decision_function(numeric_features)
        
        detection_info = {
            'method': 'isolation_forest',
            'n_features_used': n_features,
            'feature_names': numeric_features.columns.tolist(),
            'contamination': contamination,
            'outliers_count': int(outliers_count),
            'outliers_percentage': float(outliers_count / n_samples * 100),
            'outlier_indices': np.where(outliers_mask)[0].tolist(),
            'outlier_scores_stats': {
                'min': float(outlier_scores.min()),
                'max': float(outlier_scores.max()),
                'mean': float(outlier_scores.mean()),
                'std': float(outlier_scores.std())
            }
        }
        
        logger.info(f"✅ Outlier multivariati: {outliers_count} ({outliers_count/n_samples*100:.1f}%)")
        
        return outliers_mask, detection_info
        
    except Exception as e:
        logger.error(f"❌ Errore in detection multivariata: {str(e)}")
        return np.zeros(n_samples, dtype=bool), {
            'error': str(e),
            'n_features_attempted': n_features
        }