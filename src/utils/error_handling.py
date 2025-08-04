"""
Modulo per la gestione robusta degli errori e validazione.
Migliora la gestione degli errori in tutta l'applicazione.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps
import traceback
from .logger import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Eccezione base per errori di preprocessing."""
    pass


class DataValidationError(PreprocessingError):
    """Eccezione per errori di validazione dati."""
    pass


class ConfigurationError(PreprocessingError):
    """Eccezione per errori di configurazione."""
    pass


class PipelineExecutionError(PreprocessingError):
    """Eccezione per errori di esecuzione pipeline."""
    pass


def safe_execution(
    error_type: type = PreprocessingError,
    log_errors: bool = True,
    return_on_error: Any = None,
    reraise: bool = True
):
    """
    Decorator per l'esecuzione sicura di funzioni con gestione errori robusta.
    
    Args:
        error_type: Tipo di eccezione da rilanciare
        log_errors: Se True, logga gli errori
        return_on_error: Valore da restituire in caso di errore (se reraise=False)
        reraise: Se True, rilancia l'eccezione; se False, restituisce return_on_error
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"❌ Errore in {func.__name__}: {str(e)}")
                    logger.debug(f"Traceback completo: {traceback.format_exc()}")
                
                if reraise:
                    if isinstance(e, PreprocessingError):
                        raise
                    else:
                        raise error_type(f"Errore in {func.__name__}: {str(e)}") from e
                else:
                    return return_on_error
        return wrapper
    return decorator


def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    min_rows: int = 1,
    min_cols: int = 1,
    required_columns: Optional[List[str]] = None,
    check_duplicates: bool = False,
    check_nulls: bool = False,
    max_null_percentage: float = 0.5
) -> Dict[str, Any]:
    """
    Validazione robusta di un DataFrame.
    
    Args:
        df: DataFrame da validare
        name: Nome del DataFrame per i messaggi di errore
        min_rows: Numero minimo di righe richieste
        min_cols: Numero minimo di colonne richieste
        required_columns: Lista di colonne che devono essere presenti
        check_duplicates: Se True, controlla duplicati
        check_nulls: Se True, controlla percentuale di null
        max_null_percentage: Percentuale massima di null consentita
        
    Returns:
        Dictionary con risultati della validazione
        
    Raises:
        DataValidationError: Se la validazione fallisce
    """
    logger.info(f"🔍 Validazione {name}: {df.shape}")
    
    validation_results = {
        'name': name,
        'shape': df.shape,
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check 1: DataFrame non vuoto
    if df.empty:
        error_msg = f"{name} è vuoto"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
        raise DataValidationError(error_msg)
    
    # Check 2: Dimensioni minime
    if df.shape[0] < min_rows:
        error_msg = f"{name} ha solo {df.shape[0]} righe (minimo: {min_rows})"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    if df.shape[1] < min_cols:
        error_msg = f"{name} ha solo {df.shape[1]} colonne (minimo: {min_cols})"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    # Check 3: Colonne richieste
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"{name} manca delle colonne richieste: {missing_columns}"
            validation_results['errors'].append(error_msg)
            validation_results['validation_passed'] = False
    
    # Check 4: Duplicati
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['stats']['duplicate_rows'] = duplicate_count
        
        if duplicate_count > 0:
            duplicate_percentage = duplicate_count / len(df) * 100
            warning_msg = f"{name} ha {duplicate_count} righe duplicate ({duplicate_percentage:.1f}%)"
            validation_results['warnings'].append(warning_msg)
            logger.warning(f"⚠️  {warning_msg}")
    
    # Check 5: Valori null
    if check_nulls:
        total_cells = df.shape[0] * df.shape[1]
        null_count = df.isnull().sum().sum()
        null_percentage = null_count / total_cells * 100 if total_cells > 0 else 0
        
        validation_results['stats']['null_count'] = null_count
        validation_results['stats']['null_percentage'] = null_percentage
        
        if null_percentage > max_null_percentage * 100:
            error_msg = f"{name} ha {null_percentage:.1f}% di valori null (max consentito: {max_null_percentage*100:.1f}%)"
            validation_results['errors'].append(error_msg)
            validation_results['validation_passed'] = False
    
    # Statistiche aggiuntive
    validation_results['stats'].update({
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'dtypes_count': df.dtypes.value_counts().to_dict()
    })
    
    # Rilancia errore se validazione fallita
    if not validation_results['validation_passed']:
        error_summary = "; ".join(validation_results['errors'])
        raise DataValidationError(f"Validazione {name} fallita: {error_summary}")
    
    logger.info(f"✅ Validazione {name} superata")
    return validation_results


def validate_target_column(
    y_series: pd.Series,
    name: str = "Target",
    allow_negatives: bool = False,
    allow_zeros: bool = True,
    min_unique_values: int = 2,
    check_distribution: bool = True
) -> Dict[str, Any]:
    """
    Validazione specifica per colonne target.
    
    Args:
        y_series: Serie target da validare
        name: Nome per i messaggi
        allow_negatives: Se False, errore se ci sono valori negativi
        allow_zeros: Se False, errore se ci sono zeri
        min_unique_values: Numero minimo di valori unici
        check_distribution: Se True, analizza la distribuzione
        
    Returns:
        Dictionary con risultati validazione
        
    Raises:
        DataValidationError: Se validazione fallisce
    """
    logger.info(f"🎯 Validazione target {name}: {len(y_series)} valori")
    
    validation_results = {
        'name': name,
        'count': len(y_series),
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check base: non vuoto
    if y_series.empty:
        error_msg = f"Target {name} vuoto"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
        raise DataValidationError(error_msg)
    
    # Statistiche base
    stats = {
        'min': float(y_series.min()),
        'max': float(y_series.max()),
        'mean': float(y_series.mean()),
        'std': float(y_series.std()),
        'unique_values': y_series.nunique(),
        'null_count': int(y_series.isnull().sum()),
        'null_percentage': float(y_series.isnull().sum() / len(y_series) * 100)
    }
    validation_results['stats'] = stats
    
    # Check 1: Valori null
    if stats['null_count'] > 0:
        error_msg = f"Target {name} ha {stats['null_count']} valori null ({stats['null_percentage']:.1f}%)"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    # Check 2: Valori negativi
    if not allow_negatives and stats['min'] < 0:
        negative_count = (y_series < 0).sum()
        error_msg = f"Target {name} ha {negative_count} valori negativi (non consentiti)"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    # Check 3: Valori zero
    if not allow_zeros and (y_series == 0).any():
        zero_count = (y_series == 0).sum()
        error_msg = f"Target {name} ha {zero_count} valori zero (non consentiti)"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    # Check 4: Valori unici minimi
    if stats['unique_values'] < min_unique_values:
        error_msg = f"Target {name} ha solo {stats['unique_values']} valori unici (minimo: {min_unique_values})"
        validation_results['errors'].append(error_msg)
        validation_results['validation_passed'] = False
    
    # Check 5: Distribuzione
    if check_distribution:
        # Controlla outlier estremi
        if stats['std'] > 0:
            z_scores = np.abs((y_series - stats['mean']) / stats['std'])
            extreme_outliers = (z_scores > 5).sum()
            
            if extreme_outliers > 0:
                outlier_percentage = extreme_outliers / len(y_series) * 100
                warning_msg = f"Target {name} ha {extreme_outliers} outlier estremi (z>5, {outlier_percentage:.1f}%)"
                validation_results['warnings'].append(warning_msg)
                logger.warning(f"⚠️  {warning_msg}")
        
        # Controlla skewness estrema
        from scipy import stats as scipy_stats
        skewness = scipy_stats.skew(y_series)
        validation_results['stats']['skewness'] = float(skewness)
        
        if abs(skewness) > 3:
            warning_msg = f"Target {name} ha skewness estrema: {skewness:.2f}"
            validation_results['warnings'].append(warning_msg)
            logger.warning(f"⚠️  {warning_msg}")
    
    # Rilancia errore se validazione fallita
    if not validation_results['validation_passed']:
        error_summary = "; ".join(validation_results['errors'])
        raise DataValidationError(f"Validazione target {name} fallita: {error_summary}")
    
    logger.info(f"✅ Validazione target {name} superata")
    logger.info(f"📊 Stats: min={stats['min']:.2f}, max={stats['max']:.2f}, μ={stats['mean']:.2f}, σ={stats['std']:.2f}")
    
    return validation_results


def validate_config(
    config: Dict[str, Any],
    required_sections: Optional[List[str]] = None,
    required_fields: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Validazione configurazione.
    
    Args:
        config: Dictionary di configurazione
        required_sections: Sezioni obbligatorie
        required_fields: Campi obbligatori per sezione
        
    Returns:
        Dictionary con risultati validazione
        
    Raises:
        ConfigurationError: Se configurazione non valida
    """
    logger.info("⚙️  Validazione configurazione")
    
    validation_results = {
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'sections_found': list(config.keys())
    }
    
    # Check sezioni richieste
    if required_sections:
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            error_msg = f"Sezioni di configurazione mancanti: {missing_sections}"
            validation_results['errors'].append(error_msg)
            validation_results['validation_passed'] = False
    
    # Check campi richiesti per sezione
    if required_fields:
        for section, fields in required_fields.items():
            if section in config:
                section_config = config[section]
                missing_fields = [field for field in fields if field not in section_config]
                if missing_fields:
                    error_msg = f"Campi mancanti nella sezione '{section}': {missing_fields}"
                    validation_results['errors'].append(error_msg)
                    validation_results['validation_passed'] = False
    
    # Rilancia errore se validazione fallita
    if not validation_results['validation_passed']:
        error_summary = "; ".join(validation_results['errors'])
        raise ConfigurationError(f"Configurazione non valida: {error_summary}")
    
    logger.info("✅ Configurazione valida")
    return validation_results


class ValidationContext:
    """
    Context manager per la validazione robusta con rollback.
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.validations = []
        self.errors = []
        
    def __enter__(self):
        logger.info(f"🔒 Inizio validazione: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.info(f"✅ Validazione completata: {self.operation_name}")
        else:
            logger.error(f"❌ Validazione fallita: {self.operation_name} - {exc_val}")
        return False  # Non sopprime l'eccezione
    
    def add_validation(self, validation_func: Callable, *args, **kwargs):
        """Aggiunge una validazione da eseguire."""
        try:
            result = validation_func(*args, **kwargs)
            self.validations.append({
                'function': validation_func.__name__,
                'result': result,
                'success': True
            })
            return result
        except Exception as e:
            error_info = {
                'function': validation_func.__name__,
                'error': str(e),
                'success': False
            }
            self.validations.append(error_info)
            self.errors.append(error_info)
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Ottiene riassunto delle validazioni."""
        return {
            'operation': self.operation_name,
            'total_validations': len(self.validations),
            'successful_validations': len([v for v in self.validations if v['success']]),
            'failed_validations': len(self.errors),
            'errors': self.errors
        }