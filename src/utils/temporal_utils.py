"""
Utility per gestire dati temporali con anno e mese separati.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from .logger import get_logger

logger = get_logger(__name__)

def create_composite_date_column(
    df: pd.DataFrame, 
    year_column: str = 'A_AnnoStipula', 
    month_column: str = 'A_MeseStipula',
    day: int = 15
) -> pd.DataFrame:
    """
    Crea una colonna data composita da anno e mese separati.
    
    Args:
        df: DataFrame con i dati
        year_column: Nome colonna anno
        month_column: Nome colonna mese  
        day: Giorno fisso da usare (default 15 - metà mese)
        
    Returns:
        DataFrame con colonna 'temporal_date' aggiunta
    """
    df_copy = df.copy()
    
    try:
        # Verifica che le colonne esistano
        if year_column not in df_copy.columns:
            raise ValueError(f"Colonna anno '{year_column}' non trovata")
        if month_column not in df_copy.columns:
            raise ValueError(f"Colonna mese '{month_column}' non trovata")
        
        # Gestisci valori mancanti
        year_vals = df_copy[year_column].fillna(0).astype(int)
        month_vals = df_copy[month_column].fillna(1).astype(int)
        
        # Correggi mesi fuori range (1-12)
        month_vals = month_vals.clip(1, 12)
        
        # Crea date composite (YYYY-MM-DD)
        date_strings = year_vals.astype(str) + '-' + month_vals.astype(str).str.zfill(2) + f'-{day:02d}'
        
        # Converti a datetime
        df_copy['temporal_date'] = pd.to_datetime(date_strings, errors='coerce')
        
        # Log statistiche
        valid_dates = df_copy['temporal_date'].notna().sum()
        total_dates = len(df_copy)
        
        logger.info(f"Date composite create: {valid_dates}/{total_dates} valide")
        logger.info(f"Range temporale: {df_copy['temporal_date'].min()} - {df_copy['temporal_date'].max()}")
        
        return df_copy
        
    except Exception as e:
        logger.error(f"Errore nella creazione date composite: {e}")
        raise

def calculate_month_gap(
    date1: pd.Timestamp, 
    date2: pd.Timestamp
) -> int:
    """
    Calcola la differenza in mesi tra due date.
    
    Args:
        date1: Prima data
        date2: Seconda data
        
    Returns:
        Differenza in mesi (può essere negativa)
    """
    if pd.isna(date1) or pd.isna(date2):
        return 0
        
    year_diff = date2.year - date1.year
    month_diff = date2.month - date1.month
    
    return year_diff * 12 + month_diff

def validate_temporal_split_year_month(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame,
    year_column: str = 'A_AnnoStipula',
    month_column: str = 'A_MeseStipula',
    min_gap_months: int = 1
) -> dict:
    """
    Valida split temporale usando anno/mese separati.
    
    Args:
        X_train: Features di training
        X_val: Features di validation
        X_test: Features di test
        year_column: Nome colonna anno
        month_column: Nome colonna mese
        min_gap_months: Gap minimo in mesi
        
    Returns:
        Dictionary con risultati validazione
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    
    try:
        # Verifica colonne
        for df_name, df in [('train', X_train), ('val', X_val), ('test', X_test)]:
            if year_column not in df.columns:
                results['errors'].append(f"Colonna {year_column} non trovata in {df_name}")
                results['valid'] = False
            if month_column not in df.columns:
                results['errors'].append(f"Colonna {month_column} non trovata in {df_name}")
                results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Crea date composite per ogni set
        train_composite = create_composite_date_column(
            X_train[[year_column, month_column]], year_column, month_column
        )
        val_composite = create_composite_date_column(
            X_val[[year_column, month_column]], year_column, month_column
        )
        test_composite = create_composite_date_column(
            X_test[[year_column, month_column]], year_column, month_column
        )
        
        # Calcola range temporali
        train_range = (train_composite['temporal_date'].min(), train_composite['temporal_date'].max())
        val_range = (val_composite['temporal_date'].min(), val_composite['temporal_date'].max())
        test_range = (test_composite['temporal_date'].min(), test_composite['temporal_date'].max())
        
        results['stats'] = {
            'train_range': train_range,
            'val_range': val_range,
            'test_range': test_range,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        # Verifica ordine temporale
        if train_range[1] >= val_range[0]:
            results['errors'].append(
                f"Overlap temporale train-val: train fino a {train_range[1]}, val da {val_range[0]}"
            )
            results['valid'] = False
        
        if val_range[1] >= test_range[0]:
            results['errors'].append(
                f"Overlap temporale val-test: val fino a {val_range[1]}, test da {test_range[0]}"
            )
            results['valid'] = False
        
        # Calcola gap in mesi
        train_val_gap = calculate_month_gap(train_range[1], val_range[0])
        val_test_gap = calculate_month_gap(val_range[1], test_range[0])
        
        if train_val_gap < min_gap_months:
            results['warnings'].append(
                f"Gap train-val troppo piccolo: {train_val_gap} mesi < {min_gap_months}"
            )
        
        if val_test_gap < min_gap_months:
            results['warnings'].append(
                f"Gap val-test troppo piccolo: {val_test_gap} mesi < {min_gap_months}"
            )
        
        results['stats']['train_val_gap_months'] = train_val_gap
        results['stats']['val_test_gap_months'] = val_test_gap
        
        logger.info(f"Gap temporali: train-val {train_val_gap} mesi, val-test {val_test_gap} mesi")
        
    except Exception as e:
        results['errors'].append(f"Errore validazione: {str(e)}")
        results['valid'] = False
        logger.error(f"Errore validazione temporal split: {e}")
    
    return results

def temporal_sort_by_year_month(
    df: pd.DataFrame,
    year_column: str = 'A_AnnoStipula',
    month_column: str = 'A_MeseStipula'
) -> pd.DataFrame:
    """
    Ordina DataFrame per anno e mese.
    
    Args:
        df: DataFrame da ordinare
        year_column: Nome colonna anno
        month_column: Nome colonna mese
        
    Returns:
        DataFrame ordinato cronologicamente
    """
    try:
        # Crea colonna temporanea per ordinamento
        df_temp = create_composite_date_column(df, year_column, month_column)
        
        # Ordina per data composite
        df_sorted = df_temp.sort_values('temporal_date').reset_index(drop=True)
        
        # Rimuovi colonna temporanea
        if 'temporal_date' in df_sorted.columns:
            df_sorted = df_sorted.drop(columns=['temporal_date'])
        
        logger.info(f"Dataset ordinato cronologicamente per {year_column}/{month_column}")
        return df_sorted
        
    except Exception as e:
        logger.error(f"Errore nell'ordinamento temporale: {e}")
        logger.warning("Fallback: ordinamento per anno e mese separatamente")
        return df.sort_values([year_column, month_column]).reset_index(drop=True)