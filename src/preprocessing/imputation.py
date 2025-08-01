import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

def impute_missing_values(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputazione intelligente dei valori mancanti.
    
    Args:
        df: DataFrame con valori mancanti
        target_column: Nome della colonna target (da escludere dall'imputazione)
        
    Returns:
        Tuple con DataFrame imputato e informazioni sugli imputer
    """
    logger.info("Imputazione valori mancanti...")

    missing_before = df.isnull().sum().sum()
    logger.info(f"Valori mancanti totali: {missing_before}")

    if missing_before == 0:
        logger.info("Nessun valore mancante da imputare")
        return df, {}

    imputers = {}
    skipped = []

    # Numeriche: mediana
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = [col for col in num_cols if col != target_column]  # Escludi target

    if len(num_cols) > 0:
        missing_num = df[num_cols].isnull().sum()
        cols_with_missing = missing_num[missing_num > 0].index.tolist()

        if cols_with_missing:
            # Filtra solo le colonne con almeno un valore NON nullo
            usable_cols = [col for col in cols_with_missing if df[col].notnull().any()]

            if usable_cols:
                logger.info(f"Imputazione mediana per {len(usable_cols)} colonne numeriche")
                imputer_num = SimpleImputer(strategy='median')
                imputed_data = imputer_num.fit_transform(df[usable_cols])
                df[usable_cols] = pd.DataFrame(imputed_data, columns=usable_cols, index=df.index)
                imputers['numeric'] = imputer_num

            # Colonne completamente NaN
            skipped = list(set(cols_with_missing) - set(usable_cols))
            if skipped:
                logger.warning(f"Colonne numeriche interamente NaN (saltate): {skipped}")

    # Categoriche: "MISSING"
    cat_cols = df.select_dtypes(include='object').columns

    if len(cat_cols) > 0:
        missing_cat = df[cat_cols].isnull().sum()
        cols_with_missing = missing_cat[missing_cat > 0].index.tolist()

        if cols_with_missing:
            logger.info(f"Imputazione 'MISSING' per {len(cols_with_missing)} colonne categoriche")
            for col in cols_with_missing:
                df[col] = df[col].fillna('MISSING')

    # Drop delle colonne numeriche interamente NaN
    if skipped:
        logger.info(f"Droppo {len(skipped)} colonne interamente NaN: {skipped}")
        df = df.drop(columns=skipped)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Valori mancanti dopo imputazione: {missing_after}")

    return df, imputers

def handle_missing_values(
    df: pd.DataFrame, 
    target_column: str,
    strategy_numeric: str = 'median',
    strategy_categorical: str = 'constant',
    fill_value_categorical: str = 'MISSING'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Gestisce i valori mancanti con strategie configurabili.
    
    Args:
        df: DataFrame con valori mancanti
        target_column: Nome della colonna target
        strategy_numeric: Strategia per variabili numeriche ('mean', 'median', 'most_frequent')
        strategy_categorical: Strategia per variabili categoriche ('most_frequent', 'constant')
        fill_value_categorical: Valore per strategia 'constant'
        
    Returns:
        Tuple con DataFrame imputato e informazioni sugli imputer
    """
    logger.info(f"Gestione valori mancanti (numeric: {strategy_numeric}, categorical: {strategy_categorical})...")

    missing_before = df.isnull().sum().sum()
    logger.info(f"Valori mancanti totali: {missing_before}")

    if missing_before == 0:
        logger.info("Nessun valore mancante da gestire")
        return df, {}

    imputers = {}
    dropped_cols = []

    # Separazione colonne
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = [col for col in num_cols if col != target_column]
    cat_cols = df.select_dtypes(include='object').columns

    # Gestione colonne numeriche
    if len(num_cols) > 0:
        missing_num = df[num_cols].isnull().sum()
        cols_with_missing = missing_num[missing_num > 0].index.tolist()

        if cols_with_missing:
            # Identifica colonne completamente NaN
            completely_nan = [col for col in cols_with_missing if df[col].isnull().all()]
            if completely_nan:
                logger.warning(f"Rimosse {len(completely_nan)} colonne numeriche completamente NaN: {completely_nan}")
                df = df.drop(columns=completely_nan)
                dropped_cols.extend(completely_nan)
                cols_with_missing = [col for col in cols_with_missing if col not in completely_nan]
            
            # Imputa colonne rimanenti
            if cols_with_missing:
                logger.info(f"Imputazione {strategy_numeric} per {len(cols_with_missing)} colonne numeriche")
                imputer_num = SimpleImputer(strategy=strategy_numeric)
                imputed_data = imputer_num.fit_transform(df[cols_with_missing])
                df[cols_with_missing] = pd.DataFrame(imputed_data, columns=cols_with_missing, index=df.index)
                imputers['numeric'] = {
                    'imputer': imputer_num,
                    'columns': cols_with_missing,
                    'strategy': strategy_numeric
                }

    # Gestione colonne categoriche
    if len(cat_cols) > 0:
        missing_cat = df[cat_cols].isnull().sum()
        cols_with_missing = missing_cat[missing_cat > 0].index.tolist()

        if cols_with_missing:
            if strategy_categorical == 'constant':
                logger.info(f"Imputazione '{fill_value_categorical}' per {len(cols_with_missing)} colonne categoriche")
                for col in cols_with_missing:
                    df[col] = df[col].fillna(fill_value_categorical)
                
                imputers['categorical'] = {
                    'strategy': strategy_categorical,
                    'fill_value': fill_value_categorical,
                    'columns': cols_with_missing
                }
            else:
                logger.info(f"Imputazione {strategy_categorical} per {len(cols_with_missing)} colonne categoriche")
                imputer_cat = SimpleImputer(strategy=strategy_categorical)
                imputed_data = imputer_cat.fit_transform(df[cols_with_missing])
                df[cols_with_missing] = pd.DataFrame(imputed_data, columns=cols_with_missing, index=df.index)
                
                imputers['categorical'] = {
                    'imputer': imputer_cat,
                    'columns': cols_with_missing,
                    'strategy': strategy_categorical
                }

    missing_after = df.isnull().sum().sum()
    logger.info(f"Valori mancanti dopo imputazione: {missing_after}")
    
    if dropped_cols:
        imputers['dropped_columns'] = dropped_cols

    return df, imputers