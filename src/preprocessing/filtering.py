import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from typing import Tuple, List, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcola Cramér's V tra due variabili categoriche.
    
    Args:
        x: Prima variabile categorica
        y: Seconda variabile categorica
        
    Returns:
        Valore di Cramér's V (0-1)
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.values.sum()
    if n == 0:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

def analyze_cramers_correlations(df: pd.DataFrame, target_column: str, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Analizza le correlazioni Cramér's V tra variabili categoriche SENZA rimuovere colonne.
    Questa funzione può essere usata prima dello split per identificare potenziali problemi.
    
    Args:
        df: DataFrame con le variabili
        target_column: Nome della colonna target (da escludere dall'analisi)
        threshold: Soglia di correlazione per identificare correlazioni elevate
        
    Returns:
        Dictionary con informazioni sulle correlazioni trovate
    """
    logger.info(f"Analisi correlazioni Cramér's V (soglia: {threshold})...")
    
    # Seleziona categoriche "vere" escludendo il target
    cat_cols = df.select_dtypes(include='object').columns
    cat_cols = [col for col in cat_cols if col != target_column and df[col].nunique() > 1]
    
    analysis_info = {
        'threshold_used': threshold,
        'categorical_columns_analyzed': cat_cols,
        'high_correlations': [],
        'correlation_matrix': {},
        'recommended_removals': []
    }
    
    if len(cat_cols) < 2:
        logger.info("Meno di 2 variabili categoriche, skip analisi correlazione")
        return analysis_info
    
    logger.info(f"Analisi correlazione su {len(cat_cols)} variabili categoriche")
    
    # Calcola matrice Cramér's V
    cramer_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                cramer_matrix.loc[col1, col2] = 1.0
            elif pd.isna(cramer_matrix.loc[col1, col2]):
                value = cramers_v(df[col1], df[col2])
                cramer_matrix.loc[col1, col2] = value
                cramer_matrix.loc[col2, col1] = value
    
    analysis_info['correlation_matrix'] = cramer_matrix.to_dict()
    
    # Identifica correlazioni elevate
    high_correlations = []
    potential_removals = set()
    
    for col in cramer_matrix.columns:
        high_corr_cols = cramer_matrix[col][
            (cramer_matrix[col] > threshold) & (cramer_matrix[col] < 1.0)
        ].index.tolist()
        
        for other_col in high_corr_cols:
            if col < other_col:  # Evita duplicati (A,B) e (B,A)
                correlation_value = cramer_matrix.loc[col, other_col]
                high_correlations.append({
                    'column1': col,
                    'column2': other_col,
                    'cramers_v': correlation_value
                })
                
                # Suggerisci rimozione della colonna con meno valori unici
                if df[col].nunique() <= df[other_col].nunique():
                    potential_removals.add(col)
                else:
                    potential_removals.add(other_col)
    
    analysis_info['high_correlations'] = high_correlations
    analysis_info['recommended_removals'] = list(potential_removals)
    
    if high_correlations:
        logger.info(f"Trovate {len(high_correlations)} correlazioni elevate:")
        for corr in high_correlations:
            logger.info(f"  {corr['column1']} <-> {corr['column2']}: {corr['cramers_v']:.3f}")
        logger.info(f"Colonne consigliate per rimozione: {analysis_info['recommended_removals']}")
    else:
        logger.info("Nessuna correlazione elevata trovata")
    
    return analysis_info

def remove_highly_correlated_categorical(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rimuove variabili categoriche altamente correlate usando Cramér's V.
    
    Args:
        df: DataFrame con le variabili
        threshold: Soglia di correlazione
        
    Returns:
        Tuple con DataFrame filtrato e lista colonne rimosse
    """
    logger.info(f"Rimozione variabili categoriche correlate (soglia: {threshold})...")
    
    # Seleziona categoriche "vere"
    cat_cols = df.select_dtypes(include='object').columns
    cat_cols = [col for col in cat_cols if df[col].nunique() > 1]
    
    if len(cat_cols) < 2:
        logger.info("Meno di 2 variabili categoriche, skip correlazione")
        return df, []
    
    logger.info(f"Analisi correlazione su {len(cat_cols)} variabili categoriche")
    
    # Calcola matrice Cramér's V
    cramer_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                cramer_matrix.loc[col1, col2] = 1.0
            elif pd.isna(cramer_matrix.loc[col1, col2]):
                value = cramers_v(df[col1], df[col2])
                cramer_matrix.loc[col1, col2] = value
                cramer_matrix.loc[col2, col1] = value
    
    # Rimuovi categoriche con Cramér's V > soglia
    to_remove = set()
    for col in cramer_matrix.columns:
        high_corr = cramer_matrix[col][
            (cramer_matrix[col] > threshold) & (cramer_matrix[col] < 1.0)
        ].index
        for other in high_corr:
            if other not in to_remove:
                to_remove.add(other)
    
    removed_cols = list(to_remove)
    if removed_cols:
        df = df.drop(columns=removed_cols)
        logger.info(f"Rimosse {len(removed_cols)} variabili categoriche correlate: {removed_cols}")
    else:
        logger.info("Nessuna variabile categorica da rimuovere per correlazione")
    
    return df, removed_cols

def remove_highly_correlated_numeric(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Rimuove feature numeriche altamente correlate (solo su train).
    
    Args:
        X_train: Feature di training
        X_test: Feature di test
        threshold: Soglia di correlazione
        
    Returns:
        Tuple con X_train, X_test filtrati e lista colonne rimosse
    """
    logger.info(f"Rimozione feature numeriche correlate (soglia: {threshold})...")
    
    # Calcola correlazione solo su train
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        logger.info("Meno di 2 variabili numeriche, skip correlazione")
        return X_train, X_test, []
    
    corr = X_train[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        logger.info(f"Rimosse {len(to_drop)} feature numeriche correlate: {to_drop}")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        logger.info("Nessuna feature numerica da rimuovere per correlazione")
    
    return X_train, X_test, to_drop

def filter_features(
    df: pd.DataFrame = None,
    X_train: pd.DataFrame = None,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    cramer_threshold: float = 0.95,
    corr_threshold: float = 0.95
) -> Tuple:
    """
    Applica tutti i filtri alle feature.
    
    Args:
        df: DataFrame completo (per correlazioni categoriche, None se post-split)
        X_train: Feature di training (per correlazioni numeriche)
        X_val: Feature di validation (per correlazioni numeriche)
        X_test: Feature di test (per correlazioni numeriche)
        cramer_threshold: Soglia per Cramér's V
        corr_threshold: Soglia per correlazione numerica
        
    Returns:
        Tuple con df filtrato (o None), X_train filtrato, X_val filtrato (se presente), X_test filtrato (se presente), info filtri
    """
    logger.info("Avvio filtri delle feature...")
    
    filter_info = {}
    results = []
    
    # Filtro correlazioni categoriche (solo se df è fornito - pre-split)
    if df is not None:
        logger.info("Filtro correlazioni categoriche (pre-split)")
        df_filtered, removed_categorical = remove_highly_correlated_categorical(df, cramer_threshold)
        filter_info['removed_categorical'] = removed_categorical
        results.append(df_filtered)
    else:
        logger.info("Skip filtro categorico (modalità post-split)")
        filter_info['removed_categorical'] = []
        results.append(None)
    
    # Filtro correlazioni numeriche (su train/val/test se forniti)
    if X_train is not None:
        logger.info("Filtro correlazioni numeriche (post-split)")
        
        # Usa solo train e test per backwards compatibility con remove_highly_correlated_numeric
        X_train_filtered, X_test_temp, removed_numeric = remove_highly_correlated_numeric(
            X_train, X_test if X_test is not None else X_train, corr_threshold
        )
        filter_info['removed_numeric'] = removed_numeric
        
        results.append(X_train_filtered)
        
        # Applica lo stesso filtro a validation se presente
        if X_val is not None:
            # Rimuove le stesse colonne trovate nel training
            cols_to_keep = X_train_filtered.columns
            X_val_filtered = X_val[cols_to_keep]
            results.append(X_val_filtered)
        
        # Applica lo stesso filtro a test se presente
        if X_test is not None:
            cols_to_keep = X_train_filtered.columns
            X_test_filtered = X_test[cols_to_keep]
            results.append(X_test_filtered)
        
        shape_info = f"X_train {X_train_filtered.shape}"
        if X_val is not None:
            shape_info += f", X_val {X_val_filtered.shape}"
        if X_test is not None:
            shape_info += f", X_test {X_test_filtered.shape}"
            
        if df is not None:
            logger.info(f"Filtri completati. Shape: df {df_filtered.shape}, {shape_info}")
        else:
            logger.info(f"Filtri completati. Shape: {shape_info}")
    else:
        logger.info("Nessun set di training fornito per filtri numerici")
        filter_info['removed_numeric'] = []
        if df is not None:
            logger.info(f"Filtri completati. Shape df: {df_filtered.shape}")
    
    results.append(filter_info)
    return tuple(results)