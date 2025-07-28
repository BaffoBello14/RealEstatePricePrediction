import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

def remove_non_predictive_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Rimuove colonne che non dovrebbero essere disponibili a priori.
    
    Args:
        df: DataFrame da pulire
        cols_to_drop: Lista delle colonne da rimuovere
        
    Returns:
        DataFrame senza le colonne specificate
    """
    logger.info("Rimozione colonne non predittive...")
    
    initial_cols = df.shape[1]
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Rimosse {len(existing_cols)} colonne: {existing_cols}")
    else:
        logger.info("Nessuna colonna da rimuovere trovata")
    
    logger.info(f"Colonne: {initial_cols} -> {df.shape[1]}")
    return df

def remove_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Rimuove colonne costanti o quasi-costanti.
    
    Args:
        df: DataFrame da pulire
        
    Returns:
        Tuple con DataFrame pulito e lista colonne rimosse
    """
    logger.info("Rimozione colonne costanti...")
    
    initial_cols = df.shape[1]
    nunici = df.nunique(dropna=False)
    costanti = nunici[nunici <= 1].index.tolist()
    
    if costanti:
        df = df.drop(columns=costanti)
        logger.info(f"Rimosse {len(costanti)} colonne costanti: {costanti}")
    else:
        logger.info("Nessuna colonna costante trovata")
    
    logger.info(f"Colonne: {initial_cols} -> {df.shape[1]}")
    return df, costanti

def transform_target_and_detect_outliers(
    y_train: pd.Series, 
    X_train_scaled: np.ndarray,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.1,
    min_methods: int = 2
) -> Tuple[pd.Series, np.ndarray, IsolationForest]:
    """
    Trasforma il target e rileva outliers solo sul training set.
    
    Args:
        y_train: Serie target di training
        X_train_scaled: Features di training scalate
        z_threshold: Soglia per Z-score
        iqr_multiplier: Moltiplicatore per IQR
        contamination: Parametro per Isolation Forest
        min_methods: Numero minimo di metodi che devono identificare un outlier
        
    Returns:
        Tuple con target trasformato, mask outliers, detector
    """
    logger.info("Trasformazione target e rilevamento outliers...")
    
    # Step 1: Trasformazione logaritmica
    original_skew = stats.skew(y_train)
    y_train_log = np.log1p(y_train)
    log_skew = stats.skew(y_train_log)
    
    logger.info(f"Skewness originale: {original_skew:.3f}")
    logger.info(f"Skewness dopo log: {log_skew:.3f}")
    
    # Step 2: Outlier detection
    original_shape = len(y_train_log)
    
    # Metodo 1: Z-Score
    z_scores = stats.zscore(y_train_log)
    z_outliers = np.abs(z_scores) > z_threshold
    
    # Metodo 2: IQR
    Q1 = y_train_log.quantile(0.25)
    Q3 = y_train_log.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = (y_train_log < (Q1 - iqr_multiplier * IQR)) | (y_train_log > (Q3 + iqr_multiplier * IQR))
    
    # Metodo 3: Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_outliers = iso_forest.fit_predict(X_train_scaled) == -1
    
    # Combina i metodi
    combined_outliers = (
        z_outliers.astype(int) + 
        iqr_outliers.astype(int) + 
        iso_outliers.astype(int)
    ) >= min_methods
    
    # Log risultati
    logger.info(f"Z-Score (>{z_threshold}): {z_outliers.sum()} outliers ({z_outliers.sum()/len(y_train_log)*100:.2f}%)")
    logger.info(f"IQR ({iqr_multiplier}x): {iqr_outliers.sum()} outliers ({iqr_outliers.sum()/len(y_train_log)*100:.2f}%)")
    logger.info(f"Isolation Forest: {iso_outliers.sum()} outliers ({iso_outliers.sum()/len(y_train_log)*100:.2f}%)")
    logger.info(f"Combined (≥{min_methods} metodi): {combined_outliers.sum()} outliers ({combined_outliers.sum()/len(y_train_log)*100:.2f}%)")
    
    # Visualizzazione
    _plot_target_transformation(y_train, y_train_log, combined_outliers)
    
    # Test di normalità
    _test_normality(y_train, y_train_log, combined_outliers)
    
    return y_train_log, combined_outliers, iso_forest

def _plot_target_transformation(y_train: pd.Series, y_train_log: pd.Series, outliers_mask: np.ndarray) -> None:
    """
    Crea grafici per visualizzare la trasformazione del target.
    
    Args:
        y_train: Target originale
        y_train_log: Target log-trasformato
        outliers_mask: Mask degli outliers
    """
    from scipy.stats import probplot
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    original_skew = stats.skew(y_train)
    log_skew = stats.skew(y_train_log)
    
    # Row 1: Target originale
    axes[0,0].hist(y_train, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title(f'Target Originale\nSkewness: {original_skew:.3f}')
    axes[0,0].set_xlabel('Target')
    
    axes[0,1].boxplot(y_train, vert=True)
    axes[0,1].set_title('Boxplot Originale')
    axes[0,1].set_ylabel('Target')
    
    # Q-Q Plot originale
    probplot(y_train, dist="norm", plot=axes[0,2])
    axes[0,2].set_title('Q-Q Plot Originale')
    
    # Row 2: Target log-trasformato con outliers
    axes[1,0].hist(y_train_log[~outliers_mask], bins=50, alpha=0.7, label='Normal', color='blue')
    if outliers_mask.sum() > 0:
        axes[1,0].hist(y_train_log[outliers_mask], bins=20, alpha=0.7, label='Outliers', color='red')
    axes[1,0].set_title(f'Target Log-trasformato\nSkewness: {log_skew:.3f}')
    axes[1,0].set_xlabel('log(Target + 1)')
    axes[1,0].legend()
    
    axes[1,1].boxplot(y_train_log, vert=True)
    axes[1,1].set_title('Boxplot Log-trasformato')
    axes[1,1].set_ylabel('log(Target + 1)')
    
    # Q-Q Plot log-trasformato
    probplot(y_train_log, dist="norm", plot=axes[1,2])
    axes[1,2].set_title('Q-Q Plot Log-trasformato')
    
    plt.tight_layout()
    plt.show()

def _test_normality(y_train: pd.Series, y_train_log: pd.Series, outliers_mask: np.ndarray) -> None:
    """
    Esegue test di normalità sui dati.
    
    Args:
        y_train: Target originale
        y_train_log: Target log-trasformato
        outliers_mask: Mask degli outliers
    """
    logger.info("Test di Normalità:")
    
    # Test su campione per performance
    sample_size = min(5000, len(y_train))
    _, p_orig = normaltest(y_train.sample(sample_size, random_state=42))
    logger.info(f"Target originale - D'Agostino-Pearson p-value: {p_orig:.6f}")
    
    y_train_log_clean = y_train_log[~outliers_mask]
    if len(y_train_log_clean) > 0:
        sample_size_clean = min(5000, len(y_train_log_clean))
        if len(y_train_log_clean) >= sample_size_clean:
            _, p_log = normaltest(y_train_log_clean.sample(sample_size_clean, random_state=42))
            logger.info(f"Target log-trasformato (senza outliers) - D'Agostino-Pearson p-value: {p_log:.6f}")

def clean_data(
    df: pd.DataFrame, 
    target_column: str,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applica tutte le operazioni di pulizia ai dati.
    
    Args:
        df: DataFrame da pulire  
        target_column: Nome della colonna target
        config: Dizionario con i parametri di configurazione
        
    Returns:
        Tuple con DataFrame pulito e informazioni sulle operazioni
    """
    logger.info("Avvio pulizia completa dei dati...")
    
    cleaning_info = {}
    
    # Rimozione colonne non predittive
    cols_to_drop = ['A_Id', 'A_Codice', 'A_Prezzo', 'AI_Id']
    df = remove_non_predictive_columns(df, cols_to_drop)
    
    # Rimozione colonne costanti
    df, removed_constants = remove_constant_columns(df)
    cleaning_info['removed_constants'] = removed_constants
    
    logger.info(f"Pulizia completata. Shape finale: {df.shape}")
    
    return df, cleaning_info