import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any, List
from ..utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame, target_column: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pulisce i dati del dataset applicando varie operazioni di cleaning.
    
    Args:
        df: DataFrame da pulire
        target_column: Nome della colonna target
        config: Configurazione del preprocessing
        
    Returns:
        Tuple con DataFrame pulito e informazioni sul cleaning
    """
    logger.info("Inizio pulizia dati...")
    
    original_shape = df.shape
    cleaning_info = {'original_shape': original_shape}
    
    # 1. Sostituisce stringhe vuote con NaN
    logger.info("Sostituzione stringhe vuote con NaN...")
    df = df.replace('', np.nan)
    
    # 2. Rimuove righe dove il target è mancante
    if target_column in df.columns:
        target_null_count = df[target_column].isnull().sum()
        if target_null_count > 0:
            logger.info(f"Rimozione {target_null_count} righe con target mancante...")
            df = df.dropna(subset=[target_column])
            cleaning_info['target_null_removed'] = target_null_count
    
    # 3. Rimuove colonne completamente vuote
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        logger.info(f"Rimozione {len(empty_cols)} colonne completamente vuote...")
        df = df.drop(columns=empty_cols)
        cleaning_info['empty_columns_removed'] = empty_cols
    
    # 4. Rimuove duplicati completi
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Rimozione {duplicates} righe duplicate...")
        df = df.drop_duplicates()
        cleaning_info['duplicates_removed'] = duplicates
    
    # 5. Conversione automatica di tipo per colonne numeriche
    numeric_conversions = 0
    for col in df.columns:
        if col != target_column and df[col].dtype == 'object':
            try:
                # Prova conversione a numerico
                df_numeric = pd.to_numeric(df[col], errors='coerce')
                # Se non troppe perdite di dati, applica conversione
                if df_numeric.notna().sum() > 0.8 * len(df[col]):
                    df[col] = df_numeric
                    numeric_conversions += 1
            except Exception:
                pass
    
    if numeric_conversions > 0:
        logger.info(f"Convertite {numeric_conversions} colonne a tipo numerico")
        cleaning_info['numeric_conversions'] = numeric_conversions
    
    final_shape = df.shape
    cleaning_info['final_shape'] = final_shape
    cleaning_info['rows_removed'] = original_shape[0] - final_shape[0]
    cleaning_info['columns_removed'] = original_shape[1] - final_shape[1]
    
    logger.info(f"Pulizia completata: {original_shape} -> {final_shape}")
    logger.info(f"Righe rimosse: {cleaning_info['rows_removed']}")
    logger.info(f"Colonne rimosse: {cleaning_info['columns_removed']}")
    
    return df, cleaning_info

def transform_target_and_detect_outliers(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.1,
    min_methods: int = 2
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    """
    Trasforma il target e rileva outliers usando multiple metodologie.
    
    Args:
        y_train: Serie target di training
        X_train: DataFrame features di training (per Isolation Forest)
        z_threshold: Soglia per Z-score
        iqr_multiplier: Moltiplicatore per IQR
        contamination: Parametro per Isolation Forest
        min_methods: Numero minimo di metodi che devono identificare un outlier
        
    Returns:
        Tuple con target trasformato, mask outliers, info detector
    """
    logger.info("Trasformazione target e outlier detection...")
    
    # Step 1: Trasformazione logaritmica del target
    original_skew = stats.skew(y_train)
    y_train_log = np.log1p(y_train)
    log_skew = stats.skew(y_train_log)
    
    logger.info(f"Skewness originale: {original_skew:.3f}")
    logger.info(f"Skewness dopo log: {log_skew:.3f}")
    
    # Step 2: Outlier detection con multiple methods
    outlier_votes = np.zeros(len(y_train_log))
    outlier_methods = []
    
    # Metodo 1: Z-Score
    z_scores = stats.zscore(y_train_log)
    z_outliers = np.abs(z_scores) > z_threshold
    outlier_votes[z_outliers] += 1
    outlier_methods.append(('z_score', z_outliers.sum()))
    logger.info(f"Z-Score outliers (soglia {z_threshold}): {z_outliers.sum()}")
    
    # Metodo 2: IQR
    Q1 = y_train_log.quantile(0.25)
    Q3 = y_train_log.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    iqr_outliers = (y_train_log < lower_bound) | (y_train_log > upper_bound)
    outlier_votes[iqr_outliers] += 1
    outlier_methods.append(('iqr', iqr_outliers.sum()))
    logger.info(f"IQR outliers (moltiplicatore {iqr_multiplier}): {iqr_outliers.sum()}")
    
    # Metodo 3: Isolation Forest (se abbastanza features numeriche)
    numeric_features = X_train.select_dtypes(include=[np.number])
    if numeric_features.shape[1] > 0:
        try:
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=1
            )
            iso_outliers_pred = iso_forest.fit_predict(numeric_features)
            iso_outliers = iso_outliers_pred == -1
            outlier_votes[iso_outliers] += 1
            outlier_methods.append(('isolation_forest', iso_outliers.sum()))
            logger.info(f"Isolation Forest outliers (contamination {contamination}): {iso_outliers.sum()}")
        except Exception as e:
            logger.warning(f"Isolation Forest fallito: {e}")
    
    # Step 3: Combina i metodi
    # Un punto è considerato outlier se identificato da almeno min_methods
    outliers_mask = outlier_votes >= min_methods
    total_outliers = outliers_mask.sum()
    
    logger.info(f"Outliers finali (min_methods={min_methods}): {total_outliers}/{len(y_train_log)} "
               f"({total_outliers/len(y_train_log)*100:.2f}%)")
    
    # Informazioni detector
    detector_info = {
        'original_skew': float(original_skew),
        'log_skew': float(log_skew),
        'total_outliers': int(total_outliers),
        'outlier_percentage': float(total_outliers/len(y_train_log)*100),
        'methods_used': outlier_methods,
        'parameters': {
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier,
            'contamination': contamination,
            'min_methods': min_methods
        },
        'target_bounds': {
            'log_min': float(y_train_log.min()),
            'log_max': float(y_train_log.max()),
            'log_mean': float(y_train_log.mean()),
            'log_std': float(y_train_log.std())
        }
    }
    
    return y_train_log, outliers_mask, detector_info

def transform_target_and_detect_outliers_by_category(
    y_train: pd.Series,
    X_train: pd.DataFrame,  # Serve il DataFrame originale, non scaled
    category_column: str = 'AI_IdCategoriaCatastale',  # o 'CC_Id'
    z_threshold: float = 2.5, 
    iqr_multiplier: float = 1.5,
    contamination: float = 0.05, 
    min_methods: int = 2,
    min_samples_per_category: int = 30  # Minimo campioni per outlier detection
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    """
    Trasforma il target e rileva outliers stratificando per categoria catastale.
    
    Args:
        y_train: Serie target di training
        X_train: DataFrame features originali (serve per categoria)
        category_column: Nome colonna categoria catastale
        z_threshold: Soglia per Z-score (per categoria)
        iqr_multiplier: Moltiplicatore per IQR (per categoria)
        contamination: Parametro per Isolation Forest (per categoria)
        min_methods: Numero minimo di metodi che devono identificare un outlier
        min_samples_per_category: Minimo campioni per fare outlier detection
        
    Returns:
        Tuple con target trasformato, mask outliers, info dettagliate
    """
    logger.info("Trasformazione target e outlier detection stratificata per categoria...")
    
    # Verifica presenza colonna categoria
    if category_column not in X_train.columns:
        logger.warning(f"Colonna {category_column} non trovata. Fallback a detection globale.")
        return _fallback_global_detection(y_train, z_threshold, iqr_multiplier, contamination, min_methods)
    
    # Step 1: Trasformazione logaritmica
    original_skew = stats.skew(y_train)
    y_train_log = np.log1p(y_train)
    log_skew = stats.skew(y_train_log)
    
    logger.info(f"Skewness originale: {original_skew:.3f}")
    logger.info(f"Skewness dopo log: {log_skew:.3f}")
    
    # Step 2: Outlier detection per categoria
    categories = X_train[category_column].value_counts()
    logger.info(f"Categorie catastali trovate: {len(categories)}")
    
    # Inizializza mask outliers
    outliers_mask_global = np.zeros(len(y_train_log), dtype=bool)
    category_info = {}
    
    for category, count in categories.items():
        if count < min_samples_per_category:
            logger.warning(f"Categoria {category}: solo {count} campioni, skip outlier detection")
            category_info[category] = {
                'count': count,
                'outliers_detected': 0,
                'outlier_methods': [],
                'skipped': True,
                'reason': f'Meno di {min_samples_per_category} campioni'
            }
            continue
        
        # Seleziona dati per questa categoria
        category_mask = X_train[category_column] == category
        y_category = y_train_log[category_mask]
        
        logger.info(f"\nCategoria {category}: {count} campioni")
        logger.info(f"  Target range: {y_category.min():.3f} - {y_category.max():.3f}")
        logger.info(f"  Target μ±σ: {y_category.mean():.3f} ± {y_category.std():.3f}")
        
        # Applica detection methods per questa categoria
        outliers_methods = []
        
        # Metodo 1: Z-Score (per categoria)
        z_scores = stats.zscore(y_category)
        z_outliers_idx = np.where(category_mask)[0][np.abs(z_scores) > z_threshold]
        if len(z_outliers_idx) > 0:
            outliers_methods.append(('z_score', z_outliers_idx))
            logger.info(f"  Z-Score: {len(z_outliers_idx)} outliers (soglia: {z_threshold})")
        
        # Metodo 2: IQR (per categoria)
        Q1 = y_category.quantile(0.25)
        Q3 = y_category.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        iqr_outliers_local = (y_category < lower_bound) | (y_category > upper_bound)
        iqr_outliers_idx = np.where(category_mask)[0][iqr_outliers_local]
        if len(iqr_outliers_idx) > 0:
            outliers_methods.append(('iqr', iqr_outliers_idx))
            logger.info(f"  IQR: {len(iqr_outliers_idx)} outliers (bounds: {lower_bound:.3f} - {upper_bound:.3f})")
        
        # Metodo 3: Isolation Forest (per categoria, solo se abbastanza campioni)
        if count >= 50:  # Isolation Forest funziona meglio con più dati
            try:
                # Usa features numeriche per questa categoria
                X_category_numeric = X_train[category_mask].select_dtypes(include=[np.number])
                if X_category_numeric.shape[1] > 0:
                    iso_forest = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_jobs=1
                    )
                    iso_outliers_pred = iso_forest.fit_predict(X_category_numeric)
                    iso_outliers_idx = np.where(category_mask)[0][iso_outliers_pred == -1]
                    if len(iso_outliers_idx) > 0:
                        outliers_methods.append(('isolation_forest', iso_outliers_idx))
                        logger.info(f"  Isolation Forest: {len(iso_outliers_idx)} outliers")
            except Exception as e:
                logger.warning(f"  Isolation Forest fallito per categoria {category}: {e}")
        
        # Combine methods per questa categoria
        outlier_votes = np.zeros(len(y_train_log))
        method_names = []
        
        for method_name, outlier_indices in outliers_methods:
            outlier_votes[outlier_indices] += 1
            method_names.append(method_name)
        
        # Considera outlier solo se rilevato da almeno min_methods
        category_outliers_mask = outlier_votes >= min_methods
        category_outliers_count = category_outliers_mask.sum()
        
        # Applica alla mask globale
        outliers_mask_global |= category_outliers_mask
        
        # Salva info categoria
        category_info[category] = {
            'count': count,
            'outliers_detected': category_outliers_count,
            'outlier_percentage': (category_outliers_count / count) * 100,
            'outlier_methods': method_names,
            'mean_target': float(y_category.mean()),
            'std_target': float(y_category.std()),
            'skipped': False
        }
        
        logger.info(f"  Outliers finali: {category_outliers_count}/{count} ({category_outliers_count/count*100:.1f}%)")
    
    # Summary finale
    total_outliers = outliers_mask_global.sum()
    total_samples = len(y_train_log)
    
    logger.info(f"\n=== SUMMARY OUTLIER DETECTION PER CATEGORIA ===")
    logger.info(f"Outliers totali: {total_outliers}/{total_samples} ({total_outliers/total_samples*100:.2f}%)")
    
    # Log dettagliato per categoria
    for category, info in category_info.items():
        if not info['skipped']:
            logger.info(f"  {category}: {info['outliers_detected']}/{info['count']} "
                       f"({info['outlier_percentage']:.1f}%) - "
                       f"methods: {info['outlier_methods']}")
    
    # Prepara info di ritorno
    detection_info = {
        'total_outliers': int(total_outliers),
        'total_samples': int(total_samples),
        'outlier_percentage_global': float(total_outliers/total_samples*100),
        'category_info': category_info,
        'method': 'stratified_by_category',
        'category_column': category_column,
        'parameters': {
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier,
            'contamination': contamination,
            'min_methods': min_methods,
            'min_samples_per_category': min_samples_per_category
        }
    }
    
    return y_train_log, outliers_mask_global, detection_info

def _fallback_global_detection(
    y_train: pd.Series,
    z_threshold: float,
    iqr_multiplier: float,
    contamination: float,
    min_methods: int
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    """Fallback alla detection globale se categoria non disponibile."""
    logger.warning("Usando outlier detection globale come fallback")
    
    # Trasformazione log
    y_train_log = np.log1p(y_train)
    
    # Detection globale semplificata
    outlier_votes = np.zeros(len(y_train_log))
    
    # Z-Score globale
    z_scores = stats.zscore(y_train_log)
    z_outliers = np.abs(z_scores) > z_threshold
    outlier_votes[z_outliers] += 1
    
    # IQR globale
    Q1, Q3 = y_train_log.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    iqr_outliers = (y_train_log < Q1 - iqr_multiplier * IQR) | (y_train_log > Q3 + iqr_multiplier * IQR)
    outlier_votes[iqr_outliers] += 1
    
    # Outlier finale
    outliers_mask = outlier_votes >= min_methods
    
    detection_info = {
        'total_outliers': int(outliers_mask.sum()),
        'total_samples': len(y_train_log),
        'outlier_percentage_global': float(outliers_mask.sum()/len(y_train_log)*100),
        'method': 'global_fallback'
    }
    
    return y_train_log, outliers_mask, detection_info

def analyze_outliers_by_category(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    outliers_mask: np.ndarray,
    category_column: str = 'AI_IdCategoriaCatastale'
) -> pd.DataFrame:
    """
    Analizza gli outliers rilevati per categoria per validation.
    
    Args:
        y_train: Target originale
        X_train: Features originali 
        outliers_mask: Mask degli outliers
        category_column: Colonna categoria
        
    Returns:
        DataFrame con analisi outliers per categoria
    """
    if category_column not in X_train.columns:
        logger.warning(f"Colonna {category_column} non trovata per analisi")
        return pd.DataFrame()
    
    analysis_data = []
    
    for category in X_train[category_column].unique():
        category_mask = X_train[category_column] == category
        category_outliers = outliers_mask & category_mask
        
        total_in_category = category_mask.sum()
        outliers_in_category = category_outliers.sum()
        
        if total_in_category > 0:
            y_category = y_train[category_mask]
            y_category_clean = y_train[category_mask & ~outliers_mask]
            
            analysis_data.append({
                'Categoria': category,
                'Totale_Campioni': total_in_category,
                'Outliers_Rilevati': outliers_in_category,
                'Perc_Outliers': (outliers_in_category / total_in_category) * 100,
                'Target_Mean_All': y_category.mean(),
                'Target_Mean_Clean': y_category_clean.mean() if len(y_category_clean) > 0 else np.nan,
                'Target_Std_All': y_category.std(),
                'Target_Std_Clean': y_category_clean.std() if len(y_category_clean) > 0 else np.nan,
                'Target_Min': y_category.min(),
                'Target_Max': y_category.max()
            })
    
    df_analysis = pd.DataFrame(analysis_data)
    df_analysis = df_analysis.sort_values('Perc_Outliers', ascending=False)
    
    return df_analysis