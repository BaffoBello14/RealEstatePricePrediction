import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any, List
from ..utils.logger import get_logger
from .data_cleaning_core import (
    convert_to_numeric_unified,
    clean_dataframe_unified,
    remove_constant_columns_unified
)
from .target_processing_core import (
    transform_target_log,
    detect_outliers_univariate,
    detect_outliers_multivariate
)

logger = get_logger(__name__)

def remove_specific_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rimuove colonne specifiche dal DataFrame.
    
    Args:
        df: DataFrame da cui rimuovere le colonne
        columns_to_remove: Lista di nomi di colonne da rimuovere
        
    Returns:
        Tuple con DataFrame pulito e informazioni sulle colonne rimosse
    """
    logger.info(f"Rimozione colonne specifiche: {columns_to_remove}")
    
    # Trova colonne che esistono effettivamente nel DataFrame
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    removal_info = {
        'requested_columns': columns_to_remove,
        'existing_columns': existing_columns,
        'missing_columns': missing_columns,
        'columns_removed_count': len(existing_columns)
    }
    
    if existing_columns:
        df = df.drop(columns=existing_columns)
        logger.info(f"Rimosse {len(existing_columns)} colonne: {existing_columns}")
    else:
        logger.info("Nessuna colonna specifica da rimuovere trovata nel DataFrame")
    
    if missing_columns:
        logger.warning(f"Colonne richieste ma non trovate: {missing_columns}")
    
    return df, removal_info

def remove_constant_columns(df: pd.DataFrame, target_column: str, threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rimuove colonne che sono quasi costanti (sopra una certa soglia di valori uguali).
    
    Args:
        df: DataFrame da cui rimuovere le colonne costanti
        target_column: Nome della colonna target (da non rimuovere)
        threshold: Soglia per considerare una colonna costante (% di valori uguali)
        
    Returns:
        Tuple con DataFrame pulito e informazioni sulle colonne rimosse
    """
    logger.info(f"Rimozione colonne costanti (soglia: {threshold})...")
    
    constant_columns = []
    column_stats = {}
    
    for col in df.columns:
        if col == target_column:
            continue
            
        # Calcola la percentuale del valore più frequente
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if len(value_counts) > 0:
            max_frequency = value_counts.iloc[0]
            column_stats[col] = {
                'max_frequency': max_frequency,
                'unique_values': len(value_counts),
                'most_common_value': value_counts.index[0]
            }
            
            if max_frequency >= threshold:
                constant_columns.append(col)
    
    removal_info = {
        'threshold_used': threshold,
        'constant_columns': constant_columns,
        'columns_removed_count': len(constant_columns),
        'column_stats': column_stats
    }
    
    if constant_columns:
        df = df.drop(columns=constant_columns)
        logger.info(f"Rimosse {len(constant_columns)} colonne costanti: {constant_columns}")
        
        # Log delle statistiche per le colonne rimosse
        for col in constant_columns:
            stats = column_stats[col]
            logger.info(f"  {col}: {stats['max_frequency']:.3f} frequenza, "
                       f"{stats['unique_values']} valori unici, "
                       f"valore dominante: {stats['most_common_value']}")
    else:
        logger.info("Nessuna colonna costante trovata")
    
    return df, removal_info



def clean_data(df: pd.DataFrame, target_column: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    RISTRUTTURATA: Pulisce i dati del dataset utilizzando funzioni unificate modulari.
    
    Args:
        df: DataFrame da pulire
        target_column: Nome della colonna target
        config: Configurazione del preprocessing
        
    Returns:
        Tuple con DataFrame pulito e informazioni dettagliate sul cleaning
        
    Note:
        Ora utilizza il sistema modulare unificato per eliminare duplicazioni.
        Mantiene compatibilità con l'interfaccia esistente.
    """
    logger.info("🧹 Avvio pulizia dati ristrutturata...")
    
    steps_config = config.get('steps', {})
    cleaning_info = {'config_used': config, 'steps_performed': []}
    
    # STEP 1: Pulizia base unificata
    logger.info("🔧 Step 1: Pulizia base unificata...")
    df_clean, base_cleaning_info = clean_dataframe_unified(
        df=df,
        target_column=target_column,
        remove_empty_strings=True,
        remove_duplicates=True,
        remove_empty_columns=True,
        remove_target_nulls=True
    )
    cleaning_info['base_cleaning'] = base_cleaning_info
    cleaning_info['steps_performed'].append('base_cleaning_unified')
    
    # STEP 2: Rimozione colonne specifiche (se abilitato)
    if steps_config.get('enable_specific_columns_removal', True):
        columns_to_remove = config.get('columns_to_remove', [])
        if columns_to_remove:
            logger.info(f"🗑️  Step 2: Rimozione colonne specifiche: {columns_to_remove}")
            df_clean, specific_removal_info = remove_specific_columns(df_clean, columns_to_remove)
            cleaning_info['specific_columns_removal'] = specific_removal_info
            cleaning_info['steps_performed'].append('specific_columns_removal')
        else:
            logger.info("📋 Step 2: Nessuna colonna specifica da rimuovere")
    else:
        logger.info("⏭️  Step 2: Rimozione colonne specifiche DISABILITATA")
    
    # STEP 3: Rimozione colonne costanti (se abilitato)
    if steps_config.get('enable_constant_columns_removal', True):
        constant_threshold = config.get('constant_column_threshold', 0.95)
        logger.info(f"📊 Step 3: Rimozione colonne quasi-costanti (soglia: {constant_threshold:.1%})")
        df_clean, constant_removal_info = remove_constant_columns_unified(
            df=df_clean,
            target_column=target_column,
            threshold=constant_threshold
        )
        cleaning_info['constant_columns_removal'] = constant_removal_info
        cleaning_info['steps_performed'].append('constant_columns_removal')
    else:
        logger.info("⏭️  Step 3: Rimozione colonne costanti DISABILITATA")
    
    # Statistiche finali per compatibilità con codice esistente
    original_shape = cleaning_info['base_cleaning']['original_shape']
    final_shape = list(df_clean.shape)
    
    cleaning_info.update({
        'original_shape': original_shape,
        'final_shape': final_shape,
        'rows_removed': original_shape[0] - final_shape[0],
        'columns_removed': original_shape[1] - final_shape[1]
    })
    
    logger.info(f"🏁 Pulizia completata: {original_shape} → {final_shape}")
    logger.info(f"📊 Totale riduzione: {cleaning_info['rows_removed']} righe, {cleaning_info['columns_removed']} colonne")
    logger.info(f"✨ Step eseguiti: {cleaning_info['steps_performed']}")
    
    return df_clean, cleaning_info

def transform_target_and_detect_outliers(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.1,
    min_methods: int = 2
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    """
    DEPRECATA: Trasforma il target e rileva outliers usando multiple metodologie.
    
    ⚠️  PROBLEMA ARCHITETTURALE: Questa funzione fa due cose diverse insieme (trasformazione + outlier detection).
    
    ALTERNATIVA RACCOMANDATA:
    - Usa transform_target_log() dal modulo target_processing_core per trasformazione
    - Usa detect_outliers_univariate() o detect_outliers_multivariate() per outlier detection
    
    Args:
        y_train: Serie target di training
        X_train: DataFrame features di training (per Isolation Forest)
        z_threshold: Soglia per Z-score
        iqr_multiplier: Moltiplicatore per IQR
        contamination: Parametro per Isolation Forest
        min_methods: Numero minimo di metodi che devono identificare un outlier
        
    Returns:
        Tuple con target trasformato, mask outliers, info detector
        
    Note:
        Mantenuta per compatibilità con codice esistente, ma ora usa internamente
        le funzioni separate e più chiare del modulo target_processing_core.
    """
    logger.warning("⚠️  transform_target_and_detect_outliers è DEPRECATA, usa funzioni separate dal modulo target_processing_core")
    
    # RISTRUTTURAZIONE: Usa le nuove funzioni separate per chiarezza
    # Step 1: Trasformazione logaritmica del target
    y_train_log, transform_info = transform_target_log(y_train, method='log1p')
    
    # Step 2: Outlier detection usando le nuove funzioni modulari
    outliers_mask, outlier_info = detect_outliers_univariate(
        y_train_log,
        methods=['zscore', 'iqr'],
        z_threshold=z_threshold,
        iqr_multiplier=iqr_multiplier,
        min_methods_agreement=min_methods
    )
    
    # Combina le informazioni per compatibilità con l'API esistente
    # (adatta ai nuovi formati delle funzioni separate)
    detector_info = {
        'transform_info': transform_info,
        'outlier_info': outlier_info,
        # Compatibilità legacy (potrebbe essere rimossa in futuro)
        'original_skew': transform_info['original_stats']['skewness'],
        'log_skew': transform_info['transformed_stats']['skewness'] if transform_info['applied'] else transform_info['original_stats']['skewness'],
        'total_outliers': outlier_info['final_outliers']['count'],
        'outlier_percentage': outlier_info['final_outliers']['percentage'],
        'methods_used': [(method, result['outliers_count']) for method, result in outlier_info['method_results'].items()],
        'parameters': outlier_info['parameters']
    }
    
    return y_train_log, outliers_mask, detector_info

def detect_outliers_multimethod(
    y_target: pd.Series,
    X_features: pd.DataFrame = None,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.1,
    min_methods: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Rileva outliers usando multiple metodologie.
    
    Args:
        y_target: Serie target su cui applicare outlier detection
        X_features: DataFrame features (per Isolation Forest), opzionale
        z_threshold: Soglia per Z-score
        iqr_multiplier: Moltiplicatore per IQR
        contamination: Parametro per Isolation Forest
        min_methods: Numero minimo di metodi che devono identificare un outlier
        
    Returns:
        Tuple con mask outliers e informazioni sui metodi usati
    """
    logger.info("Outlier detection con metodologie multiple...")
    
    # Inizializza conteggio votes per ogni metodo
    outlier_votes = np.zeros(len(y_target))
    outlier_methods = []
    
    # Metodo 1: Z-Score
    z_scores = stats.zscore(y_target)
    z_outliers = np.abs(z_scores) > z_threshold
    outlier_votes[z_outliers] += 1
    outlier_methods.append(('z_score', z_outliers.sum()))
    logger.info(f"Z-Score outliers (soglia {z_threshold}): {z_outliers.sum()}")
    
    # Metodo 2: IQR
    Q1 = y_target.quantile(0.25)
    Q3 = y_target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    iqr_outliers = (y_target < lower_bound) | (y_target > upper_bound)
    outlier_votes[iqr_outliers] += 1
    outlier_methods.append(('iqr', iqr_outliers.sum()))
    logger.info(f"IQR outliers (moltiplicatore {iqr_multiplier}): {iqr_outliers.sum()}")
    
    # Metodo 3: Isolation Forest (se features fornite e numeriche disponibili)
    if X_features is not None:
        numeric_features = X_features.select_dtypes(include=[np.number])
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
        else:
            logger.warning("Nessuna feature numerica disponibile per Isolation Forest")
    else:
        logger.info("Features non fornite, skip Isolation Forest")
    
    # Combina i metodi: un punto è outlier se identificato da almeno min_methods
    outliers_mask = outlier_votes >= min_methods
    total_outliers = outliers_mask.sum()
    
    logger.info(f"Outliers finali (min_methods={min_methods}): {total_outliers}/{len(y_target)} "
               f"({total_outliers/len(y_target)*100:.2f}%)")
    
    # Informazioni sui metodi usati
    outlier_info = {
        'total_outliers': int(total_outliers),
        'outlier_percentage': float(total_outliers/len(y_target)*100),
        'methods_used': outlier_methods,
        'parameters': {
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier,
            'contamination': contamination,
            'min_methods': min_methods
        },
        'target_stats': {
            'min': float(y_target.min()),
            'max': float(y_target.max()),
            'mean': float(y_target.mean()),
            'std': float(y_target.std()),
            'q1': float(Q1),
            'q3': float(Q3),
            'iqr': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    }
    
    return outliers_mask, outlier_info

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