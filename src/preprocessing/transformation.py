import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

def split_dataset_with_validation(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.18,
    random_state: int = 42,
    use_temporal_split: bool = True,
    year_column: str = 'A_AnnoStipula',
    month_column: str = 'A_MeseStipula'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Divide il dataset in train, validation e test.
    
    Args:
        df: DataFrame completo
        target_column: Nome della colonna target
        test_size: Proporzione del test set
        val_size: Proporzione del validation set (rispetto al train+val)
        random_state: Seed per riproducibilità
        use_temporal_split: Se True usa split temporale, altrimenti casuale
        year_column: Nome della colonna anno per split temporale
        month_column: Nome della colonna mese per split temporale
        
    Returns:
        Tuple con X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig
    """
    logger.info(f"Divisione dataset (test_size={test_size}, val_size={val_size}, temporal={use_temporal_split})...")
    
    # Separa features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Recupera target originale se disponibile
    target_orig_column = target_column.replace('_log', '_original')
    if target_orig_column in df.columns:
        y_orig = df[target_orig_column]
    else:
        # Se non disponibile, calcola dalla scala log
        y_orig = np.exp(y)
    
    # Verifica disponibilità colonne temporali
    has_year_month = year_column in df.columns and month_column in df.columns
    
    if use_temporal_split and has_year_month:
        logger.info(f"Usando split temporale basato su colonne: {year_column}, {month_column}")
        # Usa la funzione per ordinamento temporale
        from ..utils.temporal_utils import temporal_sort_by_year_month
        df_sorted = temporal_sort_by_year_month(df, year_column, month_column)
        X_sorted = df_sorted.drop(columns=[target_column])
        y_sorted = df_sorted[target_column]
        y_orig_sorted = df_sorted[target_orig_column] if target_orig_column in df_sorted.columns else np.exp(y_sorted)
        
        # Calcola gli indici di split temporale
        n_samples = len(df_sorted)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split temporale
        X_train = X_sorted.iloc[:val_idx].copy()
        X_val = X_sorted.iloc[val_idx:test_idx].copy()
        X_test = X_sorted.iloc[test_idx:].copy()
        
        y_train = y_sorted.iloc[:val_idx].copy()
        y_val = y_sorted.iloc[val_idx:test_idx].copy()
        y_test = y_sorted.iloc[test_idx:].copy()
        
        y_train_orig = y_orig_sorted.iloc[:val_idx].copy()
        y_val_orig = y_orig_sorted.iloc[val_idx:test_idx].copy()
        y_test_orig = y_orig_sorted.iloc[test_idx:].copy()
        
        # Log delle date di split
        logger.info(f"Split temporale - Range anno/mese:")
        logger.info(f"  Train: {df_sorted[year_column].iloc[0]}/{df_sorted[month_column].iloc[0]} a {df_sorted[year_column].iloc[val_idx-1]}/{df_sorted[month_column].iloc[val_idx-1]}")
        logger.info(f"  Val: {df_sorted[year_column].iloc[val_idx]}/{df_sorted[month_column].iloc[val_idx]} a {df_sorted[year_column].iloc[test_idx-1]}/{df_sorted[month_column].iloc[test_idx-1]}")
        logger.info(f"  Test: {df_sorted[year_column].iloc[test_idx]}/{df_sorted[month_column].iloc[test_idx]} a {df_sorted[year_column].iloc[-1]}/{df_sorted[month_column].iloc[-1]}")
        
    else:
        if use_temporal_split:
            logger.warning(f"Colonne temporali '{year_column}' e '{month_column}' non trovate. Fallback a split casuale.")
        
        logger.info("Usando split casuale")
        
        # Prima divisione: train+val vs test
        X_temp, X_test, y_temp, y_test, y_temp_orig, y_test_orig = train_test_split(
            X, y, y_orig, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Seconda divisione: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Aggiusta la proporzione
        X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = train_test_split(
            X_temp, y_temp, y_temp_orig, test_size=val_size_adjusted, random_state=random_state, stratify=None
        )
    
    logger.info(f"Train set: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    logger.info(f"Validation set: {X_val.shape[0]} righe, {X_val.shape[1]} colonne") 
    logger.info(f"Test set: {X_test.shape[0]} righe, {X_test.shape[1]} colonne")
    
    # Log delle statistiche sui target ORIGINALI (prima della trasformazione log)
    logger.info(f"Target originale - Train: μ={y_train.mean():.3f}, σ={y_train.std():.3f}")
    logger.info(f"Target originale - Val: μ={y_val.mean():.3f}, σ={y_val.std():.3f}")
    logger.info(f"Target originale - Test: μ={y_test.mean():.3f}, σ={y_test.std():.3f}")
    
    # Verifica coerenza distribuzioni tra split
    train_mean, val_mean, test_mean = y_train.mean(), y_val.mean(), y_test.mean()
    max_drift = max(abs(val_mean - train_mean), abs(test_mean - train_mean)) / train_mean
    
    if max_drift > 0.1:  # 10% di drift
        logger.warning(f"⚠️  Drift significativo nella distribuzione target: {max_drift:.3f}")
    else:
        logger.info(f"✓ Distribuzione target coerente tra split (drift: {max_drift:.3f})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig

def split_dataset(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide il dataset in train e test (retrocompatibilità).
    
    Args:
        df: DataFrame completo
        target_column: Nome della colonna target
        test_size: Proporzione del test set
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train, X_test, y_train, y_test
    """
    logger.info(f"Divisione dataset (test_size={test_size})...")
    
    # Separa features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Train set: {X_train.shape[0]} righe, {X_train.shape[1]} colonne")
    logger.info(f"Test set: {X_test.shape[0]} righe, {X_test.shape[1]} colonne")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Tuple:
    """
    Scala le feature usando StandardScaler.
    
    Args:
        X_train: Feature di training
        X_val: Feature di validation (opzionale)
        X_test: Feature di test (opzionale)
        
    Returns:
        Tuple con X_train_scaled, X_val_scaled (se presente), X_test_scaled (se presente), scaler
    """
    logger.info("Scaling delle feature...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    results = [X_train_scaled]
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        results.append(X_val_scaled)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)  # Usa parametri del train
        results.append(X_test_scaled)
    
    results.append(scaler)
    
    logger.info(f"Feature scalate: {X_train_scaled.shape[1]}")
    
    return tuple(results)

def apply_pca(
    X_train_scaled: np.ndarray, 
    X_val_scaled: np.ndarray = None,
    X_test_scaled: np.ndarray = None, 
    X_train_columns: pd.Index = None,
    variance_threshold: float = 0.95,
    random_state: int = 42
) -> Tuple:
    """
    Applica PCA mantenendo la varianza specificata.
    
    Args:
        X_train_scaled: Feature di training scalate
        X_val_scaled: Feature di validation scalate (opzionale)
        X_test_scaled: Feature di test scalate (opzionale)
        X_train_columns: Nomi delle colonne originali
        variance_threshold: Soglia di varianza da mantenere (0-1) o numero di componenti (int)
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train_pca, X_val_pca (se presente), X_test_pca (se presente), pca_model, loadings_df
    """
    logger.info(f"Applicazione PCA (varianza target: {variance_threshold})...")
    
    # Gestisci variance_threshold come int o float
    if isinstance(variance_threshold, int) or variance_threshold > 1.0:
        # Numero specifico di componenti
        n_components = int(variance_threshold)
        logger.info(f"PCA con {n_components} componenti fisse")
    else:
        # Percentuale di varianza da mantenere
        n_components = variance_threshold
        logger.info(f"PCA mantenendo {variance_threshold*100:.1f}% della varianza")
    
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    results = [X_train_pca]
    
    if X_val_scaled is not None:
        X_val_pca = pca.transform(X_val_scaled)
        results.append(X_val_pca)
    
    if X_test_scaled is not None:
        X_test_pca = pca.transform(X_test_scaled)  # Usa parametri del train
        results.append(X_test_pca)
    
    explained_var = pca.explained_variance_ratio_
    total_variance_explained = explained_var.sum()
    
    logger.info(f"PCA: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} componenti")
    logger.info(f"Varianza spiegata totale: {total_variance_explained:.3f} ({total_variance_explained*100:.1f}%)")
    
    # Verifica che la varianza target sia stata raggiunta
    if isinstance(n_components, float) and total_variance_explained < variance_threshold:
        logger.warning(f"Varianza raggiunta ({total_variance_explained:.3f}) < target ({variance_threshold:.3f})")
    
    # Test di ricostruzione per validare la qualità
    X_reconstructed = pca.inverse_transform(X_train_pca)
    reconstruction_error = np.mean((X_train_scaled - X_reconstructed) ** 2)
    logger.info(f"Errore di ricostruzione PCA: {reconstruction_error:.6f}")
    
    # Loadings per interpretabilità
    feature_names = X_train_columns if X_train_columns is not None else [f'Feature_{i}' for i in range(X_train_scaled.shape[1])]
    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    results.extend([pca, loadings])
    
    # Top feature per ogni componente
    logger.info("Top 5 feature per ogni componente principale:")
    for i in range(min(5, pca.n_components_)):  # Mostra solo le prime 5 componenti
        pc_name = f'PC{i+1}'
        top_features = loadings.loc[pc_name].abs().sort_values(ascending=False).head(5)
        logger.info(f"{pc_name} (varianza: {explained_var[i]:.3f}): {dict(top_features)}")
    
    # Grafico varianza cumulativa
    _plot_pca_variance(explained_var, variance_threshold if isinstance(variance_threshold, float) else None)
    
    return tuple(results)

def _plot_pca_variance(explained_var: np.ndarray, variance_threshold: float) -> None:
    """
    Crea grafico della varianza cumulativa PCA.
    
    Args:
        explained_var: Array con varianza spiegata per componente
        variance_threshold: Soglia di varianza
    """
    plt.figure(figsize=(10, 6))
    cumvar = np.cumsum(explained_var)
    plt.plot(range(1, len(cumvar) + 1), cumvar, 'bo-', markersize=4)
    plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                label=f'{variance_threshold*100}% threshold')
    plt.xlabel('Numero di Componenti Principali')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('PCA: Varianza Spiegata Cumulativa')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def apply_feature_scaling(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None
) -> Tuple:
    """
    Applica solo feature scaling senza PCA.
    
    Args:
        X_train: Feature di training
        X_val: Feature di validation (opzionale)
        X_test: Feature di test (opzionale)
        
    Returns:
        Tuple con X_train_scaled, X_val_scaled (se presente), X_test_scaled (se presente), scaler_info
    """
    logger.info("Applicazione feature scaling...")
    
    # Usa la funzione scale_features esistente
    scaling_results = scale_features(X_train, X_val, X_test)
    scaler = scaling_results[-1]  # L'ultimo elemento è sempre il scaler
    
    # Converte array numpy in DataFrame per mantenere i nomi delle colonne
    X_train_scaled = pd.DataFrame(
        scaling_results[0], 
        columns=X_train.columns,
        index=X_train.index
    )
    
    results = [X_train_scaled]
    
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaling_results[1], 
            columns=X_val.columns,
            index=X_val.index
        )
        results.append(X_val_scaled)
    
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaling_results[2], 
            columns=X_test.columns,
            index=X_test.index
        )
        results.append(X_test_scaled)
    
    # Informazioni sul scaling
    scaling_info = {
        'scaler': scaler,
        'original_features': X_train.shape[1],
        'scaled_features': X_train.shape[1],
        'feature_means': dict(zip(X_train.columns, scaler.mean_)),
        'feature_scales': dict(zip(X_train.columns, scaler.scale_))
    }
    
    logger.info(f"Feature scaling completato: {X_train.shape[1]} features scalate")
    
    # Restituisci i dati scalati + info
    return tuple(results) + (scaling_info,)

def apply_pca_transformation(
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame = None,
    X_test_scaled: pd.DataFrame = None,
    variance_threshold: float = 0.95,
    random_state: int = 42
) -> Tuple:
    """
    Applica solo PCA su dati già scalati.
    
    Args:
        X_train_scaled: Feature di training già scalate
        X_val_scaled: Feature di validation già scalate (opzionale)
        X_test_scaled: Feature di test già scalate (opzionale)
        variance_threshold: Soglia varianza per PCA
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train_pca, X_val_pca (se presente), X_test_pca (se presente), pca_info
    """
    logger.info(f"Applicazione PCA (varianza target: {variance_threshold})...")
    
    # Usa la funzione apply_pca esistente
    pca_results = apply_pca(
        X_train_scaled.values,  # Converte a numpy array
        X_val_scaled.values if X_val_scaled is not None else None,
        X_test_scaled.values if X_test_scaled is not None else None,
        X_train_columns=X_train_scaled.columns, 
        variance_threshold=variance_threshold, 
        random_state=random_state
    )
    
    pca_model = pca_results[-2]  # Penultimo elemento è il modello PCA
    loadings = pca_results[-1]   # Ultimo elemento sono i loadings
    
    # Converte i risultati in DataFrame con nomi delle componenti principali
    component_names = [f'PC{i+1}' for i in range(pca_model.n_components_)]
    
    X_train_pca = pd.DataFrame(
        pca_results[0], 
        columns=component_names,
        index=X_train_scaled.index
    )
    
    results = [X_train_pca]
    
    if X_val_scaled is not None:
        X_val_pca = pd.DataFrame(
            pca_results[1], 
            columns=component_names,
            index=X_val_scaled.index
        )
        results.append(X_val_pca)
    
    if X_test_scaled is not None:
        result_idx = 2 if X_val_scaled is not None else 1
        X_test_pca = pd.DataFrame(
            pca_results[result_idx], 
            columns=component_names,
            index=X_test_scaled.index
        )
        results.append(X_test_pca)
    
    # Informazioni sulla PCA
    pca_info = {
        'pca_model': pca_model,
        'loadings': loadings,
        'original_features': X_train_scaled.shape[1],
        'pca_components': pca_model.n_components_,
        'variance_explained': pca_model.explained_variance_ratio_.sum(),
        'variance_explained_per_component': pca_model.explained_variance_ratio_.tolist(),
        'component_names': component_names
    }
    
    logger.info(f"PCA completata: {X_train_scaled.shape[1]} -> {pca_model.n_components_} componenti")
    logger.info(f"Varianza preservata: {pca_info['variance_explained']:.3f}")
    
    # Restituisci i dati trasformati + info
    return tuple(results) + (pca_info,)

def apply_transformations(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    use_pca: bool = True,
    pca_variance_threshold: float = 0.95,
    random_state: int = 42
) -> Tuple:
    """
    Applica tutte le trasformazioni (scaling + PCA opzionale).
    
    Args:
        X_train: Feature di training
        X_val: Feature di validation (opzionale)
        X_test: Feature di test (opzionale)
        use_pca: Se True, applica PCA dopo scaling
        pca_variance_threshold: Soglia varianza per PCA (solo se use_pca=True)
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train_transformed, X_val_transformed (se presente), X_test_transformed (se presente), transformation_info
    """
    logger.info("Applicazione trasformazioni complete...")
    
    # Scaling
    scaling_results = scale_features(X_train, X_val, X_test)
    scaler = scaling_results[-1]  # L'ultimo elemento è sempre il scaler
    
    if use_pca:
        # PCA
        logger.info("Applicazione PCA dopo scaling...")
        pca_results = apply_pca(
            *scaling_results[:-1],  # Passa tutti i set scalati eccetto il scaler
            X_train_columns=X_train.columns, 
            variance_threshold=pca_variance_threshold, 
            random_state=random_state
        )
        
        pca_model = pca_results[-2]  # Penultimo elemento è il modello PCA
        loadings = pca_results[-1]   # Ultimo elemento sono i loadings
        
        # Informazioni sulle trasformazioni
        transformation_info = {
            'scaler': scaler,
            'pca_model': pca_model,
            'loadings': loadings,
            'original_features': X_train.shape[1],
            'transformed_features': pca_results[0].shape[1],  # Prima trasformazione (train)
            'variance_explained': pca_model.explained_variance_ratio_.sum(),
            'use_pca': True
        }
        
        logger.info(f"Trasformazioni completate: {X_train.shape[1]} -> {pca_results[0].shape[1]} feature (con PCA)")
        logger.info(f"Varianza preservata: {transformation_info['variance_explained']:.3f}")
        
        # Restituisci i dati trasformati + info
        return pca_results[:-2] + (transformation_info,)
    else:
        # Solo scaling, senza PCA
        logger.info("Applicazione solo scaling (senza PCA)...")
        
        # Converte array numpy in DataFrame per mantenere i nomi delle colonne
        X_train_scaled = pd.DataFrame(
            scaling_results[0], 
            columns=X_train.columns,
            index=X_train.index
        )
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                scaling_results[1], 
                columns=X_val.columns,
                index=X_val.index
            )
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaling_results[2], 
                columns=X_test.columns,
                index=X_test.index
            )
            results.append(X_test_scaled)
        
        # Informazioni sulle trasformazioni
        transformation_info = {
            'scaler': scaler,
            'pca_model': None,
            'loadings': None,
            'original_features': X_train.shape[1],
            'transformed_features': X_train.shape[1],  # Stesso numero di feature
            'variance_explained': 1.0,  # Nessuna perdita di varianza
            'use_pca': False
        }
        
        logger.info(f"Trasformazioni completate: {X_train.shape[1]} -> {X_train.shape[1]} feature (solo scaling)")
        logger.info("Nessuna perdita di varianza (solo scaling)")
        
        # Restituisci i dati trasformati + info
        return tuple(results) + (transformation_info,)

def inverse_transform_target(y_log: pd.Series) -> pd.Series:
    """
    Inverte la trasformazione logaritmica del target.
    
    Args:
        y_log: Target log-trasformato
        
    Returns:
        Target nella scala originale
    """
    return np.expm1(y_log)  # Inverso di log1p

def transform_target_log(y_train: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Applica trasformazione logaritmica al target.
    
    Args:
        y_train: Serie target di training
        
    Returns:
        Tuple con target trasformato e informazioni sulla trasformazione
    """
    from scipy import stats
    
    logger.info("Applicazione trasformazione logaritmica al target...")
    
    # Calcola skewness originale
    original_skew = stats.skew(y_train)
    
    # Applica trasformazione log1p (log(1+x) per gestire valori zero)
    y_train_log = np.log1p(y_train)
    
    # Calcola skewness dopo trasformazione
    log_skew = stats.skew(y_train_log)
    
    logger.info(f"Skewness originale: {original_skew:.3f}")
    logger.info(f"Skewness dopo log: {log_skew:.3f}")
    logger.info(f"Miglioramento skewness: {abs(original_skew) - abs(log_skew):.3f}")
    
    # Informazioni sulla trasformazione
    transform_info = {
        'original_skew': float(original_skew),
        'log_skew': float(log_skew),
        'skew_improvement': float(abs(original_skew) - abs(log_skew)),
        'target_bounds': {
            'original_min': float(y_train.min()),
            'original_max': float(y_train.max()),
            'original_mean': float(y_train.mean()),
            'original_std': float(y_train.std()),
            'log_min': float(y_train_log.min()),
            'log_max': float(y_train_log.max()),
            'log_mean': float(y_train_log.mean()),
            'log_std': float(y_train_log.std())
        }
    }
    
    return y_train_log, transform_info