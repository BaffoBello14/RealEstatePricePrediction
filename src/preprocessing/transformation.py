import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

def split_dataset(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide il dataset in train e test.
    
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

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scala le feature usando StandardScaler.
    
    Args:
        X_train: Feature di training
        X_test: Feature di test
        
    Returns:
        Tuple con X_train_scaled, X_test_scaled, scaler
    """
    logger.info("Scaling delle feature...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Usa parametri del train
    
    logger.info(f"Feature scalate: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(
    X_train_scaled: np.ndarray, 
    X_test_scaled: np.ndarray, 
    X_train_columns: pd.Index,
    variance_threshold: float = 0.95,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, PCA, pd.DataFrame]:
    """
    Applica PCA mantenendo la varianza specificata.
    
    Args:
        X_train_scaled: Feature di training scalate
        X_test_scaled: Feature di test scalate
        X_train_columns: Nomi delle colonne originali
        variance_threshold: Soglia di varianza da mantenere
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train_pca, X_test_pca, pca_model, loadings_df
    """
    logger.info(f"Applicazione PCA (varianza target: {variance_threshold})...")
    
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)  # Usa parametri del train
    
    explained_var = pca.explained_variance_ratio_
    
    logger.info(f"PCA: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} componenti")
    logger.info(f"Varianza spiegata totale: {explained_var.sum():.3f} ({explained_var.sum()*100:.1f}%)")
    
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
    
    # Top feature per ogni componente
    logger.info("Top 5 feature per ogni componente principale:")
    for i in range(min(5, pca.n_components_)):  # Mostra solo le prime 5 componenti
        pc_name = f'PC{i+1}'
        top_features = loadings.loc[pc_name].abs().sort_values(ascending=False).head(5)
        logger.info(f"{pc_name} (varianza: {explained_var[i]:.3f}): {dict(top_features)}")
    
    # Grafico varianza cumulativa
    _plot_pca_variance(explained_var, variance_threshold)
    
    return X_train_pca, X_test_pca, pca, loadings

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

def apply_transformations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pca_variance_threshold: float = 0.95,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applica tutte le trasformazioni (scaling + PCA).
    
    Args:
        X_train: Feature di training
        X_test: Feature di test
        pca_variance_threshold: Soglia varianza per PCA
        random_state: Seed per riproducibilità
        
    Returns:
        Tuple con X_train_transformed, X_test_transformed, transformation_info
    """
    logger.info("Applicazione trasformazioni complete...")
    
    # Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # PCA
    X_train_pca, X_test_pca, pca_model, loadings = apply_pca(
        X_train_scaled, X_test_scaled, X_train.columns, pca_variance_threshold, random_state
    )
    
    # Informazioni sulle trasformazioni
    transformation_info = {
        'scaler': scaler,
        'pca_model': pca_model,
        'loadings': loadings,
        'original_features': X_train.shape[1],
        'pca_components': X_train_pca.shape[1],
        'variance_explained': pca_model.explained_variance_ratio_.sum()
    }
    
    logger.info(f"Trasformazioni completate: {X_train.shape[1]} -> {X_train_pca.shape[1]} feature")
    logger.info(f"Varianza preservata: {transformation_info['variance_explained']:.3f}")
    
    return X_train_pca, X_test_pca, transformation_info

def inverse_transform_target(y_log: pd.Series) -> pd.Series:
    """
    Inverte la trasformazione logaritmica del target.
    
    Args:
        y_log: Target log-trasformato
        
    Returns:
        Target nella scala originale
    """
    return np.expm1(y_log)  # Inverso di log1p