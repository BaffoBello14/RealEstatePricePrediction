"""
Feature Importance Analysis con SHAP e metodologie avanzate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import signal
import time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TimeoutError(Exception):
    """Exception raised when operation times out."""
    pass

def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                return None
            finally:
                # Reset the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

def calculate_basic_feature_importance(best_models: Dict[str, Any], feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola feature importance di base per i migliori modelli.
    
    Args:
        best_models: Dictionary con i migliori modelli
        feature_cols: Lista delle colonne features
        
    Returns:
        Tuple con DataFrame feature importance e summary
    """
    logger.info("Calcolo feature importance di base...")
    
    feature_importance_data = []
    feature_importance_summary = pd.DataFrame(index=feature_cols)
    
    for model_key, model_data in best_models.items():
        model_name = model_data['name']
        model = model_data['model']
        
        try:
            # Diversi metodi per ottenere feature importance
            importance = None
            method = "unknown"
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                method = "feature_importances_"
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                method = "abs(coef_)"
            
            if importance is not None:
                # Normalizza importanze
                importance_norm = importance / importance.sum()
                
                # Aggiungi ai dati
                for i, (feature, imp) in enumerate(zip(feature_cols, importance_norm)):
                    feature_importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': imp,
                        'Rank': i + 1,
                        'Method': method
                    })
                
                # Aggiungi al summary
                feature_importance_summary[model_name] = importance_norm
                
                logger.info(f"✓ Feature importance calcolata per {model_name} ({method})")
                
        except Exception as e:
            logger.error(f"✗ Errore calcolo feature importance per {model_name}: {str(e)}")
    
    # Crea DataFrame completo
    df_feature_importance = pd.DataFrame(feature_importance_data)
    
    # Calcola importanza media
    if not df_feature_importance.empty:
        avg_importance = df_feature_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
        feature_importance_summary['Average'] = avg_importance
    
    return df_feature_importance, feature_importance_summary

@with_timeout(300)  # 5 minutes timeout
def calculate_shap_importance(
    model: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    model_name: str,
    model_type: str = 'auto',
    sample_size: int = 500,  # Reduced from 1000
    background_size: int = 100  # Added separate background size
) -> Dict[str, Any]:
    """
    Calcola SHAP values per un modello con ottimizzazioni di performance.
    
    Args:
        model: Modello addestrato
        X_train: Training features per il background
        X_test: Test features per spiegazioni
        model_name: Nome del modello
        model_type: Tipo di explainer ('tree', 'linear', 'kernel', 'auto')
        sample_size: Numero di campioni per calcolo
        background_size: Numero di campioni background per kernel explainer
        
    Returns:
        Dictionary con SHAP values e informazioni
    """
    logger.info(f"Calcolo SHAP values per {model_name}...")
    
    try:
        # Scegli explainer automaticamente se richiesto
        if model_type == 'auto':
            model_type = _detect_model_type(model)
        
        # Per modelli kernel, usa campionamento più aggressivo
        if model_type == 'kernel':
            # Usa kmeans per background più rappresentativo
            background_size = min(background_size, 50)  # Molto ridotto per kernel
            sample_size = min(sample_size, 200)  # Ridotto per kernel
            
            if len(X_train) > background_size:
                # Usa kmeans per campionamento più intelligente
                try:
                    background_sample = shap.kmeans(X_train, background_size).data
                    background_sample = pd.DataFrame(background_sample, columns=X_train.columns)
                except:
                    # Fallback a campionamento casuale se kmeans fallisce
                    background_sample = X_train.sample(n=background_size, random_state=42)
            else:
                background_sample = X_train
        else:
            # Per altri modelli, usa dimensioni standard
            if len(X_train) > sample_size:
                background_sample = X_train.sample(n=sample_size, random_state=42)
            else:
                background_sample = X_train
            
        if len(X_test) > sample_size:
            test_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            test_sample = X_test
        
        logger.info(f"Usando {len(background_sample)} campioni background e {len(test_sample)} campioni test per {model_name}")
        
        # Crea explainer con parametri ottimizzati
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_sample)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, background_sample)
            shap_values = explainer.shap_values(test_sample)
        elif model_type == 'kernel':
            # Parametri ottimizzati per kernel explainer
            explainer = shap.KernelExplainer(
                model.predict, 
                background_sample,
                link="identity"  # Più veloce per regressione
            )
            # Limita ulteriormente i campioni per kernel explainer
            kernel_test_sample = test_sample.head(min(50, len(test_sample)))
            logger.info(f"Kernel explainer: usando solo {len(kernel_test_sample)} campioni test per velocità")
            shap_values = explainer.shap_values(kernel_test_sample, silent=True)
            test_sample = kernel_test_sample  # Aggiorna il riferimento
        else:
            # Fallback a Permutation explainer con campionamento ridotto
            explainer = shap.Explainer(model.predict, background_sample)
            reduced_test = test_sample.head(min(100, len(test_sample)))
            shap_values = explainer(reduced_test).values
            test_sample = reduced_test
        
        # Se shap_values è 3D (classificazione), prendi la prima classe
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]
        
        # Calcola feature importance da SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_norm = feature_importance / feature_importance.sum()
        
        # Crea DataFrame per facilità d'uso
        shap_df = pd.DataFrame({
            'feature': test_sample.columns,
            'shap_importance': feature_importance_norm
        }).sort_values('shap_importance', ascending=False)
        
        logger.info(f"✓ SHAP values calcolati per {model_name}")
        logger.info(f"Top 5 features SHAP: {list(shap_df.head()['feature'])}")
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance_norm,
            'feature_names': list(test_sample.columns),
            'shap_df': shap_df,
            'model_type': model_type,
            'sample_sizes': {
                'background': len(background_sample),
                'test': len(test_sample)
            }
        }
        
    except Exception as e:
        logger.error(f"✗ Errore calcolo SHAP per {model_name}: {str(e)}")
        return None

def calculate_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calcola permutation importance.
    
    Args:
        model: Modello addestrato
        X_test: Test features
        y_test: Test target
        model_name: Nome del modello
        n_repeats: Numero di ripetizioni
        random_state: Seed per riproducibilità
        
    Returns:
        Dictionary con permutation importance
    """
    logger.info(f"Calcolo permutation importance per {model_name}...")
    
    try:
        # Calcola permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='neg_mean_squared_error'
        )
        
        # Crea DataFrame
        perm_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        logger.info(f"✓ Permutation importance calcolato per {model_name}")
        
        return {
            'importances': perm_importance.importances,
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'feature_names': list(X_test.columns),
            'perm_df': perm_df
        }
        
    except Exception as e:
        logger.error(f"✗ Errore calcolo permutation importance per {model_name}: {str(e)}")
        return None

def compare_importance_methods(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_names: List[str],
    skip_slow_shap: bool = True  # Added option to skip slow SHAP
) -> pd.DataFrame:
    """
    Confronta diversi metodi di feature importance.
    
    Args:
        model: Modello addestrato
        X_train: Training features
        X_test: Test features
        y_test: Test target
        model_name: Nome del modello
        feature_names: Lista nomi features
        
    Returns:
        DataFrame con confronto dei metodi
    """
    logger.info(f"Confronto metodi feature importance per {model_name}...")
    
    comparison_df = pd.DataFrame(index=feature_names)
    
    # 1. Basic importance (se disponibile)
    try:
        if hasattr(model, 'feature_importances_'):
            basic_imp = model.feature_importances_
            comparison_df['basic_importance'] = basic_imp / basic_imp.sum()
        elif hasattr(model, 'coef_'):
            basic_imp = np.abs(model.coef_)
            comparison_df['basic_importance'] = basic_imp / basic_imp.sum()
    except Exception as e:
        logger.warning(f"Basic importance non disponibile: {e}")
    
    # 2. SHAP importance with optimized parameters
    model_type = _detect_model_type(model)
    if skip_slow_shap and model_type == 'kernel' and 'svr' in model.__class__.__name__.lower():
        logger.info(f"Saltando SHAP per {model_name} (troppo lento) - usando solo permutation importance")
        shap_result = None
    else:
        shap_result = calculate_shap_importance(
            model, X_train, X_test, model_name, 
            sample_size=300,  # Reduced for comparison
            background_size=50  # Reduced for comparison
        )
    
    if shap_result:
        comparison_df['shap_importance'] = shap_result['feature_importance']
    
    # 3. Permutation importance
    perm_result = calculate_permutation_importance(model, X_test, y_test, model_name)
    if perm_result:
        comparison_df['permutation_importance'] = perm_result['importances_mean']
        comparison_df['permutation_std'] = perm_result['importances_std']
    
    # Calcola media e rank
    importance_cols = [col for col in comparison_df.columns if 'importance' in col and 'std' not in col]
    if importance_cols:
        comparison_df['average_importance'] = comparison_df[importance_cols].mean(axis=1, skipna=True)
        comparison_df['rank'] = comparison_df['average_importance'].rank(ascending=False)
    
    # Ordina per importanza media
    comparison_df = comparison_df.sort_values('average_importance', ascending=False)
    
    return comparison_df

def create_shap_plots(
    shap_result: Dict[str, Any],
    model_name: str,
    output_dir: str,
    X_test: pd.DataFrame = None,
    max_display: int = 20
) -> None:
    """
    Crea visualizzazioni SHAP.
    
    Args:
        shap_result: Risultato di calculate_shap_importance
        model_name: Nome del modello
        output_dir: Directory di output
        X_test: Test features per alcuni plot
        max_display: Numero massimo di features da mostrare
    """
    if shap_result is None:
        logger.warning(f"Impossibile creare plot SHAP per {model_name}: risultato nullo")
        return
    
    logger.info(f"Creazione plot SHAP per {model_name}...")
    
    try:
        shap_values = shap_result['shap_values']
        feature_names = shap_result['feature_names']
        
        # 1. Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            features=X_test if X_test is not None else None,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=X_test if X_test is not None else None,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_importance_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Plot SHAP salvati per {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Errore creazione plot SHAP per {model_name}: {str(e)}")

def create_comparison_plot(
    comparison_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    top_n: int = 15
) -> None:
    """
    Crea plot di confronto dei metodi di feature importance.
    
    Args:
        comparison_df: DataFrame con confronto metodi
        model_name: Nome del modello
        output_dir: Directory di output
        top_n: Numero di top features da mostrare
    """
    logger.info(f"Creazione plot confronto importance per {model_name}...")
    
    try:
        # Seleziona top features
        top_features_df = comparison_df.head(top_n)
        
        # Colonne di importance (escludi std e rank)
        importance_cols = [col for col in top_features_df.columns 
                          if 'importance' in col and 'std' not in col]
        
        if len(importance_cols) < 2:
            logger.warning(f"Non abbastanza metodi per confronto ({len(importance_cols)})")
            return
        
        # Crea subplot
        fig, axes = plt.subplots(1, len(importance_cols), figsize=(5*len(importance_cols), 8))
        if len(importance_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(importance_cols):
            # Rimuovi valori NaN
            plot_data = top_features_df.dropna(subset=[col])
            
            if len(plot_data) == 0:
                continue
                
            # Bar plot
            axes[i].barh(range(len(plot_data)), plot_data[col])
            axes[i].set_yticks(range(len(plot_data)))
            axes[i].set_yticklabels(plot_data.index, fontsize=8)
            axes[i].set_xlabel('Importance')
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].invert_yaxis()
        
        plt.suptitle(f'Feature Importance Comparison - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/importance_comparison_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Plot confronto salvato per {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Errore creazione plot confronto per {model_name}: {str(e)}")

def _detect_model_type(model) -> str:
    """
    Rileva automaticamente il tipo di modello per SHAP con ottimizzazioni.
    
    Args:
        model: Modello da analizzare
        
    Returns:
        Tipo di modello ('tree', 'linear', 'kernel')
    """
    model_name = model.__class__.__name__.lower()
    
    # Tree-based models
    tree_models = ['randomforest', 'extratrees', 'gradientboosting', 'xgb', 'lgb', 'catboost',
                   'decisiontree', 'ada', 'bagging']
    
    # Linear models  
    linear_models = ['linear', 'ridge', 'lasso', 'elastic', 'logistic', 'sgd']
    
    # SVR and other complex models that should use permutation instead of kernel
    avoid_kernel_models = ['svr', 'svc', 'nusvr', 'nusvc']
    
    for tree_model in tree_models:
        if tree_model in model_name:
            return 'tree'
    
    for linear_model in linear_models:
        if linear_model in model_name:
            return 'linear'
    
    # Per SVR e modelli simili, salta SHAP se troppo lento
    for avoid_model in avoid_kernel_models:
        if avoid_model in model_name:
            logger.warning(f"Modello {model_name} rilevato - usando explainer ottimizzato")
            return 'kernel'
    
    # Default: kernel explainer (più lento ma universale)
    return 'kernel'

def run_comprehensive_feature_analysis(
    best_models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Esegue analisi completa delle feature importance.
    
    Args:
        best_models: Dictionary con i migliori modelli
        X_train: Training features
        X_test: Test features  
        y_test: Test target
        feature_cols: Lista colonne features
        output_dir: Directory di output
        
    Returns:
        Tuple con DataFrame summary e dictionary dettagliato
    """
    logger.info("=" * 60)
    logger.info("ANALISI COMPLETA FEATURE IMPORTANCE")
    logger.info("=" * 60)
    
    # 1. Feature importance di base
    df_basic, summary_basic = calculate_basic_feature_importance(best_models, feature_cols)
    
    # 2. Analisi avanzata per ogni modello
    detailed_results = {}
    comparison_summaries = []
    
    for model_key, model_data in best_models.items():
        model_name = model_data['name']
        model = model_data['model']
        
        logger.info(f"\nAnalisi avanzata per {model_name}...")
        
        # Confronto metodi con ottimizzazioni
        comparison_df = compare_importance_methods(
            model, X_train, X_test, y_test, model_name, feature_cols,
            skip_slow_shap=True  # Skip SHAP for very slow models
        )
        
        # SHAP analysis with optimized parameters
        model_type = _detect_model_type(model)
        if model_type == 'kernel' and 'svr' in model.__class__.__name__.lower():
            logger.info(f"Saltando SHAP separato per {model_name} (già fatto nel confronto)")
            shap_result = comparison_df.get('shap_importance') is not None
            if shap_result:
                # Use the result from comparison instead of recalculating
                shap_result = None  # Skip plotting for SVR
            else:
                shap_result = None
        else:
            shap_result = calculate_shap_importance(
                model, X_train, X_test, model_name,
                sample_size=400,  # Slightly larger for final analysis
                background_size=100
            )
        
        # Salva risultati
        detailed_results[model_name] = {
            'comparison_df': comparison_df,
            'shap_result': shap_result
        }
        
        # Aggiungi al summary
        if not comparison_df.empty:
            comparison_summaries.append(comparison_df[['average_importance']].rename(
                columns={'average_importance': model_name}
            ))
        
        # Crea plot
        if shap_result:
            create_shap_plots(shap_result, model_name, output_dir, X_test)
        
        create_comparison_plot(comparison_df, model_name, output_dir)
    
    # 3. Summary globale
    if comparison_summaries:
        global_summary = pd.concat(comparison_summaries, axis=1).fillna(0)
        global_summary['global_average'] = global_summary.mean(axis=1)
        global_summary = global_summary.sort_values('global_average', ascending=False)
    else:
        global_summary = summary_basic
    
    logger.info("\nTop 15 features (analisi globale):")
    for i, (feature, imp) in enumerate(global_summary.head(15)['global_average'].items(), 1):
        logger.info(f"  {i:2d}. {feature}: {imp:.4f}")
    
    return global_summary, detailed_results