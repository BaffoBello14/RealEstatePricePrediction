"""
Feature Importance Analysis con SHAP e metodologie avanzate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from ..utils.logger import get_logger

logger = get_logger(__name__)

def _create_feature_aware_predictor(model, feature_names: List[str]):
    """
    Crea un wrapper per il modello che gestisce correttamente i feature names.
    
    Args:
        model: Modello da wrappare
        feature_names: Lista dei nomi delle feature
        
    Returns:
        Funzione wrapper per le predizioni
    """
    def predict_wrapper(X):
        # Converti in DataFrame se necessario per mantenere feature names
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
        
        # Sopprimi i warning sui feature names durante SHAP
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            return model.predict(X_df)
    
    return predict_wrapper

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
            elif 'Ensemble' in model_name or 'Stacking' in model_name:
                # Usa il metodo veloce per ensemble models
                importance = get_ensemble_feature_importance_fast(model, feature_cols, model_name)
                method = "ensemble_fast"
            
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

def calculate_shap_importance(
    model: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    model_name: str,
    model_type: str = 'auto',
    sample_size: int = 100  # Ridotto da 1000 per velocizzare
) -> Dict[str, Any]:
    """
    Calcola SHAP values per un modello con ottimizzazioni per velocità.
    
    Args:
        model: Modello addestrato
        X_train: Training features per il background
        X_test: Test features per spiegazioni
        model_name: Nome del modello
        model_type: Tipo di explainer ('tree', 'linear', 'kernel', 'auto')
        sample_size: Numero di campioni per calcolo (default ridotto per velocità)
        
    Returns:
        Dictionary con SHAP values e informazioni
    """
    logger.info(f"Calcolo SHAP values per {model_name}...")
    
    try:
        # Campiona i dati aggressivamente per velocizzare
        # Per ensemble models, usa campioni molto più piccoli
        if 'Ensemble' in model_name or 'Stacking' in model_name:
            effective_sample_size = min(sample_size // 2, 50)  # Max 50 campioni per ensemble
            test_sample_size = min(20, len(X_test))  # Max 20 predizioni per ensemble
        else:
            effective_sample_size = sample_size
            test_sample_size = min(sample_size, len(X_test))
        
        logger.info(f"Usando {effective_sample_size} campioni di background e {test_sample_size} campioni di test per {model_name}")
        
        # Campiona i dati
        if len(X_train) > effective_sample_size:
            train_sample = X_train.sample(n=effective_sample_size, random_state=42)
        else:
            train_sample = X_train
            
        if len(X_test) > test_sample_size:
            test_sample = X_test.sample(n=test_sample_size, random_state=42)
        else:
            test_sample = X_test
        
        # Scegli explainer automaticamente se richiesto
        if model_type == 'auto':
            model_type = _detect_model_type(model)
        
        # Crea wrapper per gestire feature names e sopprimere warning
        feature_names = list(X_test.columns)
        predict_wrapper = _create_feature_aware_predictor(model, feature_names)
        
        # Crea explainer con gestione ottimizzata
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            warnings.filterwarnings("ignore", message=".*background data samples.*")
            
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(test_sample)
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, train_sample)
                shap_values = explainer.shap_values(test_sample)
            elif model_type == 'kernel':
                # Usa il wrapper per evitare warning sui feature names
                explainer = shap.KernelExplainer(predict_wrapper, train_sample.values)
                shap_values = explainer.shap_values(test_sample.values)
            else:
                # Fallback a Permutation explainer
                explainer = shap.Explainer(predict_wrapper, train_sample.values)
                shap_values = explainer(test_sample.values).values
        
        # Se shap_values è 3D (classificazione), prendi la prima classe
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]
        
        # Calcola feature importance da SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_importance_norm = feature_importance / feature_importance.sum()
        
        # Crea DataFrame per facilità d'uso
        shap_df = pd.DataFrame({
            'feature': X_test.columns,
            'shap_importance': feature_importance_norm
        }).sort_values('shap_importance', ascending=False)
        
        logger.info(f"✓ SHAP values calcolati per {model_name}")
        logger.info(f"Top 5 features SHAP: {list(shap_df.head()['feature'])}")
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance_norm,
            'feature_names': list(X_test.columns),
            'shap_df': shap_df,
            'model_type': model_type,
            'sample_sizes': {
                'train': len(train_sample),
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
    random_state: int = 42,
    scoring: str = 'neg_mean_squared_error'
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
            scoring=scoring
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
    scoring: str = 'neg_mean_squared_error'
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
    
    # 2. SHAP importance
    shap_result = calculate_shap_importance(model, X_train, X_test, model_name)
    if shap_result:
        comparison_df['shap_importance'] = shap_result['feature_importance']
    
    # 3. Permutation importance
    perm_result = calculate_permutation_importance(model, X_test, y_test, model_name, scoring=scoring)
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
    Rileva automaticamente il tipo di modello per SHAP.
    
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
    
    for tree_model in tree_models:
        if tree_model in model_name:
            return 'tree'
    
    for linear_model in linear_models:
        if linear_model in model_name:
            return 'linear'
    
    # Default: kernel explainer (più lento ma universale)
    return 'kernel'

def run_comprehensive_feature_analysis(
    best_models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    output_dir: str,
    scoring: str = 'neg_mean_squared_error'
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
        
        # Confronto metodi
        comparison_df = compare_importance_methods(
            model, X_train, X_test, y_test, model_name, feature_cols, scoring=scoring
        )
        
        # SHAP analysis - skip per ensemble models molto lenti
        if 'Ensemble_StackingRegressor' in model_name:
            logger.warning(f"⚠️  Skipping SHAP analysis per {model_name} - troppo computazionalmente costoso")
            logger.info(f"💡 Per calcolare SHAP per {model_name}, usa un campione più piccolo o esegui separatamente")
            shap_result = None
        else:
            shap_result = calculate_shap_importance(model, X_train, X_test, model_name)
        
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

def get_ensemble_feature_importance_fast(
    model, 
    feature_names: List[str], 
    model_name: str
) -> Optional[np.ndarray]:
    """
    Calcola feature importance veloce per ensemble models senza SHAP.
    
    Args:
        model: Modello ensemble
        feature_names: Lista nomi delle feature
        model_name: Nome del modello
        
    Returns:
        Array con feature importance o None se non disponibile
    """
    try:
        # Per StackingRegressor, prova a ottenere importance dai base estimators
        if hasattr(model, 'estimators_') and model.estimators_:
            logger.info(f"Calcolo feature importance media dai base estimators per {model_name}...")
            
            all_importances = []
            for i, (name, estimator) in enumerate(model.estimators_):
                if hasattr(estimator, 'feature_importances_'):
                    importance = estimator.feature_importances_
                    all_importances.append(importance)
                    logger.info(f"  ✓ {name}: importance estratta")
                elif hasattr(estimator, 'coef_'):
                    importance = np.abs(estimator.coef_)
                    all_importances.append(importance)
                    logger.info(f"  ✓ {name}: coefficienti estratti")
                else:
                    logger.info(f"  ⚠️ {name}: no feature importance disponibile")
            
            if all_importances:
                # Media delle importanze dai base estimators
                mean_importance = np.mean(all_importances, axis=0)
                logger.info(f"✓ Feature importance media calcolata da {len(all_importances)} base estimators")
                return mean_importance
            
        # Fallback: prova metodi standard
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
            
        logger.warning(f"⚠️ Nessun metodo disponibile per feature importance in {model_name}")
        return None
        
    except Exception as e:
        logger.error(f"✗ Errore calcolo feature importance veloce per {model_name}: {str(e)}")
        return None