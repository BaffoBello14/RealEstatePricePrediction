"""
Training e valutazione modelli.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import get_logger
from ..utils.io import load_dataframe
from .models import get_baseline_models, create_model_from_params, create_ensemble_models
from .tuning import run_full_optimization

logger = get_logger(__name__)

def evaluate_baseline_models(X_train, y_train, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valuta modelli baseline con cross-validation.
    
    Args:
        X_train: Features di training
        y_train: Target di training
        config: Configurazione del training
        
    Returns:
        Dictionary con risultati dei modelli baseline
    """
    logger.info("Valutazione modelli baseline...")
    
    baseline_models = get_baseline_models(config.get('random_state', 42))
    baseline_results = {}
    
    cv_folds = config.get('cv_folds', 5)
    random_state = config.get('random_state', 42)
    n_jobs = config.get('n_jobs', -1)
    use_time_series_cv = config.get('use_time_series_cv', False)
    
    # Scegli il tipo di cross-validation
    if use_time_series_cv:
        logger.info("Usando TimeSeriesSplit per dati temporali")
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        logger.info("Usando KFold standard")
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, model in baseline_models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=n_jobs)
            
            baseline_results[name] = {
                'cv_score_mean': -scores.mean(),
                'cv_score_std': scores.std(),
                'model': model
            }
            
            logger.info(f"{name}: {-scores.mean():.6f} ± {scores.std():.6f}")
            
        except Exception as e:
            logger.error(f"Errore con {name}: {str(e)}")
            baseline_results[name] = {'error': str(e)}
    
    return baseline_results

def evaluate_single_model(model, X_train, y_train, X_val, y_val, model_name: str) -> Dict[str, Any]:
    """
    Valuta un singolo modello sul validation set.
    
    Args:
        model: Modello da valutare
        X_train: Features di training
        y_train: Target di training
        X_val: Features di validation
        y_val: Target di validation
        model_name: Nome del modello
        
    Returns:
        Dictionary con metriche di valutazione
    """
    try:
        # Training
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predizioni
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Metriche
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # MAPE (evita divisione per zero)
        train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
        val_mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
        
        # Calcola overfit ratio e controlli di qualità
        overfit_ratio = val_rmse / train_rmse if train_rmse > 0 else float('inf')
        
        # Warning per overfitting severo
        if overfit_ratio > 2.0:
            logger.warning(f"⚠️  {model_name}: Possibile overfitting severo (ratio: {overfit_ratio:.3f})")
        elif overfit_ratio > 1.5:
            logger.warning(f"⚠️  {model_name}: Overfitting moderato (ratio: {overfit_ratio:.3f})")
        
        # Warning per performance molto bassa
        if val_r2 < -0.5:
            logger.warning(f"⚠️  {model_name}: R² molto basso ({val_r2:.4f}) - controllare preprocessing")
        
        results = {
            'model_name': model_name,
            'model': model,
            'training_time': training_time,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mape': train_mape,
            'val_mape': val_mape,
            'overfit_ratio': overfit_ratio,
            'predictions_train': y_pred_train,
            'predictions_val': y_pred_val
        }
        
        logger.info(f"✓ {model_name} - Val RMSE: {val_rmse:.6f}, Val R²: {val_r2:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"✗ Errore nella valutazione di {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'error': str(e),
            'model': None
        }

def create_optimized_models(optimization_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea modelli ottimizzati dai risultati Optuna.
    
    Args:
        optimization_results: Risultati dell'ottimizzazione
        
    Returns:
        Dictionary con modelli ottimizzati
    """
    logger.info("Creazione modelli ottimizzati...")
    
    optimized_models = {}
    
    for model_name, results in optimization_results.items():
        if results.get('best_params') is not None:
            try:
                model = create_model_from_params(model_name, results['best_params'])
                optimized_models[model_name] = model
                logger.info(f"✓ {model_name} ottimizzato creato")
                
            except Exception as e:
                logger.error(f"✗ Errore nella creazione di {model_name}: {str(e)}")
    
    return optimized_models

def evaluate_all_models(X_train, y_train, X_val, y_val, baseline_results: Dict[str, Any], 
                       optimized_models: Dict[str, Any], ensemble_models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valuta tutti i modelli (baseline, ottimizzati, ensemble).
    
    Args:
        X_train: Features di training
        y_train: Target di training
        X_val: Features di validation
        y_val: Target di validation
        baseline_results: Risultati modelli baseline
        optimized_models: Modelli ottimizzati
        ensemble_models: Modelli ensemble
        
    Returns:
        Dictionary con tutti i risultati
    """
    logger.info("=" * 60)
    logger.info("TRAINING E VALUTAZIONE SU VALIDATION SET")
    logger.info("=" * 60)
    
    all_results = {}
    
    # 1. Modelli baseline
    logger.info("Valutazione modelli baseline...")
    for name, baseline_data in baseline_results.items():
        if 'model' in baseline_data:
            results = evaluate_single_model(
                baseline_data['model'], X_train, y_train, X_val, y_val, f"Baseline_{name}"
            )
            all_results[f"Baseline_{name}"] = results
    
    # 2. Modelli ottimizzati
    logger.info("Valutazione modelli ottimizzati...")
    for name, model in optimized_models.items():
        results = evaluate_single_model(model, X_train, y_train, X_val, y_val, f"Optimized_{name}")
        all_results[f"Optimized_{name}"] = results
    
    # 3. Modelli ensemble
    logger.info("Valutazione modelli ensemble...")
    for name, model in ensemble_models.items():
        results = evaluate_single_model(model, X_train, y_train, X_val, y_val, f"Ensemble_{name}")
        all_results[f"Ensemble_{name}"] = results
    
    return all_results

def select_best_models(validation_results: Dict[str, Any], config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analizza i risultati della validazione e seleziona i migliori modelli.
    
    Args:
        validation_results: Risultati della validazione
        
    Returns:
        Tuple con DataFrame dei risultati e migliori modelli
    """
    logger.info("=" * 60)
    logger.info("ANALISI RISULTATI VALIDAZIONE")
    logger.info("=" * 60)
    
    # Filtra modelli con errori
    valid_results = {k: v for k, v in validation_results.items() if 'error' not in v}
    
    if not valid_results:
        logger.error("Nessun modello valutato con successo!")
        return pd.DataFrame(), {}
    
    # Crea DataFrame per analisi
    results_data = []
    for model_name, results in valid_results.items():
        results_data.append({
            'Model': model_name,
            'Train_RMSE': results['train_rmse'],
            'Val_RMSE': results['val_rmse'],
            'Train_R2': results['train_r2'],
            'Val_R2': results['val_r2'],
            'Train_MAE': results['train_mae'],
            'Val_MAE': results['val_mae'],
            'Train_MAPE': results['train_mape'],
            'Val_MAPE': results['val_mape'],
            'Overfit_Ratio': results['overfit_ratio'],
            'Training_Time': results['training_time']
        })
    
    df_results = pd.DataFrame(results_data)
    
    # Ottieni metrica di ranking dal config
    ranking_metric = config.get('ranking_metric', 'Val_RMSE') if config else 'Val_RMSE'
    ranking_ascending = config.get('ranking_ascending', True) if config else True
    
    # Ordina dinamicamente
    df_results = df_results.sort_values(ranking_metric, ascending=ranking_ascending).reset_index(drop=True)
    
    logger.info(f"Top 10 modelli per {ranking_metric}:")
    print("\n" + "="*100)
    print(f"{'Rank':<4} {'Model':<30} {'Val_RMSE':<10} {'Val_R²':<8} {'Val_MAE':<9} {'Overfit':<7} {'Time(s)':<8}")
    print("="*100)
    
    for i, row in df_results.head(10).iterrows():
        print(f"{i+1:<4} {row['Model']:<30} {row['Val_RMSE']:<10.6f} {row['Val_R2']:<8.4f} "
              f"{row['Val_MAE']:<9.6f} {row['Overfit_Ratio']:<7.3f} {row['Training_Time']:<8.2f}")
    
    # Selezione migliori modelli per categoria
    best_models = {}
    
    # Miglior modello overall
    best_overall = df_results.iloc[0]['Model']
    best_models['best_overall'] = {
        'name': best_overall,
        'model': valid_results[best_overall]['model'],
        'results': valid_results[best_overall]
    }
    
    # Migliori per categoria
    categories = ['Baseline', 'Optimized', 'Ensemble']
    for category in categories:
        category_models = df_results[df_results['Model'].str.contains(category)]
        if not category_models.empty:
            best_in_category = category_models.iloc[0]['Model']
            best_models[f'best_{category.lower()}'] = {
                'name': best_in_category,
                'model': valid_results[best_in_category]['model'],
                'results': valid_results[best_in_category]
            }
    
    # Modelli con miglior bilancio performance/overfitting
    df_balanced = df_results[df_results['Overfit_Ratio'] <= 1.2]  # Max 20% overfitting
    if not df_balanced.empty:
        best_balanced = df_balanced.iloc[0]['Model']
        best_models['best_balanced'] = {
            'name': best_balanced,
            'model': valid_results[best_balanced]['model'],
            'results': valid_results[best_balanced]
        }
    
    logger.info(f"\nModelli selezionati: {list(best_models.keys())}")
    
    return df_results, best_models

def run_training_pipeline(preprocessing_paths: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue la pipeline completa di training.
    
    Args:
        preprocessing_paths: Path ai file preprocessati
        config: Configurazione del training
        
    Returns:
        Dictionary con tutti i risultati del training
    """
    logger.info("=== AVVIO PIPELINE TRAINING ===")
    
    try:
        # 1. Caricamento dati
        logger.info("Caricamento dati preprocessati...")
        X_train = load_dataframe(preprocessing_paths['train_features'])
        X_val = load_dataframe(preprocessing_paths['val_features'])
        X_test = load_dataframe(preprocessing_paths['test_features'])
        y_train = load_dataframe(preprocessing_paths['train_target']).squeeze()
        y_val = load_dataframe(preprocessing_paths['val_target']).squeeze()
        y_test = load_dataframe(preprocessing_paths['test_target']).squeeze()
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # 2. Valutazione modelli baseline
        baseline_results = evaluate_baseline_models(X_train, y_train, config)
        
        # 3. Ottimizzazione iperparametri
        optimization_results = run_full_optimization(X_train, y_train, config)
        
        # 4. Creazione modelli ottimizzati
        optimized_models = create_optimized_models(optimization_results)
        
        # 5. Creazione modelli ensemble
        ensemble_models = create_ensemble_models(
            optimized_models, 
            config.get('random_state', 42),
            config.get('n_jobs', -1)
        )
        
        # 6. Valutazione completa
        validation_results = evaluate_all_models(
            X_train, y_train, X_val, y_val,
            baseline_results, optimized_models, ensemble_models
        )
        
        # 7. Selezione migliori modelli
        df_validation_results, best_models = select_best_models(validation_results, config)
        
        # 8. Preparazione risultati finali
        training_results = {
            'baseline_results': baseline_results,
            'optimization_results': optimization_results,
            'optimized_models': optimized_models,
            'ensemble_models': ensemble_models,
            'validation_results': validation_results,
            'df_validation_results': df_validation_results,
            'best_models': best_models,
            'data_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        
        logger.info("=== PIPELINE TRAINING COMPLETATA ===")
        return training_results
        
    except Exception as e:
        logger.error(f"Errore nella pipeline di training: {e}")
        raise