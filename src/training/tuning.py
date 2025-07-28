"""
Ottimizzazione iperparametri con Optuna.
"""

import numpy as np
import optuna
import optunahub
from datetime import datetime
from typing import Dict, Any, Callable
from optuna.pruners import MedianPruner
from ..utils.logger import get_logger
from .models import (
    objective_random_forest, objective_gradient_boosting, objective_xgboost,
    objective_catboost, objective_lightgbm, objective_hist_gradient_boosting
)

logger = get_logger(__name__)

def create_optimization_objective(objective_func: Callable, X_train, y_train, config: Dict[str, Any]) -> Callable:
    """
    Crea una funzione obiettivo wrapper per Optuna.
    
    Args:
        objective_func: Funzione obiettivo specifica del modello
        X_train: Features di training
        y_train: Target di training
        config: Configurazione del training
        
    Returns:
        Funzione obiettivo wrapper
    """
    def objective(trial):
        return objective_func(
            trial, X_train, y_train,
            cv_folds=config.get('cv_folds', 5),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
    return objective

def optimize_model(objective_func: Callable, model_name: str, X_train, y_train, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ottimizza un singolo modello usando Optuna.
    
    Args:
        objective_func: Funzione obiettivo del modello
        model_name: Nome del modello
        X_train: Features di training
        y_train: Target di training
        config: Configurazione del training
        
    Returns:
        Dictionary con risultati dell'ottimizzazione
    """
    logger.info(f"Ottimizzazione {model_name}...")
    
    # Configurazione Optuna
    n_trials = config.get('n_trials', 100)
    timeout = config.get('optuna_timeout', 7200)
    
    try:
        # Configurazione sampler e pruner
        module = optunahub.load_module(package="samplers/auto_sampler")
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=module.AutoSampler(),
            pruner=pruner,
            study_name=f"{model_name}_optimization"
        )
        
        # Crea objective wrapper
        objective = create_optimization_objective(objective_func, X_train, y_train, config)
        
        # Ottimizzazione
        start_time = datetime.now()
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        end_time = datetime.now()
        
        optimization_time = (end_time - start_time).total_seconds()
        
        logger.info(f"{model_name} - Migliori parametri: {study.best_params}")
        logger.info(f"{model_name} - Miglior punteggio: {study.best_value:.6f}")
        logger.info(f"{model_name} - Trials completati: {len(study.trials)}")
        logger.info(f"{model_name} - Tempo ottimizzazione: {optimization_time:.2f} secondi")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        logger.error(f"✗ Errore nell'ottimizzazione di {model_name}: {str(e)}")
        return {
            'error': str(e),
            'best_params': None,
            'best_score': None
        }

def run_full_optimization(X_train, y_train, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue l'ottimizzazione completa per tutti i modelli.
    
    Args:
        X_train: Features di training
        y_train: Target di training
        config: Configurazione del training
        
    Returns:
        Dictionary con risultati di tutti i modelli
    """
    logger.info("=" * 50)
    logger.info("INIZIO OTTIMIZZAZIONE COMPLETA")
    logger.info("=" * 50)
    
    optimization_results = {}
    
    # Definizione modelli da ottimizzare
    models_to_optimize = [
        (objective_random_forest, "RandomForest"),
        (objective_gradient_boosting, "GradientBoosting"),
        (objective_xgboost, "XGBoost"),
        (objective_catboost, "CatBoost"),
        (objective_lightgbm, "LightGBM"),
        (objective_hist_gradient_boosting, "HistGradientBoosting")
    ]
    
    total_start_time = datetime.now()
    
    for objective_func, model_name in models_to_optimize:
        try:
            result = optimize_model(objective_func, model_name, X_train, y_train, config)
            optimization_results[model_name] = result
            
            if result.get('best_score') is not None:
                logger.info(f"✓ {model_name} ottimizzato con successo")
            else:
                logger.error(f"✗ {model_name} fallito")
                
        except Exception as e:
            logger.error(f"✗ Errore nell'ottimizzazione di {model_name}: {str(e)}")
            optimization_results[model_name] = {
                'error': str(e),
                'best_params': None,
                'best_score': None
            }
    
    total_end_time = datetime.now()
    total_time = (total_end_time - total_start_time).total_seconds()
    
    logger.info("=" * 50)
    logger.info("RIEPILOGO OTTIMIZZAZIONE")
    logger.info("=" * 50)
    logger.info(f"Tempo totale: {total_time:.2f} secondi ({total_time/60:.2f} minuti)")
    
    # Classifica modelli per performance
    successful_models = {k: v for k, v in optimization_results.items() if v.get('best_score') is not None}
    if successful_models:
        sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['best_score'])
        logger.info("\nClassifica modelli (migliore al peggiore):")
        for i, (model_name, results) in enumerate(sorted_models, 1):
            logger.info(f"{i}. {model_name}: {results['best_score']:.6f}")
    
    return optimization_results