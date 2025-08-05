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

def load_data_for_model_type(preprocessing_paths: Dict[str, str], model_name: str) -> Dict[str, Any]:
    """
    Carica il dataset appropriato per il tipo di modello.
    
    Args:
        preprocessing_paths: Path ai file preprocessati
        model_name: Nome del modello per determinare quale dataset caricare
        
    Returns:
        Dictionary con i dati caricati appropriati per il modello
    """
    # Modelli che supportano feature categoriche native
    categorical_models = ['CatBoost', 'LightGBM']
    
    if model_name in categorical_models:
        # Usa dataset con feature categoriche
        logger.info(f"Caricando dati con feature categoriche per {model_name}")
        
        # Verifica che i file categorici esistano
        categorical_paths = ['train_features_categorical', 'val_features_categorical', 'test_features_categorical']
        if all(path in preprocessing_paths for path in categorical_paths):
            X_train = load_dataframe(preprocessing_paths['train_features_categorical'])
            X_val = load_dataframe(preprocessing_paths['val_features_categorical'])
            X_test = load_dataframe(preprocessing_paths['test_features_categorical'])
            logger.info(f"Dataset categorico caricato per {model_name}: X_train {X_train.shape}")
        else:
            logger.warning(f"File categorici non trovati per {model_name}, uso dataset encoded")
            X_train = load_dataframe(preprocessing_paths['train_features'])
            X_val = load_dataframe(preprocessing_paths['val_features'])
            X_test = load_dataframe(preprocessing_paths['test_features'])
    else:
        # Usa dataset encoded per tutti gli altri modelli
        logger.debug(f"Caricando dati encoded per {model_name}")
        X_train = load_dataframe(preprocessing_paths['train_features'])
        X_val = load_dataframe(preprocessing_paths['val_features'])
        X_test = load_dataframe(preprocessing_paths['test_features'])
    
    # Target è sempre lo stesso per tutti i modelli
    y_train = load_dataframe(preprocessing_paths['train_target']).squeeze()
    y_val = load_dataframe(preprocessing_paths['val_target']).squeeze()
    y_test = load_dataframe(preprocessing_paths['test_target']).squeeze()
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

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
    
    # Ottieni metrica unificata dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error')
    
    for name, model in baseline_models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            
            baseline_results[name] = {
                'cv_score_mean': scores.mean(),  # Score naturale della metrica
                'cv_score_std': scores.std(),
                'model': model
            }
            
            logger.info(f"{name}: {scores.mean():.6f} ± {scores.std():.6f}")
            
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
        # Verifiche di sanità sui dati prima del training
        logger.debug(f"Verifiche pre-training per {model_name}...")
        
        # Verifica y_train
        if not np.isfinite(y_train).all():
            n_invalid_train = np.sum(~np.isfinite(y_train))
            logger.warning(f"⚠️  {model_name}: {n_invalid_train} valori non validi in y_train")
        
        # Verifica y_val
        if not np.isfinite(y_val).all():
            n_invalid_val = np.sum(~np.isfinite(y_val))
            logger.warning(f"⚠️  {model_name}: {n_invalid_val} valori non validi in y_val")
        
        # Verifica X_train
        if not np.isfinite(X_train.values if hasattr(X_train, 'values') else X_train).all():
            n_invalid_X_train = np.sum(~np.isfinite(X_train.values if hasattr(X_train, 'values') else X_train))
            logger.warning(f"⚠️  {model_name}: {n_invalid_X_train} valori non validi in X_train")
        
        # Verifica scale del target
        y_train_std = np.std(y_train)
        y_train_range = np.ptp(y_train)
        y_val_std = np.std(y_val)
        y_val_range = np.ptp(y_val)
        
        logger.debug(f"{model_name} - Target stats:")
        logger.debug(f"  Train: mean={np.mean(y_train):.2e}, std={y_train_std:.2e}, range={y_train_range:.2e}")
        logger.debug(f"  Val:   mean={np.mean(y_val):.2e}, std={y_val_std:.2e}, range={y_val_range:.2e}")
        
        if y_train_range > 1e6:
            logger.warning(f"⚠️  {model_name}: Range del target training molto ampio ({y_train_range:.0f}). Considerare normalizzazione.")
        if y_train_std > 1e4:
            logger.warning(f"⚠️  {model_name}: Deviazione standard del target training molto alta ({y_train_std:.0f}). Considerare normalizzazione.")
        
        # Confronto train vs validation
        scale_diff = abs(y_train_std - y_val_std) / max(y_train_std, 1e-10)
        if scale_diff > 0.5:
            logger.warning(f"⚠️  {model_name}: Grande differenza di scala tra train ({y_train_std:.2e}) e val ({y_val_std:.2e})")
            
        mean_diff = abs(np.mean(y_train) - np.mean(y_val)) / max(abs(np.mean(y_train)), 1e-10)
        if mean_diff > 0.3:
            logger.warning(f"⚠️  {model_name}: Grande differenza di media tra train ({np.mean(y_train):.2e}) e val ({np.mean(y_val):.2e})")
        
        # Training
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predizioni
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Verifica predizioni
        if not np.isfinite(y_pred_train).all():
            n_invalid_pred_train = np.sum(~np.isfinite(y_pred_train))
            logger.warning(f"⚠️  {model_name}: {n_invalid_pred_train} predizioni non valide su training set")
            
        if not np.isfinite(y_pred_val).all():
            n_invalid_pred_val = np.sum(~np.isfinite(y_pred_val))
            logger.warning(f"⚠️  {model_name}: {n_invalid_pred_val} predizioni non valide su validation set")
        
        # Metriche con diagnostics dettagliati
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        # Calcolo R² con diagnostics
        try:
            train_r2 = r2_score(y_train, y_pred_train)
        except Exception as e:
            logger.error(f"Errore nel calcolo R² training per {model_name}: {e}")
            train_r2 = float('-inf')
            
        try:
            val_r2 = r2_score(y_val, y_pred_val)
            
            # Diagnostics dettagliati se R² è estremamente negativo
            if val_r2 < -1000:
                logger.warning(f"🔍 DIAGNOSTICS R² per {model_name}:")
                logger.warning(f"   Val R²: {val_r2:.2e}")
                logger.warning(f"   Val RMSE: {val_rmse:.2f}")
                
                # Calcolo manuale dei componenti R²
                y_val_mean = np.mean(y_val)
                ss_res = np.sum((y_val - y_pred_val) ** 2)
                ss_tot = np.sum((y_val - y_val_mean) ** 2)
                
                logger.warning(f"   Target mean: {y_val_mean:.2e}")
                logger.warning(f"   Target std: {np.std(y_val):.2e}")
                logger.warning(f"   Target range: {np.ptp(y_val):.2e}")
                logger.warning(f"   SS_res: {ss_res:.2e}")
                logger.warning(f"   SS_tot: {ss_tot:.2e}")
                logger.warning(f"   Pred mean: {np.mean(y_pred_val):.2e}")
                logger.warning(f"   Pred std: {np.std(y_pred_val):.2e}")
                logger.warning(f"   Pred range: {np.ptp(y_pred_val):.2e}")
                
                # Verifiche aggiuntive
                logger.warning(f"   Target type: {type(y_val)}, shape: {getattr(y_val, 'shape', 'No shape')}")
                logger.warning(f"   Pred type: {type(y_pred_val)}, shape: {getattr(y_pred_val, 'shape', 'No shape')}")
                
                # Check per allineamento dati
                if hasattr(y_val, 'index') and hasattr(y_pred_val, 'index'):
                    if not y_val.index.equals(y_pred_val.index):
                        logger.error(f"⚠️ INDICI NON ALLINEATI tra y_val e y_pred_val!")
                
                # Esempi di valori
                logger.warning(f"   Primi 5 target: {y_val.iloc[:5].tolist() if hasattr(y_val, 'iloc') else list(y_val[:5])}")
                logger.warning(f"   Primi 5 pred:   {y_pred_val[:5].tolist() if hasattr(y_pred_val, 'tolist') else list(y_pred_val[:5])}")
                
                # Calcolo R² con sklearn per verifica
                try:
                    r2_manual = 1 - (ss_res / ss_tot)
                    logger.warning(f"   R² manual: {r2_manual:.2e}")
                except:
                    logger.warning(f"   R² manual: ERROR")
                
                if ss_tot < 1e-10:
                    logger.error(f"⚠️ SS_tot troppo piccolo ({ss_tot:.2e}) - target quasi costante!")
                
                # Check per valori estremi nelle predizioni
                extreme_preds = np.abs(y_pred_val) > 1e6
                if np.any(extreme_preds):
                    n_extreme = np.sum(extreme_preds)
                    logger.error(f"⚠️ {n_extreme} predizioni estreme (>1M) che potrebbero causare overflow!")
            
        except Exception as e:
            logger.error(f"Errore nel calcolo R² validation per {model_name}: {e}")
            val_r2 = float('-inf')
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
    Valuta tutti i modelli (baseline, ottimizzati, ensemble) - versione legacy.
    
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
    logger.info("TRAINING E VALUTAZIONE SU VALIDATION SET (legacy)")
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

def evaluate_all_models_with_appropriate_data(preprocessing_paths: Dict[str, str], config: Dict[str, Any],
                                           baseline_results: Dict[str, Any], optimized_models: Dict[str, Any], 
                                           ensemble_models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valuta tutti i modelli usando i dataset appropriati per ogni tipo.
    
    Args:
        preprocessing_paths: Path ai file preprocessati
        config: Configurazione del training
        baseline_results: Risultati modelli baseline
        optimized_models: Modelli ottimizzati
        ensemble_models: Modelli ensemble
        
    Returns:
        Dictionary con tutti i risultati
    """
    logger.info("=" * 60)
    logger.info("TRAINING E VALUTAZIONE CON DATASET APPROPRIATI")
    logger.info("=" * 60)
    
    all_results = {}
    categorical_models = ['CatBoost', 'LightGBM']
    
    # 1. Modelli baseline (usano sempre dati encoded)
    logger.info("Valutazione modelli baseline con dati encoded...")
    data_encoded = load_data_for_model_type(preprocessing_paths, "StandardModel")
    
    for name, baseline_data in baseline_results.items():
        if 'model' in baseline_data:
            results = evaluate_single_model(
                baseline_data['model'], 
                data_encoded['X_train'], data_encoded['y_train'], 
                data_encoded['X_val'], data_encoded['y_val'], 
                f"Baseline_{name}"
            )
            all_results[f"Baseline_{name}"] = results
    
    # 2. Modelli ottimizzati (usa dataset appropriato per ogni modello)
    logger.info("Valutazione modelli ottimizzati...")
    data_categorical = None  # Carica solo se necessario
    
    for name, model in optimized_models.items():
        # Determina se il modello supporta categoriche
        model_supports_categorical = any(cat_model in name for cat_model in categorical_models)
        
        if model_supports_categorical:
            # Carica dati categorici se non già fatto
            if data_categorical is None:
                logger.info("Caricamento dati categorici per modelli che li supportano...")
                data_categorical = load_data_for_model_type(preprocessing_paths, "CatBoost")
            
            results = evaluate_single_model(
                model, 
                data_categorical['X_train'], data_categorical['y_train'], 
                data_categorical['X_val'], data_categorical['y_val'], 
                f"Optimized_{name}"
            )
            logger.info(f"✓ {name} valutato con feature categoriche")
        else:
            # Usa dati encoded
            results = evaluate_single_model(
                model, 
                data_encoded['X_train'], data_encoded['y_train'], 
                data_encoded['X_val'], data_encoded['y_val'], 
                f"Optimized_{name}"
            )
            logger.debug(f"✓ {name} valutato con feature encoded")
        
        all_results[f"Optimized_{name}"] = results
    
    # 3. Modelli ensemble (usano sempre dati encoded perché combinano modelli diversi)
    logger.info("Valutazione modelli ensemble con dati encoded...")
    for name, model in ensemble_models.items():
        results = evaluate_single_model(
            model, 
            data_encoded['X_train'], data_encoded['y_train'], 
            data_encoded['X_val'], data_encoded['y_val'], 
            f"Ensemble_{name}"
        )
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
    Esegue la pipeline completa di training con supporto per dataset categorici.
    
    Args:
        preprocessing_paths: Path ai file preprocessati
        config: Configurazione del training
        
    Returns:
        Dictionary con tutti i risultati del training
    """
    logger.info("=== AVVIO PIPELINE TRAINING CON SUPPORTO CATEGORICHE ===")
    
    try:
        # 1. Caricamento dati encoded (per modelli baseline e non-categorici)
        logger.info("Caricamento dati encoded per modelli standard...")
        data_encoded = load_data_for_model_type(preprocessing_paths, "StandardModel")
        logger.info(f"Dati encoded - Train: {data_encoded['X_train'].shape}, Val: {data_encoded['X_val'].shape}")
        
        # 2. Valutazione modelli baseline (usano sempre dati encoded)
        baseline_results = evaluate_baseline_models(data_encoded['X_train'], data_encoded['y_train'], config)
        
        # 3. Ottimizzazione iperparametri per modelli standard
        logger.info("Ottimizzazione modelli standard con dati encoded...")
        optimization_results_standard = run_full_optimization(data_encoded['X_train'], data_encoded['y_train'], config)
        
        # 4. Ottimizzazione per modelli che supportano categoriche
        logger.info("Ottimizzazione modelli categorici...")
        optimization_results_categorical = {}
        
        # Lista modelli che supportano categoriche dal config
        categorical_models = ['CatBoost', 'LightGBM']
        enabled_categorical = []
        
        for model in categorical_models:
            model_key = model.lower().replace('boost', 'boost').replace('gbm', 'gbm')
            if config.get('models', {}).get('advanced', {}).get(model_key, False):
                enabled_categorical.append(model)
        
        if enabled_categorical:
            logger.info(f"Ottimizzazione modelli categorici abilitati: {enabled_categorical}")
            
            # Carica dati categorici
            data_categorical = load_data_for_model_type(preprocessing_paths, "CatBoost")  # Usa CatBoost come rappresentante
            
            # Esegui ottimizzazione per ogni modello categorico
            from .tuning import optimize_model
            for model_name in enabled_categorical:
                if model_name in ['CatBoost', 'LightGBM']:
                    logger.info(f"Ottimizzazione {model_name} con feature categoriche...")
                    try:
                        result = optimize_model(model_name, data_categorical['X_train'], data_categorical['y_train'], config)
                        optimization_results_categorical[model_name] = result
                        logger.info(f"✓ {model_name} ottimizzato con score: {result.get('best_score', 'N/A')}")
                    except Exception as e:
                        logger.error(f"✗ Errore ottimizzazione {model_name}: {str(e)}")
        
        # 5. Combina risultati ottimizzazione
        optimization_results = {**optimization_results_standard, **optimization_results_categorical}
        
        # 6. Creazione modelli ottimizzati
        optimized_models = create_optimized_models(optimization_results)
        
        # 7. Creazione modelli ensemble
        ensemble_models = create_ensemble_models(
            optimized_models, 
            config.get('random_state', 42),
            config.get('n_jobs', -1)
        )
        
        # 8. Valutazione completa - usa dati appropriati per ogni modello
        validation_results = evaluate_all_models_with_appropriate_data(
            preprocessing_paths, config,
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
                'train': data_encoded['X_train'].shape,
                'val': data_encoded['X_val'].shape,
                'test': data_encoded['X_test'].shape
            }
        }
        
        logger.info("=== PIPELINE TRAINING COMPLETATA ===")
        return training_results
        
    except Exception as e:
        logger.error(f"Errore nella pipeline di training: {e}")
        raise