"""
Definizione modelli e funzioni obiettivo per Optuna.

NOTA IMPORTANTE SULLA DIREZIONE DI OTTIMIZZAZIONE:
Le funzioni obiettivo restituiscono sempre scores.mean() perché:
- Per 'neg_root_mean_squared_error': direction='maximize' (valori più vicini a 0 sono migliori)
- Per 'r2': direction='maximize' (valori più alti sono migliori)
- La direzione di ottimizzazione è gestita nella configurazione di Optuna, non nelle funzioni obiettivo.
"""

from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from ..utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)



def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Restituisce i modelli baseline per confronto.
    
    Args:
        random_state: Seed per riproducibilità
        
    Returns:
        Dictionary con modelli baseline
    """
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=random_state),
        'Lasso': Lasso(random_state=random_state),
        'ElasticNet': ElasticNet(random_state=random_state),
        'DecisionTree': DecisionTreeRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(),
        'SVR': SVR()
    }

def objective_random_forest(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': random_state,
        'n_jobs': n_jobs
    }
    
    model = RandomForestRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()

def objective_gradient_boosting(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per Gradient Boosting"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': random_state
    }
    
    model = GradientBoostingRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()

def objective_xgboost(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbosity': 0,
        'eval_metric': 'rmse'
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()

def objective_catboost(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per CatBoost con supporto feature categoriche"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'early_stopping_rounds': 50,  # Early stopping per evitare overfitting
        'random_seed': random_state,
        'logging_level': 'Silent',
        'thread_count': n_jobs if n_jobs > 0 else None,
        'train_dir': None,  # Disabilita la creazione della directory catboost_info
        'allow_writing_files': False  # Previene la scrittura di file di log
    }
    
    # Parametri aggiuntivi per bootstrap_type
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
    
    # Identifica feature categoriche automaticamente
    categorical_features = []
    if hasattr(X_train, 'dtypes'):
        categorical_features = list(X_train.select_dtypes(include=['object', 'category']).columns)
        if categorical_features:
            logger.info(f"CatBoost: {len(categorical_features)} feature categoriche identificate: {categorical_features[:5]}...")
            # Converti indices se necessario
            if isinstance(categorical_features[0], str):
                categorical_features = [X_train.columns.get_loc(col) for col in categorical_features]
    
    model = cb.CatBoostRegressor(cat_features=categorical_features, **params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()

def objective_lightgbm(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per LightGBM con supporto feature categoriche"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbose': -1
    }
    
    # Identifica feature categoriche per LightGBM
    categorical_features = 'auto'  # LightGBM può rilevare automaticamente
    if hasattr(X_train, 'dtypes'):
        categorical_columns = list(X_train.select_dtypes(include=['object', 'category']).columns)
        if categorical_columns:
            logger.info(f"LightGBM: {len(categorical_columns)} feature categoriche identificate: {categorical_columns[:5]}...")
            # Per LightGBM possiamo passare i nomi delle colonne
            categorical_features = categorical_columns
    
    model = lgb.LGBMRegressor(categorical_feature=categorical_features, **params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    
    # Per LightGBM con feature categoriche, non possiamo usare cross_val_score
    # perché deve preparare i dati categorici. Usiamo una CV manuale.
    if categorical_features != 'auto' and categorical_features:
        logger.info("Usando CV manuale per LightGBM con feature categoriche")
        scores = []
        for train_idx, val_idx in cv_strategy.split(X_train):
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # Calcola score manualmente in base alla metrica
            if scoring == 'neg_root_mean_squared_error':
                from sklearn.metrics import mean_squared_error
                score = -np.sqrt(mean_squared_error(y_val_fold, y_pred))
            elif scoring == 'r2':
                from sklearn.metrics import r2_score
                score = r2_score(y_val_fold, y_pred)
            else:
                # Default a RMSE
                from sklearn.metrics import mean_squared_error
                score = -np.sqrt(mean_squared_error(y_val_fold, y_pred))
                
            scores.append(score)
        
        return np.mean(scores)
    else:
        # Usa cross_val_score standard se non ci sono categoriche
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
        return scores.mean()

def objective_hist_gradient_boosting(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per Histogram-based Gradient Boosting"""
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 0, 10),
        'max_bins': trial.suggest_int('max_bins', 32, 255),
        'random_state': random_state
    }
    
    model = HistGradientBoostingRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()



def create_model_from_params(model_name: str, params: Dict[str, Any]) -> Any:
    """
    Crea un modello dalle migliori params di Optuna.
    
    Args:
        model_name: Nome del modello
        params: Parametri ottimizzati
        
    Returns:
        Modello istanziato
    """
    if model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**params)
    elif model_name == "CatBoost":
        # Se params non contiene cat_features, lo aggiungiamo vuoto
        # Sarà configurato durante il training con i dati appropriati
        if 'cat_features' not in params:
            params['cat_features'] = []
        return cb.CatBoostRegressor(**params)
    elif model_name == "LightGBM":
        # Se params non contiene categorical_feature, lo aggiungiamo come auto
        if 'categorical_feature' not in params:
            params['categorical_feature'] = 'auto'
        return lgb.LGBMRegressor(**params)
    elif model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

def create_ensemble_models(optimized_models: Dict[str, Any], random_state: int = 42, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Crea modelli ensemble (Voting e Stacking).
    
    Args:
        optimized_models: Dictionary con modelli ottimizzati
        random_state: Seed per riproducibilità
        n_jobs: Numero di job paralleli
        
    Returns:
        Dictionary con modelli ensemble
    """
    ensemble_models = {}
    
    if len(optimized_models) < 2:
        logger.warning("Meno di 2 modelli ottimizzati disponibili per ensemble")
        return ensemble_models
    
    # Seleziona i migliori 5 modelli per ensemble
    ensemble_estimators = [(name, model) for name, model in list(optimized_models.items())[:5]]
    
    if len(ensemble_estimators) >= 2:
        # Voting Regressor
        try:
            voting_regressor = VotingRegressor(
                estimators=ensemble_estimators,
                n_jobs=n_jobs
            )
            ensemble_models['VotingRegressor'] = voting_regressor
            logger.info(f"✓ VotingRegressor creato con {len(ensemble_estimators)} modelli")
        except Exception as e:
            logger.error(f"✗ Errore nella creazione VotingRegressor: {str(e)}")
        
        # Stacking Regressor
        try:
            stacking_regressor = StackingRegressor(
                estimators=ensemble_estimators,
                final_estimator=Ridge(random_state=random_state),
                cv=5,
                n_jobs=n_jobs
            )
            ensemble_models['StackingRegressor'] = stacking_regressor
            logger.info(f"✓ StackingRegressor creato con {len(ensemble_estimators)} modelli")
        except Exception as e:
            logger.error(f"✗ Errore nella creazione StackingRegressor: {str(e)}")
    
    return ensemble_models