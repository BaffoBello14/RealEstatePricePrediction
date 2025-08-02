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
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from tabm import TabM
from ..utils.logger import get_logger

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
    """Funzione obiettivo per CatBoost"""
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
    
    model = cb.CatBoostRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs)
    
    return scores.mean()

def objective_lightgbm(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per LightGBM"""
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
    
    model = lgb.LGBMRegressor(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
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

def objective_tabm(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per TabM"""
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
        'd_out': 1,
        'random_state': random_state,
        'n_jobs': n_jobs,
        'verbosity': 0
    }
    
    model = TabM(**params)
    
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
        return cb.CatBoostRegressor(**params)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**params)
    elif model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**params)
    elif model_name == "TabM":
        return TabM(**params)
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