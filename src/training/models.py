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
from tabm import TabM
from ..utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

class TabMWrapper:
    """
    Wrapper per TabM che gestisce automaticamente il preprocessing delle feature categoriche.
    TabM non supporta nativamente le feature categoriche, quindi le converte automaticamente.
    """
    
    def __init__(self, **params):
        """
        Inizializza TabM wrapper.
        
        Args:
            **params: Parametri per TabM
        """
        self.params = params
        self.model = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.is_fitted = False
        
    def _preprocess_data(self, X, fit_encoders=False):
        """
        Preprocessa i dati convertendo le feature categoriche in numeriche.
        
        Args:
            X: Dataset da preprocessare
            fit_encoders: Se True, fit degli encoder (fase di training)
            
        Returns:
            Dataset preprocessato
        """
        if not hasattr(X, 'dtypes'):
            return X
            
        X_processed = X.copy()
        categorical_columns = list(X.select_dtypes(include=['object', 'category']).columns)
        
        if fit_encoders:
            self.categorical_columns = categorical_columns
            self.label_encoders = {}
            
            for col in categorical_columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
                
            if categorical_columns:
                logger.info(f"TabM: Fitted label encoders per {len(categorical_columns)} feature categoriche")
        else:
            # Usa encoder già fittati
            for col in self.categorical_columns:
                if col in X_processed.columns:
                    le = self.label_encoders[col]
                    # Gestisci valori non visti durante il training
                    try:
                        X_processed[col] = le.transform(X_processed[col].astype(str))
                    except ValueError:
                        # Se ci sono valori non visti, usa un valore di default
                        X_processed[col] = X_processed[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                        
        return X_processed
    
    def fit(self, X, y, **fit_params):
        """
        Adatta il modello ai dati.
        
        Args:
            X: Feature
            y: Target
            **fit_params: Parametri aggiuntivi per fit
        """
        # Preprocessa i dati
        X_processed = self._preprocess_data(X, fit_encoders=True)
        
        # Aggiorna parametri con numero effettivo di feature
        final_params = self.params.copy()
        final_params['n_num_features'] = X_processed.shape[1]
        final_params.setdefault('cat_cardinalities', [])
        
        # Crea e adatta il modello
        self.model = TabM(**final_params)
        self.model.fit(X_processed, y, **fit_params)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predice sui dati.
        
        Args:
            X: Feature per predizione
            
        Returns:
            Predizioni
        """
        if not self.is_fitted:
            raise ValueError("Il modello deve essere fittato prima della predizione")
            
        X_processed = self._preprocess_data(X, fit_encoders=False)
        return self.model.predict(X_processed)
    
    def score(self, X, y, sample_weight=None):
        """
        Calcola lo score del modello.
        
        Args:
            X: Feature
            y: Target
            sample_weight: Pesi dei campioni
            
        Returns:
            Score del modello
        """
        if not self.is_fitted:
            raise ValueError("Il modello deve essere fittato prima del calcolo dello score")
            
        X_processed = self._preprocess_data(X, fit_encoders=False)
        return self.model.score(X_processed, y, sample_weight=sample_weight)
    
    def get_params(self, deep=True):
        """Ottiene i parametri del modello."""
        return self.params
    
    def set_params(self, **params):
        """Imposta i parametri del modello."""
        self.params.update(params)
        return self

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

def objective_tabm(trial, X_train, y_train, cv_folds=5, random_state=42, n_jobs=-1, cv_strategy=None, config=None):
    """Funzione obiettivo per TabM con gestione automatica delle categoriche"""
    
    # Determina il numero di features dopo eventuale preprocessing
    expected_n_features = X_train.shape[1]
    
    # Se non ci sono features, salta TabM
    if expected_n_features == 0:
        logger.warning("⚠️ TabM richiede almeno una feature. Skipping TabM optimization.")
        return float('-inf')  # Valore pessimo per indicare che il modello non può essere usato
    
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
        'verbosity': 0,
        'n_num_features': expected_n_features,  # Sarà aggiornato dal wrapper
        'cat_cardinalities': [],
        'k': trial.suggest_int('k', 1, min(10, max(1, expected_n_features // 4)))  # Add required k parameter
    }
    
    # Usa il wrapper che gestisce automaticamente le feature categoriche
    model = TabMWrapper(**params)
    
    # Usa cv_strategy se fornito, altrimenti KFold
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ottieni metrica dal config
    scoring = config.get('optimization_metric', 'neg_root_mean_squared_error') if config else 'neg_root_mean_squared_error'
    
    # Cross validation manuale perché TabMWrapper non è direttamente compatibile con cross_val_score
    scores = []
    for train_idx, val_idx in cv_strategy.split(X_train):
        X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # Crea nuovo modello per ogni fold
        fold_model = TabMWrapper(**params)
        fold_model.fit(X_train_fold, y_train_fold)
        y_pred = fold_model.predict(X_val_fold)
        
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
    elif model_name == "TabM":
        # Assicuriamoci che i parametri necessari per TabM siano presenti
        if 'n_num_features' not in params or 'cat_cardinalities' not in params:
            logger.warning("⚠️ TabM richiede n_num_features e cat_cardinalities. Aggiungendo valori di default.")
            # Assume che abbiamo solo features numeriche se non specificato
            params.setdefault('n_num_features', 1)  # Sarà aggiornato durante il training
            params.setdefault('cat_cardinalities', [])
        
        # Usa il wrapper che gestisce automaticamente il preprocessing delle categoriche
        return TabMWrapper(**params)
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