"""
Model definitions and utilities for training
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import logging

# Third-party models (with fallbacks if not available)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    
try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_model_class(model_name):
    """
    Get model class by name
    
    Args:
        model_name: String name of the model
        
    Returns:
        Model class
    """
    model_mapping = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'HistGradientBoosting': HistGradientBoostingRegressor,
        'SVR': SVR,
        'KNeighbors': KNeighborsRegressor
    }
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        model_mapping['XGBoost'] = xgb.XGBRegressor
    
    # Add LightGBM if available
    if LGB_AVAILABLE:
        model_mapping['LightGBM'] = lgb.LGBMRegressor
    
    # Add CatBoost if available
    if CB_AVAILABLE:
        model_mapping['CatBoost'] = cb.CatBoostRegressor
    
    if model_name not in model_mapping:
        available_models = list(model_mapping.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return model_mapping[model_name]


def create_model(model_name, params=None):
    """
    Create a model instance
    
    Args:
        model_name: String name of the model
        params: Dictionary of parameters
        
    Returns:
        Model instance
    """
    if params is None:
        params = {}
    
    model_class = get_model_class(model_name)
    
    try:
        model = model_class(**params)
        logger.debug(f"Created {model_name} with params: {params}")
        return model
    except Exception as e:
        logger.error(f"Failed to create {model_name} with params {params}: {e}")
        raise


def get_enabled_models(model_configs):
    """
    Get list of enabled models from configuration
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        List of (model_name, config) tuples for enabled models
    """
    enabled_models = []
    
    for category, models in model_configs.items():
        for model_name, config in models.items():
            if config.get('enabled', False):
                # Check if required packages are available
                if model_name == 'XGBoost' and not XGB_AVAILABLE:
                    logger.warning(f"XGBoost not available, skipping")
                    continue
                elif model_name == 'LightGBM' and not LGB_AVAILABLE:
                    logger.warning(f"LightGBM not available, skipping")
                    continue
                elif model_name == 'CatBoost' and not CB_AVAILABLE:
                    logger.warning(f"CatBoost not available, skipping")
                    continue
                
                enabled_models.append((model_name, config))
    
    logger.info(f"Enabled models: {[name for name, _ in enabled_models]}")
    return enabled_models


def create_ensemble_models(best_models, ensemble_config):
    """
    Create ensemble models from best individual models
    
    Args:
        best_models: List of (model_name, model_instance, score) tuples
        ensemble_config: Ensemble configuration
        
    Returns:
        Dictionary of ensemble models
    """
    ensemble_models = {}
    
    # Voting Regressor
    if ensemble_config['voting_regressor']['enabled']:
        n_models = ensemble_config['voting_regressor']['n_best_models']
        voting_models = best_models[:n_models]
        
        estimators = [(name, model) for name, model, _ in voting_models]
        
        weights = None
        if ensemble_config['voting_regressor']['weights'] == 'performance':
            # Weight by inverse of score (assuming lower is better)
            scores = [score for _, _, score in voting_models]
            max_score = max(scores)
            weights = [max_score / score for score in scores]
        
        voting_regressor = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        
        ensemble_models['VotingRegressor'] = voting_regressor
        logger.info(f"Created VotingRegressor with {len(estimators)} models")
    
    # Stacking Regressor
    if ensemble_config['stacking_regressor']['enabled']:
        n_models = ensemble_config['stacking_regressor']['n_base_models']
        stacking_models = best_models[:n_models]
        
        base_estimators = [(name, model) for name, model, _ in stacking_models]
        
        # Create meta-learner
        meta_learner_name = ensemble_config['stacking_regressor']['meta_learner']
        meta_learner = create_model(meta_learner_name)
        
        stacking_regressor = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=ensemble_config['stacking_regressor']['cv_folds']
        )
        
        ensemble_models['StackingRegressor'] = stacking_regressor
        logger.info(f"Created StackingRegressor with {len(base_estimators)} base models")
    
    return ensemble_models


def get_model_info(model):
    """
    Get information about a model
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model information
    """
    info = {
        'class_name': model.__class__.__name__,
        'module': model.__class__.__module__,
        'params': model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    # Add specific info for different model types
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importance'] = True
    else:
        info['has_feature_importance'] = False
    
    if hasattr(model, 'coef_'):
        info['has_coefficients'] = True
    else:
        info['has_coefficients'] = False
    
    return info


def extract_feature_importance(model, feature_names=None):
    """
    Extract feature importance from a model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature importance data or None
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_values = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importance_values))]
    
    # Sort by importance
    importance_data = list(zip(feature_names, importance_values))
    importance_data.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'feature_names': [name for name, _ in importance_data],
        'importance_values': [val for _, val in importance_data],
        'total_features': len(importance_data)
    }


def extract_coefficients(model, feature_names=None):
    """
    Extract coefficients from a linear model
    
    Args:
        model: Trained linear model
        feature_names: List of feature names
        
    Returns:
        Dictionary with coefficient data or None
    """
    if not hasattr(model, 'coef_'):
        return None
    
    coef_values = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(coef_values))]
    
    # Sort by absolute coefficient value
    coef_data = list(zip(feature_names, coef_values))
    coef_data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    result = {
        'feature_names': [name for name, _ in coef_data],
        'coefficient_values': [val for _, val in coef_data],
        'total_features': len(coef_data)
    }
    
    # Add intercept if available
    if hasattr(model, 'intercept_'):
        result['intercept'] = float(model.intercept_)
    
    return result