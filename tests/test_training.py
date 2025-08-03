"""
Test per i moduli training.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import optuna

from src.training.models import (
    get_baseline_models, objective_random_forest, 
    create_model_from_params, create_ensemble_models
)
from src.training.train import (
    evaluate_baseline_models, evaluate_single_model, run_training_pipeline
)
from src.training.tuning import optimize_model, create_objective_function


class TestModels:
    """Test per il modulo models."""

    def test_get_baseline_models(self):
        """Test ottenimento modelli baseline."""
        models = get_baseline_models(random_state=42)
        
        # Verifica che siano restituiti i modelli attesi
        expected_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 
                          'DecisionTree', 'KNN', 'SVR']
        
        assert isinstance(models, dict)
        for model_name in expected_models:
            assert model_name in models
        
        # Verifica che siano effettivamente oggetti sklearn
        assert hasattr(models['LinearRegression'], 'fit')
        assert hasattr(models['Ridge'], 'predict')

    def test_objective_random_forest(self, sample_features_and_target):
        """Test funzione obiettivo Random Forest per Optuna."""
        X, y = sample_features_and_target
        
        # Mock trial
        trial = MagicMock()
        trial.suggest_int.side_effect = [100, 10, 2, 1]  # n_estimators, max_depth, min_samples_split, min_samples_leaf
        trial.suggest_categorical.side_effect = ['sqrt', True]  # max_features, bootstrap
        
        config = {'optimization_metric': 'neg_root_mean_squared_error'}
        
        # Esegui funzione obiettivo
        score = objective_random_forest(trial, X, y, cv_folds=3, random_state=42, config=config)
        
        # Verifica che restituisca un numero
        assert isinstance(score, (int, float))
        assert not np.isnan(score)

    def test_create_model_from_params(self):
        """Test creazione modello da parametri."""
        # Test Random Forest
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        model = create_model_from_params('RandomForest', params)
        
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 100
        assert model.max_depth == 10

    def test_create_model_from_params_unsupported(self):
        """Test creazione modello non supportato."""
        with pytest.raises((ValueError, KeyError)):
            create_model_from_params('UnsupportedModel', {})

    def test_create_ensemble_models(self):
        """Test creazione modelli ensemble."""
        # Mock dei modelli base
        base_models = {
            'model1': LinearRegression(),
            'model2': Ridge()
        }
        
        ensemble_config = {
            'voting_regressor': True,
            'stacking_regressor': True
        }
        
        ensemble_models = create_ensemble_models(base_models, ensemble_config, random_state=42)
        
        # Verifica che vengano creati i modelli ensemble
        if ensemble_config['voting_regressor']:
            assert 'VotingRegressor' in ensemble_models
        if ensemble_config['stacking_regressor']:
            assert 'StackingRegressor' in ensemble_models


class TestTraining:
    """Test per il modulo train."""

    @patch('src.training.train.cross_val_score')
    def test_evaluate_baseline_models(self, mock_cv_score, sample_features_and_target, sample_config):
        """Test valutazione modelli baseline."""
        X, y = sample_features_and_target
        
        # Mock dei punteggi di cross-validation
        mock_cv_score.return_value = np.array([-0.5, -0.6, -0.4])  # RMSE negativi
        
        results = evaluate_baseline_models(X, y, sample_config['training'])
        
        # Verifica che siano stati valutati i modelli
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verifica struttura risultati
        for model_name, result in results.items():
            if 'error' not in result:
                assert 'cv_score_mean' in result
                assert 'cv_score_std' in result
                assert 'model' in result

    def test_evaluate_single_model(self, sample_features_and_target):
        """Test valutazione singolo modello."""
        X, y = sample_features_and_target
        
        # Split manuale per test
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = LinearRegression()
        
        result = evaluate_single_model(model, X_train, y_train, X_val, y_val, 'LinearRegression')
        
        # Verifica metriche
        expected_metrics = ['Train_RMSE', 'Val_RMSE', 'Train_R2', 'Val_R2', 
                           'Train_MAE', 'Val_MAE', 'training_time_seconds']
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))

    @patch('src.training.train.load_dataframe')
    @patch('src.training.train.evaluate_baseline_models')
    @patch('src.training.train.run_full_optimization')
    def test_run_training_pipeline_integration(self, mock_optimization, mock_baseline, mock_load, 
                                              sample_preprocessing_paths, sample_config,
                                              sample_features_and_target, sample_training_results):
        """Test integrazione completa pipeline training."""
        X, y = sample_features_and_target
        
        # Setup mocks
        mock_load.side_effect = [X, X, X, y, y, y, y, y]  # Per tutti i file caricati
        mock_baseline.return_value = sample_training_results['baseline_results']
        mock_optimization.return_value = sample_training_results['optimized_results']
        
        # Esegui pipeline
        results = run_training_pipeline(sample_preprocessing_paths, sample_config['training'])
        
        # Verifica struttura risultati
        assert 'baseline_results' in results
        assert 'best_models' in results
        assert 'training_summary' in results

    def test_run_training_pipeline_no_advanced_models(self, sample_preprocessing_paths, sample_config):
        """Test pipeline con solo modelli baseline."""
        # Disabilita modelli avanzati
        config = sample_config['training'].copy()
        config_models = sample_config['models'].copy()
        config_models['advanced'] = {k: False for k in config_models['advanced']}
        
        with patch('src.training.train.load_dataframe') as mock_load:
            with patch('src.training.train.evaluate_baseline_models') as mock_baseline:
                X = pd.DataFrame(np.random.randn(50, 5))
                y = pd.Series(np.random.randn(50))
                
                mock_load.side_effect = [X, X, X, y, y, y, y, y]
                mock_baseline.return_value = {
                    'LinearRegression': {'cv_score_mean': -0.5, 'model': LinearRegression()}
                }
                
                # Non dovrebbe crashare
                results = run_training_pipeline(sample_preprocessing_paths, config)
                assert 'baseline_results' in results


class TestTuning:
    """Test per il modulo tuning."""

    def test_create_objective_function(self, sample_features_and_target, sample_config):
        """Test creazione funzione obiettivo."""
        X, y = sample_features_and_target
        
        objective_func = create_objective_function('RandomForest', X, y, sample_config['training'])
        
        # Verifica che sia una funzione
        assert callable(objective_func)
        
        # Test con mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = [100, 10, 2, 1]
        mock_trial.suggest_categorical.side_effect = ['sqrt', True]
        mock_trial.suggest_float.return_value = 0.1
        
        # Non dovrebbe crashare
        score = objective_func(mock_trial)
        assert isinstance(score, (int, float))

    @patch('optuna.create_study')
    def test_optimize_model_integration(self, mock_create_study, sample_features_and_target, sample_config):
        """Test integrazione ottimizzazione modello."""
        X, y = sample_features_and_target
        
        # Mock study e optimization
        mock_study = MagicMock()
        mock_study.best_params = {'n_estimators': 100, 'max_depth': 10}
        mock_study.best_value = -0.45
        mock_study.trials = [MagicMock() for _ in range(5)]
        mock_create_study.return_value = mock_study
        
        config = sample_config['training'].copy()
        config['n_trials'] = 5  # Ridotto per test
        
        result = optimize_model('RandomForest', X, y, config)
        
        # Verifica risultati
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'model' in result
        assert 'optimization_info' in result

    @patch('optuna.create_study')
    def test_optimize_model_with_timeout(self, mock_create_study, sample_features_and_target, sample_config):
        """Test ottimizzazione con timeout."""
        X, y = sample_features_and_target
        
        mock_study = MagicMock()
        mock_study.best_params = {'n_estimators': 50}
        mock_study.best_value = -0.5
        mock_study.trials = []
        mock_create_study.return_value = mock_study
        
        config = sample_config['training'].copy()
        config['optuna_timeout'] = 1  # 1 secondo
        config['n_trials'] = 1000  # Molti trial, ma dovrebbe fermarsi per timeout
        
        # Non dovrebbe impiegare più di qualche secondo
        result = optimize_model('RandomForest', X, y, config)
        
        assert result is not None

    def test_optimize_model_unsupported(self, sample_features_and_target, sample_config):
        """Test ottimizzazione modello non supportato."""
        X, y = sample_features_and_target
        
        with pytest.raises(ValueError, match="Modello non supportato"):
            optimize_model('UnsupportedModel', X, y, sample_config['training'])


class TestTrainingIntegration:
    """Test di integrazione per moduli training."""

    def test_full_training_workflow(self, sample_features_and_target, sample_config):
        """Test workflow completo training."""
        X, y = sample_features_and_target
        
        # 1. Valuta modelli baseline
        baseline_results = evaluate_baseline_models(X, y, sample_config['training'])
        
        # 2. Trova miglior baseline
        best_baseline = min(baseline_results.items(), 
                          key=lambda x: x[1].get('cv_score_mean', float('inf')))
        
        # 3. Test singolo modello
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_result = evaluate_single_model(
            best_baseline[1]['model'], X_train, y_train, X_val, y_val, best_baseline[0]
        )
        
        # Verifica che il workflow sia completo
        assert baseline_results
        assert model_result
        assert 'Val_RMSE' in model_result

    @patch('src.training.train.run_full_optimization')
    def test_training_with_optimization_failure(self, mock_optimization, 
                                               sample_preprocessing_paths, sample_config,
                                               sample_features_and_target):
        """Test gestione fallimento ottimizzazione."""
        X, y = sample_features_and_target
        
        # Mock ottimizzazione che fallisce
        mock_optimization.side_effect = Exception("Optimization failed")
        
        with patch('src.training.train.load_dataframe') as mock_load:
            with patch('src.training.train.evaluate_baseline_models') as mock_baseline:
                mock_load.side_effect = [X, X, X, y, y, y, y, y]
                mock_baseline.return_value = {
                    'LinearRegression': {'cv_score_mean': -0.5, 'model': LinearRegression()}
                }
                
                # Dovrebbe continuare anche se l'ottimizzazione fallisce
                results = run_training_pipeline(sample_preprocessing_paths, sample_config['training'])
                
                # Dovrebbe avere almeno i risultati baseline
                assert 'baseline_results' in results

    def test_model_consistency(self, sample_features_and_target):
        """Test consistenza predizioni modelli."""
        X, y = sample_features_and_target
        
        # Addestra due volte lo stesso modello con stesso random_state
        model1 = LinearRegression()
        model2 = LinearRegression()
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Le predizioni dovrebbero essere identiche
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_cross_validation_consistency(self, sample_features_and_target, sample_config):
        """Test consistenza cross-validation."""
        X, y = sample_features_and_target
        
        # Esegui due volte la valutazione con stesso random_state
        config = sample_config['training'].copy()
        config['random_state'] = 42
        
        results1 = evaluate_baseline_models(X, y, config)
        results2 = evaluate_baseline_models(X, y, config)
        
        # I punteggi dovrebbero essere simili (può esserci piccola variabilità)
        for model_name in results1:
            if 'error' not in results1[model_name] and 'error' not in results2[model_name]:
                score1 = results1[model_name]['cv_score_mean']
                score2 = results2[model_name]['cv_score_mean']
                assert abs(score1 - score2) < 0.1  # Tolleranza per variabilità CV

    def test_feature_importance_extraction(self, sample_features_and_target):
        """Test estrazione feature importance."""
        X, y = sample_features_and_target
        
        # Modelli con feature importance
        models_with_importance = [
            ('RandomForest', RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
        
        for name, model in models_with_importance:
            model.fit(X, y)
            
            # Verifica che abbia feature importance
            assert hasattr(model, 'feature_importances_')
            importances = model.feature_importances_
            
            # Verifica proprietà
            assert len(importances) == X.shape[1]
            assert all(imp >= 0 for imp in importances)
            assert abs(sum(importances) - 1.0) < 1e-10  # Somma dovrebbe essere 1

    def test_memory_usage_monitoring(self, sample_features_and_target):
        """Test che il training non usi memoria eccessiva."""
        X, y = sample_features_and_target
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Addestra un modello semplice
        model = LinearRegression()
        model.fit(X, y)
        _ = model.predict(X)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # L'aumento di memoria dovrebbe essere ragionevole (< 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes


class TestTrainingErrorHandling:
    """Test gestione errori nel training."""

    def test_training_with_invalid_data(self):
        """Test training con dati invalidi."""
        # Dati con NaN
        X_invalid = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [np.nan, 2, 3, 4]
        })
        y_invalid = pd.Series([1, np.nan, 3, 4])
        
        model = LinearRegression()
        
        # Dovrebbe gestire o fallire gracefully
        try:
            model.fit(X_invalid.fillna(0), y_invalid.fillna(0))
            # Se non fallisce, verifica che il modello sia addestrato
            assert hasattr(model, 'coef_')
        except Exception as e:
            # È accettabile che fallisca con dati invalidi
            assert "NaN" in str(e) or "finite" in str(e)

    def test_training_with_insufficient_data(self):
        """Test training con dati insufficienti."""
        # Un solo campione
        X_single = pd.DataFrame({'feature1': [1]})
        y_single = pd.Series([1])
        
        model = LinearRegression()
        
        # Dovrebbe gestire o fallire gracefully
        try:
            model.fit(X_single, y_single)
            # Se funziona, verifica predizione
            pred = model.predict(X_single)
            assert len(pred) == 1
        except Exception:
            # È accettabile che fallisca con dati insufficienti
            pass

    def test_model_prediction_consistency(self, sample_features_and_target):
        """Test consistenza predizioni."""
        X, y = sample_features_and_target
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predizioni multiple dovrebbero essere identiche
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)

    def test_cross_validation_with_small_folds(self, sample_features_and_target, sample_config):
        """Test cross-validation con pochi fold."""
        X, y = sample_features_and_target
        
        config = sample_config['training'].copy()
        config['cv_folds'] = 2  # Pochi fold
        
        # Non dovrebbe crashare
        results = evaluate_baseline_models(X, y, config)
        assert len(results) > 0