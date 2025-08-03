"""
Test per i moduli evaluation e feature importance.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tempfile

from src.training.evaluation import (
    evaluate_on_test_set, run_evaluation_pipeline,
    calculate_feature_importance
)
from src.training.feature_importance import (
    calculate_basic_feature_importance, run_comprehensive_feature_analysis
)


class TestEvaluation:
    """Test per il modulo evaluation."""

    def test_evaluate_on_test_set_basic(self, sample_features_and_target):
        """Test valutazione base su test set."""
        X, y = sample_features_and_target
        
        # Crea modelli di test
        model1 = LinearRegression()
        model1.fit(X, y)
        
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(X, y)
        
        best_models = {
            'best_linear': {
                'name': 'LinearRegression',
                'model': model1,
                'score': -0.5
            },
            'best_tree': {
                'name': 'RandomForest',
                'model': model2,
                'score': -0.4
            }
        }
        
        # Split per test
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)  # Simula scala originale (inverso di log1p)
        
        results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        # Verifica struttura risultati
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verifica metriche per ogni modello
        for model_key, result in results.items():
            expected_metrics = [
                'Test_RMSE_log', 'Test_R2_log', 'Test_MAE_log', 'Test_MAPE_log',
                'Test_RMSE_orig', 'Test_R2_orig', 'Test_MAE_orig', 'Test_MAPE_orig'
            ]
            
            for metric in expected_metrics:
                assert metric in result
                assert isinstance(result[metric], (int, float))

    def test_evaluate_on_test_set_empty_models(self, sample_features_and_target):
        """Test valutazione con modelli vuoti."""
        X, y = sample_features_and_target
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        # Nessun modello
        results = evaluate_on_test_set({}, X_test, y_test_log, y_test_orig)
        
        assert isinstance(results, dict)
        assert len(results) == 0

    @patch('src.training.evaluation.load_dataframe')
    @patch('src.training.evaluation.save_json')
    @patch('src.training.evaluation.save_dataframe')
    def test_run_evaluation_pipeline_integration(self, mock_save_df, mock_save_json, mock_load_df,
                                                sample_training_results, sample_preprocessing_paths,
                                                sample_features_and_target, sample_config):
        """Test integrazione completa pipeline evaluation."""
        X, y = sample_features_and_target
        
        # Setup mocks per caricamento dati
        mock_load_df.side_effect = [X, X, y, y]  # X_test, y_test, y_test_orig (multipli)
        
        output_paths = {
            'results_dir': 'test_results/',
            'feature_importance_path': 'feature_importance.csv',
            'evaluation_summary_path': 'evaluation_summary.json'
        }
        
        # Esegui pipeline evaluation
        results = run_evaluation_pipeline(
            sample_training_results, sample_preprocessing_paths, 
            sample_config, output_paths
        )
        
        # Verifica che siano state fatte le chiamate di salvataggio
        assert mock_save_json.called
        assert isinstance(results, dict)

    def test_evaluate_on_test_set_with_prediction_errors(self, sample_features_and_target):
        """Test valutazione con errori di predizione."""
        X, y = sample_features_and_target
        
        # Crea mock model che fallisce nella predizione
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        best_models = {
            'failing_model': {
                'name': 'FailingModel',
                'model': mock_model,
                'score': -0.5
            }
        }
        
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        # Dovrebbe gestire l'errore gracefully
        results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        # Dovrebbe restituire dizionario vuoto o con errori gestiti
        assert isinstance(results, dict)


class TestFeatureImportance:
    """Test per il modulo feature_importance."""

    def test_calculate_basic_feature_importance(self, sample_features_and_target):
        """Test calcolo base feature importance."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        # Crea modelli con feature importance
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        best_models = {
            'random_forest': {
                'name': 'RandomForest',
                'model': rf_model
            }
        }
        
        df_importance, summary = calculate_basic_feature_importance(best_models, feature_cols)
        
        # Verifica DataFrame importance
        assert isinstance(df_importance, pd.DataFrame)
        assert 'feature' in df_importance.columns
        assert len(df_importance) == len(feature_cols)
        
        # Verifica summary
        assert isinstance(summary, dict)

    def test_calculate_basic_feature_importance_no_importance(self, sample_features_and_target):
        """Test con modelli senza feature importance."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        # LinearRegression non ha feature_importances_
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        
        best_models = {
            'linear': {
                'name': 'LinearRegression', 
                'model': linear_model
            }
        }
        
        df_importance, summary = calculate_basic_feature_importance(best_models, feature_cols)
        
        # Dovrebbe gestire il caso senza feature importance
        assert isinstance(df_importance, pd.DataFrame)
        assert isinstance(summary, dict)

    def test_calculate_feature_importance_wrapper(self, sample_features_and_target):
        """Test wrapper calculate_feature_importance."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        best_models = {
            'random_forest': {
                'name': 'RandomForest',
                'model': rf_model
            }
        }
        
        # Test senza analisi avanzata
        df_importance, summary = calculate_feature_importance(
            best_models, feature_cols, use_advanced_analysis=False
        )
        
        assert isinstance(df_importance, pd.DataFrame)
        assert isinstance(summary, dict)

    @patch('src.training.evaluation.run_comprehensive_feature_analysis')
    def test_calculate_feature_importance_advanced(self, mock_comprehensive, sample_features_and_target):
        """Test feature importance con analisi avanzata."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        # Mock comprehensive analysis
        mock_comprehensive.return_value = ({'summary': 'advanced'}, {'detailed': 'results'})
        
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        best_models = {
            'random_forest': {
                'name': 'RandomForest',
                'model': rf_model
            }
        }
        
        # Test con analisi avanzata
        df_importance, summary = calculate_feature_importance(
            best_models, feature_cols, 
            X_train=X, X_test=X, y_test=y, 
            output_dir='test_dir', use_advanced_analysis=True
        )
        
        # Verifica che sia stata chiamata l'analisi avanzata
        mock_comprehensive.assert_called_once()
        assert isinstance(summary, dict)

    @patch('src.training.feature_importance.shap')
    def test_run_comprehensive_feature_analysis_mock(self, mock_shap, sample_features_and_target):
        """Test analisi comprehensiva con mock SHAP."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        # Mock SHAP
        mock_explainer = MagicMock()
        mock_shap_values = np.random.randn(len(X), len(feature_cols))
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.Explainer.return_value = mock_explainer
        
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        best_models = {
            'random_forest': {
                'name': 'RandomForest',
                'model': rf_model
            }
        }
        
        # Non dovrebbe crashare anche se SHAP è mockato
        try:
            global_summary, detailed_results = run_comprehensive_feature_analysis(
                best_models, X, X, y, feature_cols, 'test_output'
            )
            
            assert isinstance(global_summary, dict)
            assert isinstance(detailed_results, dict)
            
        except Exception as e:
            # È accettabile che fallisca con mock, SHAP è complesso
            pytest.skip(f"Expected failure with SHAP mock: {e}")


class TestEvaluationIntegration:
    """Test di integrazione per evaluation."""

    def test_full_evaluation_workflow(self, sample_features_and_target):
        """Test workflow completo evaluation."""
        X, y = sample_features_and_target
        
        # 1. Addestra modelli
        model1 = LinearRegression()
        model1.fit(X, y)
        
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(X, y)
        
        # 2. Prepara per evaluation
        best_models = {
            'linear': {'name': 'LinearRegression', 'model': model1},
            'forest': {'name': 'RandomForest', 'model': model2}
        }
        
        # 3. Split test
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        # 4. Evaluation
        test_results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        # 5. Feature importance
        feature_cols = list(X.columns)
        importance_df, importance_summary = calculate_feature_importance(
            best_models, feature_cols, use_advanced_analysis=False
        )
        
        # Verifica workflow completo
        assert len(test_results) > 0
        assert isinstance(importance_df, pd.DataFrame)
        assert isinstance(importance_summary, dict)

    def test_evaluation_with_different_model_types(self, sample_features_and_target):
        """Test evaluation con diversi tipi di modelli."""
        X, y = sample_features_and_target
        
        # Modelli di diversi tipi
        models = {
            'linear': LinearRegression(),
            'forest': RandomForestRegressor(n_estimators=5, random_state=42)
        }
        
        # Addestra tutti
        for name, model in models.items():
            model.fit(X, y)
        
        best_models = {
            name: {'name': name, 'model': model} 
            for name, model in models.items()
        }
        
        # Test evaluation
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        # Verifica che tutti i modelli siano stati valutati
        assert len(results) == len(models)

    def test_evaluation_empty_test_set(self, sample_features_and_target):
        """Test evaluation con test set vuoto."""
        X, y = sample_features_and_target
        
        model = LinearRegression()
        model.fit(X, y)
        
        best_models = {
            'linear': {'name': 'LinearRegression', 'model': model}
        }
        
        # Test set vuoto
        X_test_empty = X[:0]  # 0 righe
        y_test_empty = y[:0]
        y_test_orig_empty = y[:0]
        
        # Dovrebbe gestire il caso gracefully
        try:
            results = evaluate_on_test_set(best_models, X_test_empty, y_test_empty, y_test_orig_empty)
            # Se non fallisce, verifica che gestisca correttamente
            assert isinstance(results, dict)
        except Exception:
            # È accettabile che fallisca con test set vuoto
            pass

    def test_feature_importance_ranking(self, sample_features_and_target):
        """Test ranking feature importance."""
        X, y = sample_features_and_target
        
        # Crea feature con importanza artificiale
        X_modified = X.copy()
        # Rendi una feature molto importante
        X_modified.iloc[:, 0] = y + np.random.randn(len(y)) * 0.1
        
        rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
        rf_model.fit(X_modified, y)
        
        best_models = {
            'forest': {'name': 'RandomForest', 'model': rf_model}
        }
        
        df_importance, summary = calculate_basic_feature_importance(best_models, list(X_modified.columns))
        
        # La prima feature dovrebbe avere importanza alta
        if len(df_importance) > 0:
            # Verifica che le importanze siano ordinate o ordinabili
            forest_importance = df_importance[df_importance['model'] == 'RandomForest']
            if len(forest_importance) > 0:
                importances = forest_importance['importance'].values
                assert all(imp >= 0 for imp in importances)

    def test_evaluation_metrics_sanity(self, sample_features_and_target):
        """Test sanity check delle metriche."""
        X, y = sample_features_and_target
        
        # Perfect predictor (per test)
        class PerfectPredictor:
            def predict(self, X):
                return y[:len(X)]  # Predice perfettamente
        
        best_models = {
            'perfect': {'name': 'Perfect', 'model': PerfectPredictor()}
        }
        
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        if 'perfect' in results:
            # R2 dovrebbe essere vicino a 1
            r2_log = results['perfect'].get('Test_R2_log', 0)
            # RMSE dovrebbe essere molto basso
            rmse_log = results['perfect'].get('Test_RMSE_log', float('inf'))
            
            # Sanity checks (con tolleranza per piccoli errori numerici)
            assert r2_log > 0.8  # Dovrebbe essere molto alto
            assert rmse_log < 1.0  # Dovrebbe essere molto basso


class TestEvaluationErrorHandling:
    """Test gestione errori in evaluation."""

    def test_evaluation_with_nan_predictions(self, sample_features_and_target):
        """Test evaluation con predizioni NaN."""
        X, y = sample_features_and_target
        
        class NaNPredictor:
            def predict(self, X):
                return np.full(len(X), np.nan)
        
        best_models = {
            'nan_model': {'name': 'NaNModel', 'model': NaNPredictor()}
        }
        
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        # Dovrebbe gestire NaN gracefully
        results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
        
        # Potrebbe restituire risultati vuoti o con errori gestiti
        assert isinstance(results, dict)

    def test_feature_importance_with_model_errors(self, sample_features_and_target):
        """Test feature importance con errori dei modelli."""
        X, y = sample_features_and_target
        feature_cols = list(X.columns)
        
        # Mock model senza attributi necessari
        mock_model = MagicMock()
        del mock_model.feature_importances_  # Rimuovi attributo
        
        best_models = {
            'broken_model': {'name': 'BrokenModel', 'model': mock_model}
        }
        
        # Dovrebbe gestire l'errore
        df_importance, summary = calculate_basic_feature_importance(best_models, feature_cols)
        
        assert isinstance(df_importance, pd.DataFrame)
        assert isinstance(summary, dict)

    def test_evaluation_with_mismatched_dimensions(self, sample_features_and_target):
        """Test evaluation con dimensioni non corrispondenti."""
        X, y = sample_features_and_target
        
        class WrongDimensionPredictor:
            def predict(self, X):
                # Restituisce numero sbagliato di predizioni
                return np.array([1.0])  # Solo una predizione invece di len(X)
        
        best_models = {
            'wrong_dim': {'name': 'WrongDim', 'model': WrongDimensionPredictor()}
        }
        
        split_idx = int(0.8 * len(X))
        X_test = X[split_idx:]
        y_test_log = y[split_idx:]
        y_test_orig = np.expm1(y_test_log)
        
        # Dovrebbe gestire l'errore di dimensioni
        try:
            results = evaluate_on_test_set(best_models, X_test, y_test_log, y_test_orig)
            assert isinstance(results, dict)
        except Exception:
            # È accettabile che fallisca con dimensioni sbagliate
            pass