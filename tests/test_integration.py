"""
Test di integrazione end-to-end per la pipeline ML.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from main import (
    run_retrieve_data, run_build_dataset, run_preprocessing,
    run_training, run_evaluation, setup_directories, main
)


class TestEndToEndIntegration:
    """Test di integrazione end-to-end della pipeline completa."""

    @patch('main.retrieve_data')
    @patch('main.check_file_exists')
    def test_run_retrieve_data_new_file(self, mock_check_exists, mock_retrieve, sample_config):
        """Test recupero dati quando il file non esiste."""
        mock_check_exists.return_value = False
        
        result_path = run_retrieve_data(sample_config)
        
        # Verifica chiamata a retrieve_data
        mock_retrieve.assert_called_once()
        expected_path = f"{sample_config['paths']['data_raw']}raw_data.parquet"
        assert result_path == expected_path

    @patch('main.check_file_exists')
    def test_run_retrieve_data_existing_file(self, mock_check_exists, sample_config):
        """Test recupero dati quando il file esiste già."""
        mock_check_exists.return_value = True
        
        result_path = run_retrieve_data(sample_config)
        
        # Non dovrebbe chiamare retrieve_data
        expected_path = f"{sample_config['paths']['data_raw']}raw_data.parquet"
        assert result_path == expected_path

    @patch('main.build_dataset')
    @patch('main.check_file_exists')
    def test_run_build_dataset(self, mock_check_exists, mock_build, sample_config):
        """Test costruzione dataset."""
        mock_check_exists.return_value = False
        raw_data_path = "test_raw.parquet"
        
        result_path = run_build_dataset(sample_config, raw_data_path)
        
        # Verifica chiamata a build_dataset
        mock_build.assert_called_once_with(raw_data_path, result_path)
        expected_path = f"{sample_config['paths']['data_processed']}dataset.parquet"
        assert result_path == expected_path

    @patch('main.run_preprocessing_pipeline')
    @patch('main.check_file_exists')
    def test_run_preprocessing(self, mock_check_exists, mock_preprocess, sample_config):
        """Test preprocessing."""
        mock_check_exists.return_value = False
        dataset_path = "test_dataset.parquet"
        
        result_paths = run_preprocessing(sample_config, dataset_path)
        
        # Verifica chiamata a preprocessing
        mock_preprocess.assert_called_once()
        assert isinstance(result_paths, dict)
        assert 'train_features' in result_paths

    @patch('main.run_training_pipeline')
    def test_run_training(self, mock_training, sample_config, sample_preprocessing_paths, sample_training_results):
        """Test training."""
        mock_training.return_value = sample_training_results
        
        results = run_training(sample_config, sample_preprocessing_paths)
        
        # Verifica chiamata a training
        mock_training.assert_called_once()
        assert isinstance(results, dict)
        assert 'baseline_results' in results

    @patch('main.run_evaluation_pipeline')
    def test_run_evaluation(self, mock_evaluation, sample_config, sample_training_results, sample_preprocessing_paths):
        """Test evaluation."""
        mock_evaluation.return_value = {'evaluation': 'results'}
        
        results = run_evaluation(sample_config, sample_training_results, sample_preprocessing_paths)
        
        # Verifica chiamata a evaluation
        mock_evaluation.assert_called_once()
        assert isinstance(results, dict)

    def test_setup_directories(self, sample_config, temp_dir):
        """Test setup directory."""
        # Modifica config per usare temp_dir
        config = sample_config.copy()
        config['paths'] = {
            'data_raw': str(temp_dir / 'raw'),
            'data_processed': str(temp_dir / 'processed'),
            'models': str(temp_dir / 'models'),
            'logs': str(temp_dir / 'logs')
        }
        
        setup_directories(config)
        
        # Verifica che le directory siano state create
        for path in config['paths'].values():
            assert Path(path).exists()

    @patch('main.load_config')
    @patch('main.setup_logger')
    @patch('main.setup_directories')
    @patch('main.run_retrieve_data')
    @patch('main.run_build_dataset')
    @patch('main.run_preprocessing')
    @patch('main.run_training')
    @patch('main.run_evaluation')
    def test_main_function_full_pipeline(self, mock_eval, mock_train, mock_preprocess, 
                                        mock_build, mock_retrieve, mock_setup_dirs,
                                        mock_setup_logger, mock_load_config, sample_config):
        """Test funzione main con pipeline completa."""
        # Setup mocks
        mock_load_config.return_value = sample_config
        mock_retrieve.return_value = 'raw_data.parquet'
        mock_build.return_value = 'dataset.parquet'
        mock_preprocess.return_value = {'train_features': 'X_train.parquet'}
        mock_train.return_value = {'baseline_results': {}}
        mock_eval.return_value = {'evaluation_results': {}}
        
        # Mock sys.argv
        with patch('sys.argv', ['main.py']):
            main()
        
        # Verifica che tutti gli step siano stati chiamati
        mock_retrieve.assert_called_once()
        mock_build.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_train.assert_called_once()
        mock_eval.assert_called_once()

    @patch('main.load_config')
    @patch('main.setup_logger')
    @patch('main.run_retrieve_data')
    def test_main_function_specific_steps(self, mock_retrieve, mock_setup_logger, mock_load_config, sample_config):
        """Test funzione main con step specifici."""
        # Setup mocks
        mock_load_config.return_value = sample_config
        mock_retrieve.return_value = 'raw_data.parquet'
        
        # Mock sys.argv per step specifici
        with patch('sys.argv', ['main.py', '--steps', 'retrieve_data']):
            main()
        
        # Verifica che solo retrieve_data sia stato chiamato
        mock_retrieve.assert_called_once()

    @patch('main.load_config')
    def test_main_function_error_handling(self, mock_load_config):
        """Test gestione errori nella funzione main."""
        # Mock errore nel caricamento config
        mock_load_config.side_effect = Exception("Config error")
        
        # Dovrebbe uscire con errore
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['main.py']):
                main()


class TestPipelineDataFlow:
    """Test del flusso dati attraverso la pipeline."""

    def test_data_flow_consistency(self, temp_dir, sample_dataframe):
        """Test consistenza dati attraverso la pipeline."""
        # Simula il flusso dati end-to-end
        
        # 1. Salva dati iniziali
        raw_path = temp_dir / 'raw_data.parquet'
        sample_dataframe.to_parquet(raw_path, index=False)
        
        # 2. Test build_dataset con dati reali
        from src.dataset.build_dataset import build_dataset
        dataset_path = temp_dir / 'dataset.parquet'
        
        build_dataset(str(raw_path), str(dataset_path))
        
        # 3. Verifica che il dataset sia stato creato
        assert dataset_path.exists()
        processed_df = pd.read_parquet(dataset_path)
        
        # 4. Verifica che i dati mantengano consistenza
        assert len(processed_df) <= len(sample_dataframe)  # Potrebbe filtrare
        assert 'AI_Prezzo_Ridistribuito' in processed_df.columns  # Target preservato

    @patch('src.preprocessing.pipeline.load_dataframe')
    @patch('src.preprocessing.pipeline.save_dataframe')
    @patch('src.preprocessing.pipeline.save_json')
    def test_preprocessing_data_flow(self, mock_save_json, mock_save_df, mock_load_df, 
                                   sample_dataframe, sample_config, temp_dir):
        """Test flusso dati nel preprocessing."""
        from src.preprocessing.pipeline import run_preprocessing_pipeline
        
        mock_load_df.return_value = sample_dataframe
        
        output_paths = {
            'train_features': str(temp_dir / 'X_train.parquet'),
            'val_features': str(temp_dir / 'X_val.parquet'),
            'test_features': str(temp_dir / 'X_test.parquet'),
            'train_target': str(temp_dir / 'y_train.parquet'),
            'val_target': str(temp_dir / 'y_val.parquet'),
            'test_target': str(temp_dir / 'y_test.parquet'),
            'val_target_orig': str(temp_dir / 'y_val_orig.parquet'),
            'test_target_orig': str(temp_dir / 'y_test_orig.parquet'),
            'preprocessing_info': str(temp_dir / 'preprocessing_info.json')
        }
        
        # Esegui preprocessing
        run_preprocessing_pipeline(
            dataset_path='input.parquet',
            target_column='AI_Prezzo_Ridistribuito',
            config=sample_config['preprocessing'],
            output_paths=output_paths
        )
        
        # Verifica che tutti i file siano stati salvati
        assert mock_save_df.call_count >= 6  # Almeno 6 file parquet
        mock_save_json.assert_called_once()  # preprocessing_info

    def test_training_evaluation_integration(self, sample_features_and_target, sample_config):
        """Test integrazione training-evaluation."""
        from src.training.train import evaluate_baseline_models, evaluate_single_model
        from src.training.evaluation import evaluate_on_test_set, calculate_feature_importance
        
        X, y = sample_features_and_target
        
        # 1. Training baseline
        baseline_results = evaluate_baseline_models(X, y, sample_config['training'])
        
        # 2. Trova miglior modello
        best_model_info = min(baseline_results.items(), 
                             key=lambda x: x[1].get('cv_score_mean', float('inf')))
        
        # 3. Evaluation dettagliata
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model_eval = evaluate_single_model(
            best_model_info[1]['model'], X_train, y_train, X_test, y_test, best_model_info[0]
        )
        
        # 4. Test set evaluation
        best_models = {
            'best': {
                'name': best_model_info[0],
                'model': best_model_info[1]['model']
            }
        }
        
        test_results = evaluate_on_test_set(best_models, X_test, y_test, np.exp(y_test))
        
        # 5. Feature importance
        importance_df, importance_summary = calculate_feature_importance(
            best_models, list(X.columns), use_advanced_analysis=False
        )
        
        # Verifica integrazione completa
        assert baseline_results
        assert model_eval
        assert test_results
        assert isinstance(importance_df, pd.DataFrame)


class TestPipelineRobustness:
    """Test robustezza della pipeline con dati problematici."""

    def test_pipeline_with_missing_data(self, sample_config):
        """Test pipeline con dati mancanti."""
        # Crea DataFrame con molti valori mancanti
        df_with_missing = pd.DataFrame({
            'A_Id': [1, 2, 3, 4, 5],
            'A_AnnoStipula': [2020, 2021, None, 2022, 2023],
            'A_MeseStipula': [1, None, 3, 4, 5],
            'AI_Superficie': [None, 90, 100, None, 85],
            'AI_IdCategoriaCatastale': ['A/2', None, 'A/4', 'C/1', None],
            'AI_Prezzo_Ridistribuito': [100000, 150000, 200000, 120000, 180000]
        })
        
        from src.preprocessing.cleaning import clean_data
        from src.preprocessing.imputation import handle_missing_values
        
        # Dovrebbe gestire i dati mancanti senza crashare
        df_cleaned, cleaning_info = clean_data(df_with_missing, 'AI_Prezzo_Ridistribuito', sample_config['preprocessing'])
        df_imputed, imputation_info = handle_missing_values(df_cleaned, 'AI_Prezzo_Ridistribuito')
        
        # Verifica che non ci siano più valori mancanti nel target
        assert not df_imputed['AI_Prezzo_Ridistribuito'].isna().any()
        
        # Verifica che le altre colonne siano state gestite
        assert not df_imputed.isna().any().any()

    def test_pipeline_with_outliers(self, sample_config):
        """Test pipeline con outlier estremi."""
        # Crea DataFrame con outlier
        df_with_outliers = pd.DataFrame({
            'A_Id': [1, 2, 3, 4, 5] * 20,
            'A_TotaleImmobili': [1, 1, 1, 1, 1] * 20,
            'AI_Id': range(1, 101),
            'AI_Superficie': [80, 90, 100, 10000, 85] * 20,  # Outlier estremo
            'AI_IdCategoriaCatastale': ['A/2', 'A/3', 'A/4', 'C/1', 'C/2'] * 20,
            'AI_Prezzo_Ridistribuito': [100000, 150000, 200000, 10000000, 180000] * 20  # Outlier prezzo
        })
        
        from src.preprocessing.cleaning import detect_outliers_multimethod
        
        # Rimuovi colonne non necessarie per test outlier
        feature_cols = ['AI_Superficie']
        X = df_with_outliers[feature_cols]
        y = df_with_outliers['AI_Prezzo_Ridistribuito']
        
        # Detection outlier dovrebbe funzionare
        outlier_mask = detect_outliers_multimethod(
            X, y, z_threshold=2.0, iqr_multiplier=1.5,
            isolation_contamination=0.1, min_methods=1
        )
        
        # Dovrebbe identificare alcuni outlier
        assert outlier_mask.sum() > 0
        assert len(outlier_mask) == len(df_with_outliers)

    def test_pipeline_with_single_category(self, sample_config):
        """Test pipeline con categorie singole."""
        # Crea DataFrame con una sola categoria
        df_single_cat = pd.DataFrame({
            'A_Id': [1, 1, 1],
            'A_TotaleImmobili': [3, 3, 3],
            'AI_Id': [1, 2, 3],
            'AI_IdCategoriaCatastale': ['A/2', 'A/2', 'A/2'],  # Una sola categoria
            'AI_Superficie': [80, 90, 100],
            'AI_Prezzo_Ridistribuito': [100000, 150000, 200000]
        })
        
        from src.preprocessing.encoding import encode_features
        
        # Encoding dovrebbe gestire categorie singole
        df_encoded, encoding_info = encode_features(
            df_single_cat, 'AI_Prezzo_Ridistribuito',
            low_card_threshold=5, apply_target_encoding=False
        )
        
        # Verifica che non crashi
        assert len(df_encoded) == len(df_single_cat)

    def test_pipeline_memory_efficiency(self, sample_config):
        """Test efficienza memoria della pipeline."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Crea dataset grande per test memoria
        large_df = pd.DataFrame({
            'A_Id': list(range(1, 5001)) * 2,  # 10000 righe
            'A_TotaleImmobili': [2] * 10000,
            'AI_Id': range(1, 10001),
            'AI_Superficie': np.random.normal(80, 20, 10000),
            'AI_IdCategoriaCatastale': np.random.choice(['A/2', 'A/3', 'A/4'], 10000),
            'AI_Prezzo_Ridistribuito': np.random.lognormal(11, 0.5, 10000)
        })
        
        from src.preprocessing.cleaning import clean_data
        
        # Dovrebbe processare senza usare memoria eccessiva
        df_cleaned, info = clean_data(large_df, 'AI_Prezzo_Ridistribuito', sample_config['preprocessing'])
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # L'aumento di memoria dovrebbe essere ragionevole (< 500MB)
        assert memory_increase < 500 * 1024 * 1024  # 500MB


class TestPipelineConfiguration:
    """Test configurazioni diverse della pipeline."""

    def test_pipeline_temporal_vs_random_split(self, sample_dataframe, sample_config):
        """Test differenze tra split temporale e casuale."""
        from src.preprocessing.transformation import split_dataset_with_validation
        
        # Split temporale
        results_temporal = split_dataset_with_validation(
            sample_dataframe, 'AI_Prezzo_Ridistribuito',
            test_size=0.2, val_size=0.2, use_temporal_split=True,
            year_column='A_AnnoStipula', month_column='A_MeseStipula'
        )
        
        # Split casuale
        results_random = split_dataset_with_validation(
            sample_dataframe, 'AI_Prezzo_Ridistribuito',
            test_size=0.2, val_size=0.2, use_temporal_split=False
        )
        
        # Entrambi dovrebbero funzionare
        assert len(results_temporal) == 9  # X_train, X_val, X_test, y_train, y_val, y_test, y_*_orig
        assert len(results_random) == 9
        
        # Verifica che le dimensioni siano coerenti
        X_train_t, X_val_t, X_test_t = results_temporal[:3]
        X_train_r, X_val_r, X_test_r = results_random[:3]
        
        total_samples = len(sample_dataframe)
        assert len(X_train_t) + len(X_val_t) + len(X_test_t) == total_samples
        assert len(X_train_r) + len(X_val_r) + len(X_test_r) == total_samples

    def test_pipeline_different_model_configs(self, sample_features_and_target, sample_config):
        """Test con configurazioni diverse dei modelli."""
        from src.training.train import evaluate_baseline_models
        
        X, y = sample_features_and_target
        
        # Configurazione solo modelli veloci
        fast_config = sample_config['training'].copy()
        fast_config['cv_folds'] = 2  # Pochi fold per velocità
        
        # Configurazione completa
        full_config = sample_config['training'].copy()
        full_config['cv_folds'] = 5
        
        # Entrambe dovrebbero funzionare
        results_fast = evaluate_baseline_models(X, y, fast_config)
        results_full = evaluate_baseline_models(X, y, full_config)
        
        assert len(results_fast) > 0
        assert len(results_full) > 0
        
        # Dovrebbero avere gli stessi modelli
        assert set(results_fast.keys()) == set(results_full.keys())

    def test_pipeline_feature_scaling_options(self, sample_features_and_target):
        """Test diverse opzioni di feature scaling."""
        from src.preprocessing.transformation import apply_feature_scaling
        
        X, y = sample_features_and_target
        
        # Test diversi metodi di scaling
        methods = ['standard', 'minmax', 'robust']
        
        for method in methods:
            X_scaled, scaler = apply_feature_scaling(X, method=method)
            
            # Verifica che lo scaling sia avvenuto
            assert X_scaled.shape == X.shape
            assert scaler is not None
            
            # Verifica proprietà specifiche del metodo
            if method == 'standard':
                # Media dovrebbe essere circa 0, std circa 1
                assert abs(X_scaled.mean().mean()) < 0.1
                assert abs(X_scaled.std().mean() - 1.0) < 0.1


class TestPipelineFailureRecovery:
    """Test recupero da fallimenti nella pipeline."""

    def test_pipeline_partial_failure_recovery(self, sample_config):
        """Test recupero da fallimenti parziali."""
        # Simula scenario dove alcuni step falliscono ma altri continuano
        
        # Mock che fallisce solo per certi input
        def failing_function(data):
            if len(data) > 50:
                raise Exception("Data too large")
            return data
        
        # Test che la pipeline possa continuare anche con alcuni fallimenti
        small_data = pd.DataFrame({'col': range(10), 'target': range(10, 20)})
        large_data = pd.DataFrame({'col': range(100), 'target': range(100, 200)})
        
        # Small data dovrebbe funzionare
        try:
            result_small = failing_function(small_data)
            assert len(result_small) == 10
        except Exception:
            pytest.fail("Small data should not fail")
        
        # Large data dovrebbe fallire gracefully
        try:
            result_large = failing_function(large_data)
            pytest.fail("Large data should fail")
        except Exception as e:
            assert "Data too large" in str(e)

    def test_pipeline_configuration_validation(self, sample_config):
        """Test validazione configurazione pipeline."""
        # Test configurazione valida
        assert 'preprocessing' in sample_config
        assert 'training' in sample_config
        assert 'models' in sample_config
        
        # Test che le configurazioni abbiano campi richiesti
        preprocessing_config = sample_config['preprocessing']
        assert 'steps' in preprocessing_config
        assert 'random_state' in preprocessing_config
        
        training_config = sample_config['training']
        assert 'cv_folds' in training_config
        assert 'random_state' in training_config
        
        # Test che i valori siano ragionevoli
        assert 1 <= training_config['cv_folds'] <= 10
        assert 0 <= preprocessing_config['random_state'] <= 2**32

    def test_pipeline_step_dependencies(self, sample_config):
        """Test dipendenze tra step della pipeline."""
        # Verifica che la configurazione abbia step in ordine logico
        execution_steps = sample_config['execution']['steps']
        
        expected_order = ['retrieve_data', 'build_dataset', 'preprocessing', 'training', 'evaluation']
        
        # Verifica che tutti gli step attesi siano presenti
        for step in expected_order:
            assert step in execution_steps
        
        # Verifica ordine (se l'ordine è importante nella configurazione)
        for i, step in enumerate(expected_order[:-1]):
            if step in execution_steps and expected_order[i+1] in execution_steps:
                step_idx = execution_steps.index(step)
                next_step_idx = execution_steps.index(expected_order[i+1])
                assert step_idx < next_step_idx, f"{step} should come before {expected_order[i+1]}"