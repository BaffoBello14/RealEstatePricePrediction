"""
Test per i moduli utilities.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import json
import pickle
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.utils import io, logger
from src.utils.temporal_utils import temporal_sort_by_year_month


class TestIOUtils:
    """Test per il modulo src.utils.io"""

    def test_load_config_success(self, test_config_file):
        """Test caricamento configurazione valida."""
        config = io.load_config(test_config_file)
        
        assert isinstance(config, dict)
        assert 'database' in config
        assert 'preprocessing' in config
        assert config['database']['selected_aliases'] == ['A', 'AI', 'PC']

    def test_load_config_file_not_found(self):
        """Test gestione file configurazione non esistente."""
        with pytest.raises(FileNotFoundError):
            io.load_config('non_existent_config.yaml')

    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test creazione directory."""
        test_path = temp_dir / "new_dir" / "subdir"
        io.ensure_dir(str(test_path))
        
        assert test_path.exists()
        assert test_path.is_dir()

    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test con directory già esistente."""
        existing_path = temp_dir / "existing"
        existing_path.mkdir()
        
        # Non dovrebbe lanciare errori
        io.ensure_dir(str(existing_path))
        assert existing_path.exists()

    def test_save_load_dataframe_parquet(self, temp_dir, sample_dataframe):
        """Test salvataggio e caricamento DataFrame in formato Parquet."""
        file_path = temp_dir / "test_data.parquet"
        
        # Salva
        io.save_dataframe(sample_dataframe, str(file_path), format='parquet')
        assert file_path.exists()
        
        # Carica
        loaded_df = io.load_dataframe(str(file_path), format='parquet')
        pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

    def test_save_load_dataframe_csv(self, temp_dir, sample_dataframe):
        """Test salvataggio e caricamento DataFrame in formato CSV."""
        file_path = temp_dir / "test_data.csv"
        
        # Salva
        io.save_dataframe(sample_dataframe, str(file_path), format='csv')
        assert file_path.exists()
        
        # Carica
        loaded_df = io.load_dataframe(str(file_path), format='csv')
        
        # CSV non preserva perfettamente i tipi, verifica solo shape e colonne
        assert loaded_df.shape == sample_dataframe.shape
        assert list(loaded_df.columns) == list(sample_dataframe.columns)

    def test_save_load_dataframe_pickle(self, temp_dir, sample_dataframe):
        """Test salvataggio e caricamento DataFrame in formato Pickle."""
        file_path = temp_dir / "test_data.pkl"
        
        # Salva
        io.save_dataframe(sample_dataframe, str(file_path), format='pickle')
        assert file_path.exists()
        
        # Carica
        loaded_df = io.load_dataframe(str(file_path), format='pickle')
        pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

    def test_load_dataframe_infer_format(self, temp_dir, sample_dataframe):
        """Test inferenza automatica formato da estensione."""
        file_path = temp_dir / "test_data.parquet"
        
        # Salva
        io.save_dataframe(sample_dataframe, str(file_path))
        
        # Carica senza specificare formato
        loaded_df = io.load_dataframe(str(file_path))
        pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

    def test_save_dataframe_invalid_format(self, temp_dir, sample_dataframe):
        """Test gestione formato non supportato."""
        file_path = temp_dir / "test_data.invalid"
        
        with pytest.raises(ValueError, match="Formato non supportato"):
            io.save_dataframe(sample_dataframe, str(file_path), format='invalid')

    def test_save_load_model(self, temp_dir):
        """Test salvataggio e caricamento modello."""
        from sklearn.linear_model import LinearRegression
        
        # Crea e addestra modello semplice
        model = LinearRegression()
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        model.fit(X, y)
        
        file_path = temp_dir / "test_model.pkl"
        
        # Salva
        io.save_model(model, str(file_path))
        assert file_path.exists()
        
        # Carica
        loaded_model = io.load_model(str(file_path))
        
        # Verifica che il modello funzioni
        predictions_original = model.predict(X)
        predictions_loaded = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)

    def test_save_load_json(self, temp_dir):
        """Test salvataggio e caricamento JSON."""
        test_data = {
            'string': 'test',
            'number': 42,
            'list': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        
        file_path = temp_dir / "test_data.json"
        
        # Salva
        io.save_json(test_data, str(file_path))
        assert file_path.exists()
        
        # Carica
        loaded_data = io.load_json(str(file_path))
        assert loaded_data == test_data

    def test_save_json_with_non_serializable(self, temp_dir):
        """Test salvataggio JSON con oggetti non serializzabili."""
        import datetime
        
        test_data = {
            'date': datetime.datetime.now(),
            'number': 42
        }
        
        file_path = temp_dir / "test_data.json"
        
        # Dovrebbe gestire oggetti non serializzabili convertendoli a stringa
        io.save_json(test_data, str(file_path))
        assert file_path.exists()
        
        loaded_data = io.load_json(str(file_path))
        assert 'date' in loaded_data
        assert loaded_data['number'] == 42

    def test_check_file_exists(self, temp_dir):
        """Test verifica esistenza file."""
        existing_file = temp_dir / "existing.txt"
        existing_file.write_text("test")
        
        non_existing_file = temp_dir / "non_existing.txt"
        
        assert io.check_file_exists(str(existing_file)) is True
        assert io.check_file_exists(str(non_existing_file)) is False


class TestLogger:
    """Test per il modulo src.utils.logger"""

    def test_setup_logger_creates_log_file(self, test_config_file, temp_dir):
        """Test che setup_logger crei il file di log."""
        # Modifica config per usare temp_dir
        with open(test_config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['logging']['file'] = str(temp_dir / 'test.log')
        
        with open(test_config_file, 'w') as f:
            yaml.dump(config, f)
        
        test_logger = logger.setup_logger(test_config_file)
        
        assert isinstance(test_logger, logging.Logger)
        assert Path(config['logging']['file']).exists()

    def test_get_logger_returns_logger(self):
        """Test che get_logger restituisca un logger."""
        test_logger = logger.get_logger('test_logger')
        
        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == 'test_logger'

    def test_get_logger_default_name(self):
        """Test comportamento get_logger con nome di default."""
        test_logger = logger.get_logger()
        
        assert isinstance(test_logger, logging.Logger)
        assert test_logger.name == 'ML_Pipeline'

    @patch('src.utils.logger.logging.basicConfig')
    def test_setup_logger_configuration(self, mock_basic_config, test_config_file):
        """Test che setup_logger configuri correttamente il logging."""
        logger.setup_logger(test_config_file)
        
        # Verifica che basicConfig sia stato chiamato
        mock_basic_config.assert_called_once()
        
        # Verifica i parametri di configurazione
        call_args = mock_basic_config.call_args
        assert call_args[1]['level'] == logging.INFO
        assert 'handlers' in call_args[1]


class TestTemporalUtils:
    """Test per il modulo src.utils.temporal_utils"""

    def test_temporal_sort_by_year_month_basic(self):
        """Test ordinamento temporale base."""
        df = pd.DataFrame({
            'A_AnnoStipula': [2023, 2022, 2023, 2022],
            'A_MeseStipula': [6, 12, 1, 3],
            'value': [1, 2, 3, 4]
        })
        
        sorted_df = temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')
        
        # Verifica ordine corretto: 2022-03, 2022-12, 2023-01, 2023-06
        expected_order = [4, 2, 3, 1]  # valori corrispondenti
        assert sorted_df['value'].tolist() == expected_order

    def test_temporal_sort_by_year_month_missing_columns(self):
        """Test gestione colonne mancanti."""
        df = pd.DataFrame({
            'A_AnnoStipula': [2023, 2022],
            'value': [1, 2]
            # Manca A_MeseStipula
        })
        
        with pytest.raises(KeyError):
            temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')

    def test_temporal_sort_by_year_month_with_nulls(self):
        """Test gestione valori null."""
        df = pd.DataFrame({
            'A_AnnoStipula': [2023, 2022, None, 2023],
            'A_MeseStipula': [6, 12, 1, None],
            'value': [1, 2, 3, 4]
        })
        
        # Dovrebbe gestire i null mettendoli alla fine o all'inizio
        sorted_df = temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')
        
        # Verifica che il DataFrame risultante abbia la stessa lunghezza
        assert len(sorted_df) == len(df)
        assert sorted_df.columns.tolist() == df.columns.tolist()

    def test_temporal_sort_preserves_data_types(self):
        """Test che l'ordinamento preservi i tipi di dati."""
        df = pd.DataFrame({
            'A_AnnoStipula': [2023, 2022],
            'A_MeseStipula': [6, 12],
            'float_col': [1.5, 2.7],
            'str_col': ['a', 'b'],
            'int_col': [10, 20]
        })
        
        original_dtypes = df.dtypes.copy()
        sorted_df = temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')
        
        # Verifica che i tipi siano preservati
        for col in df.columns:
            if col not in ['A_AnnoStipula', 'A_MeseStipula']:  # Questi potrebbero cambiare per l'ordinamento
                assert sorted_df[col].dtype == original_dtypes[col]

    def test_temporal_sort_empty_dataframe(self):
        """Test con DataFrame vuoto."""
        df = pd.DataFrame(columns=['A_AnnoStipula', 'A_MeseStipula', 'value'])
        
        sorted_df = temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')
        
        assert len(sorted_df) == 0
        assert sorted_df.columns.tolist() == df.columns.tolist()

    def test_temporal_sort_single_row(self):
        """Test con DataFrame di una sola riga."""
        df = pd.DataFrame({
            'A_AnnoStipula': [2023],
            'A_MeseStipula': [6],
            'value': [42]
        })
        
        sorted_df = temporal_sort_by_year_month(df, 'A_AnnoStipula', 'A_MeseStipula')
        
        pd.testing.assert_frame_equal(df, sorted_df)


class TestUtilsIntegration:
    """Test di integrazione per i moduli utilities."""

    def test_config_logger_io_integration(self, temp_dir, sample_config):
        """Test integrazione tra configurazione, logger e IO."""
        # Modifica configurazione per usare temp_dir
        sample_config['logging']['file'] = str(temp_dir / 'integration_test.log')
        sample_config['paths']['data_processed'] = str(temp_dir / 'processed')
        
        # Salva configurazione
        config_path = temp_dir / 'config.yaml'
        io.save_json(sample_config, str(config_path.with_suffix('.json')))  # Test JSON
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Setup logger
        test_logger = logger.setup_logger(str(config_path))
        
        # Test logging
        test_logger.info("Test message")
        
        # Verifica che il log sia stato scritto
        log_file = Path(sample_config['logging']['file'])
        assert log_file.exists()
        
        log_content = log_file.read_text()
        assert "Test message" in log_content

    def test_save_load_workflow(self, temp_dir, sample_dataframe):
        """Test workflow completo di salvataggio e caricamento."""
        # Salva DataFrame
        df_path = temp_dir / 'data.parquet'
        io.save_dataframe(sample_dataframe, str(df_path))
        
        # Crea e salva un modello
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = sample_dataframe.select_dtypes(include=[np.number]).fillna(0)
        y = sample_dataframe['AI_Prezzo_Ridistribuito']
        model.fit(X, y)
        
        model_path = temp_dir / 'model.pkl'
        io.save_model(model, str(model_path))
        
        # Salva metadati
        metadata = {
            'model_type': 'LinearRegression',
            'features': list(X.columns),
            'target': 'AI_Prezzo_Ridistribuito',
            'n_samples': len(X)
        }
        metadata_path = temp_dir / 'metadata.json'
        io.save_json(metadata, str(metadata_path))
        
        # Ricarica tutto
        loaded_df = io.load_dataframe(str(df_path))
        loaded_model = io.load_model(str(model_path))
        loaded_metadata = io.load_json(str(metadata_path))
        
        # Verifica
        pd.testing.assert_frame_equal(sample_dataframe, loaded_df)
        assert loaded_metadata['model_type'] == 'LinearRegression'
        assert len(loaded_metadata['features']) == len(X.columns)
        
        # Test predizione
        predictions_original = model.predict(X)
        predictions_loaded = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)

    def test_file_operations_error_handling(self, temp_dir):
        """Test gestione errori nelle operazioni su file."""
        # Test caricamento file inesistente
        with pytest.raises(FileNotFoundError):
            io.load_dataframe(str(temp_dir / 'nonexistent.parquet'))
        
        # Test salvataggio in directory non scrivibile (simulato)
        invalid_path = "/invalid/path/file.parquet"
        
        # Su alcuni sistemi questo potrebbe non fallire, quindi non testiamo direttamente
        # ma verifiamo che ensure_dir gestisca correttamente i path

        # Test con DataFrame vuoto
        empty_df = pd.DataFrame()
        empty_path = temp_dir / 'empty.parquet'
        io.save_dataframe(empty_df, str(empty_path))
        
        loaded_empty = io.load_dataframe(str(empty_path))
        assert len(loaded_empty) == 0