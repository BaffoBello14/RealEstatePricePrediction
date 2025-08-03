"""
Test per il modulo dataset e feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile

from src.dataset.build_dataset import (
    extract_floor_features, filter_coherent_acts, 
    calculate_omi_redistribution_coefficients, build_dataset
)


class TestFloorFeatureExtraction:
    """Test per l'estrazione features dai piani."""

    def test_extract_floor_features_simple_cases(self):
        """Test casi semplici di estrazione features dai piani."""
        # Piano terra
        features = extract_floor_features("PT")
        assert features['min_floor'] == 0
        assert features['max_floor'] == 0
        assert features['n_floors'] == 1
        assert features['has_ground'] == 1
        assert features['has_basement'] == 0
        assert features['has_upper'] == 0

        # Primo piano
        features = extract_floor_features("P1")
        assert features['min_floor'] == 1
        assert features['max_floor'] == 1
        assert features['n_floors'] == 1
        assert features['has_upper'] == 1
        assert features['has_ground'] == 0

        # Seminterrato
        features = extract_floor_features("S1")
        assert features['min_floor'] == -1
        assert features['max_floor'] == -1
        assert features['n_floors'] == 1
        assert features['has_basement'] == 1

    def test_extract_floor_features_complex_cases(self):
        """Test casi complessi con più piani."""
        # Range di piani
        features = extract_floor_features("P1-3")
        assert features['min_floor'] == 1
        assert features['max_floor'] == 3
        assert features['n_floors'] == 3
        assert features['floor_span'] == 2
        assert features['has_upper'] == 1

        # Combinazione seminterrato, terra e piani
        features = extract_floor_features("S1-T-2")
        assert features['min_floor'] == -1
        assert features['max_floor'] == 2
        assert features['n_floors'] == 3
        assert features['has_basement'] == 1
        assert features['has_ground'] == 1
        assert features['has_upper'] == 1

        # Piano terra + primo piano
        features = extract_floor_features("PT-P1")
        assert features['min_floor'] == 0
        assert features['max_floor'] == 1
        assert features['n_floors'] == 2
        assert features['has_ground'] == 1
        assert features['has_upper'] == 1

    def test_extract_floor_features_special_cases(self):
        """Test casi speciali."""
        # Seminterrato + Terra (duplex)
        features = extract_floor_features("ST")
        assert features['min_floor'] == -0.5
        assert features['max_floor'] == -0.5
        assert features['n_floors'] == 1

        # Rialzato
        features = extract_floor_features("RIAL")
        assert features['min_floor'] == 0.5
        assert features['max_floor'] == 0.5
        assert features['n_floors'] == 1

        # Solo numeri
        features = extract_floor_features("3")
        assert features['min_floor'] == 3
        assert features['max_floor'] == 3
        assert features['n_floors'] == 1
        assert features['has_upper'] == 1

        # Range numerico
        features = extract_floor_features("2-4")
        assert features['min_floor'] == 2
        assert features['max_floor'] == 4
        assert features['n_floors'] == 3

    def test_extract_floor_features_edge_cases(self):
        """Test casi edge e valori invalidi."""
        # Valori null/vuoti
        for null_value in [None, np.nan, "NULL", "", "   "]:
            features = extract_floor_features(null_value)
            assert pd.isna(features['min_floor'])
            assert pd.isna(features['max_floor'])
            assert features['n_floors'] == 0
            assert features['has_basement'] == 0
            assert features['has_ground'] == 0
            assert features['has_upper'] == 0
            assert features['floor_span'] == 0

        # Stringa non riconoscibile
        features = extract_floor_features("INVALID_FLOOR")
        assert features['n_floors'] == 0

        # Numeri fuori range
        features = extract_floor_features("99")  # Troppo alto
        assert features['n_floors'] == 0

        features = extract_floor_features("-10")  # Troppo basso
        assert features['n_floors'] == 0

    def test_extract_floor_features_weighted_average(self):
        """Test calcolo media pesata dei piani."""
        # Test con piani multipli per verificare il peso
        features = extract_floor_features("S1-T-P1-P2")
        
        # Verifica che la media pesata sia calcolata correttamente
        floors = [-1, 0, 1, 2]
        weights = [f + 3 for f in floors]  # [2, 3, 4, 5]
        expected_weighted = np.average(floors, weights=weights)
        
        assert abs(features['floor_numeric_weighted'] - expected_weighted) < 0.001

    def test_extract_floor_features_case_insensitive(self):
        """Test che il parsing sia case-insensitive."""
        features_upper = extract_floor_features("PT")
        features_lower = extract_floor_features("pt")
        features_mixed = extract_floor_features("Pt")
        
        # Tutti dovrebbero produrre lo stesso risultato
        assert features_upper['min_floor'] == features_lower['min_floor']
        assert features_upper['min_floor'] == features_mixed['min_floor']

    def test_extract_floor_features_data_types(self):
        """Test che i tipi di dati siano corretti."""
        features = extract_floor_features("P1")
        
        # Verifica tipi numerici
        assert isinstance(features['min_floor'], (int, float))
        assert isinstance(features['max_floor'], (int, float))
        assert isinstance(features['n_floors'], (int, float))
        assert isinstance(features['floor_span'], (int, float))
        assert isinstance(features['floor_numeric_weighted'], (int, float))
        
        # Verifica che i flags booleani siano 0 o 1
        assert features['has_basement'] in [0, 1]
        assert features['has_ground'] in [0, 1]
        assert features['has_upper'] in [0, 1]


class TestFilterCoherentActs:
    """Test per il filtro degli atti coerenti."""

    def test_filter_coherent_acts_basic(self):
        """Test filtro base atti coerenti."""
        # Crea dati di test
        df = pd.DataFrame({
            'A_Id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'A_TotaleImmobili': [3, 3, 3, 2, 2, 4, 4, 4, 4],
            'AI_Id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'other_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        })
        
        filtered_df = filter_coherent_acts(df)
        
        # Dovrebbe mantenere solo atti 1 e 3 (dove num_rows == TotaleImmobili)
        expected_a_ids = [1, 1, 1, 3, 3, 3, 3]
        assert filtered_df['A_Id'].tolist() == expected_a_ids
        assert len(filtered_df) == 7

    def test_filter_coherent_acts_all_coherent(self):
        """Test con tutti gli atti coerenti."""
        df = pd.DataFrame({
            'A_Id': [1, 2, 3, 3],
            'A_TotaleImmobili': [1, 1, 2, 2],
            'AI_Id': [1, 2, 3, 4]
        })
        
        filtered_df = filter_coherent_acts(df)
        
        # Tutti gli atti sono coerenti, nessuno dovrebbe essere rimosso
        assert len(filtered_df) == len(df)

    def test_filter_coherent_acts_all_incoherent(self):
        """Test con tutti gli atti incoerenti."""
        df = pd.DataFrame({
            'A_Id': [1, 1, 2, 2, 2],
            'A_TotaleImmobili': [3, 3, 1, 1, 1],  # Nessuno ha il numero corretto
            'AI_Id': [1, 2, 3, 4, 5]
        })
        
        filtered_df = filter_coherent_acts(df)
        
        # Tutti gli atti sono incoerenti, dovrebbe rimanere DataFrame vuoto
        assert len(filtered_df) == 0

    def test_filter_coherent_acts_empty_dataframe(self):
        """Test con DataFrame vuoto."""
        df = pd.DataFrame(columns=['A_Id', 'A_TotaleImmobili', 'AI_Id'])
        
        filtered_df = filter_coherent_acts(df)
        
        assert len(filtered_df) == 0
        assert list(filtered_df.columns) == list(df.columns)

    def test_filter_coherent_acts_single_act(self):
        """Test con singolo atto."""
        df = pd.DataFrame({
            'A_Id': [1, 1, 1],
            'A_TotaleImmobili': [3, 3, 3],
            'AI_Id': [1, 2, 3]
        })
        
        filtered_df = filter_coherent_acts(df)
        
        # L'atto è coerente, dovrebbe rimanere
        assert len(filtered_df) == 3
        pd.testing.assert_frame_equal(df, filtered_df)

    def test_filter_coherent_acts_preserves_other_columns(self):
        """Test che il filtro preservi tutte le altre colonne."""
        df = pd.DataFrame({
            'A_Id': [1, 1, 2, 2, 2],
            'A_TotaleImmobili': [2, 2, 2, 2, 2],  # Solo atto 1 è coerente
            'AI_Id': [1, 2, 3, 4, 5],
            'extra_col1': ['a', 'b', 'c', 'd', 'e'],
            'extra_col2': [10, 20, 30, 40, 50]
        })
        
        filtered_df = filter_coherent_acts(df)
        
        # Verifica che rimangano solo le righe dell'atto 1
        assert len(filtered_df) == 2
        assert filtered_df['A_Id'].tolist() == [1, 1]
        assert filtered_df['extra_col1'].tolist() == ['a', 'b']
        assert filtered_df['extra_col2'].tolist() == [10, 20]


class TestOMIRedistributionCoefficients:
    """Test per il calcolo coefficienti ridistribuzione OMI."""

    def test_calculate_omi_redistribution_coefficients_basic(self):
        """Test calcolo base coefficienti OMI."""
        # Crea dati con valori OMI diversi
        df = pd.DataFrame({
            'OV_Valore_normale': [1000, 1200, 1500, 2000],
            'OV_Valore_ottimo': [1200, 1440, 1800, 2400],
            'OV_Valore_scadente': [800, 960, 1200, 1600],
            'other_col': [1, 2, 3, 4]
        })
        
        result_df, coeffs = calculate_omi_redistribution_coefficients(df)
        
        # Verifica che siano stati calcolati i coefficienti
        assert 'coeff_ottimo' in coeffs
        assert 'coeff_scadente' in coeffs
        
        # Verifica che i coefficienti siano ragionevoli (ottimo > 1, scadente < 1)
        assert coeffs['coeff_ottimo'] > 1
        assert coeffs['coeff_scadente'] < 1
        
        # Verifica che sia stata aggiunta la colonna ridistribuita
        assert 'AI_Prezzo_Ridistribuito' in result_df.columns

    def test_calculate_omi_redistribution_coefficients_with_nulls(self):
        """Test gestione valori null nei dati OMI."""
        df = pd.DataFrame({
            'OV_Valore_normale': [1000, np.nan, 1500],
            'OV_Valore_ottimo': [1200, 1440, np.nan],
            'OV_Valore_scadente': [800, 960, 1200],
            'other_col': [1, 2, 3]
        })
        
        result_df, coeffs = calculate_omi_redistribution_coefficients(df)
        
        # Dovrebbe gestire i null correttamente
        assert not pd.isna(coeffs['coeff_ottimo'])
        assert not pd.isna(coeffs['coeff_scadente'])

    def test_calculate_omi_redistribution_coefficients_no_omi_columns(self):
        """Test con DataFrame senza colonne OMI."""
        df = pd.DataFrame({
            'other_col1': [1, 2, 3],
            'other_col2': ['a', 'b', 'c']
        })
        
        result_df, coeffs = calculate_omi_redistribution_coefficients(df)
        
        # Dovrebbe restituire DataFrame originale e coefficienti None/default
        pd.testing.assert_frame_equal(df, result_df)
        assert coeffs is None or len(coeffs) == 0

    def test_calculate_omi_redistribution_coefficients_edge_values(self):
        """Test con valori edge (zero, negativi)."""
        df = pd.DataFrame({
            'OV_Valore_normale': [0, 1000, -100],
            'OV_Valore_ottimo': [0, 1200, -120],
            'OV_Valore_scadente': [0, 800, -80],
        })
        
        # Non dovrebbe crashare
        result_df, coeffs = calculate_omi_redistribution_coefficients(df)
        
        # Verifica che il risultato sia valido
        assert result_df is not None
        assert len(result_df) == len(df)


class TestBuildDataset:
    """Test per la costruzione del dataset completo."""

    @patch('src.dataset.build_dataset.load_dataframe')
    @patch('src.dataset.build_dataset.save_dataframe')
    def test_build_dataset_integration(self, mock_save, mock_load):
        """Test integrazione completa build_dataset."""
        # Mock del DataFrame di input
        mock_df = pd.DataFrame({
            'A_Id': [1, 1, 1, 2, 2],
            'A_TotaleImmobili': [3, 3, 3, 2, 2],
            'AI_Id': [1, 2, 3, 4, 5],
            'AI_Piano': ['PT', 'P1', 'P2', 'S1', 'T'],
            'AI_Superficie': [80, 90, 100, 70, 85],
            'OV_Valore_normale': [1000, 1200, 1500, 1100, 1300],
            'OV_Valore_ottimo': [1200, 1440, 1800, 1320, 1560],
            'OV_Valore_scadente': [800, 960, 1200, 880, 1040]
        })
        mock_load.return_value = mock_df
        
        # Esegui build_dataset
        result = build_dataset('input.parquet', 'output.parquet')
        
        # Verifica chiamate
        mock_load.assert_called_once_with('input.parquet')
        mock_save.assert_called_once()
        
        # Verifica che il DataFrame salvato contenga le features dei piani
        saved_df = mock_save.call_args[0][0]
        
        # Verifica che siano state aggiunte le colonne delle features dei piani
        expected_floor_columns = [
            'min_floor', 'max_floor', 'n_floors', 'has_basement', 
            'has_ground', 'has_upper', 'floor_span', 'floor_numeric_weighted'
        ]
        
        for col in expected_floor_columns:
            assert col in saved_df.columns

        # Verifica che sia stata aggiunta la colonna ridistribuita
        assert 'AI_Prezzo_Ridistribuito' in saved_df.columns

    def test_build_dataset_floor_feature_integration(self):
        """Test integrazione features piani nel build_dataset."""
        # DataFrame di test con diversi tipi di piano
        df = pd.DataFrame({
            'A_Id': [1, 1, 1, 1],
            'A_TotaleImmobili': [4, 4, 4, 4],
            'AI_Id': [1, 2, 3, 4],
            'AI_Piano': ['PT', 'P1-3', 'S1', 'ST'],
            'AI_Superficie': [80, 90, 100, 70]
        })
        
        with patch('src.dataset.build_dataset.load_dataframe', return_value=df):
            with patch('src.dataset.build_dataset.save_dataframe') as mock_save:
                build_dataset('input.parquet', 'output.parquet')
                
                # Ottieni il DataFrame processato
                processed_df = mock_save.call_args[0][0]
                
                # Verifica che le features dei piani siano state calcolate correttamente
                # PT (piano terra)
                pt_row = processed_df[processed_df['AI_Piano'] == 'PT'].iloc[0]
                assert pt_row['has_ground'] == 1
                assert pt_row['has_basement'] == 0
                assert pt_row['has_upper'] == 0
                
                # P1-3 (piani 1-3)
                p13_row = processed_df[processed_df['AI_Piano'] == 'P1-3'].iloc[0]
                assert p13_row['min_floor'] == 1
                assert p13_row['max_floor'] == 3
                assert p13_row['n_floors'] == 3
                assert p13_row['has_upper'] == 1
                
                # S1 (seminterrato)
                s1_row = processed_df[processed_df['AI_Piano'] == 'S1'].iloc[0]
                assert s1_row['has_basement'] == 1

    def test_build_dataset_empty_input(self):
        """Test con input vuoto."""
        empty_df = pd.DataFrame()
        
        with patch('src.dataset.build_dataset.load_dataframe', return_value=empty_df):
            with patch('src.dataset.build_dataset.save_dataframe') as mock_save:
                result = build_dataset('input.parquet', 'output.parquet')
                
                # Dovrebbe gestire il caso vuoto senza errori
                mock_save.assert_called_once()
                saved_df = mock_save.call_args[0][0]
                assert len(saved_df) == 0

    @patch('src.dataset.build_dataset.load_dataframe')
    def test_build_dataset_error_handling(self, mock_load):
        """Test gestione errori nel build_dataset."""
        # Simula errore nel caricamento
        mock_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            build_dataset('nonexistent.parquet', 'output.parquet')


class TestDatasetIntegration:
    """Test di integrazione per il modulo dataset."""

    def test_full_pipeline_simulation(self):
        """Test simulazione pipeline completa dataset."""
        # Crea dati realistici
        raw_data = pd.DataFrame({
            'A_Id': [1, 1, 1, 2, 2],
            'A_Codice': ['ACT001', 'ACT001', 'ACT001', 'ACT002', 'ACT002'],
            'A_TotaleImmobili': [3, 3, 3, 2, 2],
            'AI_Id': [1, 2, 3, 4, 5],
            'AI_Piano': ['PT', 'P1', 'P2-3', 'S1-T', 'P1-2'],
            'AI_Superficie': [80.5, 90.0, 120.5, 70.0, 95.5],
            'AI_Vani': [3, 3, 4, 2, 3],
            'OV_Valore_normale': [2000, 2200, 2800, 1800, 2100],
            'OV_Valore_ottimo': [2400, 2640, 3360, 2160, 2520],
            'OV_Valore_scadente': [1600, 1760, 2240, 1440, 1680]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_input:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_output:
                # Salva dati di input
                raw_data.to_parquet(tmp_input.name, index=False)
                
                # Esegui build_dataset
                result = build_dataset(tmp_input.name, tmp_output.name)
                
                # Carica risultato
                processed_data = pd.read_parquet(tmp_output.name)
                
                # Verifica che il processing sia andato a buon fine
                assert len(processed_data) == 5  # Tutti gli atti sono coerenti
                
                # Verifica presenza nuove colonne
                floor_features = ['min_floor', 'max_floor', 'n_floors', 'has_basement', 'has_ground', 'has_upper']
                for feature in floor_features:
                    assert feature in processed_data.columns
                
                # Verifica calcoli specifici
                pt_row = processed_data[processed_data['AI_Piano'] == 'PT'].iloc[0]
                assert pt_row['has_ground'] == 1
                assert pt_row['min_floor'] == 0
                
                p23_row = processed_data[processed_data['AI_Piano'] == 'P2-3'].iloc[0]
                assert p23_row['min_floor'] == 2
                assert p23_row['max_floor'] == 3
                assert p23_row['n_floors'] == 2

    def test_feature_extraction_robustness(self):
        """Test robustezza estrazione features con dati reali problematici."""
        problematic_floors = [
            "PT-1-2",  # Combinazione complessa
            "S2-S1-T-P1",  # Molti livelli
            "PIANO TERRA",  # Descrizione testuale
            "1° PIANO",  # Con simboli
            "",  # Vuoto
            "???",  # Non riconoscibile
            "P10",  # Piano molto alto
            "SOTTOTETTO",  # Descrizione non standard
        ]
        
        df = pd.DataFrame({
            'A_Id': range(1, len(problematic_floors) + 1),
            'A_TotaleImmobili': [1] * len(problematic_floors),
            'AI_Id': range(1, len(problematic_floors) + 1),
            'AI_Piano': problematic_floors,
            'AI_Superficie': [80] * len(problematic_floors)
        })
        
        with patch('src.dataset.build_dataset.load_dataframe', return_value=df):
            with patch('src.dataset.build_dataset.save_dataframe') as mock_save:
                # Non dovrebbe crashare
                result = build_dataset('input.parquet', 'output.parquet')
                
                processed_df = mock_save.call_args[0][0]
                
                # Verifica che tutte le righe siano processate
                assert len(processed_df) == len(df)
                
                # Verifica che le colonne features siano presenti
                assert 'min_floor' in processed_df.columns
                assert 'n_floors' in processed_df.columns
                
                # Verifica che i valori problematici vengano gestiti
                # (potrebbero essere NaN o 0 a seconda della logica)
                assert not processed_df['n_floors'].isna().all()  # Almeno alcuni dovrebbero essere processati