"""
Test per i moduli database.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import os
from sqlalchemy import create_engine
import tempfile

from src.db.connect import get_engine, test_connection
from src.db.retrieve import (
    build_select_clause_dual_omi, get_poi_categories_query,
    generate_poi_counts_subquery, generate_ztl_subquery
)


class TestDatabaseConnect:
    """Test per il modulo src.db.connect"""

    @patch.dict(os.environ, {
        'SERVER': 'test_server',
        'DATABASE': 'test_database',
        'USER': 'test_user',
        'PASSWORD': 'test_password'
    })
    @patch('src.db.connect.create_engine')
    def test_get_engine_success(self, mock_create_engine):
        """Test creazione engine con credenziali valide."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_create_engine.return_value = mock_engine
        
        engine = get_engine()
        
        # Verifica che create_engine sia stato chiamato con la connection string corretta
        mock_create_engine.assert_called_once()
        connection_string = mock_create_engine.call_args[0][0]
        
        assert 'test_server' in connection_string
        assert 'test_database' in connection_string
        assert 'test_user' in connection_string
        assert 'test_password' in connection_string
        assert 'ODBC+Driver+18+for+SQL+Server' in connection_string
        assert engine == mock_engine

    @patch.dict(os.environ, {
        'SERVER': 'test_server',
        'DATABASE': '',  # Missing database
        'USER': 'test_user',
        'PASSWORD': 'test_password'
    })
    def test_get_engine_missing_credentials(self):
        """Test gestione credenziali mancanti."""
        with pytest.raises(ValueError, match="Variabili d'ambiente mancanti"):
            get_engine()

    @patch.dict(os.environ, {
        'SERVER': 'test_server',
        'DATABASE': 'test_database',
        'USER': 'test_user',
        'PASSWORD': 'test@password#123'  # Password con caratteri speciali
    })
    @patch('src.db.connect.create_engine')
    def test_get_engine_special_characters_in_password(self, mock_create_engine):
        """Test encoding password con caratteri speciali."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_create_engine.return_value = mock_engine
        
        engine = get_engine()
        
        # Verifica che la password sia stata encoded correttamente
        connection_string = mock_create_engine.call_args[0][0]
        assert 'test%40password%23123' in connection_string  # URL encoded

    @patch('src.db.connect.create_engine')
    def test_get_engine_connection_failure(self, mock_create_engine):
        """Test gestione fallimento connessione."""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        with patch.dict(os.environ, {
            'SERVER': 'test_server',
            'DATABASE': 'test_database',
            'USER': 'test_user',
            'PASSWORD': 'test_password'
        }):
            with pytest.raises(Exception, match="Connection failed"):
                get_engine()

    @patch('src.db.connect.get_engine')
    def test_test_connection_success(self, mock_get_engine):
        """Test connessione di test riuscita."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = [1]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_get_engine.return_value = mock_engine
        
        result = test_connection()
        
        assert result is True
        mock_connection.execute.assert_called_once_with("SELECT 1 as test")

    @patch('src.db.connect.get_engine')
    def test_test_connection_failure(self, mock_get_engine):
        """Test connessione di test fallita."""
        mock_get_engine.side_effect = Exception("Connection error")
        
        result = test_connection()
        
        assert result is False

    @patch('src.db.connect.get_engine')
    def test_test_connection_unexpected_result(self, mock_get_engine):
        """Test connessione con risultato inaspettato."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = [2]  # Valore inaspettato
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_get_engine.return_value = mock_engine
        
        result = test_connection()
        
        assert result is False


class TestDatabaseRetrieve:
    """Test per il modulo src.db.retrieve"""

    def test_build_select_clause_dual_omi_basic(self, sample_schema):
        """Test costruzione clausola SELECT base."""
        selected_aliases = ['A', 'AI']
        
        select_clause = build_select_clause_dual_omi(sample_schema, selected_aliases)
        
        # Verifica che vengano incluse solo le tabelle selezionate
        assert 'A.Id AS A_Id' in select_clause
        assert 'AI.IdCategoriaCatastale AS AI_IdCategoriaCatastale' in select_clause
        
        # Verifica formato separato da virgole e newline
        assert ',\n ' in select_clause

    def test_build_select_clause_dual_omi_no_aliases(self, sample_schema):
        """Test costruzione clausola SELECT senza filtro alias."""
        select_clause = build_select_clause_dual_omi(sample_schema, None)
        
        # Dovrebbe includere tutte le tabelle
        assert 'A.Id AS A_Id' in select_clause
        assert 'AI.IdCategoriaCatastale AS AI_IdCategoriaCatastale' in select_clause

    def test_build_select_clause_dual_omi_skip_non_retrieve(self):
        """Test che le colonne con retrieve=False vengano saltate."""
        schema_with_non_retrieve = {
            "TestTable": {
                "alias": "TT",
                "columns": [
                    {"name": "Id", "type": "int", "retrieve": True},
                    {"name": "SkipMe", "type": "varchar", "retrieve": False},
                    {"name": "IncludeMe", "type": "varchar", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(schema_with_non_retrieve, ['TT'])
        
        assert 'TT.Id AS TT_Id' in select_clause
        assert 'TT.IncludeMe AS TT_IncludeMe' in select_clause
        assert 'SkipMe' not in select_clause

    def test_build_select_clause_dual_omi_geometry_columns(self):
        """Test gestione colonne geometry/geography."""
        schema_with_geometry = {
            "Geometries": {
                "alias": "GEO",
                "columns": [
                    {"name": "Id", "type": "int", "retrieve": True},
                    {"name": "Shape", "type": "geometry", "retrieve": True},
                    {"name": "Location", "type": "geography", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(schema_with_geometry, ['GEO'])
        
        assert 'GEO.Id AS GEO_Id' in select_clause
        assert 'GEO.Shape.STAsText() AS GEO_Shape' in select_clause
        assert 'GEO.Location.STAsText() AS GEO_Location' in select_clause

    def test_build_select_clause_dual_omi_ov_special_handling(self):
        """Test gestione speciale per tabella OV (OmiValori)."""
        schema_with_ov = {
            "OmiValori": {
                "alias": "OV",
                "columns": [
                    {"name": "Id", "type": "int", "retrieve": True},
                    {"name": "Valore", "type": "float", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(schema_with_ov, ['OV'])
        
        # Dovrebbe generare tre varianti per ogni colonna OV
        assert 'OVN.Id AS OV_Id_normale' in select_clause
        assert 'OVO.Id AS OV_Id_ottimo' in select_clause
        assert 'OVS.Id AS OV_Id_scadente' in select_clause
        assert 'OVN.Valore AS OV_Valore_normale' in select_clause
        assert 'OVO.Valore AS OV_Valore_ottimo' in select_clause
        assert 'OVS.Valore AS OV_Valore_scadente' in select_clause

    def test_get_poi_categories_query(self):
        """Test query per categorie POI."""
        query = get_poi_categories_query()
        
        assert 'SELECT DISTINCT Id, Denominazione' in query
        assert 'FROM PuntiDiInteresseTipologie' in query
        assert 'ORDER BY Denominazione' in query

    def test_generate_poi_counts_subquery(self):
        """Test generazione subquery conteggi POI."""
        subquery = generate_poi_counts_subquery()
        
        # Verifica elementi chiave della subquery
        assert 'POI_COUNTS AS' in subquery
        assert 'PC_MAIN.Id as IdParticella' in subquery
        assert 'COUNT(PDI.Id) as ConteggioPOI' in subquery
        assert 'PC_MAIN.Isodistanza.STContains' in subquery
        assert 'GROUP BY' in subquery

    def test_generate_ztl_subquery(self):
        """Test generazione subquery ZTL."""
        subquery = generate_ztl_subquery()
        
        # Verifica elementi chiave della subquery
        assert 'ZTL_CHECK AS' in subquery
        assert 'PC_MAIN.Id as IdParticella' in subquery
        assert 'CASE' in subquery

    def test_build_select_clause_dual_omi_empty_schema(self):
        """Test con schema vuoto."""
        empty_schema = {}
        
        select_clause = build_select_clause_dual_omi(empty_schema, [])
        
        assert select_clause == ""

    def test_build_select_clause_dual_omi_filtered_aliases(self, sample_schema):
        """Test filtro alias che esclude alcune tabelle."""
        selected_aliases = ['A']  # Solo tabella Atti
        
        select_clause = build_select_clause_dual_omi(sample_schema, selected_aliases)
        
        # Dovrebbe includere solo colonne della tabella A
        assert 'A.Id AS A_Id' in select_clause
        assert 'A.Codice AS A_Codice' in select_clause
        assert 'AI_' not in select_clause  # Non dovrebbe includere colonne AI


class TestDatabaseIntegration:
    """Test di integrazione per moduli database."""

    @patch('src.db.retrieve.get_engine')
    @patch('src.db.retrieve.load_json')
    def test_retrieve_data_mock_integration(self, mock_load_json, mock_get_engine, sample_schema):
        """Test integrazione retrieve_data con mock."""
        # Setup mocks
        mock_load_json.return_value = sample_schema
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_get_engine.return_value = mock_engine
        
        # Mock del risultato della query
        mock_result_df = pd.DataFrame({
            'A_Id': [1, 2],
            'A_Codice': ['ACT001', 'ACT002'],
            'AI_IdCategoriaCatastale': ['A/2', 'A/3']
        })
        
        with patch('pandas.read_sql', return_value=mock_result_df) as mock_read_sql:
            with patch('src.db.retrieve.save_dataframe') as mock_save_df:
                from src.db.retrieve import retrieve_data
                
                # Esegui retrieve_data
                result = retrieve_data('test_schema.json', ['A', 'AI'], 'output.parquet')
                
                # Verifica chiamate
                mock_load_json.assert_called_once_with('test_schema.json')
                mock_save_df.assert_called_once()
                
                # Verifica che read_sql sia stato chiamato con query corretta
                mock_read_sql.assert_called_once()
                query_used = mock_read_sql.call_args[0][0]
                assert 'SELECT' in query_used
                assert 'A.Id AS A_Id' in query_used

    def test_schema_parsing_edge_cases(self):
        """Test parsing schema con casi edge."""
        edge_case_schema = {
            "TableWithMixedTypes": {
                "alias": "MIX",
                "columns": [
                    {"name": "Id", "type": "int", "retrieve": True},
                    {"name": "JsonData", "type": "nvarchar(max)", "retrieve": True},
                    {"name": "BinaryData", "type": "varbinary", "retrieve": False},
                    {"name": "GeometryCol", "type": "geometry", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(edge_case_schema, ['MIX'])
        
        # Verifica gestione tipi misti
        assert 'MIX.Id AS MIX_Id' in select_clause
        assert 'MIX.JsonData AS MIX_JsonData' in select_clause
        assert 'BinaryData' not in select_clause  # retrieve=False
        assert 'MIX.GeometryCol.STAsText() AS MIX_GeometryCol' in select_clause

    @patch.dict(os.environ, {}, clear=True)  # Rimuovi tutte le variabili d'ambiente
    def test_connection_without_environment_variables(self):
        """Test comportamento senza variabili d'ambiente."""
        with pytest.raises(ValueError, match="Variabili d'ambiente mancanti"):
            get_engine()

    def test_build_select_clause_consistency(self, sample_schema):
        """Test consistenza nella generazione della clausola SELECT."""
        # Genera la clausola più volte per verificare consistenza
        select_clause_1 = build_select_clause_dual_omi(sample_schema, ['A', 'AI'])
        select_clause_2 = build_select_clause_dual_omi(sample_schema, ['A', 'AI'])
        
        assert select_clause_1 == select_clause_2

    def test_alias_generation_consistency(self):
        """Test consistenza nella generazione degli alias."""
        schema_no_explicit_alias = {
            "TableName": {
                # Nessun alias esplicito
                "columns": [
                    {"name": "Id", "type": "int", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(schema_no_explicit_alias, None)
        
        # Dovrebbe usare prime due lettere del nome tabella come alias
        assert 'TA.Id AS TA_Id' in select_clause

    def test_special_characters_in_column_names(self):
        """Test gestione caratteri speciali nei nomi colonne."""
        schema_special_chars = {
            "SpecialTable": {
                "alias": "SP",
                "columns": [
                    {"name": "Column_With_Underscore", "type": "varchar", "retrieve": True},
                    {"name": "Column-With-Dash", "type": "varchar", "retrieve": True},
                    {"name": "Column With Space", "type": "varchar", "retrieve": True}
                ]
            }
        }
        
        select_clause = build_select_clause_dual_omi(schema_special_chars, ['SP'])
        
        # Verifica che i nomi con caratteri speciali vengano gestiti
        assert 'SP.Column_With_Underscore AS SP_Column_With_Underscore' in select_clause
        assert 'SP.Column-With-Dash AS SP_Column-With-Dash' in select_clause
        assert 'SP.Column With Space AS SP_Column With Space' in select_clause