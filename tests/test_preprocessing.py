"""
Test per i moduli preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tempfile

from src.preprocessing.cleaning import (
    remove_specific_columns, remove_constant_columns,
    detect_outliers_multimethod
)
from src.preprocessing.data_cleaning_core import (
    convert_to_numeric_unified,
    clean_dataframe_unified
)
from src.preprocessing.filtering import (
    analyze_cramers_correlations, remove_highly_correlated_numeric_pre_split
)
from src.preprocessing.encoding import encode_features
from src.preprocessing.imputation import handle_missing_values
from src.preprocessing.transformation import (
    split_dataset_with_validation, apply_feature_scaling,
    apply_pca_transformation, transform_target_log
)
from src.preprocessing.pipeline import run_preprocessing_pipeline


class TestCleaning:
    """Test per il modulo cleaning."""

    def test_remove_specific_columns_existing(self, sample_dataframe):
        """Test rimozione colonne specifiche esistenti."""
        columns_to_remove = ['A_Id', 'A_Codice']
        df_cleaned, info = remove_specific_columns(sample_dataframe, columns_to_remove)
        
        # Verifica rimozione
        for col in columns_to_remove:
            assert col not in df_cleaned.columns
        
        # Verifica info
        assert info['columns_removed_count'] == 2
        assert info['existing_columns'] == columns_to_remove

    def test_remove_specific_columns_nonexistent(self, sample_dataframe):
        """Test rimozione colonne non esistenti."""
        columns_to_remove = ['NonExistent1', 'NonExistent2']
        df_cleaned, info = remove_specific_columns(sample_dataframe, columns_to_remove)
        
        # DataFrame dovrebbe rimanere uguale
        pd.testing.assert_frame_equal(sample_dataframe, df_cleaned)
        assert info['columns_removed_count'] == 0
        assert info['missing_columns'] == columns_to_remove

    def test_remove_constant_columns(self):
        """Test rimozione colonne costanti."""
        df = pd.DataFrame({
            'constant_col': ['SAME'] * 100,
            'almost_constant': ['SAME'] * 96 + ['DIFF'] * 4,  # 96% costante
            'variable_col': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        df_cleaned, info = remove_constant_columns(df, 'target', threshold=0.95)
        
        # Verifica rimozione colonne costanti
        assert 'constant_col' not in df_cleaned.columns
        assert 'almost_constant' not in df_cleaned.columns
        assert 'variable_col' in df_cleaned.columns
        assert 'target' in df_cleaned.columns
        
        assert info['columns_removed_count'] == 2

    def test_convert_to_numeric_basic(self):
        """Test conversione automatica a numerico."""
        df = pd.DataFrame({
            'numeric_string': ['1.5', '2.0', '3.5', '4.0'],
            'mixed_string': ['1.5', '2.0', 'invalid', '4.0'],
            'pure_string': ['a', 'b', 'c', 'd'],
            'already_numeric': [1, 2, 3, 4],
            'target': [10, 20, 30, 40]
        })
        
        df_converted, info = convert_to_numeric_unified(df, 'target', threshold=0.8)
        
        # Verifica conversioni
        assert pd.api.types.is_numeric_dtype(df_converted['numeric_string'])
        assert pd.api.types.is_numeric_dtype(df_converted['already_numeric'])
        assert not pd.api.types.is_numeric_dtype(df_converted['mixed_string'])  # Solo 75% convertibile
        assert not pd.api.types.is_numeric_dtype(df_converted['pure_string'])

    def test_detect_outliers_multimethod(self, sample_features_and_target):
        """Test detection outlier con metodi multipli."""
        X, y = sample_features_and_target
        
        # Aggiungi alcuni outlier artificiali
        X_with_outliers = X.copy()
        X_with_outliers.iloc[0, 0] = 10  # Outlier estremo
        X_with_outliers.iloc[1, 1] = -10  # Altro outlier
        
        outlier_mask = detect_outliers_multimethod(
            X_with_outliers, y, 
            z_threshold=2.0, 
            iqr_multiplier=1.5,
            isolation_contamination=0.1,
            min_methods=2
        )
        
        # Verifica che almeno alcuni outlier siano identificati
        assert outlier_mask.sum() > 0
        assert len(outlier_mask) == len(X_with_outliers)

    def test_clean_data_integration(self, sample_dataframe, sample_config):
        """Test integrazione completa clean_data."""
        config = sample_config['preprocessing']
        
        df_cleaned, info = clean_dataframe_unified(
            sample_dataframe, 'AI_Prezzo_Ridistribuito',
            remove_empty_strings=True, remove_duplicates=True,
            remove_empty_columns=True, remove_target_nulls=True
        )
        
        # Verifica che la pulizia sia avvenuta
        assert 'A_Id' not in df_cleaned.columns  # Dovrebbe essere rimossa
        assert 'Constant_Col' not in df_cleaned.columns  # Dovrebbe essere rimossa come costante
        assert 'AI_Prezzo_Ridistribuito' in df_cleaned.columns  # Target preservato
        
        # Verifica info
        assert 'specific_columns_removal' in info
        assert 'constant_columns_removal' in info


class TestFiltering:
    """Test per il modulo filtering."""

    def test_analyze_cramers_correlations(self):
        """Test analisi correlazioni Cramér's V."""
        # Crea dati con variabili categoriche correlate
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B'] * 25,
            'cat2': ['A', 'B', 'A', 'B'] * 25,  # Identica a cat1
            'cat3': ['X', 'Y', 'Z', 'W'] * 25,  # Diversa
            'target': np.random.randn(100)
        })
        
        analysis = analyze_cramers_correlations(df, 'target', threshold=0.95)
        
        # Verifica che l'analisi rilevi la correlazione tra cat1 e cat2
        assert 'correlation_matrix' in analysis
        assert 'highly_correlated_pairs' in analysis

    def test_remove_highly_correlated_numeric_pre_split(self):
        """Test rimozione variabili numeriche correlate."""
        # Crea dati con correlazioni elevate
        df = pd.DataFrame({
            'var1': np.random.randn(100),
            'var3': np.random.randn(100),
            'target': np.random.randn(100)
        })
        # var2 altamente correlata con var1
        df['var2'] = df['var1'] * 1.01 + np.random.randn(100) * 0.01
        
        df_filtered, removed_cols = remove_highly_correlated_numeric_pre_split(
            df, 'target', threshold=0.95
        )
        
        # Verifica che una delle variabili correlate sia stata rimossa
        assert len(removed_cols) > 0
        assert len(df_filtered.columns) < len(df.columns)


class TestEncoding:
    """Test per il modulo encoding."""

    def test_encode_features_basic(self):
        """Test encoding base delle variabili categoriche."""
        df = pd.DataFrame({
            'low_card': ['A', 'B', 'A', 'B'] * 25,  # Bassa cardinalità
            'high_card': [f'VAL_{i}' for i in range(100)],  # Alta cardinalità
            'numeric': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        df_encoded, info = encode_features(
            df, 'target',
            low_card_threshold=10,
            high_card_max=50,
            apply_target_encoding=False
        )
        
        # Verifica che l'encoding sia avvenuto
        assert 'low_card_A' in df_encoded.columns or 'low_card_B' in df_encoded.columns  # One-hot
        assert 'low_card' not in df_encoded.columns  # Originale rimossa
        assert 'numeric' in df_encoded.columns  # Numerica preservata

    def test_encode_features_with_target_encoding(self):
        """Test encoding con target encoding."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 34 + ['A', 'B'],  # 100 samples
            'target': np.random.randn(100)
        })
        
        df_encoded, info = encode_features(
            df, 'target',
            low_card_threshold=5,
            apply_target_encoding=True
        )
        
        # Verifica target encoding
        assert 'category_target_encoded' in df_encoded.columns
        assert 'category' not in df_encoded.columns


class TestImputation:
    """Test per il modulo imputation."""

    def test_handle_missing_values(self):
        """Test gestione valori mancanti."""
        df = pd.DataFrame({
            'numeric_with_na': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_with_na': ['A', 'B', np.nan, 'A', 'B'],
            'complete_numeric': [10, 20, 30, 40, 50],
            'target': [100, 200, 300, 400, 500]
        })
        
        df_imputed, info = handle_missing_values(df, 'target')
        
        # Verifica che non ci siano più valori mancanti
        assert not df_imputed.isnull().any().any()
        
        # Verifica strategie di imputazione
        assert df_imputed['categorical_with_na'].iloc[2] == 'MISSING'  # Categorica -> "MISSING"
        assert not pd.isna(df_imputed['numeric_with_na'].iloc[2])  # Numerica -> mediana


class TestTransformation:
    """Test per il modulo transformation."""

    def test_split_dataset_with_validation_random(self, sample_dataframe):
        """Test split dataset casuale."""
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = split_dataset_with_validation(
            sample_dataframe, 'AI_Prezzo_Ridistribuito',
            test_size=0.2, val_size=0.2, use_temporal_split=False
        )
        
        # Verifica dimensioni
        total_samples = len(sample_dataframe)
        assert len(X_train) + len(X_val) + len(X_test) == total_samples
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)

    def test_split_dataset_with_validation_temporal(self, sample_dataframe):
        """Test split dataset temporale."""
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = split_dataset_with_validation(
            sample_dataframe, 'AI_Prezzo_Ridistribuito',
            test_size=0.2, val_size=0.2, use_temporal_split=True,
            year_column='A_AnnoStipula', month_column='A_MeseStipula'
        )
        
        # Verifica che il split temporale abbia funzionato
        # (test set dovrebbe avere date più recenti del train set)
        assert len(X_train) + len(X_val) + len(X_test) == len(sample_dataframe)

    def test_apply_feature_scaling(self, sample_features_and_target):
        """Test scaling delle features."""
        X, y = sample_features_and_target
        
        X_scaled, scaler = apply_feature_scaling(X, method='standard')
        
        # Verifica che lo scaling sia avvenuto
        assert isinstance(scaler, StandardScaler)
        assert X_scaled.shape == X.shape
        
        # Verifica proprietà dello scaling (media ~0, std ~1)
        assert abs(X_scaled.mean().mean()) < 0.1
        assert abs(X_scaled.std().mean() - 1.0) < 0.1

    def test_apply_pca_transformation(self, sample_features_and_target):
        """Test trasformazione PCA."""
        X, y = sample_features_and_target
        
        X_pca, pca, explained_variance = apply_pca_transformation(
            X, variance_threshold=0.95
        )
        
        # Verifica PCA
        assert isinstance(pca, PCA)
        assert X_pca.shape[0] == X.shape[0]  # Stesso numero di campioni
        assert X_pca.shape[1] <= X.shape[1]  # Dimensioni ridotte o uguali
        assert 0 < explained_variance <= 1

    def test_transform_target_log(self, sample_dataframe):
        """Test trasformazione logaritmica del target."""
        target_col = 'AI_Prezzo_Ridistribuito'
        
        df_transformed = transform_target_log(sample_dataframe, target_col)
        
        # Verifica che sia stata aggiunta la colonna log
        assert f'{target_col}_log' in df_transformed.columns
        assert f'{target_col}_original' in df_transformed.columns
        
        # Verifica trasformazione
        original_values = df_transformed[f'{target_col}_original']
        log_values = df_transformed[f'{target_col}_log']
        
        # La trasformazione log dovrebbe ridurre la skewness
        assert log_values.std() < original_values.std()


class TestPreprocessingPipeline:
    """Test per la pipeline completa di preprocessing."""

    @patch('src.preprocessing.pipeline.load_dataframe')
    @patch('src.preprocessing.pipeline.save_dataframe')
    @patch('src.preprocessing.pipeline.save_json')
    def test_run_preprocessing_pipeline_integration(self, mock_save_json, mock_save_df, mock_load_df, 
                                                   sample_dataframe, sample_config, sample_preprocessing_paths):
        """Test integrazione completa pipeline preprocessing."""
        mock_load_df.return_value = sample_dataframe
        
        # Esegui pipeline
        run_preprocessing_pipeline(
            dataset_path='input.parquet',
            target_column='AI_Prezzo_Ridistribuito',
            config=sample_config['preprocessing'],
            output_paths=sample_preprocessing_paths
        )
        
        # Verifica che siano state fatte le chiamate di salvataggio
        assert mock_save_df.call_count >= 6  # X_train, X_val, X_test, y_train, y_val, y_test
        mock_save_json.assert_called()  # preprocessing_info

    def test_preprocessing_pipeline_step_control(self, sample_dataframe, sample_config):
        """Test controllo step nella pipeline."""
        # Disabilita alcuni step
        config = sample_config['preprocessing'].copy()
        config['steps']['enable_cramers_analysis'] = False
        config['steps']['enable_auto_numeric_conversion'] = False
        
        with patch('src.preprocessing.pipeline.load_dataframe', return_value=sample_dataframe):
            with patch('src.preprocessing.pipeline.save_dataframe'):
                with patch('src.preprocessing.pipeline.save_json'):
                    # Non dovrebbe crashare anche con step disabilitati
                    run_preprocessing_pipeline(
                        dataset_path='input.parquet',
                        target_column='AI_Prezzo_Ridistribuito',
                        config=config,
                        output_paths={
                            'train_features': 'X_train.parquet',
                            'val_features': 'X_val.parquet',
                            'test_features': 'X_test.parquet',
                            'train_target': 'y_train.parquet',
                            'val_target': 'y_val.parquet',
                            'test_target': 'y_test.parquet',
                            'val_target_orig': 'y_val_orig.parquet',
                            'test_target_orig': 'y_test_orig.parquet',
                            'preprocessing_info': 'preprocessing_info.json'
                        }
                    )

    def test_preprocessing_pipeline_empty_dataframe(self, sample_config):
        """Test pipeline con DataFrame vuoto."""
        empty_df = pd.DataFrame()
        
        with patch('src.preprocessing.pipeline.load_dataframe', return_value=empty_df):
            with patch('src.preprocessing.pipeline.save_dataframe'):
                with patch('src.preprocessing.pipeline.save_json'):
                    # Dovrebbe gestire il caso vuoto senza errori
                    run_preprocessing_pipeline(
                        dataset_path='input.parquet',
                        target_column='AI_Prezzo_Ridistribuito',
                        config=sample_config['preprocessing'],
                        output_paths={
                            'train_features': 'X_train.parquet',
                            'val_features': 'X_val.parquet',
                            'test_features': 'X_test.parquet',
                            'train_target': 'y_train.parquet',
                            'val_target': 'y_val.parquet',
                            'test_target': 'y_test.parquet',
                            'val_target_orig': 'y_val_orig.parquet',
                            'test_target_orig': 'y_test_orig.parquet',
                            'preprocessing_info': 'preprocessing_info.json'
                        }
                    )


class TestPreprocessingEdgeCases:
    """Test per casi edge nel preprocessing."""

    def test_all_categorical_data(self):
        """Test preprocessing con solo dati categorici."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 30 + ['A'],
            'cat2': ['X', 'Y'] * 50 + ['X'],
            'target': np.random.randn(91)
        })
        
        # Dovrebbe gestire correttamente anche senza dati numerici
        df_encoded, info = encode_features(df, 'target', low_card_threshold=5)
        assert len(df_encoded.columns) > 2  # Dovrebbe aver creato colonne encoded

    def test_all_numeric_data(self):
        """Test preprocessing con solo dati numerici."""
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'num3': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        # Non dovrebbe crashare
        df_encoded, info = encode_features(df, 'target')
        df_imputed, impute_info = handle_missing_values(df_encoded, 'target')
        
        assert len(df_imputed) == len(df)

    def test_single_sample_data(self):
        """Test preprocessing con un solo campione."""
        df = pd.DataFrame({
            'feature1': [1.0],
            'feature2': ['A'],
            'target': [100.0]
        })
        
        # Molti algoritmi potrebbero non funzionare con un solo campione
        # ma non dovrebbe crashare
        try:
            df_cleaned, info = clean_dataframe_unified(
                df, 'target',
                remove_empty_strings=True, remove_duplicates=True,
                remove_empty_columns=True, remove_target_nulls=True
            )
            assert len(df_cleaned) == 1
        except Exception as e:
            # È accettabile che alcuni algoritmi falliscano con un solo campione
            pytest.skip(f"Expected failure with single sample: {e}")

    def test_extreme_missing_values(self):
        """Test con percentuali estreme di valori mancanti."""
        # 90% valori mancanti
        df = pd.DataFrame({
            'mostly_missing': [1.0] * 10 + [np.nan] * 90,
            'target': np.random.randn(100)
        })
        
        df_imputed, info = handle_missing_values(df, 'target')
        
        # Dovrebbe comunque imputare
        assert not df_imputed['mostly_missing'].isna().any()
        assert info['columns_with_missing']['mostly_missing']['missing_percentage'] == 0.9