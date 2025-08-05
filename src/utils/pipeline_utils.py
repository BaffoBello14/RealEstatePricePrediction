"""
Utility functions per la pipeline di ML per eliminare duplicazione di codice.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .logger import get_logger
from .io import load_dataframe, save_dataframe, check_file_exists

logger = get_logger(__name__)

class PathManager:
    """Gestisce la costruzione e validazione dei path per la pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il manager dei path.
        
        Args:
            config: Configurazione della pipeline
        """
        self.config = config
        self.paths = config.get('paths', {})
        
    def get_data_paths(self) -> Dict[str, str]:
        """
        Ottiene tutti i path base per i dati.
        
        Returns:
            Dictionary con i path base
        """
        return {
            'data_raw': self.paths.get('data_raw', 'data/raw/'),
            'data_processed': self.paths.get('data_processed', 'data/processed/'),
            'models': self.paths.get('models', 'models/'),
            'logs': self.paths.get('logs', 'logs/')
        }
    
    def get_raw_data_path(self) -> str:
        """Ottiene il path per i dati grezzi."""
        return f"{self.paths.get('data_raw', 'data/raw/')}raw_data.parquet"
    
    def get_dataset_path(self) -> str:
        """Ottiene il path per il dataset processato."""
        return f"{self.paths.get('data_processed', 'data/processed/')}dataset.parquet"
    
    def get_preprocessing_paths(self) -> Dict[str, str]:
        """
        Ottiene tutti i path per i file di preprocessing.
        
        Returns:
            Dictionary con tutti i path di preprocessing
        """
        data_processed = self.paths.get('data_processed', 'data/processed/')
        
        return {
            'train_features': f"{data_processed}X_train.parquet",
            'val_features': f"{data_processed}X_val.parquet",
            'test_features': f"{data_processed}X_test.parquet",
            'train_target': f"{data_processed}y_train.parquet",
            'val_target': f"{data_processed}y_val.parquet",
            'test_target': f"{data_processed}y_test.parquet",
            'val_target_orig': f"{data_processed}y_val_orig.parquet",
            'test_target_orig': f"{data_processed}y_test_orig.parquet",
            'preprocessing_info': f"{data_processed}preprocessing_info.json"
        }
    
    def get_evaluation_paths(self) -> Dict[str, str]:
        """
        Ottiene i path per i file di evaluation.
        
        Returns:
            Dictionary con i path di evaluation
        """
        return {
            'results_dir': self.paths.get('logs', 'logs/'),
            'feature_importance_path': f"{self.paths.get('data_processed', 'data/processed/')}feature_importance.csv",
            'evaluation_summary_path': f"{self.paths.get('data_processed', 'data/processed/')}evaluation_summary.json"
        }

class ConfigManager:
    """Gestisce l'accesso centralizzato alla configurazione."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il manager della configurazione.
        
        Args:
            config: Configurazione della pipeline
        """
        self.config = config
        self._cache = {}
    
    def get_database_config(self) -> Dict[str, Any]:
        """Ottiene la configurazione del database."""
        if 'database' not in self._cache:
            self._cache['database'] = self.config.get('database', {})
        return self._cache['database']
    
    def get_target_config(self) -> Dict[str, Any]:
        """Ottiene la configurazione del target."""
        if 'target' not in self._cache:
            self._cache['target'] = self.config.get('target', {})
        return self._cache['target']
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Ottiene la configurazione del preprocessing."""
        if 'preprocessing' not in self._cache:
            self._cache['preprocessing'] = self.config.get('preprocessing', {})
        return self._cache['preprocessing']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Ottiene la configurazione del training."""
        if 'training' not in self._cache:
            self._cache['training'] = self.config.get('training', {})
        return self._cache['training']
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Ottiene la configurazione dell'esecuzione."""
        if 'execution' not in self._cache:
            self._cache['execution'] = self.config.get('execution', {})
        return self._cache['execution']
    
    def should_force_reload(self) -> bool:
        """Verifica se forzare il ricaricamento."""
        return self.get_execution_config().get('force_reload', False)
    
    def get_target_column(self) -> str:
        """Ottiene il nome della colonna target."""
        return self.get_target_config().get('column', 'AI_Prezzo_Ridistribuito')
    
    def get_steps_to_run(self, args_steps: Optional[List[str]] = None) -> List[str]:
        """
        Ottiene gli step da eseguire.
        
        Args:
            args_steps: Step specificati da argomenti comando
            
        Returns:
            Lista degli step da eseguire
        """
        if args_steps:
            return args_steps
        return self.get_execution_config().get('steps', [])

class FileManager:
    """Gestisce le operazioni sui file e le verifiche di esistenza."""
    
    @staticmethod
    def check_files_exist(file_paths: Dict[str, str]) -> bool:
        """
        Verifica se tutti i file esistono.
        
        Args:
            file_paths: Dictionary con i path dei file
            
        Returns:
            True se tutti i file esistono
        """
        return all(check_file_exists(path) for path in file_paths.values())
    
    @staticmethod
    def check_files_exist_with_details(file_paths: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
        """
        Verifica esistenza file con dettagli.
        
        Args:
            file_paths: Dictionary con i path dei file
            
        Returns:
            Tuple con (tutti_esistono, file_esistenti, file_mancanti)
        """
        existing_files = []
        missing_files = []
        
        for name, path in file_paths.items():
            if check_file_exists(path):
                existing_files.append(name)
            else:
                missing_files.append(name)
        
        all_exist = len(missing_files) == 0
        return all_exist, existing_files, missing_files
    
    @staticmethod
    def log_file_status(file_paths: Dict[str, str], operation_name: str) -> bool:
        """
        Logga lo stato dei file e ritorna se tutti esistono.
        
        Args:
            file_paths: Dictionary con i path dei file
            operation_name: Nome dell'operazione per il log
            
        Returns:
            True se tutti i file esistono
        """
        all_exist, existing, missing = FileManager.check_files_exist_with_details(file_paths)
        
        if all_exist:
            logger.info(f"{operation_name}: Tutti i file esistono già")
        else:
            logger.info(f"{operation_name}: File mancanti: {missing}")
            if existing:
                logger.info(f"{operation_name}: File esistenti: {existing}")
        
        return all_exist

class DataLoader:
    """Gestisce il caricamento centralizzato dei dati."""
    
    @staticmethod
    def load_multiple_dataframes(file_paths: Dict[str, str], 
                                required_files: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Carica multipli DataFrame con gestione errori.
        
        Args:
            file_paths: Dictionary con i path dei file
            required_files: Lista dei file obbligatori (se None, tutti sono obbligatori)
            
        Returns:
            Dictionary con i DataFrame caricati
        """
        dataframes = {}
        errors = []
        
        if required_files is None:
            required_files = list(file_paths.keys())
        
        for name, path in file_paths.items():
            try:
                if check_file_exists(path):
                    dataframes[name] = load_dataframe(path)
                    logger.info(f"Caricato {name}: {dataframes[name].shape}")
                elif name in required_files:
                    errors.append(f"File obbligatorio mancante: {name} ({path})")
                else:
                    logger.warning(f"File opzionale mancante: {name} ({path})")
            except Exception as e:
                error_msg = f"Errore caricamento {name}: {e}"
                if name in required_files:
                    errors.append(error_msg)
                else:
                    logger.warning(error_msg)
        
        if errors:
            raise FileNotFoundError(f"Errori nel caricamento dati: {'; '.join(errors)}")
        
        return dataframes
    
    @staticmethod
    def save_multiple_dataframes(dataframes: Dict[str, pd.DataFrame], 
                                file_paths: Dict[str, str],
                                format: str = 'parquet') -> None:
        """
        Salva multipli DataFrame in batch.
        
        Args:
            dataframes: Dictionary con i DataFrame da salvare
            file_paths: Dictionary con i path dei file
            format: Formato di salvataggio
        """
        saved_count = 0
        errors = []
        
        for name, df in dataframes.items():
            if name in file_paths and df is not None:
                try:
                    save_dataframe(df, file_paths[name], format)
                    saved_count += 1
                except Exception as e:
                    errors.append(f"Errore salvataggio {name}: {e}")
            elif df is None:
                logger.info(f"Skipping salvataggio {name}: DataFrame è None")
            else:
                logger.warning(f"Path non trovato per {name}")
        
        if errors:
            logger.error(f"Errori nel salvataggio: {'; '.join(errors)}")
        else:
            logger.info(f"Salvati con successo {saved_count} DataFrame")
    
    @staticmethod
    def load_preprocessing_data(preprocessing_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Carica tutti i dati di preprocessing con nomi standardizzati.
        
        Args:
            preprocessing_paths: Path dei file di preprocessing
            
        Returns:
            Dictionary con tutti i dati caricati
        """
        # File delle features (obbligatori)
        feature_paths = {
            'X_train': preprocessing_paths['train_features'],
            'X_val': preprocessing_paths['val_features'],
            'X_test': preprocessing_paths['test_features']
        }
        
        # File dei target (obbligatori)
        target_paths = {
            'y_train': preprocessing_paths['train_target'],
            'y_val': preprocessing_paths['val_target'],
            'y_test': preprocessing_paths['test_target']
        }
        
        # File dei target originali (opzionali)
        target_orig_paths = {
            'y_val_orig': preprocessing_paths.get('val_target_orig'),
            'y_test_orig': preprocessing_paths.get('test_target_orig')
        }
        
        # Carica features e target (obbligatori)
        features = DataLoader.load_multiple_dataframes(feature_paths)
        targets = DataLoader.load_multiple_dataframes(target_paths)
        
        # Carica target originali (opzionali)
        targets_orig = DataLoader.load_multiple_dataframes(
            target_orig_paths, 
            required_files=[]  # Nessun file obbligatorio
        )
        
        # Carica info preprocessing se disponibile
        preprocessing_info = None
        if 'preprocessing_info' in preprocessing_paths:
            info_path = preprocessing_paths['preprocessing_info']
            if check_file_exists(info_path):
                from .io import load_json
                preprocessing_info = load_json(info_path)
        
        # Combina tutto
        result = {**features, **targets, **targets_orig}
        if preprocessing_info:
            result['preprocessing_info'] = preprocessing_info
        
        logger.info(f"Dati preprocessing caricati: {list(result.keys())}")
        return result

def create_pipeline_managers(config: Dict[str, Any]) -> Tuple[PathManager, ConfigManager, FileManager]:
    """
    Crea tutti i manager della pipeline.
    
    Args:
        config: Configurazione della pipeline
        
    Returns:
        Tuple con (PathManager, ConfigManager, FileManager)
    """
    path_manager = PathManager(config)
    config_manager = ConfigManager(config)
    file_manager = FileManager()
    
    return path_manager, config_manager, file_manager