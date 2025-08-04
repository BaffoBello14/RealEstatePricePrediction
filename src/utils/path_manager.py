"""
Modulo centralizzato per la gestione dei path del progetto.
Elimina duplicazioni e standardizza la gestione dei path in tutta l'applicazione.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from .logger import get_logger

logger = get_logger(__name__)


class PathManager:
    """
    Gestore centralizzato per tutti i path del progetto.
    Elimina duplicazioni e fornisce un'interfaccia consistente per gestire i path.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il PathManager con la configurazione.
        
        Args:
            config: Configurazione del progetto contenente sezioni paths, database, etc.
        """
        self.config = config
        self.paths_config = config.get('paths', {})
        self.db_config = config.get('database', {})
        
        # Path di base dal config
        self.data_raw = self.paths_config.get('data_raw', 'data/raw/')
        self.data_processed = self.paths_config.get('data_processed', 'data/processed/')
        self.models = self.paths_config.get('models', 'models/')
        self.logs = self.paths_config.get('logs', 'logs/')
        
        logger.info(f"🗂️  PathManager inizializzato con path base: raw={self.data_raw}, processed={self.data_processed}")
    
    def get_raw_data_path(self, filename: str = "raw_data.parquet") -> str:
        """
        Ottiene il path per i dati grezzi.
        
        Args:
            filename: Nome del file (default: raw_data.parquet)
            
        Returns:
            Path completo del file dati grezzi
        """
        return str(Path(self.data_raw) / filename)
    
    def get_dataset_path(self, filename: str = "dataset.parquet") -> str:
        """
        Ottiene il path per il dataset processato.
        
        Args:
            filename: Nome del file (default: dataset.parquet)
            
        Returns:
            Path completo del file dataset
        """
        return str(Path(self.data_processed) / filename)
    
    def get_preprocessing_paths(self) -> Dict[str, str]:
        """
        Ottiene tutti i path per i file di preprocessing.
        Elimina duplicazione della costruzione path in main.py.
        
        Returns:
            Dictionary con tutti i path per preprocessing output
        """
        processed_dir = Path(self.data_processed)
        
        paths = {
            # Feature sets per train/val/test
            'train_features': str(processed_dir / "X_train.parquet"),
            'val_features': str(processed_dir / "X_val.parquet"),
            'test_features': str(processed_dir / "X_test.parquet"),
            
            # Target sets per train/val/test
            'train_target': str(processed_dir / "y_train.parquet"),
            'val_target': str(processed_dir / "y_val.parquet"),
            'test_target': str(processed_dir / "y_test.parquet"),
            
            # Target originali per validation e test
            'val_target_orig': str(processed_dir / "y_val_orig.parquet"),
            'test_target_orig': str(processed_dir / "y_test_orig.parquet"),
            
            # Metadati preprocessing
            'preprocessing_info': str(processed_dir / "preprocessing_info.json")
        }
        
        logger.debug(f"📋 Generati {len(paths)} path di preprocessing")
        return paths
    
    def get_evaluation_paths(self) -> Dict[str, str]:
        """
        Ottiene i path per i file di evaluation.
        
        Returns:
            Dictionary con i path per evaluation output
        """
        paths = {
            'results_dir': self.logs,
            'feature_importance_path': str(Path(self.data_processed) / "feature_importance.csv"),
            'evaluation_summary_path': str(Path(self.data_processed) / "evaluation_summary.json")
        }
        
        logger.debug(f"📊 Generati {len(paths)} path di evaluation")
        return paths
    
    def get_database_schema_path(self) -> str:
        """
        Ottiene il path per lo schema del database.
        
        Returns:
            Path del file schema database
        """
        return self.db_config.get('schema_path', 'data/db_schema.json')
    
    def get_model_path(self, model_name: str, extension: str = ".pkl") -> str:
        """
        Ottiene il path per salvare un modello.
        
        Args:
            model_name: Nome del modello
            extension: Estensione del file (default: .pkl)
            
        Returns:
            Path completo per il file modello
        """
        filename = f"{model_name}{extension}"
        return str(Path(self.models) / filename)
    
    def get_log_path(self, log_name: str = "pipeline.log") -> str:
        """
        Ottiene il path per i file di log.
        
        Args:
            log_name: Nome del file di log
            
        Returns:
            Path completo del file di log
        """
        return str(Path(self.logs) / log_name)
    
    def ensure_all_directories(self) -> None:
        """
        Crea tutte le directory necessarie per il progetto.
        Sostituisce la funzione setup_directories in main.py.
        """
        logger.info("📁 Creazione/verifica directory del progetto...")
        
        directories_to_create = [
            self.data_raw,
            self.data_processed, 
            self.models,
            self.logs
        ]
        
        created_count = 0
        for directory in directories_to_create:
            path = Path(directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_count += 1
                logger.info(f"  ➕ Creata directory: {directory}")
            else:
                logger.debug(f"  ✅ Directory esistente: {directory}")
        
        if created_count > 0:
            logger.info(f"🎯 Completato: {created_count} nuove directory create")
        else:
            logger.info("✅ Tutte le directory già esistenti")
    
    def validate_paths_exist(self, paths: List[str]) -> Dict[str, bool]:
        """
        Valida che una lista di path esista.
        
        Args:
            paths: Lista di path da validare
            
        Returns:
            Dictionary con path -> bool (True se esiste)
        """
        logger.info(f"🔍 Validazione esistenza di {len(paths)} path...")
        
        results = {}
        existing_count = 0
        
        for path in paths:
            exists = Path(path).exists()
            results[path] = exists
            if exists:
                existing_count += 1
            else:
                logger.warning(f"  ❌ Path mancante: {path}")
        
        logger.info(f"📊 Validazione completata: {existing_count}/{len(paths)} path esistenti")
        return results
    
    def get_all_preprocessing_paths_for_step(self, step_name: str) -> Optional[Dict[str, str]]:
        """
        Ottiene i path necessari per uno step specifico della pipeline.
        Utile per riavvii parziali della pipeline.
        
        Args:
            step_name: Nome dello step ('retrieve_data', 'build_dataset', 'preprocessing', etc.)
            
        Returns:
            Dictionary con i path necessari per lo step, None se step non riconosciuto
        """
        if step_name == 'retrieve_data':
            return {
                'output': self.get_raw_data_path()
            }
        elif step_name == 'build_dataset':
            return {
                'input': self.get_raw_data_path(),
                'output': self.get_dataset_path()
            }
        elif step_name == 'preprocessing':
            return {
                'input': self.get_dataset_path(),
                **self.get_preprocessing_paths()
            }
        elif step_name == 'training':
            return self.get_preprocessing_paths()
        elif step_name == 'evaluation':
            return {
                **self.get_preprocessing_paths(),
                **self.get_evaluation_paths()
            }
        else:
            logger.warning(f"⚠️  Step '{step_name}' non riconosciuto")
            return None
    
    def __repr__(self) -> str:
        """Rappresentazione string del PathManager per debugging."""
        return (f"PathManager(raw={self.data_raw}, processed={self.data_processed}, "
               f"models={self.models}, logs={self.logs})")


def create_path_manager(config: Dict[str, Any]) -> PathManager:
    """
    Factory function per creare un PathManager.
    
    Args:
        config: Configurazione del progetto
        
    Returns:
        Istanza di PathManager configurata
    """
    return PathManager(config)