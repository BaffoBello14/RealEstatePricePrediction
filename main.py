"""
Entry point principale per la pipeline di Machine Learning.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logger, get_logger
from src.utils.io import load_config, ensure_dir
from src.utils.pipeline_utils import create_pipeline_managers, PathManager, ConfigManager, FileManager
from src.db.retrieve import retrieve_data
from src.dataset.build_dataset import build_dataset
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.training.train import run_training_pipeline
from src.training.evaluation import run_evaluation_pipeline

def setup_directories(path_manager: PathManager) -> None:
    """
    Crea le directory necessarie per il progetto.
    
    Args:
        path_manager: Manager dei path
    """
    logger = get_logger(__name__)
    
    paths = path_manager.get_data_paths()
    for path_name, path_value in paths.items():
        ensure_dir(path_value)
        logger.info(f"Directory {path_name} verificata: {path_value}")

def run_retrieve_data(path_manager: PathManager, config_manager: ConfigManager, file_manager: FileManager) -> str:
    """
    Esegue il recupero dati dal database.
    
    Args:
        path_manager: Manager dei path
        config_manager: Manager della configurazione
        file_manager: Manager dei file
        
    Returns:
        Path del file dati grezzi
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 1: RECUPERO DATI ===")
    
    db_config = config_manager.get_database_config()
    output_path = path_manager.get_raw_data_path()
    
    schema_path = db_config.get('schema_path', 'data/db_schema.json')
    selected_aliases = db_config.get('selected_aliases', [])
    
    # Verifica se i dati esistono già
    if file_manager.log_file_status({'raw_data': output_path}, "Recupero dati") and not config_manager.should_force_reload():
        return output_path
    
    # Recupera dati
    retrieve_data(schema_path, selected_aliases, output_path)
    return output_path

def run_build_dataset(path_manager: PathManager, config_manager: ConfigManager, file_manager: FileManager, raw_data_path: str) -> str:
    """
    Esegue la costruzione del dataset.
    
    Args:
        path_manager: Manager dei path
        config_manager: Manager della configurazione
        file_manager: Manager dei file
        raw_data_path: Path ai dati grezzi
        
    Returns:
        Path del dataset processato
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 2: COSTRUZIONE DATASET ===")
    
    output_path = path_manager.get_dataset_path()
    
    # Verifica se il dataset esiste già
    if file_manager.log_file_status({'dataset': output_path}, "Costruzione dataset") and not config_manager.should_force_reload():
        return output_path
    
    # Costruisce dataset
    build_dataset(raw_data_path, output_path)
    return output_path

def run_preprocessing(path_manager: PathManager, config_manager: ConfigManager, file_manager: FileManager, dataset_path: str) -> Dict[str, str]:
    """
    Esegue il preprocessing completo.
    
    Args:
        path_manager: Manager dei path
        config_manager: Manager della configurazione
        file_manager: Manager dei file
        dataset_path: Path al dataset
        
    Returns:
        Dizionario con i path dei file processati
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 3: PREPROCESSING ===")
    
    target_column = config_manager.get_target_column()
    preprocessing_config = config_manager.get_preprocessing_config()
    output_paths = path_manager.get_preprocessing_paths()
    
    # Verifica se i file esistono già
    if file_manager.check_files_exist(output_paths) and not config_manager.should_force_reload():
        logger.info("File preprocessing già esistenti")
        return output_paths
    
    # Esegue preprocessing
    run_preprocessing_pipeline(
        dataset_path=dataset_path,
        target_column=target_column,
        config=preprocessing_config,
        output_paths=output_paths
    )
    
    return output_paths

def run_training(config_manager: ConfigManager, preprocessing_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Esegue il training dei modelli.
    
    Args:
        config_manager: Manager della configurazione
        preprocessing_paths: Path dei file preprocessati
        
    Returns:
        Dictionary con risultati del training
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 4: TRAINING ===")
    
    training_config = config_manager.get_training_config()
    
    # Esegue training pipeline
    training_results = run_training_pipeline(preprocessing_paths, training_config)
    
    logger.info("Training completato con successo")
    return training_results

def run_evaluation(path_manager: PathManager, config_manager: ConfigManager, training_results: Dict[str, Any], preprocessing_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Esegue la valutazione dei modelli.
    
    Args:
        path_manager: Manager dei path
        config_manager: Manager della configurazione
        training_results: Risultati del training
        preprocessing_paths: Path ai file preprocessati
        
    Returns:
        Dictionary con risultati dell'evaluation
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 5: EVALUATION ===")
    
    output_paths = path_manager.get_evaluation_paths()
    
    # Esegue evaluation pipeline
    evaluation_results = run_evaluation_pipeline(
        training_results, preprocessing_paths, config_manager.config, output_paths
    )
    
    logger.info("Evaluation completata con successo")
    return evaluation_results

def main():
    """Funzione principale della pipeline."""
    
    # Parse argomenti
    parser = argparse.ArgumentParser(description='Pipeline ML per analisi immobiliare')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path al file di configurazione')
    parser.add_argument('--steps', nargs='+', 
                       choices=['retrieve_data', 'build_dataset', 'preprocessing', 'training', 'evaluation'],
                       help='Step specifici da eseguire')
    parser.add_argument('--force-reload', action='store_true',
                       help='Forza il ricaricamento di tutti i dati')
    
    args = parser.parse_args()
    
    try:
        # Carica configurazione
        config = load_config(args.config)
        
        # Override force_reload se specificato
        if args.force_reload:
            config.setdefault('execution', {})['force_reload'] = True
        
        # Setup logger
        logger = setup_logger(args.config)
        logger.info("=== AVVIO PIPELINE ML ===")
        logger.info(f"Configurazione caricata da: {args.config}")
        
        # Crea manager delle utility
        path_manager, config_manager, file_manager = create_pipeline_managers(config)
        
        # Setup directory
        setup_directories(path_manager)
        
        # Determina step da eseguire
        steps_to_run = config_manager.get_steps_to_run(args.steps)
        logger.info(f"Step da eseguire: {steps_to_run}")
        
        # Variabili per passaggio dati tra step
        raw_data_path = None
        dataset_path = None
        preprocessing_paths = None
        training_results = None
        
        # Esecuzione step
        if 'retrieve_data' in steps_to_run:
            raw_data_path = run_retrieve_data(path_manager, config_manager, file_manager)
        
        if 'build_dataset' in steps_to_run:
            if raw_data_path is None:
                raw_data_path = path_manager.get_raw_data_path()
            dataset_path = run_build_dataset(path_manager, config_manager, file_manager, raw_data_path)
        
        if 'preprocessing' in steps_to_run:
            if dataset_path is None:
                dataset_path = path_manager.get_dataset_path()
            preprocessing_paths = run_preprocessing(path_manager, config_manager, file_manager, dataset_path)
        
        if 'training' in steps_to_run:
            if preprocessing_paths is None:
                preprocessing_paths = path_manager.get_preprocessing_paths()
            training_results = run_training(config_manager, preprocessing_paths)
        
        if 'evaluation' in steps_to_run:
            if training_results is None:
                logger.error("Training results non disponibili per evaluation. Eseguire prima il training.")
            else:
                run_evaluation(path_manager, config_manager, training_results, preprocessing_paths)
        
        logger.info("=== PIPELINE COMPLETATA CON SUCCESSO ===")
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Errore nella pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()