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
from src.utils.io import load_config, check_file_exists, ensure_dir
from src.db.retrieve import retrieve_data
from src.dataset.build_dataset import build_dataset
from src.preprocessing.pipeline import run_preprocessing_pipeline

def setup_directories(config: Dict[str, Any]) -> None:
    """
    Crea le directory necessarie per il progetto.
    
    Args:
        config: Configurazione del progetto
    """
    logger = get_logger(__name__)
    
    paths = config.get('paths', {})
    for path_name, path_value in paths.items():
        ensure_dir(path_value)
        logger.info(f"Directory {path_name} verificata: {path_value}")

def run_retrieve_data(config: Dict[str, Any]) -> str:
    """
    Esegue il recupero dati dal database.
    
    Args:
        config: Configurazione del progetto
        
    Returns:
        Path del file dati grezzi
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 1: RECUPERO DATI ===")
    
    db_config = config.get('database', {})
    paths = config.get('paths', {})
    
    schema_path = db_config.get('schema_path', 'db_schema.json')
    selected_aliases = db_config.get('selected_aliases', [])
    output_path = f"{paths.get('data_raw', 'data/raw/')}raw_data.parquet"
    
    # Verifica se i dati esistono già
    if check_file_exists(output_path) and not config.get('execution', {}).get('force_reload', False):
        logger.info(f"Dati grezzi già esistenti: {output_path}")
        return output_path
    
    # Recupera dati
    retrieve_data(schema_path, selected_aliases, output_path)
    return output_path

def run_build_dataset(config: Dict[str, Any], raw_data_path: str) -> str:
    """
    Esegue la costruzione del dataset.
    
    Args:
        config: Configurazione del progetto
        raw_data_path: Path ai dati grezzi
        
    Returns:
        Path del dataset processato
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 2: COSTRUZIONE DATASET ===")
    
    paths = config.get('paths', {})
    output_path = f"{paths.get('data_processed', 'data/processed/')}dataset.parquet"
    
    # Verifica se il dataset esiste già
    if check_file_exists(output_path) and not config.get('execution', {}).get('force_reload', False):
        logger.info(f"Dataset già esistente: {output_path}")
        return output_path
    
    # Costruisce dataset
    build_dataset(raw_data_path, output_path)
    return output_path

def run_preprocessing(config: Dict[str, Any], dataset_path: str) -> Dict[str, str]:
    """
    Esegue il preprocessing completo.
    
    Args:
        config: Configurazione del progetto
        dataset_path: Path al dataset
        
    Returns:
        Dizionario con i path dei file processati
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 3: PREPROCESSING ===")
    
    paths = config.get('paths', {})
    target_config = config.get('target', {})
    preprocessing_config = config.get('preprocessing', {})
    
    target_column = target_config.get('column', 'AI_Prezzo_Ridistribuito')
    
    # Path di output
    output_paths = {
        'train_features': f"{paths.get('data_processed', 'data/processed/')}X_train.parquet",
        'test_features': f"{paths.get('data_processed', 'data/processed/')}X_test.parquet", 
        'train_target': f"{paths.get('data_processed', 'data/processed/')}y_train.parquet",
        'test_target': f"{paths.get('data_processed', 'data/processed/')}y_test.parquet",
        'preprocessing_info': f"{paths.get('data_processed', 'data/processed/')}preprocessing_info.json"
    }
    
    # Verifica se i file esistono già
    all_exist = all(check_file_exists(path) for path in output_paths.values())
    if all_exist and not config.get('execution', {}).get('force_reload', False):
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

def run_training(config: Dict[str, Any], preprocessing_paths: Dict[str, str]) -> None:
    """
    Esegue il training dei modelli.
    
    Args:
        config: Configurazione del progetto
        preprocessing_paths: Path dei file preprocessati
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 4: TRAINING ===")
    
    # TODO: Implementare training
    logger.info("Training non ancora implementato")

def run_evaluation(config: Dict[str, Any]) -> None:
    """
    Esegue la valutazione dei modelli.
    
    Args:
        config: Configurazione del progetto
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 5: EVALUATION ===")
    
    # TODO: Implementare evaluation
    logger.info("Evaluation non ancora implementato")

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
        
        # Setup directory
        setup_directories(config)
        
        # Determina step da eseguire
        steps_to_run = args.steps if args.steps else config.get('execution', {}).get('steps', [])
        logger.info(f"Step da eseguire: {steps_to_run}")
        
        # Variabili per passaggio dati tra step
        raw_data_path = None
        dataset_path = None
        preprocessing_paths = None
        
        # Esecuzione step
        if 'retrieve_data' in steps_to_run:
            raw_data_path = run_retrieve_data(config)
        
        if 'build_dataset' in steps_to_run:
            if raw_data_path is None:
                paths = config.get('paths', {})
                raw_data_path = f"{paths.get('data_raw', 'data/raw/')}raw_data.parquet"
            dataset_path = run_build_dataset(config, raw_data_path)
        
        if 'preprocessing' in steps_to_run:
            if dataset_path is None:
                paths = config.get('paths', {})
                dataset_path = f"{paths.get('data_processed', 'data/processed/')}dataset.parquet"
            preprocessing_paths = run_preprocessing(config, dataset_path)
        
        if 'training' in steps_to_run:
            if preprocessing_paths is None:
                paths = config.get('paths', {})
                preprocessing_paths = {
                    'train_features': f"{paths.get('data_processed', 'data/processed/')}X_train.parquet",
                    'test_features': f"{paths.get('data_processed', 'data/processed/')}X_test.parquet",
                    'train_target': f"{paths.get('data_processed', 'data/processed/')}y_train.parquet",
                    'test_target': f"{paths.get('data_processed', 'data/processed/')}y_test.parquet"
                }
            run_training(config, preprocessing_paths)
        
        if 'evaluation' in steps_to_run:
            run_evaluation(config)
        
        logger.info("=== PIPELINE COMPLETATA CON SUCCESSO ===")
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Errore nella pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()