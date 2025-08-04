"""
Entry point principale per la pipeline di Machine Learning.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src import (
    setup_logger, get_logger,
    load_config, check_file_exists,
    PipelineOrchestrator
)

def setup_directories(config: Dict[str, Any]) -> None:
    """
    DEPRECATA: Usa PathManager.ensure_all_directories().
    Crea le directory necessarie per il progetto.
    
    Args:
        config: Configurazione del progetto
    """
    logger = get_logger(__name__)
    logger.warning("⚠️  setup_directories è DEPRECATA, usa PathManager.ensure_all_directories()")
    
    path_manager = create_path_manager(config)
    path_manager.ensure_all_directories()

def run_retrieve_data(config: Dict[str, Any]) -> str:
    """
    RISTRUTTURATA: Esegue il recupero dati dal database usando PathManager.
    
    Args:
        config: Configurazione del progetto
        
    Returns:
        Path del file dati grezzi
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 1: RECUPERO DATI ===")
    
    # Usa PathManager per gestire i path
    path_manager = create_path_manager(config)
    db_config = config.get('database', {})
    
    schema_path = path_manager.get_database_schema_path()
    selected_aliases = db_config.get('selected_aliases', [])
    output_path = path_manager.get_raw_data_path()
    
    # Verifica se i dati esistono già
    if check_file_exists(output_path) and not config.get('execution', {}).get('force_reload', False):
        logger.info(f"📁 Dati grezzi già esistenti: {output_path}")
        return output_path
    
    # Recupera dati
    logger.info(f"🔄 Recupero dati da schema: {schema_path}")
    retrieve_data(schema_path, selected_aliases, output_path)
    logger.info(f"✅ Dati grezzi salvati: {output_path}")
    return output_path

def run_build_dataset(config: Dict[str, Any], raw_data_path: str) -> str:
    """
    RISTRUTTURATA: Esegue la costruzione del dataset usando PathManager.
    
    Args:
        config: Configurazione del progetto
        raw_data_path: Path ai dati grezzi
        
    Returns:
        Path del dataset processato
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 2: COSTRUZIONE DATASET ===")
    
    # Usa PathManager per gestire i path
    path_manager = create_path_manager(config)
    output_path = path_manager.get_dataset_path()
    
    # Verifica se il dataset esiste già
    if check_file_exists(output_path) and not config.get('execution', {}).get('force_reload', False):
        logger.info(f"📁 Dataset già esistente: {output_path}")
        return output_path
    
    # Costruisce dataset
    logger.info(f"🔄 Costruzione dataset: {raw_data_path} → {output_path}")
    build_dataset(raw_data_path, output_path)
    logger.info(f"✅ Dataset costruito: {output_path}")
    return output_path

def run_preprocessing(config: Dict[str, Any], dataset_path: str) -> Dict[str, str]:
    """
    RISTRUTTURATA: Esegue il preprocessing completo usando PathManager.
    
    Args:
        config: Configurazione del progetto
        dataset_path: Path al dataset
        
    Returns:
        Dizionario con i path dei file processati
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 3: PREPROCESSING ===")
    
    # Usa PathManager per gestire i path
    path_manager = create_path_manager(config)
    target_config = config.get('target', {})
    preprocessing_config = config.get('preprocessing', {})
    
    target_column = target_config.get('column', 'AI_Prezzo_Ridistribuito')
    
    # Ottieni path di output dal PathManager (elimina duplicazione)
    output_paths = path_manager.get_preprocessing_paths()
    
    # Verifica se i file esistono già
    all_exist = all(check_file_exists(path) for path in output_paths.values())
    if all_exist and not config.get('execution', {}).get('force_reload', False):
        logger.info("📁 File preprocessing già esistenti")
        return output_paths
    
    # Esegue preprocessing
    logger.info(f"🔄 Avvio preprocessing: {dataset_path}")
    logger.info(f"🎯 Target column: {target_column}")
    run_preprocessing_pipeline(
        dataset_path=dataset_path,
        target_column=target_column,
        config=preprocessing_config,
        output_paths=output_paths
    )
    logger.info(f"✅ Preprocessing completato, {len(output_paths)} file generati")
    
    return output_paths

def run_training(config: Dict[str, Any], preprocessing_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Esegue il training dei modelli.
    
    Args:
        config: Configurazione del progetto
        preprocessing_paths: Path dei file preprocessati
        
    Returns:
        Dictionary con risultati del training
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 4: TRAINING ===")
    
    training_config = config.get('training', {})
    
    # Esegue training pipeline
    training_results = run_training_pipeline(preprocessing_paths, training_config)
    
    logger.info("Training completato con successo")
    return training_results

def run_evaluation(config: Dict[str, Any], training_results: Dict[str, Any], preprocessing_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    RISTRUTTURATA: Esegue la valutazione dei modelli usando PathManager.
    
    Args:
        config: Configurazione del progetto
        training_results: Risultati del training
        preprocessing_paths: Path ai file preprocessati
        
    Returns:
        Dictionary con risultati dell'evaluation
    """
    logger = get_logger(__name__)
    logger.info("=== FASE 5: EVALUATION ===")
    
    # Usa PathManager per gestire i path
    path_manager = create_path_manager(config)
    output_paths = path_manager.get_evaluation_paths()
    
    # Esegue evaluation pipeline
    logger.info(f"🔄 Avvio evaluation con {len(output_paths)} output path")
    evaluation_results = run_evaluation_pipeline(
        training_results, preprocessing_paths, config, output_paths
    )
    
    logger.info("✅ Evaluation completata con successo")
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
        logger.info("=== AVVIO PIPELINE ML RISTRUTTURATA ===")
        logger.info(f"📄 Configurazione caricata da: {args.config}")
        
        # Inizializza orchestratore pipeline
        orchestrator = PipelineOrchestrator(config)
        orchestrator.setup_environment()
        
        # Determina e esegue step
        steps_to_run = args.steps if args.steps else None
        results = orchestrator.run_pipeline(steps_to_run)
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Errore nella pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()