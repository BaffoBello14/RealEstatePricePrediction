"""
Entry point principale per la pipeline di Machine Learning.

Questo modulo fornisce l'interfaccia principale per l'esecuzione della pipeline
di Machine Learning ristrutturata, utilizzando il PipelineOrchestrator per
gestire l'orchestrazione degli step.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Aggiungi src al path per import locali
sys.path.append(str(Path(__file__).parent / "src"))

# Import consolidati e ottimizzati
from src import setup_logger, get_logger, load_config, PipelineOrchestrator


def parse_arguments() -> argparse.Namespace:
    """
    Parse degli argomenti da linea di comando.
    
    Returns:
        Argomenti parsati
    """
    parser = argparse.ArgumentParser(
        description='Pipeline ML ristrutturata per analisi immobiliare',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python main.py                                    # Esegue pipeline completa
  python main.py --steps retrieve_data build_dataset # Esegue solo alcuni step
  python main.py --force-reload                     # Forza ricaricamento dati
  python main.py --config config/custom.yaml       # Usa configurazione custom
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config/config.yaml',
        help='Path al file di configurazione (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--steps', 
        nargs='+',
        choices=['retrieve_data', 'build_dataset', 'preprocessing', 'training', 'evaluation'],
        help='Step specifici da eseguire (default: tutti dalla configurazione)'
    )
    
    parser.add_argument(
        '--force-reload', 
        action='store_true',
        help='Forza il ricaricamento di tutti i dati esistenti'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Mostra gli step che verrebbero eseguiti senza eseguirli'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Abilita logging più dettagliato'
    )
    
    return parser.parse_args()


def setup_configuration(args: argparse.Namespace) -> dict:
    """
    Configura e carica la configurazione del progetto.
    
    Args:
        args: Argomenti da linea di comando
        
    Returns:
        Configurazione caricata e aggiornata
    """
    # Carica configurazione base
    config = load_config(args.config)
    
    # Applica override da argomenti
    execution_config = config.setdefault('execution', {})
    
    if args.force_reload:
        execution_config['force_reload'] = True
    
    if args.verbose:
        config.setdefault('logging', {})['level'] = 'DEBUG'
    
    return config


def run_dry_run(orchestrator: PipelineOrchestrator, steps: Optional[List[str]] = None) -> None:
    """
    Esegue una simulazione della pipeline senza effettivo processing.
    
    Args:
        orchestrator: Istanza dell'orchestratore
        steps: Step da simulare
    """
    logger = get_logger(__name__)
    logger.info("=== MODALITÀ DRY-RUN ATTIVATA ===")
    
    steps_to_run = orchestrator.get_steps_to_run(steps)
    
    logger.info("📋 Step che verrebbero eseguiti:")
    for i, step in enumerate(steps_to_run, 1):
        logger.info(f"  {i}. {step}")
    
    state = orchestrator.get_pipeline_state()
    logger.info(f"🔧 PathManager: {state['path_manager_info']}")
    logger.info("=== FINE DRY-RUN ===")


def main() -> None:
    """
    Funzione principale della pipeline ML ristrutturata.
    
    Gestisce l'orchestrazione completa utilizzando PipelineOrchestrator
    per garantire modularità, robustezza e manutenibilità.
    """
    # Parse argomenti e configurazione
    args = parse_arguments()
    config = setup_configuration(args)
    
    # Setup del sistema di logging
    logger = setup_logger(args.config)
    logger.info("🚀 === AVVIO PIPELINE ML RISTRUTTURATA ===")
    logger.info(f"📄 Configurazione caricata da: {args.config}")
    
    if args.verbose:
        logger.info(f"🎛️  Argomenti: {vars(args)}")
    
    try:
        # Inizializza orchestratore pipeline
        orchestrator = PipelineOrchestrator(config)
        
        # Setup ambiente (directory, validazioni, etc.)
        orchestrator.setup_environment()
        
        # Modalità dry-run
        if args.dry_run:
            run_dry_run(orchestrator, args.steps)
            return
        
        # Esecuzione pipeline vera e propria
        logger.info("▶️  Avvio esecuzione pipeline...")
        results = orchestrator.run_pipeline(args.steps)
        
        # Reporting finale
        logger.info("🎉 === PIPELINE COMPLETATA CON SUCCESSO ===")
        logger.info(f"📊 Step eseguiti: {len(results)}")
        
        if args.verbose:
            for step, result in results.items():
                if isinstance(result, dict) and 'total_rows' in result:
                    logger.info(f"  • {step}: {result.get('total_rows', 'N/A')} righe processate")
                elif isinstance(result, str):
                    logger.info(f"  • {step}: {result}")
                else:
                    logger.info(f"  • {step}: completato")
        
        # Stato finale della pipeline
        final_state = orchestrator.get_pipeline_state()
        logger.info(f"📁 File generati consultabili tramite PathManager")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Pipeline interrotta dall'utente")
        sys.exit(130)  # Standard exit code per SIGINT
        
    except Exception as e:
        logger.error(f"❌ Errore critico nella pipeline: {e}")
        if args.verbose:
            logger.exception("📋 Traceback completo:")
        sys.exit(1)


if __name__ == "__main__":
    main()