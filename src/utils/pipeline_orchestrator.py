"""
Orchestratore centrale per la pipeline di Machine Learning.

Questo modulo fornisce una classe per gestire l'esecuzione coordinata
di tutti i step della pipeline ML, centralizzando la logica di orchestrazione
e migliorando la manutenibilità.
"""

import sys
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .logger import get_logger
from .path_manager import PathManager, create_path_manager
from .error_handling import PipelineExecutionError, safe_execution


class PipelineOrchestrator:
    """
    Orchestratore centrale per la pipeline di Machine Learning.
    
    Gestisce l'esecuzione coordinata di tutti i step della pipeline,
    centralizzando la logica di passaggio dati tra step e la gestione
    degli errori.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza l'orchestratore.
        
        Args:
            config: Configurazione completa del progetto
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.path_manager = create_path_manager(config)
        
        # State management per passaggio dati tra step
        self._state = {
            'raw_data_path': None,
            'dataset_path': None, 
            'preprocessing_paths': None,
            'training_results': None
        }
        
        # Step disponibili e loro funzioni di esecuzione
        self._available_steps = {
            'retrieve_data': self._execute_retrieve_data,
            'build_dataset': self._execute_build_dataset,
            'preprocessing': self._execute_preprocessing,
            'training': self._execute_training,
            'evaluation': self._execute_evaluation
        }
    
    def setup_environment(self) -> None:
        """
        Configura l'ambiente necessario per l'esecuzione della pipeline.
        """
        self.logger.info("🔧 Configurazione ambiente pipeline...")
        
        # Assicura che tutte le directory esistano
        self.path_manager.ensure_all_directories()
        self.logger.info(f"🗂️  PathManager attivato: {self.path_manager}")
    
    def get_steps_to_run(self, requested_steps: Optional[List[str]] = None) -> List[str]:
        """
        Determina quali step eseguire basandosi sulla configurazione.
        
        Args:
            requested_steps: Step specificamente richiesti (override configurazione)
            
        Returns:
            Lista degli step da eseguire
        """
        if requested_steps:
            steps = requested_steps
        else:
            steps = self.config.get('execution', {}).get('steps', [])
        
        # Validazione step
        invalid_steps = [step for step in steps if step not in self._available_steps]
        if invalid_steps:
            raise PipelineExecutionError(f"Step non validi: {invalid_steps}")
        
        self.logger.info(f"📋 Step da eseguire: {steps}")
        return steps
    
    @safe_execution(reraise=True)
    def run_pipeline(self, steps_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Esegue la pipeline completa con gli step specificati.
        
        Args:
            steps_to_run: Lista specifica di step da eseguire
            
        Returns:
            Risultati finali della pipeline
            
        Raises:
            PipelineExecutionError: In caso di errore nell'esecuzione
        """
        self.logger.info("=== AVVIO PIPELINE ML RISTRUTTURATA ===")
        
        steps = self.get_steps_to_run(steps_to_run)
        results = {}
        
        # Esecuzione sequenziale degli step
        for step_name in steps:
            self.logger.info(f"▶️  Esecuzione step: {step_name}")
            
            try:
                step_result = self._available_steps[step_name]()
                results[step_name] = step_result
                self.logger.info(f"✅ Step '{step_name}' completato con successo")
                
            except Exception as e:
                self.logger.error(f"❌ Errore nello step '{step_name}': {e}")
                raise PipelineExecutionError(f"Fallimento step '{step_name}': {e}") from e
        
        self.logger.info("=== PIPELINE COMPLETATA CON SUCCESSO ===")
        return results
    
    def _get_or_reconstruct_path(self, state_key: str, path_method: str) -> str:
        """
        Ottiene un path dallo state o lo ricostruisce usando PathManager.
        
        Args:
            state_key: Chiave nello state dell'orchestratore
            path_method: Nome del metodo PathManager da chiamare
            
        Returns:
            Path ricostruito o recuperato dallo state
        """
        if self._state[state_key] is not None:
            return self._state[state_key]
        
        # Ricostruisci path usando PathManager
        path = getattr(self.path_manager, path_method)()
        self.logger.info(f"🔗 Path {state_key} ricostruito: {path}")
        return path
    
    @safe_execution(reraise=True)
    def _execute_retrieve_data(self) -> str:
        """Esegue il recupero dati dal database."""
        from ..db.retrieve import retrieve_data
        
        self.logger.info("=== FASE 1: RECUPERO DATI ===")
        
        schema_path = self.path_manager.get_database_schema_path()
        output_path = self.path_manager.get_raw_data_path()
        
        # Esecuzione recupero dati
        result_path = retrieve_data(
            config=self.config,
            schema_path=schema_path,
            output_path=output_path
        )
        
        self._state['raw_data_path'] = result_path
        self.logger.info(f"📊 Dati recuperati in: {result_path}")
        return result_path
    
    @safe_execution(reraise=True)
    def _execute_build_dataset(self) -> str:
        """Esegue la costruzione del dataset."""
        from ..dataset.build_dataset import build_dataset
        
        self.logger.info("=== FASE 2: COSTRUZIONE DATASET ===")
        
        # Ottieni path dati grezzi (da state o ricostruisci)
        raw_data_path = self._get_or_reconstruct_path('raw_data_path', 'get_raw_data_path')
        output_path = self.path_manager.get_dataset_path()
        
        # Esecuzione costruzione dataset
        result_path = build_dataset(
            config=self.config,
            input_path=raw_data_path,
            output_path=output_path
        )
        
        self._state['dataset_path'] = result_path
        self.logger.info(f"🏗️  Dataset costruito in: {result_path}")
        return result_path
    
    @safe_execution(reraise=True)
    def _execute_preprocessing(self) -> Dict[str, str]:
        """Esegue il preprocessing dei dati."""
        from ..preprocessing.pipeline import run_preprocessing_pipeline
        
        self.logger.info("=== FASE 3: PREPROCESSING ===")
        
        # Ottieni path dataset (da state o ricostruisci)
        dataset_path = self._get_or_reconstruct_path('dataset_path', 'get_dataset_path')
        output_paths = self.path_manager.get_preprocessing_paths()
        
        # Esecuzione preprocessing
        result_paths = run_preprocessing_pipeline(
            config=self.config,
            dataset_path=dataset_path,
            output_paths=output_paths
        )
        
        self._state['preprocessing_paths'] = result_paths
        self.logger.info(f"🔄 Preprocessing completato: {len(result_paths)} file generati")
        return result_paths
    
    @safe_execution(reraise=True)
    def _execute_training(self) -> Dict[str, Any]:
        """Esegue il training dei modelli."""
        from ..training.train import run_training_pipeline
        
        self.logger.info("=== FASE 4: TRAINING ===")
        
        # Ottieni path preprocessing (da state o ricostruisci)
        if self._state['preprocessing_paths'] is None:
            preprocessing_paths = self.path_manager.get_preprocessing_paths()
            self.logger.info(f"🔗 Path preprocessing ricostruiti: {len(preprocessing_paths)} file")
        else:
            preprocessing_paths = self._state['preprocessing_paths']
        
        # Esecuzione training
        results = run_training_pipeline(
            config=self.config,
            preprocessing_paths=preprocessing_paths
        )
        
        self._state['training_results'] = results
        self.logger.info("🎯 Training completato con successo")
        return results
    
    @safe_execution(reraise=True)
    def _execute_evaluation(self) -> Dict[str, Any]:
        """Esegue la valutazione dei modelli."""
        from ..training.evaluation import run_evaluation_pipeline
        
        self.logger.info("=== FASE 5: EVALUATION ===")
        
        # Verifica prerequisiti
        if self._state['training_results'] is None:
            raise PipelineExecutionError(
                "Training results non disponibili per evaluation. "
                "Eseguire prima il step 'training'."
            )
        
        # Ottieni path preprocessing
        if self._state['preprocessing_paths'] is None:
            preprocessing_paths = self.path_manager.get_preprocessing_paths()
        else:
            preprocessing_paths = self._state['preprocessing_paths']
        
        evaluation_paths = self.path_manager.get_evaluation_paths()
        
        # Esecuzione evaluation
        results = run_evaluation_pipeline(
            config=self.config,
            training_results=self._state['training_results'],
            preprocessing_paths=preprocessing_paths,
            output_paths=evaluation_paths
        )
        
        self.logger.info("📈 Evaluation completata con successo")
        return results
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Ritorna lo stato corrente della pipeline.
        
        Returns:
            Stato corrente con path e risultati
        """
        return {
            'state': self._state.copy(),
            'path_manager_info': str(self.path_manager),
            'available_steps': list(self._available_steps.keys())
        }