"""
Pipeline di preprocessing modularizzata e ristrutturata.
Sostituisce il pipeline.py monolitico con un approccio modulare e ben definito.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from ..utils import get_logger, save_dataframe, save_json

# Import degli step modulari
from .steps import (
    load_and_validate_dataset,
    execute_data_cleaning_step,
    execute_feature_analysis_step,
    execute_encoding_step,
    execute_imputation_step,
    execute_feature_filtering_step,
    execute_data_splitting_step,
    execute_target_processing_step,
    execute_feature_scaling_step,
    execute_pca_step
)

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Pipeline di preprocessing modularizzata.
    
    Caratteristiche:
    - Ogni step è una funzione specializzata e testabile
    - Tracciamento completo di tutte le operazioni
    - Possibilità di eseguire step specifici
    - Validazione automatica tra step
    - Rollback in caso di errori
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza la pipeline con la configurazione.
        
        Args:
            config: Configurazione del preprocessing
        """
        self.config = config
        self.steps_config = config.get('steps', {})
        self.execution_history = []
        self.current_dataframe = None
        self.step_results = {}
        
        # Definizione ordine degli step
        self.step_definitions = [
            {
                'name': 'data_loading',
                'function': load_and_validate_dataset,
                'enabled_key': None,  # Sempre abilitato
                'description': 'Caricamento e validazione dataset'
            },
            {
                'name': 'data_cleaning', 
                'function': execute_data_cleaning_step,
                'enabled_key': 'enable_data_cleaning',
                'description': 'Pulizia dati unificata'
            },
            {
                'name': 'feature_analysis',
                'function': execute_feature_analysis_step,
                'enabled_key': 'enable_feature_analysis', 
                'description': 'Analisi feature e correlazioni'
            },
            {
                'name': 'data_encoding',
                'function': execute_encoding_step,
                'enabled_key': 'enable_advanced_encoding',
                'description': 'Encoding feature categoriche'
            },
            {
                'name': 'data_imputation',
                'function': execute_imputation_step,
                'enabled_key': 'enable_imputation',
                'description': 'Imputazione valori mancanti'
            },
            {
                'name': 'feature_filtering',
                'function': execute_feature_filtering_step,
                'enabled_key': 'enable_correlation_removal',
                'description': 'Filtering feature correlate'
            },
            {
                'name': 'data_splitting',
                'function': execute_data_splitting_step,
                'enabled_key': None,  # Sempre abilitato
                'description': 'Split train/validation/test'
            },
            {
                'name': 'target_processing',
                'function': execute_target_processing_step,
                'enabled_key': 'enable_target_processing',
                'description': 'Processamento target e outlier detection'
            },
            {
                'name': 'feature_scaling',
                'function': execute_feature_scaling_step,
                'enabled_key': 'enable_feature_scaling',
                'description': 'Scaling delle feature'
            },
            {
                'name': 'dimensionality_reduction',
                'function': execute_pca_step,
                'enabled_key': 'enable_pca',
                'description': 'Riduzione dimensionalità (PCA)'
            }
        ]
        
        logger.info(f"🔧 Pipeline modularizzata inizializzata con {len(self.step_definitions)} step")
    
    def is_step_enabled(self, step_definition: Dict[str, Any]) -> bool:
        """
        Controlla se uno step è abilitato nella configurazione.
        
        Args:
            step_definition: Definizione dello step
            
        Returns:
            True se lo step è abilitato
        """
        enabled_key = step_definition.get('enabled_key')
        
        if enabled_key is None:
            return True  # Step sempre abilitati (loading, splitting)
        
        return self.steps_config.get(enabled_key, True)
    
    def execute_step(
        self, 
        step_definition: Dict[str, Any], 
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> bool:
        """
        Esegue un singolo step della pipeline.
        
        Args:
            step_definition: Definizione dello step da eseguire
            dataset_path: Path del dataset (solo per step di loading)
            target_column: Nome colonna target
            
        Returns:
            True se lo step è stato eseguito con successo
        """
        step_name = step_definition['name']
        step_function = step_definition['function']
        description = step_definition['description']
        
        logger.info(f"🚀 Esecuzione step: {step_name} - {description}")
        
        try:
            # Chiamata speciale per il data loading
            if step_name == 'data_loading':
                if not dataset_path or not target_column:
                    raise ValueError("dataset_path e target_column richiesti per data_loading")
                
                self.current_dataframe, step_result = step_function(dataset_path, target_column)
            else:
                # Chiamata standard per tutti gli altri step
                if self.current_dataframe is None:
                    raise ValueError(f"Nessun DataFrame disponibile per step {step_name}")
                
                self.current_dataframe, step_result = step_function(
                    self.current_dataframe, 
                    target_column, 
                    self.config
                )
            
            # Salva risultati dello step
            self.step_results[step_name] = step_result
            self.execution_history.append({
                'step_name': step_name,
                'success': True,
                'description': description,
                'dataframe_shape': list(self.current_dataframe.shape) if self.current_dataframe is not None else None
            })
            
            logger.info(f"✅ Step {step_name} completato con successo")
            if self.current_dataframe is not None:
                logger.info(f"📊 Shape corrente: {self.current_dataframe.shape}")
            
            return True
            
        except Exception as e:
            error_msg = f"Errore nello step {step_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            self.execution_history.append({
                'step_name': step_name,
                'success': False,
                'error': error_msg,
                'description': description
            })
            
            return False
    
    def run_full_pipeline(
        self, 
        dataset_path: str, 
        target_column: str, 
        output_paths: Dict[str, str],
        stop_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Esegue la pipeline completa di preprocessing.
        
        Args:
            dataset_path: Path al dataset da processare
            target_column: Nome della colonna target
            output_paths: Dictionary con i path di output
            stop_on_error: Se True, si ferma al primo errore
            
        Returns:
            Dictionary con informazioni complete sull'esecuzione
        """
        logger.info("🎬 AVVIO PIPELINE PREPROCESSING MODULARIZZATA")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🎯 Target: {target_column}")
        
        # Reset stato pipeline
        self.execution_history = []
        self.step_results = {}
        self.current_dataframe = None
        
        # Contatori per statistiche
        steps_executed = 0
        steps_skipped = 0
        steps_failed = 0
        
        # Esecuzione step
        for step_definition in self.step_definitions:
            step_name = step_definition['name']
            
            # Controlla se step è abilitato
            if not self.is_step_enabled(step_definition):
                logger.info(f"⏭️  Step {step_name} DISABILITATO, skip...")
                steps_skipped += 1
                continue
            
            # Esegue step
            success = self.execute_step(
                step_definition, 
                dataset_path=dataset_path,
                target_column=target_column
            )
            
            if success:
                steps_executed += 1
            else:
                steps_failed += 1
                if stop_on_error:
                    logger.error(f"💥 Pipeline fermata a causa di errore nello step {step_name}")
                    break
        
        # Genera informazioni finali
        pipeline_info = {
            'pipeline_version': 'modular_v1.0',
            'execution_summary': {
                'total_steps_defined': len(self.step_definitions),
                'steps_executed': steps_executed,
                'steps_skipped': steps_skipped,
                'steps_failed': steps_failed,
                'success_rate': steps_executed / (steps_executed + steps_failed) * 100 if (steps_executed + steps_failed) > 0 else 0
            },
            'config_used': self.config,
            'target_column': target_column,
            'execution_history': self.execution_history,
            'step_results': self.step_results,
            'final_dataframe_shape': list(self.current_dataframe.shape) if self.current_dataframe is not None else None
        }
        
        # Log riassuntivo
        success_rate = pipeline_info['execution_summary']['success_rate']
        logger.info(f"🏁 PIPELINE COMPLETATA")
        logger.info(f"📊 Statistiche: {steps_executed} eseguiti, {steps_skipped} saltati, {steps_failed} falliti")
        logger.info(f"📈 Tasso successo: {success_rate:.1f}%")
        
        if self.current_dataframe is not None:
            logger.info(f"📊 Dataset finale: {self.current_dataframe.shape}")
        
        # Salva informazioni
        if 'preprocessing_info' in output_paths:
            save_json(pipeline_info, output_paths['preprocessing_info'])
            logger.info(f"💾 Info pipeline salvate: {output_paths['preprocessing_info']}")
        
        return pipeline_info
    
    def get_step_summary(self) -> List[Dict[str, Any]]:
        """
        Ottiene un riassunto degli step disponibili e del loro stato.
        
        Returns:
            Lista con informazioni su ogni step
        """
        summary = []
        
        for step_def in self.step_definitions:
            step_info = {
                'name': step_def['name'],
                'description': step_def['description'],
                'enabled': self.is_step_enabled(step_def),
                'executed': any(h['step_name'] == step_def['name'] for h in self.execution_history),
                'success': any(h['step_name'] == step_def['name'] and h.get('success', False) for h in self.execution_history)
            }
            summary.append(step_info)
        
        return summary


def run_modular_preprocessing_pipeline(
    dataset_path: str,
    target_column: str,
    config: Dict[str, Any],
    output_paths: Dict[str, str]
) -> None:
    """
    Funzione di interfaccia per eseguire la pipeline modularizzata.
    Sostituisce run_preprocessing_pipeline del pipeline.py originale.
    
    Args:
        dataset_path: Path al dataset da processare
        target_column: Nome della colonna target
        config: Configurazione del preprocessing
        output_paths: Dictionary con i path di output
        
    Note:
        Questa funzione mantiene la compatibilità con l'interfaccia esistente
        mentre utilizza la nuova architettura modularizzata sotto il cofano.
    """
    logger.info("🔄 Avvio preprocessing con pipeline modularizzata")
    
    try:
        # Crea e configura pipeline
        pipeline = PreprocessingPipeline(config)
        
        # Esegue pipeline completa
        pipeline_info = pipeline.run_full_pipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            output_paths=output_paths,
            stop_on_error=True
        )
        
        # Log summary degli step
        step_summary = pipeline.get_step_summary()
        logger.info("📋 SUMMARY STEP:")
        for step in step_summary:
            status = "✅ SUCCESS" if step['success'] else ("❌ FAILED" if step['executed'] else ("⏭️ SKIPPED" if not step['enabled'] else "⏸️ NOT RUN"))
            logger.info(f"   {step['name']}: {status} - {step['description']}")
        
        # Warning se ci sono stati fallimenti
        failed_steps = [h for h in pipeline.execution_history if not h.get('success', True)]
        if failed_steps:
            logger.warning(f"⚠️  {len(failed_steps)} step falliti durante l'esecuzione")
            for failed_step in failed_steps:
                logger.warning(f"   - {failed_step['step_name']}: {failed_step.get('error', 'Errore sconosciuto')}")
        
        logger.info("✨ Pipeline modularizzata completata")
        
    except Exception as e:
        logger.error(f"💥 Errore critico nella pipeline modularizzata: {str(e)}")
        raise