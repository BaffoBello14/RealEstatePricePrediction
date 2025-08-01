"""
Evaluation e analisi delle performance sui modelli addestrati.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_json, save_dataframe

logger = get_logger(__name__)

def calculate_feature_importance(
    best_models: Dict[str, Any], 
    feature_cols: List[str],
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    output_dir: str = None,
    use_advanced_analysis: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola feature importance per i migliori modelli.
    DEPRECATED: Usa run_comprehensive_feature_analysis per analisi completa.
    
    Args:
        best_models: Dictionary con i migliori modelli
        feature_cols: Lista delle colonne features
        X_train: Training features (per SHAP)
        X_test: Test features (per SHAP e permutation)
        y_test: Test target (per permutation)
        output_dir: Directory per salvare plot
        use_advanced_analysis: Se usare analisi avanzata con SHAP
        
    Returns:
        Tuple con DataFrame feature importance e summary
    """
    from .feature_importance import run_comprehensive_feature_analysis, calculate_basic_feature_importance
    
    if use_advanced_analysis and X_train is not None and X_test is not None and y_test is not None and output_dir is not None:
        logger.info("Usando analisi feature importance avanzata...")
        global_summary, detailed_results = run_comprehensive_feature_analysis(
            best_models, X_train, X_test, y_test, feature_cols, output_dir
        )
        
        # Estrai DataFrame base per compatibilità
        df_basic, summary_basic = calculate_basic_feature_importance(best_models, feature_cols)
        
        return df_basic, global_summary
    else:
        logger.info("Usando analisi feature importance di base...")
        return calculate_basic_feature_importance(best_models, feature_cols)

def evaluate_on_test_set(best_models: Dict[str, Any], X_test, y_test_log, y_test_orig) -> Dict[str, Any]:
    """
    Valuta i modelli migliori sul test set finale.
    
    Args:
        best_models: Dictionary con i migliori modelli
        X_test: Features del test set
        y_test_log: Target test in scala logaritmica
        y_test_orig: Target test in scala originale
        
    Returns:
        Dictionary con risultati del test set
    """
    logger.info("=" * 60)
    logger.info("VALUTAZIONE FINALE SU TEST SET")
    logger.info("=" * 60)
    
    test_results = {}
    seen_models = set()
    model_counter = {}

    for key, model_data in best_models.items():
        model = model_data['model']
        model_name = model_data['name']

        # Evita doppie valutazioni sullo stesso oggetto modello
        model_id = id(model)
        if model_id in seen_models:
            continue
        seen_models.add(model_id)

        # Crea alias univoco se ci sono più modelli con lo stesso nome
        model_counter[model_name] = model_counter.get(model_name, 0) + 1
        alias = f"{model_name}__{model_counter[model_name]}" if model_counter[model_name] > 1 else model_name

        try:
            # Predizioni in scala log
            y_pred_test_log = model.predict(X_test)

            # Metriche log
            test_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_test_log))
            test_r2_log = r2_score(y_test_log, y_pred_test_log)
            test_mae_log = mean_absolute_error(y_test_log, y_pred_test_log)
            test_mape_log = mean_absolute_percentage_error(y_test_log, y_pred_test_log) * 100

            # Scala originale
            y_pred_test_orig = np.exp(y_pred_test_log)

            # Metriche originali
            test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
            test_r2_orig = r2_score(y_test_orig, y_pred_test_orig)
            test_mae_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)
            test_mape_orig = mean_absolute_percentage_error(y_test_orig, y_pred_test_orig) * 100

            test_results[alias] = {
                'model_name': model_name,
                'predictions_log': y_pred_test_log,
                'predictions_orig': y_pred_test_orig,
                'rmse_log': test_rmse_log,
                'r2_log': test_r2_log,
                'mae_log': test_mae_log,
                'mape_log': test_mape_log,
                'rmse_orig': test_rmse_orig,
                'r2_orig': test_r2_orig,
                'mae_orig': test_mae_orig,
                'mape_orig': test_mape_orig
            }

            logger.info(f"✓ {alias}:")
            logger.info(f"    Log scale  - RMSE: {test_rmse_log:.6f}, R²: {test_r2_log:.4f}, MAE: {test_mae_log:.6f}")
            logger.info(f"    Orig scale - RMSE: {test_rmse_orig:.2f}, R²: {test_r2_orig:.4f}, MAE: {test_mae_orig:.2f}")

        except Exception as e:
            logger.error(f"✗ Errore nella valutazione di {alias} su test set: {str(e)}")
            test_results[alias] = {'error': str(e)}

    return test_results

def create_evaluation_plots(df_validation_results: pd.DataFrame, best_models: Dict[str, Any], 
                          feature_importance_summary: pd.DataFrame, test_results: Dict[str, Any],
                          y_val_log, output_dir: str) -> None:
    """
    Crea grafici per l'analisi delle performance.
    
    Args:
        df_validation_results: DataFrame con risultati validazione
        best_models: Dictionary con migliori modelli
        feature_importance_summary: DataFrame con feature importance
        test_results: Risultati del test set
        y_val_log: Target validation in scala log
        output_dir: Directory per salvare i grafici
    """
    logger.info("=" * 60)
    logger.info("CREAZIONE GRAFICI DI ANALISI")
    logger.info("=" * 60)
    
    # Setup per i grafici
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Confronto performance modelli
    ax1 = plt.subplot(4, 3, 1)
    if not df_validation_results.empty:
        top_models = df_validation_results.head(10)
        bars = ax1.barh(range(len(top_models)), top_models['Val_RMSE'])
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels([name.replace('_', '\n') for name in top_models['Model']], fontsize=8)
        ax1.set_xlabel('Validation RMSE')
        ax1.set_title('Top 10 Models by Validation RMSE')
        ax1.grid(True, alpha=0.3)
        
        # Colora diverse categorie
        colors = []
        for model in top_models['Model']:
            if 'Baseline' in model:
                colors.append('lightcoral')
            elif 'Ensemble' in model:
                colors.append('gold')
            else:
                colors.append('lightblue')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # 2. R² vs RMSE scatter
    ax2 = plt.subplot(4, 3, 2)
    if not df_validation_results.empty:
        colors_map = {'Baseline': 'red', 'Optimized': 'blue', 'Ensemble': 'orange'}
        for category, color in colors_map.items():
            mask = df_validation_results['Model'].str.contains(category)
            if mask.any():
                ax2.scatter(df_validation_results[mask]['Val_RMSE'], 
                           df_validation_results[mask]['Val_R2'],
                           c=color, label=category, alpha=0.7, s=60)
        
        ax2.set_xlabel('Validation RMSE')
        ax2.set_ylabel('Validation R²')
        ax2.set_title('R² vs RMSE by Model Category')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting analysis
    ax3 = plt.subplot(4, 3, 3)
    if not df_validation_results.empty:
        ax3.scatter(df_validation_results['Train_RMSE'], df_validation_results['Val_RMSE'], alpha=0.6)
        
        # Linea di riferimento (no overfitting)
        min_rmse = min(df_validation_results['Train_RMSE'].min(), df_validation_results['Val_RMSE'].min())
        max_rmse = max(df_validation_results['Train_RMSE'].max(), df_validation_results['Val_RMSE'].max())
        ax3.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', alpha=0.5, label='No Overfitting')
        
        ax3.set_xlabel('Train RMSE')
        ax3.set_ylabel('Validation RMSE')
        ax3.set_title('Overfitting Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance heatmap
    ax4 = plt.subplot(4, 3, 4)
    if not feature_importance_summary.empty:
        # Top 20 features
        top_features = feature_importance_summary.iloc[:20]
        if len(top_features.columns) > 1:
            sns.heatmap(top_features.T, annot=False, cmap='viridis', ax=ax4)
            ax4.set_title('Feature Importance Heatmap (Top 20)')
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Models')
    
    # 5. Training time vs Performance
    ax5 = plt.subplot(4, 3, 5)
    if not df_validation_results.empty:
        scatter = ax5.scatter(df_validation_results['Training_Time'], df_validation_results['Val_RMSE'],
                             c=df_validation_results['Val_R2'], cmap='RdYlBu', alpha=0.7, s=60)
        ax5.set_xlabel('Training Time (seconds)')
        ax5.set_ylabel('Validation RMSE')
        ax5.set_title('Training Time vs Performance')
        ax5.set_xscale('log')
        plt.colorbar(scatter, ax=ax5, label='R²')
        ax5.grid(True, alpha=0.3)
    
    # 6. Predictions vs Actual (miglior modello)
    ax6 = plt.subplot(4, 3, 6)
    if best_models and 'best_overall' in best_models:
        best_model_results = best_models['best_overall']['results']
        if 'predictions_val' in best_model_results:
            ax6.scatter(y_val_log, best_model_results['predictions_val'], alpha=0.6)
            
            # Linea di riferimento perfetta
            min_val = min(y_val_log.min(), best_model_results['predictions_val'].min())
            max_val = max(y_val_log.max(), best_model_results['predictions_val'].max())
            ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
            
            ax6.set_xlabel('Actual (log scale)')
            ax6.set_ylabel('Predicted (log scale)')
            ax6.set_title(f'Predictions vs Actual\n{best_models["best_overall"]["name"]}')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # 7. Residuals plot
    ax7 = plt.subplot(4, 3, 7)
    if best_models and 'best_overall' in best_models:
        best_model_results = best_models['best_overall']['results']
        if 'predictions_val' in best_model_results:
            residuals = y_val_log - best_model_results['predictions_val']
            ax7.scatter(best_model_results['predictions_val'], residuals, alpha=0.6)
            ax7.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax7.set_xlabel('Predicted Values')
            ax7.set_ylabel('Residuals')
            ax7.set_title('Residuals Plot')
            ax7.grid(True, alpha=0.3)
    
    # 8. Distribution of residuals
    ax8 = plt.subplot(4, 3, 8)
    if best_models and 'best_overall' in best_models:
        best_model_results = best_models['best_overall']['results']
        if 'predictions_val' in best_model_results:
            residuals = y_val_log - best_model_results['predictions_val']
            ax8.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax8.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax8.set_xlabel('Residuals')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Distribution of Residuals')
            ax8.grid(True, alpha=0.3)
    
    # 9. Feature importance bar plot
    ax9 = plt.subplot(4, 3, 9)
    if not feature_importance_summary.empty and 'Average' in feature_importance_summary.columns:
        top_features = feature_importance_summary['Average'].head(15)
        bars = ax9.barh(range(len(top_features)), top_features.values)
        ax9.set_yticks(range(len(top_features)))
        ax9.set_yticklabels(top_features.index, fontsize=8)
        ax9.set_xlabel('Average Importance')
        ax9.set_title('Top 15 Most Important Features')
        ax9.grid(True, alpha=0.3)
        
        # Gradient color
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # 10. Test set performance comparison
    ax10 = plt.subplot(4, 3, 10)
    if test_results:
        valid_test_results = {k: v for k, v in test_results.items() if 'error' not in v}
        if valid_test_results:
            model_names = [v['model_name'] for v in valid_test_results.values()]
            rmse_values = [v['rmse_orig'] for v in valid_test_results.values()]
            
            bars = ax10.bar(range(len(model_names)), rmse_values)
            ax10.set_xticks(range(len(model_names)))
            ax10.set_xticklabels([name.replace('_', '\n') for name in model_names], 
                                rotation=45, ha='right', fontsize=8)
            ax10.set_ylabel('Test RMSE (Original Scale)')
            ax10.set_title('Test Set Performance')
            ax10.grid(True, axis='y', alpha=0.3)
            
            # Colora la barra del miglior modello
            best_idx = np.argmin(rmse_values)
            bars[best_idx].set_color('gold')
    
    # Aggiungi altre visualizzazioni se necessario...
    
    # Regola il layout
    plt.tight_layout(pad=3.0)
    
    # Salva il grafico
    plots_path = f'{output_dir}/evaluation_plots.png'
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    
    # Mostra il grafico
    plt.show()
    
    logger.info(f"Grafici di analisi salvati in: {plots_path}")

def generate_evaluation_summary(df_validation_results: pd.DataFrame, best_models: Dict[str, Any], 
                               feature_importance_summary: pd.DataFrame, test_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un summary completo dell'analisi.
    
    Args:
        df_validation_results: DataFrame con risultati validazione
        best_models: Dictionary con migliori modelli
        feature_importance_summary: DataFrame con feature importance
        test_results: Risultati del test set
        
    Returns:
        Dictionary con summary dell'analisi
    """
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY DELL'ANALISI")
    logger.info("=" * 60)

    summary = {}
    
    try:
        # Best model overall
        if best_models and 'best_overall' in best_models:
            best_model = best_models['best_overall']
            results = best_model.get('results', {})
            
            summary['best_model'] = {
                'name': best_model.get('name', 'N/A'),
                'val_rmse': results.get('val_rmse', float('nan')),
                'val_r2': results.get('val_r2', float('nan')),
                'training_time': results.get('training_time', None)
            }
            
            logger.info(f"\n🏆 MIGLIOR MODELLO OVERALL:")
            logger.info(f"   Nome: {summary['best_model']['name']}")
            logger.info(f"   Validation RMSE: {summary['best_model']['val_rmse']:.4f}")
            logger.info(f"   Validation R²: {summary['best_model']['val_r2']:.4f}")
            
            if isinstance(summary['best_model']['training_time'], (int, float)):
                logger.info(f"   Training Time: {summary['best_model']['training_time']:.2f}s")
        
        # Performance statistics
        if not df_validation_results.empty:
            summary['performance_stats'] = {
                'total_models': len(df_validation_results),
                'avg_rmse': df_validation_results['Val_RMSE'].mean(),
                'std_rmse': df_validation_results['Val_RMSE'].std(),
                'avg_r2': df_validation_results['Val_R2'].mean(),
                'avg_training_time': df_validation_results['Training_Time'].mean()
            }
            
            logger.info(f"\n📊 STATISTICHE PERFORMANCE:")
            logger.info(f"   Numero totale modelli: {summary['performance_stats']['total_models']}")
            logger.info(f"   RMSE medio: {summary['performance_stats']['avg_rmse']:.4f}")
            logger.info(f"   RMSE std: {summary['performance_stats']['std_rmse']:.4f}")
            logger.info(f"   R² medio: {summary['performance_stats']['avg_r2']:.4f}")
            logger.info(f"   Tempo training medio: {summary['performance_stats']['avg_training_time']:.2f}s")
        
        # Feature importance insights
        if not feature_importance_summary.empty and 'Average' in feature_importance_summary.columns:
            top_5_features = feature_importance_summary['Average'].sort_values(ascending=False).head(5)
            summary['top_features'] = top_5_features.to_dict()
            
            logger.info(f"\n🔍 TOP 5 FEATURE PIÙ IMPORTANTI:")
            for i, (feature, importance) in enumerate(top_5_features.items(), 1):
                logger.info(f"   {i}. {feature}: {importance:.4f}")
        
        # Test results summary
        if test_results:
            valid_test_results = {k: v for k, v in test_results.items() if 'error' not in v}
            if valid_test_results:
                summary['test_results'] = {}
                logger.info(f"\n🎯 PERFORMANCE SU TEST SET:")
                for model_key, results in valid_test_results.items():
                    summary['test_results'][model_key] = {
                        'rmse_orig': results['rmse_orig'],
                        'r2_orig': results['r2_orig']
                    }
                    logger.info(f"   {results['model_name']}: RMSE = {results['rmse_orig']:.4f}, R² = {results['r2_orig']:.4f}")
        
        # Overfitting analysis
        if not df_validation_results.empty:
            overfitting_ratio = (df_validation_results['Val_RMSE'] / df_validation_results['Train_RMSE']).mean()
            summary['overfitting_analysis'] = {
                'avg_ratio': overfitting_ratio,
                'status': 'good' if 1.05 <= overfitting_ratio <= 1.2 else 'warning'
            }
            
            logger.info(f"\n⚠️  ANALISI OVERFITTING:")
            logger.info(f"   Ratio Val/Train RMSE medio: {overfitting_ratio:.3f}")
            if overfitting_ratio > 1.2:
                logger.info("   ⚠️  Possibile overfitting detectato!")
            elif overfitting_ratio < 1.05:
                logger.info("   ⚠️  Possibile underfitting detectato!")
            else:
                logger.info("   ✅ Buon bilanciamento bias-variance!")
    
    except Exception as e:
        logger.error(f"Errore nella generazione del summary: {e}")
        summary['error'] = str(e)
    
    logger.info("=" * 60)
    return summary

def run_evaluation_pipeline(training_results: Dict[str, Any], preprocessing_paths: Dict[str, str], 
                          config: Dict[str, Any], output_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Esegue la pipeline completa di evaluation.
    
    Args:
        training_results: Risultati del training
        preprocessing_paths: Path ai file preprocessati
        config: Configurazione
        output_paths: Path di output
        
    Returns:
        Dictionary con risultati dell'evaluation
    """
    logger.info("=== AVVIO PIPELINE EVALUATION ===")
    
    try:
        # Carica dati di test
        X_test = load_dataframe(preprocessing_paths['test_features'])
        y_test_log = load_dataframe(preprocessing_paths['test_target']).squeeze()
        y_test_orig = load_dataframe(preprocessing_paths['test_target_orig']).squeeze()
        y_val_log = load_dataframe(preprocessing_paths['val_target']).squeeze()
        
        # Carica dati di training per feature importance avanzata
        X_train = load_dataframe(preprocessing_paths['train_features'])
        
        # Ottieni feature columns
        feature_cols = X_test.columns.tolist()
        
        # 1. Calcola feature importance (analisi avanzata con SHAP)
        df_feature_importance, feature_importance_summary = calculate_feature_importance(
            training_results['best_models'], 
            feature_cols,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test_orig,
            output_dir=output_paths.get('results_dir', 'results'),
            use_advanced_analysis=True
        )
        
        # 2. Valutazione su test set
        test_results = evaluate_on_test_set(
            training_results['best_models'], X_test, y_test_log, y_test_orig
        )
        
        # 3. Crea grafici di analisi
        create_evaluation_plots(
            training_results['df_validation_results'], 
            training_results['best_models'],
            feature_importance_summary, 
            test_results,
            y_val_log,
            output_paths.get('results_dir', 'results')
        )
        
        # 4. Genera summary
        evaluation_summary = generate_evaluation_summary(
            training_results['df_validation_results'],
            training_results['best_models'],
            feature_importance_summary,
            test_results
        )
        
        # 5. Salva risultati
        if not df_feature_importance.empty:
            save_dataframe(df_feature_importance, output_paths.get('feature_importance_path', 'feature_importance.csv'))
        
        save_json(evaluation_summary, output_paths.get('evaluation_summary_path', 'evaluation_summary.json'))
        
        evaluation_results = {
            'feature_importance': df_feature_importance,
            'feature_importance_summary': feature_importance_summary,
            'test_results': test_results,
            'evaluation_summary': evaluation_summary
        }
        
        logger.info("=== PIPELINE EVALUATION COMPLETATA ===")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Errore nella pipeline di evaluation: {e}")
        raise