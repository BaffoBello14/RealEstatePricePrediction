"""
Dimensionality reduction module with PCA and other techniques
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


def apply_pca(X_train, X_test, config):
    """
    Apply PCA dimensionality reduction
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        config: Configuration dictionary with PCA settings
        
    Returns:
        tuple: (X_train_pca, X_test_pca, pca_model, loadings_df)
    """
    variance_threshold = config.get('variance_threshold', 0.95)
    random_state = config.get('random_state', 42)
    
    logger.info(f"Applying PCA (variance target: {variance_threshold})...")
    
    # Initialize PCA
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    
    # Fit on training data and transform both sets
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Get explained variance information
    explained_var = pca.explained_variance_ratio_
    total_variance = explained_var.sum()
    
    logger.info(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} components")
    logger.info(f"Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)")
    
    # Calculate reconstruction error
    X_reconstructed = pca.inverse_transform(X_train_pca)
    reconstruction_error = np.mean((X_train - X_reconstructed) ** 2)
    logger.info(f"PCA reconstruction error: {reconstruction_error:.6f}")
    
    # Create loadings DataFrame for interpretability
    feature_names = (X_train.columns if hasattr(X_train, 'columns') 
                    else [f'Feature_{i}' for i in range(X_train.shape[1])])
    
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    # Log top features for each component
    logger.info("Top 5 features for each principal component:")
    for i in range(min(5, pca.n_components_)):  # Show first 5 components
        pc_name = f'PC{i+1}'
        top_features = loadings_df.loc[pc_name].abs().sort_values(ascending=False).head(5)
        logger.info(f"{pc_name} (variance: {explained_var[i]:.3f}): {dict(top_features)}")
    
    return X_train_pca, X_test_pca, pca, loadings_df


def visualize_pca_variance(pca, save_path=None):
    """
    Visualize PCA variance explained
    
    Args:
        pca: Fitted PCA model
        save_path: Optional path to save the plot
    """
    explained_var = pca.explained_variance_ratio_
    cumvar = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual variance explained
    ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title('Variance Explained by Each Principal Component')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance explained
    ax2.plot(range(1, len(cumvar) + 1), cumvar, 'bo-', markersize=4)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.axhline(y=0.99, color='orange', linestyle='--', label='99% threshold')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PCA variance plot saved to {save_path}")
    
    plt.show()


def visualize_pca_loadings(loadings_df, n_components=5, n_features=10, save_path=None):
    """
    Visualize PCA loadings for top components
    
    Args:
        loadings_df: DataFrame with PCA loadings
        n_components: Number of components to show
        n_features: Number of top features to show per component
        save_path: Optional path to save the plot
    """
    n_components = min(n_components, len(loadings_df))
    
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
    if n_components == 1:
        axes = [axes]
    
    for i, pc in enumerate(loadings_df.index[:n_components]):
        ax = axes[i]
        
        # Get top features by absolute loading
        loadings = loadings_df.loc[pc]
        top_features = loadings.abs().sort_values(ascending=False).head(n_features)
        
        # Create horizontal bar plot
        colors = ['red' if loadings[feat] < 0 else 'blue' for feat in top_features.index]
        y_pos = range(len(top_features))
        
        ax.barh(y_pos, [loadings[feat] for feat in top_features.index], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Loading Value')
        ax.set_title(f'{pc} - Top {n_features} Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PCA loadings plot saved to {save_path}")
    
    plt.show()


def create_pca_biplot(X_pca, loadings_df, target=None, pc1=0, pc2=1, 
                     n_features=10, save_path=None):
    """
    Create PCA biplot showing samples and feature vectors
    
    Args:
        X_pca: PCA-transformed data
        loadings_df: DataFrame with PCA loadings
        target: Optional target variable for coloring points
        pc1: First principal component to plot (0-indexed)
        pc2: Second principal component to plot (0-indexed)
        n_features: Number of top features to show as vectors
        save_path: Optional path to save the plot
    """
    if X_pca.shape[1] <= max(pc1, pc2):
        logger.warning(f"Not enough components for biplot (have {X_pca.shape[1]}, need {max(pc1, pc2)+1})")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot samples
    if target is not None:
        scatter = ax.scatter(X_pca[:, pc1], X_pca[:, pc2], 
                           c=target, alpha=0.6, s=20, cmap='viridis')
        plt.colorbar(scatter)
    else:
        ax.scatter(X_pca[:, pc1], X_pca[:, pc2], alpha=0.6, s=20)
    
    # Get top features for both components
    pc1_name = f'PC{pc1+1}'
    pc2_name = f'PC{pc2+1}'
    
    loadings_pc1 = loadings_df.loc[pc1_name]
    loadings_pc2 = loadings_df.loc[pc2_name]
    
    # Select top features based on combined loading magnitude
    combined_loadings = np.sqrt(loadings_pc1**2 + loadings_pc2**2)
    top_features = combined_loadings.sort_values(ascending=False).head(n_features).index
    
    # Plot feature vectors
    scale_factor = 0.8 * max(np.abs(X_pca[:, [pc1, pc2]]).max())
    
    for feature in top_features:
        x_load = loadings_pc1[feature] * scale_factor
        y_load = loadings_pc2[feature] * scale_factor
        
        ax.arrow(0, 0, x_load, y_load, head_width=0.05*scale_factor, 
                head_length=0.05*scale_factor, fc='red', ec='red', alpha=0.8)
        ax.text(x_load*1.1, y_load*1.1, feature, fontsize=8, 
               ha='center', va='center', rotation=0)
    
    ax.set_xlabel(f'{pc1_name} ({loadings_df.index[pc1]})')
    ax.set_ylabel(f'{pc2_name} ({loadings_df.index[pc2]})')
    ax.set_title(f'PCA Biplot - {pc1_name} vs {pc2_name}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PCA biplot saved to {save_path}")
    
    plt.show()


def analyze_pca_components(pca, loadings_df, feature_names=None):
    """
    Analyze and interpret PCA components
    
    Args:
        pca: Fitted PCA model
        loadings_df: DataFrame with PCA loadings
        feature_names: Optional list of feature names
        
    Returns:
        dict: Analysis results
    """
    analysis = {
        'n_components': pca.n_components_,
        'total_variance_explained': pca.explained_variance_ratio_.sum(),
        'individual_variance': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'component_interpretations': {}
    }
    
    # Analyze each component
    for i, pc in enumerate(loadings_df.index):
        loadings = loadings_df.loc[pc]
        
        # Top positive and negative loadings
        positive_loadings = loadings[loadings > 0].sort_values(ascending=False).head(5)
        negative_loadings = loadings[loadings < 0].sort_values(ascending=True).head(5)
        
        analysis['component_interpretations'][pc] = {
            'variance_explained': pca.explained_variance_ratio_[i],
            'top_positive_features': positive_loadings.to_dict(),
            'top_negative_features': negative_loadings.to_dict(),
            'max_absolute_loading': loadings.abs().max(),
            'feature_with_max_loading': loadings.abs().idxmax()
        }
    
    logger.info("PCA component analysis completed")
    return analysis


def apply_tsne_visualization(X_pca, target=None, perplexity=30, random_state=42, save_path=None):
    """
    Apply t-SNE for 2D visualization of PCA-reduced data
    
    Args:
        X_pca: PCA-transformed data
        target: Optional target variable for coloring
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        save_path: Optional path to save the plot
        
    Returns:
        ndarray: 2D t-SNE embedding
    """
    logger.info(f"Applying t-SNE visualization (perplexity={perplexity})...")
    
    # Limit to reasonable number of samples for t-SNE
    if len(X_pca) > 5000:
        logger.warning(f"Large dataset ({len(X_pca)} samples), sampling 5000 for t-SNE")
        indices = np.random.choice(len(X_pca), 5000, replace=False)
        X_tsne_input = X_pca[indices]
        target_tsne = target[indices] if target is not None else None
    else:
        X_tsne_input = X_pca
        target_tsne = target
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    # Plot
    plt.figure(figsize=(10, 8))
    if target_tsne is not None:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_tsne, 
                            alpha=0.7, s=20, cmap='viridis')
        plt.colorbar(scatter)
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=20)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of PCA-reduced Data')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"t-SNE plot saved to {save_path}")
    
    plt.show()
    
    return X_tsne


def save_pca_results(pca, loadings_df, analysis, save_dir):
    """
    Save PCA results to files
    
    Args:
        pca: Fitted PCA model
        loadings_df: DataFrame with PCA loadings
        analysis: PCA analysis results
        save_dir: Directory to save results
    """
    import pickle
    import json
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PCA model
    with open(save_dir / 'pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    
    # Save loadings
    loadings_df.to_csv(save_dir / 'pca_loadings.csv')
    
    # Save analysis (convert numpy types for JSON serialization)
    analysis_json = {}
    for key, value in analysis.items():
        if isinstance(value, (np.ndarray, np.integer, np.floating)):
            analysis_json[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
        else:
            analysis_json[key] = value
    
    with open(save_dir / 'pca_analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    logger.info(f"PCA results saved to {save_dir}")