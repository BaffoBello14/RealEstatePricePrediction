"""
Outlier detection module with support for grouped detection by categorical variables
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)


def detect_outliers_zscore(data, threshold=3.0):
    """
    Detect outliers using Z-score method
    
    Args:
        data: Array-like, data to check for outliers
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    z_scores = np.abs(zscore(data, nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Args:
        data: Array-like, data to check for outliers
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Detect outliers using Isolation Forest
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of outliers
        random_state: Random state for reproducibility
        
    Returns:
        Boolean array indicating outliers
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    outlier_labels = iso_forest.fit_predict(X)
    return outlier_labels == -1


def detect_outliers_grouped(df, target_col, group_col, X_scaled, config):
    """
    Detect outliers grouped by categorical variable
    
    Args:
        df: DataFrame with target and grouping columns
        target_col: Name of target column
        group_col: Name of grouping column
        X_scaled: Scaled feature matrix for isolation forest
        config: Configuration dictionary with outlier detection parameters
        
    Returns:
        tuple: (target_transformed, outliers_mask, outlier_detector)
    """
    logger.info(f"Detecting outliers grouped by {group_col}...")
    
    # Log transform target
    target_log = np.log1p(df[target_col])
    
    # Initialize outlier mask
    outliers_mask = np.zeros(len(df), dtype=bool)
    
    # Get unique groups
    unique_groups = df[group_col].dropna().unique()
    logger.info(f"Found {len(unique_groups)} unique groups in {group_col}")
    
    # Track outliers by method and group
    outlier_stats = {
        'z_score': {},
        'iqr': {},
        'isolation_forest': {},
        'combined': {}
    }
    
    for group in unique_groups:
        group_mask = df[group_col] == group
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) < config['min_group_size']:
            logger.debug(f"Skipping group {group}: only {len(group_indices)} samples")
            continue
            
        group_target = target_log[group_mask]
        group_X = X_scaled[group_mask]
        
        # Z-score outliers for this group
        z_outliers = detect_outliers_zscore(group_target, config['z_threshold'])
        
        # IQR outliers for this group
        iqr_outliers = detect_outliers_iqr(group_target, config['iqr_multiplier'])
        
        # Isolation Forest outliers for this group
        if len(group_X) >= 10:  # Minimum samples for isolation forest
            iso_outliers = detect_outliers_isolation_forest(
                group_X, 
                config['isolation_contamination'],
                42
            )
        else:
            iso_outliers = np.zeros(len(group_X), dtype=bool)
        
        # Combine methods (outlier if detected by at least min_methods)
        outlier_counts = (
            z_outliers.astype(int) + 
            iqr_outliers.astype(int) + 
            iso_outliers.astype(int)
        )
        group_combined_outliers = outlier_counts >= config['min_methods_outlier']
        
        # Update global outlier mask
        outliers_mask[group_indices] = group_combined_outliers
        
        # Store statistics
        outlier_stats['z_score'][group] = z_outliers.sum()
        outlier_stats['iqr'][group] = iqr_outliers.sum()
        outlier_stats['isolation_forest'][group] = iso_outliers.sum()
        outlier_stats['combined'][group] = group_combined_outliers.sum()
        
        logger.info(f"Group {group} ({len(group_indices)} samples): "
                   f"Z-score: {z_outliers.sum()}, IQR: {iqr_outliers.sum()}, "
                   f"Isolation: {iso_outliers.sum()}, Combined: {group_combined_outliers.sum()}")
    
    # Overall statistics
    total_outliers = outliers_mask.sum()
    logger.info(f"Total outliers detected: {total_outliers} ({total_outliers/len(df)*100:.2f}%)")
    
    # Create a simple isolation forest detector for the interface
    # (This is mainly for compatibility with the existing pipeline)
    iso_forest = IsolationForest(
        contamination=config['isolation_contamination'],
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    
    return target_log, outliers_mask, iso_forest, outlier_stats


def detect_outliers_global(df, target_col, X_scaled, config):
    """
    Detect outliers globally (not grouped)
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        X_scaled: Scaled feature matrix
        config: Configuration dictionary with outlier detection parameters
        
    Returns:
        tuple: (target_transformed, outliers_mask, outlier_detector)
    """
    logger.info("Detecting outliers globally...")
    
    # Log transform target
    target_log = np.log1p(df[target_col])
    
    # Z-score outliers
    z_outliers = detect_outliers_zscore(target_log, config['z_threshold'])
    
    # IQR outliers
    iqr_outliers = detect_outliers_iqr(target_log, config['iqr_multiplier'])
    
    # Isolation Forest outliers
    iso_outliers = detect_outliers_isolation_forest(
        X_scaled, 
        config['isolation_contamination'],
        42
    )
    
    # Combine methods
    outlier_counts = (
        z_outliers.astype(int) + 
        iqr_outliers.astype(int) + 
        iso_outliers.astype(int)
    )
    combined_outliers = outlier_counts >= config['min_methods_outlier']
    
    # Statistics
    logger.info(f"Z-Score (>{config['z_threshold']}): {z_outliers.sum()} outliers "
               f"({z_outliers.sum()/len(target_log)*100:.2f}%)")
    logger.info(f"IQR ({config['iqr_multiplier']}x): {iqr_outliers.sum()} outliers "
               f"({iqr_outliers.sum()/len(target_log)*100:.2f}%)")
    logger.info(f"Isolation Forest: {iso_outliers.sum()} outliers "
               f"({iso_outliers.sum()/len(target_log)*100:.2f}%)")
    logger.info(f"Combined (≥{config['min_methods_outlier']} methods): {combined_outliers.sum()} outliers "
               f"({combined_outliers.sum()/len(target_log)*100:.2f}%)")
    
    # Create isolation forest detector
    iso_forest = IsolationForest(
        contamination=config['isolation_contamination'],
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    
    return target_log, combined_outliers, iso_forest


def visualize_outliers(df, target_original, target_log, outliers_mask, target_col, group_col=None):
    """
    Visualize outliers with original and transformed target
    
    Args:
        df: DataFrame
        target_original: Original target values
        target_log: Log-transformed target values
        outliers_mask: Boolean mask for outliers
        target_col: Name of target column
        group_col: Optional grouping column for colored visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Original target
    axes[0,0].hist(target_original, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title(f'Target Originale\nSkewness: {target_original.skew():.3f}')
    axes[0,0].set_xlabel(target_col)
    
    axes[0,1].boxplot(target_original, vert=True)
    axes[0,1].set_title('Boxplot Originale')
    axes[0,1].set_ylabel(target_col)
    
    # Scatter plot original
    if group_col and group_col in df.columns:
        # Color by group
        groups = df[group_col].fillna('Unknown')
        unique_groups = groups.unique()[:10]  # Limit to first 10 groups for visibility
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        
        for i, group in enumerate(unique_groups):
            group_mask = groups == group
            axes[0,2].scatter(
                np.where(group_mask)[0], 
                target_original[group_mask],
                alpha=0.6, 
                s=20, 
                color=colors[i],
                label=f'{group}'
            )
        axes[0,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0,2].scatter(range(len(target_original)), target_original, alpha=0.6, s=20)
    
    axes[0,2].set_title('Target Originale vs Index')
    axes[0,2].set_xlabel('Index')
    axes[0,2].set_ylabel(target_col)
    
    # Row 2: Log-transformed target with outliers
    axes[1,0].hist(target_log[~outliers_mask], bins=50, alpha=0.7, label='Normal', color='blue')
    if outliers_mask.sum() > 0:
        axes[1,0].hist(target_log[outliers_mask], bins=20, alpha=0.7, label='Outliers', color='red')
    axes[1,0].set_title(f'Target Log-trasformato\nSkewness: {target_log.skew():.3f}')
    axes[1,0].set_xlabel(f'log({target_col} + 1)')
    axes[1,0].legend()
    
    axes[1,1].boxplot(target_log, vert=True)
    axes[1,1].set_title('Boxplot Log-trasformato')
    axes[1,1].set_ylabel(f'log({target_col} + 1)')
    
    # Scatter plot log-transformed with outliers highlighted
    normal_mask = ~outliers_mask
    axes[1,2].scatter(
        np.where(normal_mask)[0], 
        target_log[normal_mask], 
        alpha=0.6, 
        s=20, 
        color='blue', 
        label='Normal'
    )
    if outliers_mask.sum() > 0:
        axes[1,2].scatter(
            np.where(outliers_mask)[0], 
            target_log[outliers_mask], 
            alpha=0.8, 
            s=30, 
            color='red', 
            label='Outliers'
        )
    axes[1,2].set_title('Target Log vs Index (Outliers evidenziati)')
    axes[1,2].set_xlabel('Index')
    axes[1,2].set_ylabel(f'log({target_col} + 1)')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()


def visualize_outliers_by_group(df, target_log, outliers_mask, group_col, max_groups=12):
    """
    Visualize outliers by categorical group
    
    Args:
        df: DataFrame
        target_log: Log-transformed target values
        outliers_mask: Boolean mask for outliers
        group_col: Name of grouping column
        max_groups: Maximum number of groups to show
    """
    # Get top groups by frequency
    top_groups = df[group_col].value_counts().head(max_groups).index
    
    # Create subplots
    n_cols = 4
    n_rows = (len(top_groups) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, group in enumerate(top_groups):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        group_mask = df[group_col] == group
        group_target = target_log[group_mask]
        group_outliers = outliers_mask[group_mask]
        
        # Histogram with outliers highlighted
        ax.hist(group_target[~group_outliers], bins=20, alpha=0.7, 
                label='Normal', color='blue', edgecolor='black')
        if group_outliers.sum() > 0:
            ax.hist(group_target[group_outliers], bins=10, alpha=0.7, 
                    label='Outliers', color='red', edgecolor='black')
        
        ax.set_title(f'Group: {group}\n(n={group_mask.sum()}, outliers={group_outliers.sum()})')
        ax.set_xlabel('Log Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(top_groups), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()