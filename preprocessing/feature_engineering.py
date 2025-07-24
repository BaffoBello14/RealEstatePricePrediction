"""
Feature engineering and selection module
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import logging

logger = logging.getLogger(__name__)


def cramers_v(confusion_matrix):
    """
    Calculate Cramer's V statistic for categorical-categorical association
    
    Args:
        confusion_matrix: Contingency table
        
    Returns:
        Cramer's V value (0-1)
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def identify_duplicate_columns(df):
    """
    Identify duplicate columns in DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names to drop (keeps first occurrence)
    """
    logger.info("Identifying duplicate columns...")
    
    cols = df.columns.tolist()
    to_drop = set()
    
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
            
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
                
            # Compare columns (handling NaN values)
            s1 = df[cols[i]].fillna('##NAN##')
            s2 = df[cols[j]].fillna('##NAN##')
            
            if s1.equals(s2):
                to_drop.add(cols[j])
                logger.debug(f"Duplicate found: {cols[j]} == {cols[i]}")
    
    logger.info(f"Found {len(to_drop)} duplicate columns")
    return list(to_drop)


def identify_constant_columns(df, threshold=0.99):
    """
    Identify constant or near-constant columns
    
    Args:
        df: Input DataFrame
        threshold: Minimum fraction of same values to consider constant
        
    Returns:
        List of column names to drop
    """
    logger.info("Identifying constant columns...")
    
    constant_cols = []
    
    for col in df.columns:
        # Calculate the most frequent value proportion
        if df[col].dtype == 'object':
            # For categorical columns
            most_frequent_prop = df[col].value_counts(dropna=False).iloc[0] / len(df)
        else:
            # For numerical columns
            most_frequent_prop = df[col].value_counts(dropna=False).iloc[0] / len(df)
        
        if most_frequent_prop >= threshold:
            constant_cols.append(col)
            logger.debug(f"Constant column: {col} ({most_frequent_prop:.3f} same values)")
    
    logger.info(f"Found {len(constant_cols)} constant columns")
    return constant_cols


def identify_low_variance_columns(df, threshold=0.01):
    """
    Identify low variance numerical columns
    
    Args:
        df: Input DataFrame (should contain only numerical columns)
        threshold: Minimum variance threshold
        
    Returns:
        List of column names to drop
    """
    logger.info("Identifying low variance columns...")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        logger.info("No numerical columns found for variance analysis")
        return []
    
    # Use sklearn's VarianceThreshold
    variance_selector = VarianceThreshold(threshold=threshold)
    
    # Fit on numerical data (handle NaN values)
    numerical_data = df[numerical_cols].fillna(df[numerical_cols].median())
    variance_selector.fit(numerical_data)
    
    # Get columns to keep
    selected_features = variance_selector.get_support()
    low_variance_cols = numerical_cols[~selected_features].tolist()
    
    logger.info(f"Found {len(low_variance_cols)} low variance columns")
    return low_variance_cols


def identify_highly_correlated_numerical(df, threshold=0.95):
    """
    Identify highly correlated numerical columns
    
    Args:
        df: Input DataFrame
        threshold: Correlation threshold above which to remove features
        
    Returns:
        List of column names to drop
    """
    logger.info("Identifying highly correlated numerical columns...")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        logger.info("Less than 2 numerical columns found")
        return []
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr().abs()
    
    # Find pairs with correlation above threshold
    high_corr_pairs = []
    to_drop = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                
                # Keep the column with higher variance
                var1 = df[col1].var()
                var2 = df[col2].var()
                
                if var1 >= var2:
                    to_drop.add(col2)
                else:
                    to_drop.add(col1)
    
    logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs")
    logger.info(f"Removing {len(to_drop)} columns due to high correlation")
    
    return list(to_drop)


def identify_highly_correlated_categorical(df, threshold=0.85):
    """
    Identify highly correlated categorical columns using Cramer's V
    
    Args:
        df: Input DataFrame
        threshold: Cramer's V threshold above which to remove features
        
    Returns:
        List of column names to drop
    """
    logger.info("Identifying highly correlated categorical columns...")
    
    # Select only categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) < 2:
        logger.info("Less than 2 categorical columns found")
        return []
    
    high_corr_pairs = []
    to_drop = set()
    
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            col1, col2 = categorical_cols[i], categorical_cols[j]
            
            try:
                # Create contingency table
                contingency_table = pd.crosstab(
                    df[col1].fillna('Missing'), 
                    df[col2].fillna('Missing')
                )
                
                # Calculate Cramer's V
                cramers_v_value = cramers_v(contingency_table.values)
                
                if cramers_v_value >= threshold:
                    high_corr_pairs.append((col1, col2, cramers_v_value))
                    
                    # Keep the column with more unique values
                    nunique1 = df[col1].nunique()
                    nunique2 = df[col2].nunique()
                    
                    if nunique1 >= nunique2:
                        to_drop.add(col2)
                    else:
                        to_drop.add(col1)
                        
            except Exception as e:
                logger.debug(f"Error calculating Cramer's V for {col1} vs {col2}: {e}")
                continue
    
    logger.info(f"Found {len(high_corr_pairs)} highly correlated categorical pairs")
    logger.info(f"Removing {len(to_drop)} categorical columns due to high correlation")
    
    return list(to_drop)


def remove_features(df, config):
    """
    Remove features based on configuration
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary with removal settings
        
    Returns:
        tuple: (cleaned_df, removal_report)
    """
    logger.info("Starting feature removal process...")
    
    removal_report = {
        'original_shape': df.shape,
        'duplicate_columns': [],
        'constant_columns': [],
        'low_variance_columns': [],
        'high_corr_numerical': [],
        'high_corr_categorical': []
    }
    
    df_clean = df.copy()
    
    # Remove duplicate columns
    if config.get('remove_duplicates', True):
        duplicate_cols = identify_duplicate_columns(df_clean)
        removal_report['duplicate_columns'] = duplicate_cols
        df_clean = df_clean.drop(columns=duplicate_cols)
        logger.info(f"Removed {len(duplicate_cols)} duplicate columns")
    
    # Remove constant columns
    if config.get('remove_constants', True):
        constant_cols = identify_constant_columns(df_clean)
        removal_report['constant_columns'] = constant_cols
        df_clean = df_clean.drop(columns=constant_cols)
        logger.info(f"Removed {len(constant_cols)} constant columns")
    
    # Remove low variance columns
    if config.get('remove_low_variance', True):
        low_var_cols = identify_low_variance_columns(
            df_clean, 
            config.get('variance_threshold', 0.01)
        )
        removal_report['low_variance_columns'] = low_var_cols
        df_clean = df_clean.drop(columns=low_var_cols)
        logger.info(f"Removed {len(low_var_cols)} low variance columns")
    
    # Remove highly correlated numerical columns
    if config.get('remove_high_correlation', True):
        high_corr_num = identify_highly_correlated_numerical(
            df_clean, 
            config.get('corr_threshold', 0.95)
        )
        removal_report['high_corr_numerical'] = high_corr_num
        df_clean = df_clean.drop(columns=high_corr_num)
        logger.info(f"Removed {len(high_corr_num)} highly correlated numerical columns")
    
    # Remove highly correlated categorical columns
    if config.get('remove_high_correlation', True):
        high_corr_cat = identify_highly_correlated_categorical(
            df_clean, 
            config.get('cramer_threshold', 0.85)
        )
        removal_report['high_corr_categorical'] = high_corr_cat
        df_clean = df_clean.drop(columns=high_corr_cat)
        logger.info(f"Removed {len(high_corr_cat)} highly correlated categorical columns")
    
    removal_report['final_shape'] = df_clean.shape
    removal_report['total_removed'] = df.shape[1] - df_clean.shape[1]
    
    logger.info(f"Feature removal complete: {df.shape[1]} -> {df_clean.shape[1]} columns "
               f"({removal_report['total_removed']} removed)")
    
    return df_clean, removal_report


def encode_categorical_features(df, target, config):
    """
    Encode categorical features using appropriate strategies
    
    Args:
        df: Input DataFrame
        target: Target variable for target encoding
        config: Configuration dictionary with encoding settings
        
    Returns:
        tuple: (encoded_df, encoding_info)
    """
    logger.info("Encoding categorical features...")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        logger.info("No categorical columns found")
        return df.copy(), {}
    
    encoding_info = {
        'label_encoded': [],
        'onehot_encoded': [],
        'target_encoded': [],
        'encoders': {}
    }
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        nunique = df[col].nunique()
        
        if nunique <= config.get('low_cardinality_threshold', 20):
            # One-hot encoding for low cardinality
            logger.debug(f"One-hot encoding {col} (cardinality: {nunique})")
            
            # Simple label encoding first (for compatibility)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].fillna('Missing').astype(str))
            
            encoding_info['label_encoded'].append(col)
            encoding_info['encoders'][col] = le
            
        elif nunique <= config.get('high_cardinality_max', 100):
            # Target encoding for medium cardinality
            logger.debug(f"Target encoding {col} (cardinality: {nunique})")
            
            te = TargetEncoder(
                smooth=config.get('target_smoothing', 10),
                random_state=42
            )
            df_encoded[col] = te.fit_transform(
                df[col].fillna('Missing').values.reshape(-1, 1), 
                target
            ).ravel()
            
            encoding_info['target_encoded'].append(col)
            encoding_info['encoders'][col] = te
            
        else:
            # Label encoding for very high cardinality
            logger.debug(f"Label encoding {col} (cardinality: {nunique})")
            
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].fillna('Missing').astype(str))
            
            encoding_info['label_encoded'].append(col)
            encoding_info['encoders'][col] = le
    
    logger.info(f"Encoded {len(categorical_cols)} categorical columns: "
               f"{len(encoding_info['label_encoded'])} label encoded, "
               f"{len(encoding_info['target_encoded'])} target encoded")
    
    return df_encoded, encoding_info


def impute_missing_values(df, config):
    """
    Impute missing values in numerical and categorical columns
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary with imputation settings
        
    Returns:
        tuple: (imputed_df, imputers_dict)
    """
    logger.info("Imputing missing values...")
    
    df_imputed = df.copy()
    imputers = {}
    
    # Numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(
            strategy=config.get('numerical_strategy', 'median')
        )
        df_imputed[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        imputers['numerical'] = num_imputer
        
        missing_before = df[numerical_cols].isnull().sum().sum()
        logger.info(f"Imputed {missing_before} missing values in {len(numerical_cols)} numerical columns")
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(
            strategy=config.get('categorical_strategy', 'most_frequent')
        )
        df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols].astype(str))
        imputers['categorical'] = cat_imputer
        
        missing_before = df[categorical_cols].isnull().sum().sum()
        logger.info(f"Imputed {missing_before} missing values in {len(categorical_cols)} categorical columns")
    
    return df_imputed, imputers


def scale_features(X, config, fitted_scaler=None):
    """
    Scale features using specified method
    
    Args:
        X: Feature matrix
        config: Configuration dictionary with scaling settings
        fitted_scaler: Pre-fitted scaler (for test set)
        
    Returns:
        tuple: (scaled_X, scaler)
    """
    method = config.get('method', 'standard')
    
    if fitted_scaler is not None:
        # Transform using pre-fitted scaler
        X_scaled = fitted_scaler.transform(X)
        return X_scaled, fitted_scaler
    
    # Fit new scaler
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Features scaled using {method} scaler")
    
    return X_scaled, scaler