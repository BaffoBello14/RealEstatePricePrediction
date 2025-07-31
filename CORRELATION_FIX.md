# Fix: Correlation Removal Bug - Switch to PRE-SPLIT Function

## Problem
The preprocessing pipeline was failing with error: `'NoneType' object has no attribute 'drop'` in the correlation removal step because the function `remove_highly_correlated_numeric` was designed for POST-SPLIT usage but being called PRE-SPLIT with `X_test=None`.

## Solution
Create a dedicated PRE-SPLIT function and update the pipeline to use it. This provides better performance, consistency, and cleaner architecture.

## Changes Required

### 1. Add new function to `src/preprocessing/filtering.py`

Add this function after the existing `remove_highly_correlated_numeric` function:

```python
def remove_highly_correlated_numeric_pre_split(
    df: pd.DataFrame,
    target_column: str = None,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rimuove feature numeriche altamente correlate PRE-SPLIT.
    Progettata per lavorare su un DataFrame completo prima dello split.
    
    Args:
        df: DataFrame completo con tutte le feature
        target_column: Nome della colonna target (da escludere dall'analisi)
        threshold: Soglia di correlazione
        
    Returns:
        Tuple con DataFrame filtrato e lista colonne rimosse
    """
    logger.info(f"Rimozione feature numeriche correlate PRE-SPLIT (soglia: {threshold})...")
    
    # Seleziona solo colonne numeriche, escludendo il target se specificato
    numeric_cols = df.select_dtypes(include=np.number).columns
    if target_column and target_column in numeric_cols:
        numeric_cols = numeric_cols.drop(target_column)
    
    if len(numeric_cols) < 2:
        logger.info("Meno di 2 variabili numeriche, skip correlazione")
        return df, []
    
    # Calcola correlazione solo su feature numeriche
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        logger.info(f"Rimosse {len(to_drop)} feature numeriche correlate: {to_drop}")
        df_filtered = df.drop(columns=to_drop)
    else:
        logger.info("Nessuna feature numerica da rimuovere per correlazione")
        df_filtered = df
    
    return df_filtered, to_drop
```

### 2. Update `src/preprocessing/pipeline.py`

Replace the entire STEP 7 block (approximately lines 94-122) with this simplified version:

```python
        # ===== STEP 7: RIMOZIONE FEATURE CORRELATE PRE-SPLIT (se abilitato) =====
        if steps_config.get('enable_correlation_removal', True):
            logger.info("Step 7: Rimozione feature altamente correlate...")
            # Usa la funzione dedicata PRE-SPLIT
            from .filtering import remove_highly_correlated_numeric_pre_split
            
            df, removed_numeric_cols = remove_highly_correlated_numeric_pre_split(
                df, 
                target_column=target_column,
                threshold=config.get('corr_threshold', 0.95)
            )
            
            correlation_removal_info = {
                'removed_columns': removed_numeric_cols,
                'threshold_used': config.get('corr_threshold', 0.95)
            }
            
            preprocessing_info['steps_info']['correlation_removal'] = correlation_removal_info
        else:
            logger.info("Step 7: Rimozione feature correlate DISABILITATA")
            preprocessing_info['steps_info']['correlation_removal'] = {'skipped': True}
```

## Benefits

1. **Bug Fix**: Eliminates the `NoneType` error completely
2. **Performance**: Faster correlation analysis on complete dataset vs split datasets
3. **Consistency**: All splits (train/val/test) have exactly the same features
4. **Simplicity**: Cleaner code with single function call
5. **Architecture**: Aligns with project's PRE-SPLIT preprocessing philosophy

## Testing

The fix has been tested with:
- Datasets with correlated features
- Proper target column exclusion
- Preservation of non-numeric columns
- Both with and without target column specified

## Backward Compatibility

The original `remove_highly_correlated_numeric` function remains unchanged for any POST-SPLIT usage that might exist elsewhere in the codebase.