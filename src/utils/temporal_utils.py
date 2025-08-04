"""
Temporal utilities for time-series data handling
"""
import pandas as pd

def temporal_sort_by_year_month(df, year_column, month_column):
    """
    Sort DataFrame by year and month columns for temporal splits.
    
    Args:
        df: DataFrame to sort
        year_column: Name of year column
        month_column: Name of month column
        
    Returns:
        Sorted DataFrame
    """
    # Create a datetime-like column for sorting
    df_copy = df.copy()
    df_copy['_temporal_sort_key'] = df_copy[year_column] * 12 + df_copy[month_column]
    df_sorted = df_copy.sort_values('_temporal_sort_key').drop(columns=['_temporal_sort_key'])
    
    return df_sorted