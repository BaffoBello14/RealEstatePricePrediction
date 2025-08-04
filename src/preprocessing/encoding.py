import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder
from typing import Tuple, Dict, Any, List
from ..utils.logger import get_logger
from .data_cleaning_core import convert_to_numeric_unified

logger = get_logger(__name__)



def advanced_categorical_encoding(
    df: pd.DataFrame, 
    target_column: str, 
    low_card_threshold: int = 10,
    high_card_max: int = 100,
    random_state: int = 42,
    apply_target_encoding: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encoding avanzato per variabili categoriche.
    
    Args:
        df: DataFrame con variabili categoriche
        target_column: Nome della colonna target
        low_card_threshold: Soglia per bassa cardinalità (one-hot)
        high_card_max: Soglia massima per alta cardinalità (target encoding)
        random_state: Seed per riproducibilità
        apply_target_encoding: Se True applica target encoding (ATTENZIONE: può causare data leakage pre-split)
        
    Returns:
        Tuple con DataFrame encodato e info degli encoder
    """
    logger.info("Encoding variabili categoriche...")
    
    cat_cols = df.select_dtypes(include='object').columns
    cat_cols = [col for col in cat_cols if col != target_column]  # Escludi target se categorico
    
    if len(cat_cols) == 0:
        logger.info("Nessuna variabile categorica da encodare")
        return df, {}
    
    encoding_info = {}
    
    # Separazione per cardinalità
    low_card = [col for col in cat_cols if df[col].nunique() < low_card_threshold]
    high_card = [
        col for col in cat_cols 
        if df[col].nunique() >= low_card_threshold and df[col].nunique() < high_card_max
    ]
    very_high_card = [col for col in cat_cols if df[col].nunique() >= high_card_max]
    
    logger.info(f"Variabili categoriche - Bassa cardinalità: {len(low_card)}, "
               f"Alta: {len(high_card)}, Molto alta: {len(very_high_card)}")
    
    # One-hot encoding per bassa cardinalità
    if low_card:
        logger.info(f"One-hot encoding per {len(low_card)} variabili a bassa cardinalità")
        df = pd.get_dummies(df, columns=low_card, drop_first=True)
        encoding_info['one_hot'] = low_card
    
    # Target encoding per alta cardinalità (solo se target è numerico)
    if high_card and df[target_column].dtype in ['int64', 'float64'] and apply_target_encoding:
        logger.info(f"Target encoding per {len(high_card)} variabili ad alta cardinalità")
        target_encoders = {}
        
        for col in high_card:
            encoder = TargetEncoder(random_state=random_state)
            df[col] = encoder.fit_transform(df[[col]], df[target_column])
            target_encoders[col] = encoder
        
        encoding_info['target_encoding'] = target_encoders
    
    # Label encoding per cardinalità molto alta (fallback)
    if very_high_card:
        logger.warning(f"Label encoding per {len(very_high_card)} variabili a cardinalità molto alta")
        label_encoders = {}
        
        for col in very_high_card:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            label_encoders[col] = encoder
        
        encoding_info['label_encoding'] = label_encoders
    
    return df, encoding_info

def encode_features(
    df: pd.DataFrame,
    target_column: str,
    low_card_threshold: int = 10,
    high_card_max: int = 100,
    random_state: int = 42,
    apply_target_encoding: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applica tutti gli encoding alle feature categoriche.
    
    Args:
        df: DataFrame con feature da encodare
        target_column: Nome della colonna target
        low_card_threshold: Soglia per bassa cardinalità
        high_card_max: Soglia massima per alta cardinalità
        random_state: Seed per riproducibilità
        apply_target_encoding: Se True applica target encoding (ATTENZIONE: può causare data leakage pre-split)
        
    Returns:
        Tuple con DataFrame encodato e informazioni sugli encoder
    """
    logger.info("Avvio encoding delle feature...")
    
    # Conversione automatica a numerico
    df, conversion_info = convert_to_numeric_unified(df, target_column, threshold=0.8)
    converted_cols = conversion_info.get('successful_conversions', [])
    
    # Encoding categoriche avanzato
    df, encoding_info = advanced_categorical_encoding(
        df, target_column, low_card_threshold, high_card_max, random_state, apply_target_encoding
    )
    
    encoding_info['converted_to_numeric'] = converted_cols
    
    logger.info(f"Encoding completato. Shape finale: {df.shape}")
    
    return df, encoding_info