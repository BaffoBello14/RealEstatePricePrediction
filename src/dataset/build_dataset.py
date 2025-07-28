import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from ..utils.logger import get_logger
from ..utils.io import save_dataframe

logger = get_logger(__name__)

def extract_floor_features(floor_str: str) -> Dict[str, float]:
    """
    Estrae features multiple dai dati del piano per ML.
    
    Args:
        floor_str: Stringa del piano (es. "P1-3", "S1-T-2", "PT-ST-1")
        
    Returns:
        dict: Dizionario con features numeriche del piano
    """
    if pd.isna(floor_str) or floor_str in ['NULL', '', None]:
        return {
            'min_floor': np.nan,
            'max_floor': np.nan,
            'n_floors': np.float64(0),
            'has_basement': np.float64(0),
            'has_ground': np.float64(0),
            'has_upper': np.float64(0),
            'floor_span': np.float64(0),
            'floor_numeric_weighted': np.nan
        }
    
    floor_str = str(floor_str).strip().upper()
    
    # Mappatura dei piani a valori numerici
    floor_mapping = {
        'S2': -2, 'S1': -1, 'S': -1, 'SEMI': -1,
        'T': 0, 'PT': 0, 'RIAL': 0.5,
        'ST': -0.5,  # Seminterrato+Terra (duplex)
        'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5, 'P6': 6,
        'P7': 7, 'P8': 8, 'P9': 9, 'P10': 10, 'P11': 11, 'P12': 12
    }
    
    # Estrai tutti i componenti del piano
    floors_found = []
    
    # Pattern per trovare tutti i riferimenti ai piani
    patterns = [
        r'P(\d+)',      # P1, P2, etc.
        r'S(\d*)',      # S, S1, S2, etc.
        r'PT',          # Piano Terra
        r'T(?![0-9])',  # T (ma non T-1, T-2)
        r'ST',          # Seminterrato+Terra
        r'SEMI',        # Seminterrato
        r'RIAL'         # Rialzato
    ]
    
    # Cerca pattern specifici
    for pattern in patterns:
        matches = re.findall(pattern, floor_str)
        if pattern == r'P(\d+)':
            for match in matches:
                key = f'P{match}'
                if key in floor_mapping:
                    floors_found.append(floor_mapping[key])
        elif pattern == r'S(\d*)':
            for match in matches:
                if match == '':
                    floors_found.append(floor_mapping['S'])
                else:
                    key = f'S{match}'
                    if key in floor_mapping:
                        floors_found.append(floor_mapping[key])
        else:
            if re.search(pattern, floor_str):
                key = pattern.replace(r'(?![0-9])', '').replace(r'\b', '')
                if key in floor_mapping:
                    floors_found.append(floor_mapping[key])
    
    # Aggiungi numeri isolati come piani (es. "3", "4-5")
    isolated_numbers = re.findall(r'\b(\d+)\b', floor_str)
    for num in isolated_numbers:
        num_val = int(num)
        if 1 <= num_val <= 12:  # Considera solo numeri ragionevoli come piani
            floors_found.append(num_val)
    
    # Rimuovi duplicati e ordina
    floors_found = sorted(list(set(floors_found)))
    
    if not floors_found:
        # Fallback: prova a interpretare come numero singolo
        try:
            single_floor = float(floor_str)
            if -5 <= single_floor <= 15:
                floors_found = [single_floor]
        except:
            pass
    
    # Calcola features
    if floors_found:
        min_floor = float(min(floors_found))
        max_floor = float(max(floors_found))
        n_floors = np.float64(len(floors_found))
        floor_span = np.float64(max_floor - min_floor)
        
        # Calcola media pesata (piani più alti hanno più peso)
        weights = [f + 3 for f in floors_found]  # Shift per evitare pesi negativi
        floor_numeric_weighted = float(np.average(floors_found, weights=weights))
        
        # Flags booleani convertiti in 0/1
        has_basement = np.float64(1) if any(f < 0 for f in floors_found) else np.float64(0)
        has_ground = np.float64(1) if any(-0.5 <= f <= 0.5 for f in floors_found) else np.float64(0)
        has_upper = np.float64(1) if any(f >= 1 for f in floors_found) else np.float64(0)
        
    else:
        # Valori di default se non si riesce a parsare
        min_floor = max_floor = floor_numeric_weighted = np.nan
        n_floors = floor_span = np.float64(0)
        has_basement = has_ground = has_upper = np.float64(0)
    
    return {
        'min_floor': min_floor,
        'max_floor': max_floor,
        'n_floors': n_floors,
        'has_basement': has_basement,
        'has_ground': has_ground,
        'has_upper': has_upper,
        'floor_span': floor_span,
        'floor_numeric_weighted': floor_numeric_weighted
    }

def filter_coherent_acts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra atti coerenti (numero righe = TotaleImmobili).
    
    Args:
        df: DataFrame da filtrare
        
    Returns:
        DataFrame filtrato
    """
    logger.info("Filtro atti coerenti...")
    
    initial_rows = len(df)
    
    grouped = df.groupby("A_Id").agg(
        num_rows=('A_Id', 'size'),
        expected=('A_TotaleImmobili', 'first')
    ).reset_index()

    valid_ids = grouped[grouped["num_rows"] == grouped["expected"]]["A_Id"]
    df_filtered = df[df["A_Id"].isin(valid_ids)].copy()
    
    logger.info(f"Atti filtrati: {initial_rows} -> {len(df_filtered)} righe "
                f"({len(df_filtered)/initial_rows*100:.1f}% mantenute)")
    
    return df_filtered

def estimate_prices_dual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stima i prezzi degli immobili e ridistribuisce il prezzo totale.
    
    Args:
        df: DataFrame con dati degli immobili
        
    Returns:
        DataFrame con prezzi ridistribuiti
    """
    logger.info("Stima prezzi e ridistribuzione...")
    
    # Calcola prezzo stimato per m2 (media tra min e max OMI normale)
    df["prezzo_m2"] = (df["OV_ValoreMercatoMin_normale"] + df["OV_ValoreMercatoMax_normale"]) / 2
    df["prezzo_stimato_immobile"] = df["prezzo_m2"] * df["AI_Superficie"]

    # Calcola coefficiente di ridistribuzione per ogni atto
    prezzi = df.groupby("A_Id").agg(
        prezzo_stimato_totale=('prezzo_stimato_immobile', 'sum'),
        A_Prezzo=('A_Prezzo', 'first')
    ).reset_index()

    prezzi["coefficiente"] = prezzi["A_Prezzo"] / prezzi["prezzo_stimato_totale"]

    # Merge e calcolo prezzo finale ridistribuito
    df = df.merge(prezzi[["A_Id", "coefficiente"]], on="A_Id", how="left")
    df["AI_Prezzo_Ridistribuito"] = df["prezzo_stimato_immobile"] * df["coefficiente"]

    # Pulizia colonne temporanee
    df.drop(columns=["prezzo_m2", "prezzo_stimato_immobile", "coefficiente"], inplace=True)
    
    logger.info(f"Prezzi ridistribuiti calcolati per {len(df)} immobili")
    
    return df

def process_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elabora le features del piano e le aggiunge al DataFrame.
    
    Args:
        df: DataFrame con colonna AI_Piano
        
    Returns:
        DataFrame con features del piano aggiunte
    """
    if 'AI_Piano' not in df.columns:
        logger.warning("Colonna AI_Piano non trovata nel DataFrame")
        return df
    
    logger.info("Elaborazione features del piano...")
    
    # Applica la funzione extract_floor_features
    floor_features = df['AI_Piano'].apply(extract_floor_features)
    
    # Converti in DataFrame
    floor_df = pd.DataFrame(floor_features.tolist())
    
    # Aggiungi prefisso per evitare conflitti
    floor_df.columns = ['floor_' + col for col in floor_df.columns]
    
    # Concatena con il DataFrame originale
    df_with_floors = pd.concat([df, floor_df], axis=1)
    
    # Statistiche di debug
    logger.info(f"Features del piano aggiunte: {list(floor_df.columns)}")
    
    non_null_floors = df_with_floors['floor_floor_numeric_weighted'].notna().sum()
    logger.info(f"Valori non-null in floor_numeric_weighted: {non_null_floors}/{len(df_with_floors)}")
    
    if non_null_floors > 0:
        min_floor = df_with_floors['floor_floor_numeric_weighted'].min()
        max_floor = df_with_floors['floor_floor_numeric_weighted'].max()
        logger.info(f"Range floor_numeric_weighted: {min_floor:.2f} - {max_floor:.2f}")
    
    return df_with_floors

def remove_all_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove colonne interamente NaN.
    
    Args:
        df: DataFrame da pulire
        
    Returns:
        DataFrame senza colonne interamente NaN
    """
    logger.info("Rimozione colonne interamente NaN...")
    
    all_nan_cols = df.columns[df.isna().all()].tolist()
    
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        logger.info(f"Colonne interamente NaN rimosse: {all_nan_cols}")
    else:
        logger.info("Nessuna colonna interamente NaN trovata")
    
    return df

def build_dataset(raw_data_path: str, output_path: str) -> pd.DataFrame:
    """
    Costruisce il dataset finale partendo dai dati grezzi.
    
    Args:
        raw_data_path: Path ai dati grezzi
        output_path: Path dove salvare il dataset processato
        
    Returns:
        DataFrame del dataset finale
    """
    logger.info(f"Costruzione dataset da {raw_data_path}")
    
    try:
        # Carica dati grezzi
        from ..utils.io import load_dataframe
        df = load_dataframe(raw_data_path)
        logger.info(f"Dati grezzi caricati: {df.shape}")
        
        # Applica trasformazioni
        df = filter_coherent_acts(df)
        df = estimate_prices_dual(df)
        df = process_floor_features(df)
        df = remove_all_nan_columns(df)
        
        # Salva dataset processato
        save_dataframe(df, output_path, format='parquet')
        logger.info(f"Dataset finale salvato: {output_path} (shape: {df.shape})")
        
        return df
        
    except Exception as e:
        logger.error(f"Errore nella costruzione del dataset: {e}")
        raise