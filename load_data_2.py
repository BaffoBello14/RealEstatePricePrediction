import json
import pandas as pd
import numpy as np
import re
from db.connection import get_engine

def extract_floor_features(floor_str):
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
            'has_basement': np.float64(0),  # 0/1 invece di bool
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

def build_select_clause_dual_omi(schema, selected_aliases=None):
    """
    Costruisce la SELECT SQL includendo i valori OMI 'Normale', 'Ottimo' e 'Scadente' con suffissi diversi.
    """
    selects = []

    for table_name, table_info in schema.items():
        alias = table_info.get("alias", table_name[:2].upper())

        if selected_aliases is not None and alias not in selected_aliases:
            continue

        for col in table_info["columns"]:
            if not col.get("retrieve", False):
                continue

            col_name = col["name"]
            col_type = col["type"].lower()

            if alias == "OV":
                for stato_alias, stato_suffix in [("OVN", "normale"), ("OVO", "ottimo"), ("OVS", "scadente")]:
                    selects.append(f"{stato_alias}.{col_name} AS {alias}_{col_name}_{stato_suffix}")
            else:
                if col_type in ["geometry", "geography"]:
                    selects.append(f"{alias}.{col_name}.STAsText() AS {alias}_{col_name}")
                else:
                    selects.append(f"{alias}.{col_name} AS {alias}_{col_name}")

    return ",\n ".join(selects)

def generate_query_dual_omi(select_clause):
    return f"""
    SELECT
        {select_clause}
    FROM
        Atti A
        INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
        INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
        INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
        INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
        INNER JOIN ParticelleCatastali_OmiZone PCOZ ON PCOZ.IdParticella = PC.Id
        INNER JOIN OmiZone OZ ON PCOZ.IdZona = OZ.Id
        -- Join su OmiValori per stato Normale (necessaria)
        INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
            AND OVN.Stato = 'Normale'
            AND AI.IdTipologiaEdilizia = OVN.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Ottimo (opzionale)
        LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
            AND OVO.Stato = 'Ottimo'
            AND AI.IdTipologiaEdilizia = OVO.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Scadente (opzionale)
        LEFT JOIN OmiValori OVS ON OZ.Id = OVS.IdZona
            AND OVS.Stato = 'Scadente'
            AND AI.IdTipologiaEdilizia = OVS.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
    WHERE 
        A.TotaleFabbricati = A.TotaleImmobili
        AND AI.IdTipologiaEdilizia IS NOT NULL
        AND A.Id NOT IN (
            SELECT IdAtto
            FROM AttiImmobili
            WHERE Superficie IS NULL
            OR IdTipologiaEdilizia IS NULL
        )
    ORDER BY A.Id
    """

def clean_dataframe(df):
    df.replace('', np.nan, inplace=True)
    return df

def drop_duplicate_columns(df):
    cols = df.columns.tolist()
    to_drop = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue

            s1 = df[cols[i]].fillna('##nan##')
            s2 = df[cols[j]].fillna('##nan##')

            if s1.equals(s2):
                to_drop.add(cols[j])

    if to_drop:
        df = df.drop(columns=list(to_drop))
        print(f"Colonne duplicate rimosse: {to_drop}")
    else:
        print("Nessuna colonna duplicata trovata.")

    return df

def filter_coherent_acts(df):
    grouped = df.groupby("A_Id").agg(
        num_rows=('A_Id', 'size'),
        expected=('A_TotaleImmobili', 'first')
    ).reset_index()

    valid_ids = grouped[grouped["num_rows"] == grouped["expected"]]["A_Id"]
    return df[df["A_Id"].isin(valid_ids)].copy()

def estimate_prices_dual(df):
    df["prezzo_m2"] = (df["OV_ValoreMercatoMin_normale"] + df["OV_ValoreMercatoMax_normale"]) / 2
    df["prezzo_stimato_immobile"] = df["prezzo_m2"] * df["AI_Superficie"]

    prezzi = df.groupby("A_Id").agg(
        prezzo_stimato_totale=('prezzo_stimato_immobile', 'sum'),
        A_Prezzo=('A_Prezzo', 'first')
    ).reset_index()

    prezzi["coefficiente"] = prezzi["A_Prezzo"] / prezzi["prezzo_stimato_totale"]

    df = df.merge(prezzi[["A_Id", "coefficiente"]], on="A_Id", how="left")
    df["AI_Prezzo_Ridistribuito"] = df["prezzo_stimato_immobile"] * df["coefficiente"]

    df.drop(columns=["prezzo_m2", "prezzo_stimato_immobile", "coefficiente"], inplace=True)
    return df

def process_floor_features(df):
    """
    Elabora le features del piano e le aggiunge al DataFrame.
    """
    if 'AI_Piano' not in df.columns:
        print("Attenzione: colonna AI_Piano non trovata nel DataFrame")
        return df
    
    print("Elaborazione features del piano...")
    
    # Applica la funzione extract_floor_features
    floor_features = df['AI_Piano'].apply(extract_floor_features)
    
    # Converti in DataFrame
    floor_df = pd.DataFrame(floor_features.tolist())
    
    # Aggiungi prefisso per evitare conflitti
    floor_df.columns = ['floor_' + col for col in floor_df.columns]
    
    # Concatena con il DataFrame originale
    df_with_floors = pd.concat([df, floor_df], axis=1)
    
    # Statistiche di debug
    print(f"Features del piano aggiunte: {list(floor_df.columns)}")
    print(f"Valori nulli in floor_numeric_weighted: {df_with_floors['floor_floor_numeric_weighted'].isna().sum()}")
    print(f"Range floor_numeric_weighted: {df_with_floors['floor_floor_numeric_weighted'].min():.2f} - {df_with_floors['floor_floor_numeric_weighted'].max():.2f}")
    
    return df_with_floors

def load_data_dual_omi(json_path, selected_aliases=["A", "AI", "PC", "ISC", "II", "PCOZ", "OZ", "OV"]):
    with open(json_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    select_clause = build_select_clause_dual_omi(schema, selected_aliases)
    query = generate_query_dual_omi(select_clause)

    engine = get_engine()
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    df = clean_dataframe(df)
    df = drop_duplicate_columns(df)
    df = filter_coherent_acts(df)
    df = estimate_prices_dual(df)
    
    # Elaborazione features del piano
    df = process_floor_features(df)

    # Drop colonne interamente NaN
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        print(f"⚠️ Colonne interamente NaN rimosse: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    return df
