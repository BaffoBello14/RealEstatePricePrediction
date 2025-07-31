import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any
from .connect import get_engine
from ..utils.logger import get_logger
from ..utils.io import save_dataframe, load_json

logger = get_logger(__name__)

def build_select_clause_dual_omi(schema: Dict[str, Any], selected_aliases: Optional[List[str]] = None) -> str:
    """
    Costruisce la SELECT SQL includendo i valori OMI 'Normale', 'Ottimo' e 'Scadente'.
    
    Args:
        schema: Schema del database
        selected_aliases: Lista degli alias da includere
        
    Returns:
        Clausola SELECT SQL
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
                # Gestione speciale per OmiValori con stati diversi
                for stato_alias, stato_suffix in [("OVN", "normale"), ("OVO", "ottimo"), ("OVS", "scadente")]:
                    selects.append(f"{stato_alias}.{col_name} AS {alias}_{col_name}_{stato_suffix}")
            else:
                if col_type in ["geometry", "geography"]:
                    selects.append(f"{alias}.{col_name}.STAsText() AS {alias}_{col_name}")
                else:
                    selects.append(f"{alias}.{col_name} AS {alias}_{col_name}")

    return ",\n ".join(selects)

def generate_query_dual_omi(select_clause: str) -> str:
    """
    Genera la query SQL completa per il recupero dati con OMI multi-stato.
    
    Args:
        select_clause: Clausola SELECT
        
    Returns:
        Query SQL completa
    """
    return f"""
    SELECT
        {select_clause}
    FROM
        Atti A
        INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
        INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
        INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
        INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
        INNER JOIN ParticelleCatastali_OmiZone PC_OZ ON PC_OZ.IdParticella = PC.Id
        INNER JOIN OmiZone OZ ON PC_OZ.IdZona = OZ.Id
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

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce il DataFrame sostituendo stringhe vuote con NaN.
    
    Args:
        df: DataFrame da pulire
        
    Returns:
        DataFrame pulito
    """
    logger.info("Pulizia DataFrame: sostituzione stringhe vuote con NaN")
    df.replace('', np.nan, inplace=True)
    return df

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove colonne duplicate dal DataFrame.
    
    Args:
        df: DataFrame da cui rimuovere duplicati
        
    Returns:
        DataFrame senza colonne duplicate
    """
    logger.info("Rimozione colonne duplicate...")
    
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
        logger.info(f"Colonne duplicate rimosse: {list(to_drop)}")
    else:
        logger.info("Nessuna colonna duplicata trovata")

    return df

def retrieve_data(schema_path: str, selected_aliases: List[str], output_path: str) -> pd.DataFrame:
    """
    Recupera i dati dal database e li salva su disco.
    
    Args:
        schema_path: Path al file schema JSON
        selected_aliases: Lista degli alias da includere
        output_path: Path dove salvare i dati
        
    Returns:
        DataFrame con i dati recuperati
    """
    logger.info(f"Avvio recupero dati dal database...")
    logger.info(f"Schema: {schema_path}")
    logger.info(f"Alias selezionati: {selected_aliases}")
    
    try:
        # Carica schema
        schema = load_json(schema_path)
        logger.info(f"Schema caricato con {len(schema)} tabelle")
        
        # Costruisce query
        select_clause = build_select_clause_dual_omi(schema, selected_aliases)
        query = generate_query_dual_omi(select_clause)
        
        logger.info("Query SQL generata")
        
        # Esegue query
        engine = get_engine()
        with engine.connect() as connection:
            logger.info("Esecuzione query in corso...")
            df = pd.read_sql(query, connection)
            logger.info(f"Query completata: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Pulizia dati
        df = clean_dataframe(df)
        df = drop_duplicate_columns(df)
        
        # Salva risultati
        save_dataframe(df, output_path, format='parquet')
        logger.info(f"Dati salvati in: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Errore nel recupero dati: {e}")
        raise