import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
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

def get_poi_categories_query() -> str:
    """
    Ottiene tutte le categorie di punti di interesse disponibili.
    
    Returns:
        Query per ottenere le categorie POI
    """
    return """
    SELECT DISTINCT Id, Denominazione
    FROM PuntiDiInteresseTipologie
    ORDER BY Denominazione
    """

def generate_poi_counts_subquery() -> str:
    """
    Genera una subquery che calcola il conteggio dei POI per tipologia
    per ogni particella usando l'isodistanza.
    
    Returns:
        Subquery SQL per conteggi POI
    """
    return """
    -- Subquery per conteggi POI per tipologia
    POI_COUNTS AS (
        SELECT 
            PC_MAIN.Id as IdParticella,
            PDIT.Id as TipologiaPOI,
            PDIT.Denominazione as DenominazionePOI,
            COUNT(PDI.Id) as ConteggioPOI
        FROM 
            ParticelleCatastali PC_MAIN
            CROSS JOIN PuntiDiInteresseTipologie PDIT
            LEFT JOIN (
                PuntiDiInteresse PDI 
                INNER JOIN PuntiDiInteresse_Tipologie PDI_T ON PDI.Id = PDI_T.IdPuntoDiInteresse
            ) ON PDI_T.IdTipologia = PDIT.Id 
                AND PC_MAIN.Isodistanza.STContains(PDI.Posizione) = 1
        GROUP BY PC_MAIN.Id, PDIT.Id, PDIT.Denominazione
    )"""

def generate_ztl_subquery() -> str:
    """
    Genera una subquery che verifica se le particelle sono in ZTL.
    
    Returns:
        Subquery SQL per verifica ZTL
    """
    return """
    -- Subquery per verifica ZTL
    ZTL_CHECK AS (
        SELECT 
            PC_MAIN.Id as IdParticella,
            CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM ZoneTrafficoLimitato ZTL 
                    WHERE ZTL.Poligono.STContains(PC_MAIN.Centroide) = 1
                ) THEN 1 
                ELSE 0 
            END as InZTL
        FROM ParticelleCatastali PC_MAIN
    )"""

def generate_query_with_poi_and_ztl(select_clause: str, poi_categories: List[str]) -> str:
    """
    Genera la query SQL completa per il recupero dati con OMI multi-stato,
    conteggi POI per tipologia e verifica ZTL.
    
    Args:
        select_clause: Clausola SELECT
        poi_categories: Lista delle categorie POI da includere
        
    Returns:
        Query SQL completa
    """
    poi_subquery = generate_poi_counts_subquery()
    ztl_subquery = generate_ztl_subquery()
    
    # Genera i LEFT JOIN per ogni categoria POI
    poi_joins = []
    poi_selects = []
    
    for category in poi_categories:
        safe_category = category.replace('-', '_').replace(' ', '_').replace('.', '_')
        alias = f"POI_{safe_category}"
        
        poi_joins.append(f"""
        LEFT JOIN POI_COUNTS {alias} ON PC.Id = {alias}.IdParticella 
            AND {alias}.TipologiaPOI = '{category}'""")
        
        poi_selects.append(f"COALESCE({alias}.ConteggioPOI, 0) AS POI_{safe_category}_count")
    
    poi_joins_str = "".join(poi_joins)
    poi_selects_str = ",\n        ".join(poi_selects)
    
    return f"""
    WITH 
    {poi_subquery},
    {ztl_subquery}
    
    SELECT
        {select_clause},
        -- Conteggi POI per tipologia
        {poi_selects_str},
        -- Informazioni ZTL
        ZTL_INFO.InZTL as InZTL
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
        -- Join per informazioni ZTL
        LEFT JOIN ZTL_CHECK ZTL_INFO ON PC.Id = ZTL_INFO.IdParticella
        -- Join per conteggi POI{poi_joins_str}
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

def generate_query_dual_omi(select_clause: str) -> str:
    """
    Genera la query SQL completa per il recupero dati con OMI multi-stato.
    DEPRECATO: Usare generate_query_with_poi_and_ztl per funzionalità complete.
    
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

def get_poi_categories(engine) -> List[str]:
    """
    Recupera tutte le categorie di punti di interesse disponibili.
    
    Args:
        engine: Engine di connessione al database
        
    Returns:
        Lista degli ID delle categorie POI
    """
    try:
        query = get_poi_categories_query()
        with engine.connect() as connection:
            result = pd.read_sql(query, connection)
            categories = result['Id'].tolist()
            logger.info(f"Trovate {len(categories)} categorie POI: {categories}")
            return categories
    except Exception as e:
        logger.warning(f"Errore nel recupero categorie POI: {e}")
        return []

def retrieve_data(schema_path: str, selected_aliases: List[str], output_path: str, 
                 include_poi: bool = True, include_ztl: bool = True, 
                 poi_categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Recupera i dati dal database e li salva su disco.
    
    Args:
        schema_path: Path al file schema JSON
        selected_aliases: Lista degli alias da includere
        output_path: Path dove salvare i dati
        include_poi: Se includere i conteggi dei punti di interesse
        include_ztl: Se includere le informazioni ZTL
        poi_categories: Lista specifica di categorie POI (se None, usa tutte)
        
    Returns:
        DataFrame con i dati recuperati
    """
    logger.info(f"Avvio recupero dati dal database...")
    logger.info(f"Schema: {schema_path}")
    logger.info(f"Alias selezionati: {selected_aliases}")
    logger.info(f"Include POI: {include_poi}, Include ZTL: {include_ztl}")
    
    try:
        # Carica schema
        schema = load_json(schema_path)
        logger.info(f"Schema caricato con {len(schema)} tabelle")
        
        # Costruisce query
        select_clause = build_select_clause_dual_omi(schema, selected_aliases)
        
        # Ottieni engine
        engine = get_engine()
        
        # Determina se usare la query estesa con POI e ZTL
        if include_poi or include_ztl:
            # Ottieni categorie POI se necessarie
            if include_poi:
                if poi_categories is None:
                    poi_categories = get_poi_categories(engine)
                logger.info(f"Usando {len(poi_categories)} categorie POI")
            else:
                poi_categories = []
            
            query = generate_query_with_poi_and_ztl(select_clause, poi_categories)
        else:
            query = generate_query_dual_omi(select_clause)
        
        logger.info("Query SQL generata")
        
        # Esegue query
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

def get_poi_categories_info() -> pd.DataFrame:
    """
    Recupera informazioni dettagliate sulle categorie di punti di interesse.
    
    Returns:
        DataFrame con ID e denominazione delle categorie POI
    """
    logger.info("Recupero informazioni categorie POI...")
    
    try:
        engine = get_engine()
        query = get_poi_categories_query()
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
            logger.info(f"Trovate {len(df)} categorie POI")
            return df
            
    except Exception as e:
        logger.error(f"Errore nel recupero informazioni POI: {e}")
        raise

def test_poi_and_ztl_sample(schema_path: str, selected_aliases: List[str], 
                           limit: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Testa il recupero POI e ZTL su un campione limitato di dati.
    
    Args:
        schema_path: Path al file schema JSON
        selected_aliases: Lista degli alias da includere
        limit: Numero massimo di righe da recuperare
        
    Returns:
        Tuple con (dati_campione, categorie_poi)
    """
    logger.info(f"Test recupero POI e ZTL su campione di {limit} righe...")
    
    try:
        # Carica schema
        schema = load_json(schema_path)
        
        # Costruisce query
        select_clause = build_select_clause_dual_omi(schema, selected_aliases)
        
        # Ottieni engine e categorie POI
        engine = get_engine()
        poi_categories = get_poi_categories(engine)[:5]  # Usa solo le prime 5 per test
        
        # Genera query con LIMIT
        base_query = generate_query_with_poi_and_ztl(select_clause, poi_categories)
        test_query = f"SELECT TOP {limit} * FROM ({base_query}) AS test_data"
        
        # Esegue query
        with engine.connect() as connection:
            logger.info("Esecuzione query di test...")
            df = pd.read_sql(test_query, connection)
            logger.info(f"Test completato: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Ottieni info categorie POI
        poi_info = get_poi_categories_info()
        
        return df, poi_info
        
    except Exception as e:
        logger.error(f"Errore nel test POI e ZTL: {e}")
        raise