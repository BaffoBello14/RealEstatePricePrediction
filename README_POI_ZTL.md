# Funzionalità POI e ZTL - Guida

## Panoramica

Il sistema di recupero dati è stato esteso per includere:

1. **Conteggi Punti di Interesse (POI)**: Numero di POI per tipologia nell'area di isodistanza di ogni particella
2. **Zone a Traffico Limitato (ZTL)**: Verifica se una particella si trova in una ZTL

## Nuove Funzionalità

### 1. Conteggi POI per Tipologia

Il sistema conta automaticamente i punti di interesse per ogni tipologia (ristoranti, farmacie, scuole, ecc.) che si trovano nell'area di **isodistanza a piedi** (500m) di ogni particella catastale.

**Caratteristiche:**
- Usa il campo `Isodistanza` delle particelle catastali
- Conta POI per ogni tipologia definita in `PuntiDiInteresseTipologie`
- Crea colonne dinamiche: `POI_{tipologia}_count`
- Gestisce automaticamente nomi di tipologie con caratteri speciali

### 2. Verifica ZTL

Per ogni particella, verifica se il **centroide** è contenuto in una Zona a Traffico Limitato.

**Campi aggiunti:**
- `InZTL`: Flag binario (1 = in ZTL, 0 = non in ZTL)

## Utilizzo del Codice

### Funzione Base con POI e ZTL

```python
from src.db.retrieve import retrieve_data

# Recupero completo con tutte le categorie POI
df = retrieve_data(
    schema_path="data/db_schema.json",
    selected_aliases=["A", "AI", "PC", "OV"],
    output_path="dati_completi.parquet",
    include_poi=True,      # Include conteggi POI
    include_ztl=True       # Include informazioni ZTL
)
```

### Recupero con Categorie POI Specifiche

```python
# Solo alcune categorie POI
specific_poi = ["restaurant", "pharmacy", "bank", "school"]

df = retrieve_data(
    schema_path="data/db_schema.json",
    selected_aliases=["A", "AI", "PC"],
    output_path="dati_poi_specifici.parquet",
    include_poi=True,
    include_ztl=True,
    poi_categories=specific_poi  # Lista specifica
)
```

### Funzioni di Utilità

```python
from src.db.retrieve import get_poi_categories_info, test_poi_and_ztl_sample

# Visualizza tutte le categorie POI disponibili
poi_info = get_poi_categories_info()
print(poi_info)

# Test su campione limitato
sample_data, poi_info = test_poi_and_ztl_sample(
    schema_path="data/db_schema.json",
    selected_aliases=["A", "AI", "PC"],
    limit=10
)
```

## Struttura SQL Generata

### Subquery POI

```sql
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
)
```

### Subquery ZTL

```sql
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
)
```

## Colonne nel Dataset Finale

### Colonne POI
- `POI_restaurant_count`: Numero di ristoranti
- `POI_pharmacy_count`: Numero di farmacie  
- `POI_bank_count`: Numero di banche
- `POI_school_count`: Numero di scuole
- ... (una per ogni categoria POI)

### Colonne ZTL
- `InZTL`: Flag binario (0/1)

## Test delle Funzionalità

Esegui il file di test per verificare il funzionamento:

```bash
python test_poi_ztl.py
```

Il test:
1. Mostra tutte le categorie POI disponibili
2. Testa su un campione di 5 righe
3. Esegue un recupero completo con categorie specifiche
4. Analizza i risultati POI e ZTL

## Prestazioni e Considerazioni

### Ottimizzazioni
- Le subquery usano `CROSS JOIN` per garantire tutte le combinazioni
- I `LEFT JOIN` gestiscono categorie POI senza occorrenze
- L'uso di `STContains` è ottimizzato per indici spaziali

### Scalabilità
- Il numero di colonne POI dipende dalle categorie disponibili
- Ogni categoria POI aggiunge una colonna al risultato
- L'elaborazione ZTL è efficiente grazie all'uso di `EXISTS`

### Personalizzazione
- Usa `poi_categories` per limitare le categorie
- Imposta `include_poi=False` o `include_ztl=False` per disabilitare
- Le funzioni mantengono retrocompatibilità con il codice esistente

## Esempio di Risultato

```
A_Id | PC_Superficie | POI_restaurant_count | POI_pharmacy_count | InZTL
-----|---------------|---------------------|-------------------|-------
001  | 120.5         | 3                   | 1                 | 1     
002  | 85.2          | 1                   | 0                 | 0     
003  | 200.0         | 5                   | 2                 | 1     
```

## Backwards Compatibility

Il codice esistente continua a funzionare:

```python
# Questo continua a funzionare come prima
df = retrieve_data(schema_path, selected_aliases, output_path)
# Equivale a: include_poi=True, include_ztl=True, poi_categories=None
```