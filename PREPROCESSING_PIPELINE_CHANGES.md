# Ristrutturazione Pipeline di Preprocessing

## Panoramica delle Modifiche

La pipeline di preprocessing è stata completamente ristrutturata per seguire una sequenza logica e configurabile secondo le specifiche richieste.

## Nuova Sequenza di Step

### 1. **Caricamento Dati** ✅
- Carica il dataset dal percorso specificato
- Log delle dimensioni originali

### 2. **Pulizia Dati** ✅ 
- **NUOVO**: Rimozione colonne specifiche configurabile: `['A_Id', 'A_Codice', 'A_Prezzo', 'AI_Id']`
- **NUOVO**: Rimozione colonne costanti (soglia configurabile)
- Rimozione righe con target mancante
- Rimozione colonne completamente vuote
- Rimozione duplicati completi

### 3. **Analisi Cramér's V** ✅ **NUOVO**
- Analizza correlazioni tra variabili categoriche SENZA rimuoverle
- Identifica potenziali problemi prima dello split
- Configurabile on/off

### 4. **Conversione Automatica a Numerico** ✅ **NUOVO**
- Converte colonne object a numerico quando possibile
- Soglia configurabile per tasso di successo
- Configurabile on/off

### 5. **Encoding Categoriche di Base** ✅
- Encoding senza target encoding (evita data leakage)
- Configurabile on/off

### 6. **Imputazione Valori Nulli** ✅
- Gestisce valori mancanti con strategie appropriate

### 7. **Rimozione Feature Correlate** ✅ **MIGLIORATO**
- Rimuove feature numeriche altamente correlate PRIMA dello split
- Configurabile on/off

### 8. **Split Train/Test/Val** ✅
- Split temporale o casuale
- Mantiene proporzioni target originali

### 9. **Target Encoding Post-Split** ✅ **MIGLIORATO**
- Target encoding SOLO dopo lo split per evitare data leakage
- Configurabile on/off

### 10. **Feature Scaling** ✅ **SEPARATO**
- Scaling applicato solo su train, poi trasferito a val/test
- Funzione dedicata per controllo granulare
- Configurabile on/off

### 11. **Trasformazione Log + Outlier Detection** ✅
- Trasformazione logaritmica del target
- Outlier detection con metodi multipli
- Entrambi configurabili separatamente

### 12. **PCA** ✅ **SEPARATO**
- PCA applicata solo su train, poi trasferita
- Funzione dedicata per controllo granulare
- Configurabile on/off

### 13. **Salvataggi** ✅
- Salva tutti i risultati e metadati completi

## Nuove Configurazioni Aggiunte

```yaml
preprocessing:
  # Controllo step
  steps:
    enable_specific_columns_removal: true
    enable_constant_columns_removal: true
    enable_cramers_analysis: true  
    enable_auto_numeric_conversion: true
    enable_correlation_removal: true
    enable_advanced_encoding: true
    enable_feature_scaling: true
    enable_log_transformation: true
    enable_outlier_detection: true
    enable_pca: false
  
  # Configurazioni pulizia
  columns_to_remove: ['A_Id', 'A_Codice', 'A_Prezzo', 'AI_Id']
  constant_column_threshold: 0.95
  auto_numeric_threshold: 0.8
```

## Nuove Funzioni Aggiunte

### `src/preprocessing/cleaning.py`
- `remove_specific_columns()` - Rimuove colonne specifiche
- `remove_constant_columns()` - Rimuove colonne quasi costanti
- `convert_to_numeric()` - Conversione automatica a numerico

### `src/preprocessing/filtering.py`
- `analyze_cramers_correlations()` - Analisi correlazioni pre-split

### `src/preprocessing/transformation.py`
- `apply_feature_scaling()` - Solo feature scaling
- `apply_pca_transformation()` - Solo PCA

## Vantaggi della Nuova Pipeline

### ✅ **Controllo Data Leakage**
- Target encoding SOLO post-split
- Feature scaling SOLO post-split
- PCA SOLO post-split

### ✅ **Configurabilità Completa**
- Ogni step può essere abilitato/disabilitato
- Tutte le soglie sono configurabili
- Controllo granulare su ogni operazione

### ✅ **Modularità**
- Funzioni separate per ogni operazione
- Riutilizzabilità dei singoli step
- Easier testing e debugging

### ✅ **Tracciabilità**
- Log dettagliati per ogni step
- Informazioni complete su tutte le trasformazioni
- Summary finale degli step eseguiti

### ✅ **Sequenza Logica**
- Pulizia completa prima dello split
- Operazioni che potrebbero causare leakage dopo lo split
- Ordine ottimizzato per prestazioni

## Breaking Changes

⚠️ **ATTENZIONE**: Questa è una ristrutturazione completa che cambia:

1. **Ordine degli step**: Nuova sequenza logica
2. **Configurazione**: Nuove opzioni nel config.yaml
3. **Funzioni**: Nuove funzioni e interfacce
4. **Output**: Struttura diversa delle informazioni salvate

## Test della Pipeline

Per testare la nuova pipeline:

```python
from src.preprocessing.pipeline import run_preprocessing_pipeline

# La pipeline userà automaticamente le nuove configurazioni
# e la nuova sequenza di step
```

## Compatibilità

- ✅ Mantiene compatibilità con il resto del sistema
- ✅ Stesso formato di output per i file
- ✅ Stesse interfacce principali
- ⚠️ Nuovo formato per preprocessing_info.json (più dettagliato)