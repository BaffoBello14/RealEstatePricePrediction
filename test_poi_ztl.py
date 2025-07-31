#!/usr/bin/env python3
"""
Script di test per le funzionalità POI e ZTL nel sistema di recupero dati.
"""

import sys
sys.path.append('src')

from db.retrieve import (
    get_poi_categories_info, 
    test_poi_and_ztl_sample,
    retrieve_data
)
from utils.logger import get_logger

logger = get_logger(__name__)

def test_poi_categories():
    """Test per visualizzare le categorie POI disponibili."""
    print("\n=== TEST CATEGORIE POI ===")
    try:
        poi_info = get_poi_categories_info()
        print(f"Categorie POI trovate: {len(poi_info)}")
        print(poi_info.head(10))
        return poi_info
    except Exception as e:
        print(f"Errore: {e}")
        return None

def test_sample_data():
    """Test su un campione di dati con POI e ZTL."""
    print("\n=== TEST CAMPIONE DATI CON POI E ZTL ===")
    
    schema_path = "data/db_schema.json"
    selected_aliases = ["A", "AI", "PC", "OV"]  # Alias base per test
    
    try:
        sample_data, poi_info = test_poi_and_ztl_sample(
            schema_path, selected_aliases, limit=5
        )
        
        print(f"Dati campione: {sample_data.shape}")
        print("Colonne disponibili:")
        for col in sorted(sample_data.columns):
            print(f"  - {col}")
        
        # Mostra colonne POI
        poi_columns = [col for col in sample_data.columns if col.startswith('POI_')]
        print(f"\nColonne POI trovate: {len(poi_columns)}")
        for col in poi_columns:
            print(f"  - {col}")
        
        # Mostra colonne ZTL
        ztl_columns = [col for col in sample_data.columns if 'ZTL' in col]
        print(f"\nColonne ZTL trovate: {len(ztl_columns)}")
        for col in ztl_columns:
            print(f"  - {col}")
        
        return sample_data, poi_info
        
    except Exception as e:
        print(f"Errore: {e}")
        return None, None

def test_specific_poi_categories():
    """Test con categorie POI specifiche."""
    print("\n=== TEST CON CATEGORIE POI SPECIFICHE ===")
    
    schema_path = "data/db_schema.json"
    selected_aliases = ["A", "AI", "PC"]
    output_path = "test_poi_ztl_output.parquet"
    
    # Categorie POI specifiche da testare
    specific_poi = ["restaurant", "pharmacy", "bank", "school"]
    
    try:
        df = retrieve_data(
            schema_path=schema_path,
            selected_aliases=selected_aliases,
            output_path=output_path,
            include_poi=True,
            include_ztl=True,
            poi_categories=specific_poi
        )
        
        print(f"Dati recuperati: {df.shape}")
        
        # Analizza risultati POI
        poi_columns = [col for col in df.columns if col.startswith('POI_')]
        print(f"Colonne POI create: {len(poi_columns)}")
        
        for col in poi_columns:
            max_count = df[col].max()
            mean_count = df[col].mean()
            print(f"  {col}: max={max_count}, media={mean_count:.2f}")
        
        # Analizza risultati ZTL
        if 'InZTL' in df.columns:
            ztl_count = df['InZTL'].sum()
            ztl_percentage = (ztl_count / len(df)) * 100
            print(f"\nImmobili in ZTL: {ztl_count}/{len(df)} ({ztl_percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"Errore: {e}")
        return None

if __name__ == "__main__":
    print("Avvio test funzionalità POI e ZTL...")
    
    # Test 1: Categorie POI disponibili
    poi_info = test_poi_categories()
    
    # Test 2: Campione dati
    sample_data, _ = test_sample_data()
    
    # Test 3: Categorie specifiche
    full_data = test_specific_poi_categories()
    
    print("\n=== TEST COMPLETATI ===")
    
    if poi_info is not None:
        print(f"✓ Categorie POI: {len(poi_info)} trovate")
    else:
        print("✗ Errore nel recupero categorie POI")
    
    if sample_data is not None:
        print(f"✓ Test campione: {sample_data.shape[0]} righe recuperate")
    else:
        print("✗ Errore nel test campione")
    
    if full_data is not None:
        print(f"✓ Test completo: {full_data.shape[0]} righe recuperate")
    else:
        print("✗ Errore nel test completo")