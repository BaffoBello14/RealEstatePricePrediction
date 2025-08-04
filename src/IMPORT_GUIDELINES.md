# Guidelines per l'Organizzazione degli Import

## Principi Generali

### 1. Ordine degli Import
Mantenere sempre questo ordine:
```python
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 3. Local imports (dalle utilità condivise)
from ..utils import get_logger, load_dataframe, save_dataframe

# 4. Local imports (da moduli specifici)
from .data_cleaning_core import convert_to_numeric_unified
```

### 2. Import Preferiti per Utilità

#### Utilità Comuni (Utilizzare sempre import da ..utils)
```python
# ✅ GIUSTO
from ..utils import get_logger, load_dataframe, save_dataframe, PathManager

# ❌ SBAGLIATO 
from ..utils.logger import get_logger
from ..utils.io import load_dataframe, save_dataframe
from ..utils.path_manager import PathManager
```

#### Import da src (per main.py e test)
```python
# ✅ GIUSTO (da main.py)
from src import (
    setup_logger, get_logger,
    load_config, check_file_exists,
    PipelineOrchestrator
)

# ❌ SBAGLIATO
from src.utils.logger import setup_logger, get_logger
from src.utils.io import load_config, check_file_exists
```

### 3. Import Relativi vs Assoluti

#### All'interno di Package
```python
# ✅ GIUSTO (import relativi)
from .data_cleaning_core import convert_to_numeric_unified
from ..utils import get_logger

# ❌ SBAGLIATO (import assoluti quando non necessari)
from src.preprocessing.data_cleaning_core import convert_to_numeric_unified
```

#### Dal Root del Progetto
```python
# ✅ GIUSTO (import assoluti da root)
from src import PipelineOrchestrator
from src.preprocessing import run_modular_preprocessing_pipeline
```

### 4. Organizzazione nei Package

#### File __init__.py
Ogni package deve avere un `__init__.py` che esporta le funzionalità principali:

```python
# src/preprocessing/__init__.py
from .pipeline_modular import run_modular_preprocessing_pipeline
from .data_cleaning_core import convert_to_numeric_unified
# ... altre esportazioni

__all__ = [
    'run_modular_preprocessing_pipeline',
    'convert_to_numeric_unified',
    # ... lista completa
]
```

### 5. Best Practices per Import di Gruppo

#### Raggruppare Import Simili
```python
# ✅ GIUSTO
from ..utils import (
    get_logger, 
    load_dataframe, save_dataframe,
    validate_dataframe, validate_target_column
)

# ❌ SBAGLIATO (linee multiple separate)
from ..utils import get_logger
from ..utils import load_dataframe
from ..utils import save_dataframe
from ..utils import validate_dataframe
```

### 6. Import Specifici per Moduli

#### Preprocessing Core Modules
```python
# Per data_cleaning_core.py
from ..utils import get_logger  # Sempre così

# Per steps/
from ...utils import get_logger  # Tre livelli su per steps
```

#### Training Modules
```python
# Per train.py
from ..utils import get_logger, load_dataframe, save_model

# Non importare direttamente da preprocessing
from ..preprocessing import run_preprocessing_pipeline  # Usa l'API pubblica
```

### 7. Import di Compatibilità

#### Mantenere Compatibilità con Versioni Legacy
```python
# Nel __init__.py del preprocessing
# Import delle funzioni legacy per retrocompatibilità
from .cleaning import convert_to_numeric, clean_data  # DEPRECATED ma mantenute
from .encoding import auto_convert_to_numeric  # DEPRECATED ma mantenute
```

### 8. Anti-Pattern da Evitare

#### Import Circolari
```python
# ❌ MAI FARE
# In module_a.py
from .module_b import function_b

# In module_b.py  
from .module_a import function_a  # CIRCULAR IMPORT!
```

#### Import di Tutto
```python
# ❌ EVITARE
from .some_module import *

# ✅ MEGLIO
from .some_module import specific_function, specific_class
```

#### Import Troppo Specifici
```python
# ❌ EVITARE (troppo specifico)
from ..utils.error_handling import PreprocessingError

# ✅ MEGLIO (usa il package utils)
from ..utils import PreprocessingError
```

### 9. Convenzioni per Testing

#### Import nei Test
```python
# In tests/
import pytest
from src import PipelineOrchestrator
from src.preprocessing import convert_to_numeric_unified
```

### 10. Checklist per Revisione

Quando si revisionano import:
- [ ] Ordine corretto (stdlib, third-party, local)
- [ ] Utilizzo di import relativi appropriati
- [ ] Raggruppamento logico degli import
- [ ] Nessun import circolare
- [ ] Utilizzo dei package __init__.py per import pubblici
- [ ] Import specifici, non wildcard (*)
- [ ] Coerenza con le convenzioni del progetto

### 11. Strumenti di Verifica

#### Verifica Automatica
```bash
# Usando isort per ordinare gli import
isort src/

# Usando flake8 per verificare stile
flake8 src/ --select=E401,E402

# Usando pylint per import analysis
pylint src/ --disable=all --enable=import-error,cyclic-import
```

## Esempi di Migrazione

### Prima (Disorganizzato)
```python
from sklearn.preprocessing import StandardScaler
from ..utils.logger import get_logger
import pandas as pd
from .data_cleaning_core import convert_to_numeric_unified
from ..utils.io import load_dataframe
from typing import Dict, Any
```

### Dopo (Organizzato)
```python
import pandas as pd
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler

from ..utils import get_logger, load_dataframe
from .data_cleaning_core import convert_to_numeric_unified
```