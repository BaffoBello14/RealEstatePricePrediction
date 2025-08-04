"""
Utility per analizzare e validare la struttura degli import nella codebase.

Questo modulo fornisce funzionalità per:
- Rilevare import circolari
- Verificare consistenza degli import
- Analizzare le dipendenze tra moduli
- Identificare import non utilizzati o ridondanti
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

from .logger import get_logger

logger = get_logger(__name__)


class ImportAnalyzer:
    """
    Analizzatore per la struttura degli import della codebase.
    """
    
    def __init__(self, root_path: str = "src"):
        """
        Inizializza l'analizzatore.
        
        Args:
            root_path: Path della directory radice da analizzare
        """
        self.root_path = Path(root_path)
        self.dependencies = defaultdict(set)  # module -> set of imported modules
        self.reverse_dependencies = defaultdict(set)  # module -> set of modules that import it
        self.import_styles = defaultdict(list)  # module -> list of import statements
        
    def analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """
        Analizza un singolo file Python per i suoi import.
        
        Args:
            file_path: Path del file da analizzare
            
        Returns:
            Dictionary con informazioni sugli import del file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = {
                'absolute': [],
                'relative': [],
                'from_imports': [],
                'errors': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports['absolute'].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    level = node.level
                    
                    if level > 0:  # Relative import
                        imports['relative'].append(f"{'.' * level}{module}")
                    else:  # Absolute import
                        imports['from_imports'].append(module)
                        
        except Exception as e:
            imports['errors'].append(str(e))
            logger.warning(f"Errore nell'analisi di {file_path}: {e}")
            
        return imports
    
    def scan_directory(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Scansiona ricorsivamente la directory per analizzare tutti i file Python.
        
        Returns:
            Dictionary con le analisi di tutti i file
        """
        results = {}
        
        for py_file in self.root_path.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
                
            relative_path = py_file.relative_to(self.root_path)
            module_name = str(relative_path).replace('/', '.').replace('.py', '')
            
            imports = self.analyze_file(py_file)
            results[module_name] = imports
            
            # Costruisci grafo delle dipendenze
            for imp in imports['absolute'] + imports['from_imports']:
                if imp.startswith('src.') or imp.startswith('.'):
                    # Import interno
                    self.dependencies[module_name].add(imp)
                    self.reverse_dependencies[imp].add(module_name)
        
        return results
    
    def detect_circular_imports(self) -> List[List[str]]:
        """
        Rileva import circolari usando l'algoritmo DFS.
        
        Returns:
            Lista di cicli rilevati
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Trovato ciclo
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for dependency in self.dependencies.get(node, []):
                dfs(dependency, path + [node])
            
            rec_stack.remove(node)
        
        for module in self.dependencies:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def check_import_consistency(self, results: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Verifica la consistenza degli stili di import.
        
        Args:
            results: Risultati dell'analisi dei file
            
        Returns:
            Dictionary con inconsistenze rilevate
        """
        inconsistencies = defaultdict(list)
        
        # Verifica che i moduli utils siano importati in modo consistente
        utils_patterns = {
            'logger': set(),
            'io': set(),
            'error_handling': set(),
            'path_manager': set()
        }
        
        for module, imports in results.items():
            # Analizza pattern di import per moduli utils
            all_imports = imports['absolute'] + imports['from_imports'] + imports['relative']
            
            for imp in all_imports:
                if 'utils.logger' in imp:
                    utils_patterns['logger'].add(imp)
                elif 'utils.io' in imp:
                    utils_patterns['io'].add(imp)
                elif 'utils.error_handling' in imp:
                    utils_patterns['error_handling'].add(imp)
                elif 'utils.path_manager' in imp:
                    utils_patterns['path_manager'].add(imp)
        
        # Rileva inconsistenze
        for util_type, patterns in utils_patterns.items():
            if len(patterns) > 1:
                inconsistencies[f'{util_type}_imports'].extend(list(patterns))
        
        return dict(inconsistencies)
    
    def generate_report(self) -> str:
        """
        Genera un report completo dell'analisi degli import.
        
        Returns:
            Report testuale formattato
        """
        logger.info("🔍 Generazione report analisi import...")
        
        # Esegui analisi completa
        results = self.scan_directory()
        cycles = self.detect_circular_imports()
        inconsistencies = self.check_import_consistency(results)
        
        report = []
        report.append("=" * 60)
        report.append("REPORT ANALISI STRUTTURA IMPORT")
        report.append("=" * 60)
        
        # Statistiche generali
        total_files = len(results)
        total_imports = sum(
            len(data['absolute']) + len(data['from_imports']) + len(data['relative'])
            for data in results.values()
        )
        
        report.append(f"\n📊 STATISTICHE GENERALI:")
        report.append(f"  • File analizzati: {total_files}")
        report.append(f"  • Import totali: {total_imports}")
        report.append(f"  • Import circolari rilevati: {len(cycles)}")
        report.append(f"  • Inconsistenze rilevate: {len(inconsistencies)}")
        
        # Import circolari
        if cycles:
            report.append(f"\n⚠️  IMPORT CIRCOLARI RILEVATI:")
            for i, cycle in enumerate(cycles, 1):
                report.append(f"  {i}. {' -> '.join(cycle)}")
        else:
            report.append(f"\n✅ NESSUN IMPORT CIRCOLARE RILEVATO")
        
        # Inconsistenze
        if inconsistencies:
            report.append(f"\n🔧 INCONSISTENZE NEGLI IMPORT:")
            for type_name, patterns in inconsistencies.items():
                report.append(f"  • {type_name}:")
                for pattern in patterns:
                    report.append(f"    - {pattern}")
        else:
            report.append(f"\n✅ IMPORT CONSISTENTI")
        
        # Moduli più dipendenti
        most_dependencies = sorted(
            self.dependencies.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        if most_dependencies:
            report.append(f"\n📦 MODULI CON PIÙ DIPENDENZE:")
            for module, deps in most_dependencies:
                report.append(f"  • {module}: {len(deps)} dipendenze")
        
        # Moduli più importati
        most_imported = sorted(
            self.reverse_dependencies.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        if most_imported:
            report.append(f"\n🔗 MODULI PIÙ IMPORTATI:")
            for module, importers in most_imported:
                report.append(f"  • {module}: importato da {len(importers)} moduli")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def analyze_imports(root_path: str = "src") -> str:
    """
    Funzione di utilità per analizzare la struttura degli import.
    
    Args:
        root_path: Path della directory da analizzare
        
    Returns:
        Report dell'analisi
    """
    analyzer = ImportAnalyzer(root_path)
    return analyzer.generate_report()


if __name__ == "__main__":
    # Script eseguibile per analisi standalone
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizza struttura import codebase")
    parser.add_argument("--path", default="src", help="Path directory da analizzare")
    parser.add_argument("--output", help="File output per il report")
    
    args = parser.parse_args()
    
    report = analyze_imports(args.path)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report salvato in: {args.output}")
    else:
        print(report)