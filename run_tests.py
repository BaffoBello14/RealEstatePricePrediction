#!/usr/bin/env python3
"""
Script per eseguire i test della ML Pipeline.
Fornisce un'interfaccia semplice per eseguire diversi tipi di test.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, verbose=True):
    """Esegue un comando shell e gestisce l'output."""
    if verbose:
        print(f"🔄 Eseguendo: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        if verbose:
            print("✅ Comando completato con successo")
        return result
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"❌ Comando fallito con exit code {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Esegui test per la ML Pipeline")
    
    # Tipi di test
    parser.add_argument('--all', action='store_true', help='Esegui tutti i test')
    parser.add_argument('--unit', action='store_true', help='Esegui solo unit test')
    parser.add_argument('--integration', action='store_true', help='Esegui solo integration test')
    parser.add_argument('--fast', action='store_true', help='Esegui solo test veloci')
    parser.add_argument('--slow', action='store_true', help='Esegui solo test lenti')
    
    # Test per moduli specifici
    parser.add_argument('--utils', action='store_true', help='Test moduli utilities')
    parser.add_argument('--db', action='store_true', help='Test moduli database')
    parser.add_argument('--dataset', action='store_true', help='Test moduli dataset')
    parser.add_argument('--preprocessing', action='store_true', help='Test moduli preprocessing')
    parser.add_argument('--training', action='store_true', help='Test moduli training')
    parser.add_argument('--evaluation', action='store_true', help='Test moduli evaluation')
    parser.add_argument('--e2e', action='store_true', help='Test end-to-end')
    
    # Opzioni
    parser.add_argument('--coverage', action='store_true', help='Genera report di coverage')
    parser.add_argument('--parallel', action='store_true', help='Esegui test in parallelo')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output verboso')
    parser.add_argument('--quiet', '-q', action='store_true', help='Output minimo')
    parser.add_argument('--fail-fast', action='store_true', help='Ferma al primo fallimento')
    parser.add_argument('--markers', type=str, help='Filtri per markers pytest (es: "not slow")')
    
    # Test specifici
    parser.add_argument('--file', type=str, help='Esegui test da file specifico')
    parser.add_argument('--function', type=str, help='Esegui funzione di test specifica')
    
    # Setup
    parser.add_argument('--install-deps', action='store_true', help='Installa dipendenze prima dei test')
    parser.add_argument('--clean', action='store_true', help='Pulisci file temporanei prima dei test')
    
    args = parser.parse_args()
    
    # Se nessuna opzione è specificata, esegui tutti i test
    if not any([args.all, args.unit, args.integration, args.fast, args.slow,
                args.utils, args.db, args.dataset, args.preprocessing, 
                args.training, args.evaluation, args.e2e, args.file]):
        args.all = True
    
    # Setup environment
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # Clean se richiesto
    if args.clean:
        print("🧹 Pulizia file temporanei...")
        run_command(['make', 'clean'], args.verbose)
    
    # Install dependencies se richiesto
    if args.install_deps:
        print("📦 Installazione dipendenze...")
        run_command(['pip', 'install', '-r', 'requirements.txt'], args.verbose)
    
    # Costruisci comando pytest
    cmd = ['pytest']
    
    # Aggiungi opzioni verbosità
    if args.verbose:
        cmd.append('-v')
    elif args.quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')  # Default verboso
    
    # Aggiungi fail-fast
    if args.fail_fast:
        cmd.append('-x')
    
    # Aggiungi parallel
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # Aggiungi coverage
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term-missing'])
    
    # Aggiungi markers
    if args.markers:
        cmd.extend(['-m', args.markers])
    
    # Specifica test da eseguire
    test_files = []
    
    if args.file:
        if args.function:
            test_files.append(f"{args.file}::{args.function}")
        else:
            test_files.append(args.file)
    else:
        # Aggiungi file per tipo di test
        if args.all:
            test_files.append('tests/')
        else:
            if args.unit:
                cmd.extend(['-m', 'unit'])
                test_files.append('tests/')
            if args.integration:
                cmd.extend(['-m', 'integration'])
                test_files.append('tests/')
            if args.fast:
                cmd.extend(['-m', 'not slow'])
                test_files.append('tests/')
            if args.slow:
                cmd.extend(['-m', 'slow'])
                test_files.append('tests/')
            
            # Test per moduli specifici
            if args.utils:
                test_files.append('tests/test_utils.py')
            if args.db:
                test_files.append('tests/test_database.py')
            if args.dataset:
                test_files.append('tests/test_dataset.py')
            if args.preprocessing:
                test_files.append('tests/test_preprocessing.py')
            if args.training:
                test_files.append('tests/test_training.py')
            if args.evaluation:
                test_files.append('tests/test_evaluation.py')
            if args.e2e:
                test_files.append('tests/test_integration.py')
    
    # Aggiungi file di test al comando
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append('tests/')
    
    # Esegui test
    print("🚀 Avvio test...")
    print(f"📝 Comando: {' '.join(cmd)}")
    print("=" * 60)
    
    result = run_command(cmd, verbose=True)
    
    if result:
        print("=" * 60)
        print("✅ Test completati con successo!")
        
        if args.coverage:
            print("📊 Report di coverage generato in htmlcov/index.html")
    else:
        print("=" * 60)
        print("❌ Alcuni test sono falliti!")
        sys.exit(1)

def show_examples():
    """Mostra esempi di utilizzo."""
    examples = [
        "# Esegui tutti i test",
        "python run_tests.py --all",
        "",
        "# Esegui solo test veloci con coverage",
        "python run_tests.py --fast --coverage",
        "",
        "# Esegui test di un modulo specifico",
        "python run_tests.py --preprocessing",
        "",
        "# Esegui test specifico",
        "python run_tests.py --file tests/test_utils.py --function test_load_config",
        "",
        "# Esegui test in parallelo escludendo quelli lenti",
        "python run_tests.py --parallel --markers 'not slow'",
        "",
        "# Setup completo e test",
        "python run_tests.py --install-deps --clean --all --coverage"
    ]
    
    print("Esempi di utilizzo:")
    print("=" * 50)
    for example in examples:
        print(example)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--examples':
        show_examples()
    else:
        main()