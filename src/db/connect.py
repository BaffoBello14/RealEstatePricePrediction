from sqlalchemy import create_engine, Engine
import urllib.parse
import os
from dotenv import load_dotenv
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Carica variabili d'ambiente
load_dotenv()

def get_engine() -> Engine:
    """
    Crea e ritorna un engine SQLAlchemy per la connessione al database.
    
    Returns:
        Engine SQLAlchemy configurato
        
    Raises:
        ValueError: Se le variabili di ambiente non sono configurate
        Exception: Se la connessione non può essere stabilita
    """
    try:
        # Recupera variabili d'ambiente
        server = os.getenv('SERVER')
        database = os.getenv('DATABASE')
        user = os.getenv('USER')
        password = os.getenv('PASSWORD')
        
        # Verifica che tutte le variabili siano presenti
        if not all([server, database, user, password]):
            missing = [var for var, val in [
                ('SERVER', server), ('DATABASE', database), 
                ('USER', user), ('PASSWORD', password)
            ] if not val]
            raise ValueError(f"Variabili d'ambiente mancanti: {missing}")
        
        # Codifica credenziali per URL
        user_encoded = urllib.parse.quote_plus(user)
        password_encoded = urllib.parse.quote_plus(password)
        
        # Costruisce connection string
        connection_string = (
            f"mssql+pyodbc://{user_encoded}:{password_encoded}@{server}/"
            f"{database}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
            f"TrustServerCertificate=no&Connection+Timeout=30"
        )
        
        # Crea engine
        engine = create_engine(connection_string)
        
        # Test connessione
        with engine.connect() as conn:
            logger.info(f"Connessione al database {database} su {server} stabilita con successo")
        
        return engine
        
    except Exception as e:
        logger.error(f"Errore nella connessione al database: {e}")
        raise

def test_connection() -> bool:
    """
    Testa la connessione al database.
    
    Returns:
        True se la connessione è riuscita, False altrimenti
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            test_value = result.fetchone()[0]
            if test_value == 1:
                logger.info("Test connessione database: SUCCESSO")
                return True
            else:
                logger.error("Test connessione database: FALLITO - Valore inaspettato")
                return False
    except Exception as e:
        logger.error(f"Test connessione database: FALLITO - {e}")
        return False