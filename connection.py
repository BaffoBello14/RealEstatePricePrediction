from sqlalchemy import create_engine, text
import urllib
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    SERVER = os.getenv('SERVER')
    DATABASE = os.getenv('DATABASE')
    USER = urllib.parse.quote_plus(os.getenv('USER'))
    PASSWORD = urllib.parse.quote_plus(os.getenv('PASSWORD'))
    
    connection_string = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
    
    engine = create_engine(connection_string)
    return engine


if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        for row in result:
            print(row)