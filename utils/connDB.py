# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import psycopg
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DB_HOST", "localhost")
port = int(os.getenv("DB_PORT", 5433))
dbname = os.getenv("POSTGRES_DB", "my_database")
user = os.getenv("POSTGRES_USER", "admin")
password = os.getenv("POSTGRES_PASSWORD", "securepasswordhere")

conn = psycopg.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password
)

print("Connexion réussie")
