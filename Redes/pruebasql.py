import sqlite3
import pandas as pd

# Conectarse
conn = sqlite3.connect("transacciones.db")

# Ver tablas
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("Tablas:", tables)

# Cargar primeras 10 filas en un DataFrame
df = pd.read_sql_query("SELECT * FROM transacciones LIMIT 10;", conn)
print(df)

conn.close()