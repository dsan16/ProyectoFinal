import sqlite3
import pandas as pd

conn = sqlite3.connect("transacciones.db")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("Tablas:", tables)

df = pd.read_sql_query("SELECT * FROM transacciones where step > 742 LIMIT 10;", conn)
print(df)

conn.close()