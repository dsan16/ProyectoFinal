import pandas as pd
import sqlite3

df = pd.read_csv("DS_Banca_modify.csv")

# Asegurarse de que los nombres de columnas coincidan con los de tus scripts
df = df[[
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud'
]]

conn = sqlite3.connect("transacciones.db")

# Guardar el DataFrame en la base de datos
df.to_sql("transacciones", conn, if_exists="replace", index=False)

print("Base de datos creada")
conn.close()
