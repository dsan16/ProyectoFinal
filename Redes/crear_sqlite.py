import pandas as pd
import sqlite3

# Leer el CSV original
df = pd.read_csv("DS_Banca_modify.csv")

# Asegurarse de que los nombres de columnas coincidan con los de tus scripts
df = df[[
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud'
]]

# Crear la conexión SQLite
conn = sqlite3.connect("transacciones.db")

# Guardar el DataFrame en la base de datos
df.to_sql("transacciones", conn, if_exists="replace", index=False)

print("✅ Base de datos creada correctamente con los datos.")
conn.close()
