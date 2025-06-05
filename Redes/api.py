from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal, Optional, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# cd /d E:\Dani\ProyectoFinal\frontend


# --------------------------
# Conexión SQLite
# --------------------------
DB_PATH = "transacciones.db"
def get_connection():
    return sqlite3.connect(DB_PATH)

# --------------------------
# Pydantic para entrada del /predict
# --------------------------
class Transaction(BaseModel):
    type: Literal['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int = 0
    isFlaggedFraud: int = 0

# --------------------------
# Pydantic para las nuevas consultas
# --------------------------
class LimitRequest(BaseModel):
    limit: int = 10

class OriginRequest(BaseModel):
    nameOrig: str
    limit: int = 10

# --------------------------
# Definición de la red
# --------------------------
class Red(nn.Module):
    def __init__(self, n_entradas: int):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 128)
        self.linear2 = nn.Linear(128, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        x = torch.sigmoid(self.linear1(inputs))
        x = torch.sigmoid(self.linear2(x))
        return torch.sigmoid(self.linear3(x))

# --------------------------
# Carga de pesos (state_dict)
# --------------------------
state_dict = torch.load("modelo_fraude_weights.pth", map_location="cpu")
model = Red(n_entradas=13)
model.load_state_dict(state_dict)
model.eval()

# --------------------------
# LabelEncoder
# --------------------------
encoder = LabelEncoder()
encoder.classes_ = np.array(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

# --------------------------
# Funciones auxiliares
# --------------------------
def get_tx_counts(nameOrig: str, nameDest: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transacciones WHERE nameOrig = ?", (nameOrig,))
    orig_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM transacciones WHERE nameDest = ?", (nameDest,))
    dest_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return orig_count, dest_count

def transformar_transaccion(tx: dict, orig_count: int, dest_count: int):
    step = tx["step"]
    type_encoded = encoder.transform([tx["type"]])[0]
    amount = tx["amount"]
    oldbalanceOrg = tx["oldbalanceOrg"]
    newbalanceOrig = tx["newbalanceOrig"]
    oldbalanceDest = tx["oldbalanceDest"]
    newbalanceDest = tx["newbalanceDest"]
    is_transfer = int(tx["type"] == 'TRANSFER')
    is_cashout = int(tx["type"] == 'CASH_OUT')
    hour_of_day = step % 24
    is_merchant = int(tx["nameDest"].startswith('M'))

    features = np.array([
        step,
        type_encoded,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        is_transfer,
        is_cashout,
        orig_count,
        dest_count,
        hour_of_day,
        is_merchant
    ], dtype=np.float32).reshape(1, -1)

    return torch.tensor(features, dtype=torch.float32)

def get_next_step() -> int:
    """
    Lee el mayor valor de 'step' en la tabla y devuelve +1,
    o 1 si la tabla está vacía.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(step) FROM transacciones")
    max_step = cur.fetchone()[0] or 0
    cur.close()
    conn.close()
    return max_step + 1

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint original de predicción
@app.post("/predict")
def predecir_fraude(req: Transaction):
    # 1) Calculamos el nuevo step
    step = get_next_step()

    # 2) Construimos el dict completo, inyectando el step
    tx = req.dict()
    tx["step"] = step

    # 3) Conteos en base de datos
    orig_count, dest_count = get_tx_counts(tx["nameOrig"], tx["nameDest"])

    # 4) Transformamos y predecimos
    tensor_entrada = transformar_transaccion(tx, orig_count, dest_count)
    with torch.no_grad():
        pred = model(tensor_entrada).item()
        resultado = int(pred >= 0.5)

    # 5) (Opcional) podrías aquí insertar la transacción en la BD si quieres
    #    registrar history, pero no es estrictamente necesario para predecir.

    return {
        "probabilidad_fraude": round(pred, 4),
        "es_fraude": resultado,
        "orig_tx_count": orig_count,
        "dest_tx_count": dest_count
    }

@app.get("/transactions/all")
def all_transactions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM transacciones LIMIT 8000")
    cols = [c[0] for c in cur.description]

    def row_generator():
        yield "["
        first = True
        for row in cur:
            if not first:
                yield ","
            else:
                first = False
            yield json.dumps(dict(zip(cols, row)))
        yield "]"
        cur.close()
        conn.close()

    return StreamingResponse(row_generator(), media_type="application/json")

@app.get("/transactions/allFraud")
def all_transactions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM transacciones WHERE isFraud = 1")
    cols = [c[0] for c in cur.description]

    def row_generator():
        yield "["
        first = True
        for row in cur:
            if not first:
                yield ","
            else:
                first = False
            yield json.dumps(dict(zip(cols, row)))
        yield "]"
        cur.close()
        conn.close()

    return StreamingResponse(row_generator(), media_type="application/json")

@app.get("/transactions/by_origin", response_model=List[dict])
def transactions_by_origin(
    nameOrig: str = Query(..., description="ID del origen"),
    limit: int = Query(10, ge=1, description="Número máximo de registros")
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM transacciones WHERE nameDest = ? LIMIT ?",
        (nameOrig, limit)
    )
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    cur.close(); conn.close()
    return [dict(zip(cols, row)) for row in rows]

@app.post("/transactions/insert")
def insert_transaction(transaccion: Transaction):
    conn = get_connection()
    cur = conn.cursor()
    valores = (
        get_next_step(),
        transaccion.type,
        transaccion.amount,
        transaccion.nameOrig,
        transaccion.oldbalanceOrg,
        transaccion.newbalanceOrig,
        transaccion.nameDest,
        transaccion.oldbalanceDest,
        transaccion.newbalanceDest,
        transaccion.isFraud,
        transaccion.isFlaggedFraud
    )
    try:
        cur.execute(
            """
            INSERT INTO transacciones
            (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
            nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            valores
        )
    except sqlite3.IntegrityError as e:
        return {"error": "Error de integridad: " + str(e)}
    
    conn.commit()
    cur.close()
    conn.close()
    return {"mensaje": "Transacción insertada correctamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
