from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import psycopg2

# --------------------------
# Configuración PostgreSQL
# --------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "transaccionesBancarias",
    "user": "admin",
    "password": "admin"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# --------------------------
# Clase Pydantic para la entrada
# --------------------------
class Transaction(BaseModel):
    step: int
    type: Literal['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int
    isFlaggedFraud: int

# --------------------------
# Cargar el modelo completo
# --------------------------
class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        x = torch.sigmoid(self.linear1(inputs))
        x = torch.sigmoid(self.linear2(x))
        return torch.sigmoid(self.linear3(x))
    
model = torch.load(
    "modelo_fraude.pth",
    map_location=torch.device("cpu"),
    weights_only=False 
)
model.eval()
# --------------------------
# LabelEncoder (clases conocidas)
# --------------------------
encoder = LabelEncoder()
encoder.classes_ = np.array(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

# --------------------------
# Función para obtener conteos desde PostgreSQL
# --------------------------
def get_tx_counts(nameOrig: str, nameDest: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transacciones WHERE nameOrig = %s", (nameOrig,))
    orig_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transacciones WHERE nameDest = %s", (nameDest,))
    dest_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()
    return orig_count, dest_count

# --------------------------
# Transformar la transacción para el modelo
# --------------------------
def transformar_transaccion(transaction: dict, orig_count: int, dest_count: int):
    step = transaction["step"]
    type_str = transaction["type"]
    amount = transaction["amount"]
    oldbalanceOrg = transaction["oldbalanceOrg"]
    newbalanceOrig = transaction["newbalanceOrig"]
    oldbalanceDest = transaction["oldbalanceDest"]
    newbalanceDest = transaction["newbalanceDest"]
    nameDest = transaction["nameDest"]

    type_encoded = encoder.transform([type_str])[0]
    is_transfer = int(type_str == 'TRANSFER')
    is_cashout = int(type_str == 'CASH_OUT')
    hour_of_day = step % 24
    is_merchant = int(nameDest.startswith('M'))

    entrada = np.array([
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

    return torch.tensor(entrada, dtype=torch.float32)

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()

@app.post("/predict")
def predecir_fraude(transaccion: Transaction):
    orig_count, dest_count = get_tx_counts(transaccion.nameOrig, transaccion.nameDest)
    tensor_entrada = transformar_transaccion(transaccion.dict(), orig_count, dest_count)

    with torch.no_grad():
        pred = model(tensor_entrada).item()
        resultado = int(pred >= 0.5)

    return {
        "probabilidad_fraude": round(pred, 4),
        "es_fraude": resultado,
        "orig_tx_count": orig_count,
        "dest_tx_count": dest_count
    }
"""
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    orig_tx_count: int
    dest_tx_count: int
    hour_of_day: int
    is_merchant: int """