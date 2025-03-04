import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

datos = pd.read_csv("DS_Banca.csv")

datos_y = datos["isFraud"]
print(datos_y.head())
datos_x = datos.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "isFraud"])

datos_x = pd.get_dummies(datos_x)
print(datos_x.head())

escalar = StandardScaler()
datos_x = escalar.fit_transform(datos_x)

x_train, x_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size=0.3, random_state=2)

print("X Train: {}, X Test: {}, Y Train: {}, Y Test: {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

n_entradas1 = x_train.shape[1]
print(n_entradas1)

t_x_train = torch.from_numpy(x_train).float().to("cpu")
t_x_test = torch.from_numpy(x_test).float().to("cpu")
t_y_train = torch.from_numpy(y_train.values).float().to("cpu")
t_y_test = torch.from_numpy(y_test.values).float().to("cpu")
t_y_train = t_y_train[:,None]
t_y_test = t_y_test[:,None]

test = TensorDataset(t_x_test, t_y_test)
print(test[0])

print(t_x_train)

class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 12)
        self.linear2 = nn.Linear(12, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        pred_f = torch.sigmoid(input=self.linear3(pred_2))

        return pred_f
    
lr = 0.001
epochs = 2000
estatus_print = 100

model = Red(n_entradas=n_entradas1)
print(model.parameters())
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
print(model)
historico = pd.DataFrame()

for epoch in range(1, epochs + 1):
    y_pred = model(t_x_train)
    loss = loss_fn(input= y_pred, target= t_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % estatus_print == 0:
        print(f"\nEpoch {epoch} Loss: {round(loss.item(), 4)}")

    with torch.no_grad():
        y_pred = model(t_x_test)
        y_pred_class = y_pred.round()
        correct = (y_pred_class == t_y_test).sum()
        accuracy = 100 * correct / float(len(t_y_test))
        if epoch % estatus_print == 0:
            print("Accuracy: {}".format(accuracy.item()))

    df_tmp = pd.DataFrame(data={
        'Epoch': epoch,
        'Loss': round(loss.item(), 4),
        'Accuracy': round(accuracy.item(), 4)
    },
    index=[0])
    historico = pd.concat(objs= [historico, df_tmp], ignore_index=True, sort=False)

print("Accuracy final: {}".format(round(accuracy.item(), 4)))
