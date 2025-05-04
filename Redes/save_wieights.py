import torch
import torch.nn as nn

# --------------------------
# Definición de la misma clase Red usada al crear el modelo
# --------------------------
class Red(nn.Module):
    def __init__(self, n_entradas: int):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        x = torch.sigmoid(self.linear1(inputs))
        x = torch.sigmoid(self.linear2(x))
        return torch.sigmoid(self.linear3(x))

# --------------------------
# Carga completa del modelo (pickle)
# --------------------------
# Asegúrate de tener 'modelo_fraude.pth' en este directorio
full_model = torch.load(
    "modelo_fraude.pth",
    map_location="cpu",
    weights_only=False
)

# --------------------------
# Guardado solo de los pesos (state_dict)
# --------------------------
weights_file = "modelo_fraude_weights.pth"
torch.save(full_model.state_dict(), weights_file)
print(f"✔️ Pesos guardados en '{weights_file}'")
