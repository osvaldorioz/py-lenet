from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import lenet
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

# Transformación para normalizar las imágenes
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Carga del conjunto de datos MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    
@app.post("/lenet")
def calculo(epochs: int, lr: float):
    output_file1 = 'dispersion.png'
    output_file2 = 'perdida.png'
    
    # Obtener datos y etiquetas
    data, targets = next(iter(trainloader))
    data = data.view(data.size(0), 1, 28, 28)  # Formato adecuado para LeNet

    # Entrenamiento del modelo
    #epochs=10
    learning_rate=lr
    output = lenet.train(data, targets, epochs, learning_rate)

    # Gráfica de dispersión de los resultados
    plt.figure(figsize=(12, 5))
    plt.scatter(np.arange(len(output)), output.detach().cpu().numpy().argmax(axis=1), c=targets.numpy(), cmap='viridis')
    plt.title("Resultados del Modelo LeNet")
    plt.colorbar(label="Etiquetas Reales")
    plt.tight_layout()
    plt.savefig(output_file1)
    #plt.show()

    # Gráfica de pérdidas
    losses = np.random.rand(10)  # Ejemplo de pérdidas generadas
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(1, 11), losses, marker='o')
    plt.title("Pérdida durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.grid(True)
    #plt.show()
    plt.tight_layout()
    plt.savefig(output_file2)

    plt.close()
    
    j1 = {
        "Grafica de dispersion": output_file1,
        "Grafica de perdida": output_file2
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/lenet-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
