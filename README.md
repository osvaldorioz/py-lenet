
LeNet es una arquitectura de red neuronal convolucional (CNN) propuesta por Yann LeCun en 1998, diseñada principalmente para la clasificación de imágenes. Es especialmente efectiva en tareas como el reconocimiento de dígitos escritos a mano, por ejemplo, en el conjunto de datos MNIST.

La arquitectura de LeNet incluye:
- **Capas convolucionales** para extraer características.
- **Capas de activación** como Tanh o ReLU.
- **Capas de agrupación (pooling)** para reducir la dimensionalidad.
- **Capas completamente conectadas (fully connected)** para realizar la clasificación.

---

### **Implementación en C++ con Pybind11**
El programa integra el modelo LeNet en C++ y expone su funcionalidad mediante Pybind11 para que pueda ser utilizado desde Python. 

#### **Aspectos clave de la implementación**
1. **Definición del modelo LeNet en C++:**  
   El modelo incluye capas convolucionales, de activación y totalmente conectadas.

2. **Entrenamiento en C++:**  
   La función `train` realiza el proceso de entrenamiento directamente en C++.

3. **Integración con Pybind11:**  
   Se emplea `py::gil_scoped_release release;` dentro de la función `train`. Esta línea es fundamental porque:

   - **`py::gil_scoped_release`** libera el Global Interpreter Lock (GIL) de Python durante las operaciones intensivas de C++ (como el entrenamiento).
   - Esto mejora el rendimiento, ya que evita que el flujo de Python bloquee la ejecución de otros hilos mientras se entrena el modelo.

#### **Fragmento clave del código en C++**
```cpp
py::array_t<float> train(torch::Tensor data, torch::Tensor targets, int epochs, float learning_rate) {
    py::gil_scoped_release release;  // Libera el GIL para optimizar el rendimiento

    auto model = std::make_shared<LeNet>();
    auto optimizer = torch::optim::SGD(model->parameters(), learning_rate);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, targets);
        loss.backward();
        optimizer.step();
    }

    return output;
}
```

---

### **Visualización en Python**
El script de Python:

✅ Carga el conjunto de datos MNIST.  
✅ Realiza el entrenamiento utilizando el modelo implementado en C++.  
✅ Genera dos gráficas:  
   - **Gráfica de dispersión:** Muestra las predicciones del modelo comparadas con las etiquetas reales.  
   - **Gráfica de pérdida:** Visualiza cómo evoluciona el error durante el entrenamiento.

---

### **Ventajas de esta implementación**
✅ Mayor rendimiento al delegar el cálculo a C++.  
✅ Uso eficiente del GIL para permitir operaciones concurrentes en Python.  
✅ Visualización clara del comportamiento del modelo con gráficos útiles para el análisis.  
