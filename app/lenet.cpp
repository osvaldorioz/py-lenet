#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace torch;

// Definición de la red LeNet
struct LeNet : nn::Module {
    nn::Conv2d conv1, conv2;
    nn::Linear fc1, fc2, fc3;

    LeNet() :
        conv1(nn::Conv2dOptions(1, 6, 5)),
        conv2(nn::Conv2dOptions(6, 16, 5)),
        fc1(16 * 4 * 4, 120),
        fc2(120, 84),
        fc3(84, 10) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = relu(conv1->forward(x));
        x = max_pool2d(x, 2);
        x = relu(conv2->forward(x));
        x = max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = relu(fc1->forward(x));
        x = relu(fc2->forward(x));
        return log_softmax(fc3->forward(x), 1);
    }
};

// Entrenamiento del modelo
torch::Tensor train(torch::Tensor data, torch::Tensor targets, int epochs, double learning_rate) {
    py::gil_scoped_release release;  
    
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    LeNet model;
    model.to(device);

    auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(learning_rate));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train();
        auto output = model.forward(data.to(device));
        auto loss = torch::nll_loss(output, targets.to(device));

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Epoch [" << epoch + 1 << "] Loss: " << loss.item<float>() << std::endl;
    }

    return model.forward(data.to(device));
}

PYBIND11_MODULE(lenet, m) {
    py::class_<LeNet, std::shared_ptr<LeNet>>(m, "LeNet")
        .def(py::init<>())
        .def("forward", &LeNet::forward);

    m.def("train", &train, "Train LeNet model",
          py::arg("data"), py::arg("targets"), py::arg("epochs") = 10, py::arg("learning_rate") = 0.01);
}
