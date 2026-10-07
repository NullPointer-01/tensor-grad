import random
from tensor import Tensor


class Neuron:
    def __init__(self, n_inputs, activation=True):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Tensor(0)
        self.activation = activation

    def __call__(self, X):
        # Process each sample independently, computes dot(w, x) + b, optionally followed by ReLU;
        y_pred = []
        for x in X:
            y = self.b
            for wi, xi in zip(self.w, x):
                y += wi * xi

            if self.activation:
                y = y.relu()

            y_pred.append(y)

        return y_pred

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, layer_size, activation=True):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(layer_size)]

    def __call__(self, X):
        result = [neuron(X) for neuron in self.neurons]
        if len(self.neurons) == 1:
            return result[0]

        # Transposes (neurons, batch) → (batch, neurons).
        return [list(row) for row in zip(*result)]

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(self, n_inputs, layer_sizes):
        self.layers = []

        prev_n_inputs = n_inputs
        for i, layer_size in enumerate(layer_sizes):
            # Last layer has no activation
            self.layers.append(
                Layer(prev_n_inputs, layer_size, i != len(layer_sizes) - 1)
            )
            prev_n_inputs = layer_size

    def __call__(self, X):
        x_input = X
        for layer in self.layers:
            y_pred = layer(x_input)
            x_input = y_pred

        return y_pred

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
