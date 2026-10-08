import numpy as np
from tensor import Tensor


class Linear:
    def __init__(self, nin, nout, activation=True):
        # He initialization
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2 / nin), requires_grad=True)
        self.b = Tensor(np.zeros(nout), requires_grad=True)
        self.activation = activation

    def __call__(self, X):
        out = X @ self.W + self.b
        return out.relu() if self.activation else out

    def zero_grad(self):
        self.W.zero_grad()
        self.b.zero_grad()

    def parameters(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, nin, nouts):
        self.layers = []
        prev = nin

        for i, nout in enumerate(nouts):
            self.layers.append(Linear(prev, nout, i != len(nouts) - 1))
            prev = nout

    def __call__(self, X_train):
        X = X_train
        for layer in self.layers:
            X = layer(X)

        return X

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
