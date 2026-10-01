# Each class computes dL/d(inputs) given dL/d(result) and stores parent refs
# for the topological traversal in Tensor.backward().

class NoOpBackward:
    # Used by leaf tensors (weights, biases, inputs) — no parents, no grad to propagate.
    def __init__(self):
        self.parents = ()

    def __call__(self):
        pass


class AddBackward:
    # d(x+y)/dx = 1, d(x+y)/dy = 1
    def __init__(self, x, y, result):
        self.x = x
        self.y = y
        self.result = result
        self.parents = (x, y)

    def __call__(self):
        self.x.grad += self.result.grad
        self.y.grad += self.result.grad


class MulBackward:
    # d(x*y)/dx = y, d(x*y)/dy = x
    def __init__(self, x, y, result):
        self.x = x
        self.y = y
        self.result = result
        self.parents = (x, y)

    def __call__(self):
        self.x.grad += self.y.data * self.result.grad
        self.y.grad += self.x.data * self.result.grad


class ReluBackward:
    # d(relu(x))/dx = 1 if x >= 0 else 0
    def __init__(self, x, result):
        self.x = x
        self.result = result
        self.parents = (x,)

    def __call__(self):
        self.x.grad += self.result.grad if self.x.data >= 0 else 0


class ExpBackward:
    # d(e^x)/dx = e^x = result.data
    def __init__(self, x, result):
        self.x = x
        self.result = result
        self.parents = (x,)

    def __call__(self):
        self.x.grad += self.result.data * self.result.grad


class LogBackward:
    # d(log x)/dx = 1/x
    def __init__(self, x, result):
        self.x = x
        self.result = result
        self.parents = (x,)

    def __call__(self):
        self.x.grad += (1 / self.x.data) * self.result.grad


class PowBackward:
    # d(x^n)/dx = n * x^(n-1)
    def __init__(self, x, exp, result):
        self.x = x
        self.exp = exp
        self.result = result
        self.parents = (x,)

    def __call__(self):
        self.x.grad += self.exp * (self.x.data ** (self.exp - 1)) * self.result.grad
