import math
from backward import (
    NoOpBackward,
    AddBackward,
    MulBackward,
    ReluBackward,
    ExpBackward,
    LogBackward,
    PowBackward,
)


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0  # accumulated gradient (dL/dself)
        self.grad_fn = NoOpBackward()  # leaf nodes have no-op backward

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data)
        result.grad_fn = AddBackward(self, other, result)

        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data)
        result.grad_fn = MulBackward(self, other, result)

        return result

    def __truediv__(self, other):
        return self * other**-1

    def relu(self):
        result = Tensor(self.data if self.data >= 0 else 0)
        result.grad_fn = ReluBackward(self, result)

        return result

    def exp(self):
        result = Tensor(math.exp(self.data))
        result.grad_fn = ExpBackward(self, result)

        return result

    def log(self):
        result = Tensor(math.log(self.data))
        result.grad_fn = LogBackward(self, result)

        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Expected int or float exponent"
        result = Tensor(self.data**other)
        result.grad_fn = PowBackward(self, other, result)

        return result

    def backward(self):
        # Build topological order so each node is visited after all its parents.
        topo = []
        visited = set()

        def topological_sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for par in tensor.grad_fn.parents:
                    topological_sort(par)
                topo.append(tensor)

        topological_sort(self)

        self.grad = 1.0  # dL/dL = 1
        for tensor in reversed(topo):
            tensor.grad_fn()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data})"
