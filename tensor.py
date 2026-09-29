import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data)
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data)
        return result

    def __truediv__(self, other):
        return self * other** -1

    def relu(self):
        result = Tensor(self.data if self.data >= 0 else 0)
        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Expected int or float exponent"
        
        result = Tensor(self.data ** other)
        return result

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self** -1

    def __repr__(self):
        return f"Tensor(data={self.data})"
