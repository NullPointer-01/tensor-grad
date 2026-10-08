import numpy as np


def _unbroadcast(grad, shape):
    """Reduce a broadcasted gradient back to the original shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    return grad


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self.grad_fn = lambda: None
        self.parents = ()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data + other.data, requires_grad=requires_grad)
        out.parents = (self, other)

        def _grad_fn():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)

            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out.grad_fn = _grad_fn
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data * other.data, requires_grad=requires_grad)
        out.parents = (self, other)

        def _grad_fn():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)

            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out.grad_fn = _grad_fn
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data @ other.data, requires_grad=requires_grad)
        out.parents = (self, other)

        def _grad_fn():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out.grad_fn = _grad_fn
        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Tensor(self.data**exp, requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += exp * self.data ** (exp - 1) * out.grad

        out.grad_fn = _grad_fn
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += out.grad * (self.data > 0)

        out.grad_fn = _grad_fn
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += out.grad / self.data

        out.grad_fn = _grad_fn
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += np.broadcast_to(out.grad, self.data.shape)

        out.grad_fn = _grad_fn
        return out

    def softmax(self):
        # Subtract row-max for numerical stability
        exp_data = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        p = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out = Tensor(p, requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += out.data * (
                out.grad - (out.grad * out.data).sum(axis=1, keepdims=True)
            )

        out.grad_fn = _grad_fn
        return out

    def cross_entropy_softmax(self, y):
        n = len(y)

        logits = self.data
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits

        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)

        log_sum_exp = max_logits + np.log(sum_exp)

        losses = log_sum_exp[:, 0] - logits[np.arange(n), y]
        loss = np.mean(losses)

        out = Tensor(loss)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            probs = exp_shifted / sum_exp
            probs[np.arange(n), y] -= 1.0
            probs /= n
            probs *= out.grad

            self.grad += probs

        out.grad_fn = _grad_fn
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out.parents = (self,)

        def _grad_fn():
            if not self.requires_grad:
                return

            self.grad += -out.grad

        out.grad_fn = _grad_fn
        return out

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0)

    def backward(self):
        topo = []
        visited = set()
        stack = [self]

        while stack:
            tensor = stack[-1]

            # First visit
            if tensor not in visited:
                visited.add(tensor)
                stack.extend(p for p in tensor.parents if p not in visited)

            # Second visit
            else:
                stack.pop()
                topo.append(tensor)

        self.grad = np.ones_like(self.data)

        for tensor in reversed(topo):
            tensor.grad_fn()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"
