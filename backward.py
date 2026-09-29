class NoOpBackward:
    def __init__(self):
        self.parents = ()

    def __call__(self):
        pass

class AddBackward:
    def __init__(self, x, y, result):
        self.x = x
        self.y = y
        self.result = result
        self.parents = (x, y)

    def __call__(self):
        self.x.grad += self.result.grad
        self.y.grad += self.result.grad

class MulBackward:
    def __init__(self, x, y, result):
        self.x = x
        self.y = y
        self.result = result
        self.parents = (x, y)

    def __call__(self):
        self.x.grad += self.y.data * self.result.grad
        self.y.grad += self.x.data * self.result.grad

class ReluBackward:
    def __init__(self, x, result):
        self.x = x
        self.result = result
        self.parents = (x, )

    def __call__(self):
        self.x.grad += self.result.grad if self.x.data >= 0 else 0

class PowBackward:
    def __init__(self, x, exp, result):
        self.x = x
        self.exp = exp
        self.result = result
        self.parents = (x, )

    def __call__(self):
        self.x.grad += self.exp * (self.x.data**(self.exp - 1)) * self.result.grad
