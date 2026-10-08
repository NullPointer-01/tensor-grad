# tensor-grad

An automatic differentiation engine built from scratch using Python and NumPy.

This project implements basic tensors, backpropagation, and neural network layers to understand how deep learning frameworks work internally.

## Features

* Tensor operations
* Automatic differentiation
* Backpropagation
* ReLU activation
* Softmax and cross-entropy loss
* Linear layers
* Multi-layer perceptron (MLP)
* MNIST example

## Installation

Clone the repository:

```bash
git clone https://github.com/NullPointer-01/tensor-grad.git
cd tensor-grad
```

Install NumPy:

```bash
pip install numpy
```

## Usage

```python
from tensor import Tensor

x = Tensor(2.0)
y = Tensor(3.0)

z = x * y
z.backward()

print(x.grad)
print(y.grad)
```

## Neural Network

A simple MLP can be created using:

```python
from nn import MLP

model = MLP(784, [128, 64, 10])
```

An MNIST training example is available in `demo.ipynb`.


## License

MIT
