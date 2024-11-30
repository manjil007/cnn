import numpy as np
import matplotlib.pyplot as plt


class Activation:
    def __init__(self, name="relu"):
        self.name = name

    def identity(self, a):
        return a

    def relu(self, a):
        return np.maximum(0, a)

    def tanh(self, a):
        return np.tanh(a)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def softmax(self, a):
        return np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)

    def dev_identity(self, x):
        return 1

    def dev_relu(self, x):
        return np.where(x > 0, 1, 0)

    def dev_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def dev_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def dev_softmax(self, z):
        ## for CE backpropagration softmax is combined with CE.
        return z

    def forward(self, a):
        if self.name == "identity":
            return self.identity(a)

        elif self.name == "relu":
            return self.relu(a)

        elif self.name == "tanh":
            return self.tanh(a)

        elif self.name == "sigmoid":
            return self.sigmoid(a)

        elif self.name == "softmax":
            return self.softmax(a)

        else:
            raise ValueError(
                f"Unknown activation function: {self.name}. Please select 'identity', 'relu', 'tanh', 'sigmoid', 'cross_entropy' "
            )

    def backward(self, a):
        if self.name == "identity":
            return self.dev_identity(a)

        elif self.name == "relu":
            return self.dev_relu(a)

        elif self.name == "tanh":
            return self.dev_tanh(a)

        elif self.name == "sigmoid":
            return self.dev_sigmoid(a)

        elif self.name == "softmax":
            return self.dev_softmax(a)

        else:
            raise ValueError(
                f"Unknown activation function: {self.name}. Please select 'identity', 'relu', 'tanh', 'sigmoid', 'cross_entropy' "
            )