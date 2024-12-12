import numpy as np

from relu import ReLU


class TwoLayerNetwork:
    """
    Creating the 2 layer linear network.
    """

    # def __init__(self, input: np.ndarray, hidden_size: int = 12, out_size: int = 10):
    def __init__(self, input_size: int, hidden_size: int = 12, out_size: int = 10):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.in_size = input_size
        self._weight_init()
        self._gradient_init()

    def _weight_init(self):
        # Weight and bias initialization
        self.W1 = np.abs(np.random.rand(self.in_size, self.hidden_size) * 0.01)
        self.B1 = np.zeros(self.hidden_size)

        self.W2 = np.abs(np.random.rand(self.hidden_size, self.out_size) * 0.01)
        self.B2 = np.zeros(self.out_size) 

    def _gradient_init(self):
        # Gradient initialization
        self.dw1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.B1)

        self.dw2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.B2)

    def forward(self, input: np.ndarray, act: ReLU):
        self.input = input
        self.Z1 = self.input @ self.W1 + self.B1  # z = x2 + b
        self.A1 = act.forward(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.B2
        return self.Z2

    def backward(self, dz, act_der: ReLU):
        # Second layer backpropagation
        self.dw2 = (dz.T @ self.A1).T / len(self.input)
        self.db2 = np.sum(dz, axis=0) / len(self.input)
        dx2 = dz @ self.W2.T

        # Activation layer backpropagation,
        dz1 =  act_der.backward(dx2)


        self.dw1 = (dz1.T @ self.input).T / len(self.input)
        self.db1 = np.sum(dz1, axis=0) / len(self.input)
        dx1 = dz1 @ self.W1.T / len(self.input)

        return dx1
    
    def update_params(self, lr, reg = 0.01):
        self.W1 -= lr * self.dw1 + reg * self.W1
        self.B1 -= lr * self.db1

        self.W2 -= lr * self.dw2 + reg * self.W2
        self.B2 -= lr * self.db2


    def zero_gradient(self):
        self._gradient_init()


