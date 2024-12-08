import numpy as np


class TwoLayerNetwork:
    """
    Creating the 2 layer linear network.
    """
    def __init__(self, input: np.ndarray, hidden_size: int = 12, out_size: int = 10):
        np.random.seed(5)
        self.input = input
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch, self.in_size = self.input.shape
        self._weight_init()
        self._gradient_init()

    def _weight_init(self):
        # Weight and bias initialization
        self.W1 = np.random.rand(self.in_size, self.hidden_size)
        self.B1 = np.random.rand(self.hidden_size)

        self.W2 = np.random.rand(self.hidden_size, self.out_size)
        self.B2 = np.random.rand(self.hidden_size)

    def _gradient_init(self):
        # Gradient initialization
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.B1)

        self.dw2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.B2)

    def forward(self, act):

        self.Z1 = self.input @ self.W1 + self.B1  # z = x2 + b
        self.A1 = act(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.B2
        return self.Z2

    def backward(self, dz, act_der):
        # Second layer backpropagation
        self.dw2 = dz @ self.A1
        self.db2 = np.sum(dz)
        dx2 = dz @ self.W2

        # Activation layer backpropagation,
        # Activation layer derivation with respect to what was passed during forward propagation which is Z1
        dz1 = dx2 * act_der(self.Z1)

        # First layer backprogation (same as second layer last layer output is dz1)
        self.dw1 = dz1 @ self.input
        self.db1 = np.sum(dz1)
        dx1 = dz1 @ self.W1

        # Returning dx1 becouse this will be unflatten to become the dl/dz for convolution layer.
        return dx1
