import numpy as np

class ReLU:
    def __init__(self):
        self.cache = None
        self.dx = None

    def forward(self, x):
        self.dx = None
        out = np.maximum(x, 0)

        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache
        out = np.where(out > 0, 1, out)
        dx = dout * out
        self.dx = dx
        return dx