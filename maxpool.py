import numpy as np
import math


class MaxPool:
    def __init__(self, size: int):
        self.size = size

    def forward(self, input: np.ndarray):
        """Forward propagation of max pool

        Args:
            input (np.ndarray): Normally it is the output of the convolution layer

        Returns:
            _type_: np.ndarray,
            It will be use as the input for the next layer.
        """
        self.input = input
        # it is for backproagration of input.
        # same sape as input we can know the index for it when needed
        x = input
        self.forward_index = np.full(x.shape, None)
        B, C, H, W = x.shape
        max_shape = int(math.floor(H - self.size) / self.size + 1)
        pool_x = np.zeros((B, C, max_shape, max_shape))
        for b in range(B):
            for c in range(C):
                h1 = 0
                for h in range(0, H, self.size):
                    w1 = 0
                    for w in range(0, W, self.size):
                        max_value = x[b, c, h, w]
                        max_index = (b, c, h, w)
                        for i in range(self.size):
                            for j in range(self.size):
                                if x[b, c, h + i, w + j] > max_value:
                                    max_value = x[b, c, h + i, w + j]
                                    max_index = (b, c, h + i, w + j)
                        self.forward_index[
                            max_index[0], max_index[1], max_index[2], max_index[3]
                        ] = max_value
                        pool_x[max_index[0], max_index[1], h1, w1] = max_value
                        w1 += 1
                    h1 += 1
        return pool_x

    def backward(self, dz: np.ndarray):
        """Max pool backpropagation

        Args:
            dz (np.ndarray): dl/dz of the las layer.

        Returns:
            _type_: np.ndarray,
            It return the dl/dx which will be used in previous layer as dl/dx if we have any previous layer before.
        """
        dx = np.zeros_like(self.forward_index)
        B, C, H, W = self.forward_index.shape
        for b in range(B):
            for c in range(C):
                h1 = 0
                for h in range(H):
                    w1 = 0
                    for w in range(W):
                        if self.forward_index[b, c, h, w]:
                            dx[b, c, h, w] = float(dz[b, c, h1, w1])
        return dx
