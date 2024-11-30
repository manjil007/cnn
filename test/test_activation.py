import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np

from activation import Activation


def test_relu(x: np.ndarray):
    activation: Activation = Activation()
    relu_f = activation.forward(x)
    print("Forward Output")
    print(relu_f)
    relu_b = activation.backward(x)
    print("Backward output")
    print(relu_b)


if __name__ == "__main__":
    input = np.random.uniform(-10, 1, (3, 28, 28))
    print(input)
    test_relu(input)
