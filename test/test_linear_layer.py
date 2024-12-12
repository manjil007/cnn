import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())


import numpy as np

from linear import TwoLayerNetwork
from relu import ReLU
from utils import *


if __name__ == "__main__":
    np.random.seed(5)

    batch, feature, num_class = 100, 225, 10
    input = np.random.rand(batch, feature)

    y = np.random.randint(0, 10, batch)
    one_hot_y = one_hot_y(y, num_classes=10)

    act: ReLU = ReLU()

    linear_layer: TwoLayerNetwork = TwoLayerNetwork(feature)
    logits = linear_layer.forward(input, act)

    prob = softmax(logits)
    loss = cross_entropy_loss(logits, one_hot_y)
    print(f"Loss: {loss}")

    dz = logits - one_hot_y

    dz = linear_layer.backward(dz, act)
    print(f"Print the dl_dz[0, :20] : {dz[0, :20]}")
    print(f"checking the shape of input and it's gradient: {dz.shape == input.shape}")
   
