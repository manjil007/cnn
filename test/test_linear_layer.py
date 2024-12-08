import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())


import numpy as np

from linear import TwoLayerNetwork
from activation import Activation
from utils import *


if __name__ == "__main__":
    np.random.seed(5)

    batch, feature, num_class = 100, 225, 10
    input = np.random.rand(batch, feature)

    y = np.random.randint(0,10,batch)
    one_hot_y = one_hot_y(y, num_classes=10)

    act: Activation = Activation("relu")

    linear_layer: TwoLayerNetwork = TwoLayerNetwork(input)
    logits = linear_layer.forward(act)

    prob = softmax(logits)
    loss = cross_entropy_loss(logits, one_hot_y)
    print(loss)

    dz = logits - one_hot_y

    dz = linear_layer.backward(dz, act)
    print("checking the shape of input and it's gradient")
    print(dz.shape == input.shape)
