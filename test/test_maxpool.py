import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from maxpool import MaxPool


if __name__ == "__main__":
    np.random.seed(20)

    B, C, H, W = 1, 2, 4, 4
    input = np.random.rand(B, C, H, W)
    print("INPUT")
    print(input)

    max_pool = MaxPool(2)
    pooled_x = max_pool.forward(input)
    print("POOLED OUTPUT")
    print(pooled_x)

    dz = np.random.random_sample(pooled_x.shape)
    print("RANDOM DZ FOR BCKPOPAGATION")
    print(dz)

    dx = max_pool.backward(dz)
    print("BACKPROPAGATION OF dz (PRINTING dx)")
    print(dx)
