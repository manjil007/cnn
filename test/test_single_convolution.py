import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from single_convolution import SingleConvolution


if __name__ == "__main__":
    np.random.seed(6)
    H, W = 5, 5
    k = 2
    dz_shape = 4
    # Random int of 0 and 1 for input and kernal for easy caclulation in testing.
    input = np.random.randint(0, 2, (H, W))
    single_conv = SingleConvolution(H, k)

    print("INPUT")
    print(input)
    print("KERNAL")
    print(single_conv.kernel)

    out = single_conv.forward(input)
    print("Forward Output")
    print(out)

    dz = np.random.randint(0, 2, (dz_shape, dz_shape))
    print("dl/dz")
    print(dz)
    print("BACKPROPAGATION")
    single_conv.backward(dz)
    print("dl/dw")
    print(single_conv.weight_gradient)
    print("dl/db")
    print(single_conv.bias_gradient)
    print("dl/dx")
    print(single_conv.dl_dx)
