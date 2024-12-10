import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
from CNN_model import cnn


def test_cnn(x: np.ndarray, kernel_size: int, stride: int, padding: int):
    in_channel = x.shape[1]
    out_channel = 6
    cnn_model = cnn(in_channel, out_channel, kernel_size, stride, padding)
    output = cnn_model.forward(x)
    dl_dk, dl_db, dl_dx = cnn_model.backward(output)

    print("Shape of input = ", x.shape)
    print(
        f"in_channel, out_channel, stride, padding = {in_channel}, {out_channel}, {stride}, {padding}"
    )
    print(f"Kernel size = {kernel_size}")
    print("Shape out output of forward function: ", output.shape)
    print(
        "Shape of partial derivative with respect to kernel (weight or filter): ",
        dl_dk.shape,
    )
    print("Shape of partial derivative with respect to bias: ", dl_db.shape)
    print("Shape of partial derivative with respect to input: ", dl_dx.shape)


if __name__ == "__main__":
    stride = 1
    padding = 0
    input = np.random.rand(5, 3, 6, 6)
    kernel_size = 2

    test_cnn(input, kernel_size, stride, padding)
