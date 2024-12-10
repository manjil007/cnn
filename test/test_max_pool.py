import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
from max_pool import MaxPool, max_pool_backward

def test_max_pool(x:np.ndarray, kernel_size:int):
    output_forward = MaxPool(x, kernel_size)

    output_backward = max_pool_backward(output_forward, x, kernel_size, stride=1)



    return output_forward, output_backward



if __name__ == "__main__":
    stride = 1
    padding = 0
    kernel_size = 2

    input =  np.random.randint(low=1, high=10, size=(2,2,2,2))

    output_forward, output_backward = test_max_pool(input, kernel_size)

    print("size of Max pool forward output = ", output_forward.shape)
    print("size of Max pool backward output = ", output_backward.shape)


    
