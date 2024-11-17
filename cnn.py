import numpy as np

class cnn:
    def __init__(self, num_input, num_output, kernel_size, stride, padding):
        self.num_input = num_input
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = np.zeros(kernel_size)

    def __repr__(self):
        return (f"cnn(num_input={self.num_input}, num_output={self.num_output}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")



if __name__ == "__main__":
    num_input = 3  
    num_output = 2  
    kernel_size = (3, 3, 3)  
    stride = 1  
    padding = 0  

    cnn_model = cnn(num_input, num_output, kernel_size, stride, padding)

    print("Initialized CNN model:")
    print(cnn_model)

    # Print the kernel to verify its dimensions and values
    print("\nKernel initialized with zeros:")
    print(cnn_model.kernel)
    print(f"Kernel shape: {cnn_model.kernel.shape}")



