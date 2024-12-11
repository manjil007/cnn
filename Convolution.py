import numpy as np
import scipy.signal as signal


class cnn:
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        np.random.seed(5)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels_shape = (
            self.out_channel,
            self.in_channel,
            self.kernel_size,
            self.kernel_size,
        )
        self.kernels = np.random.randint(1, 4, self.kernels_shape) # * 0.01
        self.biases = np.zeros(self.out_channel)


    def convolve(self, img, kernel):

        img_h, img_w = img.shape
        kernel_h, kernel_w = kernel.shape

        # Calculate the output dimensions
        output_h = (img_h - kernel_h) // self.stride + 1
        output_w = (img_w - kernel_w) // self.stride + 1

        # Initialize the output matrix
        convoluted_output = np.zeros((output_h, output_w))

        # Perform convolution
        for i in range(0, output_h):
            for j in range(0, output_w):
                # Determine the region of the input image to apply the kernel
                start_i = i * self.stride
                start_j = j * self.stride
                region = img[start_i : start_i + kernel_h, start_j : start_j + kernel_w]

                # Compute the convolution (element-wise multiplication and sum)
                convoluted_output[i, j] = np.sum(region * kernel)

        return convoluted_output

    def forward(self, X):
        self.input = X


        if self.padding > 0:
            # Apply padding to all images in the batch
            X_padded = np.pad(
                X,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            X_padded = X

        batch_size, _, input_h, input_w = X_padded.shape
        output_h = (input_h - self.kernel_size) // self.stride + 1
        output_w = (input_w - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, self.out_channel, output_h, output_w)
        self.output = np.zeros(self.output_shape)

        for i, image in enumerate(X_padded):
            for j in range(self.out_channel):
                for m in range(self.in_channel):
                    cur_kernel = self.kernels[j, m]
                    conv_img = self.convolve(image[m], cur_kernel)
                    self.output[i, j] += conv_img
                self.output[i, j] += self.biases[j]

        return self.output

    def backward(self, dl_dz):  # dl_dz is the same size as ouput
        dl_dk = np.zeros(
            (
                self.input.shape[0],
                self.out_channel,
                self.in_channel,
                self.kernel_size,
                self.kernel_size,
            )
        )

        for b in range(self.input.shape[0]):
            weight_gradient = np.zeros((self.kernel_size, self.kernel_size))
            for o in range(self.out_channel):
                for k in range(self.in_channel):
                    curr_x = self.input[b, k]  # dz_dk
                    curr_dl_dz = dl_dz[b, o]  # k_h, k_w = curr_dl_dz.shape
                    for l in range(self.kernel_size):
                        for m in range(self.kernel_size):
                            patch = curr_x[
                                l : l + curr_dl_dz.shape[0], m : m + curr_dl_dz.shape[1]
                            ]
                            weight_gradient[l, m] = (patch * curr_dl_dz).sum()

                    dl_dk[b, o, k] = weight_gradient
        self.weight_gradient = np.sum(dl_dk, axis=0)

        # print("dl_dk = ", dl_dk[0][1][2])

        dl_db = np.zeros(self.out_channel)

        for b in range(dl_dz.shape[0]):
            for o in range(dl_dz.shape[1]):
                dl_db[o] = dl_db[o] + dl_dz[b, o].sum()

        self.bias_gradient = dl_db

        # print("dl_db = ", dl_db[0])

        dl_dx = np.zeros(self.input.shape)

        padding_height = self.kernels.shape[2] - 1
        padding_width = self.kernels.shape[3] - 1

        dl_dz_padded = np.pad(
            dl_dz,
            pad_width=(
                (0, 0),
                (0, 0),
                (padding_height, padding_width),
                (padding_height, padding_width),
            ),
            mode="constant",
            constant_values=0,
        )
        rot_kernels = self.kernels[:, :, ::-1, ::-1]
        B,C, H,W = self.input.shape
        for b in range(B):
            for c in range(C):
                y = np.zeros((H, W))
                for o in range(self.out_channel):
                    current = self.conv2d_stride_1(dl_dz_padded[b, o], rot_kernels[o, c], y.shape)
                    y += current
                self.assigning_value_matrix(b, c, dl_dx, y)    

        self.dl_dx = dl_dx

        return  self.weight_gradient, self.bias_gradient, self.dl_dx
    
    def update_params(self, lr):
        self.kernels -= lr * self.weight_gradient
        self.biases -= lr * self.bias_gradient


    def conv2d_stride_1(self, X, K, out_shape):
        """It takes the image and does the convolution on it."""
        Y = np.zeros(out_shape)
        i = 0
        k_h, k_w = K.shape
        while i < out_shape[0]:
            j = 0
            while j < out_shape[1]:
                curr_x = X[i : i + k_h, j : j + k_w]
                Y[i, j] = (curr_x * K).sum()
                j += 1
            i += 1
        return Y
    

    def assigning_value_matrix(self, b, c, out, y):
        """Plat the y into the out matrix given batch size as b and c as channel"""
        for h in range(y.shape[0]):
            for w in range(y.shape[1]):
                out[b, c, h, w] = y[h, w]
