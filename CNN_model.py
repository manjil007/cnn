import numpy as np
import scipy.signal as signal

class cnn:
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size) 
        self.kernels = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(self.out_channel)
        

    
    def generate_cnn_patches(self, image):
        num_channel, image_h, image_w = image.shape

        if isinstance(self.kernel_size, int):
            kernel_h = kernel_w = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            kernel_h, kernel_w = self.kernel_size


        num_channel, image_h, image_w = image.shape
        # Initialize storage for patches
        patches = []
        for channel in range(num_channel):
            channel_patch = []
            for i in range(0, image_h - kernel_h + 1, self.stride):
                for j in range(0, image_w - kernel_w + 1, self.stride):
                    patch = image[channel, i:i+kernel_h, j:j+kernel_w]
                    channel_patch.append(patch)
            channel_patch = np.array(channel_patch)
            patches.append(channel_patch)
        patches = np.array(patches)

        return patches
    
    # def convolute(self, patch, kernel):
    #     conv_mat = signal.correlate2d(patch, kernel, mode='valid')
    #     return conv_mat
    
    # def convolve(self, img, kernel):
    #     patches = self.generate_cnn_patches(img)
    #     _, _, output_h, output_w = self.output_shape
        
    #     convoluted_output_shape = (output_h, output_w)
    #     convoluted_output = np.zeros((convoluted_output_shape))

    #     for j in range(self.in_channel):
    #         row, col = 0, 0
    #         for patch in patches[j]:
    #             convoluted_output[row][col] = self.convolute(patch, kernel)
    #             if col >= output_w - 1:
    #                 col = 0
    #                 row += 1
    #             else:
    #                 col += 1
        
    #     return convoluted_output

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
                region = img[start_i:start_i + kernel_h, start_j:start_j + kernel_w]
                
                # Compute the convolution (element-wise multiplication and sum)
                convoluted_output[i, j] = np.sum(region * kernel)

        return convoluted_output
    
    def forward(self, X):
        if self.padding > 0:
            # Apply padding to all images in the batch
            X_padded = np.pad(
                X,
                pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
        else:
            X_padded = X

        batch_size, _, input_h, input_w = X_padded.shape
        output_h = (input_h - self.kernel_size) // self.stride + 1
        output_w = (input_w - self.kernel_size) // self.stride + 1
        self.output_shape = (batch_size, self.out_channel, output_h, output_w)
        self.output = np.zeros((batch_size, self.out_channel, output_h, output_w))
        self.input = X_padded

        for i, image in enumerate(X_padded):
            for j in range(self.out_channel):
                        for m in range(self.in_channel):
                            cur_kernel = self.kernels[j, m]
                            conv_img = self.convolve(image[m], cur_kernel)
                            if self.output[i, j].shape != (conv_img + self.biases[j]).shape:
                                print("different shape")
                            self.output[i, j] += conv_img + self.biases[j]

        return self.output
    
    def backward(self, dl_dz): #dl_dz is the same size as ouput
        dl_dk = np.zeros((self.input.shape[0], self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        
        for i in range (self.input.shape[0]):
            weight_gradient = np.zeros((self.kernel_size, self.kernel_size))
            for j in range(self.out_channel):
                for k in range(self.in_channel):
                    curr_x = self.input[i, k] #dz_dk
                    curr_dl_dz = dl_dz[i, j] #k_h, k_w = curr_dl_dz.shape
                    for l in range(self.kernel_size):
                        for m in range(self.kernel_size):
                            patch = curr_x[l:l+curr_dl_dz.shape[0], m:m+curr_dl_dz.shape[1]]
                            weight_gradient[l, m] = (patch * curr_dl_dz).sum()
                    
                    dl_dk[i, j, k] = weight_gradient
        self.weight_gradient = np.sum(dl_dk, axis=0)

        dl_db = np.zeros(self.out_channel)

        for i in range(dl_dz.shape[0]):
            for j in range(dl_dz.shape[0]):
                dl_db[j] = dl_db[j] + dl_dz[i, j].sum()

        self.bias_gradient = dl_db


        dl_dx = np.zeros(self.input.shape)
        
        padding_height = self.kernels.shape[2] - 1
        padding_height = self.kernels.shape[3] - 1

        dl_dz_padded = np.pad(
                dl_dz,
                pad_width=((0, 0), (0, 0), (padding_height, padding_height), (padding_height, padding_height)),
                mode='constant',
                constant_values=0
            )
        rot_kernels = self.kernels[:, :, ::-1, ::-1]


        for i in range(dl_dz_padded.shape[0]):
            y = np.zeros((self.in_channel, self.input.shape[2], self.input.shape[3]))
            for j in range(self.out_channel):
                for k in range(self.in_channel):
                    curr_kernel = rot_kernels[j, k]
                    curr_dl_dz = dl_dz_padded[i, j]
                    out = np.zeros((self.input.shape[2:]))
                    for l in range(out.shape[0]):
                        for m in range(out.shape[1]):
                            patch = curr_dl_dz[l : l + self.kernel_size, m : m + self.kernel_size]
                            out[l, m] = (patch * curr_kernel).sum()
                    y[k] += out
            
            for p in range(self.in_channel):
                for n in range(self.input.shape[2]):
                    for o in range(self.input.shape[3]):
                        dl_dx[i, p, n, o] = y[p, n, o]
        
        self.dl_dx = dl_dx

        return self.weight_gradient, self.bias_gradient, self.dl_dx

                    




                    






        




    
