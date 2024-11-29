import numpy as np
import scipy.signal as signal

class cnn:
    def __init__(self, 
                 input_size, 
                 num_kernels, 
                 kernel_size):
        """
        Constructor for the cnn class. Initializes key attributes for the CNN: 
            - input_size: Size of the input image 
            - num_kernels : Number of convolutional filters to use
            - kernel_size : Size of the convolutional kernel/filter  
        """  
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        self.image = None 

    
    def __repr__(self):
        """
        Summarizes the initialized CNN object. 
        """
        
        
        return (f"cnn(num_input={self.num_input}, num_output={self.num_output}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
    
    def generate_cnn_patches(self, image, stride, padding):
        """
        
        Extracts sub-regions or small patches from the input image that the convolution operation will use. 
        Steps: 
            1. kernel_height (kernel_h) and width (kernel_w) based on the given kernel_size. 
            2. pads image with 0 if padding is specified 
            3. loops over the image with the specified stride, extracting kernel_{h} x kernel_{w} patches for each channel 
            4. returns a collection of these patches 
        
        """
        
        
        num_channel, image_h, image_w = image.shape

        # Determine kernel size
        if isinstance(self.kernel_size, int):
            kernel_h = kernel_w = self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            kernel_h, kernel_w = self.kernel_size

        # Add padding if necessary
        if padding > 0:
            padded_image = np.pad(
                image,
                pad_width=((0, 0), (padding, padding), (padding, padding)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_image = image


        # Initialize storage for patches
        patches = []
        for channel in range(num_channel):
            channel_patch = []
            for i in range(0, image_h - kernel_h + 1, stride):
                for j in range(0, image_w - kernel_w + 1, stride):
                    patch = padded_image[channel, i:i+kernel_h, j:j+kernel_w]
                    channel_patch.append(patch)
            channel_patch = np.array(channel_patch)
            patches.append(channel_patch)
        patches = np.array(patches)
        
        return patches
    
    def convolute(self, patch, kernel):
        """
        Performs the convolution operation between a single patch and a kernel 
        Steps: 
            1. Uses scipy.signal.correlate2d to compute the valid cross-correlation between the patch and kernel 
            2. Returns the resulting matrix
        """
        
        conv_mat = signal.correlate2d(patch, kernel, mode='valid')
        
        return conv_mat
    
    def convolve(self, img, stride, padding):
        """
        The core function that orchestrates the convolution process for the entire image for the entire image  
        """
         
        patches = self.generate_cnn_patches(img, stride, padding)
        num_channels, input_h, input_w = img.shape ##need to indexing value for right shape
        self.output_shape = (self.num_kernels, input_h - self.kernel_size + 1, input_w - self.kernel_size + 1)
        self.kernels_shape = (self.num_kernels, num_channels, self.kernel_size, self.kernel_size) 
        self.kernels = np.random.rand(*self.kernels_shape)
        self.biases = np.random.rand(*self.output_shape)
        self.output = np.zeros((self.num_kernels, input_h - self.kernel_size + 1, input_w - self.kernel_size + 1))

        print("kernels = ", self.kernels)

        for i in range(self.num_kernels):
            y = 0
            for kernel in self.kernels[i]:
                l = 0
                k = 0
                for patch in patches[y]:
                    self.output[i][k][l] += self.convolute(patch, kernel)
                    l += 1
                    
                    if l >= input_w - self.kernel_size + 1:
                        l = 0
                        k += 1
                y += 1
        
        
        return self.output

    def MaxPool(self, kernel_size , stride=1): 
        """
        Subsampling layer... 
        """
        
        
    def MinPool(kernel_size , stride =1): 
        """
        Subsampling layer... 
        """
    def AvgPool(kernel_size , stride = 1): 
        """
        Subsampling layer...
        """

    #max_pooling(self, convolved_img, stride)               


        
    #def max_pooling(self, convolved_img, stride)

                                


    # def generate_cnn_patches(self, image):
    #     num_channel, image_h, image_w = image.shape

    #     if isinstance(self.kernel_size, int):
    #         kernel_h = kernel_w = self.kernel_size
    #     elif isinstance(self.kernel_size, tuple):
    #         kernel_h, kernel_w = self.kernel_size

    #     if self.padding > 0:
    #         image = np.pad(image, pad_width=((self.padding, self.padding), (self.padding, self.padding) (0,0)), mode='constant', constant_values=0)

    #     output_height = (image_h -  kernel_h) // self.stride + 1
    #     output_width = (image_w - kernel_w) // self.stride + 1

    #     patches = []

    #     for i in range(0, output_height - kernel_h + 1, self.stride):
    #         for j in range(0, output_width - kernel_w + 1, self.stride):
    #             patch = image[i:i+kernel_h, j:kernel_w, :]
    #             patches.append(patch)

    #     return patches