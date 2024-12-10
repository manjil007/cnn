"""
This file contains utility functions used in cnn operation. 
    - padding
"""

import numpy as np
import sys
import os


def padding(image, pad):
    """
    Add zero-padding to each channel of a 3D umpy tensor.

    Parameters:
    ----------

    image : numpy.ndarray

        A 3D numpy array of shape (C, H, W), where:
        - C is the number  of channels (3 for RGB iamges)
        - H is the height of the image
        - W is the width of the image

    pad : int
        The amount of zero-padding to add around the borders on each channels.
        Padding ill be added symmetrically so that there is equal padding on all sides.

    Returns:
    --------

    numpy.ndarray
        A 3D array with shape (C, H + 2* pad , W + 2 * pad)

    """

    # There are three channels; each channel will have paddings added
    C, H, W = image.shape
    output_shape = (C, H + 2 * pad, W + 2 * pad)
    output = np.zeros(output_shape)

    for channel_index in range(C):
        output[channel_index] = np.pad(
            image[channel_index],
            pad_width=((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )
    return output


# revised
def MaxPool(conv_image, kernel_size, stride=1):
    """
    Subsampling layer.

    Parameters:
    -----------
        - conv_image : image in numpy and dimension (B, C, H, W)
            - B is the batch size
            - C is the channel
            - H is the height
            - W is the wieght

        - kernel_size : kernel_size of the pooling layer. The kernel stays rectangular
        - stride : stide of the pooling layer

    Returns:
    --------
    pooled batch in numpy

    """
    batch_size, input_channel, input_height, input_width = conv_image.shape

    output_height = ((input_height - kernel_size) // stride) + 1
    output_width = ((input_width - kernel_size) // stride) + 1

    """
        print("input height : " , input_height ) 
        print("input width : " , input_width ) 
        print("output_height : ", output_height )
        print("output_width : ", output_width ) 
        """

    pooled = np.zeros((batch_size, input_channel, output_height, output_width))

    # 2. Scan through the input image and get the max

    for b in range(batch_size):  # iterate over batch

        for c in range(input_channel):  # iterate over channels

            for j in range(0, input_height - kernel_size + 1, stride):  # height

                for i in range(0, input_width - kernel_size + 1, stride):  # width

                    window = conv_image[
                        b, c, j : j + kernel_size, i : i + kernel_size
                    ]  # window for the current iteration

                    max_value = np.max(window)  # max value from the current window

                    pooled[b, c, j // stride, i // stride] = (
                        max_value  # Store the max value in the output array
                    )

    return pooled


# revised
def backward_wrt_input(dL_dz, kernel, stride=1):
    """
    Description:
    -----------
    The image input is an input of dimension (1, H, W).
    KEY INSIGHT : The CNN Backpropagation operation with stride>1 is identical to a stride = 1.

    Parameters:
    -----------
        dL_dz  : Partial derivative of Loss wrt the kernel
        kernel : The kernel/filter of the current layer
        stride : Although defined, the CNN backpropagation operation with stride >1  is identical to stride=1. Source: https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

    Returns:
    --------
        The partial derivative dL_dx
    ----

    Note:
        i)  See if stride is a factor here. ->> The CNN backpropagation operation with stride >1  is identical to stride=1. Source: https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
        ii) The weight update will happen outside this function
        iii)

        key formula : dL_dx = full_convolution(kernel, loss gradient dL_dz )
        dL_dO is dL_dz in MMP's code

        z/O is the output of the convolution layer
        dL_do is same as dL_dz

    ----

    """

    print("dL_dz shape ", dL_dz.shape)  # the loss should be a

    # sys.exit(-1)
    # Since this is a single layer, let us ignore the dimension C.

    # Step 1. filp the filter/kernel 180 degree; assume that a single kernel is 3D i.e. (1, H, W)
    kernel_rotated = kernel[:, ::-1, ::-1]  # replace this with

    # Step 2. pad the dL_dz
    dL_dz_padded_1 = padding(dL_dz_test, 1)  # padding dL_dZ with 1)

    """
    
    dL_dz_test_padded_1 shape : (1, 30, 30) 
    kernel_rotated : (1, 5, 5) 
    
    convolutional arithmetic : ( ((30-5) + 2* padding) / stride) ) + 1  
        
    """

    image_height = dL_dz_padded_1.shape[-1]
    kernel_height = kernel_rotated.shape[-1]
    stride_ = stride

    # print("---------")

    # print("dz_dz after padding, length : ", image_height)
    # print("kernel height, length : ", kernel_height)
    # print("stride, length : ", stride_)

    output_dimension = (image_height - kernel_height // stride) + 1

    # print("output dimension : ", output_dimension)

    # print("---------")
    # sys.exit(-1)

    # print("dL_dz before padding : " , dL_dz_test.shape )
    # print("dL_dz_padded_1 , after initialization : ", dL_dz_padded_1.shape )

    # print("Update : Padding applied successfully" )
    output = np.zeros((output_dimension, output_dimension))
    print("output shape ", output.shape)

    # sys.exit(-2)

    # Step 3:
    # perform convolution between dL_dz_padded_1 and kernel_rotated- the results will be in output matrix

    print("--------------")
    # Step 3: Perform the convolution operation --->> This should be a separate function.,

    # add another loop to iterate across all the channels
    # (Batch Size, Channel, Height, Width) --> in my code (Channel (C), Height (i), Width (j))

    for i in range(0, output_dimension):

        for j in range(0, output_dimension):

            # Extract the patch of the same size as the kernel
            patch = dL_dz_padded_1[0, i : i + kernel_height, j : j + kernel_height]

            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel_rotated[0])

            print("i, j: (", i, j, ")")

    # print("dL_dx: After convolution is applied")
    # print(output)
    print("--------------")

    """
    output is dL/dX 
    """
    # for i in range(dl_dz_padded_1.shape[0]): #6, 32, 32 #x = 32, 32
    #    convolve(dl_dz_padded[i], kernel)

    #    output[] = convolute(patch, kernel)

    # print("dL_dx : after convolution is applied", )

    # print("output " , output.shape)

    return output  # this is dL/dx


def max_pool_backward(d_out, input_matrix, pool_size, stride):
    """
    Description:
    ------------
    - Backpropagation through max pooling layer for batch inputs.
    - The pooling layers are square
    - The batch of the input image is taken into consideration



    Parameters:
    -----------
    - d_out (numpy.ndarray): Gradient of the loss w.r.t. the output of the max-pooling layer,
                             shape (B, C, out_height, out_width).


    - input_matrix (numpy.ndarray): max-pooling layer input during forward pass, shape is (B, C, H, W).
    - pool_size (int): square pooling window

    - stride (int): stride operation

    Returns:
    --------
    - d_input (numpy.ndarray): Gradient of the loss w.r.t. the input of the max-pooling layer,
                               shape (batch, channel, height, width).
    """
    batch_size, channels, input_h, input_w = input_matrix.shape
    _, _, output_h, output_w = d_out.shape

    d_input = np.zeros_like(input_matrix)  # initialize gradient wrt  input matrix

    for b in range(batch_size):

        for c in range(channels):

            for i in range(output_h):

                for j in range(output_w):

                    start_h = i * stride
                    start_w = j * stride

                    end_h = start_h + pool_size
                    end_w = start_w + pool_size

                    pooling_region = input_matrix[b, c, start_h:end_h, start_w:end_w]

                    max_val = np.max(pooling_region)

                    for m in range(pool_size):

                        for n in range(pool_size):

                            if pooling_region[m, n] == max_val:

                                d_input[b, c, start_h + m, start_w + n] += d_out[
                                    b, c, i, j
                                ]
                                break  # exit once when the max value gradient is assigned -

    return d_input
