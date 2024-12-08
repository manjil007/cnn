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
        output[channel_index] = np.pad(image[channel_index],
                                       pad_width=((pad, pad), (pad, pad)),
                                       mode='constant', 
                                       constant_values=0)
    return output


def MaxPool( conv_image , kernel_size , stride=1): 
        """
        Subsampling layer. 

        Parameters:
        -----------
            - conv_image : image in numpy and dimension (C, H, W) 
            - kernel_size : kernel_size of the pooling layer. The kernel stays rectangular  
            - stride : stide of the pooling layer 

        Returns:
        --------
        pooled in numpy 
        
        """
        # Get input dimensions 
        input_channel ,input_height , input_width = conv_image.shape
    
        # Get the output dimensions 
        output_height = (input_height - kernel_size) // stride+1 
        output_width = (input_width - kernel_size) // stride+1 

        """
        print("input height : " , input_height ) 
        print("input width : " , input_width ) 
        print("output_height : ", output_height )
        print("output_width : ", output_width ) 
        """
        
    
        # initialize a np 3D array for pooling kernel 
        pooled_ = np.zeros((input_channel, output_height, output_width )) 
        #print(pooled_.shape) 

        #sys.exit(-1) 
            
        # 2. Scan through the input image and get the max 

        # The iteration below will be honsistent for 3 dimensions; as of now it is for 2 . 
        # Apply pooling operation per channel
    
        for c in range(input_channel): # iterate across the channel 
            
            for j in range(0, input_height - kernel_size + 1, stride): # iterate across the height
                
                for i in range(0, input_width - kernel_size + 1, stride): # iterate across the width 
                    
                    # Define the current window
                    
                    window = conv_image[c, j:j + kernel_size, i:i + kernel_size]
        
                    # Get the max value in the window
                    max_value = np.max(window)
        
                    # Map max value to the output matrix
                    pooled_[c, j // stride, i // stride] = max_value # fixed to get integer size
        
        return pooled_

# Get the backprop wrt to the input 