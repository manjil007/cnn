import numpy as np 
import sys 
import os 

class pool: 
    #def __init__(self, convolved image, stride ,type="max"): 
    #    """
    #    The constructor of the pooling layer.asd
    #    """
    #    # incomplete 

    def MaxPool( conv_image , kernel_size , stride=1): 
        """
        Subsampling layer... 
        
        """
        # Get input dimensions 
        input_height , input_width = conv_image.shape
    
        # Get the output dimensions 
        output_height = (input_height - kernel_size) // stride+1 
        output_width = (input_width - kernel_size) // stride+1 
    
        # 1. Initialize an np 2-D array for pooling kernel 
        pooled_ = np.zeros(( output_height, output_width )) 
        
        #print(pooled_.shape) 
            
        # 2. Scan through the input image and get the max 
        
        for j in range(0, input_height - kernel_size + 1, stride):
            for i in range(0, input_width - kernel_size + 1, stride):
                
                # Define the current window
                window = conv_image[j:j + kernel_size, i:i + kernel_size]
    
                #print("--- window.shape :" , window.shape) 
    
                # Get the max value in the window
                max_value = np.max(window)
    
                # Map max value to the output matrix
                pooled_[j // stride, i // stride] = max_value
    
        return pooled_ 

    def MinPool(conv_image, kernel_size, stride=1):
        """
        Min pooling function.
        """
        # Get input dimensions
        input_height, input_width = conv_image.shape
    
        # Get the output dimensions
        output_height = (input_height - kernel_size) // stride + 1
        output_width = (input_width - kernel_size) // stride + 1
    
        # Initialize an output matrix for pooling
        pooled_ = np.zeros((output_height, output_width))
    
        # Scan through the input image
        for j in range(0, input_height - kernel_size + 1, stride):
            for i in range(0, input_width - kernel_size + 1, stride):
                # Define the current window
                window = conv_image[j:j + kernel_size, i:i + kernel_size]
                
                # Get the min value in the window
                min_value = np.min(window)
                
                # Map min value to the output matrix
                pooled_[j // stride, i // stride] = min_value
    
        return pooled_
     
        
    def AvgPool(conv_image, kernel_size, stride=1):
        """
        Average pooling function.
        """
        # Get input dimensions
        input_height, input_width = conv_image.shape
    
        # Get the output dimensions
        output_height = (input_height - kernel_size) // stride + 1
        output_width = (input_width - kernel_size) // stride + 1
    
        # Initialize an output matrix for pooling
        pooled_ = np.zeros((output_height, output_width))
    
        # Scan through the input image
        for j in range(0, input_height - kernel_size + 1, stride):
            for i in range(0, input_width - kernel_size + 1, stride):
                # Define the current window
                window = conv_image[j:j + kernel_size, i:i + kernel_size]
                
                # Get the average value in the window
                avg_value = np.mean(window)
                
                # Map average value to the output matrix
                pooled_[j // stride, i // stride] = avg_value
    
        return pooled_


    #max_pooling(self, convolved_img, stride)       