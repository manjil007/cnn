import numpy as np 
import sys 
import os 


"""
So far: There is only one pooling layer that will pool from each channel 
"""

def backward_max_pool(pooled_image): 
    """
    
    Description: 
    -----------
    - The backpropagation of max pool layer. The parameter is a single 3D channel  

    Parameters: 
    -----------
    - pooled_layer : For a single 3D channel, it is the output of the pooling layer. Say, it's shape is HxW  
    
    
    Returns : 
    ---------
    - HxW matrix which has for every entry in the matrix pooled_layer that is greater than 0, return 1. Otherwise 0. 
    
    """

    
    dL_wrt_pooling_layer  = (pooled_layer > 0).astype(int) 
    return dL_wrt_pooling_layer
