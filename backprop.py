import numpy as np 
import sys 
import os 

def backward_max_pool(pooled_layer): 
    """
    
    Description: 
    -----------
    - The backpropagation of max pool layer. The parameter is a single 2D channel  

    Parameters: 
    -----------
    - pooled_layer : For a single 2D channel, it is the output of the pooling layer. Say, it's shape is mxn  
    
    
    Returns : 
    ---------
    - mxn matrix which has for every entry in the matrix pooled_layer that is greater than 0, return 1. Otherwise 0. 
    
    """

    dL_wrt_pooling_layer  = (pooled_layer > 0).astype(int) 
    return dL_wrt_pooling_layer



