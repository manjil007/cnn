import numpy as np 
import sys 
import os 


"""
So far: There is only one pooling layer that will pool from each channel 
"""

def backward_max_pool(image_batch, pooling_layers): 
    """
    
    Description: 
    -----------
    - The backpropagation of max pool layer. The parameter is a single 3D channel  

    Parameters: 
    -----------
    - pooled_layer : The batch of input image. 
    
    Returns : 
    ---------
    - HxW matrix which has for every entry in the matrix pooled_layer that is greater than 0, return 1. Otherwise 0. 
    
    """




    
    dL_wrt_pooling_layer  = (pooled_layers > 0).astype(int) 
    return dL_wrt_pooling_layer


"""

- generate dummy dL_dz 
- generate dummy output 


"""

dL_dz_test = np.random.rand(6, 28, 28)
z = np.random.rand( 6, 28, 28) 
kernel = np.random.rand(6,5,5)

# revised 
def backward_wrt_input(dL_dz, kernel ,stride=1):     
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
    
    print("dL_dz shape ", dL_dz.shape) # the loss should be a
    

    #sys.exit(-1) 
    # Since this is a single layer, let us ignore the dimension C.
    
    # Step 1. filp the filter/kernel 180 degree; assume that a single kernel is 3D i.e. (1, H, W) 
    kernel_rotated = kernel[: , ::-1, ::-1 ] # replace this with
    
    # Step 2. pad the dL_dz
    dL_dz_padded_1 = padding(dL_dz_test, 1) # padding dL_dZ with 1) 

    """
    
    dL_dz_test_padded_1 shape : (1, 30, 30) 
    kernel_rotated : (1, 5, 5) 
    
    convolutional arithmetic : ( ((30-5) + 2* padding) / stride) ) + 1  
        
    """

    image_height = dL_dz_padded_1.shape[-1]
    kernel_height = kernel_rotated.shape[-1]
    stride_ = stride   


    #print("---------") 
    
    #print("dz_dz after padding, length : ", image_height) 
    #print("kernel height, length : ", kernel_height) 
    #print("stride, length : ", stride_) 
    
    output_dimension = (image_height - kernel_height // stride) + 1

    #print("output dimension : ", output_dimension)

    #print("---------") 
    #sys.exit(-1) 
    
    #print("dL_dz before padding : " , dL_dz_test.shape ) 
    #print("dL_dz_padded_1 , after initialization : ", dL_dz_padded_1.shape ) 

    #print("Update : Padding applied successfully" ) 
    output = np.zeros( (output_dimension, output_dimension) ) 
    print("output shape ", output.shape) 
    
    #sys.exit(-2) 
    
    # Step 3: 
    # perform convolution between dL_dz_padded_1 and kernel_rotated- the results will be in output matrix 
    
    print("--------------") 
    # Step 3: Perform the convolution operation --->> This should be a separate function., 
    
    # add another loop to iterate across all the channels 
    # (Batch Size, Channel, Height, Width) --> in my code (Channel (C), Height (i), Width (j)) 
    
    for i in range(0, output_dimension):
        
        for j in range(0, output_dimension):
            
            # Extract the patch of the same size as the kernel
            patch = dL_dz_padded_1[0, i:i+kernel_height, j:j+kernel_height] 

            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(patch * kernel_rotated[0])

            print("i, j: (", i, j , ")")


     
    
    #print("dL_dx: After convolution is applied")
    #print(output)
    print("--------------") 


    """
    output is dL/dX 
    """
    #for i in range(dl_dz_padded_1.shape[0]): #6, 32, 32 #x = 32, 32     
    #    convolve(dl_dz_padded[i], kernel)
                                 
        #    output[] = convolute(patch, kernel)

    #print("dL_dx : after convolution is applied", ) 

    #print("output " , output.shape) 

    return output # this is dL/dx
         

#backward_wrt_input(dL_dz_test, kernel, stride=1)    



