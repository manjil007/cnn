"""
This file contains utility functions used in cnn operation. 
    - padding
"""

import numpy as np 

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