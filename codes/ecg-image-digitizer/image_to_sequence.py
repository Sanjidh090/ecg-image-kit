"""
IMAGE_TO_SEQUENCE Extracts a sequence/time-series from an image.

This function processes an image to extract a time-series representation,
for example an ECG image. The method to extract the sequence depends on
the image's characteristics (e.g., whether the foreground is darker or
brighter than the background) and the filtering approach. The function
returns a vector that has the same length as the width of the input image
(the second dimension of the input image matrix). The method used
for extracting the sequence can be justified using a maximum likelihood
estimate of adjacent temporal samples when studied in a probabilistic
framework.

Syntax:
    data = image_to_sequence(img, mode, method, windowlen=3, plot_result=False)

Inputs:
    img - A 2D or 3D matrix representing the image.
    mode - A string specifying the foreground type: 'dark-foreground' or
           'bright-foreground'.
    method - A string specifying the filtering method to use. Options are
             'max_finder', 'moving_average', 'hor_smoothing', 
             'all_left_right_neighbors', 'combined_all_neighbors'.
    windowlen - (optional) Length of the moving average window. Default is 3.
    plot_result - (optional) Boolean to plot the result. Default is False.

Outputs:
    data - Extracted sequence or time-series from the image.

Example:
    from PIL import Image
    import numpy as np
    
    img = np.array(Image.open('path/to/ecg_image.jpg'))
    data = image_to_sequence(img, 'dark-foreground', 'moving_average', windowlen=5, plot_result=True)

Notes:
    - The function assumes the image is either grayscale or RGB.
    - The 'max_finder' method simply extracts the max value per column.
    - Other methods apply different filters to smoothen or highlight features.

Reference:
    Reza Sameni, 2023-2024, ECG-Image-Kit: A toolkit for ECG image analysis.
    Available at: https://github.com/alphanumericslab/ecg-image-kit

Revision History:
    2022: First release
    2024: Converted to Python
"""

import numpy as np
from scipy.ndimage import convolve


def image_to_sequence(img, mode, method, windowlen=3, plot_result=False):
    """
    Extract time-series from an ECG segment
    
    Parameters
    ----------
    img : ndarray
        A 2D or 3D matrix representing the image
    mode : str
        'dark-foreground' or 'bright-foreground'
    method : str
        Filtering method: 'max_finder', 'moving_average', 'hor_smoothing',
        'all_left_right_neighbors', 'combined_all_neighbors'
    windowlen : int, optional
        Length of the moving average window. Default is 3
    plot_result : bool, optional
        Whether to plot the result. Default is False
    
    Returns
    -------
    data : ndarray
        Extracted sequence or time-series from the image
    """
    # Convert image to grayscale if it's in RGB format
    if img.ndim == 3:
        # RGB to grayscale conversion
        img_gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        img_gray = img.astype(float)
    
    # Process image based on specified foreground mode
    if mode == 'dark-foreground':
        img_flipped = 255 - img_gray if img_gray.max() > 1 else 1.0 - img_gray
    elif mode == 'bright-foreground':
        img_flipped = img_gray
    else:
        raise ValueError("mode must be 'dark-foreground' or 'bright-foreground'")
    
    # Apply different methods for sequence extraction
    if method == 'max_finder':
        img_filtered = img_flipped
    
    elif method == 'moving_average':
        h = np.ones((windowlen, windowlen))
        h = h / np.sum(h)
        img_filtered = convolve(img_flipped, h, mode='reflect')
    
    elif method == 'hor_smoothing':
        h = np.ones((1, windowlen))
        h = h / np.sum(h)
        img_filtered = convolve(img_flipped, h, mode='reflect')
    
    elif method == 'all_left_right_neighbors':
        h = np.array([[1, 0, 1], 
                      [1, 1, 1], 
                      [1, 0, 1]], dtype=float)
        h = h / np.sum(h)
        img_filtered = convolve(img_flipped, h, mode='reflect')
    
    elif method == 'combined_all_neighbors':
        h1 = np.array([[1, 1, 1]], dtype=float)
        h1 = h1 / np.sum(h1)
        z1 = convolve(img_flipped, h1, mode='reflect')
        
        h2 = np.array([[1, 0, 0], 
                       [0, 1, 0], 
                       [0, 0, 1]], dtype=float)
        h2 = h2 / np.sum(h2)
        z2 = convolve(img_flipped, h2, mode='reflect')
        
        h3 = np.array([[0, 0, 1], 
                       [0, 1, 0], 
                       [1, 0, 0]], dtype=float)
        h3 = h3 / np.sum(h3)
        z3 = convolve(img_flipped, h3, mode='reflect')
        
        img_filtered = np.minimum(np.minimum(z1, z2), z3)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Find the maximum pixel value in each column to represent the ECG signal
    I = np.argmax(img_filtered, axis=0)
    img_height = img_filtered.shape[0]
    data = img_height - I  # Convert to vertical position (ECG amplitude with offset)
    
    # Plot the result if requested
    if plot_result:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
            plt.plot(np.arange(img.shape[1]), img.shape[0] - data, 'g', linewidth=3)
            plt.title('Extracted ECG Signal')
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    return data
