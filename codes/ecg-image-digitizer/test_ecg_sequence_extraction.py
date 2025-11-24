"""
Test script for image_to_sequence to convert ECG images into time-series

This script reads ECG images and applies different methods provided by
the image_to_sequence function to convert these images into time-series
data. It processes each image using various sequence extraction methods
and then visualizes the results for comparison.

Note: This method works on single-channel ECG or segments of ECG. For
multichannel data, the leads should be separated before applying this
function. To note, the function is only provided as proof-of-concept. A
full ECG digitization pipeline requires additional elements.

Reference:
    Reza Sameni, 2023-2024, ECG-Image-Kit: A toolkit for ECG image analysis.
    Available at: https://github.com/alphanumericslab/ecg-image-kit

Revision History:
    2023: First release
    2024: Converted to Python
"""

import os
import numpy as np
from PIL import Image
import glob

from image_to_sequence import image_to_sequence

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def test_ecg_sequence_extraction():
    """Test script for ECG sequence extraction from images"""
    
    # Define the path to the folder containing ECG image segments
    data_path = '../../sample-data/ecg-images/sample-segments/'
    
    # Get a list of all image files in the folder
    image_files = glob.glob(os.path.join(data_path, '*.*'))
    
    # Filter for common image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [f for f in image_files if f.lower().endswith(valid_extensions)]
    
    # Loop over all files, reading and processing each image
    for image_fname in image_files:
        if os.path.isfile(image_fname):
            try:
                print(f"\nProcessing: {os.path.basename(image_fname)}")
                
                # Read the image
                img = np.array(Image.open(image_fname))
                
                # Apply different sequence extraction methods to the image
                z0 = image_to_sequence(img, 'dark-foreground', 'max_finder', windowlen=3, plot_result=False)
                z1 = image_to_sequence(img, 'dark-foreground', 'hor_smoothing', windowlen=3, plot_result=False)
                z2 = image_to_sequence(img, 'dark-foreground', 'all_left_right_neighbors', windowlen=3, plot_result=False)
                z3 = image_to_sequence(img, 'dark-foreground', 'combined_all_neighbors', windowlen=3, plot_result=False)
                z4 = image_to_sequence(img, 'dark-foreground', 'moving_average', windowlen=3, plot_result=False)
                
                # Combine results from all methods. The median operator is used
                # to perform a type of voting between multiple algorithms.
                z_combined = np.median(np.vstack([z0, z1, z2, z3, z4]), axis=0)
                
                if HAS_MATPLOTLIB:
                    # Prepare for plotting
                    lgnd = ['max_finder', 'hor_smoothing', 'all_left_right_neighbors',
                           'combined_all_neighbors', 'moving_average', 'combined methods']
                    nn = np.arange(img.shape[1])
                    img_height = img.shape[0]
                    
                    # Display the original image and overlay the extracted sequences
                    plt.figure(figsize=(12, 8))
                    if img.ndim == 3:
                        plt.imshow(img)
                    else:
                        plt.imshow(img, cmap='gray')
                    
                    plt.plot(nn, img_height - z0, linewidth=2, label=lgnd[0])
                    plt.plot(nn, img_height - z1, linewidth=2, label=lgnd[1])
                    plt.plot(nn, img_height - z2, linewidth=2, label=lgnd[2])
                    plt.plot(nn, img_height - z3, linewidth=2, label=lgnd[3])
                    plt.plot(nn, img_height - z4, linewidth=2, label=lgnd[4])
                    plt.plot(nn, img_height - z_combined, linewidth=3, label=lgnd[5])
                    
                    # Add legend and title
                    plt.legend(loc='best')
                    plt.title(f'Paper ECG vs recovered signal for: {os.path.basename(image_fname)}')
                    plt.tight_layout()
                    
                    # Uncomment to save the figure
                    # plt.savefig(image_fname.replace('.', '-rec.'))
                    
                    plt.show()
                
                print(f"Successfully extracted sequences from {os.path.basename(image_fname)}")
                
            except (IOError, ValueError, RuntimeError) as e:
                print(f'Error processing file {os.path.basename(image_fname)}: {str(e)}')


if __name__ == '__main__':
    test_ecg_sequence_extraction()
