"""
ECG Image Digitizer
===================

Software tools for ECG image processing and digitization.

This package provides Python implementations of algorithms for:
- ECG grid size estimation from scanned ECG images
- Time-series extraction from ECG image segments
- Signal processing utilities for ECG digitization

Functions
---------

Grid Estimation:
    - ecg_grid_size_from_paper: Estimate grid size from paper dimensions
    - ecg_gridest_margdist: Grid estimation using marginal distribution approach
    - ecg_gridest_spectral: Grid estimation using spectral approach
    - ecg_gridest_matchedfilt: Grid estimation using matched filter approach

Image Processing:
    - image_to_sequence: Extract time-series from ECG image segments
    - tanh_sat: Saturate signal/image intensity with tanh function

Test Scripts:
    - test_ecg_grid_size_estimator: Compare grid estimation methods
    - test_ecg_sequence_extraction: Test sequence extraction methods

Installation
------------

Install the required dependencies:

    pip install -r requirements.txt

Usage
-----

Example: Estimate grid size from an ECG image

    import numpy as np
    from PIL import Image
    from ecg_grid_size_from_paper import ecg_grid_size_from_paper
    from ecg_gridest_spectral import ecg_gridest_spectral
    
    # Load image
    img = np.array(Image.open('ecg_image.jpg'))
    
    # Estimate based on paper size
    coarse_grid, fine_grid = ecg_grid_size_from_paper(img, 11.0, 'in')
    
    # Estimate using spectral method
    grid_hor, grid_ver = ecg_gridest_spectral(img)
    
    print(f"Fine grid size (paper-based): {fine_grid:.2f} pixels")
    print(f"Grid size (spectral): horizontal={grid_hor[0]:.2f}, vertical={grid_ver[0]:.2f} pixels")

Example: Extract time-series from ECG segment

    from image_to_sequence import image_to_sequence
    
    # Extract sequence using different methods
    data = image_to_sequence(img, 'dark-foreground', 'moving_average', 
                            windowlen=5, plot_result=True)

References
----------

Reza Sameni, 2023-2024, ECG-Image-Kit: A toolkit for ECG image analysis.
Available at: https://github.com/alphanumericslab/ecg-image-kit

Citation:
    Kshama Kodthalu Shivashankara, Deepanshi, Afagh Mehri Shervedani, 
    Matthew A. Reyna, Gari D. Clifford, Reza Sameni (2024). 
    ECG-image-kit: a synthetic image generation toolbox to facilitate 
    deep learning-based electrocardiogram digitization. 
    In Physiological Measurement. IOP Publishing. 
    doi: 10.1088/1361-6579/ad4954
"""

__version__ = '1.0.0'
__author__ = 'Reza Sameni and contributors'

from .ecg_grid_size_from_paper import ecg_grid_size_from_paper
from .ecg_gridest_margdist import ecg_gridest_margdist
from .ecg_gridest_spectral import ecg_gridest_spectral
from .ecg_gridest_matchedfilt import ecg_gridest_matchedfilt
from .image_to_sequence import image_to_sequence
from .tanh_sat import tanh_sat

__all__ = [
    'ecg_grid_size_from_paper',
    'ecg_gridest_margdist',
    'ecg_gridest_spectral',
    'ecg_gridest_matchedfilt',
    'image_to_sequence',
    'tanh_sat',
]
