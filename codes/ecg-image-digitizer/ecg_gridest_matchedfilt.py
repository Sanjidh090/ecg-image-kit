"""
ecg_gridest_matchedfilt - Estimates grid size in ECG images using matched filtering.

This function analyzes an ECG image to estimate the grid size using a
matched filter-based approach. The image is segmented into smaller
regular or random patches, which are processed with an edge-only
square-shaped matched filter of variable size. The matched filter output
powers are calculated and averaged across all patches. The matched filter
sizes yielding the maximum average power are returned as potential grid
sizes.

Note: This function only detects regular grids without rotation. The
returned values should be evaluated based on the image DPI and ECG image
style to map the grid resolutions to physical time and amplitude units.

Syntax:
    grid_sizes, grid_size_prominences, mask_size, matched_filter_powers_avg, I_peaks = ecg_gridest_matchedfilt(img, params=None)

Inputs:
    img - A 2D matrix representing the ECG image in grayscale or RGB formats.
    params - (optional) A dict containing various parameters to control
             the image processing and grid detection algorithm.

Outputs:
    grid_sizes - Vector of estimated grid sizes.
    grid_size_prominences - Vector of prominences for each grid size.
    mask_size - Array of all studied grid sizes.
    matched_filter_powers_avg - Average matched filter output powers for each mask size.
    I_peaks - The selected local peaks of matched_filter_powers_avg

Reference:
    Reza Sameni, 2023-2024, ECG-Image-Kit: A toolkit for ECG image analysis.
    Available at: https://github.com/alphanumericslab/ecg-image-kit

Revision History:
    2023: First release
    2024: Converted to Python
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import find_peaks
try:
    from .tanh_sat import tanh_sat
except ImportError:
    from tanh_sat import tanh_sat


def boundary_mask(sz):
    """Create a boundary mask for matched filtering"""
    B = np.zeros((sz, sz))
    B[0, :] = 1
    B[-1, :] = 1
    B[:, 0] = 1
    B[:, -1] = 1
    B = B / np.sum(B)
    return B


def ecg_gridest_matchedfilt(img, params=None):
    """
    ECG grid size estimation - matched filter-based approach
    
    Parameters
    ----------
    img : ndarray
        A 2D or 3D matrix representing the ECG image
    params : dict, optional
        Dictionary containing algorithm parameters
    
    Returns
    -------
    grid_sizes : ndarray
        Vector of estimated grid sizes
    grid_size_prominences : ndarray
        Vector of prominences for each grid size
    mask_size : ndarray
        Array of all studied grid sizes
    matched_filter_powers_avg : ndarray
        Average matched filter output powers for each mask size
    I_peaks : ndarray
        The selected local peaks of matched_filter_powers_avg
    """
    # Parse algorithm parameters
    if params is None:
        params = {}
    
    params.setdefault('blur_sigma_in_inch', 1.0)
    params.setdefault('paper_size_in_inch', [11, 8.5])
    params.setdefault('remove_shadows', True)
    params.setdefault('sat_pre_grid_det', True)
    params.setdefault('sat_level_pre_grid_det', 0.7)
    params.setdefault('num_seg_hor', 4)
    params.setdefault('num_seg_ver', 4)
    params.setdefault('tiling_method', 'RANDOM_TILING')
    params.setdefault('total_segments', 16)
    params.setdefault('max_grid_size', 30)
    params.setdefault('min_grid_size', 2)
    params.setdefault('power_avg_prctile_th', 95.0)
    params.setdefault('detailed_plots', 0)
    
    height, width = img.shape[:2]
    
    # Convert image to gray scale
    if img.ndim == 3:
        img_gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        img_gray = img_gray / np.max(img_gray)
    else:
        img_gray = img.astype(float)
        img_gray = 1.0 - (img_gray / np.max(img_gray))
    
    # Shadow removal and intensity normalization
    if params['remove_shadows']:
        blurring_sigma = np.mean([
            width * params['blur_sigma_in_inch'] / params['paper_size_in_inch'][0],
            height * params['blur_sigma_in_inch'] / params['paper_size_in_inch'][1]
        ])
        img_gray_blurred = gaussian_filter(img_gray, sigma=blurring_sigma, mode='reflect')
        
        img_gray_normalized = img_gray / img_gray_blurred
        img_gray_normalized = (img_gray_normalized - np.min(img_gray_normalized)) / \
                             (np.max(img_gray_normalized) - np.min(img_gray_normalized))
    else:
        img_gray_normalized = img_gray
    
    # Image density saturation
    if params['sat_pre_grid_det']:
        img_sat = tanh_sat(1.0 - img_gray_normalized.flatten(),
                          params['sat_level_pre_grid_det'], 'ksigma')
        img_gray_normalized = img_sat.reshape(img_gray_normalized.shape)
    
    # Segmentation
    seg_width = width // params['num_seg_hor']
    seg_height = height // params['num_seg_ver']
    mask_size = np.arange(params['min_grid_size'], params['max_grid_size'] + 1)
    
    if params['tiling_method'] == 'REGULAR_TILING':
        total_segments = params['num_seg_hor'] * params['num_seg_ver']
        matched_filter_powers = np.zeros((total_segments, len(mask_size)))
        k = 0
        for i in range(params['num_seg_ver']):
            for j in range(params['num_seg_hor']):
                segment = img_gray_normalized[
                    i * seg_height:(i + 1) * seg_height,
                    j * seg_width:(j + 1) * seg_width
                ]
                segment = (segment - np.mean(segment)) / np.std(segment)
                
                for g in range(len(mask_size)):
                    B = boundary_mask(mask_size[g])
                    B = B - np.mean(B)
                    matched_filtered = convolve(segment, B, mode='reflect')
                    pm = matched_filtered.flatten()**2
                    pm_th = np.percentile(pm, params['power_avg_prctile_th'])
                    matched_filter_powers[k, g] = 10 * np.log10(np.mean(pm[pm > pm_th]))
                k += 1
    
    else:  # RANDOM_TILING
        total_segments = params['total_segments']
        matched_filter_powers = np.zeros((total_segments, len(mask_size)))
        for k in range(total_segments):
            start_hor = np.random.randint(0, width - seg_width)
            start_ver = np.random.randint(0, height - seg_height)
            segment = img_gray_normalized[
                start_ver:start_ver + seg_height,
                start_hor:start_hor + seg_width
            ]
            segment = (segment - np.mean(segment)) / np.std(segment)
            
            for g in range(len(mask_size)):
                B = boundary_mask(mask_size[g])
                B = B - np.mean(B)
                matched_filtered = convolve(segment, B, mode='reflect')
                pm = matched_filtered.flatten()**2
                pm_th = np.percentile(pm, params['power_avg_prctile_th'])
                matched_filter_powers[k, g] = 10 * np.log10(np.mean(pm[pm > pm_th]))
    
    matched_filter_powers_avg = np.mean(matched_filter_powers, axis=0)
    I_peaks, properties = find_peaks(matched_filter_powers_avg)
    grid_size_prominences = properties['prominences'] if 'prominences' in properties else np.ones(len(I_peaks))
    grid_sizes = mask_size[I_peaks] - 1  # -1 is to convert mask size to period
    
    return grid_sizes, grid_size_prominences, mask_size, matched_filter_powers_avg, I_peaks
