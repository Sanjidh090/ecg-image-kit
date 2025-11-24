"""
ecg_gridest_spectral - Estimates grid size in ECG images using spectral approach.

This function analyzes an ECG image to estimate the grid size in both
horizontal and vertical directions using a spectral approach. The image
is segmentized into smaller regular or random patches, the 2D spectrum of
the patches are estimated and averaged. The local peaks of the average
spectra are used to estimate the potential grid resolutions (both
horizontally and vertically), and returned as vectors.

Note: This function only detects regular grids. The returned values should
  be evaluated based on the image DPI and ECG image style to map the grid
  resolutions to physical time and amplitude units.

Syntax:
    grid_sizes_hor, grid_sizes_ver = ecg_gridest_spectral(img, params=None)

Inputs:
    img - A 2D matrix representing the ECG image in grayscale or RGB formats.
    params - (optional) A dict containing various parameters to control
             the image processing and grid detection algorithm.

Outputs:
    grid_sizes_hor - A vector of estimated grid sizes in the horizontal
        direction (in pixels), sorted in order of prominence
    grid_sizes_ver - A vector of estimated grid sizes in the vertical
        direction (in pixels), sorted in order of prominence

Reference:
    Reza Sameni, 2023-2024, ECG-Image-Kit: A toolkit for ECG image analysis.
    Available at: https://github.com/alphanumericslab/ecg-image-kit

Revision History:
    2023: First release
    2024: Converted to Python
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
try:
    from .tanh_sat import tanh_sat
except ImportError:
    from tanh_sat import tanh_sat


def ecg_gridest_spectral(img, params=None):
    """
    ECG grid size estimation - spectral approach
    
    Parameters
    ----------
    img : ndarray
        A 2D or 3D matrix representing the ECG image
    params : dict, optional
        Dictionary containing algorithm parameters
    
    Returns
    -------
    grid_sizes_hor : ndarray
        Estimated grid sizes in the horizontal direction (in pixels)
    grid_sizes_ver : ndarray
        Estimated grid sizes in the vertical direction (in pixels)
    """
    # Parse algorithm parameters
    if params is None:
        params = {}
    
    params.setdefault('blur_sigma_in_inch', 1.0)
    params.setdefault('paper_size_in_inch', [11, 8.5])
    params.setdefault('remove_shadows', True)
    params.setdefault('apply_edge_detection', False)
    params.setdefault('sat_pre_grid_det', True)
    params.setdefault('sat_level_pre_grid_det', 0.7)
    params.setdefault('num_seg_hor', 5)
    params.setdefault('num_seg_ver', 5)
    params.setdefault('spectral_tiling_method', 'RANDOM_TILING')
    params.setdefault('total_segments', 100)
    params.setdefault('min_grid_resolution', 1)
    params.setdefault('min_grid_peak_prominence', 1.0)
    params.setdefault('detailed_plots', 0)
    params.setdefault('smooth_spectra', True)
    params.setdefault('gauss_win_sigma', 0.3)
    params.setdefault('patch_avg_method', 'MEDIAN')
    params.setdefault('seg_width_rand_dev', 0.1)
    params.setdefault('seg_height_rand_dev', 0.1)
    
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
    
    # Segmentize and estimate spectra
    seg_width = width // params['num_seg_hor']
    seg_height = height // params['num_seg_ver']
    
    if params['spectral_tiling_method'] == 'REGULAR_TILING':
        total_segments = params['num_seg_hor'] * params['num_seg_ver']
        if params['smooth_spectra']:
            y, x = np.ogrid[:seg_height, :seg_width]
            cy, cx = seg_height / 2, seg_width / 2
            sigma = params['gauss_win_sigma'] * np.mean([seg_width, seg_height])
            mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        else:
            mask = np.ones((seg_height, seg_width))
        
        spectra_stacked = np.zeros((seg_height, seg_width, total_segments))
        k = 0
        for i in range(params['num_seg_ver']):
            for j in range(params['num_seg_hor']):
                patch = img_gray_normalized[
                    i * seg_height:(i + 1) * seg_height,
                    j * seg_width:(j + 1) * seg_width
                ]
                spectra_stacked[:, :, k] = np.abs(np.fft.fft2(mask * patch))**2 / (seg_width * seg_height)
                k += 1
    
    elif params['spectral_tiling_method'] == 'RANDOM_TILING':
        total_segments = params['total_segments']
        if params['smooth_spectra']:
            y, x = np.ogrid[:seg_height, :seg_width]
            cy, cx = seg_height / 2, seg_width / 2
            sigma = params['gauss_win_sigma'] * np.mean([seg_width, seg_height])
            mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        else:
            mask = np.ones((seg_height, seg_width))
        
        spectra_stacked = np.zeros((seg_height, seg_width, total_segments))
        for k in range(total_segments):
            start_hor = np.random.randint(0, width - seg_width)
            start_ver = np.random.randint(0, height - seg_height)
            patch = img_gray_normalized[
                start_ver:start_ver + seg_height,
                start_hor:start_hor + seg_width
            ]
            spectra_stacked[:, :, k] = np.abs(np.fft.fft2(mask * patch))**2 / (seg_width * seg_height)
    
    else:  # RANDOM_VAR_SIZE_TILING
        total_segments = params['total_segments']
        spectra_stacked = np.zeros((seg_height, seg_width, total_segments))
        for k in range(total_segments):
            seg_width_randomized = min(width - 1, 
                seg_width + np.random.randint(0, int(params['seg_width_rand_dev'] * seg_width) + 1))
            seg_height_randomized = min(height - 1,
                seg_height + np.random.randint(0, int(params['seg_height_rand_dev'] * seg_height) + 1))
            
            if params['smooth_spectra']:
                y, x = np.ogrid[:seg_height_randomized, :seg_width_randomized]
                cy, cx = seg_height_randomized / 2, seg_width_randomized / 2
                sigma = params['gauss_win_sigma'] * np.mean([seg_width_randomized, seg_height_randomized])
                mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            else:
                mask = np.ones((seg_height_randomized, seg_width_randomized))
            
            start_hor = np.random.randint(0, width - seg_width_randomized)
            start_ver = np.random.randint(0, height - seg_height_randomized)
            patch = img_gray_normalized[
                start_ver:start_ver + seg_height_randomized,
                start_hor:start_hor + seg_width_randomized
            ]
            fft_result = np.fft.fft2(mask * patch, s=(seg_height, seg_width))
            spectra_stacked[:, :, k] = np.abs(fft_result)**2 / (seg_height * seg_width)
    
    # Average spectra across patches
    if params['patch_avg_method'] == 'MEDIAN':
        spectral_avg = np.median(spectra_stacked, axis=2)
    else:  # MEAN
        spectral_avg = np.mean(spectra_stacked, axis=2)
    
    spectral_avg_hor = 10 * np.log10(np.mean(spectral_avg, axis=1))
    spectral_avg_ver = 10 * np.log10(np.mean(spectral_avg, axis=0))
    
    # Estimate grid resolution - find local spectral peaks
    peaks_hor, properties_hor = find_peaks(spectral_avg_hor,
                                           distance=params['min_grid_resolution'],
                                           prominence=params['min_grid_peak_prominence'])
    peaks_ver, properties_ver = find_peaks(spectral_avg_ver,
                                           distance=params['min_grid_resolution'],
                                           prominence=params['min_grid_peak_prominence'])
    
    # Limit range to Nyquist frequency
    ff_hor = np.arange(len(spectral_avg_hor)) / len(spectral_avg_hor)
    ff_ver = np.arange(len(spectral_avg_ver)) / len(spectral_avg_ver)
    
    # Keep only peaks below Nyquist frequency
    nyq_mask_hor = ff_hor[peaks_hor] < 0.5
    peaks_hor = peaks_hor[nyq_mask_hor]
    pk_prominence_hor = properties_hor['prominences'][nyq_mask_hor] if 'prominences' in properties_hor else np.ones(len(peaks_hor))
    
    nyq_mask_ver = ff_ver[peaks_ver] < 0.5
    peaks_ver = peaks_ver[nyq_mask_ver]
    pk_prominence_ver = properties_ver['prominences'][nyq_mask_ver] if 'prominences' in properties_ver else np.ones(len(peaks_ver))
    
    # Sort spectral peaks in order of prominence
    if len(peaks_hor) > 0:
        I_hor_sorted = np.argsort(pk_prominence_hor)[::-1]
        grid_sizes_hor = 1.0 / ff_hor[peaks_hor[I_hor_sorted]]
    else:
        grid_sizes_hor = np.array([])
    
    if len(peaks_ver) > 0:
        I_ver_sorted = np.argsort(pk_prominence_ver)[::-1]
        grid_sizes_ver = 1.0 / ff_ver[peaks_ver[I_ver_sorted]]
    else:
        grid_sizes_ver = np.array([])
    
    return grid_sizes_hor, grid_sizes_ver
