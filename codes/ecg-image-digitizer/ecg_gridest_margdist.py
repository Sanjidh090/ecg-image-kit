"""
ecg_gridest_margdist - Estimates grid size in ECG images.

This function analyzes an ECG image to estimate the grid size in both
horizontal and vertical directions using the average marginal pixel 
densities of regular or random patches of the ECG. Potential horizontal
and vertical grid sizes are returned for further evaluation  

Note: This function only detects regular grids. The returned values should
  be evaluated based on the image DPI and ECG image style to map the grid
  resolutions to physical time and amplitude units.

Syntax:
    grid_size_hor, grid_size_ver, peak_gaps_hor, peak_gaps_ver, peak_amps_hor, peak_amps_ver = ecg_gridest_margdist(img, params=None)

Inputs:
    img - A 2D matrix representing the ECG image in grayscale or RGB formats.
    params - (optional) A dict containing various parameters to control
             the image processing and grid detection algorithm. Default
             values are used if this argument is not provided or is partially
             provided.

Outputs:
    grid_size_hor - Estimated grid size in the horizontal direction (in pixels).
    grid_size_ver - Estimated grid size in the vertical direction (in pixels).
    peak_gaps_hor - Grid spacing for all segments in the horizontal direction (in pixels).
    peak_gaps_ver - Grid spacing for all segments in the vertical direction (in pixels).
    peak_amps_hor - Peak amplitudes in the horizontal direction
    peak_amps_ver - Peak amplitudes in the vertical direction

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
from sklearn.cluster import KMeans
try:
    from .tanh_sat import tanh_sat
except ImportError:
    from tanh_sat import tanh_sat


def ecg_gridest_margdist(img, params=None):
    """
    ECG grid size estimation - marginal image intensity approach
    
    Parameters
    ----------
    img : ndarray
        A 2D or 3D matrix representing the ECG image
    params : dict, optional
        Dictionary containing algorithm parameters
    
    Returns
    -------
    grid_size_hor : float
        Estimated grid size in the horizontal direction (in pixels)
    grid_size_ver : float
        Estimated grid size in the vertical direction (in pixels)
    peak_gaps_hor : ndarray
        Grid spacing for all segments in the horizontal direction
    peak_gaps_ver : ndarray
        Grid spacing for all segments in the vertical direction
    peak_amps_hor : ndarray
        Peak amplitudes in the horizontal direction
    peak_amps_ver : ndarray
        Peak amplitudes in the vertical direction
    """
    # Parse algorithm parameters
    if params is None:
        params = {}
    
    params.setdefault('blur_sigma_in_inch', 1.0)
    params.setdefault('paper_size_in_inch', [11, 8.5])
    params.setdefault('remove_shadows', True)
    params.setdefault('apply_edge_detection', False)
    params.setdefault('cluster_peaks', True)
    params.setdefault('avg_quartile', 50.0)
    params.setdefault('num_seg_hor', 4)
    params.setdefault('num_seg_ver', 4)
    params.setdefault('hist_grid_det_method', 'RANDOM_TILING')
    params.setdefault('total_segments', 100)
    params.setdefault('min_grid_resolution', 1)
    params.setdefault('min_grid_peak_prom_prctile', 2)
    params.setdefault('sat_pre_grid_det', True)
    params.setdefault('sat_level_pre_grid_det', 0.7)
    params.setdefault('detailed_plots', 0)
    
    if params['cluster_peaks']:
        params.setdefault('max_clusters', 3)
        params.setdefault('cluster_selection_method', 'GAP_MIN_VAR')
    
    if params['avg_quartile'] > 100.0:
        raise ValueError('avg_quartile parameter must be between 0 and 100.0')
    
    height, width = img.shape[:2]
    
    # Convert image to gray scale if in RGB
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
    
    if params['hist_grid_det_method'] == 'REGULAR_TILING':
        total_segments = params['num_seg_hor'] * params['num_seg_ver']
        segments_stacked = np.zeros((seg_height, seg_width, total_segments))
        k = 0
        for i in range(params['num_seg_ver']):
            for j in range(params['num_seg_hor']):
                segments_stacked[:, :, k] = img_gray_normalized[
                    i * seg_height:(i + 1) * seg_height,
                    j * seg_width:(j + 1) * seg_width
                ]
                k += 1
    else:  # RANDOM_TILING
        total_segments = params['total_segments']
        segments_stacked = np.zeros((seg_height, seg_width, total_segments))
        for k in range(total_segments):
            start_hor = np.random.randint(0, width - seg_width)
            start_ver = np.random.randint(0, height - seg_height)
            segments_stacked[:, :, k] = img_gray_normalized[
                start_ver:start_ver + seg_height,
                start_hor:start_hor + seg_width
            ]
    
    # Horizontal/vertical histogram estimation per patch
    peak_amps_hor = []
    peak_gaps_hor = []
    peak_amps_ver = []
    peak_gaps_ver = []
    
    for k in range(total_segments):
        # Horizontal histogram
        hist_hor = 1.0 - np.mean(segments_stacked[:, :, k], axis=1)
        min_grid_peak_prominence = np.percentile(hist_hor, params['min_grid_peak_prom_prctile']) - np.min(hist_hor)
        peaks_hor, properties_hor = find_peaks(hist_hor, 
                                               distance=params['min_grid_resolution'],
                                               prominence=min_grid_peak_prominence)
        if len(peaks_hor) > 1:
            peak_amps_hor.extend(hist_hor[peaks_hor[1:]])
            peak_gaps_hor.extend(np.diff(peaks_hor))
        
        # Vertical histogram
        hist_ver = 1.0 - np.mean(segments_stacked[:, :, k], axis=0)
        min_grid_peak_prominence = np.percentile(hist_ver, params['min_grid_peak_prom_prctile']) - np.min(hist_ver)
        peaks_ver, properties_ver = find_peaks(hist_ver,
                                               distance=params['min_grid_resolution'],
                                               prominence=min_grid_peak_prominence)
        if len(peaks_ver) > 1:
            peak_amps_ver.extend(hist_ver[peaks_ver[1:]])
            peak_gaps_ver.extend(np.diff(peaks_ver))
    
    peak_amps_hor = np.array(peak_amps_hor)
    peak_gaps_hor = np.array(peak_gaps_hor)
    peak_amps_ver = np.array(peak_amps_ver)
    peak_gaps_ver = np.array(peak_gaps_ver)
    
    # Calculate horizontal/vertical grid sizes
    if not params['cluster_peaks']:
        # Direct method
        peak_gaps_prctiles = np.percentile(peak_gaps_hor, 
            [50.0 - params['avg_quartile']/2, 50.0 + params['avg_quartile']/2])
        mask = (peak_gaps_hor >= peak_gaps_prctiles[0]) & (peak_gaps_hor <= peak_gaps_prctiles[1])
        grid_size_hor = np.mean(peak_gaps_hor[mask])
        
        peak_gaps_prctiles = np.percentile(peak_gaps_ver,
            [50.0 - params['avg_quartile']/2, 50.0 + params['avg_quartile']/2])
        mask = (peak_gaps_ver >= peak_gaps_prctiles[0]) & (peak_gaps_ver <= peak_gaps_prctiles[1])
        grid_size_ver = np.mean(peak_gaps_ver[mask])
    else:
        # Indirect method (cluster the local peaks)
        # Horizontal clustering
        if len(peak_amps_hor) > params['max_clusters']:
            best_k = min(params['max_clusters'], len(peak_amps_hor))
            kmeans_hor = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            IDX_hor = kmeans_hor.fit_predict(peak_amps_hor.reshape(-1, 1))
            
            if params['cluster_selection_method'] == 'GAP_MIN_VAR':
                peak_gaps_per_cluster = [np.std(peak_gaps_hor[IDX_hor == cc]) 
                                        for cc in range(best_k)]
                selected_cluster_hor = np.argmin(peak_gaps_per_cluster)
            else:  # MAX_AMP_PEAKS
                peak_amps_per_cluster = [np.median(peak_amps_hor[IDX_hor == cc])
                                        for cc in range(best_k)]
                selected_cluster_hor = np.argmax(peak_amps_per_cluster)
            
            peak_gaps_selected = peak_gaps_hor[IDX_hor == selected_cluster_hor]
            peak_gaps_prctiles = np.percentile(peak_gaps_selected,
                [50.0 - params['avg_quartile']/2, 50.0 + params['avg_quartile']/2])
            mask = (peak_gaps_selected >= peak_gaps_prctiles[0]) & \
                   (peak_gaps_selected <= peak_gaps_prctiles[1])
            grid_size_hor = np.mean(peak_gaps_selected[mask])
        else:
            grid_size_hor = np.mean(peak_gaps_hor) if len(peak_gaps_hor) > 0 else 0
        
        # Vertical clustering
        if len(peak_amps_ver) > params['max_clusters']:
            best_k = min(params['max_clusters'], len(peak_amps_ver))
            kmeans_ver = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            IDX_ver = kmeans_ver.fit_predict(peak_amps_ver.reshape(-1, 1))
            
            if params['cluster_selection_method'] == 'GAP_MIN_VAR':
                peak_gaps_per_cluster = [np.std(peak_gaps_ver[IDX_ver == cc])
                                        for cc in range(best_k)]
                selected_cluster_ver = np.argmin(peak_gaps_per_cluster)
            else:  # MAX_AMP_PEAKS
                peak_amps_per_cluster = [np.median(peak_amps_ver[IDX_ver == cc])
                                        for cc in range(best_k)]
                selected_cluster_ver = np.argmax(peak_amps_per_cluster)
            
            peak_gaps_selected = peak_gaps_ver[IDX_ver == selected_cluster_ver]
            peak_gaps_prctiles = np.percentile(peak_gaps_selected,
                [50.0 - params['avg_quartile']/2, 50.0 + params['avg_quartile']/2])
            mask = (peak_gaps_selected >= peak_gaps_prctiles[0]) & \
                   (peak_gaps_selected <= peak_gaps_prctiles[1])
            grid_size_ver = np.mean(peak_gaps_selected[mask])
        else:
            grid_size_ver = np.mean(peak_gaps_ver) if len(peak_gaps_ver) > 0 else 0
    
    return grid_size_hor, grid_size_ver, peak_gaps_hor, peak_gaps_ver, peak_amps_hor, peak_amps_ver
