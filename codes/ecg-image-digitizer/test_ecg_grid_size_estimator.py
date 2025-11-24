"""
A test script for multiple ECG grid size estimation algorithms

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

from ecg_grid_size_from_paper import ecg_grid_size_from_paper
from ecg_gridest_margdist import ecg_gridest_margdist
from ecg_gridest_spectral import ecg_gridest_spectral
from ecg_gridest_matchedfilt import ecg_gridest_matchedfilt


def test_ecg_grid_size_estimator():
    """Test script for ECG grid size estimation algorithms"""
    
    data_path = '../../sample-data/ecg-images/'
    
    # Get a list of all image files in the folder
    image_files = glob.glob(os.path.join(data_path, '*.*'))
    
    # Filter for common image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [f for f in image_files if f.lower().endswith(valid_extensions)]
    
    for image_fname in image_files:
        try:
            print(f"\nProcessing: {os.path.basename(image_fname)}")
            
            # Read the image
            img = np.array(Image.open(image_fname))
            
            # Estimate grid resolution based on paper size
            paper_size = [11.0, 8.5]
            coarse_grid_size_paper_based, fine_grid_size_paper_based = \
                ecg_grid_size_from_paper(img, paper_size[0], 'in')
            
            # Marginal distribution-based method
            params_margdist = {
                'blur_sigma_in_inch': 1.0,
                'paper_size_in_inch': paper_size,
                'remove_shadows': True,
                'apply_edge_detection': False,
                'sat_pre_grid_det': False,
                'sat_level_pre_grid_det': 0.7,
                'num_seg_hor': 4,
                'num_seg_ver': 4,
                'hist_grid_det_method': 'RANDOM_TILING',
                'total_segments': 100,
                'min_grid_resolution': 1,
                'min_grid_peak_prom_prctile': 2.0,
                'cluster_peaks': True,
                'max_clusters': 3,
                'cluster_selection_method': 'GAP_MIN_VAR',
                'avg_quartile': 50.0,
                'detailed_plots': 0
            }
            
            gridsize_hor_margdist, gridsize_ver_margdist, grid_spacings_hor, grid_spacing_ver, _, _ = \
                ecg_gridest_margdist(img, params_margdist)
            
            # Spectral-based method
            params_spectral = {
                'blur_sigma_in_inch': 1.0,
                'paper_size_in_inch': paper_size,
                'remove_shadows': True,
                'apply_edge_detection': False,
                'sat_pre_grid_det': False,
                'sat_level_pre_grid_det': 0.7,
                'num_seg_hor': 4,
                'num_seg_ver': 4,
                'spectral_tiling_method': 'RANDOM_TILING',
                'total_segments': 100,
                'min_grid_resolution': 1,
                'min_grid_peak_prominence': 1.0,
                'detailed_plots': 0
            }
            
            gridsize_hor_spectral, gridsize_ver_spectral = \
                ecg_gridest_spectral(img, params_spectral)
            
            # Find closest spectral estimates to paper-based resolution
            if len(gridsize_hor_spectral) > 0:
                closest_ind_hor = np.argmin(np.abs(gridsize_hor_spectral - fine_grid_size_paper_based))
            else:
                closest_ind_hor = None
            
            if len(gridsize_ver_spectral) > 0:
                closest_ind_ver = np.argmin(np.abs(gridsize_ver_spectral - fine_grid_size_paper_based))
            else:
                closest_ind_ver = None
            
            # Matched filter-based method
            params_matchfilt = params_margdist.copy()
            params_matchfilt['sat_pre_grid_det'] = True
            params_matchfilt['sat_level_pre_grid_det'] = 0.7
            params_matchfilt['total_segments'] = 10
            params_matchfilt['tiling_method'] = 'RANDOM_TILING'
            
            grid_sizes_matchedfilt, grid_size_prom_matchedfilt, mask_size_matchedfilt, \
                matchedfilt_powers_avg, I_peaks_matchedfilt = \
                ecg_gridest_matchedfilt(img, params_matchfilt)
            
            # Display results
            print(f'Grid resolution estimate per 0.1mV x 40ms (paper size-based): {fine_grid_size_paper_based:.2f} pixels')
            print(f'Grid resolution estimates per 0.1mV x 40ms (matched filter-based): {grid_sizes_matchedfilt}')
            print(f'Horizontal grid resolution estimate (margdist): {gridsize_hor_margdist:.2f} pixels')
            print(f'Vertical grid resolution estimate (margdist): {gridsize_ver_margdist:.2f} pixels')
            print(f'Horizontal grid resolution estimate (spectral): {gridsize_hor_spectral}')
            print(f'Vertical grid resolution estimate (spectral): {gridsize_ver_spectral}')
            
            if closest_ind_hor is not None:
                print(f'Closest spectral horizontal grid resolution estimate from paper-based resolution (per 0.1mV x 40ms): '
                      f'{gridsize_hor_spectral[closest_ind_hor]:.2f} pixels')
            if closest_ind_ver is not None:
                print(f'Closest spectral vertical grid resolution estimate from paper-based resolution (per 0.1mV x 40ms): '
                      f'{gridsize_ver_spectral[closest_ind_ver]:.2f} pixels')
            
            print('---')
            
        except Exception as e:
            print(f"Error processing {image_fname}: {str(e)}")


if __name__ == '__main__':
    test_ecg_grid_size_estimator()
