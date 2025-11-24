"""
tanh_sat - Saturates outlier samples using a tanh shape function

Usage:
    y = tanh_sat(x, param, mode='ksigma')

Inputs:
    x: Input data, can be a vector or a matrix (channels x time)
    param: Scaling factor for the saturation level:
        - If mode == 'ksigma' or no mode defined: k times the standard deviation of each channel
        - If mode == 'absolute': a vector of absolute thresholds used to
            saturate each channel. If param is a scalar, the same value is
            used for all channels
    mode (optional): 'ksigma' or 'absolute'. Default is 'ksigma'

Output:
    y: Saturated data with outliers replaced by the saturation level

Revision History:
    2020: First release
    2023: Renamed from deprecated version TanhSaturation()
    2024: Converted to Python

References:
    Reza Sameni, 2020-2024
    The Open-Source Electrophysiological Toolbox
    https://github.com/alphanumericslab/OSET
"""

import numpy as np


def tanh_sat(x, param, mode='ksigma'):
    """
    Saturate signal/image intensity with a tanh function
    
    Parameters
    ----------
    x : array_like
        Input data, can be a vector or a matrix (channels x time)
    param : float or array_like
        Scaling factor for the saturation level
    mode : str, optional
        'ksigma' or 'absolute'. Default is 'ksigma'
    
    Returns
    -------
    y : ndarray
        Saturated data with outliers replaced by the saturation level
    """
    x = np.asarray(x)
    
    # Ensure x is at least 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    if mode == 'ksigma':
        # Compute the scaling factor based on the standard deviation of each channel
        alpha = param * np.std(x, axis=1, keepdims=True)
    elif mode == 'absolute':
        if np.isscalar(param):
            alpha = param * np.ones((x.shape[0], 1))
        else:
            param = np.asarray(param)
            if len(param) == x.shape[0]:
                alpha = param.reshape(-1, 1)
            else:
                raise ValueError('Parameter must be a scalar or a vector with the same number of elements as the data channels')
    else:
        raise ValueError('Undefined mode')
    
    # Scale the input data and apply the tanh function to saturate outliers
    y = alpha * np.tanh(x / alpha)
    
    return y
