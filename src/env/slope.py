import numpy as np
from scipy.signal import savgol_filter

# Slope limit for discontinuity
def constraint_slope(f:np.ndarray, fx:np.ndarray, dx:float, axis:int, direction:int):
    fx_fd = (f - np.roll(f, direction, axis=axis))/dx

    if direction == 1:
        fx = np.maximum(0.0, np.minimum(1.0, fx_fd / (fx + 1.0e-8 * (fx == 0)))) * fx
    elif direction == -1:
        fx = np.maximum(0.0, np.minimum(1.0, (-1) * fx_fd / (fx + 1.0e-8 * (fx == 0)))) * fx
        
    return fx

def slopelimit(f:np.ndarray, fx:np.ndarray, fy:np.ndarray, dx:float):
    R = -1
    L = 1
    f_dx = constraint_slope(f, fx, dx, axis = 0, direction = L)
    f_dx = constraint_slope(f, f_dx, dx, axis = 0, direction = R)
    f_dy = constraint_slope(f, fy, dx, axis = 1, direction = L)
    f_dy = constraint_slope(f, f_dy, dx, axis = 1, direction = R)
    return f_dx, f_dy

def get_median_filtered(f:np.ndarray, threshold=3):
 
    df = np.abs(f - np.median(f))
    df_median = np.median(df)
 
    if df_median == 0:
        s = 0
    else:
        s = df / float(df_median)
        
    mask = s > threshold
    f[mask] = np.median(f)
    return f

def smoothing(f:np.ndarray):
    
    # Option 1. 5-point weighting sum
    # R = -1
    # L = 1
    # fxr = np.roll(f, R, axis=0)
    # fxl = np.roll(f, L, axis=0)
    # fyr = np.roll(f, R, axis=1)
    # fyl = np.roll(f, L, axis=1)
    # f_smooth = 0.15 * (fxr + fxl + fyr + fyl) + 0.4 * f
    
    # Option 2. Median filtering method
    f_smooth = get_median_filtered(f, threshold = 3.0)
    
    return f_smooth
