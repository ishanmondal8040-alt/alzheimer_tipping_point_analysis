import numpy as np
import pandas as pd
from scipy.signal.windows import gaussian
from scipy.stats import kendalltau

def gaussian_detrend(series, bandwidth=5):
    # Kernel length should scale with bandwidth, NOT with len(series).
    # A kernel as long as the whole series causes heavy zero-padding
    # edge artifacts near the start/end of short clinical trajectories.
    kernel_len = min(len(series), max(int(6 * bandwidth) | 1, 3))  # odd, >=3
    kernel = gaussian(kernel_len, std=bandwidth)
    kernel /= kernel.sum()
    smooth = np.convolve(series, kernel, mode='same')
    return series - smooth

def compute_csd_indicators(series, window_ratio=0.5):
    n = len(series)
    win_size = int(n * window_ratio)
    detrended = gaussian_detrend(series)
    s = pd.Series(detrended)
    rolling_var = s.rolling(window=win_size).var().dropna().values
    
    ac1 = []
    for i in range(win_size, n + 1):
        sub = detrended[i - win_size:i]
        val = np.corrcoef(sub[:-1], sub[1:])[0, 1] if len(sub) > 1 else 0
        ac1.append(val)
    ac1 = np.array(ac1)
    
    tau_var, _ = kendalltau(np.arange(len(rolling_var)), rolling_var)
    tau_ac1, _ = kendalltau(np.arange(len(ac1)), ac1)
    
    return {
        "detrended": detrended,
        "rolling_variance": rolling_var,
        "autocorrelation_ac1": ac1,
        "kendall_tau_var": tau_var,
        "kendall_tau_ac1": tau_ac1
    }