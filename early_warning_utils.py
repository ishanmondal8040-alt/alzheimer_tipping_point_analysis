"""
early_warning_utils.py
-----------------------
Updated with Gaussian Kernel Detrending and Robustness Checks per Dakos et al. (2012).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import kendalltau


def gaussian_detrend(signal, sigma=5.0):
    """
    Applies Gaussian filter detrending to remove non-linear baseline trends.
    Recommended over global linear detrending for non-linear biological trajectories.
    """
    smooth_trend = gaussian_filter1d(signal, sigma=sigma)
    fluctuations = signal - smooth_trend
    return fluctuations, smooth_trend


def process_patient_trajectory(times, values, target_points=50, window_size=None, sigma=5.0):
    """
    Interpolates irregular timepoints, applies Gaussian detrending, and computes 
    rolling CSD indicators (Variance, Lag-1 Autocorrelation, and Kendall's Tau).
    """
    # 1. Resample irregular timepoints to regular grid
    reg_times = np.linspace(np.min(times), np.max(times), target_points)
    interp_func = interp1d(times, values, kind='linear')
    resampled_vals = interp_func(reg_times)

    # 2. Gaussian Detrending (Kernel-based)
    detrended_vals, trend = gaussian_detrend(resampled_vals, sigma=sigma)

    df = pd.DataFrame({
        'time': reg_times,
        'raw_resampled': resampled_vals,
        'trend': trend,
        'signal': detrended_vals
    })

    # 3. Dynamic half-length window assertion
    if window_size is None:
        window_size = max(5, int(target_points * 0.5))

    # 4. Compute Rolling CSD Metrics
    df['rolling_var'] = df['signal'].rolling(window=window_size).var()
    df['rolling_autocorr'] = df['signal'].rolling(window=window_size).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan, 
        raw=True
    )

    # 5. Calculate Kendall's Tau for Indicator Trends
    clean_var = df['rolling_var'].dropna()
    tau_var, p_val_var = kendalltau(clean_var.index, clean_var.values)

    clean_ac1 = df['rolling_autocorr'].dropna()
    tau_ac1, p_val_ac1 = kendalltau(clean_ac1.index, clean_ac1.values)

    metrics = {
        'tau_var': tau_var,
        'p_val_var': p_val_var,
        'tau_ac1': tau_ac1,
        'p_val_ac1': p_val_ac1
    }

    return df, metrics


def plot_patient_csd(df, patient_id="Patient_001"):
    """
    Plots resampled raw signal, detrended noise, rolling variance, and rolling AC1.
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(df['time'], df['raw_resampled'], label='Resampled Signal', color='blue')
    axes[0].plot(df['time'], df['trend'], label='Gaussian Trend', color='red', linestyle='--')
    axes[0].set_ylabel('Biomarker')
    axes[0].set_title(f'CSD & Detrending Analysis for {patient_id}')
    axes[0].legend(loc='upper left')

    axes[1].plot(df['time'], df['signal'], color='purple', label='Detrended Fluctuation')
    axes[1].set_ylabel('Fluctuation')
    axes[1].legend(loc='upper left')

    axes[2].plot(df['time'], df['rolling_var'], color='orange', label='Rolling Variance')
    axes[2].set_ylabel('Variance')
    axes[2].legend(loc='upper left')

    axes[3].plot(df['time'], df['rolling_autocorr'], color='green', label='Rolling Lag-1 Autocorr')
    axes[3].set_xlabel('Time (Years)')
    axes[3].set_ylabel('Autocorrelation')
    axes[3].legend(loc='upper left')

    plt.tight_layout()
    plt.show()