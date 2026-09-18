import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_csd_indicators(time_points, original_signal, detrended_signal, variance_series, ac1_series, save_path="outputs/figures/csd_indicators.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time_points, original_signal, color='#2b5c8f', lw=2, label='Raw Biomarker Trajectory')
    axes[0].legend(loc='upper left')
    axes[1].plot(time_points, detrended_signal, color='#e74c3c', lw=1.5, linestyle='--', label='Detrended Fluctuation')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle=':')
    axes[1].legend(loc='upper left')
    axes[2].plot(time_points[-len(variance_series):], variance_series, color='#27ae60', lw=2, label='Rolling Variance')
    axes[2].legend(loc='upper left')
    axes[3].plot(time_points[-len(ac1_series):], ac1_series, color='#8e44ad', lw=2, label='Lag-1 Autocorrelation (AC1)')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()