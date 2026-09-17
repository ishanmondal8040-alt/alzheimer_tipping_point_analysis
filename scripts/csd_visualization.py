"""
csd_visualization.py
--------------------
Plots temporal biomarker trajectories alongside Critical Slowing Down (CSD)
early-warning indicators: Rolling Variance, Lag-1 Autocorrelation (AC1),
and Kendall's Tau trend indicators.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_csd_indicators(time_points, original_signal, detrended_signal, variance_series, ac1_series, save_path="outputs/figures/csd_indicators.png"):
    """
    Plots a 4-panel publication-ready CSD diagnostic figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. Raw Biomarker Trajectory
    axes[0].plot(time_points, original_signal, color='#2b5c8f', lw=2, label='Raw Biomarker Trajectory')
    axes[0].set_ylabel('Biomarker Level')
    axes[0].legend(loc='upper left')
    axes[0].set_title("Critical Slowing Down (CSD) Diagnostic Pipeline")

    # 2. Gaussian Detrended Fluctuation
    axes[1].plot(time_points, detrended_signal, color='#e74c3c', lw=1.5, linestyle='--', label='Detrended Fluctuation (y - y_smooth)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle=':')
    axes[1].set_ylabel('Fluctuation')
    axes[1].legend(loc='upper left')

    # 3. Rolling Variance
    axes[2].plot(time_points[-len(variance_series):], variance_series, color='#27ae60', lw=2, label='Rolling Variance')
    axes[2].set_ylabel('Variance')
    axes[2].legend(loc='upper left')

    # 4. Lag-1 Autocorrelation (AC1)
    axes[3].plot(time_points[-len(ac1_series):], ac1_series, color='#8e44ad', lw=2, label='Lag-1 Autocorrelation (AC1)')
    axes[3].set_ylabel('AC(1)')
    axes[3].set_xlabel('Time (Months / Years from Baseline)')
    axes[3].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"CSD Indicator figure saved to: {save_path}")


if __name__ == "__main__":
    # Test script with synthetic sequence
    t = np.linspace(0, 10, 100)
    signal = 0.5 + 0.05 * t + np.random.normal(0, 0.02, size=100)
    detrended = signal - (0.5 + 0.05 * t)
    var = np.convolve(detrended**2, np.ones(10)/10, mode='valid')
    ac1 = np.corrcoef(detrended[:-1], detrended[1:])[0, 1] * np.ones_like(var)

    plot_csd_indicators(t, signal, detrended, var, ac1)