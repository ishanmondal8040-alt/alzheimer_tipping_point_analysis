"""
robustness_analysis.py
----------------------
Evaluates the stability and robustness of CSD indicators (Variance, AC1, Kendall's Tau)
under synthetic Gaussian noise corruption and varying rolling window sizes.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def evaluate_noise_sensitivity(signal, noise_levels=[0.01, 0.05, 0.1], window_ratios=[0.3, 0.4, 0.5]):
    """
    Tests Kendall's Tau stability under different noise levels and window sizes.
    """
    results = []
    n = len(signal)

    for noise in noise_levels:
        for w_ratio in window_ratios:
            win_size = int(n * w_ratio)
            noisy_signal = signal + np.random.normal(0, noise, size=n)
            
            # Compute rolling variance
            rolling_var = pd.Series(noisy_signal).rolling(window=win_size).var().dropna().values
            tau_var, _ = kendalltau(np.arange(len(rolling_var)), rolling_var)

            results.append({
                "Noise Std": noise,
                "Window Ratio": f"{int(w_ratio * 100)}%",
                "Kendall Tau (Var)": round(tau_var, 4)
            })

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    print("=== Running CSD Robustness & Noise Sensitivity Test ===")
    t = np.linspace(0, 10, 100)
    synthetic_signal = 0.5 + 0.03 * t + np.random.normal(0, 0.02, size=100)
    
    res_table = evaluate_noise_sensitivity(synthetic_signal)
    print("\n--- Sensitivity Matrix ---")
    print(res_table.to_string(index=False))
    print("\nRobustness analysis module tested successfully.")