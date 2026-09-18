"""
robustness_analysis.py
----------------------
Evaluates noise sensitivity and window ratio robustness for CSD indicators.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def evaluate_noise_sensitivity(signal, noise_levels=[0.01, 0.05, 0.1], window_ratios=[0.3, 0.4, 0.5]):
    results = []
    n = len(signal)
    for noise in noise_levels:
        for w_ratio in window_ratios:
            win_size = int(n * w_ratio)
            noisy_signal = signal + np.random.normal(0, noise, size=n)
            rolling_var = pd.Series(noisy_signal).rolling(window=win_size).var().dropna().values
            tau_var, _ = kendalltau(np.arange(len(rolling_var)), rolling_var)
            results.append({
                "Noise Std": noise, 
                "Window Ratio": f"{int(w_ratio * 100)}%", 
                "Kendall Tau (Var)": round(tau_var, 4)
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Running Noise & Window Sensitivity Analysis ===")
    
    # Generate baseline dummy biomarker signal
    np.random.seed(42)
    time_points = np.linspace(0, 24, 24)
    synthetic_signal = 1000 - 5 * time_points + np.random.normal(0, 2, size=24)
    
    df_sensitivity = evaluate_noise_sensitivity(synthetic_signal)
    print("\nSensitivity Analysis Results:")
    print(df_sensitivity.to_string(index=False))

    # Save to outputs
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "tables")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "noise_sensitivity.csv")
    df_sensitivity.to_csv(csv_path, index=False)
    print(f"\n[Saved] Sensitivity table saved to: {csv_path}")