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
            results.append({"Noise Std": noise, "Window Ratio": f"{int(w_ratio * 100)}%", "Kendall Tau (Var)": round(tau_var, 4)})
    return pd.DataFrame(results)