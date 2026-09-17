import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# 1. Generate synthetic longitudinal time-series data with increasing noise/fluctuation
np.random.seed(42)
time_points = np.linspace(0, 10, 100)

# Simulate progressive signal with increasing variance over time
signal = 0.5 * time_points + np.random.normal(0, scale=0.1 + 0.05 * time_points, size=len(time_points))

df = pd.DataFrame({
    'time': time_points, 
    'signal': signal
})

# 2. Compute rolling window statistics (Window size = 20)
window_size = 20

# Calculate Rolling Variance
df['rolling_var'] = df['signal'].rolling(window=window_size).var()

# Helper function for Lag-1 Autocorrelation
def calculate_lag1_autocorr(x):
    if len(x) < 2:
        return np.nan
    return pd.Series(x).autocorr(lag=1)

# Calculate Rolling Lag-1 Autocorrelation
df['rolling_autocorr'] = df['signal'].rolling(window=window_size).apply(calculate_lag1_autocorr, raw=True)

# 3. Calculate Kendall's Tau to assess indicator trend direction
clean_variance = df['rolling_var'].dropna()
tau_var, p_val_var = kendalltau(clean_variance.index, clean_variance.values)

print(f"Variance Kendall's Tau: {tau_var:.3f} (p-value: {p_val_var:.4f})")

# 4. Visualization setup
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Plot Raw Signal
axes[0].plot(df['time'], df['signal'], label='Synthetic Biomarker Signal', color='blue')
axes[0].set_ylabel('Biomarker Level')
axes[0].legend(loc='upper left')

# Plot Rolling Variance
axes[1].plot(df['time'], df['rolling_var'], label='Rolling Variance', color='orange')
axes[1].set_ylabel('Variance')
axes[1].legend(loc='upper left')

# Plot Rolling Lag-1 Autocorrelation
axes[2].plot(df['time'], df['rolling_autocorr'], label='Rolling Lag-1 Autocorrelation', color='green')
axes[2].set_xlabel('Time (Years)')
axes[2].set_ylabel('Autocorrelation')
axes[2].legend(loc='upper left')

plt.tight_layout()
plt.savefig('early_warning_signals_plot.png', dpi=300)
plt.show()