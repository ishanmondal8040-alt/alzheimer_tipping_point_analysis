import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.interpolate import interp1d
from scipy.stats import kendalltau

# 1. Simulate irregular timepoints (Real patient-like data)
np.random.seed(42)
irregular_times = np.array([0.0, 0.7, 1.5, 2.8, 3.4, 4.2, 5.1, 6.8, 7.9, 8.5, 9.3, 10.0])
raw_signal = 0.4 * irregular_times + np.random.normal(0, scale=0.1 + 0.03 * irregular_times, size=len(irregular_times))

# 2. Resample to regular time intervals via Linear Interpolation
regular_times = np.linspace(irregular_times.min(), irregular_times.max(), 50)
interpolation_function = interp1d(irregular_times, raw_signal, kind='linear')
resampled_signal = interpolation_function(regular_times)

# 3. Apply Detrending (Remove overall growth trend)
detrended_signal = detrend(resampled_signal)

df = pd.DataFrame({
    'time': regular_times,
    'signal': resampled_signal,
    'detrended': detrended_signal
})

# 4. Calculate Rolling Variance on Detrended Signal
window_size = 10
df['rolling_var'] = df['detrended'].rolling(window=window_size).var()

# Helper function for Lag-1 Autocorrelation
def calculate_lag1_autocorr(x):
    if len(x) < 2:
        return np.nan
    return pd.Series(x).autocorr(lag=1)

df['rolling_autocorr'] = df['detrended'].rolling(window=window_size).apply(calculate_lag1_autocorr, raw=True)

# 5. Kendall's Tau Test
clean_var = df['rolling_var'].dropna()
tau_val, p_val = kendalltau(clean_var.index, clean_var.values)
print(f"Detrended Variance Kendall's Tau: {tau_val:.3f} (p-value: {p_val:.4f})")

# 6. Visualization
fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

# Raw vs Resampled Signal
axes[0].plot(irregular_times, raw_signal, 'ro', label='Irregular Patient Data')
axes[0].plot(regular_times, resampled_signal, 'b-', label='Resampled Linear Trend')
axes[0].set_ylabel('Raw Level')
axes[0].legend(loc='upper left')

# Detrended Signal (Noise/Fluctuation Only)
axes[1].plot(regular_times, detrended_signal, color='purple', label='Detrended Signal (Fluctuations)')
axes[1].set_ylabel('Fluctuation')
axes[1].legend(loc='upper left')

# Rolling Variance
axes[2].plot(df['time'], df['rolling_var'], color='orange', label='Rolling Variance (CSD Indicator)')
axes[2].set_ylabel('Variance')
axes[2].legend(loc='upper left')

# Rolling Autocorrelation
axes[3].plot(df['time'], df['rolling_autocorr'], color='green', label='Rolling Lag-1 Autocorr')
axes[3].set_xlabel('Time (Years)')
axes[3].set_ylabel('Autocorrelation')
axes[3].legend(loc='upper left')

plt.tight_layout()
plt.savefig('detrended_csd_analysis.png', dpi=300)
plt.show()