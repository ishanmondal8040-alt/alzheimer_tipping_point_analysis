"""
app.py
------
Streamlit Dashboard for Alzheimer's Tipping Point Analysis & CSD Visualization
"""

import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from early_warning_utils import compute_csd_indicators

st.set_page_config(page_title="AD Tipping Point Visualizer", layout="wide")

st.title("🧠 Alzheimer's Tipping Point & CSD Indicator Visualizer")
st.markdown("Interactive analysis of Critical Slowing Down (CSD) indicators across ADNI biomarker trajectories.")

# Sidebar Controls
st.sidebar.header("Simulation Settings")
n_months = st.sidebar.slider("Trajectory Length (Months)", min_value=12, max_value=60, value=24, step=6)
noise_level = st.sidebar.slider("Biomarker Noise Level", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# Generate Synthetic Trajectory with a Tipping Point
rng = np.random.default_rng(42)
time_points = np.linspace(0, n_months, n_months)
signal = 1000 - 5 * time_points + (1 + time_points / 10) * rng.normal(0, 10 * noise_level, size=n_months)

# Compute CSD
csd_results = compute_csd_indicators(signal)

# Layout: 2 Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Summary Metrics")
    st.metric(label="Kendall Tau (Rolling Variance)", value=f"{csd_results['kendall_tau_var']:.4f}")
    st.metric(label="Kendall Tau (AC1)", value=f"{csd_results['kendall_tau_ac1']:.4f}")
    
    st.info("A positive Kendall Tau indicates increasing fluctuation/memory, signaling an approaching critical transition (Tipping Point).")

with col2:
    st.subheader("CSD Indicator Plots")
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    axes[0].plot(time_points, signal, color='#2b5c8f', label='Biomarker Trajectory')
    axes[0].set_ylabel("Value")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(time_points[-len(csd_results['rolling_variance']):], csd_results['rolling_variance'], color='#27ae60', label='Rolling Variance')
    axes[1].set_ylabel("Variance")
    axes[1].legend(loc='upper right')
    
    # Lag-1 AC1 Plot with proper Y-limit padding
    axes[2].plot(time_points[-len(csd_results['autocorrelation_ac1']):], csd_results['autocorrelation_ac1'], color='#8e44ad', label='Lag-1 AC1')
    axes[2].set_xlabel("Months")
    axes[2].set_ylabel("AC1")
    axes[2].set_ylim(0.70, 1.12)  # 1.12 পর্যন্ত বাড়ানো হলো যাতে ডানদিকের উপরে লেজেন্ডের জন্য ফাঁকা জায়গা থাকে
    axes[2].legend(loc='upper right')  # আগের মতো সব কয়টি ডানদিকের উপরে
    
    fig.tight_layout()
    st.pyplot(fig)