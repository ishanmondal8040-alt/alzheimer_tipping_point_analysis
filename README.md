# 🧠 Continuous-Time Modeling & Critical Slowing Down (CSD) Analysis for Alzheimer's Disease

An end-to-end computational pipeline designed to detect **Critical Slowing Down (CSD)** early-warning signals and evaluate longitudinal biomarker trajectories in Alzheimer's Disease progression.

---

## 📌 Project Overview
Alzheimer's Disease (AD) progression involves dynamic non-linear shifts in biomarker levels (such as Amyloid-β and Tau). This project identifies candidate **tipping points (critical transitions)** prior to clinical conversion (MCI → AD) by analyzing temporal fluctuations and comparing predictive dynamical models.

### Key Focus Areas:
1. **Statistical Early-Warning Signals (EWS):** Compute Rolling Variance, Lag-1 Autocorrelation (AC1), and Kendall's $\tau$ trend dynamics on non-linearly detrended biomarker series.
2. **Model-Based Stability Diagnostics:** Extract continuous-time hidden state trajectories $h(t)$ from Liquid Neural Networks (LTC/CfC) to calculate PCA order parameters $z(t)$ and leading Jacobian eigenvalues ($\text{Re}(\lambda_{\max}) \to 0^-$).
3. **Rigorous Model Evaluation:** Perform non-parametric paired Wilcoxon signed-rank tests, bootstrap confidence interval estimation, and MCI-to-AD conversion timing accuracy comparisons between baseline Graph-Laplacian diffusion models and LNNs.

---

## 🛠 Project Structure

```text
alzheimer_tipping_point_analysis/
│
├── early_warning_utils.py       # Resampling, Gaussian kernel detrending, dynamic windowing & CSD metrics
├── hidden_state_extraction.py   # Latent trajectory extraction, PCA reduction & Jacobian eigenvalue solver
├── model_evaluation.py          # Paired Wilcoxon tests, bootstrap CIs & conversion timing evaluation
├── detrending_csd.py            # Sandbox module for non-linear fluctuation extraction
├── early_warning_test.py        # Initial synthetic pipeline verification script
└── outputs/                     # Generated visual plots and statistical metric logs