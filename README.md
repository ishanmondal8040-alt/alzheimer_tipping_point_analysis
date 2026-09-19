# 🧠 Alzheimer's Disease Tipping Point Analysis (CSD Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational neuroscience and bioinformatics framework designed to detect dynamic early warning signals (**Critical Slowing Down - CSD**) prior to irreversible neurodegenerative transitions in Alzheimer's Disease progression.

---

## 📌 Executive Summary

Alzheimer's Disease (AD) progression is non-linear; patients often maintain cognitive stability for years before experiencing a rapid, irreversible clinical tipping point. This repository integrates longitudinal ADNI biomarker processing, non-parametric statistical CSD indicators, dynamic spatial correlation across multi-biomarker networks, and model benchmarking (**Graph Laplacian Dynamics vs. Continuous-Time Liquid Neural Networks**).

### 🌟 Key Responsibilities & Deliverables (Lead Integration)
* **ADNI Data Pipeline (`adni_data_loader.py`)**: Built cubic spline & linear fallback interpolation engines with global Z-score normalization for longitudinal patient trajectories.
* **Early Warning CSD Engine (`early_warning_utils.py`)**: Implemented Gaussian detrending, Rolling Variance, Lag-1 Autocorrelation (AC1), and Kendall's Tau trend statistics.
* **Spatial Cross-Correlation Module (`advanced_csd.py`)**: Quantified network-level inter-biomarker dynamic synchronization near critical transitions.
* **Robustness & Sensitivity Sweep (`robustness_analysis.py`)**: Tested indicator stability across varying noise standard deviations ($0.01$ to $0.10$) and rolling window ratios ($30\%$ to $50\%$).
* **Model Benchmarking Pipeline (`main_pipeline.py`)**: Benchmarked predictions over 1,000 bootstrap iterations.
* **Interactive Visualizer (`scripts/app.py`)**: Deployed a real-time Streamlit web app for dynamic signal analysis and interactive parameter tweaking.

---

## 📂 Project Architecture

```text
alzheimer_tipping_point_analysis/
├── data/
│   ├── raw/                        # ADNI longitudinal biomarker CSVs
│   └── processed/                  # Normalized & interpolated trajectories
├── models/
│   ├── graph_laplacian/            # Graph Laplacian dynamics weights & binaries
│   └── lnn/                        # Liquid Neural Network (LNN) checkpoints
├── outputs/
│   ├── figures/                    # Exported CSD trajectory & dynamic correlation plots
│   └── tables/                     # Statistical export tables & LaTeX source outputs
│       ├── model_performance.tex   # Model benchmarking metrics
│       ├── noise_sensitivity.csv   # Kendall's Tau window & noise sweep results
│       └── robustness_results.csv  # Robustness outputs
├── scripts/
│   ├── adni_data_loader.py         # Data preprocessing & spline interpolation
│   ├── early_warning_utils.py      # Gaussian detrending, Rolling Variance, AC1, Kendall Tau
│   ├── advanced_csd.py             # Dynamic multi-biomarker spatial cross-correlation
│   ├── robustness_analysis.py      # Noise sensitivity & window sweep testing
│   ├── main_pipeline.py            # Main execution pipeline & model benchmarking
│   ├── app.py                      # Streamlit interactive visualizer
│   └── export_latex.py             # LaTeX table formatter
├── paper_draft.md                  # Comprehensive research paper manuscript
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation