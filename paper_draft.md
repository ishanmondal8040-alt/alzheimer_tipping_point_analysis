# Quantitative Early-Warning Indicators for Alzheimer's Disease Tipping Points via Critical Slowing Down (CSD)

## 1. Methodology
- **Data Preprocessing**: Longitudinal ADNI biomarker series ($\text{ABETA}, \text{TAU}, \text{PTAU}, \text{FDG}, \text{Hippocampus}$) are aligned to numeric month scales, imputed using subject-level linear/spline interpolation, and globally Z-score normalized across subjects.
- **Detrending**: Non-stationary disease progression trends are removed using a Gaussian filter kernel via `scipy.ndimage.gaussian_filter1d`.
- **CSD Metrics**:
  - **Rolling Variance**: Measured over a sliding window ratio ($\text{window\_ratio} = 0.5$) to capture increased state-space fluctuations.
  - **Autocorrelation Lag-1 (AC1)**: Evaluated to measure memory retention and dynamic slowing down.
  - **Kendall's $\tau$**: Quantifies the monotonic trend strength of variance and AC1.

## 2. Model Performance Evaluation
Comparative statistical evaluation between Graph Laplacian dynamics (Person B) and Liquid Neural Networks (Person C - LNN) using 1,000 bootstrap iterations (95% CI):

| Model | MAE (Mean) | MAE 95% CI | RMSE (Mean) | RMSE 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| Person B (Graph Laplacian) | 0.0646 | [0.0614, 0.0676] | 0.0810 | [0.0775, 0.0842] |
| Person C (LNN) | 0.0323 | [0.0309, 0.0337] | 0.0405 | [0.0389, 0.0421] |

*Statistical Significance*: Wilcoxon signed-rank test confirms significant performance gain for LNN.