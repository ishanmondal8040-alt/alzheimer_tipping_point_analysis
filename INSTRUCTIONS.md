# 📋 Team Data Exchange & Integration Guidelines

To seamlessly run the final analysis pipeline (`scripts/main_pipeline.py`), Person A, B, and C should format and export their respective outputs according to the following specifications:

---

## 1. Person A (Data Lead — Clinical & Biomarker Data)
* **File Location:** Save the preprocessed ADNI patient time-series data to `data/adni_processed.npy` or `data/adni_processed.csv`.
* **Array Shape:** `(n_patients, timepoints, n_regions)`
* **Timepoints:** Include absolute visit intervals (in months or years from baseline) to ensure precise Gaussian detrending and interpolation.

---

## 2. Person B (Baseline Model Lead — Graph Laplacian Diffusion)
* **File Location:** Save predicted biomarker trajectories to `outputs/y_pred_B.npy`.
* **Array Shape:** `(n_patients, timepoints, n_regions)` — matching Person A's data dimensions.
* **Conversion Estimates:** Save predicted MCI-to-AD conversion times (if applicable) in `outputs/conversion_times_B.npy`.

---

## 3. Person C (Advanced Model Lead — Liquid Neural Network)
* **File Location:** Save predicted biomarker trajectories to `outputs/y_pred_C.npy`.
* **Hidden States ($h(t)$):** Export continuous-time hidden state sequence to `outputs/hidden_states_C.npy` with shape `(n_patients, timepoints, hidden_dim)` for PCA reduction and Jacobian eigenvalue analysis.
* **Model Interface:** Ensure the trained PyTorch model exposes its step-wise cell (`.rnn_cell`) to compute the leading Jacobian eigenvalues $\text{Re}(\lambda_{\max})$.

---

## 4. Person D (Dipanjan — Analysis & Pipeline Lead)
Once files are placed in `data/` and `outputs/`, run:
```bash
python scripts/main_pipeline.py