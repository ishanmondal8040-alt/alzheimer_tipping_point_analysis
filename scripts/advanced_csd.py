"""
advanced_csd.py
---------------
Computes spatial cross-correlation and dynamic instability indicators 
across multiple biomarker trajectories.
"""

import numpy as np
import pandas as pd


def compute_cross_correlation_matrix(df_subject, biomarker_cols=None):
    """
    Computes pairwise Pearson correlation between biomarkers for a single subject.
    """
    if biomarker_cols is None:
        biomarker_cols = ["ABETA", "TAU", "PTAU", "FDG", "Hippocampus"]

    valid_cols = [c for c in biomarker_cols if c in df_subject.columns]
    if len(valid_cols) < 2:
        return None

    corr_matrix = df_subject[valid_cols].corr()
    return corr_matrix


def compute_mean_spatial_correlation(df_subject, biomarker_cols=None):
    """
    Calculates the average off-diagonal correlation as an indicator 
    of system-wide dynamic synchronization (CSD indicator).
    """
    corr_matrix = compute_cross_correlation_matrix(df_subject, biomarker_cols)
    if corr_matrix is None:
        return np.nan

    values = corr_matrix.values
    mask = ~np.eye(values.shape[0], dtype=bool)
    mean_corr = np.nanmean(values[mask])
    return mean_corr


if __name__ == "__main__":
    dummy_df = pd.DataFrame({
        "ABETA": [1.2, 1.0, 0.8, 0.5, 0.2],
        "TAU": [0.1, 0.3, 0.6, 0.9, 1.2],
        "FDG": [1.5, 1.3, 1.1, 0.8, 0.4]
    })
    mean_corr = compute_mean_spatial_correlation(dummy_df)
    print(f"[Advanced CSD] Mean Spatial Cross-Correlation: {mean_corr:.4f}")