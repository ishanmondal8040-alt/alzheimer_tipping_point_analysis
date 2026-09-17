"""
main_pipeline.py
----------------
Orchestrates the entire Person D pipeline:
1. Loads/Simulates biomarker data
2. Processes CSD indicators
3. Evaluates Model B vs. Model C performance
4. Generates summary figures
"""

import sys
import os
import numpy as np

# Ensure modules in scripts folder can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from early_warning_utils import process_patient_trajectory
from model_evaluation import evaluate_models
from plot_utils import plot_model_comparison


def run_pipeline():
    print("=== Starting AD Tipping Point Analysis Pipeline ===")
    
    # Simulate Synthetic Cohort for End-to-End Test
    rng = np.random.default_rng(42)
    n_patients, T, n_regions = 20, 10, 5

    y_true = rng.normal(0.5, 0.1, size=(n_patients, T, n_regions))
    y_pred_B = y_true + rng.normal(0, 0.08, size=y_true.shape)
    y_pred_C = y_true + rng.normal(0, 0.04, size=y_true.shape)

    # 1. Run Model Evaluation
    report = evaluate_models(y_true, y_pred_B, y_pred_C)
    print("\n--- Model Evaluation Summary ---")
    print(report["summary_table"].to_string(index=False))

    # 2. Save Figure
    metrics_summary = {
        'mae_B': report["summary_table"].loc[0, "MAE (mean)"],
        'mae_C': report["summary_table"].loc[1, "MAE (mean)"],
        'rmse_B': report["summary_table"].loc[0, "RMSE (mean)"],
        'rmse_C': report["summary_table"].loc[1, "RMSE (mean)"]
    }
    plot_model_comparison(metrics_summary)
    print("\n=== Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline()