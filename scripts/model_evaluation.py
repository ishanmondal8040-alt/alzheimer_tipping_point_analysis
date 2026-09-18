import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def bootstrap_ci(metric_series, n_bootstrap=1000, ci=95):
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(metric_series, size=len(metric_series), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

def evaluate_models(y_true, y_pred_B, y_pred_C):
    mae_B = np.mean(np.abs(y_true - y_pred_B), axis=(1, 2))
    mae_C = np.mean(np.abs(y_true - y_pred_C), axis=(1, 2))
    rmse_B = np.sqrt(np.mean((y_true - y_pred_B)**2, axis=(1, 2)))
    rmse_C = np.sqrt(np.mean((y_true - y_pred_C)**2, axis=(1, 2)))
    
    stat_mae, p_mae = wilcoxon(mae_B, mae_C)
    
    ci_mae_B = bootstrap_ci(mae_B)
    ci_mae_C = bootstrap_ci(mae_C)
    ci_rmse_B = bootstrap_ci(rmse_B)
    ci_rmse_C = bootstrap_ci(rmse_C)
    
    df_summary = pd.DataFrame([
        {"Model": "Person B (Graph Laplacian)", "MAE (mean)": np.mean(mae_B), "MAE 95% CI": f"[{ci_mae_B[0]:.4f}, {ci_mae_B[1]:.4f}]", "RMSE (mean)": np.mean(rmse_B), "RMSE 95% CI": f"[{ci_rmse_B[0]:.4f}, {ci_rmse_B[1]:.4f}]"},
        {"Model": "Person C (LNN)", "MAE (mean)": np.mean(mae_C), "MAE 95% CI": f"[{ci_mae_C[0]:.4f}, {ci_mae_C[1]:.4f}]", "RMSE (mean)": np.mean(rmse_C), "RMSE 95% CI": f"[{ci_rmse_C[0]:.4f}, {ci_rmse_C[1]:.4f}]"}
    ])
    return {"summary_table": df_summary, "p_value_mae": p_mae}