"""
model_evaluation.py
--------------------
Compares Person B's baseline model against Person C's Liquid Neural Network.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def mae(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def per_patient_errors(y_true, y_pred, metric_fn):
    n_patients = y_true.shape[0]
    errors = np.zeros(n_patients)
    for i in range(n_patients):
        errors[i] = metric_fn(y_true[i], y_pred[i])
    return errors


def bootstrap_ci(errors, n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(errors)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(errors, size=n, replace=True)
        boot_means[b] = sample.mean()

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    lower = np.percentile(boot_means, lower_pct)
    upper = np.percentile(boot_means, upper_pct)
    return errors.mean(), lower, upper


def paired_model_comparison(errors_B, errors_C):
    stat, p_value = wilcoxon(errors_B, errors_C)
    median_B, median_C = np.median(errors_B), np.median(errors_C)
    better_model = "Person C (LNN)" if median_C < median_B else "Person B (Graph Laplacian)"

    return {
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "median_error_B": median_B,
        "median_error_C": median_C,
        "lower_median_error_model": better_model,
    }


def conversion_time_error(true_times, pred_times):
    mask = ~np.isnan(true_times) & ~np.isnan(pred_times)
    diffs = pred_times[mask] - true_times[mask]

    return {
        "n_converters_matched": int(mask.sum()),
        "mean_timing_error_months": float(np.mean(diffs)) if mask.any() else np.nan,
        "mae_timing_months": float(np.mean(np.abs(diffs))) if mask.any() else np.nan,
        "pct_within_6_months": float(np.mean(np.abs(diffs) <= 6) * 100) if mask.any() else np.nan,
    }


def evaluate_models(
    y_true,
    y_pred_B,
    y_pred_C,
    conversion_times_true=None,
    conversion_times_B=None,
    conversion_times_C=None,
):
    mae_B = per_patient_errors(y_true, y_pred_B, mae)
    mae_C = per_patient_errors(y_true, y_pred_C, mae)
    rmse_B = per_patient_errors(y_true, y_pred_B, rmse)
    rmse_C = per_patient_errors(y_true, y_pred_C, rmse)

    mean_mae_B, lo_mae_B, hi_mae_B = bootstrap_ci(mae_B)
    mean_mae_C, lo_mae_C, hi_mae_C = bootstrap_ci(mae_C)
    mean_rmse_B, lo_rmse_B, hi_rmse_B = bootstrap_ci(rmse_B)
    mean_rmse_C, lo_rmse_C, hi_rmse_C = bootstrap_ci(rmse_C)

    summary_table = pd.DataFrame(
        {
            "Model": ["Person B (Graph Laplacian)", "Person C (LNN)"],
            "MAE (mean)": [mean_mae_B, mean_mae_C],
            "MAE 95% CI": [f"[{lo_mae_B:.4f}, {hi_mae_B:.4f}]", f"[{lo_mae_C:.4f}, {hi_mae_C:.4f}]"],
            "RMSE (mean)": [mean_rmse_B, mean_rmse_C],
            "RMSE 95% CI": [f"[{lo_rmse_B:.4f}, {hi_rmse_B:.4f}]", f"[{lo_rmse_C:.4f}, {hi_rmse_C:.4f}]"],
        }
    )

    significance = paired_model_comparison(mae_B, mae_C)

    result = {
        "summary_table": summary_table,
        "significance": significance,
    }

    if conversion_times_true is not None:
        if conversion_times_B is not None:
            result["timing_B"] = conversion_time_error(conversion_times_true, conversion_times_B)
        if conversion_times_C is not None:
            result["timing_C"] = conversion_time_error(conversion_times_true, conversion_times_C)

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_patients, T, n_regions = 30, 10, 5

    y_true = rng.normal(0.5, 0.1, size=(n_patients, T, n_regions))
    y_pred_B = y_true + rng.normal(0, 0.08, size=y_true.shape)
    y_pred_C = y_true + rng.normal(0, 0.05, size=y_true.shape)

    report = evaluate_models(y_true, y_pred_B, y_pred_C)
    print(report["summary_table"].to_string(index=False))