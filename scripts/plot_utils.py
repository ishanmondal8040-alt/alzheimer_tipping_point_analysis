import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="ticks", font_scale=1.1)

def plot_model_comparison(metrics_dict, save_path="outputs/figures/model_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = ['Person B (Graph Laplacian)', 'Person C (LNN)']
    mae_vals = [metrics_dict['mae_B'], metrics_dict['mae_C']]
    rmse_vals = [metrics_dict['rmse_B'], metrics_dict['rmse_C']]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, mae_vals, width, label='MAE', color='#4C72B0')
    ax.bar(x + width/2, rmse_vals, width, label='RMSE', color='#DD8452')
    ax.set_ylabel('Error Score')
    ax.set_title('Model Performance Comparison (ADNI Cohort)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()