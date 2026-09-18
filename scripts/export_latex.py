import os
import pandas as pd

def export_to_latex(summary_df, save_path="outputs/tables/model_performance.tex"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    latex_code = summary_df.to_latex(
        index=False,
        caption="Comparative performance evaluation between Graph Laplacian and Liquid Neural Network (LNN) models on ADNI trajectories.",
        label="tab:model_performance",
        column_format="lcccc",
        position="htbp"
    )
    with open(save_path, "w") as f:
        f.write(latex_code)