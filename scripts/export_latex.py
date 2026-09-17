"""
export_latex.py
---------------
Exports model evaluation metrics and CSD statistical summaries
directly into publication-ready LaTeX table format (.tex).
"""

import os
import pandas as pd


def export_to_latex(summary_df, save_path="outputs/tables/model_performance.tex"):
    """
    Converts a pandas DataFrame into a formatted LaTeX table file.
    """
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
        
    print(f"LaTeX table exported successfully to: {save_path}")


if __name__ == "__main__":
    # Test script with sample data
    dummy_data = {
        "Model": ["Person B (Graph Laplacian)", "Person C (LNN)"],
        "MAE (mean)": [0.0646, 0.0323],
        "MAE 95% CI": ["[0.0615, 0.0676]", "[0.0309, 0.0337]"],
        "RMSE (mean)": [0.0810, 0.0405],
        "RMSE 95% CI": ["[0.0776, 0.0842]", "[0.0389, 0.0421]"]
    }
    df = pd.DataFrame(dummy_data)
    export_to_latex(df)