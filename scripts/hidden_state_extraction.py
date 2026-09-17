"""
hidden_state_extraction.py
---------------------------
Extracts hidden state trajectory h(t) from Person C's Liquid Neural Network (LTC / CfC)
and computes PCA-reduced z(t) and Jacobian leading eigenvalues.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA


def get_hidden_states(model, x_seq, device="cpu"):
    """
    Run the trained liquid model in inference mode and collect hidden states.
    """
    model.eval()
    x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs, hidden_seq = model(x_tensor, return_sequences=True)

    h_traj = hidden_seq.squeeze(0).cpu().numpy()
    return h_traj


def reduce_hidden_states(h_traj, n_components=1):
    """
    PCA-reduce the hidden trajectory to a low-dimensional surrogate for CSD analysis.
    """
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(h_traj)
    explained_var = pca.explained_variance_ratio_.sum()

    if n_components == 1:
        z = z.ravel()

    return z, explained_var


def compute_jacobian_eigs(model, h_traj, x_seq, device="cpu"):
    """
    Compute the leading eigenvalue of the Jacobian of the learned ODE vector field.
    Approaching 0 indicates Critical Slowing Down.
    """
    model.eval()
    T = h_traj.shape[0]
    leading_eig = np.zeros(T)

    def ode_func(h, x_t):
        h_next, _ = model.rnn_cell(x_t, h)
        return h_next

    for t in range(T):
        h_t = torch.tensor(h_traj[t], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        x_t = torch.tensor(x_seq[t], dtype=torch.float32, device=device).unsqueeze(0)

        J = torch.autograd.functional.jacobian(
            lambda h: ode_func(h, x_t), h_t
        )
        J = J.squeeze().detach().cpu().numpy()

        eigvals = np.linalg.eigvals(J)
        leading_eig[t] = eigvals.real.max()

    return leading_eig


def extract_ews_candidates(model, x_seq, device="cpu"):
    """
    Extracts hidden states, PCA trajectory, and Jacobian eigenvalues for early-warning metrics.
    """
    h_traj = get_hidden_states(model, x_seq, device=device)
    z_pca, explained_var = reduce_hidden_states(h_traj, n_components=1)
    leading_eig = compute_jacobian_eigs(model, h_traj, x_seq, device=device)

    return {
        "h_traj": h_traj,
        "z_pca": z_pca,
        "explained_var": explained_var,
        "leading_eig": leading_eig,
    }


if __name__ == "__main__":
    print("hidden_state_extraction module loaded successfully.")

