import numpy as np
from sklearn.decomposition import PCA

def extract_latent_states(hidden_states_matrix, n_components=2):
    pca = PCA(n_components=n_components)
    latent_trajectories = pca.fit_transform(hidden_states_matrix)
    return latent_trajectories, pca

def compute_leading_eigenvalues(jacobian_matrices):
    leading_eigs = []
    for J in jacobian_matrices:
        eigs = np.linalg.eigvals(J)
        max_real_eig = np.max(np.real(eigs))
        leading_eigs.append(max_real_eig)
    return np.array(leading_eigs)