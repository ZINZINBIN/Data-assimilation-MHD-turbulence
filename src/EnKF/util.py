import numpy as np

def generate_observation_matrix(n_grid:int, n_obs_point:int, return_all:bool = False):
    H = np.zeros((n_obs_point, n_grid * n_grid))
    
    indice = [i for i in range(n_grid * n_grid)]
    obs_indice = np.random.choice(indice, n_obs_point, replace = False)
    
    for idx, obs_idx in enumerate(obs_indice):
        H[idx, obs_idx] = 1.0
        
    if return_all:
        return H, obs_indice
    else:
        return H

def compute_l2_norm(quantity:np.ndarray, quantity_filtered:np.ndarray):
    l2_norm = np.linalg.norm(quantity - quantity_filtered)
    return l2_norm

def generate_observation_points(obs_indices:np.ndarray, X_mesh:np.ndarray, Y_mesh:np.ndarray):
    X_mesh_1D = X_mesh.ravel()
    Y_mesh_1D = Y_mesh.ravel()
    
    x_obs = X_mesh_1D[obs_indices]
    y_obs = Y_mesh_1D[obs_indices]
    return x_obs, y_obs