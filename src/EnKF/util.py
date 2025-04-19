import numpy as np

def generate_observation_matrix(n_grid:int, n_obs_point:int):
    H = np.zeros((n_obs_point, n_grid * n_grid))
    
    indice = [i for i in range(n_grid * n_grid)]
    obs_indice = np.random.choice(indice, n_obs_point, replace = False)
    
    for idx, obs_idx in enumerate(obs_indice):
        H[idx, obs_idx] = 1.0
    
    return H