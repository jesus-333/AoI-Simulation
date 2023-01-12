# Imports
import numpy as np
import matplotlib.pyplot as plt

#%% Simulation function

def simulation(N, T, p_tx, alpha):
    current_age = np.zeros(N)
    age_list = []
    idx_tx = 0

    for t in range(T):
        current_age += 1

        # Reset the AoI of the sensor IF do the transmission during its turn
        if np.random.rand(1) < p_tx:
            current_age[idx_tx] = 0

            # With probability alpha reset the AoI of all the sensor
            if np.random.rand(1) < alpha:
                current_age[:] = 0

        # Advance the transmission index
        idx_tx += 1
        if idx_tx >= N: idx_tx = 0

        # Save the current age
        age_list.append(current_age.copy())

    return np.asarray(age_list)

#%% Settings
N = 4
T = 500

p_tx = 10
alpha = 0.1


#%% Simulation (SINGLE SET OF PARAMETER)

aoi_history = simulation(N, T, p_tx, alpha)
mean_age = aoi_history.mean(0)
print(mean_age)
    
#%%

p_tx_array = np.geomspace(1e-2, 0.3, 1920)
alpha_array = np.geomspace(1e-3, 1, 1080)

mean_age_list = []
mean_age_surface = np.zeros((len(p_tx_array), len(alpha_array)))

for i in range(len(p_tx_array)):
    p_tx = p_tx_array[i]
    for j in range(len(alpha_array)):
        alpha = alpha_array[j]
        aoi_history = simulation(N, T, p_tx, alpha)
        
        mean_age_list.append(aoi_history.mean(0))
        
        mean_age_surface[i, j] = mean_age_list[-1].mean()
                    
    print(round((i + 1)/len(p_tx_array) * 100, 2))

mean_age = np.asarray(mean_age_list)

#%% Plot results

def plot_aoi_surface(p_tx_array, alpha_array, aoi_surface,
                     plot_type = 0, use_log_scale = True, figsize = (15, 10)):
    extent = [p_tx_array[0], p_tx_array[-1], alpha_array[0], alpha_array[-1]]
    
    fig, ax = plt.subplots(figsize = figsize)
    if plot_type == 0: cs = ax.contourf(p_tx_array, alpha_array, aoi_surface.T, levels = 100)
    if plot_type == 1: cs = ax.imshow(aoi_surface.T, extent = extent, origin = 'lower')
    cbar = fig.colorbar(cs)
    
    if use_log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    ax.set_xlabel('P tx')
    ax.set_ylabel('alpha')
    
    cbar.ax.set_ylabel('Average AoI')
    plt.show()

plot_aoi_surface(p_tx_array, alpha_array, mean_age_surface, 0)
plot_aoi_surface(p_tx_array, alpha_array, mean_age_surface, 1)
