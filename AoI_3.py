import numpy as np
import matplotlib.pyplot as plt


def simulation(N, T, p_tx, alpha):
    current_age = np.zeros(N)
    age_list = []
    idx_tx = 0

    for t in range(T):
        current_age += 1
        
        # With probability alpha reset the AoI of all the sensor
        if np.random.rand(1) < alpha: 
            current_age[:] = 0
        
        # Reset the AoI of the sensor that do the transmission 
        current_age[idx_tx] = 0
        
        # Advance the transmission index
        idx_tx += 1
        if idx_tx >= N: idx_tx = 0
        
        # Save the current age
        age_list.append(current_age.copy())
    
    return np.asarray(age_list)

#%% Settings
N = 10
T = 1000

p_tx = 10
alpha = 0.1


#%% Simulation (SINGLE SET OF PARAMETER)

aoi_history = simulation(N, T, p_tx, alpha)
mean_age = aoi_history.mean(0)
    
#%%

p_tx_array = np.geomspace(1e-2, 1, 100)
alpha_array = np.geomspace(1e-3, 1, 100)

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

#%%
from matplotlib import ticker, cm
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(p_tx_array, alpha_array, mean_age_surface, 
#                        cmap = cm.coolwarm, linewidth=0, antialiased=False)


fig, ax = plt.subplots()
# cs = ax.contourf(p_tx_array, alpha_array, mean_age_surface)
cs = ax.imshow(mean_age_surface)
cbar = fig.colorbar(cs)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('P tx')
ax.set_ylabel('alpha')

cbar.ax.set_ylabel('Average AoI')

# plt.imshow(mean_age_surface)