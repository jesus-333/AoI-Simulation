"""
@Author: Alberto Zancanaro (Jesus)
@Organization: University of Padua

Function used to simulate a system with N sensor transmit with a TDMA protocol
The simulation is used to track the AoI of the various sensors
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt


"""
%load_ext autoreload
%autoreload 2

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Simulation function

def simulation(N, T, p_tx, alpha):
    """
    N = number of sensors
    T = unit of time to simulate. (i.e. the simulation advance with a discrete step and each iteration corresponde to a single unit of time)
    p_tx = probability of transmission for the various sensor
    alpha = probability that a transmission of a sensor reset the AoI of all sensors
    """

    current_age = np.zeros(N)
    age_list = []
    idx_tx = 0

    for t in range(T):
        current_age += 1
        
        # Reset the AoI of the sensor that do the transmission 
        if np.random.rand(1) < p_tx[idx_tx]:
            current_age[idx_tx] = 0

        # With probability alpha reset the AoI of all the sensor (I.e. the transmission is usefull for all the sensor)
        if np.random.rand(1) < alpha: 
            current_age[:] = 0
        
        # Advance the transmission index
        idx_tx += 1
        if idx_tx >= N: idx_tx = 0
        
        # Save the current age
        age_list.append(current_age.copy())
    
    return np.asarray(age_list)


def simulate_multiple_value():
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot function

def plot_config():
    config = dict(
        use_imshow = True,
        add_color_bar = True
    )

    return config

def plot_AoI_surface(p_tx_array, alpha_array, mean_age_surface, config):
    fig, ax = plt.subplots()
    
    if config['use_imshow']:
        cs = ax.imshow(mean_age_surface)
        plt.xticks(p_tx_array)
        plt.yticks(alpha_array)
    else:
        cs = ax.contourf(p_tx_array, alpha_array, mean_age_surface)
    
    if config['add_color_bar']:
        cbar = fig.colorbar(cs)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Transmission probability p')
    ax.set_ylabel('alpha')

    cbar.ax.set_ylabel('Average AoI')
