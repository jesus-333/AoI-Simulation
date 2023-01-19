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
    Simple simulation where each sensor can transmit only during its turn according to a transmission probability

    N = number of sensors
    T = unit of time to simulate. (i.e. the simulation advance with a discrete step and each iteration corresponde to a single unit of time)
    p_tx = probability of transmission for the various sensor
    alpha = probability that a transmission of a sensor reset the AoI of all sensors
    """

    current_age = np.zeros(N)
    age_list = []
    idx_tx = 0

    if type(p_tx) == float: p_tx = np.ones(N) * p_tx

    for t in range(T):
        current_age += 1

        # Reset the AoI of the sensor IF do the transmission during its turn
        if np.random.rand(1) < p_tx[idx_tx]:
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


def simulation_V2():
    """
    Simulation where each sensor can transmit only during its turn according to a transmission probability
    In this simulation the transmission probability is a increasing function of the AoI, i.e. higher the AoI higher the transmission probability

    N = number of sensors
    T = unit of time to simulate. (i.e. the simulation advance with a discrete step and each iteration corresponde to a single unit of time)
    p_tx = probability of transmission for the various sensor
    alpha = probability that a transmission of a sensor reset the AoI of all sensors
    """

    current_age = np.zeros(N)
    age_list = []
    idx_tx = 0

    if type(p_tx) == float: p_tx = np.ones(N) * p_tx

    for t in range(T):
        current_age += 1

        # Reset the AoI of the sensor IF do the transmission during its turn
        if np.random.rand(1) < p_tx[idx_tx]:
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

def get_simualtion_config():
    config = dict(
        N = 10,
        T = 500,
        p_tx_array = np.geomspace(1e-2, 0.3, 400),
        alpha_array = np.geomspace(1e-3, 1, 200),
        print_var = True
    )

    return config


def simulate_multiple_parameters():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']

    mean_age_list = []
    mean_age_surface = np.zeros((len(p_tx_array), len(alpha_array)))

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]
            aoi_history = simulation(config['N'], config['T'], p_tx, alpha)
            
            mean_age_list.append(aoi_history.mean(0))
            
            mean_age_surface[i, j] = mean_age_list[-1].mean()
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    mean_age = np.asarray(mean_age_list)

    return mean_age_surface, mean_age

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot function

def get_plot_config():
    config = dict(
        use_imshow = True,
        add_color_bar = True
    )

    return config

def plot_AoI_surface(p_tx_array, alpha_array, mean_age_surface):
    config = get_plot_config()

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Main function

def main():
    pass


if __name__ == "__main__":
    main()
