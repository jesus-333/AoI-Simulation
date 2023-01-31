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

    # if type(p_tx) == float: p_tx = np.ones(N) * p_tx
    p_tx = np.ones(N) * p_tx

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


def simulation_V2(N, T, initial_p_tx, alpha, increase_function):
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

    # if type(initial_p_tx) == float: p_tx = np.ones(N) * initial_p_tx
    # else: raise ValueError("initial_p_tx must be a float between 0 and 1")
    p_tx = np.ones(N) * initial_p_tx
    
    for t in range(T):
        current_age += 1

        
        if np.random.rand(1) < p_tx[idx_tx]: # Transmission 
            # Reset the AoI of the sensor IF do the transmission during its turn    
            current_age[idx_tx] = 0

            # With probability alpha reset the AoI of all the sensor
            if np.random.rand(1) < alpha:
                current_age[:] = 0
            
            # Reset the transmission probability
            p_tx = np.ones(N) * initial_p_tx
        else: # No transmission
            new_p_tx = increase_function(p_tx[0])
            p_tx = np.ones(N) * new_p_tx

        # Advance the transmission index
        idx_tx += 1
        if idx_tx >= N: idx_tx = 0

        # Save the current age
        age_list.append(current_age.copy())

    return np.asarray(age_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation settings and multiple parameter simulation

def get_simualtion_config():
    config = dict(
        N = 10,
        T = 18000,
        p_tx_array = np.geomspace(1e-2, 0.3, 100),
        alpha_array = np.geomspace(1e-3, 1, 50),
        print_var = True
    )

    return config


def simulate_multiple_parameters_V1():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']

    mean_age_list = []
    mean_age_surface = np.zeros((len(p_tx_array), len(alpha_array)))

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]

            # Compute the simulation
            aoi_history = simulation(config['N'], config['T'], p_tx, alpha)
            
            # Compute the mean AoI for this set of parameter and save the results
            mean_age_list.append(aoi_history.mean(0))
            mean_age_surface[i, j] = mean_age_list[-1].mean()
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    mean_age = np.asarray(mean_age_list)

    return mean_age_surface, mean_age

def simulate_multiple_parameters_V2():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']

    mean_age_list = []
    mean_age_surface = np.zeros((len(p_tx_array), len(alpha_array)))

    increase_function = lambda x: x + 0.01 if x < 1 else 1

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]

            # Compute the simulation
            aoi_history = simulation_V2(config['N'], config['T'], p_tx, alpha, increase_function)
            
            # Compute the mean AoI for this set of parameter and save the results
            mean_age_list.append(aoi_history.mean(0))
            mean_age_surface[i, j] = mean_age_list[-1].mean()
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    mean_age = np.asarray(mean_age_list)

    return mean_age_surface, mean_age

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theoretical model (PARTIAL SUM)

def compute_AoI_theory_partial_sum(p_tx, N, alpha, n_iterations = 2000):
    p = p_tx
    
    A1 = compute_A1_partial_sum(p, N, alpha, n_iterations)
    
    A2 = compute_A2_partial_sum(p, N, alpha, n_iterations)
    
    aoi = A1 + A2

    return aoi

def compute_A1_partial_sum(p, N, alpha, n_iterations = 2000):
    A1 = 0
    i = 0
    while i < n_iterations:
        a1_i = ((1 - p) ** i) * ((1 - alpha * p) ** (i * (N - 1))) * p * N * i
        A1 += a1_i

        i += 1

    return A1

def compute_A2_partial_sum(p, N, alpha, n_iterations = 2000):
    A2 = 0
    j = 0
    while j < n_iterations:
        k = 1
        term_1 = alpha * p * ((1 - p) ** (j + 1))
        tmp_a2 = 0
        while k <= N - 1:
            term_2 = (1 - alpha * p) ** (j * N + k - 1)
            term_3 = j * N + k
            tmp_a2 += term_2 * term_3
            k += 1
 
        a2_j = term_1 * tmp_a2 
        A2 += a2_j
        j += 1
    
    return A2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theoretical model (CLOSED FORM)

def compute_AoI_theory_closed_form(p_tx, N, alpha):
    p = p_tx

    A1 = compute_A1_closed_form(p, N, alpha)

    A2 = compute_A2_closed_form(p, N, alpha)
    
    aoi = A1 + A2

    return aoi

def compute_A1_closed_form(p, N, alpha):
    q =  (1 - p) * ((1 - alpha * p) ** (N - 1))
    A1 = (N * p * q) / ((1 - q) ** 2)

    return A1

def compute_A2_closed_form(p, N, alpha):
    q = (1 - alpha * p)
    r = ((1 - p) * (q ** N))

    B = (alpha * p * (1 - p) * N) / q
    C = (q - (q ** N)) / (1 - q)
    D = ( (N - 1) * (q ** (N + 1)) - (N * (q ** N)) + q) / ((1 - q) ** 2)

    A2 = B * (((C * r) / ((1 - r) ** 2)) + (D / (1 - r)))
    
    return A2

def compute_AoI_theory_multiple_parameter(compute_with_partial_sum = False):
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']

    mean_age_surface = np.zeros((len(p_tx_array), len(alpha_array)))

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]

            
            # Compute the mean AoI for this set of parameter and save the results
            if compute_with_partial_sum:
                mean_age_surface[i, j] = compute_AoI_theory_partial_sum(p_tx, config['N'], alpha)
            else:
                mean_age_surface[i, j] = compute_AoI_theory_closed_form(p_tx, config['N'], alpha)
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    return mean_age_surface

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot function

def get_plot_config():
    config = dict(
        figsize = (15, 10),
        fontsize = 15,
        use_imshow = False,
        levels_countourf = 20,
        add_color_bar = True,
    )

    return config

def plot_AoI_surface(p_tx_array, alpha_array, mean_age_surface):
    # Get config and check them
    config = get_plot_config()
    if 'save_fig' not in config: config['save_fig'] = False
    if 'levels_countourf' not in config: config['levels_countourf'] = 10
    if 'add_color_bar' not in config: config['add_color_bar'] = True

    fig, ax = plt.subplots(figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})

    if config['use_imshow']:
        # extent = [xmin,xmax,ymin,ymax]
        extent = [p_tx_array[0], p_tx_array[-1], alpha_array[0], alpha_array[-1]]

        cs = ax.imshow(mean_age_surface, extent = extent, interpolation='nearest', aspect='auto')
        # ax.set_xticks(p_tx_array)
        # ax.set_yticks(alpha_array)
    else:
        cs = ax.contourf(p_tx_array, alpha_array, mean_age_surface, levels = 20)
    
        ax.set_yscale('log')
        ax.set_xscale('log')

    if config['add_color_bar']:
        cbar = fig.colorbar(cs)

    ax.set_xlabel('Transmission probability p')
    ax.set_ylabel('q')

    cbar.ax.set_ylabel('Average AoI')
    
    plt.tight_layout()
    plt.show()

    if config['save_fig']:
        name = "aoi_surface_N{}".format(N)

        file_type = 'png'
        filename = "{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'eps'
        filename = "{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Main function

def main():
    pass


if __name__ == "__main__":
    main()

    # a1_closed = np.zeros((len(p), len(alpha)))
    # a1_sum = np.zeros((len(p), len(alpha)))
    # a2_closed = np.zeros((len(p), len(alpha)))
    # a2_sum = np.zeros((len(p), len(alpha)))
    #
    # N = 10
    # for i in range(len(p)):
    #     for j in range(len(alpha)):
    #         a1_closed[i, j] = AoI.compute_A1_closed_form(p[i], N, alpha[j])
    #         a2_closed[i, j] = AoI.compute_A2_closed_form(p[i], N, alpha[j])
    #         a1_sum[i, j] = AoI.compute_A1_partial_sumsum(p[i], N, alpha[j])
    #         a2_sum[i, j] = AoI.compute_A2_partial_sum(p[i], N, alpha[j])
