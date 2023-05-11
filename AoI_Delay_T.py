"""
@Author: Alberto Zancanaro (Jesus)
@Organization: University of Padua

Simulation for the the AoI work with only propagation delay (T)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def config_settings():
    config = dict(
        t_points = 100,
        t_type = 'uniform',
        t_max_delay = 0.08,
        M_list = [2,3,4],
        use_sum_for_theory = False,
    )

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theory section

@jit(nopython = True, parallel = False)
def aoi_theory_formula(M : int, T : float):
    num = 1 + (M - 1) * (2 * T - (T ** 2))
    den = 2 * M

    aoi = num / den

    return aoi

@jit(nopython = True, parallel = False)
def aoi_theory_sum(M : int, T : float):
    aoi = 0
    y_array = compute_y_array(M, T)

    for i in range(M + 1):
        y_i = y_array[i]

        aoi += 0.5 * (y_i ** 2) 
        
        if i != M: aoi += T * y_i

    return aoi

@jit(nopython = True, parallel = False)
def compute_y_array(M : int, T : float):
    """
    Compute the inter-transmission interval
    M = Number of tx
    T = (average) value of the propagation delay
    """

    y_array = np.zeros(M + 1)

    for i in range(M + 1):
        if i != M:
            y_i = (1 - T) / M
        elif i == M:
            y_i = (1 + (M - 1) * T) / M

        y_array[i] = y_i

    return y_array

def compute_aoi_theory_multiple_value(config : dict):
    t_values = np.geomspace(0.005, config['t_max_delay'], config['t_points'])
    results = np.zeros((len(config['M_list']), len(t_values)))

    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        for j in range(len(t_values)):
            T = t_values[j]
            
            if config['use_sum_for_theory']:
                results[i, j] = aoi_theory_sum(M, T)
            else:
                results[i, j] = aoi_theory_formula(M, T)

    return results
            

def compare_aoi_formula():
    config = config_settings()

    config['use_sum_for_theory'] = False
    results_formula = compute_aoi_theory_multiple_value(config)

    config['use_sum_for_theory'] = True
    results_sum = compute_aoi_theory_multiple_value(config)
    
    t_values = np.geomspace(0.005, config['t_max_delay'], config['t_points'])
    
    M_list = config['M_list']
    for i in range(results_formula.shape[0]):
        plt.plot(t_values, results_formula[i], label = M_list[i])
        plt.plot(t_values, results_sum[i], label = M_list[i])
        # plt.plot(t_values, results_sum[i] / results_formula[i])

    # plt.xscale('log')
    plt.legend()
    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

def simulation(L : int, M : int, average_t :float, t_type : str):
    y_array = compute_y_array(M, average_t)
    tx_instant_array = compute_transmission_instant(y_array)
    t_delay_list = sample_distribution(average_t, t_type, M)
    
    # Variable to save AoI
    current_aoi = 0
    aoi_history = np.zeros(L)
    
    idx_tx_instant = 0
    current_tx_instant = tx_instant_array[0]
    tx_arrival = current_tx_instant + t_delay_list[0]
    
    # Variable used to measure time in the simulation
    simulation_step = 1 / L
    simulation_time = 0

    for i in range(L):

        if simulation_time >= tx_arrival:
            # Compute the correction factor due to the discrete time of the simulation
            correction_factor = simulation_time - (tx_arrival)

            # Reset the AoI
            current_aoi = t_delay_list[current_tx_instant] + correction_factor
            
            # Retrieve the next tx instant and corresponding delay
            idx_tx_instant += 1
            current_tx_instant = tx_instant_array[idx_tx_instant]
            tx_arrival = current_tx_instant + t_delay_list[idx_tx_instant]

        aoi_history.append(current_aoi)

        current_aoi += simulation_step
        simulation_time += simulation_step

@jit(nopython = True, parallel = False)
def compute_transmission_instant(y_array):
    """
    Convert the transmission interval in the corresponding transmission instant

    Note that y_j = t_(j+1) - t_j and t_0 = 0 and t_M = 1
    Due to this constraints the vector length of tx_instant_array will be len(y_array) - 2
    """
    tx_instant_array = np.zeros(len(y_array) - 2)

    for i in range(len(y_array) - 1):
        if i == 0: 
            t_i = y_array[i]
        else:
            t_i = y_array[i] + tx_instant_array[i - 1]

        tx_instant_array[i] = t_i

    return tx_instant_array

@jit(nopython = True, parallel = False)
def sample_distribution(distribution_average : float, distribution_type : str, size : int):
    """
    Sample from a random variable
    distribution_average: average value (i.e. expected_value) of the distribution
    distribution_type: string that specify the distribution type. It can only have value uniform or exponential
    """
    
    if distribution_type == 'uniform': x = np.random.uniform(float(0), 2 * distribution_average, size = size)
    elif distribution_type  == 'exponential': x = np.random.exponential(scale = distribution_average, size = size)
    else: raise ValueError("Wrong distribution type")

    return x

