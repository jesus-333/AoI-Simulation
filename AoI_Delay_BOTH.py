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

"""
%load_ext autoreload
%autoreload 2
import AoI_Delay_T as ad
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_computation():
    config = dict(
        # Parameter for both theory and simulation
        d_points = 100,
        d_type = 'uniform',
        d_min_delay = 0.005,
        d_max_delay = 0.15,
        t_points = 100,
        t_type = 'uniform',
        t_min_delay = 0.005,
        t_max_delay = 0.15,
        M_list = [4, 5, 8, 10],
        # M_list = np.arange(3, 20),
        # Parameter only for the simulation 
        use_sum_for_theory = False,
        compute_not_optimized = False,
        L = 500, # Number of simulation step. Used only for the simulation
        integration_type = 1, # used only for the simulation
        repeat_simulation = 100, # Number of time each simulation is repeated
        # Other
        print_var = False,
    )

    config['d_values'] = np.geomspace(config['d_min_delay'], config['d_max_delay'], config['d_points'])
    config['t_values'] = np.geomspace(config['t_min_delay'], config['t_max_delay'], config['t_points'])

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theory Computation single values 

@jit(nopython = True, parallel = False)
def aoi_theory_formula(M : int, D : float, T : float, d_type : str):
    Q = 1/(M + 1)
    
    aoi = Q + (2 * M * var(D, d_type)) - (M * Q * (T**2)) + (2 * Q * M * T)
    aoi /= 2

    return aoi

@jit(nopython = True, parallel = False)
def aoi_theory_formula_NOT_OPTIMIZED(M : int, D : float, T : float):
    aoi = 0
    # TODO
    return aoi


@jit(nopython = True, parallel = False)
def aoi_theory_sum(M : int, D: float, T : float, d_type : str):
    aoi = 0
    y_array = compute_y_array(M, D, T)

    for i in range(M + 1):
        y_i = y_array[i]

        aoi += 0.5 * (y_i ** 2) 
        
        if i != M: aoi += T * y_i

    aoi += (y_array[0] - y_array[-1]) * D
    aoi += M * var(D, d_type) + (D ** 2) + D * T

    return aoi

@jit(nopython = True, parallel = False)
def compute_y_array(M : int, D : float, T : float):
    """
    Compute the inter-transmission interval
    M = Number of tx
    D = (average) value of the computation delay
    T = (average) value of the propagation delay
    """

    y_array = np.zeros(M + 1)
    Q = 1 / (M + 1)

    for i in range(M + 1):
        if i == 0: # y_0
            y_i = Q - D - (Q * T)
        elif i != M: # y_i
            y_i = Q * (1 - T)
        elif i == M: # y_m
            y_i = Q + D + ( M * Q * T )

        y_array[i] = y_i

    return y_array

@jit(nopython = True, parallel = False)
def var(x, x_type):
    """
    Compute the variance as expected_value(x^2) - expected_value(x)^2
    """

    if x_type == 'uniform': expected_value_squared = (4/3) * (x ** 2)
    elif x_type == 'exponential': expected_value_squared = 2 * (x ** 2)
    else: raise ValueError("Wrong distribution type")

    return expected_value_squared - (x ** 2)

def probability_overflow(M : int, D : float, T : float, d_type : str, t_type : str):
    # TODO 
    prob = -1
    if t_type == 'uniform':
        pass
    elif t_type == 'exponential':
        pass
    else:
        raise ValueError("Wrong distribution type")

    return prob

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theory. Computation multiple values and similar stuff

def compute_aoi_theory_multiple_value(config : dict):
    d_values = config['d_values']
    t_values = config['t_values']
    results = np.zeros((len(config['M_list']), len(d_values), len(t_values)))

    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        if config['print_var']: print("Theory M = {}".format(M))
        for j in range(len(d_values)):
            D = d_values[j]
            for k in range(len(t_values)):
                T = t_values[k]
                
                if config['compute_not_optimized']:
                    results[i, j, k] = aoi_theory_formula_NOT_OPTIMIZED(M, D, T)
                else:
                    if config['use_sum_for_theory']:
                        results[i, j, k] = aoi_theory_sum(M, D, T, config['d_type'])
                    else:
                        results[i, j, k] = aoi_theory_formula(M, D, T, config['d_type'])

    return results
            

def compare_aoi_formula():
    config = get_config_computation()

    config['use_sum_for_theory'] = False
    results_formula = compute_aoi_theory_multiple_value(config)

    config['use_sum_for_theory'] = True
    results_sum = compute_aoi_theory_multiple_value(config)
    
    d_values = np.geomspace(0.005, config['d_max_delay'], config['d_points'])
    t_values = np.geomspace(0.005, config['t_max_delay'], config['t_points'])
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))

    M_list = config['M_list']
    for i in range(results_formula.shape[0]):
        idx_0 = np.random.randint(results_formula.shape[2])
        ax[0].plot(d_values, results_formula[i, :, idx_0], label = M_list[i])
        ax[0].plot(d_values, results_sum[i, :, idx_0], label = M_list[i])
        ax[0].set_xlabel('D Delay')

        idx_1 = np.random.randint(results_formula.shape[1])
        ax[1].plot(t_values, results_formula[i, idx_1, :], label = M_list[i])
        ax[1].plot(t_values, results_sum[i, idx_1, :], label = M_list[i])
        ax[1].set_xlabel('T Delay')

    for i in range(len(ax)):
        ax[i].set_xscale('log')
        ax[i].grid(True)
        ax[i].legend()
    fig.tight_layout()
    fig.show()


def compute_probability_overflow_multiple_value(config : dict):
    t_values = config['t_values']
    d_values = config['d_values']
    results = np.zeros((len(config['M_list']), len(d_values), len(t_values)))

    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        for j in range(len(d_values)):
            D = d_values[j]
            for k in range(len(t_values)):
                T = t_values[k]
                results[i, j, k] = probability_overflow(M, D, T, config['d_type'], config['t_type'])           

    return results

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

@jit(nopython = True, parallel = False)
def simulation(L : int, M : int, average_d :float, d_type : str, average_t :float, t_type : str):
    y_array = compute_y_array(M, average_d, average_t)
    tx_instant_array = compute_transmission_instant(y_array)
    d_delay_array = sample_distribution(average_d, d_type, M)
    t_delay_array = sample_distribution(average_t, t_type, M)
    
    # Variable to save AoI
    current_aoi = 0
    aoi_history = np.zeros(L)
    
   # Variable used for the tx
    idx_tx_instant = 0
    current_tx_instant = tx_instant_array[0] + d_delay_array[0]
    tx_arrival = current_tx_instant + t_delay_array[0]
    
    # Variable used to measure time in the simulation
    simulation_step = 1 / L
    simulation_time = 0
    
    for i in range(L):
        if simulation_time >= tx_arrival:
            # Compute the correction factor due to the discrete time of the simulation
            correction_factor = simulation_time - tx_arrival
            
            # Reset the AoI
            current_aoi = t_delay_array[idx_tx_instant] + correction_factor
            
            # Retrieve the next tx instant and corresponding delay
            idx_tx_instant += 1
            if idx_tx_instant < len(tx_instant_array): # If there are transmissions to do
                current_tx_instant = tx_instant_array[idx_tx_instant] + d_delay_array[idx_tx_instant]
                tx_arrival = current_tx_instant + t_delay_array[idx_tx_instant]
            else: # If the transmissions are finished
                current_tx_instant = 1
                tx_arrival = 1e10

        
        # Save AoI for the current iteration of the simulation
        aoi_history[i] = current_aoi
        
        # Advance simulation of 1 step
        current_aoi += simulation_step
        simulation_time += simulation_step
    
    return aoi_history, current_tx_instant

@jit(nopython = True, parallel = False)
def aoi_simulation(L : int, M : int, D : float, d_type : str, T : float, t_type: str, repeat_simulation : int = 1, integration_type : int = 0):
    tmp_results = np.zeros(repeat_simulation)
    for k in range(repeat_simulation):
        last_tx_instant = 0
        i = 0
        while last_tx_instant != 1:
            i+=1
            aoi_history, last_tx_instant = simulation(L, M, D, d_type, T, t_type)

        if integration_type == 0: aoi_average = np.mean(aoi_history)
        elif integration_type == 1: aoi_average = np.trapz(aoi_history, np.linspace(0, 1, len(aoi_history)))

        tmp_results[k] = aoi_average

    return np.mean(tmp_results)

@jit(nopython = True, parallel = False)
def compute_transmission_instant(y_array):
    """
    Convert the transmission interval in the corresponding transmission instant

    Note that y_j = t_(j+1) - t_j and t_0 = 0 and t_M = 1
    """

    # Note that with decision the last element of the array will remain a 1.
    # In this case it can be used during the simulation to check if all the tx have been performed
    tx_instant_array = np.ones(len(y_array))

    for i in range(len(y_array) - 1):
        if i == 0: 
            t_i = y_array[i]
        else:
            t_i = y_array[i] + tx_instant_array[i - 1]

        tx_instant_array[i] = t_i
    
    # Return all the transmission instant apart from the last one that is a 1
    return tx_instant_array[0:-1]

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


def compute_aoi_simulation_multiple_value(config : dict, print_var = False):
    d_values = config['d_values']
    t_values = config['t_values']
    results = np.zeros((len(config['M_list']), len(d_values), len(t_values)))

    for i in range(len(config['M_list'])): # Iterate through number of tx
        M = config['M_list'][i]
        if config['print_var']: print("Simulation M = {}".format(M))
        for j in range(len(d_values)): # Iterate through different values of D delay
            D = d_values[j]
            for k in range(len(t_values)): # Iterate through different values of T delay
                T = t_values[k]
                if print_var: 
                    M_percentage = round((i + 1) / len(config['M_list']) * 100)
                    d_percentage = round((j + 1) / len(config['d_values']) * 100)
                    t_percentage = round((k + 1) / len(config['t_values']) * 100)
                    print("{}% M_list ({})\t {}% d_values ({})\t {}% t_values ({})".format(M_percentage, M, d_percentage, D, t_percentage, T))

                results[i, j, k] = aoi_simulation(config['L'], M, D, config['d_type'], T, config['t_type'], config['repeat_simulation'], config['integration_type'])
        
        # Save the results after each iteration through the M list
        with open('results_simulation.npy', 'wb') as f:
            np.save(f, results)

    return results