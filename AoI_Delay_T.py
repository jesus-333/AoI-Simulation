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
    for i in range(M + 1):
        if i != M:
            y_i = (1 - T) / M
        elif i == M:
            y_i = (1 + (M - 1) * T) / M

        aoi += 0.5 * (y_i ** 2) 
        
        if i != M: aoi += T * y_i

    return aoi

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

def simulation():
    pass

@jit(nopython = True, parallel = False)
def compute_transmission_interval(N_tx : int, average_t : float):
    y_array = np.zeros(N_tx)

    for i in range(N_tx):
        if i != N_tx - 1:
            y_array[i] = Q + average_d + 2 * N_tx * Q * average_t
        else:
            y_array[i] = Q - 2 * Q * average_t
    
    return y_array

