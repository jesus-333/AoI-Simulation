import numpy as np
import plot_aoi as plt_aoi
import matplotlib.pyplot as plt
import time
from numba import jit, prange

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

"""
%load_ext autoreload
%autoreload 2
import AoI_Delay as ad
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def config_settings():
    config = dict(
        d_points = 50,
        d_type = 'uniform',
        max_d_delay = 0.08,
        t_points = 50,
        t_type = 'uniform',
        max_t_delay = 0,
        M_list = [3, 4, 5],
        L = 500, # Number of simulation step. Used only for the simulation
        integration_type = 1, # used only for the simulation
    )
    
    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Implementation of theory formulas

def aoi_delay(Q : float, M : int, d : float, t : float, d_type : str, t_type : str):
    aoi_1 = aoi_triangle(Q, M, d, t, d_type, t_type)
    aoi_2 = aoi_rectangle(Q, M, d, t, d_type, t_type)

    return aoi_1 + aoi_2

def aoi_triangle(Q, M, d, t, d_type, t_type):
    """
    Calculate the part of AoI that depends on the triangles    
    """

    aoi = Q/2 
    aoi += M * var(d, d_type) 
    aoi += (2 * (Q ** 2) - 4 * Q) * (expected_value(t) ** 2)
    aoi += (M - 1) * var(t, t_type)
    aoi -= expected_value(d) * expected_value(t)
    aoi += expected_value(t) ** 2

    return aoi

def aoi_rectangle(Q, M, d, t, d_type, t_type):
    """
    Calculate the part of AoI that depends on the rectangle    
    """
    
    aoi = M * Q * expected_value(t)
    aoi += 2 * Q * expected_value(t) ** 2
    aoi -= expected_value(t) ** 2

    return aoi

def expected_value(x):
    """
    False function. Used only to write in a nicer way some terms in the calculation of the aoi
    """
    return x

def var(x, x_type):
    """
    Compute the variance as expected_value(x^2) - expected_value(x)^2
    """

    if x_type == 'uniform': e_x_square = (4/3) * (x ** 2)
    elif x_type == 'exponential': e_x_square = 2 * (x ** 2)
    else: raise ValueError("Wrong distribution type")

    return e_x_square - (x ** 2)


def compute_aoi_multiple_value_theory(config):
    """
    Compute the average AoI using the theoretical formula.
    The results are saved in a matrix of dimension possible_tx x D x T

    possible_tx = list with the number of transmission (e.g. [4,5,6,7,8] indicate that the results are computer for 4 transmissions, 5 transmissions etc)
    D = possible values of D delay
    T = possible values of T delay
    """
    
    # Check d_values
    if config['max_d_delay'] != 0 and config['max_d_delay'] > 0.005:
        d_values = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    else: d_values = np.zeros(2)
    
    # Check t_values
    if config['max_t_delay'] != 0 and config['max_t_delay'] > 0.005:
        t_values = np.geomspace(0.005, config['max_t_delay'], config['t_points'])
    else: t_values = np.zeros(2)
    
    # Matrix to save the results
    results = np.zeros((len(config['M_list']), config['d_points'], config['t_points']))
    
    for i in range(len(config['M_list'])): # Cycle through number of transmission
        M = config['M_list'][i]
        Q = 1 / (M + 1)
        for j in range(len(d_values)): # Cycle through possible value of D delay
            d = d_values[j]
            for k in range(len(t_values)): # Cycle through possible value of T delay
                t = t_values[k]
                
                results[i, j, k] = aoi_delay(Q, M, d, t, config['d_type'], config['t_type'])

    return results, d_values, t_values

def compute_theory_all_distribution():
    config =  config_settings()

    config['d_type'] = 'uniform'
    config['t_type'] = 'uniform'
    results_theory_uu, _, _ = compute_aoi_multiple_value_theory(config)
    
    config['d_type'] = 'exponential'
    config['t_type'] = 'exponential'
    results_theory_ee, _, _ = compute_aoi_multiple_value_theory(config)

    config['d_type'] = 'uniform'
    config['t_type'] = 'exponential'
    results_theory_ue, _, _ = compute_aoi_multiple_value_theory(config)

    config['d_type'] = 'uniform'
    config['t_type'] = 'uniform'
    results_theory_eu, _, _ = compute_aoi_multiple_value_theory(config)

    return results_theory_uu, results_theory_ee, results_theory_ue, results_theory_eu

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

def compute_aoi_multiple_value_simulation(config):

    if config['max_d_delay'] != 0 and config['max_d_delay'] > 0.005:
        d_values = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    else: d_values = np.zeros(2)
    if config['max_t_delay'] != 0 and config['max_t_delay'] > 0.005:
        t_values = np.geomspace(0.005, config['max_t_delay'], config['t_points'])
    else: t_values = np.zeros(2)
    
    results = np.zeros((len(config['M_list']), config['d_points'], config['t_points']))

    integration_type = config['integration_type']
    
    for i in range(len(config['M_list'])):
        N_tx = config['M_list'][i]
        for j in range(len(d_values)):
            d = d_values[j]
            for k in range(len(t_values)):
                t = t_values[k]
                current_interval = 0
                tentativi = 0
                while current_interval < 9000:
                    # If the last interval of the simulation has value 1000000 it means that all transmissions have been made
                    # If it has a small value means that some transmission remain.
                    # Thi while cylce iterate until all transmission are made

                    aoi, current_interval = simulation(config['L'], N_tx, d, config['d_type'], t, config['t_type'])
                    if integration_type == 0: aoi_average = np.mean(aoi)
                    elif integration_type == 1: aoi_average = np.trapz(aoi, np.linspace(0, 1, len(aoi)))
                    tentativi += 1

                results[i, j, k] = aoi_average
                # print(N_tx, d, t, tentativi)
                    
    return results, d_values, t_values

@jit(nopython = True, parallel = False)
def simulation(L : int, N_tx : int, average_d : float, d_type : str, average_t :float, t_type : str):
    """
    Simulation to compute the average AoI.
    L = lenghts in unit of time of the simulation.
    N_tx = Number of transmissions to be made
    average_d = Average value of the activation delay. Express as a percentage
    d_type = distribution of d
    average_t = Average value of the transmission delay. Express as a percentage
    t_type = distribution of t
    """
    
    current_aoi = 0
    aoi_history = np.zeros(L)

    transmission_interval = compute_transmission_instants(N_tx, average_d, average_t)
    idx_current_tx = 0

    d_delay_list = sample_distribution(average_d, d_type, N_tx)
    t_delay_list = sample_distribution(average_t, t_type, N_tx)
    current_interval = transmission_interval[idx_current_tx] + d_delay_list[idx_current_tx] + t_delay_list[idx_current_tx]
    
    simulation_step = 1 / L
    
    # PRINT DI DEBUG
    # print("L = ", L)
    # print("Simulation step = ", simulation_step)
    # print("M = ", N_tx)
    # print("Q = ", 1 / (N_tx + 1))
    # print("average d = ", average_d)
    # print("average t = ", average_t)
    # print("d_delay = ", d_delay)
    # print("t_delay = ", t_delay)
    # print("transmission_interval = ", transmission_interval)
    # print("d_type", d_type)
    # print("t_type", t_type, "\n")

    for i in range(L):
        current_aoi += simulation_step
        current_interval -= simulation_step

        if current_interval <= 0: # i.e. the transmission has taken place
            # This value is used to correct the results given by the discrete nature of the simulation
            adjustment_value = np.abs(current_interval)
            current_aoi = t_delay_list[idx_current_tx] + adjustment_value
            
            # Advance the idx of the current tx
            idx_current_tx += 1
            
            # Compute the new interval
            if idx_current_tx <= len(transmission_interval) - 1: 
                current_interval = transmission_interval[idx_current_tx] + d_delay_list[idx_current_tx] + t_delay_list[idx_current_tx]
            else: # IF the transmission are ended sets the next interval to a value too large to finish 
                current_interval = int(1000000)

        aoi_history[i] = current_aoi
    
    return aoi_history, current_interval

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

@jit(nopython = True, parallel = False)
def compute_transmission_instants(N_tx : int, average_d: float, average_t : float):
    Q = 1 / (N_tx + 1)
    y_array = np.zeros(N_tx)

    for i in range(N_tx):
        if i == 0:
            y_array[i] = Q - average_d - 2 * Q * average_t
        elif i == N_tx - 1:
            y_array[i] = Q + average_d + 2 * N_tx * Q * average_t
        else:
            y_array[i] = Q - 2 * Q * average_t
    
    return y_array

def average_multiple_simulation(n_simulation, config):
    results_simulation = np.zeros([len(config['M_list']), config['d_points'], config['t_points']]) 
    for i in range(n_simulation): 
        tmp_results_simulation, _, _ = compute_aoi_multiple_value_simulation(config) 
        results_simulation += tmp_results_simulation

    results_simulation /= n_simulation

    return results_simulation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    
    results_theory_uu, results_theory_ee, results_theory_ue, results_theory_eu = compute_theory_all_distribution()

    n_simulation = 50
    config =  config_settings()

    start = time.time()
    config['d_type'] = 'uniform'
    config['t_type'] = 'uniform'
    results_sim_uu  = average_multiple_simulation(n_simulation, config)
    print("Simulation time (uu): ", time.time() - start)
    
    start = time.time()
    config['d_type'] = 'exponential'
    config['t_type'] = 'exponential'
    results_sim_ee = average_multiple_simulation(n_simulation, config)
    print("Simulation time (ee): ", time.time() - start)

    plt_aoi.plot_delay_theory_vs_sim_single_delay(results_theory_uu, results_sim_uu, 'uniform', 'D', config)
    plt_aoi.plot_delay_theory_vs_sim_single_delay(results_theory_ee, results_sim_ee, 'exponential', 'D', config)

if __name__ == "__main__":
    main()
