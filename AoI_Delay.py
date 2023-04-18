import numpy as np
import matplotlib.pyplot as plt
# from numba import jit, prange

"""
%load_ext autoreload
%autoreload 2
import AoI_Delay as ad
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def config_settings():
    config = dict(
        d_points = 100,
        d_type = 'exponential',
        max_d_delay = 0.08,
        t_points = 10,
        t_type = 'uniform',
        max_t_delay = 0,
        M_list = [2, 5],
        L = 500, # used only for the simulation
        integration_type = 0, # used only for the simulation
    )
    
    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Implementation of theory formulas

def aoi_delay(Q : float, M : int, d : float, t : float, d_type : str, t_type : str):
    aoi_1 = aoi_triangle(Q, M, d, t, d_type, t_type)
    aoi_2 = aoi_rectangle(Q, M, d, t, d_type, t_type)

    return aoi_1 + aoi_2

    # aoi = (M + 1) * (Q ** 2)
    #
    # if d_type == 'uniform': aoi += 2 * M * (4/3) * (expected_value(d) ** 2)
    # elif d_type == 'exponential': aoi += 2 * M * 2 * (expected_value(d) ** 2)
    # else: raise ValueError("Error d_type")
    #
    # aoi -= (M + 1) * (expected_value(d) ** 2)
    # aoi /= 2
    #
    # return aoi

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
    
    if config['max_d_delay'] != 0 and config['max_d_delay'] > 0.005:
        d_values = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    else: d_values = np.zeros(2)
    if config['max_t_delay'] != 0 and config['max_t_delay'] > 0.005:
        t_values = np.geomspace(0.005, config['max_t_delay'], config['t_points'])
    else: t_values = np.zeros(2)

    results = np.zeros((len(config['M_list']), config['d_points'], config['t_points']))
    
    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        Q = 1 / (M + 1)
        for j in range(len(d_values)):
            d = d_values[j]
            for k in range(len(t_values)):
                t = t_values[k]
                
                results[i, j, k] = aoi_delay(Q, M, d, t, config['d_type'], config['t_type'])

    # plt.plot(d_values, results[0, :, 0])
    # plt.plot(d_values, results[1, :, 0])
    # plt.xscale('log')
    # plt.xlim([0.005, 0.08])
    # plt.show()

    return results, d_values, t_values

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
                
                aoi = simulation(config['L'], N_tx, d, config['d_type'], t, config['t_type'])
                if integration_type == 0: aoi_average = np.mean(aoi)
                elif integration_type == 1: aoi_average = np.trapz(aoi, np.linspace(0, 1, len(aoi)))

                results[i, j, k] = aoi_average

    return results, d_values, t_values

# @jit(nopython = True, parallel = False)
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

    d_delay = sample_distribution(average_d, d_type)
    t_delay = sample_distribution(average_t, t_type)
    current_interval = transmission_interval[idx_current_tx] + d_delay + t_delay
    
    simulation_step = 1 / L

    for i in range(L):
        current_aoi += simulation_step
        current_interval -= simulation_step

        if current_interval < 0: # i.e. the transmission has taken place
            # This value is used to correct the results given by the discrete nature of the simulation
            adjustment_value = np.abs(current_interval)
            current_aoi = t_delay + adjustment_value
            
            # Sample new values for the activation and transmission delay
            d_delay = sample_distribution(average_d, d_type)
            t_delay = sample_distribution(average_t, t_type)
            
            # Advance the idx of the current tx
            idx_current_tx += 1
            
            # Compute the new interval
            if idx_current_tx <= len(transmission_interval) - 1: current_interval = transmission_interval[idx_current_tx] + d_delay + t_delay
            else: current_interval = 10000

        aoi_history[i] = current_aoi

    return aoi_history

def sample_distribution(distribution_average : float, distribution_type : str):
    """
    Sample from a random variable
    distribution_average: average value (i.e. expected_value) of the distribution
    distribution_type: string that specify the distribution type. It can only have value uniform or exponential
    """
    
    if distribution_type == 'uniform': x = np.random.uniform(low = 0, high = distribution_average)
    elif distribution_type  == 'exponential': x = np.random.exponential(scale = distribution_average)
    else: raise ValueError("Wrong distribution type")

    return x

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

# import scipy.integrate as integrate
# def compute_average(aoi_history):
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    config =  config_settings()

    config['d_type'] = 'uniform'
    results_theory_uniform, _, _ = compute_aoi_multiple_value_theory(config)
    
    config['d_type'] = 'exponential'
    results_theory_exponential, _, _ = compute_aoi_multiple_value_theory(config)

    config['d_type'] = 'uniform'
    results_simulation = np.zeros(results_theory_uniform.shape)
    n_simulation = 10 
    for i in range(n_simulation):
        tmp_results_simulation, _, _ = compute_aoi_multiple_value_simulation(config)
        results_simulation += tmp_results_simulation

    results_simulation /= n_simulation
    
    x = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    # plt.plot(x, results_theory_uniform[0, :, 0], label = 'uniform M = {}'.format(config['M_list'][0]))
    # plt.plot(x, results_theory_uniform[1, :, 0], label = 'uniform M = {}'.format(config['M_list'][1]))
    plt.plot(x, results_theory_exponential[0, :, 0], label = 'exponential M = {}'.format(config['M_list'][0]))
    plt.plot(x, results_theory_exponential[1, :, 0], label = 'exponential M = {}'.format(config['M_list'][1]))
    plt.plot(x, results_simulation[0, :, 0], label = 'simulation M = {}'.format(config['M_list'][0]))
    plt.plot(x, results_simulation[1, :, 0], label = 'simulation M = {}'.format(config['M_list'][1]))
    
    plt.xscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
