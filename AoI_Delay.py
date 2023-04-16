import numpy as np
import matplotlib.pyplot as plt

"""
%load_ext autoreload
%autoreload 2
import AoI_Delay as ad
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def config_settings():
    config = dict(
        d_points = 100,
        d_type = 'uniform',
        max_d_delay = 0.08,
        t_points = 100,
        t_type = 'uniform',
        max_t_delay = 0.08,
        M_list = [4, 5],
        L = 1000, # Used only for the simulation
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
    False function. Used only to better write some terms in the calculation of the aoi
    """
    return x

def var(x, x_type):
    """
    Compute the variance as expected_value(x^2) - expected_value(x)^2
    """

    if x_type == 'uniform': e_x_square = (4/3) * x
    elif x_type == 'exponential': e_x_square = 2 * (x ** 2)
    else: raise ValueError("Wrong distribution type")

    return e_x_square - (x ** 2)


def compute_aoi_multiple_value_theory():
    config = config_settings()

    d_values = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    t_values = np.geomspace(0.005, config['max_t_delay'], config['t_points'])

    results = np.zeros((len(config['M_list']), config['d_points'], config['t_points']))
    
    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        Q = 1 / (M + 1)
        for j in range(len(d_values)):
            d = d_values[j]
            for k in range(len(t_values)):
                t = t_values[k]
                
                results[i, j, k] = aoi_delay(Q, M, d, t, config['d_type'], config['t_type'])

    plt.plot(d_values, results[0, :, 0])
    plt.plot(d_values, results[1, :, 0])
    plt.xscale('log')
    plt.xlim([0.005, 0.08])
    plt.show()

    return results, d_values, t_values

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

def compute_aoi_multiple_value_simulation():
    config = config_settings()

    d_values = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    t_values = np.geomspace(0.005, config['max_t_delay'], config['t_points'])

    results = np.zeros((len(config['M_list']), config['d_points'], config['t_points']))
    
    for i in range(len(config['M_list'])):
        N_tx = config['M_list'][i]
        for j in range(len(d_values)):
            d = d_values[j]
            for k in range(len(t_values)):
                t = t_values[k]
                
                results[i, j, k] = simulation(config['L'], N_tx,
                                              d, config['d_type'],
                                              t, config['t_type'])

    plt.plot(d_values, results[0, :, 0])
    plt.plot(d_values, results[1, :, 0])
    plt.xscale('log')
    plt.xlim([0.005, 0.08])
    plt.show()

    return results, d_values, t_values

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

    transmission_instants = compute_transmission_instants(N_tx, average_d, average_t)
    idx_current_tx = 0

    current_d = sample_distribution(average_d, d_type)
    current_t = sample_distribution(average_t, t_type)
    
    t_array = np.linspace(0, 1, L)

    for i in range(len(t_array)):
        t = t_array[i]

        current_aoi += t

        if t >= transmission_instants[idx_current_tx] + current_d + current_t: # i.e. the transmission has taken place
            # This value is used to correct the results given by the discrete nature of the simulation
            adjustment_value = t - (transmission_instants + current_d + current_t)
            current_aoi = current_t + adjustment_value
            
            # Sample new values for the activation and transmission delay
            current_d = sample_distribution(average_d, d_type)
            current_t = sample_distribution(average_t, t_type)
            
            # Advance the idx of the current tx
            idx_current_tx += 1
        
        aoi_history[i] = current_aoi

    return aoi_history.mean()

def sample_distribution(distribution_average : float, distribution_type : str):
    """
    Sample from a random variable
    distribution_average: average value (i.e. expected_value) of the distribution
    distribution_type: string that specify the distribution type. It can only have value uniform or exponential
    """

    if distribution_type == 'uniform': x = np.random.uniform(low = 0, high = 2 * distribution_average)
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