import numpy as np
import matplotlib.pyplot as plt

"""
%load_ext autoreload
%autoreload 2
import AoI_Delay as ad
"""
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

def config_theory():
    config = dict(
        d_points = 100,
        d_type = 'uniform',
        max_d_delay = 0.08,
        t_points = 100,
        t_type = 'uniform',
        max_t_delay = 0.08,
        M_list = [4, 5]
    )
    
    return config

def compute_aoi_multiple_value_theory():
    config = config_theory()

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

