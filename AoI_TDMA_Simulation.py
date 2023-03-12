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
import time
import multiprocessing as mp
from numba import jit, prange

"""
%load_ext autoreload
%autoreload 2

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Simulation function

@jit(nopython = True, parallel = True)
def simulation(N, T, p_tx, alpha):
    """
    Simple simulation where each sensor can transmit only during its turn according to a transmission probability

    N = number of sensors
    T = unit of time to simulate. (i.e. the simulation advance with a discrete step and each iteration corresponde to a single unit of time)
    p_tx = probability of transmission for the various sensor
    alpha = probability that a transmission of a sensor reset the AoI of all sensors
    """

    current_age = np.zeros(N)
    age_list = np.zeros((N, T))
    idx_tx = 0

    # if type(p_tx) == float: p_tx = np.ones(N) * p_tx
    p_tx = np.ones(N) * p_tx

    for t in prange(T):
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
        """ age_list.append(current_age.copy()) """
        age_list[:, t] = current_age
    
    return age_list


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
        T = 15000,
        # p_tx_array = np.geomspace(1e-2, 0.3, 120),
        alpha_array = np.geomspace(1e-3, 1, 90),
        # N_array = np.linspace(1, 60, 60),
        N_array = [10, 30],
        p_tx_array = [0.01, 0.05],
        # alpha_array = [0.1],
        print_var = True,
        n_iterations = 50000,
    )

    return config

def simulate_multiple_parameters_V1():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']
    N_array = config['N_array']

    mean_age_list = []
    mean_age_array = np.zeros((len(p_tx_array), len(alpha_array), len(N_array)))

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]
            for k in range(len(N_array)):
                N = int(N_array[k])
                
                # Compute the simulation
                aoi_history = simulation(N, config['T'], p_tx, alpha)
                
                # Compute the mean AoI for this set of parameter and save the results
                mean_age_list.append(aoi_history.mean(0))
                mean_age_array [i, j, k] = mean_age_list[-1].mean()
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    mean_age = np.asarray(mean_age_list)

    return mean_age_array, mean_age

def simulate_multiple_parameters_V2():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']
    N_array = config['N_array']

    mean_age_list = []
    mean_age_array = np.zeros((len(p_tx_array), len(alpha_array), len(N_array)))

    increase_function = lambda x: x + 0.01 if x < 1 else 1

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]
            for k in range(len(N_array)):
                N = int(N_array[k])

                # Compute the simulation
                aoi_history = simulation_V2(N, config['T'], p_tx, alpha, increase_function)
                
                # Compute the mean AoI for this set of parameter and save the results
                mean_age_list.append(aoi_history.mean(0))
                mean_age_array[i, j, k] = mean_age_list[-1].mean()
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    mean_age = np.asarray(mean_age_list)

    return mean_age_array, mean_age

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

    B = (alpha * p * (1 - p)) / q
    C = ((q - (q ** N)) / (1 - q)) * N
    D = ( (N - 1) * (q ** (N + 1)) - (N * (q ** N)) + q) / ((1 - q) ** 2)

    A2 = B * (((C * r) / ((1 - r) ** 2)) + (D / (1 - r)))
    
    return A2

def compute_AoI_theory_multiple_parameter(compute_with_partial_sum = False):
    config = get_simualtion_config()
    
    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']
    N_array = config['N_array']

    mean_age_array = np.zeros((len(p_tx_array), len(alpha_array), len(N_array)))

    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in range(len(alpha_array)):
            alpha = alpha_array[j]
            for k in range(len(N_array)):
                N = N_array[k]
            
                # Compute the mean AoI for this set of parameter and save the results
                if compute_with_partial_sum:
                    mean_age_array [i, j, k] = compute_AoI_theory_partial_sum(p_tx, N, alpha, config['n_iterations'])
                else:
                    mean_age_array [i, j, k] = compute_AoI_theory_closed_form(p_tx, N, alpha)
        
        if config['print_var']:
            print(round((i + 1)/len(p_tx_array) * 100, 2))

    return mean_age_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Parallel function

def create_args_for_parallel(p_tx_array, q_array, N_array, n_iteration, add_index = True):
    args = []

    for i in range(len(N_array)):
        N = N_array[i]
        for j in range(len(p_tx_array)):
            p_tx = p_tx_array[j]
            for k in range(len(q_array)):
                q = q_array[k]
                if add_index: tmp_args = [N, n_iteration, p_tx, q, i, j, k]
                else: tmp_args = [N, n_iteration, p_tx, q]

                args.append(tmp_args)

    return args

def simulate_multiple_parameters_V1_parallel():
    config = get_simualtion_config()

    p_tx_array = config['p_tx_array']
    alpha_array = config['alpha_array']
    N_array = config['N_array']

    return __simulate_multiple_parameters_V1_parallel(p_tx_array, alpha_array, N_array, config['T'])

# NOT WORKING 
@jit(nopython = True)
def __simulate_multiple_parameters_V1_parallel(p_tx_array, alpha_array, N_array, T):
    mean_age_list = []
    mean_age_array = np.zeros((len(p_tx_array), len(alpha_array), len(N_array)))
    
    for i in range(len(p_tx_array)):
        p_tx = p_tx_array[i]
        for j in prange(len(alpha_array)):
            alpha = alpha_array[j]
            for k in prange(len(N_array)):
                N = int(N_array[k])
                
                # Compute the simulation
                aoi_history = simulation(N, T, p_tx, alpha)
                
                # Compute the mean AoI for this set of parameter and save the results
                mean_age_list.append(aoi_history.mean(0))
                mean_age_array [i, j, k] = mean_age_list[-1].mean()

    mean_age = np.asarray(mean_age_list)

    return mean_age_array, mean_age

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Main function

def test_simulation(multiprocess = False):
    p_tx_array = [5 * 1e-3, 5 * 1e-2, 1e-1]
    alpha_array = np.geomspace(1e-5, 1, 200)
    N_array = [1, 5, 10, 25]
    n_iteration = 5000

    if multiprocess:
        args = create_args_for_parallel(p_tx_array, alpha_array, N_array, n_iteration, False)
        
        st = time.time()
        with mp.Pool(processes = mp.cpu_count()) as pool:
            results = pool.starmap(simulation, args)
        print("Time simulation: {} (multiprocess)".format(time.time() - st)) 
    else:
        results = np.zeros((len(N_array), len(p_tx_array), len(alpha_array)))

        st = time.time()
        for i in range(len(N_array)):
            N = N_array[i]
            for j in range(len(p_tx_array)):
                p_tx = p_tx_array[j]

                for k in range(len(alpha_array)):
                    alpha = alpha_array[k]

                    tmp_result = simulation(N, n_iteration, p_tx, alpha)
                    results[i,j,k] = tmp_result.mean()

        print("Total time: {} (single thread)".format(time.time() - st))


def main():
    pass


if __name__ == "__main__":
    main()

