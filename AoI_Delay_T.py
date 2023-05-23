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
        t_points = 100,
        t_type = 'uniform',
        t_min_delay = 0.005,
        t_max_delay = 0.3,
        M_list = [4, 5, 8, 10],
        # M_list = np.arange(3, 20),
        # Parameter only for the simulation 
        use_sum_for_theory = True,
        compute_not_optimized = False,
        L = 500, # Number of simulation step. Used only for the simulation
        integration_type = 1, # used only for the simulation
        repeat_simulation = 100, # Number of time each simulation is repeated
        # Other
        print_var = False,
    )

    config['t_values'] = np.geomspace(config['t_min_delay'], config['t_max_delay'], config['t_points'])

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Theory section

@jit(nopython = True, parallel = False)
def aoi_theory_formula(M : int, T : float):
    
    Q = 1/(M + 1)

    aoi = Q + Q * M * T * (2 - T)
    aoi /= 2

    return aoi

@jit(nopython = True, parallel = False)
def aoi_theory_formula_NOT_OPTIMIZED(M : int, T : float):
    Q = 1/(M + 1)
    aoi = (Q * (1 + 2 * M * T)) / 2   

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
            y_i = (1 - T) / (M + 1)
        elif i == M:
            # y_i = (1 + (M - 1) * T) / M
            y_i = (1 + M * T) / (M + 1)

        y_array[i] = y_i

    return y_array

def compute_aoi_theory_multiple_value(config : dict):
    t_values = config['t_values']
    results = np.zeros((len(config['M_list']), len(t_values)))

    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        if config['print_var']: print("Theory M = {}".format(M))
        for j in range(len(t_values)):
            T = t_values[j]
            
            if config['compute_not_optimized']:
                results[i, j] = aoi_theory_formula_NOT_OPTIMIZED(M, T)
            else:
                if config['use_sum_for_theory']:
                    results[i, j] = aoi_theory_sum(M, T)
                else:
                    results[i, j] = aoi_theory_formula(M, T)

    return results
            

def compare_aoi_formula():
    config = get_config_computation()

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

    plt.xscale('log')
    plt.legend()
    plt.show()

def probability_overflow(M : int, T : float, t_type : str):
    if t_type == 'uniform':
        num = T * (M + 2) - 1
        # num = M * (1 - T)
        den = 2 * T * (M + 1)
        prob = num/den
    elif t_type == 'exponential':
        num = 1 + M * T
        den = (M + 1) * T
        prob = np.exp(-num/den)
    else:
        raise ValueError("Wrong distribution type")
    

    return prob

def compute_probability_overflow_multiple_value(config : dict):
    t_values = config['t_values']
    results = np.zeros((len(config['M_list']), len(t_values)))

    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        for j in range(len(t_values)):
            T = t_values[j]
            results[i, j] = probability_overflow(M, T, config['t_type'])           

    return results
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

@jit(nopython = True, parallel = False)
def simulation(L : int, M : int, average_t :float, t_type : str):
    y_array = compute_y_array(M, average_t)
    tx_instant_array = compute_transmission_instant(y_array)
    t_delay_array = sample_distribution(average_t, t_type, M)
    
    # Variable to save AoI
    current_aoi = 0
    aoi_history = np.zeros(L)
    
   # Variable used for the tx
    idx_tx_instant = 0
    current_tx_instant = tx_instant_array[0]
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
            if idx_tx_instant <= len(tx_instant_array) - 1:
                current_tx_instant = tx_instant_array[idx_tx_instant]
                tx_arrival = current_tx_instant + t_delay_array[idx_tx_instant]
            else:
                current_tx_instant = 1
                tx_arrival = 1e10
        
        # Save AoI for the current iteration of the simulation
        aoi_history[i] = current_aoi
        
        # Advance simulation of 1 step
        current_aoi += simulation_step
        simulation_time += simulation_step
    
    # print(tx_instant_array)
    return aoi_history, current_tx_instant

@jit(nopython = True, parallel = False)
def aoi_simulation(L : int, M : int, T : float, t_type: str, repeat_simulation : int = 1, integration_type : int = 0):
    tmp_results = np.zeros(repeat_simulation)
    for k in range(repeat_simulation):
        last_tx_instant = 0
        i = 0
        while last_tx_instant != 1:
            i+=1
            aoi_history, last_tx_instant = simulation(L, M, T, t_type)

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


def compute_aoi_simulation_multiple_value(config : dict):
    t_values = config['t_values']
    results = np.zeros((len(config['M_list']), len(t_values)))

    for i in range(len(config['M_list'])): # Iterate through number of tx
        M = config['M_list'][i]
        if config['print_var']: print("Simulation M = {}".format(M))
        for j in range(len(t_values)): # Iterate through different values of T delay
            T = t_values[j]

            results[i, j] = aoi_simulation(config['L'], M, T, config['t_type'], config['repeat_simulation'], config['integration_type'])

    return results

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Computation for paper

def aoi_vs_T():
    config_computation = get_config_computation()
    results = compute_aoi_theory_multiple_value(config_computation)
    
    config_plot = get_config_plot()
    plot_aoi_vs_T(results, config_computation, config_plot)

def aoi_vs_M():
    config_computation = get_config_computation()
    config_computation['t_values'] = [0.005, 0.02, 0.04, 0.08]
    config_computation['M_list'] = np.arange(3, 20)
    results = compute_aoi_theory_multiple_value(config_computation)

    config_plot = get_config_plot()
    plot_aoi_vs_M(results, config_computation, config_plot)

def theory_vs_simulation():
    config_computation = get_config_computation()
    results_theory = compute_aoi_theory_multiple_value(config_computation)
    results_sim = compute_aoi_simulation_multiple_value(config_computation)
    
    config_plot = get_config_plot()
    plot_theory_vs_sim_delay_T(results_theory, results_sim, config_computation, config_plot)

def optimized_vs_NOT_optimized():
    config_computation = get_config_computation()
    config_computation['compute_not_optimized'] = False
    results_optimized = compute_aoi_theory_multiple_value(config_computation)

    config_computation['compute_not_optimized'] = True 
    results_NOT_optimized = compute_aoi_theory_multiple_value(config_computation)
    
    config_plot = get_config_plot()
    plot_ratio = True
    plot_optimized_vs_NOT_optimized(results_optimized, results_NOT_optimized, config_computation, config_plot, plot_ratio)

def overflow():
    config_computation = get_config_computation()
    config_plot = get_config_plot()

    config_computation['t_type'] = 'exponential'
    results = compute_probability_overflow_multiple_value(config_computation)
    results[results < 0] = 0
    plot_probability_overflow(results, config_computation, config_plot)

    config_computation['t_type'] = 'uniform'
    results = compute_probability_overflow_multiple_value(config_computation)
    results[results < 0] = 0
    plot_probability_overflow(results, config_computation, config_plot)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot stuff

def get_config_plot():
    config = dict(
        figsize = (10, 8),
        use_log_scale = True,
        fontsize = 20,
        linewidth = 2.5,
        n_ticks = 7,
        save_fig = True,
        save_path = 'Plot/delay_T/',
    )

    return config

def plot_aoi_vs_T(results, config_computation, config_plot):
    t_values = config_computation['t_values']

    fig, ax = plt.subplots(figsize = config_plot['figsize'])
    plt.rcParams.update({'font.size': config_plot['fontsize']})

    linestyle_theory = ['solid', 'dotted', 'dashdot', 'dashed']
    for i in range(results.shape[0]):
        ax.plot(t_values, results[i], 
                label = 'M = {}'.format(config_computation['M_list'][i]),
                linewidth = config_plot['linewidth'], linestyle = linestyle_theory[i]
                )
    
    ax.legend()

    ax.set_ylabel("Average AoI")
    ax.set_xlabel("Average Delay")
    ax.set_xlim([t_values[0], t_values[-1]])
    if config_plot['use_log_scale']: ax.set_xscale('log') 

    ticks = np.round(np.geomspace(config_computation['t_min_delay'], config_computation['t_max_delay'], config_plot['n_ticks']), 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xticks(ticks, minor = True)
    ax.grid(True)
    
    fig.tight_layout()
    if config_plot['save_fig']: 
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_AoI_VS_T.png")
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_AoI_VS_T.eps")
    plt.show()


def plot_aoi_vs_M(results, config_computation, config_plot):
    M_values = config_computation['M_list']
    t_values = config_computation['t_values']

    fig, ax = plt.subplots(figsize = config_plot['figsize'])
    plt.rcParams.update({'font.size': config_plot['fontsize']})
    plt.rcParams['text.usetex'] = False

    linestyle_theory = ['solid', 'dotted', 'dashdot', 'dashed']
    for i in range(results.shape[1]):
        ax.plot(M_values, results[:, i], 
                label = r'E[T] = {}'.format(t_values[i]),
                linewidth = config_plot['linewidth'], linestyle = linestyle_theory[i]
                )
    ax.legend()

    ax.set_ylabel("Average AoI")
    ax.set_xlabel("Number of transmissions (M)")
    ax.set_xlim([M_values[0], M_values[-1]])
    ax.grid(True)
    
    fig.tight_layout()
    if config_plot['save_fig']: 
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_AoI_VS_M.eps")
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_AoI_VS_M.png")
    plt.show()

def plot_theory_vs_sim_delay_T(results_theory, results_sim, config_computation, config_plot):
    t_values = np.geomspace(config_computation['t_min_delay'], config_computation['t_max_delay'], config_computation['t_points'])

    fig, ax = plt.subplots(figsize = config_plot['figsize'])
    plt.rcParams.update({'font.size': config_plot['fontsize']})

    linestyle_theory = ['solid', 'dotted', 'solid']
    linestyle_sim = ['dashdot', 'dashed', 'solid']
    for i in range(results_theory.shape[0]):
        ax.plot(t_values, results_theory[i], 
                label = 'Theory M = {}'.format(config_computation['M_list'][i]),
                linewidth = config_plot['linewidth'], linestyle = linestyle_theory[i]
                )
        ax.plot(t_values, results_sim[i], 
                label = 'Simulation M = {}'.format(config_computation['M_list'][i]),
                linewidth = config_plot['linewidth'], linestyle = linestyle_sim[i]
                )
    
    ax.legend()

    ax.set_ylabel("Average AoI")
    ax.set_xlabel("Average Delay")
    ax.set_xlim([t_values[0], t_values[-1]])
    if config_plot['use_log_scale']: ax.set_xscale('log') 

    ticks = np.round(np.geomspace(config_computation['t_min_delay'], config_computation['t_max_delay'], config_plot['n_ticks']), 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xticks(ticks, minor = True)
    ax.grid(True)
    
    fig.tight_layout()
    if config_plot['save_fig']: 
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_Sim_VS_Theory_{}.eps".format(config_computation['t_type']))
        fig.savefig(config_plot['save_path'] + "Single_Delay_T_Sim_VS_Theory_{}.png".format(config_computation['t_type']))
    plt.show()

def plot_probability_overflow(results, config_computation, config_plot):
    M_values = config_computation['M_list']
    t_values = config_computation['t_values']

    fig, ax = plt.subplots(figsize = config_plot['figsize'])
    plt.rcParams.update({'font.size': config_plot['fontsize']})

    linestyle_theory = ['solid', 'dotted', 'dashdot', 'dashed']
    for i in range(results.shape[0]):
        ax.plot(t_values, results[i], 
                label = 'M = {}'.format(M_values[i]),
                linewidth = config_plot['linewidth'], linestyle = linestyle_theory[i]
                )
    ax.legend()

    ax.set_ylabel("Probability of overflow")
    ax.set_xlabel("Average Delay")
    ax.set_xlim([t_values[0], t_values[-1]])
    # ax.set_ylim([0, 1])
    ax.grid(True)
    
    fig.tight_layout()
    if config_plot['save_fig']: 
        fig.savefig(config_plot['save_path'] + "probability_fail_tx_{}.eps".format(config_computation['t_type']))
        fig.savefig(config_plot['save_path'] + "probability_fail_tx_{}.png".format(config_computation['t_type']))
    plt.show()

def plot_optimized_vs_NOT_optimized(results_optimized, results_NOT_optimized, config_computation, config_plot, plot_ratio = False):

    M_list = config_computation['M_list']
    t_values = config_computation['t_values']

    fig, ax = plt.subplots(figsize = config_plot['figsize'])
    plt.rcParams.update({'font.size': config_plot['fontsize']})

    linestyle = ['solid', 'dotted', 'dashdot', 'dashed']
    for i in range(results_optimized.shape[0]):
        if plot_ratio:
            ax.plot(t_values,  results_NOT_optimized[i] / results_optimized[i],
                    label = "M = {}".format(M_list[i]), linestyle = linestyle[i])
        else:
            ax.plot(t_values, results_optimized[i], 
                    label = "Optimized M = {}".format(M_list[i]))
            ax.plot(t_values, results_NOT_optimized[i], 
                    label = "Not Optimized M = {}".format(M_list[i]))

    ax.legend()
    if plot_ratio:
        ax.set_ylabel("Not optimized/Optimized AoI ratio")
    else:
        ax.set_ylabel("Average AoI")
    ax.set_xlabel("Average Delay")
    ax.set_xlim([t_values[0], t_values[-1]])
    if config_plot['use_log_scale']: ax.set_xscale('log') 

    ticks = np.round(np.geomspace(config_computation['t_min_delay'], config_computation['t_max_delay'], config_plot['n_ticks']), 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xticks(ticks, minor = True)
    ax.grid(True)
    
    fig.tight_layout()

    if config_plot['save_fig']: 
        if plot_ratio:
            title = config_plot['save_path'] + "optimized_vs_NOT_optimized_ratio" 
        else:
            title = config_plot['save_path'] + "optimized_vs_NOT_optimized" 
        fig.savefig(title + ".png")
        fig.savefig(title + ".eps")

    plt.show()
