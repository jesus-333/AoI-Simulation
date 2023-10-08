"""
@Author: Alberto Zancanaro (Jesus)
@Organization: University of Padua

Functions used to plot the AoI obtained from the simulation or other figures to insert in various paper
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

import AoI_Delay_BOTH

"""
%load_ext autoreload
%autoreload 2

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot config

def get_plot_config():
    config = dict(
        # - - - - - - - - - - - - -
        # Common parameters
        figsize = (15, 10),
        fontsize = 24,
        markersize = 11,
        markevery = 5,
        save_fig = True,
        # - - - - - - - - - - - - -
        # Parameters for aoi surface
        use_imshow = False,
        levels_countourf = 20,
        add_color_bar = True,
        # - - - - - - - - - - - - -
        # Parameters for aoi vs N
        labels = ["p = {}".format(p) for p in [0.005, 0.01, 0.02, 0.05]],
        idx_to_plot = [(i, 0) for i in range(4) ],
        linestyle = ['solid', 'dotted', 'dashed', 'dashdot']
        # - - - - - - - - - - - - -
        # Parameters for delay plot (globecom)
    )

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot function

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
    ax.set_ylabel('Probability $q$ of a useful update from a neighbor')

    cbar.ax.set_ylabel('Average AoI')
    
    fig.tight_layout()
    plt.tight_layout()

    if config['save_fig']:
        N = 10
        type_aoi = 'theory'
        name = "aoi_surface_N{}_{}".format(N, type_aoi)

        file_type = 'png'
        filename = "{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'eps'
        filename = "{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

    plt.show()


def plot_aoi_vs_N(N_array, aoi_array, std_array = None):
    config = get_plot_config()
    
    fig, ax = plt.subplots(1, 1, figsize = (15, 10)) 
    plt.rcParams.update({'font.size': config['fontsize']})
    
    for i in range(len(config['idx_to_plot'])):
        tmp_idx = config['idx_to_plot'][i]
        
        tmp_aoi = aoi_array[tmp_idx[0], tmp_idx[1], :]
        tmp_label = config['labels'][i] + " (mean)"
        ax.plot(N_array, tmp_aoi, 
                label = tmp_label, linestyle = config['linestyle'][i], linewidth = 2.5)
        
        if std_array is not None:
            tmp_label = config['labels'][i] + " (std)"
            tmp_std = std_array[tmp_idx[0], tmp_idx[1], :] 
            ax.fill_between(N_array, tmp_aoi + tmp_std, tmp_aoi - tmp_std, 
                            label = tmp_label, alpha = 0.25)

    ax.legend(ncol = 2)
    
    ax.set_xlabel("Number of neighbors")
    ax.set_ylabel("Average AoI")
    ax.grid(True)

    ax.set_xlim([N_array[0], N_array[-1]])
    ax.set_yscale('log')

    fig.tight_layout()
    fig.tight_layout()

    plt.show()

def plot_aoi_simulation_vs_theroy(alpha_array, aoi_array_theory, aoi_array_sim, std_array_sim = None):
    config = get_plot_config()
    
    fig, ax = plt.subplots(1, 1, figsize = (15, 10)) 
    plt.rcParams.update({'font.size': config['fontsize']})
    
    p_tx_array = [0.01, 0.05]
    N_array = [10, 30]
    
    color_closed_form = ['blue', 'black']
    color_simulation = ['green', 'red']

    for i in range(aoi_array_theory.shape[0]):
        tmp_idx_1 = i
        tmp_idx_2 = i
        
        tmp_label = "Closed form p = {} N = {}".format(p_tx_array[i], N_array[i])
        tmp_aoi = aoi_array_theory[tmp_idx_1, :, tmp_idx_2]
        ax.plot(alpha_array, tmp_aoi, 
                label = tmp_label, linestyle = config['linestyle'][i], linewidth = 2.5, color = color_closed_form[i])

        tmp_label = "Simulation p = {} N = {} (mean)".format(p_tx_array[i], N_array[i])
        tmp_aoi = aoi_array_sim[tmp_idx_1, :, tmp_idx_2]
        ax.plot(alpha_array, tmp_aoi, 
                label = tmp_label, linestyle = config['linestyle'][i + 2], linewidth = 2.5, color = color_simulation[i])
        
        if std_array_sim is not None:
            tmp_label = "Simulation p = {} N = {} (std)".format(p_tx_array[i], N_array[i])
            tmp_std = std_array_sim[tmp_idx_1, :, tmp_idx_2] 
            ax.fill_between(alpha_array, tmp_aoi + tmp_std, tmp_aoi - tmp_std, 
                            label = tmp_label, alpha = 0.25, color = color_simulation[i])

    ax.legend()
    
    ax.set_xlabel("Probability $q$ of a useful update from a neighbor")
    ax.set_ylabel("Average AoI")
    ax.grid(True)

    ax.set_xlim([alpha_array[0], alpha_array[-1]])
    ax.set_yscale('log')
    ax.set_xscale('log')

    fig.tight_layout()
    fig.tight_layout()

    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot AoI Evolution

def plot_aoi_evolution_TDMA(aoi):
    """
    Function that create the typical figure of AoI evolution presents in various paper (the plot with a sawtooth trend)
    
    The AoI input must be a vector where each value represents the AoI in that specific temporal moment
    E.g. aoi = [0,1,2,3,0,1,0,1,2,3,4,5,6,7,0,1, ....]
    """

    config = dict(
        figsize = (8, 6),
        fontsize = 18,
        x_limit = 16, # Points to plot (i.e. simulation step)
        y_limit = 8,
        linewidth = 1.7,
        N = aoi.shape[1], # Number of sensors
        plot_type = 1,
        label_setted_for_reset_correlation = False, # NOT MODIFY
        label_setted_for_reset_tx = False,          # NOT MODIFY
    )
    
    # Create ticks for the grid and labels
    config_ticks = dict(
        major_x_ticks = np.arange(0, config['x_limit'] + 1, 1),
        minor_x_ticks = np.arange(0, config['x_limit'] + 1, 1) + 0.5,
        major_y_ticks = np.arange(0, config['y_limit'], 1),
        minor_x_ticks_fontsize = 8,
    )
    config_ticks['x_ticks_labels'] = get_xticks_labels(config['N'], len(config_ticks['minor_x_ticks']))

    # Correct AoI
    aoi_correct = []
    t_array = []
    for i in range(aoi.shape[1]):
        tmp_aoi_correct, t = correct_aoi_for_evolution_plot(aoi[:, i])
        
        aoi_correct.append(np.asarray(tmp_aoi_correct))
        t_array.append(t)
    
    # Find point where AoI was reset
    reset_all_list, reset_single_list = check_aoi_for_reset(aoi, config['x_limit'])
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_list = ['black' for i in range(aoi.shape[1])]

    plt.rcParams.update({'font.size': config['fontsize']})
    
    if config['plot_type'] == 0: # Plot in 4 different plot and same figure
        n_figure_row = n_figure_column = int(np.ceil(np.sqrt(config['N'])))
        fig, ax_array = plt.subplots(n_figure_row, n_figure_column, figsize = config['figsize'])

        i = 0
        for ax in ax_array.flatten():
            config['title'] = config_ticks['x_ticks_labels'][i]
            config['label'] = 'AoI'
            config['color'] = color_list[i]
            reset_list = [reset_all_list, reset_single_list[i]]

            plot_aoi_evolution(ax, t_array[i], aoi_correct[i], reset_list, config, config_ticks)
            fig.tight_layout()
            i += 1

    elif config['plot_type'] == 1: # Plot in 4 different plot and different figure
        for i in range(aoi.shape[1]):
            fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
            config['title'] = config_ticks['x_ticks_labels'][i]
            config['label'] = ''
            config['color'] = color_list[i]
            config['label_setted_for_reset_correlation'] = False
            config['label_setted_for_reset_tx']          = False
            reset_list = [reset_all_list, reset_single_list[i]]

            plot_aoi_evolution(ax, t_array[i], aoi_correct[i], reset_list, config, config_ticks)
            fig.tight_layout()
            
            # fig.savefig('Plot/TDMA_aoi_evolution_separated/TDMA_aoi_evolution_separated_{}.png'.format(chr(65 + i)))
            # fig.savefig('Plot/TDMA_aoi_evolution_separated/TDMA_aoi_evolution_separated_{}.eps'.format(chr(65 + i)))
            fig.savefig('TMP/TDMA_aoi_evolution_separated_{}.png'.format(chr(65 + i)))
            fig.savefig('TMP/TDMA_aoi_evolution_separated_{}.eps'.format(chr(65 + i)))

    elif config['plot_type'] == 2: # Plot in same plot and same figure
        fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
        for i in range(aoi.shape[1]):
            config['title'] = config_ticks['x_ticks_labels'][i]
            config['label'] = 'AoI Sensor {}'.format(config_ticks['x_ticks_labels'][i])
            config['color'] = color_list[i]
            config['label_setted_for_reset_correlation'] = True if i != aoi.shape[1] - 1 else False
            config['label_setted_for_reset_tx']          = True if i != aoi.shape[1] - 1 else False
            reset_list = [reset_all_list, reset_single_list] # TODO CORRECT

            plot_aoi_evolution(ax, t_array[i], aoi_correct[i], reset_list, config, config_ticks)
            fig.tight_layout()


    else: raise ValueError("config['plot_type'] must have value 0 or 1 or 2")
    
    # Plot Transmission bar
    plot_bar_transmission(reset_all_list, reset_single_list, config, config_ticks)

    plt.show()

def plot_aoi_evolution(ax, t_array, aoi, reset_list, config, config_ticks):
    reset_all_list = reset_list[0]
    reset_single_list = reset_list[1]

    for i in range(int(config['x_limit'] / config['N'])):
        ax.axvline(i * config['N'], color = 'grey', alpha = 0.5)
    
    ax.plot(t_array, aoi, 
            linewidth = config['linewidth'], color = config['color'],
            label = config['label']
            )
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

    ax.set_yticks(config_ticks['major_y_ticks'])
    ax.set_xticks(config_ticks['major_x_ticks'])
    ax.set_xticks(config_ticks['minor_x_ticks'], minor = True)
    # ax.set_xticklabels(config_ticks['x_ticks_labels'], minor = True)
    # ax.tick_params(axis = 'x', which = 'minor',  labelsize = config_ticks['minor_x_ticks_fontsize'])
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    ax.set_xlim([0, config['x_limit']])
    ax.set_ylim([0, config['y_limit']])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    ax.set_xlabel("Time [Units of time]")
    ax.set_ylabel("AoI")
    # ax.set_title("Sensor {}".format(config['title']))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Plot point where AoI was reset

    # Through correlation
    for point in reset_all_list:
        x, y = point
        label = '' if config['label_setted_for_reset_correlation']  else 'Correlation reset'
        ax.scatter(x, y, marker = 'o', color = 'green', linewidths = 3, label = label, s = 60)
        config['label_setted_for_reset_correlation'] = True

    # Through Transmission
    for point in reset_single_list:
        x, y = point
        label = '' if config['label_setted_for_reset_tx']  else 'Transmission reset'
        ax.scatter(x, y, marker = 'o', linewidths = 2, label = label, facecolors='none', edgecolors = 'red', s = 100)
        config['label_setted_for_reset_tx'] = True

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    ax.grid(True)
    ax.legend()

def plot_bar_transmission(reset_all_list, reset_single_list, config, config_ticks):
    fig, ax = plt.subplots(1, 1, figsize = (10, 1))

    ax.set_yticks(config_ticks['major_y_ticks'])
    ax.set_yticklabels('')
    ax.set_xticks(config_ticks['major_x_ticks'])
    
    ax.grid(True)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, config['x_limit']])
    
    for point in reset_all_list:
        x, y = point
        y = 0.5
        ax.scatter(x, y, marker = 'o', color = 'green', linewidths = 3, s = 60)
    
    for i in range(len(reset_single_list)):
        tmp_list = reset_single_list[i]
        for point in tmp_list:
            x, y = point
            y = 0.5
            x += 0.05

            marker = "$" + chr(65 + i) + "$"
            ax.scatter(x, y, marker = marker, linewidths = 1, edgecolors = 'black', s = 50)

    fig.tight_layout()

def correct_aoi_for_evolution_plot(aoi, epsilon = 0.00001):
    """
    Since matplotlib connect the points of a plot through direct line if the AoI go to zero I normaly obtain a diagon line that go 
    """

    t = []
    new_aoi = []

    for i in range(0, len(aoi)):
        current_aoi = aoi[i]
        
        if current_aoi == 0:
            t.append(i - 1 + epsilon)
            new_aoi.append(0)

        t.append(i)
        new_aoi.append(current_aoi)

    return new_aoi, t

def check_aoi_for_reset(aoi, T_limit = -1):
    """
    aoi = array of shape (T, N) obtained with the simulation in the TDMA file. T = Number of simulation steps, N = numbers of sensors

    Create a list containing the momement where the AoI is reset.
    The list are set of points to be plot in the AoI figure through the scatter function
    """
    
    reset_all_list = []
    reset_single_list = [[] for i in range(aoi.shape[1])]

    if T_limit <= 0: n_samples = aoi.shape[0]
    else: n_samples = T_limit

    for i in range(n_samples):
        if aoi[i].sum() == 0: # If the sum of the row is zero they must be all zero so the AoI was reset through the help factor 
            reset_all_list.append([i - 1, aoi[i - 1, i % aoi.shape[1]]]) 
            reset_single_list[i % aoi.shape[1]].append([i - 1, aoi[i - 1, i % aoi.shape[1]]]) 
        elif (aoi[i] == 0).sum() > 0: # Check if there is at least 1 zero (i.e. at least 1 sensor resets its aoi)
            for j in range(len(aoi[i])):
                if aoi[i, j] == 0:
                    reset_single_list[j].append([i - 1, aoi[i - 1, i % aoi.shape[1]]]) 
                    # reset_single_list.append([i, 0])

    return reset_all_list, reset_single_list


def get_xticks_labels(N, L):
    j = 65
    labels = []
    for i in range(L):
        tick_string = "({})".format(chr(j))
        tick_string = chr(j)
        labels.append(tick_string)

        j+= 1

        if j - 65 >= N: j = 65
    
    return labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot globecom

def plot_single_delay(results_uu, results_ee, config, delay_type):
    """
    Plot the results when we have only a single type of Delay (i.e. only D or only T)
    """
    x = np.geomspace(0.005, config['max_d_delay'], 100)

    fig, ax = plt.subplots()

    if delay_type == 'D':
        uu = results_uu[:, : , 0]
        ee = results_ee[:, : , 0]
    elif delay_type == 'T':
        uu = results_uu[:, 0 , :]
        ee = results_ee[:, 0 , :]
    else:
        raise ValueError("Wrong delay type")

    ax.plot(x, uu[0], 
             label = 'uniform M = {}'.format(config['M_list'][0]), 
             marker = '*', markevery = 10, markersize = 10)
    ax.plot(x, uu[1], 
             label = 'uniform M = {}'.format(config['M_list'][1]),
             marker = '^', markevery = 10, markersize = 10)
    ax.plot(x, ee[0], 
             label = 'exponential M = {}'.format(config['M_list'][0]),
             marker = 'o', markevery = 10, markersize = 10)
    ax.plot(x, ee[1], 
             label = 'exponential M = {}'.format(config['M_list'][1]),
             marker = 's', markevery = 10, markersize = 10)
    
    """ ax.plot(x, results_simulation[0, 0], label = 'simulation M = {}'.format(config['M_list'][0])) """
    """ ax.plot(x, results_simulation[1, 0], label = 'simulation M = {}'.format(config['M_list'][1])) """
    
    name = "single_delay_only_{}".format(delay_type)

    ax.set_ylabel("Average AoI (optimized)")
    ax.set_xlabel("Average Delay ({})".format(delay_type))
    ax.set_xlim([min(x), max(x)])
        
    ax.set_xscale('log')
    
    labels = [0.005, 0.01, 0.02, 0.05, 0.08]
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.grid(True)
    ax.legend()
    
    fig.tight_layout()

    file_type = 'eps'
    filename = "Plot/delay/{}.{}".format(name, file_type)
    plt.savefig(filename, format=file_type)

    file_type = 'png'
    filename = "Plot/delay/{}.{}".format(name, file_type)
    plt.savefig(filename, format=file_type)
    
    plt.show()

def plot_delay_comparison(results, config_computation : dict, max_delay : float = 0.08, idx_M : int = 0):
    """
    Create a plot with 3 line: only D delay, only T delay, equal combination of D + T delay (D/2 + T/2)
    
    results: numpy array of dimensions M x D x T 
    config_computation: dictionary with the config used for the computation
    max_delay: max value of the combination of the two delay
    idx_M: index of the first dimension of the results matrix.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Compute values used for the plot

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])
    
    # Indices for the combined delay and valued of the combined delay
    idx_both_delays = np.zeros(results.shape)
    idx_both_delays, delay_values= compute_idx_both_delays(idx_both_delays, d_values, t_values, max_delay, idx_M)
    both_delay = results[idx_both_delays]
    
    idx_single_delay_d = np.logical_and(d_values >= min(delay_values), d_values <= max(delay_values))
    idx_single_delay_t = np.logical_and(t_values >= min(delay_values), t_values <= max(delay_values))

    only_D = results[idx_M, idx_single_delay_d, 0]
    only_T = results[idx_M, 0, idx_single_delay_t]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Plot 
    config = get_plot_config()
    
    fig, ax = plt.subplots(1, 1, figsize = config['figsize']) 
    plt.rcParams.update({'font.size': config['fontsize']})
    
    ax.plot(delay_values, both_delay, label = "Both delays (theory)", color = 'red', marker = "o", markevery = config['markevery'], markersize = config['markersize'])
    ax.plot(d_values[idx_single_delay_d], only_D, label = "Only D delay (theory)", color = 'skyblue', marker = "x", markevery = config['markevery'], markersize = config['markersize'])
    ax.plot(t_values[idx_single_delay_t], only_T, label = "Only T delay (theory)", color = 'green', marker = "v", markevery = config['markevery'], markersize = config['markersize'])

    # Axis stuff
    ax.set_xscale('log')
    ticks = [0.011, 0.015, 0.02, 0.03, 0.045, 0.07]
    ax.set_xticks(ticks, labels = ticks, minor = False)
    ax.set_xticks(ticks, labels = ticks,minor = True)
    xlim = [max(min(d_values[idx_single_delay_d]), min(delay_values)),min(max(d_values[idx_single_delay_d]), max(delay_values))]
    ax.set_xlim(xlim)
    ax.grid(True, 'major')
    
    # Legend and label
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Average AoI")
    ax.legend()

    fig.tight_layout()
    
    if config['save_fig']:
        name = "delay_comparison"
        file_type = 'eps'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'png'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)
    
    plt.show()


def compute_idx_both_delays(idx_both_delays, d_values, t_values, max_delay, idx_M):
    actual_delay = d_values[0] + t_values[0]
    delay_values = []
    
    i = 0
    while actual_delay <= max_delay:
        delay_values.append(actual_delay)
        idx_both_delays[idx_M, i, i] = 1
        
        i+= 1
        actual_delay = d_values[i] + t_values[i]

    return idx_both_delays == 1, np.asarray(delay_values)

def plot_delay_theory_vs_sim_single_delay(results_theory, results_sim, delay_distribution, delay_type, config):
    x = np.geomspace(0.005, config['max_d_delay'], config['d_points'])
    for i in range(len(config['M_list'])):
        M = config['M_list'][i]
        if delay_type == 'D':
            plt.plot(x, results_theory[i, :, 0], label = "M = {} ({})(theory)".format(M, delay_distribution))
            plt.plot(x, results_sim[i, :, 0], label = "M = {} ({})(sim)".format(M, delay_distribution))
        elif delay_type == 'T':
            plt.plot(x, results_theory[i, 0, :], label = "M = {} ({})(theory)".format(M, delay_distribution))
            plt.plot(x, results_sim[i, 0, :], label = "M = {} ({})(sim)".format(M, delay_distribution))
        else:
            raise ValueError("WRONG DELAY TYPE")
    
    ticks = [0.011, 0.015, 0.02, 0.03, 0.045, 0.07]
    plt.xlabel("Average Delay")
    plt.ylabel("Average AoI")
    plt.legend()
    plt.title("Only {} Delay ({})".format(delay_type, delay_distribution))
    plt.show()

def plot_delay_theory_vs_sim_both_delays(results_theory, results_sim_average, results_sim_std, config_computation : dict, max_delay : float = 0.08, idx_M : int = 0):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Compute values used for the plot

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])
    
    # Indices for the combined delay and valued of the combined delay
    idx_both_delays = np.zeros(results_theory.shape)
    idx_both_delays, delay_values = compute_idx_both_delays(idx_both_delays, d_values, t_values, max_delay, idx_M)
    both_delay_theory = results_theory[idx_both_delays]
    both_delay_sim_avg = results_sim_average[idx_both_delays]
    both_delay_sim_std = results_sim_std[idx_both_delays]
    
    idx_single_delay_d = np.logical_and(d_values >= min(delay_values), d_values <= max(delay_values))
    idx_single_delay_t = np.logical_and(t_values >= min(delay_values), t_values <= max(delay_values))

    only_D_theory = results_theory[idx_M, idx_single_delay_d, 0]
    only_T_theory = results_theory[idx_M, 0, idx_single_delay_t]
    only_D_sim_avg = results_sim_average[idx_M, idx_single_delay_d, 0]
    only_T_sim_avg = results_sim_average[idx_M, 0, idx_single_delay_t]
    only_D_sim_std = results_sim_std[idx_M, idx_single_delay_d, 0]
    only_T_sim_std = results_sim_std[idx_M, 0, idx_single_delay_t]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the figure

    config = get_plot_config()
    
    plt.rcParams.update({'font.size': config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = config['figsize']) 
    
    ax.plot(delay_values, both_delay_theory, label = "Both delays (theory)", color = 'red', marker = "o", markevery = config['markevery'], markersize = config['markersize'])
    ax.plot(d_values[idx_single_delay_d], only_D_theory, label = "Only D delay (theory)", color = 'skyblue', marker = "x", markevery = config['markevery'], markersize = config['markersize'])
    ax.plot(t_values[idx_single_delay_t], only_T_theory, label = "Only T delay (theory)", color = 'green', marker = "v", markevery = config['markevery'], markersize = config['markersize'])

    # ax.plot(delay_values, both_delay_sim, label = "Both delays (sim)", color = 'darkred', marker = "^", markevery = config['markevery'], markersize = config['markersize'])
    # ax.plot(d_values[idx_single_delay_d], only_D_sim, label = "Only D delay (sim)", color = 'royalblue', marker = "8", markevery = config['markevery'], markersize = config['markersize'])
    # ax.plot(t_values[idx_single_delay_t], only_T_sim, label = "Only T delay (sim)", color = 'darkgreen', marker = "s", markevery = config['markevery'], markersize = config['markersize'])

    ax.fill_between(delay_values, both_delay_sim_avg - both_delay_sim_std, both_delay_sim_avg + both_delay_sim_std, 
                    label = "Both delays (sim)", color = 'darkred', alpha = 0.2)
    ax.fill_between(d_values[idx_single_delay_d], only_D_sim_avg - only_D_sim_std, only_D_sim_avg + only_D_sim_std, 
                    label = "Only D delay (sim)", color = 'royalblue', alpha = 0.2)
    ax.fill_between(t_values[idx_single_delay_t], only_T_sim_avg - only_T_sim_std, only_T_sim_avg + only_T_sim_std, 
                    label = "Only T delay (sim)", color = 'darkgreen', alpha = 0.2)

    # Axis stuff
    ax.set_xscale('log')
    xlim = [max(min(d_values[idx_single_delay_d]), min(delay_values)),min(max(d_values[idx_single_delay_d]), max(delay_values))]
    ax.set_xlim(xlim)
    ticks = np.round(np.geomspace(xlim[0], xlim[-1], 8), 3)
    print(ticks)
    ax.set_xticks(ticks, labels = ticks, minor = False)
    ax.set_xticks(ticks, labels = ticks,minor = True)
    ax.grid(True, 'major')
    
    # Legend and label
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Average AoI")
    ax.legend()

    fig.tight_layout()
    
    if config['save_fig']:
        name = "delay_comparison_with_simulation"

        file_type = 'pdf'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'png'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)
    
    plt.show()


def plot_both_delay_different_distribution(results_uniform, results_exp, config_computation, max_delay, idx_M):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Compute values used for the plot

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])
    
    # Indices for the combined delay and valued of the combined delay
    idx_both_delays = np.zeros(results_uniform.shape)
    idx_both_delays, delay_values = compute_idx_both_delays(idx_both_delays, d_values, t_values, max_delay, idx_M)
    both_delay_uniform = results_uniform[idx_both_delays]
    both_delay_exp = results_exp[idx_both_delays]
    
    idx_single_delay_d = np.logical_and(d_values >= min(delay_values), d_values <= max(delay_values))
    idx_single_delay_t = np.logical_and(t_values >= min(delay_values), t_values <= max(delay_values))

    only_D_uniform = results_uniform[idx_M, idx_single_delay_d, 0]
    only_T_uniform = results_uniform[idx_M, 0, idx_single_delay_t]
    only_D_exp = results_exp[idx_M, idx_single_delay_d, 0]
    only_T_exp = results_exp[idx_M, 0, idx_single_delay_t]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    plot_config = get_plot_config()

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    plt.rcParams.update({'font.size': plot_config['fontsize']})

    ax.plot(delay_values, both_delay_uniform, label = "Both delays (uniform)", color = 'red', marker = "o", markevery = plot_config['markevery'], markersize = plot_config['markersize'])
    ax.plot(d_values[idx_single_delay_d], only_D_uniform, label = "Only D delay (uniform)", color = 'skyblue', marker = "x", markevery = plot_config['markevery'], markersize = plot_config['markersize'])
    ax.plot(t_values[idx_single_delay_t], only_T_uniform, label = "Only T delay (uniform)", color = 'green', marker = "v", markevery = plot_config['markevery'], markersize = plot_config['markersize'])

    ax.plot(delay_values, both_delay_exp, label = "Both delays (exp)", color = 'darkred', marker = "^", markevery = plot_config['markevery'], markersize = plot_config['markersize'])
    ax.plot(d_values[idx_single_delay_d], only_D_exp, label = "Only D delay (exp)", color = 'royalblue', marker = "8", markevery = plot_config['markevery'], markersize = plot_config['markersize'])
    ax.plot(t_values[idx_single_delay_t], only_T_exp, label = "Only T delay (exp)", color = 'darkgreen', marker = "s", markevery = plot_config['markevery'], markersize = plot_config['markersize'])

    # Axis stuff
    ax.set_xscale('log')
    xlim = [max(min(d_values[idx_single_delay_d]), min(delay_values)),min(max(d_values[idx_single_delay_d]), max(delay_values))]
    ax.set_xlim(xlim)
    ticks = np.round(np.geomspace(xlim[0], xlim[-1], 8), 3)
    print(ticks)
    ax.set_xticks(ticks, labels = ticks, minor = False)
    ax.set_xticks(ticks, labels = ticks,minor = True)
    ax.grid(True, 'major')
    
    # Legend and label
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Average AoI")
    ax.legend()

    fig.tight_layout()
    
    if plot_config['save_fig']:
        name = "both_delay_different_distribution"

        file_type = 'eps'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'png'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)
    
    plt.show()

def ratio_aoi_with_no_delay(results, config_computation : dict, max_delay : float = 0.08):
    """
    Create a plot with 3 line: only D delay, only T delay, equal combination of D + T delay (D/2 + T/2)
    
    results: numpy array of dimensions M x D x T 
    config_computation: dictionary with the config used for the computation
    max_delay: max value of the combination of the two delay
    idx_M: index of the first dimension of the results matrix.
    """
    
    config = get_plot_config()
        
    plt.rcParams.update({'font.size': config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = config['figsize']) 

    for i in range(len(config_computation['M_list'])):
        normalization_factor = AoI_Delay_BOTH.aoi_theory_formula(config_computation['M_list'][i], 0, 0, 'uniform')

        normalized_results = results / normalization_factor

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Compute values used for the plot

        # Compute d-values and t-values
        d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
        t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])
        
        # Indices for the combined delay and valued of the combined delay
        idx_both_delays = np.zeros(results.shape)
        idx_both_delays, delay_values= compute_idx_both_delays(idx_both_delays, d_values, t_values, max_delay,i)
        both_delay = normalized_results[idx_both_delays]
        
        idx_single_delay_d = np.logical_and(d_values >= min(delay_values), d_values <= max(delay_values))
        idx_single_delay_t = np.logical_and(t_values >= min(delay_values), t_values <= max(delay_values))

        only_D = normalized_results[i, idx_single_delay_d, 0]
        only_T = normalized_results[i, 0, idx_single_delay_t]

        color_list = [['red', 'skyblue', 'green'], ['darkred', 'royalblue', 'darkgreen']]
        marker_list = [["o", "x", "v"], ["^", "8", "s"]]
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Plot 
        
        ax.plot(delay_values, both_delay, label = "Both delays (M = {})".format(config_computation['M_list'][i]), color = color_list[i][0], marker = marker_list[i][0], markevery = config['markevery'], markersize = config['markersize'])
        ax.plot(d_values[idx_single_delay_d], only_D, label = "Only D delay (M = {})".format(config_computation['M_list'][i]), color = color_list[i][1], marker = marker_list[i][1], markevery = config['markevery'], markersize = config['markersize'])
        ax.plot(t_values[idx_single_delay_t], only_T, label = "Only T delay (M = {})".format(config_computation['M_list'][i]), color = color_list[i][2], marker = marker_list[i][2], markevery = config['markevery'], markersize = config['markersize'])

    # Axis stuff
    ax.set_xscale('log')
    ticks = [0.011, 0.015, 0.02, 0.03, 0.045, 0.07]
    ax.set_xticks(ticks, labels = ticks, minor = False)
    ax.set_xticks(ticks, labels = ticks,minor = True)
    xlim = [max(min(d_values[idx_single_delay_d]), min(delay_values)),min(max(d_values[idx_single_delay_d]), max(delay_values))]
    ax.set_xlim(xlim)
    ax.grid(True, 'major')
    
    # Legend and label
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Average AoI")
    ax.legend()

    fig.tight_layout()
    
    if config['save_fig']:
        name = "delay_comparison"
        file_type = 'eps'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)

        file_type = 'png'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format=file_type)
        
    fig.show()


def fix_one_delay(results, config_computation, fix_delay : float):
    """
    Create a plot with the average delay on the x-axis and use it as variable for a delay while keeping the other delay fix
    E.g. keep the D delay fix at 0.02 and use the value of x-axis as variable for the T delay

    results : numpy array of dimensions M x D x T 
    config_computation : dictionary with the config used for the computation
    fix_delay : value of the fixed delay (float)
    """

    if config_computation['d_max_delay'] != config_computation['t_max_delay']:
        raise ValueError("The max delay must be the same for T and D")

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])

    # Find the indices for D and T of the closest value to fix delay
    idx_for_D = np.argmin(np.abs(t_values - fix_delay))
    idx_for_T = np.argmin(np.abs(d_values - fix_delay))
    
    # Get plot config and create figure
    config_plot = get_plot_config()
    plt.rcParams.update({'font.size': config_plot['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = config_plot['figsize']) 

    color_list = [['skyblue', 'green'], ['royalblue', 'darkgreen']]
    marker_list = [["x", "v"], ["8", "s"]]

    for i in range(len(config_computation['M_list'])):
        delay_D_with_T_fix = results[i, :, idx_for_D]
        delay_T_with_D_fix = results[i, idx_for_T, :]

        ax.plot(d_values, delay_D_with_T_fix, label = "D delay (M = {})".format(config_computation['M_list'][i]),
                color = color_list[i][0], marker = marker_list[i][0], markevery = config_plot['markevery'], markersize = config_plot['markersize'])

        ax.plot(t_values, delay_T_with_D_fix, label = "T delay (M = {})".format(config_computation['M_list'][i]),
                color = color_list[i][1], marker = marker_list[i][1], markevery = config_plot['markevery'], markersize = config_plot['markersize'])
                
    ax.set_xscale('log')
    ticks = np.round(np.geomspace(0.005, config_computation['d_max_delay'], 6), 3) 
    ax.set_xticks(ticks, labels = ticks, minor = False)
    ax.set_xticks(ticks, labels = ticks,minor = True)
    ax.set_xlim([min(d_values), max(d_values)])
    ax.grid(True, 'major')
    
    # Legend and label
    ax.set_xlabel("Average Delay")
    ax.set_ylabel("Average AoI")
    ax.legend()
    fig.tight_layout()

    if config_plot['save_fig']:
        name = "fix_delay_{}_D_{}_T_{}".format(round(fix_delay * 100), config_computation['d_type'], config_computation['t_type'])
        file_type = 'pdf'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format = file_type)

        file_type = 'png'
        filename = "Plot/delay/{}.{}".format(name, file_type)
        plt.savefig(filename, format = file_type)

    fig.show()


def change_proportion(results, config_computation : dict, fixed_values_list : list):
    """
    Given the fixed value of delay show how the AoI change based on the proportion a * D + (1 - a) * T
    """

    alpha_list = np.linspace(0, 1, 100)

    # Get plot config and create figure
    config_plot = get_plot_config()
    plt.rcParams.update({'font.size': config_plot['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = config_plot['figsize']) 

    color_list = [['skyblue', 'green'], ['royalblue', 'darkgreen']]
    marker_list = [["x", "v"], ["8", "s"]]
    
    for i in range(len(config_computation['M_list'])):
        for j in range(len(fixed_values_list)):
            fixed_value = fixed_values_list[j]


def find_aoi_for_chagen_proportion_function(results, config_computation : dict, fixed_value : float, alpha_list):
    """
    Find the AoI for each value of the linear combination a * D + (1 - a) + T
    results : matrix of shape D x T
    """

    if config_computation['d_points'] != config_computation['t_points']:
        raise ValueError("You have to compute the AoI on the same number of points for D and T (i.e. d_points must be equal to t_points)")

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['d_max_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['t_max_delay'], config_computation['t_points'])
    for i in range(len(d_values)):
        for k in range(len(t_values)):
            pass
    
    aoi_list = []
    for alpha in alpha_list:
        delays_sum = alpha * d_values + (1 - alpha) * t_values
        
        closest_idx = np.unravel_index(np.argmin(delays_sum - fixed_value), delays_sum.shape)

        
        
