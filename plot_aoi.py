"""
@Author: Alberto Zancanaro (Jesus)
@Organization: University of Padua

Functions used to plot the AoI obtained from the simulation or other figures to insert in various paper
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt


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
        fontsize = 20,
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

def plot_delay_comparison(results, config_computation : dict, max_delay : float = 0.08, idx_M :int = 0):
    """
    Create a plot with 3 line: only D delay, only T delay, equal combination of D + T delay (D/2 + T/2)
    
    results: numpy array of dimensions possible_tx x D x T (see AoI_delay.py for the output dimension)
    config_computation: dictionary with the config used for the computation
    max_delay: max value of the combination of the two delay
    idx_M: index of the first dimension of the results matrix.
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Compute values used for the plot

    # Compute d-values and t-values
    d_values = np.geomspace(0.005, config_computation['max_d_delay'], config_computation['d_points'])
    t_values = np.geomspace(0.005, config_computation['max_t_delay'], config_computation['t_points'])
    
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
    
    ax.plot(delay_values, both_delay, label = "Both Delay")
    ax.plot(d_values[idx_single_delay_d], only_D, label = "Only D delay")
    ax.plot(t_values[idx_single_delay_t], only_T, label = "Only T delay")

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

def plot_delay_theory_vs_sim_single_delat(results_theory, results_sim, delay_distribution, delay_type, config):
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
