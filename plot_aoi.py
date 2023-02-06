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
        fontsize = 28,
        save_fig = False,
        # - - - - - - - - - - - - -
        # Parameters for aoi surface
        use_imshow = False,
        levels_countourf = 20,
        add_color_bar = True,
        # - - - - - - - - - - - - -
        # Parameters for aoi vs N
        labels = ["p = {}".format(p) for p in [0.005, 0.01, 0.02, 0.05]],
        idx_to_plot = [(i, 1) for i in range(4) ],
        linestyle = ['solid', 'dotted', 'dashed', 'dashdot']
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


def plot_aoi_vs_N(N_array, aoi_array):
    config = get_plot_config()
    
    fig, ax = plt.subplots(1, 1, figsize = (15, 10)) 
    plt.rcParams.update({'font.size': config['fontsize']})
    
    for i in range(len(config['idx_to_plot'])):
        tmp_idx = config['idx_to_plot'][i]
        
        ax.plot(N_array, aoi_array[tmp_idx[0], tmp_idx[1], :], 
                label = config['labels'][i], linestyle = config['linestyle'][i], linewidth = 2.5)

    ax.legend()
    
    ax.set_xlabel("Number of neighbors")
    ax.set_ylabel("AoI")
    ax.grid(True)

    ax.set_xlim([N_array[0], N_array[-1]])
    ax.set_yscale('log')

    fig.tight_layout()
    plt.tight_layout()

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
        figsize = (15, 6),
        fontsize = 18,
        x_limit = 32, # Points to plot (i.e. simulation step)
        y_limit = 10,
        linewidth = 1.7,
        N = 4, # Number of sensors
    )
    
    # Create ticks for the grid and labels
    major_x_ticks = np.arange(0, config['x_limit'], 1)
    minor_x_ticks = np.arange(0, config['x_limit'], 1) + 0.5
    major_y_ticks = np.arange(0, config['y_limit'], 1)
    
    # Correct AoI
    aoi_correct = []
    t_array = []
    for i in range(aoi.shape[1]):
        tmp_aoi_correct, t = correct_aoi_for_evolution_plot(aoi[:, i])
        
        aoi_correct.append(np.asarray(tmp_aoi_correct))
        t_array.append(t)
    
    # Find point where AoI was reset
    reset_all_list, reset_single_list = check_aoi_for_reset(aoi, config['x_limit'])

    plt.rcParams.update({'font.size': config['fontsize']})
    
    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    
    for i in range(int(config['x_limit'] / config['N'])):
        ax.axvline(i * config['N'], color = 'black', alpha = 0.5)
    
    shift = 0.025
    for i in range(aoi.shape[1]):
        ax.plot(t_array[i], aoi_correct[i] + shift, 
                linewidth = config['linewidth'], label = "Sensor {}".format(i + 1))
        shift += shift

    ax.set_xticks(major_x_ticks)
    ax.set_yticks(major_y_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.grid(True)
    ax.legend()
    
    ax.set_xlim([0, config['x_limit']])
    ax.set_ylim([0, config['y_limit']])

    
    x_ticks_labels = get_xticks_labels(config['N'], len(minor_x_ticks))
    print(x_ticks_labels)
    ax.set_xticklabels('')
    ax.set_xticklabels(x_ticks_labels, minor = True)
    ax.set_xlabel("Time [Units of time]")
    ax.set_ylabel("AoI")
    
    for point in reset_all_list:
        x, y = point
        ax.scatter(x, y, marker = 'o', color = 'black', linewidths = 5)

    # for point in reset_single_list:
    #     x, y = point
    #     ax.scatter(x, y, marker = 'o')

    fig.tight_layout()
    plt.show()

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
    reset_single_list = []

    if T_limit <= 0: n_samples = aoi.shape[0]
    else: n_samples = T_limit

    for i in range(n_samples):
        if aoi[i].sum() == 0: # If the sum of the row is zero they must be all zero so the AoI was reset through the help factor 
            reset_all_list.append([i - 1, aoi[i - 1, i % aoi.shape[1]]]) 
        elif (aoi[i] == 0).sum() > 0: # Check if there is at least 1 zero (i.e. at least 1 sensor resets its aoi)
            for j in range(len(aoi[i])):
                if aoi[i, j] == 0:
                    reset_single_list.append([i, 0])

    return reset_all_list, reset_single_list


def get_xticks_labels(N, L):
    j = 65
    labels = []
    for i in range(L):
        labels.append(chr(j))

        j+= 1

        if j - 65 >= N: j = 65
    
    return labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function to check/rewrite

def plot_old_1(alpha, c):
    plt.figure(figsize = (15, 10))
    plt.plot(alpha, c[0, :, 0], label = "p = 0.01  N = 10", linestyle = 'solid')
    plt.plot(alpha, c[0, :, 1], label = "p = 0.01  N = 30", linestyle = 'dashed')
    plt.plot(alpha, c[1, :, 0], label = "p = 0.05  N = 10", linestyle = 'dotted')
    plt.plot(alpha, c[1, :, 1], label = "p = 0.05  N = 30", linestyle = 'dashdot')

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1e-3, 1])
    plt.grid(True)
    plt.xlabel('Probability $q$ of a useful update from a neighbor')
    plt.ylabel('AoI')
    plt.title('AoI vs $q$ TDMA Simulation')
    plt.show()


def plot_old_2(alpha, a , c):
    plt.figure(figsize = (15, 10))
    plt.plot(alpha, a[0, :, 0], label =  "Simulation   p = 0.01  N = 10", linestyle = 'solid', linewidth = 2)
    plt.plot(alpha, c[0, :, 0], label =  "Closed form  p = 0.01  N = 10", linestyle = 'dashed',linewidth = 2)
    plt.plot(alpha, a[57, :, 1], label = "Simulation   p = 0.05  N = 30", linestyle = 'dotted', linewidth = 2)
    plt.plot(alpha, c[57, :, 1], label = "Closed form  p = 0.05  N = 30", linestyle = 'dashdot', linewidth = 2)

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1e-3, 1])
    plt.grid(True)
    plt.xlabel('Probability $q$ of a useful update from a neighbor')
    plt.ylabel('AoI')
    plt.title('Simulation vs Closed Form')
    plt.show()
