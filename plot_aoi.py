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
