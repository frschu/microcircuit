"""potential.py
    
    Further command line arguments:
        c       script will close all open plots
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Analysis of membrane potential distribution.
"""
from __future__ import print_function
from imp import reload
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.abspath('../')) # include path with style
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specificaitons
sys.path.append(os.path.abspath('./simulation/')) # include path with simulation specificaitons
import style; reload(style)
# Close other plots by adding 'c' after 'run <script>' 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'
######################################################

# Import specific moduls
import network_params as net; reload(net)
import user_params as user; reload(user)

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)
plotting = True

# Data path
data_sup_path = user.data_dir
simulation_spec = user.simulation_spec
print(simulation_spec)

simulation_path = data_sup_path + simulation_spec
pynest_path = simulation_path + 'pynest/'
if 'sli' in sys.argv:
    sli = True
    data_path = simulation_path + 'sli/' 
    npy_path = simulation_path + 'npy_data_sli/' 
else:
    sli = False
    data_path = simulation_path + 'pynest/' 
    npy_path = simulation_path + 'npy_data/' 

figure_path = simulation_path + 'figures/' 
if not os.path.exists(figure_path):
    os.mkdir(figure_path)

# Get data specified for pynest simulation
populations = net.populations
#populations = ['L4e']
layers = net.layers
types= net.types
n_populations = len(populations)
n_layers = len(layers)
n_types = len(types)
n_rec_spike = np.load(pynest_path + 'n_neurons_rec_spike.npy')
# labels & colors: need to be adapted if n_types != (e, i)
layer_colors = style.colors[:n_layers]
colors = np.array([color for color in layer_colors for i in range(n_types)])
colors[1::2] = colors[1::2] * 0.4   #### adapt for more than two types!
if reverse_order:
    n_rec_spike = n_rec_spike[::-1]
    populations = populations[::-1]
    layers = layers[::-1]
    types = types[::-1]
    colors = colors[::-1]

# Simulation parameters
area    = float(simulation_spec.split("_")[0][1:])          # mm**2
t_sim   = float(simulation_spec.split("_")[1][1:]) * 10**3  # ms
t_trans = 200. # ms; starting point of analysis (avoid transients)
t_measure = t_sim - t_trans

# Further derived paramters
lower_GIDs = np.insert(np.cumsum(n_rec_spike)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spike_layer = np.sum(n_rec_spike.reshape(n_layers, n_types), 1) # recorded spikes per layer

# Plotting: Prepare figures
if plotting:
    fig = plt.figure()
    suptitle = 'Simulation for: area = %.1f, time = %ims'%(area, t_sim)
    suptitle += '\nfile: ' + simulation_spec
    if sli: 
        suptitle += '  SLI'
    fig.suptitle(suptitle, y=0.98)
    # Membrane pot over time
    ax0 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    # Histogram of membrane pot
    ax1 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)


############################################################################################
V_times_file    = 'V_times.npy'
V_times         = np.load(npy_path + V_times_file)
for i, population in enumerate(populations):
    print(population)
    # Get membrane potentials
    volts_file  = 'voltages_' + population + '.npy'
    Vs_all      = np.load(npy_path + volts_file)
    #n_GIDs = Vs_all.shape()[0]
    
    # Histogram
    j = 0
    Vs = Vs_all[j]
    #ax0.plot(V_times*1e-3, Vs, 
    #    '-', color=colors[j], alpha=0.3, linewidth=0.5, label=population)
    n_bins_Vs   = 200 - 1
    
    if population[-1] == "e":
        ax = ax0
    else:
        ax = ax1
    V_min = -100
    V_max = -50
    bin_edges = np.linspace(V_min, V_max, n_bins_Vs + 1)
    Vs_mean, b_e  = np.histogram(Vs_all, bin_edges, normed=True)
    ax.plot(bin_edges[:-1], Vs_mean, linewidth=3., color=colors[i], label=population)
    n_hist = min(20, len(Vs_all))
    ax.hist(Vs_all[:n_hist].T, bins=n_bins_Vs, normed=True, histtype='step', 
        fill=False, linewidth=1.0, color=[colors[i]]*n_hist, alpha=0.2)

    

if plotting:
    # Potential over time
    xlim = (0, t_sim * 1e-3)
    ax0.set_xlabel('simulation time / s')
    ax0.set_ylabel('Membrane potential / V')
    ax0.set_xlim(*xlim)
    ax0.grid(True)
    
    
    for ax in fig.axes:
        ax.set_ylabel('Probability P(V)')
        ax.set_xlabel('Membrane potential $V$ / V')
        ax.set_xlim(V_min, V_max)
        ax.grid(True)
        ax.legend(loc='best')
    
    
    for ax in fig.axes:
        style.fixticks(ax)
    fig_name = 'potential'
    if sli:
        fig_name += '_sli'
    #fig.savefig(figure_path + fig_name + picture_format)
    
    fig.show()
