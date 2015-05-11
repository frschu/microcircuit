'''
    analysis.py
    
    Further command line arguments:
        c       script will close all open plots
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Produces raster plots and calculates firing ratios
'''
from __future__ import print_function
from imp import reload
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.abspath('../../')) # include path for import
sys.path.append(os.path.abspath('../')) # include path for import
import style
# Close other plots by adding 'c' after 'run <script>' 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'
######################################################

# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import functions

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)

# Data path
data_path = np.load('data_path.npy')[0]
if not 'sli' in sys.argv:
    sli = False
else:
    sli = True
    data_path = '/users/schuessler/uni/microcircuit/data_sli/test_01/' 
npy_path = np.load('data_path.npy')[0]
sim_spec = data_path.split('/')[-2]

# Get data specified for pynest simulation
populations = net.populations
layers = net.layers
types= net.types
n_populations = len(populations)
n_layers = len(layers)
n_types = len(types)
# labels & colors: need to be adapted if n_types != (e, i)
layer_colors = style.colors[:n_layers]
colors = np.array([color for color in layer_colors for i in range(n_types)])
colors[1::2] = colors[1::2] * 0.4

if reverse_order:
    populations = populations[::-1]
    layers = layers[::-1]
    types = types[::-1]
    colors = colors[::-1]

T0 = 200. # ms; starting point of analysis (avoid transients)
t_measured = sim.t_sim - T0


# Get data from files
# Analyze each population independently
def get_GIDs_times(population, data_path, sli=False):
    GIDs = np.int_([])
    times = np.array([])
    file_names  = os.listdir(data_path)
    if sli:
        pop_id      = net.populations.index(population)
        type_id     = int(pop_id % len(net.types))
        layer_id    = int(pop_id / len(net.types))
        print(pop_id, layer_id, type_id)
        prefix = 'spikes_' + str(layer_id) + '_' + str(type_id)
        print(population, prefix)
    else:
        prefix = 'spikes_' + population
    for file_name in file_names:
        if file_name.startswith(prefix):
            file = open(data_path + file_name, 'r')
            for line in file:
                GID, time = line.split()
                time = float(time)
                if time > T0:
                    GIDs = np.append(GIDs, int(GID))
                    times = np.append(times, time)
            file.close()
    return GIDs, times



if sli:         # statistic generated with sli code (not quite optimal, as parameters must 
    lower_GIDs = np.zeros(n_populations)
    GID_file = open(data_path + 'population_GIDs.dat', 'r')
    for i, line in enumerate(GID_file):
        lower_GIDs[i] = np.int_(line.split()[0])
    GID_file.close()
    if reverse_order:
        lower_GIDs = lower_GIDs[::-1]

n_rec_spikes = np.load(npy_path + 'n_neurons_rec_spikes.npy')
if reverse_order:
    n_rec_spikes = n_rec_spikes[::-1]
lower_cum_n_GIDs = np.insert(np.cumsum(n_rec_spikes)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spikes_layer = np.sum(n_rec_spikes.reshape(n_layers, n_types), 1) # recorded spikes per layer

# Mean and Std of firing rates
r_mean  = np.zeros(n_populations)
r_std   = np.zeros(n_populations)
# Histogram
bin_width = 1.  # ms
n_bins = int(t_measured / bin_width) 
hist_all = np.zeros((n_populations, n_bins)).astype(int)


# Plotting
fig = plt.figure()
fig.suptitle('Simulation for: area = %.1f, time = %ims'%(net.area, sim.t_sim))

# Raster plot
ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
# Rates
ax1 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=2)
# Histogram
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1)


############################################################################################
for i, population in enumerate(populations):
    GIDs, times = get_GIDs_times(population, data_path, sli=sli)
    # Get measured GIDs and transform to numbers  
    unique_GIDs, spike_indices, n_spikes = np.unique(GIDs, return_inverse=True,
        return_counts=True) 

    # Analysis
    # Firing rate:
    rates = n_spikes / (t_measured) * 1e3 # Hz
    # Mean and std dev
    r_mean[i] = np.mean(rates)
    r_std[i] = np.std(rates)
    
    # Histogram over time
    hist_all[i], bin_edges = np.histogram(times, bins=n_bins, range=(T0, sim.t_sim))

    # Plotting
    if sli:
        rec_spikes_GIDs = lower_GIDs[i] + np.arange(n_rec_spikes[i])
    else:
        rec_spikes_GIDs = np.load(npy_path + 'rec_spikes_GIDs_' + population + '.npy')
    # Since not every neurons fired, the set of unique GIDs is smaller than that of rec. spikes.
    # Get the indices of those neurons that fired respective to all recorded ones:
    GID_indices = np.where(np.in1d(rec_spikes_GIDs, unique_GIDs, assume_unique=True))[0]
    # Add the offset of the previously analyzed populations.
    y_raster    = GID_indices[spike_indices] + lower_cum_n_GIDs[i]
    y_rate      = GID_indices + lower_cum_n_GIDs[i]
    bar_height = 1. 
    ax0.plot(times*1e-3, y_raster, '.', ms=3., color=colors[i], label=population)
    ax1.barh(y_rate, rates, height=bar_height, color=colors[i], linewidth=0)


# Plot histogram
sum_hist = np.sum(hist_all, 0)
t_edges = bin_edges[:-1] * 1e-3
ax2.bar(t_edges, sum_hist, width=bin_width*1e-3, 
    fc=colors[-1], ec=colors[-1], linewidth=0, fill=True)
ax2.set_xlabel('simulation time / s')
ax2.set_ylabel('Counts')
ax2.grid(False)


ylim = (0, sum(n_rec_spikes))
# set yticks to center of recorded n spikes of one layer
yticks = lower_cum_n_GIDs[::n_types] + 0.5 * n_rec_spikes_layer


ax0.set_yticks(yticks)
ax0.set_yticklabels(layers)
#ax0.set_xlabel('simulation time / s')
ax0.set_ylabel('Layer')
ax0.set_ylim(*ylim)
ax0.grid(False)

ax1.set_yticks(yticks)
ax1.set_yticklabels(layers)
#ax1.set_xlabel('firing rate / Hz')
#ax1.set_ylabel('Layer')
ax1.set_ylim(*ylim)
ax1.set_xlim(0, 20)
ax1.grid(False)

# Legend; order is reversed, such that labels appear correctly
for i in range(n_types):
    ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
ax1.legend(loc='best')

fig.savefig(data_path + 'raster_and_rates_' + sim_spec + picture_format)
fig.show()

 
