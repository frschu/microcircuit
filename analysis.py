'''
    analysis.py

    Produces raster plots and calculates firing ratios
'''
from __future__ import print_function
import nest
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath('../../')) # include path for import
sys.path.append(os.path.abspath('../')) # include path for import
import style
# Close other plots by adding 'c' after 'run <script>' 
if len(sys.argv) > 1:
    if sys.argv[1] == 'c':
        plt.close('all')
picture_format = '.pdf'
######################################################

# Import specific moduls
import network_params as net
import sim_params as sim
import user_params as user
import functions

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)
y_axis_GIDs = False # use GIDs as point on axis; otherwise number measured neurons

# Get data
data_path = np.load('data_path.npy')[0]
sim_spec = data_path.split('/')[-2]

populations = net.populations
layers = net.layers
types= net.types
n_populations = len(populations)
n_layers = len(layers)
n_types = len(types)
n_rec_spikes = np.load(data_path + 'n_neurons_rec_spikes.npy')
lower_GIDs = np.load(data_path + 'lower_GIDs.npy')
upper_GIDs = np.load(data_path + 'upper_GIDs.npy')
# labels & colors: need to be adapted if n_types != (e, i)
colors = style.colors[:n_types]
rev = lambda arr: arr[::-1]
if reverse_order:
    populations = rev(populations)
    layers = rev(layers)
    types = rev(types)
    colors = rev(colors)
    n_rec_spikes = rev(n_rec_spikes)
    lower_GIDs = rev(lower_GIDs)
    upper_GIDs = rev(upper_GIDs)

T0 = 200. # ms; starting point of analysis (avoid transients)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Simulation for: area = %.1f, time = %ims'%(net.area, sim.t_sim))
# Raster plot
ax0 = ax[0]  
# Rates
ax1 = ax[1]  
GID_maxs = np.zeros(n_populations) # maximum GID of each population
lower_cum_n_GIDs = np.insert(np.cumsum(n_rec_spikes)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spikes_layer = np.sum(n_rec_spikes.reshape(n_layers, n_types), 1) # recorded spikes per layer

# Get data from files
# Analyze each population independently
file_names  = os.listdir(data_path)
for i, population in enumerate(populations):
    rec_spikes_GIDs = np.load(data_path + 'rec_spikes_GIDs_' + population + '.npy')
    GIDs = np.int_([])
    times = np.array([])
    for file_name in file_names:
        parts = file_name.split('_')
        if parts[0] == 'spikes':        # only worry about spikes at this point
            if parts[1] == population:  # is this the correct population?
                file = open(data_path + file_name, 'r')
                for line in file:
                    GID, time = line.split()
                    time = float(time)
                    if time > T0:
                        GIDs = np.append(GIDs, int(GID))
                        times = np.append(times, time)

    # Get measured GIDs and transform to numbers  
    unique_GIDs, spike_indices, n_spikes = np.unique(GIDs, return_inverse=True,
        return_counts=True) 
    # Since not every neurons fired, the set of unique GIDs is smaller than that of rec. spikes.
    # Get the indices of those neurons that fired respective to all recorded ones:
    GID_indices = np.where(np.in1d(rec_spikes_GIDs, unique_GIDs, assume_unique=True))[0]
    # Add the offset of the previously analyzed populations.
    y_raster    = GID_indices[spike_indices] + lower_cum_n_GIDs[i]
    y_rate      = GID_indices + lower_cum_n_GIDs[i]

    # Firing rate:
    rates = n_spikes / (sim.t_sim - T0) * 1e3 # Hz

    # Plotting
    if y_axis_GIDs: # use GIDs a y values
        bar_height = 10. 
        ax0.plot(times*1e-3, GIDs, '.', ms=3., color=colors[i % n_types], label=population)
        ax1.barh(unique_GIDs, rates, height=bar_height, color=colors[i % n_types], linewidth=0)
        GID_maxs[i] = rec_spikes_GIDs[-1]
    else:           # use nth measured GID as y values
        bar_height = 1. 
        ax0.plot(times*1e-3, y_raster, '.', ms=3., color=colors[i % n_types], label=population)
        ax1.barh(y_rate, rates, height=bar_height, color=colors[i % n_types], linewidth=0)

if y_axis_GIDs:
    ylim = (0, max(GID_maxs))
    # set yticks to center of GIDs of all neurons of the according layer
    yticks = (lower_GIDs[::n_types] + upper_GIDs[1::n_types]) * 0.5
else:
    ylim = (0, sum(n_rec_spikes))
    # set yticks to center of recorded n spikes of one layer
    yticks = lower_cum_n_GIDs[::n_types] + 0.5 * n_rec_spikes_layer


ax0.set_yticks(yticks)
ax0.set_yticklabels(layers)
ax0.set_xlabel('simulation time / s')
ax0.set_ylabel('Layer')
ax0.set_ylim(*ylim)
ax0.grid(False)

ax1.set_yticks(yticks)
ax1.set_yticklabels(layers)
ax1.set_xlabel('firing rate / Hz')
ax1.set_ylabel('Layer')
ax1.set_ylim(*ylim)
ax1.grid(False)

# Legend; order is reversed, such that labels appear correctly
for i in range(n_types):
    ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
ax1.legend(loc='best')

fig.savefig(data_path + 'raster_and_rates' + sim_spec + picture_format)
fig.show()

 
