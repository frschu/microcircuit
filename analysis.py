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
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.abspath('../../')) # include path for import
sys.path.append(os.path.abspath('../')) # include path for import
import style; reload(style)
# Close other plots by adding 'c' after 'run <script>' 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'
######################################################

# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import functions; reload(functions)

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)


# Data path
#data_path = np.load('data_path.npy')[0]
data_sup_path = '/users/schuessler/uni/microcircuit/data/'
file_name = 'a1.0_t20.0_00/'
data_path = data_sup_path + file_name
npy_path = data_path
if not 'sli' in sys.argv:
    sli = False
else:
    sli = True
    data_path = '/users/schuessler/uni/microcircuit/data_sli/test_02/' 
sim_spec = data_path.split('/')[-2]


#########
# THIS IST NOT GOOD: parameters in net and sim must remain unchanged...
#########

# Get data specified for pynest simulation
populations = net.populations
layers = net.layers
types= net.types
n_populations = len(populations)
n_layers = len(layers)
n_types = len(types)
n_rec_spike = np.load(npy_path + 'n_neurons_rec_spike.npy')
# labels & colors: need to be adapted if n_types != (e, i)
layer_colors = style.colors[:n_layers]
colors = np.array([color for color in layer_colors for i in range(n_types)])
colors[1::2] = colors[1::2] * 0.4
if reverse_order:
    n_rec_spike = n_rec_spike[::-1]
    populations = populations[::-1]
    layers = layers[::-1]
    types = types[::-1]
    colors = colors[::-1]


# Simulation time
t_sim = sim.t_sim
T0 = 200. # ms; starting point of analysis (avoid transients)
t_measured = t_sim - T0


# Further derived paramters
lower_cum_n_GIDs = np.insert(np.cumsum(n_rec_spike)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spike_layer = np.sum(n_rec_spike.reshape(n_layers, n_types), 1) # recorded spikes per layer
# statistics generated with sli code (not quite optimal, as parameters must correspond to 
# those of the pynest simulation!
if sli:         
    lower_GIDs = np.zeros(n_populations)
    GID_file = open(data_path + 'population_GIDs.dat', 'r')
    for i, line in enumerate(GID_file):
        lower_GIDs[i] = np.int_(line.split()[0])
    GID_file.close()
    if reverse_order:
        lower_GIDs = lower_GIDs[::-1]


# Mean and Std of firing rates and CV of ISI
rates_mean  = np.zeros(n_populations)
rates_std   = np.zeros(n_populations)
cv_isi_mean = np.zeros(n_populations)
cv_isi_std  = np.zeros(n_populations)
# Histogram
bin_width = 10.  # ms
n_bins = int(t_measured / bin_width) 
rate_inst_all = np.zeros((n_populations, n_bins))
# ISI
no_isi = []


# Plotting: Prepare figures
y_mean = np.arange(n_populations) + 0.1
ylim_mean = (0, n_populations)
yticks_mean = np.arange(n_types * 0.5, n_populations, n_types)
bar_height = 0.8 
fig = plt.figure()
suptitle = 'Simulation for: area = %.1f, time = %ims'%(net.area, t_sim)
suptitle += '\nfile: ' + file_name
if sli: 
    suptitle += '  SLI'
fig.suptitle(suptitle, y=0.98)
# Raster plot
ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
# Rates
ax1 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=2)
# Histogram
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1)
# CV of interspike interval (ISI)
ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)

############################################################################################
for i, population in enumerate(populations):
    print(population)
    # Get data
    GIDs, times = functions.get_GIDs_times(population, data_path, T0, sli=sli)
    unique_GIDs, spike_indices, n_spikes = np.unique(GIDs, return_inverse=True,
        return_counts=True) 
    if sli:
        rec_spike_GIDs = lower_GIDs[i] + np.arange(n_rec_spike[i])
    else:
        rec_spike_GIDs = np.load(npy_path + 'rec_spike_GIDs_' + population + '.npy')

    # Analysis
    # Irregularity: CV of interspike intervals (ISI)
    n_spikes_all = np.zeros(n_rec_spike[i])
    cv_isi_all = np.empty(0)
    for j, GID in enumerate(rec_spike_GIDs):
        times_GID = times[GID == GIDs]
        n_spikes = len(times_GID)
        n_spikes_all[j] = n_spikes
        if n_spikes > 1:
            isi = np.diff(times_GID)
            mean_isi = np.mean(isi)
            var_isi = np.var(isi)
            cv_isi = var_isi / mean_isi**2
            cv_isi_all = np.append(cv_isi_all, cv_isi)
        else:
            no_isi.append(GID)
            #cv_isi = 0

    # Firing rate:
    rates = n_spikes_all / t_measured * 1e3 # Hz

    # Means
    rates_mean[i] = np.mean(rates)
    rates_std[i] = np.std(rates)
    cv_isi_mean[i] = np.mean(cv_isi_all)
    cv_isi_std[i] = np.std(cv_isi_all)

    # Synchrony
    # Histogram over time
    hist, bin_edges = np.histogram(times, bins=n_bins, range=(T0, t_sim))
    t_edges = bin_edges[:-1] * 1e-3
    # convolve 
    n_box = 10
    bc = signal.boxcar(n_box)
    rate_inst = signal.convolve(hist, bc, mode='same') / n_box # normed convolution
    ax2.bar(t_edges, rate_inst, width=bin_width*1e-3, 
        fc=colors[i], ec=colors[i], linewidth=0.2, fill=False)
    rate_inst_all[i] = rate_inst 

    # Plotting
    print('Plotting')
    # Since not every neurons fired, the set of unique GIDs is smaller than that of rec. spikes.
    # Get the indices of those neurons that fired respective to all recorded ones:
    GID_indices = np.where(np.in1d(rec_spike_GIDs, unique_GIDs, assume_unique=True))[0]
    # Add the offset of the previously analyzed populations.
    y_raster    = GID_indices[spike_indices] + lower_cum_n_GIDs[i]
    y_rate      = np.arange(len(rec_spike_GIDs)) + lower_cum_n_GIDs[i]
    #ax0.plot(times*1e-3, y_raster, '.', ms=3., color=colors[i], label=population)
    '''
    # Plot rates and cv_isi for single GIDs
    ax1.barh(y_rate, rates, height=bar_height, color=colors[i], linewidth=0)
    ax3.barh(y_rate, cv_isi_all, height=bar_height, color=colors[i], linewidth=0)
    '''

    ax1.barh(y_mean[i], rates_mean[i], height=bar_height, color=colors[i], linewidth=0)
    ax3.barh(y_mean[i], cv_isi_mean[i], height=bar_height, color=colors[i], linewidth=0)

# Raster Plot
xlim = (0, t_sim * 1e-3)
ylim = (0, sum(n_rec_spike))
# set yticks to center of recorded n spikes of one layer
yticks = lower_cum_n_GIDs[::n_types] + 0.5 * n_rec_spike_layer
ax0.set_yticks(yticks)
ax0.set_yticklabels(layers)
#ax0.set_xlabel('simulation time / s')
ax0.set_ylabel('Layer')
ax0.set_xlim(*xlim)
ax0.set_ylim(*ylim)
ax0.grid(False)

# Rates
ax1.set_yticks(yticks_mean)
ax1.set_yticklabels(layers)
ax1.set_xlabel('firing rate / Hz')
#ax1.set_ylabel('Layer')
ax1.set_ylim(*ylim_mean)
#ax1.set_xlim(0, 20)
ax1.grid(False)

# Synchrony: Instantaneous rates
sum_rate_inst = np.sum(rate_inst_all, 0)
ax2.bar(t_edges, sum_rate_inst, width=bin_width*1e-3, 
    fc='red', ec='red', linewidth=0, fill=True, alpha=0.2, zorder=1)
ax2.set_xlabel('simulation time / s')
ax2.set_ylabel('Counts')
ax2.set_xlim(*xlim)
ax2.grid(False)

# CV of ISI
ax3.set_yticks(yticks_mean)
ax3.set_yticklabels(layers)
ax3.set_xlabel('CV of interspike intervals / Hz')
#ax3.set_ylabel('Layer')
ax3.set_ylim(*ylim_mean)
#ax3.set_xlim(0, 20)
ax3.grid(False)

# Legend; order is reversed, such that labels appear correctly
for i in range(n_types):
    ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
ax1.legend(loc='best')

for ax in fig.axes:
    style.fixticks(ax)
fig.savefig(data_path + 'plot_' + sim_spec + picture_format)

fig.show()

