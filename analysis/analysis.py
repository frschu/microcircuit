'''
    analysis.py
    
    Further command line arguments:
        c       script will close all open plots
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Produces raster plots and calculates firing ratios
    
    Notes:
    # THIS IS NOT GOOD: parameters in net and sim must remain unchanged...

'''
from __future__ import print_function
from imp import reload
from scipy import signal
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
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import functions_analysis as functions; reload(functions)

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)
plotting = True
choose_population = 'all'
#choose_population = 'L4e' # 'all' or [names]

# Data path
data_sup_path = user.data_dir
file_name = 'a0.1_t1.2_00/'
print(file_name)

simulation_path = data_sup_path + file_name
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


# Simulation time
t_sim = sim.t_sim
T0 = sim.t_trans # ms; starting point of analysis (avoid transients)
t_measure = sim.t_measure

# Further derived paramters
lower_GIDs = np.insert(np.cumsum(n_rec_spike)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spike_layer = np.sum(n_rec_spike.reshape(n_layers, n_types), 1) # recorded spikes per layer

# Mean and Std of firing rates and CV of ISI
rates_mean  = np.zeros(n_populations)
rates_std   = np.zeros(n_populations)
cv_isi_mean = np.zeros(n_populations)
cv_isi_std  = np.zeros(n_populations)
# Spike histogram
bin_width_spikes = 10.  # ms
n_bins_spikes = int(t_measure / bin_width_spikes) 
hist_spikes = np.zeros([n_populations, n_bins_spikes])
bin_edges_spikes = np.histogram([], bins=n_bins_spikes, range=(0, t_measure))[1]
t_edges_spikes = bin_edges_spikes[:-1] * 1e-3
rate_smooth = np.zeros((n_populations, n_bins_spikes))
# Rates
max_rate = 30
n_bins_rate = 40
hist_rate = np.zeros([n_populations, n_bins_rate])
bin_edges_rate = np.histogram([], bins=n_bins_rate, range=(0, max_rate))[1]
# ISI
bin_width_isi = 2.  # ms
n_bins_isi = int(t_measure / bin_width_isi) 
hist_isi = np.zeros([n_populations, n_bins_isi])
bin_edges_isi = np.histogram([], bins=n_bins_isi, range=(0, t_measure))[1]
t_edges_isi = bin_edges_isi[:-1] * 1e-3
no_isi = []


# Choose population
if choose_population == 'all':
    print('analyze all populations')
elif choose_population in populations:
    populations = [choose_population]
    print('analyze only ' + choose_population)
else: 
    print('Expected different parameter: choose_population')
    print('analyze all populations')


# Plotting: Prepare figures
if plotting:
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
    times_file = 'times_' + population + '.npy'
    n_GIDs_file = 'n_GIDs_' + population + '.npy'
    times_all   = np.load(npy_path + times_file)
    n_GIDs  = np.load(npy_path + n_GIDs_file)
    
    # Firing rate:
    n_spikes = np.diff(n_GIDs)
    rates = n_spikes / t_measure * 1e3 # Hz
    hist_rate[i] += np.histogram(rates, bins=n_bins_rate, range=(0, max_rate))[0]

    
    cv_isi_all = np.empty(0)
    for j in range(len(n_GIDs) - 1):
        times = times_all[n_GIDs[j]:n_GIDs[j+1]]
        hist_spikes[i] += np.histogram(times, bins=n_bins_spikes, range=(T0, t_sim))[0]
        if n_spikes[j] > 1:
            isi = np.diff(times)
            hist_isi[i] += np.histogram(isi, bins=n_bins_isi, range=(0, t_measure))[0]
            mean_isi = np.mean(isi)
            var_isi = np.var(isi)
            cv_isi = var_isi / mean_isi**2
            cv_isi_all = np.append(cv_isi_all, cv_isi)
        else:
            no_isi.append(population + '_' + str(j))
        if plotting:
            ax0.plot(times*1e-3, [j + lower_GIDs[i]]*n_spikes[j], 
                '.', ms=3, color=colors[i], label=population)

    # Means
    rates_mean[i] = np.mean(rates)
    rates_std[i] = np.std(rates)
    cv_isi_mean[i] = np.mean(cv_isi_all)
    cv_isi_std[i] = np.std(cv_isi_all)
   
    # Synchrony
    # Histogram over time
    # convolve 
    n_box = 10
    bc = signal.boxcar(n_box)
    rate_smooth[i] = signal.convolve(hist_spikes[i], bc, mode='same') / n_box # normed convolution

    # Plotting
    if plotting:
        print('Plotting')
        #ax1.barh(y_mean[i], rates_mean[i], height=bar_height, color=colors[i], linewidth=0)
        #ax2.bar(t_edges_spikes, rate_smooth[i], width=bin_width_spikes*1e-3, 
        #    fc=colors[i], ec=colors[i], linewidth=0.2, fill=False)
        #ax3.barh(y_mean[i], cv_isi_mean[i], height=bar_height, color=colors[i], linewidth=0)

        ax1.barh(bin_edges_rate[:-1], hist_rate[i], height=bar_height, color=colors[i], linewidth=0)
        ax3.barh(t_edges_isi, hist_isi[i], height=bar_height, color=colors[i], linewidth=0)

if plotting:
    # Raster Plot
    xlim = (0, t_sim * 1e-3)
    ylim = (0, sum(n_rec_spike))
    # set yticks to center of recorded n spikes of one layer
    yticks = lower_GIDs[::n_types] + 0.5 * n_rec_spike_layer
    ax0.set_yticks(yticks)
    ax0.set_yticklabels(layers)
    ax0.set_xlabel('simulation time / s')
    ax0.set_ylabel('Layer')
    ax0.set_xlim(*xlim)
    ax0.set_ylim(*ylim)
    ax0.grid(False)
    
    # Rates
    ax1.set_yticks(yticks_mean)
    ax1.set_yticklabels(layers)
    ax1.set_xlabel('firing rate / Hz')
    #ax1.set_ylabel('Layer')
    #ax1.set_ylim(*ylim_mean)
    #ax1.set_xlim(0, 20)
    ax1.grid(False)
    
    # Synchrony: Instantaneous rates
    sum_rate_inst = np.sum(rate_smooth, 0)
    ax2.bar(t_edges_spikes, sum_rate_inst, width=bin_width_spikes*1e-3, 
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
    #ax3.set_ylim(*ylim_mean)
    #ax3.set_xlim(0, 20)
    ax3.grid(False)
    
    # Legend; order is reversed, such that labels appear correctly
    for i in range(n_types):
        ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
    ax1.legend(loc='best')
    
    for ax in fig.axes:
        style.fixticks(ax)
    fig_name = 'rates_etc'
    if sli:
        fig_name += '_sli'
    fig.savefig(figure_path + fig_name + picture_format)
    
    fig.show()
