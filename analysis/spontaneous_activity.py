"""spontaneous_activity.py

    Further command line arguments:
        c       script will close all open plots
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Overview over all populations: Raster plot, mean rates, mean CV of ISI per population.
"""
from imp import reload
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import sys, os
sys.path.append(os.path.abspath("../")) # include path with style
sys.path.append(os.path.abspath("../presentation/")) # include path with simulation specifications
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specifications
import pres_style as style; reload(style)
# Close other plots by adding "c" after "run <script>" 
if "c" in sys.argv:
    plt.close("all")
picture_format = ".pdf"
figure_path = "./figures"
######################################################
# Import specific moduls
import network_params as net; reload(net)
import user_params as user; reload(user)

def enum(arr1, *args):
    i_range = range(len(arr1))
    return zip(i_range, arr1 ,*args)

def print_perc(i, i_total):
    i_perc = round(100 * i / i_total)
    print("%i"%(i_perc), end="\r")

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)
show_fig = False
save_fig = True
xfactor = 2.6
rcParams["figure.figsize"] = (xfactor*4, xfactor*5.)  
figure_path = os.path.join(".", "figures")


#####################################################################
# Data path
data_path = user.data_dir
sim_spec = user.simulation_spec
print(data_path)
print(sim_spec)

# Open file: results
res_file_name = sim_spec + "_res.hdf5"
res_file = h5py.File(os.path.join(data_path, res_file_name), "w")
res_grp = res_file.create_group("micro")
res_raster = res_grp.create_group("raster") 
    
# Paths
simulation_path = os.path.join(data_path, sim_spec)
pynest_npy_path = os.path.join(simulation_path, "npy_data") 
if "sli" in sys.argv:
    sli = True
    npy_path = os.path.join(simulation_path, "npy_data_sli")
else:
    sli = False
    npy_path = pynest_npy_path
if not os.path.exists(figure_path):
    os.mkdir(figure_path)

# Get data specified for pynest simulation
populations = net.populations
layers = net.layers
types= net.types
n_populations = len(populations)
n_layers = len(layers)
n_types = len(types)
n_rec_spike = np.load(os.path.join(pynest_npy_path, "n_neurons_rec_spike.npy"))
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
area    = float(sim_spec.split("_")[0][1:])          # mm**2
t_sim   = float(sim_spec.split("_")[1][1:])          # s
t_trans = 0.2  # s; starting point of analysis (avoid transients)
t_measure = t_sim - t_trans

# Further derived paramters
lower_GIDs = np.insert(np.cumsum(n_rec_spike)[:-1], 0, 0)    # lower boundaries incl. zero
n_rec_spike_layer = np.sum(n_rec_spike.reshape(n_layers, n_types), 1) # recorded spikes per layer

# Mean and Std of firing rates and CV of ISI
rates_mean  = np.zeros(n_populations)
rates_std   = np.zeros(n_populations)
cv_isi_mean = np.zeros(n_populations)
cv_isi_std  = np.zeros(n_populations)
synchrony   = np.zeros(n_populations)
no_isi = []

#####################################################################
# Raster plot
t_min_raster = 0.2           # s
t_max_raster = min(t_sim, 1) # s
max_plot = 1.0 # part of recorded neurons plotten in raster plot
offsets = np.append([0], np.cumsum(n_rec_spike)) * max_plot
    
spikes_raster = []
times_raster = []

# Spike histogram
bin_width_spikes = 3e-3  # s
n_bins_spikes = int(t_measure / bin_width_spikes) 

print("Prepare data")
for i, population in enumerate(populations):
    res_raster_pop = res_raster.create_group(str(population))
    
    spikes_raster.append([])
    times_raster.append([])
    # Get data
    times_file = "times_" + population + ".npy"
    n_GIDs_file = "n_GIDs_" + population + ".npy"
    raw_times_all   = np.load(os.path.join(npy_path, times_file)) * 1e-3 # in seconds
    n_GIDs  = np.load(os.path.join(npy_path, n_GIDs_file))
    

    
    # Firing rate:
    n_spikes = np.diff(n_GIDs)
    rates = n_spikes / t_measure # Hz

    cv_isi_all  = np.empty(0)
    hist_spikes = np.zeros(n_bins_spikes)
    for j in range(len(n_GIDs) - 1):
        times = raw_times_all[n_GIDs[j]:n_GIDs[j+1]]
        hist_spikes += np.histogram(times, bins=n_bins_spikes, range=(t_trans, t_sim))[0]
        if n_spikes[j] > 1:
            isi = np.diff(times)
            mean_isi = np.mean(isi)
            var_isi = np.var(isi)
            cv_isi = var_isi / mean_isi**2
            cv_isi_all = np.append(cv_isi_all, cv_isi)
        else:
            no_isi.append(population + "_" + str(j))
                    
        # data for raster plot
        if j < len(n_GIDs) * max_plot:
            raster_mask = (times > t_min_raster) * (times < t_max_raster)
            n_spikes_raster = min(n_spikes[j], sum(raster_mask))
            neuron_ids_raster = [j]*n_spikes_raster + offsets[i]
            raster_data = np.vstack((times[raster_mask], neuron_ids_raster))
            res_raster_pop.create_dataset(str(j), data=raster_data)
            
            times_raster[i].append(times[raster_mask])
            spikes_raster[i].append(np.array([j + lower_GIDs[i]]*n_spikes_raster))

    # Means
    rates_mean[i]   = np.mean(rates)
    rates_std[i]    = np.std(rates)
    cv_isi_mean[i]  = np.mean(cv_isi_all)
    cv_isi_std[i]   = np.std(cv_isi_all)
    synchrony[i]    = np.var(hist_spikes) / np.mean(hist_spikes)

    
res_raster.attrs["t_min_raster"] = t_min_raster
res_raster.attrs["t_max_raster"] = t_max_raster
res_raster.attrs["ymax_raster"] = offsets[-1]
res_raster.attrs["yticks"] = (offsets[1:] - offsets[:-1]) * 0.5 + offsets[:-1]
        
res_grp.create_dataset("rates_mean", data=rates_mean)
res_grp.create_dataset("rates_std", data=rates_std)
res_grp.create_dataset("cv_isi_mean", data=cv_isi_mean)
res_grp.create_dataset("cv_isi_std", data=cv_isi_std)
res_grp.create_dataset("synchrony", data=synchrony)

res_file.close()

# Free memory from its chains
raw_times_all = None
times = None
raster_mask = None
n_spikes_raster = None
spikes_raster = None
raster_data = None



######################################################################
# Plotting
######################################################################
print("Plotting")

# Open file: results
data_path = user.data_dir
sim_spec = user.simulation_spec
res_file_name = sim_spec + "_res.hdf5"
res_file = h5py.File(os.path.join(data_path, res_file_name), "r")
res_grp = res_file["micro"]
res_raster = res_grp["raster"]
    
rates_mean   = res_grp["rates_mean"]
rates_std    = res_grp["rates_std"]
cv_isi_mean  = res_grp["cv_isi_mean"]
cv_isi_std   = res_grp["cv_isi_std"]
synchrony    = res_grp["synchrony"]

fig = plt.figure()
if not save_fig:
    suptitle = "Simulation for: area = %.1f, time = %ims"%(area, t_sim)
    suptitle += "\nfile: " + sim_spec
    if sli: 
        suptitle += "  SLI"
    fig.suptitle(suptitle, y=0.98)

# Raster plot
ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=3)
# Rates
ax1 = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=1)
# CV of interspike interval (ISI)
ax2 = plt.subplot2grid((3, 2), (1, 1), colspan=1, rowspan=1)
# Synchrony
ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=1, rowspan=1)

y_mean = np.arange(n_populations) + 0.1
bar_height = 0.8 

for i, population in enumerate(populations):
    print(population)
    res_raster_pop = res_raster[str(population)]
    for times, neuron_ids in res_raster_pop.values():
        ax0.plot(times, neuron_ids, '.', ms=3, color=colors[i], label=population)
    #for times, spikes in zip(times_raster[i], spikes_raster[i]):
    #    ax0.plot(times, spikes, ".", ms=3, color=colors[i], label=population)
    ax1.barh(y_mean[i], rates_mean[i],  height=bar_height, color=colors[i], linewidth=0)
    ax2.barh(y_mean[i], cv_isi_mean[i], height=bar_height, color=colors[i], linewidth=0)
    ax3.barh(y_mean[i], synchrony[i],   height=bar_height, color=colors[i], linewidth=0)

ylim_mean = (0, n_populations)
yticks_mean = np.arange(n_types * 0.5, n_populations, n_types)

# Raster Plot
t_min_raster = res_raster.attrs["t_min_raster"]
t_max_raster = res_raster.attrs["t_max_raster"]
ymax_raster = res_raster.attrs["ymax_raster"]
yticks = res_raster.attrs["yticks"]
xlim = (t_min_raster, t_max_raster)
ylim = (0, ymax_raster)
ax0.set_yticks(yticks)
ax0.set_yticklabels(populations)
ax0.set_xlabel('simulation time / s')
ax0.set_ylabel('Layer')
ax0.set_xlim(*xlim)
ax0.set_ylim(*ylim)
ax0.grid(False)

# Rates
ax1.set_xlabel("firing rate / Hz")
# CV of ISI
ax2.set_xlabel("CV of interspike intervals")
ax2.set_xlim(0, 1)
# Synchrony
ax3.set_xlabel("Synchrony")

for ax in (ax1, ax2, ax3):
    #ax.set_ylabel("Layer")
    ax.set_yticks(yticks_mean)
    ax.set_yticklabels(layers)
    ax.set_ylim(*ylim_mean)
    ax.grid(False)
    
# Legend; order is reversed, such that labels appear correctly
for i in range(n_types):
    ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
ax1.legend(loc="best")

for ax in fig.axes:
    style.fixticks(ax)
fig_name = "spon_activity_" + sim_spec
if sli:
    fig_name += "_sli"
fig_name += picture_format

if save_fig:
    print("save figure to " + fig_name)
    fig.savefig(os.path.join(figure_path,fig_name))

if show_fig:
    fig.show()
   
res_file.close()
