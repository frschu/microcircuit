"""prep_data.py
    
    Calculates date for further analysis;

    Specifically: 
        raster plot data, population activity, mean rates. cv_isi, synchrony.
        histogram of membrane potentials

    Writes data to ..._res.hdf5 file.
"""
from imp import reload
import numpy as np
import h5py
import sys, os
import time
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specifications
# Import specific moduls
import sim_params as sim; reload(sim)

reverse_order = True # do analysis such that plots resemble those of the paper (starting with L6i)

######################################################
# File and path
######################################################
data_path = os.path.join(sim.data_dir, "vary_d")
data_path = os.path.join(sim.data_dir, "micro")
sim_spec = "a1.0_t20.4_totalN_00"

# Original data
file_name  = sim_spec + ".hdf5"  
res_file_name = sim_spec + "_res.hdf5"

# Open file: data and results
data_file = h5py.File(os.path.join(data_path, file_name), "r")
res_file = h5py.File(os.path.join(data_path, res_file_name), "w")


######################################################
# Basic data
######################################################
# Simulation attributes
t_sim   = data_file.attrs["t_sim"]  
t_trans = data_file.attrs["t_trans"]
dt      = data_file.attrs["dt"]    
populations   = data_file.attrs["populations"].astype("|U4")
n_populations = data_file.attrs["n_populations"]

t_measure = t_sim - t_trans

if reverse_order:
    populations = populations[::-1]


######################################################
# Spikes and membrane potentials
######################################################
# Raster plot
t_min_raster = 0.2
t_max_raster = min(t_sim, 1)
max_plot = 0.1 # part of recorded neurons plotted in raster plot

# Spike histogram
bin_width_spikes = dt * 1e-3  # s
n_bins_spikes    = int(t_measure / bin_width_spikes) 
bin_edges_spikes = np.arange(t_trans, t_sim, bin_width_spikes)

# Membrane potential parameters
n_bins_volt   = 200
V_min = -100
V_max = -50
bin_edges_volt = np.linspace(V_min, V_max, n_bins_volt + 1)
n_hist_max  = 10 # maximum number of single histogram to be shown

for sim_spec2 in data_file.keys():
    print(sim_spec2)
    t0 = time.time()
    # Results
    res_grp = res_file.create_group(sim_spec2)

    ######################################################
    # Analyze spikes
    ######################################################
    if "spikes" in data_file[sim_spec2]:
        # Data
        grp = data_file[sim_spec2 + "/spikes"]
    
        res_raster = res_grp.create_group("raster") 
        dt = grp.attrs["dt"]
        n_neurons_rec_spike = grp.attrs["n_neurons_rec_spike"][:]
    
        if reverse_order:
            n_neurons_rec_spike = n_neurons_rec_spike[::-1]
        offsets = np.append([0], np.cumsum(n_neurons_rec_spike)) * max_plot
        
        # Mean and Std of firing rates and CV of ISI
        rates_mean  = np.zeros(n_populations)
        rates_std   = np.zeros(n_populations)
        cv_isi_mean = np.zeros(n_populations)
        cv_isi_std  = np.zeros(n_populations)
        synchrony   = np.zeros(n_populations)
        n_rec_spikes = np.zeros(n_populations)
        hist_spikes = np.zeros((n_populations, n_bins_spikes))
        no_isi = []
        
        for i, population in enumerate(populations):
            res_raster_pop = res_raster.create_group(str(population))
            
            # Get data
            subgrp = grp[str(population)]
            raw_times_all   = subgrp["times"][:] * dt * 1e-3 # in seconds
            indices         = subgrp["rec_neuron_i"][:]
            n_rec_spikes_i  = len(indices) - 1
            
            # Firing rate:
            n_spikes = np.diff(indices)
            rates = n_spikes / t_measure # Hz
    
            cv_isi_all = np.empty(0)    
            rates = []
            hist_spikes_i = np.zeros(n_bins_spikes)
            
            for j in range(n_rec_spikes_i):
                times = raw_times_all[indices[j]:indices[j+1]]
                times = times[times > t_trans]
                
                # raster, histogram, etc
                n_spikes = len(times)
                rates.append(n_spikes / t_measure)
                hist_spikes_i += np.histogram(times, bins=n_bins_spikes, range=(t_trans, t_sim), density=False)[0]
                if n_spikes > 1:
                    isi         = np.diff(times)
                    mean_isi    = np.mean(isi)
                    var_isi     = np.var(isi)
                    cv_isi      = var_isi / mean_isi**2
                    cv_isi_all  = np.append(cv_isi_all, cv_isi)
                else:
                    no_isi.append(str(population) + '_' + str(j))
                
                # data for raster plot
                if j < len(indices) * max_plot:
                    raster_mask         = (times > t_min_raster) * (times < t_max_raster)
                    n_spikes_raster     = min(n_spikes, sum(raster_mask))
                    neuron_ids_raster   = [j]*n_spikes_raster + offsets[i]
                    raster_data         = np.vstack((times[raster_mask], neuron_ids_raster))
                    res_raster_pop.create_dataset(str(j), data=raster_data)
    
            # Means
            rates_mean[i]   = np.mean(rates)
            rates_std[i]    = np.std(rates)
            cv_isi_mean[i]  = np.mean(cv_isi_all)
            cv_isi_std[i]   = np.std(cv_isi_all)
            synchrony[i]    = np.var(hist_spikes_i) / np.mean(hist_spikes_i)
            n_rec_spikes[i] = n_rec_spikes_i
            hist_spikes[i]  = hist_spikes_i
            
        res_raster.attrs["t_min_raster"] = t_min_raster
        res_raster.attrs["t_max_raster"] = t_max_raster
        res_raster.attrs["ymax_raster"] = offsets[-1]
        res_raster.attrs["yticks"] = (offsets[1:] - offsets[:-1]) * 0.5 + offsets[:-1]
            
        res_grp.create_dataset("rates_mean", data=rates_mean)
        res_grp.create_dataset("rates_std", data=rates_std)
        res_grp.create_dataset("cv_isi_mean", data=cv_isi_mean)
        res_grp.create_dataset("cv_isi_std", data=cv_isi_std)
        res_grp.create_dataset("synchrony", data=synchrony)
        res_grp.create_dataset("n_rec_spikes", data=n_rec_spikes)
        res_grp.create_dataset("hist_spikes", data=hist_spikes)
        dset_hist_times = res_grp.create_dataset("hist_times", data=bin_edges_spikes)
        dset_hist_times.attrs["bin_size"] = bin_width_spikes
    

    ######################################################
    # Membrane potentials
    ######################################################
    if "voltage" in data_file[sim_spec2]:
        # Data
        volt_grp = data_file[sim_spec2 + "/voltage"]
        
        # Times
        dt_volt = volt_grp.attrs["dt_volt"]
        t_min_volt  = volt_grp.attrs["t_min"]
        t_max_volt  = volt_grp.attrs["t_max"]
        times_volt  = np.arange(t_min_volt, t_max_volt, dt_volt) *1e-3 # s
        
        volt_plot         = np.zeros((n_populations, n_hist_max, len(times_volt)))    
        volt_histo_means     = np.zeros((n_populations, n_bins_volt))
        volt_histo_single    = np.zeros((n_populations, n_hist_max, n_bins_volt))    
        for i, population in enumerate(populations):
            print(population)
            # Get membrane potentials
            volt_all = volt_grp[population][:]
            n_hist = min(n_hist_max, len(volt_all))
            volt_plot[i][:n_hist] = volt_all[:n_hist]
            
            volt_histo_means[i] = np.histogram(volt_all, bin_edges_volt, density=True)[0]
            for j in range(n_hist):
                volt_histo_single[i, j] = np.histogram(volt_all[j], bin_edges_volt, density=True)[0]
    
        dset_times_volt     = res_grp.create_dataset("times_volt", data=times_volt)
        dset_times_volt.attrs["t_min_volt"] = t_min_volt * 1e-3 # s
        dset_times_volt.attrs["t_max_volt"] = t_max_volt * 1e-3 # s
        res_grp.create_dataset("volt_plot", data=volt_plot)
    
        dset_volt           = res_grp.create_dataset("volt_histo_means", data=volt_histo_means)
        dset_volt.attrs["V_min"] = V_min
        dset_volt.attrs["V_max"] = V_max
        dset_volt.attrs["n_bins_volt"] = n_bins_volt
        dset_volt.attrs["n_hist_max"] = n_hist_max
        res_grp.create_dataset("volt_histo_single", data=volt_histo_single)
    
    t_calc = time.time() - t0
    print("Time for calculation: ", t_calc)
          
#data_file.close()
res_file.close()

# Free memory from its chains
raw_times_all = None
times = None
raster_mask = None
n_spikes_raster = None
neuron_ids_raster = None
raster_data = None

