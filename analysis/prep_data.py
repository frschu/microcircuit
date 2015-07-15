"""prep_data.py
    
    Calculates date for further analysis;

    Specifically: 
        population activity, mean rates. cv_isi, synchrony.
        histogram of membrane potentials

    Writes data to ..._res.hdf5 file.
"""
import numpy as np
import h5py
import sys, os
import time

######################################################
# File and path
######################################################
data_file = "micro"
data_sup_path = "/export/data-schuessler/data_microcircuit/"
#data_sup_path    = "/users/schuessler/uni/microcircuit/data"
data_path = os.path.join(data_sup_path, data_file)
#sim_spec = "a1.0_t20.4_00"
#sim_spec = "a1.0_t20.2_fixindeg_01" 
sim_spec = "spon_act_10_times"

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
area          = data_file.attrs["area"]   
t_sim         = data_file.attrs["t_sim"]  
t_trans       = data_file.attrs["t_trans"]
dt            = data_file.attrs["dt"]    
populations   = data_file.attrs["populations"].astype("|U4")
layers        = data_file.attrs["layers"].astype("|U4")        
types         = data_file.attrs["types"].astype("|U4")     
n_populations = data_file.attrs["n_populations"]
n_layers      = data_file.attrs["n_layers"]       
n_types       = data_file.attrs["n_types"] 

t_measure = t_sim - t_trans

# Pass data to res_file:
res_file.attrs["area"]             = data_file.attrs["area"]   
res_file.attrs["t_sim"]            = data_file.attrs["t_sim"]  
res_file.attrs["t_trans"]          = data_file.attrs["t_trans"]
res_file.attrs["dt"]               = data_file.attrs["dt"]    
res_file.attrs["populations"]      = data_file.attrs["populations"]
res_file.attrs["layers"]           = data_file.attrs["layers"]       
res_file.attrs["types"]            = data_file.attrs["types"]     
res_file.attrs["n_populations"]    = data_file.attrs["n_populations"]
res_file.attrs["n_layers"]         = data_file.attrs["n_layers"]       
res_file.attrs["n_types"]          = data_file.attrs["n_types"] 

######################################################
# Spikes and membrane potentials
######################################################
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
        print("spikes")
        # Data
        grp = data_file[sim_spec2 + "/spikes"]
        dt = grp.attrs["dt"]
        n_neurons_rec_spike = grp.attrs["n_neurons_rec_spike"][:]

        # Mean and Std of firing rates and CV of ISI
        rates_mean  = np.zeros(n_populations)
        rates_std   = np.zeros(n_populations)
        cv_isi_mean = np.zeros(n_populations)
        cv_isi_std  = np.zeros(n_populations)
        synchrony   = np.zeros(n_populations)
        n_rec_spikes = np.zeros(n_populations)
        hist_spikes = np.zeros((n_populations, n_bins_spikes))
        
        for i, population in enumerate(populations):
            print(population)
            # Get data
            subgrp = grp[str(population)]
            raw_times_all   = subgrp["times"][:] * dt * 1e-3 # in seconds
            indices         = subgrp["rec_neuron_i"][:]
            
            rates           = []
            cv_isi_all      = []
            no_isi          = 0
            hist_spikes_i   = np.zeros(n_bins_spikes)
            
            for j in range(n_neurons_rec_spike[i]):
                times = raw_times_all[indices[j]:indices[j+1]]
                times = times[times > t_trans] # ignore transitional period!
                
                # histogram, isi
                n_spikes = len(times)
                rates.append(n_spikes / t_measure) # Hz; single neuron firing rate
                hist_spikes_i += np.histogram(times, bins=n_bins_spikes, range=(t_trans, t_sim), density=False)[0]
                if n_spikes > 2:
                    isi         = np.diff(times)
                    mean_isi    = np.mean(isi)
                    var_isi     = np.var(isi)
                    cv_isi      = var_isi / mean_isi**2
                    cv_isi_all.append(cv_isi)
                else:
                    no_isi += 1
                
            rates = np.array(rates)
            cv_isi_all = np.array(cv_isi_all)

            # Means
            rates_mean[i]   = np.mean(rates)
            rates_std[i]    = np.std(rates)
            cv_isi_mean[i]  = np.mean(cv_isi_all)
            cv_isi_std[i]   = np.std(cv_isi_all)
            synchrony[i]    = np.var(hist_spikes_i) / np.mean(hist_spikes_i)
            hist_spikes[i]  = hist_spikes_i

            # Save single rates and CV_ISI
            res_grp.create_dataset("single_rates/" + str(population), data=rates)
            res_grp.create_dataset("single_cv_isi/" + str(population), data=cv_isi_all)
            
        res_grp.create_dataset("rates_mean", data=rates_mean)
        res_grp.create_dataset("rates_std", data=rates_std)
        res_grp.create_dataset("cv_isi_mean", data=cv_isi_mean)
        res_grp.create_dataset("cv_isi_std", data=cv_isi_std)
        res_grp.create_dataset("synchrony", data=synchrony)
        res_grp.create_dataset("n_neurons_rec_spike", data=n_neurons_rec_spike)
        res_grp.create_dataset("hist_spikes", data=hist_spikes)
        dset_hist_times = res_grp.create_dataset("hist_times", data=bin_edges_spikes)
        dset_hist_times.attrs["bin_size"] = bin_width_spikes
    
    t1 = time.time()
    t_spikes = t1 - t0
    ######################################################
    # Membrane potentials
    ######################################################
    if "voltage" in data_file[sim_spec2]:
        print("voltage")
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
            for j in range(n_hist):
                volt_histo_single[i, j] = np.histogram(volt_all[j], bin_edges_volt,
                                                       density=False)[0]
            volt_histo_means[i] = np.histogram(volt_all, bin_edges_volt, density=False)[0]

        dset_times_volt     = res_grp.create_dataset("times_volt", data=times_volt)
        dset_times_volt.attrs["t_min_volt"] = t_min_volt * 1e-3 # s
        dset_times_volt.attrs["t_max_volt"] = t_max_volt * 1e-3 # s
        dset_times_volt.attrs["dt_volt"] = dt_volt * 1e-3 # s
        res_grp.create_dataset("volt_plot", data=volt_plot)
        res_grp.create_dataset("volt_histo_means", data=volt_histo_means)
        res_grp.create_dataset("volt_histo_single", data=volt_histo_single)
        res_grp.attrs["n_hist_max"] = n_hist_max
        res_grp.attrs["V_min"] = V_min
        res_grp.attrs["V_max"] = V_max
        res_grp.attrs["n_bins_volt"] = n_bins_volt
        res_grp.attrs["n_neurons_rec_voltage"] = volt_grp.attrs["n_neurons_rec_voltage"]

        t_volt = time.time() - t1
        print("Time for voltage   : ", t_volt)
    
    t_calc = time.time() - t0
    print("Time for spikes    : ", t_spikes)
    print("Total time for calc: ", t_calc)
          
data_file.close()
res_file.close()

print("sim_spec = " + sim_spec)
# Free memory from its chains
raw_times_all = None
times = None

