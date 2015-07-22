"""prep_data.py
    
    Calculates date for further analysis;

    Specifically: 
        population activity, mean rates. cv_isi, synchrony.
        histogram of membrane potentials

    Writes data to ..._res.hdf5 file.
"""
from __future__ import print_function
import numpy as np
import h5py
import sys, os
import time

######################################################
# File and path
######################################################
data_file = "brunel"
data_sup_path = "/export/data-schuessler/data_microcircuit/"
data_sup_path    = "/users/schuessler/uni/microcircuit/data"
data_path = os.path.join(data_sup_path, data_file)
#sim_spec = "a1.0_t20.4_00"
sim_spec = "a1.0_t20.2_fixindeg_01" 
sim_spec = "test_brunel_C250_exp"
#sim_spec = "spontaneous_activity_sli"
#sim_spec = "spon_act_statistic_sli"
#sim_spec = "simulation_at_mf"

# Original data
file_name  = sim_spec + ".hdf5"  
res_file_name = sim_spec + "_res.hdf5"
# Attributes are not save for sli simulations
if sim_spec.endswith("sli"):
    sim_spec_attrs = sim_spec[:-4]
    #sim_spec_attrs = "spon_act_10_times" # in case the non-sli file is not existing (yet)
else:
    sim_spec_attrs = sim_spec
attrs_file_name = sim_spec_attrs + ".hdf5"

# Open file: data and results
data_file = h5py.File(os.path.join(data_path, file_name), "r")
res_file = h5py.File(os.path.join(data_path, res_file_name), "w")

######################################################
# Basic data
######################################################
with h5py.File(os.path.join(data_path, attrs_file_name), "r") as attrs_file:
    # Simulation attributes
    area          = attrs_file.attrs["area"]   
    t_sim         = attrs_file.attrs["t_sim"]  
    t_trans       = attrs_file.attrs["t_trans"]
    dt            = attrs_file.attrs["dt"]    
    populations   = attrs_file.attrs["populations"].astype("|U4")
    layers        = attrs_file.attrs["layers"].astype("|U4")        
    types         = attrs_file.attrs["types"].astype("|U4")     
    n_populations = attrs_file.attrs["n_populations"]
    n_layers      = attrs_file.attrs["n_layers"]       
    n_types       = attrs_file.attrs["n_types"] 

    t_measure = t_sim - t_trans

    # Pass data to res_file:
    res_file.attrs["area"]             = attrs_file.attrs["area"]   
    res_file.attrs["t_sim"]            = attrs_file.attrs["t_sim"]  
    res_file.attrs["t_trans"]          = attrs_file.attrs["t_trans"]
    res_file.attrs["dt"]               = attrs_file.attrs["dt"]    
    res_file.attrs["populations"]      = attrs_file.attrs["populations"]
    res_file.attrs["layers"]           = attrs_file.attrs["layers"]       
    res_file.attrs["types"]            = attrs_file.attrs["types"]     
    res_file.attrs["n_populations"]    = attrs_file.attrs["n_populations"]
    res_file.attrs["n_layers"]         = attrs_file.attrs["n_layers"]       
    res_file.attrs["n_types"]          = attrs_file.attrs["n_types"] 


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
print_volt_length_warning = False

for k, sim_spec2 in enumerate(data_file.keys()):
    print(sim_spec2)
    t0 = time.time()
    # Results
    res_grp = res_file.create_group(sim_spec2)

    ######################################################
    # Analyze spikes
    ######################################################
    if "spikes" in data_file[sim_spec2]:
        print("spikes")
        print("n neurons fired    0 |    1 spikes:")
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
            print(population, end="")
            # Get data
            subgrp = grp[str(population)]
            raw_times_all   = subgrp["times"][:] * dt * 1e-3 # in seconds
            try:
                indices        = subgrp["rec_neuron_i"][:]
                print("")
            except:
                #print("No rec_neuron_i save -- change this in ../simulation/functions.py!!!")
                # What is about to come is ambiguous! What if neuron i fired its last spike k 
                # at t_i^k and neuron j = i+1 its first spike at t_j_1 > t_i_k???
                # This is very improbable for the given condition: 
                # Of 1000 neurons, 996 fired (using this technique), thus at maximum 4 neurons have been 
                # falsely identified.   
                indices                 = np.zeros(n_neurons_rec_spike[i] + 1) 
                indices_fired_only      = np.where(np.diff(raw_times_all) < 0)[0] + 1
                max_index               = len(indices_fired_only)  # number of neurons that fired >= 1 spike
                indices[1:max_index+1]  = indices_fired_only     # the rest remains zero
                indices[max_index+1:]   = indices_fired_only[-1] # at last the neurons that didn't fire
                if max_index < n_neurons_rec_spike[i]:
                    n_neurons_didnt_fire    = n_neurons_rec_spike[i] - max_index
                    sorted_indices          = np.sort(np.diff(indices_fired_only))
                    n_neurons_fired_once    = int(np.sum(sorted_indices == 1))
                    print("\t\t{0:4d} | {1:4d}".format(n_neurons_didnt_fire, n_neurons_fired_once))

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
                    std_isi      = np.std(isi)
                    cv_isi      = std_isi / mean_isi
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
            if k ==0:
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
        times_volt  = (np.arange(t_min_volt, t_max_volt, dt_volt) + dt_volt) *1e-3 # s
        if len(volt_grp[populations[0]][0]) < len(times_volt):
            print_volt_length_warning = True
            times_volt = times_volt[:-1]
        
        volt_plot            = np.zeros((n_populations, n_hist_max, len(times_volt)))    
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

if print_volt_length_warning:
    print("Warning: Length of volt array too short. Where ever that error comes from...")

