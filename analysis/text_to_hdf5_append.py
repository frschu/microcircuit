"""text_to_hdf5_append.py
   
    Transforms raw simulation data of sli simulation in hdf5 file
    with structure equivalent to the one of the pyNEST simulations. 

    Appends the data to existing file.
"""
from __future__ import print_function
from imp import reload
import numpy as np
import h5py
import sys, os, time
sys.path.append(os.path.abspath('../')) # include path with style
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specifications
import sim_params as sim; reload(sim)
import model_class; reload(model_class)
######################################################
def get_GIDs_times(model, population, data_path, t_trans):
    """
    Get spike data from files.

    Expects: population, data_path, t_trans
    Option: sli (boolean) -- Get data of sli simulation.

    Returns GIDs and times (ordered by times)
    """
    GIDs = []
    times = []
    pop_id      = np.where(model.populations == population)[0][0]
    type_id     = int(pop_id % len(model.types))
    layer_id    = int(pop_id / len(model.types))
    prefix = "spikes_" + str(layer_id) + "_" + str(type_id)
    file_names_all  = os.listdir(data_path)
    file_names = [file_name for file_name in file_names_all if file_name.startswith(prefix)]
    for file_name in file_names:
        with open(os.path.join(data_path, file_name), "r") as file:
            for line in file:
                GID, time = line.split()
                time = float(time)
                if time >= t_trans:
                    GIDs.append(int(GID))
                    times.append(time)
    GIDs = np.array(GIDs)
    times = np.array(times) 
    
    mask = np.argsort(times)
    GIDs = GIDs[mask]
    times = times[mask]
    return GIDs, times


def save_sli_to_hdf5(sim_spec="a1.0_t60.2", start_file=False):
    n_rec   =  1000
    t_sim   = 60200.0 # ms
    t_trans =   200.0 # ms
    
    # Model
    connection_rule = "fixed_total_number" # "fixed_indegree", "fixed_total_number"
    PSC_rel_sd      = 0.1 # 0.1 for  Potjans' model
    model           = model_class.model(connection_rule=connection_rule,
                                        PSC_rel_sd=PSC_rel_sd) 
    # Parameters
    populations = model.populations
    # record from 1000 neurons
    n_neurons_rec_spike = (np.ones_like(model.n_neurons) * n_rec).astype(int)
    n_neurons_rec_voltage = (np.ones_like(model.n_neurons) * 0).astype(int)
    
    # Data path
    sub_path        = "sli"
    data_sup_path = sim.data_dir
    data_path = os.path.join(data_sup_path, sub_path)
    file_name   = sim_spec + "_all.hdf5"
    
    sli_data_path = os.path.join(data_path, sim_spec)
    path_hdf5_file = os.path.join(data_path, file_name)

    if start_file: # overwrites existing file!
        write_mode = "w"
    else:
        write_mode = "r+"
    
    with h5py.File(path_hdf5_file, write_mode) as data_file:
        # Get last group number
        max_grp = 0
        if write_mode == "r+":
            for key in data_file.keys():
                max_grp = max(max_grp, int(key))
    
        print("new group: ", max_grp)
        grp         = data_file.create_group(str(max_grp + 1))
        spikes_grp = grp.create_group("spikes")
        spikes_grp.attrs["dt"]  = sim.dt 
        spikes_grp.attrs["info"]  = "times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]"
        spikes_grp.attrs["info2"]  = "times in units of dt; dt in ms  =>  times/ms = times * dt"
        spikes_grp.attrs["n_neurons_rec_spike"] = n_neurons_rec_spike
    
        for j, population in enumerate(populations):
            senders, times = get_GIDs_times(model, population, sli_data_path, t_trans)
            times   = np.uint(times / sim.dt) # in units of dt!
    
            # Create array of indices for data: 
            # times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]
            n_spikes_per_neuron = np.zeros(n_neurons_rec_spike[j])
            rec_neuron_i        = np.zeros(n_neurons_rec_spike[j] + 1)
    
            # Get corresponding reduced GIDs: nth neuron recorded
            n_spikes_fired = np.unique(senders, return_counts=True)[1]
            max_index       = len(n_spikes_fired) 
            n_spikes_per_neuron[:max_index] = n_spikes_fired
            n_spikes_per_neuron = np.random.permutation(n_spikes_per_neuron)
            nth_neuron      = np.cumsum(n_spikes_per_neuron)
            rec_neuron_i[1 : ] = nth_neuron        # leave out 0th index
    
            # sort times
            sorted_times = times[np.argsort(senders)] 
            for i in range(len(rec_neuron_i) - 1):
                i0, i1 = (rec_neuron_i[i], rec_neuron_i[i+1])
                sorted_times[i0:i1] = np.sort(sorted_times[i0:i1])
    
            # save data to HDF5 file:
            spikes_subgrp   = spikes_grp.create_group(population)
            if len(sorted_times) > 0:
                dset_times      = spikes_subgrp.create_dataset("times", data=sorted_times)
            else:
                dset_times      = spikes_subgrp.create_dataset("times", data=np.array([0]))
            dset_indices    = spikes_subgrp.create_dataset("rec_neuron_i", data=rec_neuron_i)
