"""functions.py

Functions for simulating, 
applied in: 
simulate_microcircuit.py
simulate_transition.py

Contains:
# Pre-loop
    initialize_data_file
    initialize_seeds
    initialize_info_file
# Functions inside the loop
    prepare_simulation
    derive_parameters
    create_nodes    
    connect         
    save_data
"""
from __future__ import print_function
import nest
import numpy as np
import os
import h5py
# Import specific moduls
from imp import reload
import sim_params as sim; reload(sim)

#######################################################
# Pre-loop functions
#######################################################
def initialize_data_file(sub_path, model, verbose=True, name=None):
    """Creates data_path and file_name for HDF5-file where all data is saved to.

    file_name contains global conditions of simulations: 
    - area, simulated time, thalamus, background, connection_rule
    e.g. 'a1.0_t20.2_th_dc_00.hdf5'

    Further saves attributes to this data_file.
    """
    # Data path
    data_sup_path = sim.data_dir
    data_path = os.path.join(data_sup_path, sub_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # File name
    if name==None:
        sim_spec = "a%.1f_t%.1f"%(model.area, sim.t_sim * 1e-3)
        if not model.n_th == 0:
            sim_spec += "_th"
        if not model.dc_amplitude == 0:
            sim_spec += "_dc"
        if model.connection_rule=="fixed_indegree":
            sim_spec += "_fixindeg"
    else:
        sim_spec = name
    file_name   = sim_spec + "_00.hdf5"
    
    # don't overwrite existing files...
    if file_name in os.listdir(data_path):
        max_n = 0
        for some_file in os.listdir(data_path):
            if some_file.startswith(sim_spec):
                max_n = max(max_n, int(some_file[len(sim_spec)+1: len(sim_spec) + 3])) 
        file_name = sim_spec + "_" + str(max_n + 1).zfill(2) + ".hdf5"
    if verbose: print("Filename: micro/" + file_name)
    
    data_file = h5py.File(os.path.join(data_path, file_name), "w")
    
    # Attributes
    data_file.attrs["area"]     = model.area
    data_file.attrs["t_sim"]    = sim.t_sim*1e-3
    data_file.attrs["t_trans"]  = sim.t_trans*1e-3
    data_file.attrs["n_vp"]     = sim.n_vp
    data_file.attrs["dt"]       = sim.dt
    data_file.attrs["connection_rule"]  = model.connection_rule
    data_file.attrs["populations"]      = model.populations 
    data_file.attrs["layers"]           = model.layers 
    data_file.attrs["types"]            = model.types 
    data_file.attrs["n_populations"]    = model.n_populations 
    data_file.attrs["n_layers"]         = model.n_layers 
    data_file.attrs["n_types"]          = model.n_types 
    data_file.attrs["delay_e"]          = model.delay_e 
    data_file.attrs["delay_i"]          = model.delay_i 
    
    return (data_file, file_name, data_path)

def initialize_seeds():
    """Creates initial master_seed from file or sim_spec (if no previous seed is found)."""
    seed_file   = open(os.path.join(sim.log_path, "seeds.log"), "a+")    # File containing previous seed numbers
    old_seeds   = seed_file.readlines()
    if old_seeds == []:
        master_seed = sim.master_seed
        seed_file.write("seed\t\tdate       time\t\t file\n")
    else:
        last_line   = old_seeds[-1]
        try: 
            master_seed = int(last_line.split("\t")[0]) + 1
        except: 
            master_seed = sim.master_seed

    return (seed_file, master_seed)

def initialize_info_file(file_name, data_path):
    """Save simulation details to info file."""
    info_file_dir = os.path.join(data_path, "info.log")
    info_file   = open(info_file_dir, "a+")     # save the parameters of the simulation(s)
    info_file.write("\nfilename: " + file_name + "\n")
    info_str0   = "i    area t_sim  T_conn   T_sim      T_save n_vp master_seed  date       time      groupname"
    info_file.write(info_str0 + "\n")

    return info_file


#######################################################
# Functions inside the loop
#######################################################
def prepare_simulation(master_seed, n_populations):
    """Prepare random generators with master seed."""
    nest.ResetKernel()
    # set global kernel parameters
    nest.SetKernelStatus(
        {"communicate_allgather": sim.allgather,
        "print_time": True, 
        "overwrite_files": sim.overwrite_existing_files,
        "resolution": sim.dt,
        "total_num_virtual_procs": sim.n_vp})
    if sim.to_text_file:
        nest.SetKernelStatus({"data_path": os.path.join(sim.data_dir, "text")})
   
    # Set random seeds
    nest.sli_run('0 << /rngs [%i %i] Range { rngdict/gsl_mt19937 :: exch CreateRNG } Map >> SetStatus'%(
                 master_seed, master_seed + sim.n_vp - 1))
    #nest.SetKernelStatus({"rng_seeds" : range(master_seed, master_seed + sim.n_vp)})
    nest.sli_run('0 << /grng rngdict/gsl_mt19937 :: %i CreateRNG >> SetStatus'%(master_seed + sim.n_vp))
    #nest.SetKernelStatus({"grng_seed" : master_seed + sim.n_vp})
    pyrngs = [np.random.RandomState(s) for s in 
                range(master_seed + sim.n_vp + 1, master_seed + 2 * sim.n_vp + 1)]
    return pyrngs

def derive_parameters(model):
    """Derives numbers of neurons from which to record spikes and membrane potentials:
    either rate of population 
    or simply a fixed number regardless of population size
    """
    if sim.record_fraction_neurons_spike:
        n_neurons_rec_spike = (model.n_neurons * sim.frac_rec_spike).astype(int)
    else:
        n_neurons_rec_spike = (np.ones_like(model.n_neurons) * sim.n_rec_spike).astype(int)

    if sim.record_fraction_neurons_voltage:
        n_neurons_rec_voltage = (model.n_neurons * sim.frac_rec_voltage).astype(int)
    else:
        n_neurons_rec_voltage = (np.ones_like(model.n_neurons) * sim.n_rec_voltage).astype(int)

    return n_neurons_rec_spike, n_neurons_rec_voltage

def create_nodes(model, pyrngs):
    """Creates the following GIDs:
        neuron_GIDs
        ext_poisson
        ext_dc
        th_parrots
        th_poisson
        spike_detectors
        multimeters
        th_spike_detector
    
        Further initializes the neurons" membrane potentials.
    """
    neuron_GIDs     = []
    spike_detectors = []
    multimeters     = []
    ext_poisson     = []
    ext_dc          = []
    for pop_index, population in enumerate(model.populations):
        # Neurons
        neuron_GIDs.append(nest.Create(model.neuron_model, model.n_neurons[pop_index], params=model.model_params))
        # Initialize membrane potentials locally
        # drawn from normal distribution with mu=Vm0_mean, sigma=Vm0_std
        neurons_info    = nest.GetStatus(neuron_GIDs[pop_index])
        for ni in neurons_info:                 
            if ni["local"]:                         # only adapt local nodes
                Vm_init = pyrngs[ni["vp"]].normal(model.Vm0_mean, model.Vm0_std)
                nest.SetStatus([ni["global_id"]], {"V_m": Vm_init})

        # Devices
        if sim.record_cortical_spikes:
            spike_detector_dict = {"label": sim.spike_detector_label + population + "_", 
                                    "to_file": sim.to_text_file}
            spike_detectors.append(nest.Create("spike_detector", 1, params=spike_detector_dict))

        if sim.record_voltage:
            multimeter_dict = {"label": sim.multimeter_label + population + "_", 
                                "to_file": sim.to_text_file, 
                                "start": sim.t_rec_volt_start,   
                                "stop": sim.t_rec_volt_stop, 
                                "interval": 1.0, # ms
                                "withtime": True, 
                                "record_from": ["V_m"]}
            multimeters.append(nest.Create("multimeter", 1, params=multimeter_dict))
        
        # External input
        # One poisson generator per population. 
        #Rate is determined by base rate times in-degree[population]
        ext_poisson_params = {"rate": model.rate_ext * model.C_aext[pop_index]}
        ext_poisson.append(nest.Create("poisson_generator", 1, params=ext_poisson_params))
        # One dc generator per population. 
        # Amplitude is determined by base amplitude times in-degree[population]
        ext_dc_params = {"amplitude": model.dc_amplitude * model.C_aext[pop_index]}
        ext_dc.append(nest.Create("dc_generator", 1, params=ext_dc_params))
        
    # Thalamic neurons: parrot neurons and Poisson bg
    if not model.n_th == 0:
        th_parrots  = nest.Create("parrot_neuron", model.n_th, params=None)
        th_poisson  = nest.Create("poisson_generator", 1, 
            params={"rate": model.th_rate, 
                "start": model.th_start, 
                "stop": model.th_start + model.th_duration})
        if sim.record_thalamic_spikes:
            th_spike_detector_dict = {"label": sim.th_spike_detector_label, 
                                    "to_file": sim.to_text_file}
            th_spike_detector  = nest.Create("spike_detector", 1, params=th_spike_detector_dict)
        else:
            th_spike_detector = None
    else:
        th_parrots, th_poisson, th_spike_detector = (None, None, None)
        
    return (neuron_GIDs, 
            spike_detectors, multimeters,
            ext_poisson, ext_dc, 
            th_parrots, th_poisson, th_spike_detector)

def connect(model, all_GIDs, 
            n_neurons_rec_spike, n_neurons_rec_voltage,
            verbose):
    """Connect input GIDs according to connection_rule and dictionaries
    given in network_params.py
    """
    (neuron_GIDs, 
     spike_detectors, multimeters,
     ext_poisson, ext_dc, 
     th_parrots, th_poisson, th_spike_detector) = all_GIDs
    
    # Connect target populations...
    for target_index, target_pop in enumerate(model.populations):
        if verbose: print("Connecting target " + target_pop)
        if verbose: print("with source")
        target_GIDs = neuron_GIDs[target_index]    # transform indices to GIDs of target population
    
        # ...to source populations
        for source_index, source_pop in enumerate(model.populations):
            source_GIDs = neuron_GIDs[source_index] # transform indices to GIDs of source population
            n_synapses  = model.C_ab[target_index, source_index]  # connection probability
            if not n_synapses == 0:
                if verbose: print("\t" + source_pop)
    
                conn_dict       = model.conn_dict.copy()
                if model.connection_rule == "fixed_total_number":
                    conn_dict["N"]  = n_synapses
                elif model.connection_rule == "fixed_indegree":
                    conn_dict["indegree"]  = n_synapses
    
                mean_weight             = model.PSCs[target_index, source_index]
                std_weight              = abs(mean_weight * model.PSC_rel_sd)
                if mean_weight >= 0:
                    weight_dict = model.weight_dict_exc.copy()
                else:
                    weight_dict = model.weight_dict_inh.copy()
                weight_dict["mu"]       = mean_weight
                weight_dict["sigma"]    = std_weight
    
                mean_delay              = model.delays[target_index, source_index]
                std_delay               = mean_delay * model.delay_rel_sd 
                delay_dict              = model.delay_dict.copy()
                delay_dict["mu"]        = mean_delay
                delay_dict["sigma"]     = std_delay
    
                syn_dict                = model.syn_dict.copy()
                syn_dict["weight"]      = weight_dict
                syn_dict["delay"]       = delay_dict
    
                nest.Connect(source_GIDs, target_GIDs, conn_dict, syn_dict)
        
        # ...to thalamic population
        if not model.n_th == 0:
            n_synapses_th   = model.C_th_scaled[target_index]
            if not n_synapses_th == 0:
                if verbose: print("\tthalamus")
                conn_dict_th        = model.conn_dict.copy()
                conn_dict_th["N"]   = n_synapses_th
                
                mean_weight_th      = model.PSC_th
                std_weight_th       = mean_weight_th * model.PSC_rel_sd
                weight_dict_th      = model.weight_dict_exc.copy()
                weight_dict_th["mu"]    = mean_weight_th
                weight_dict_th["sigma"] = std_weight_th
    
                mean_delay_th       = model.delay_th
                std_delay_th        = mean_delay_th * model.delay_th_rel_sd 
                delay_dict_th       = model.delay_dict.copy()
                delay_dict_th["mu"]     = mean_delay_th
                delay_dict_th["sigma"]  = std_delay_th
    
                syn_dict_th             = model.syn_dict.copy()
                syn_dict_th["weight"]   = weight_dict_th
                syn_dict_th["delay"]    = delay_dict_th
    
                nest.Connect(th_parrots, target_GIDs, conn_dict_th, syn_dict_th)
    
        # ...to spike detector
        if sim.record_cortical_spikes:
            if verbose: print("\tspike detector")
            # Choose only a fixed fraction/number of neurons to record spikes from
            rec_spike_GIDs = target_GIDs[:n_neurons_rec_spike[target_index]]
            nest.Connect(list(rec_spike_GIDs), spike_detectors[target_index], "all_to_all")
    
        # ...to multimeter
        if sim.record_voltage:
            if verbose: print("\tmultimeter")
            # Choose only a fixed fraction/number of neurons to record membrane voltage from
            rec_voltage_GIDs = target_GIDs[:n_neurons_rec_voltage[target_index]]
            nest.Connect(multimeters[target_index], list(rec_voltage_GIDs), "all_to_all")
     
        # ...to background
        if not model.rate_ext == 0:
            if verbose: print("\tpoisson background")
            nest.Connect(ext_poisson[target_index], target_GIDs, 
                conn_spec={"rule": "all_to_all"}, 
                syn_spec={"weight": model.PSC_ext, "delay": model.delay_ext}) 
        if not model.dc_amplitude == 0:
            if verbose: print("\tDC background" )
            nest.Connect(ext_dc[target_index], target_GIDs, "all_to_all")
    
    # Connect Thalamic population, if existing.
    if not model.n_th == 0:
        if verbose: print("Connect thalamus: poisson to parrots")
        nest.Connect(th_poisson, th_parrots, "all_to_all")
        if sim.record_thalamic_spikes:
            if verbose: print("Connect thalamus to th_spike_detector")
            nest.Connect(th_parrots, th_spike_detector, "all_to_all")

def save_data(grp, all_GIDs, populations, n_neurons_rec_spike, n_neurons_rec_voltage):
    """Save spike data and membrane potentials (if specified so in sim_params.py).
    Thalamic spikes not yet included.
    """
    if sim.record_cortical_spikes:
        spike_detectors = all_GIDs[1]
        spikes_grp = grp.create_group("spikes")
        spikes_grp.attrs["dt"]  = sim.dt 
        spikes_grp.attrs["info"]  = "times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]"
        spikes_grp.attrs["info2"]  = "times in units of dt; dt in ms  =>  times/ms = times * dt"
        spikes_grp.attrs["n_neurons_rec_spike"] = n_neurons_rec_spike

        for j, population in enumerate(populations):
            senders = nest.GetStatus(spike_detectors[j])[0]["events"]["senders"]
            times   = nest.GetStatus(spike_detectors[j])[0]["events"]["times"]
            times   = np.uint(times / sim.dt) # in units of dt!

            # Create array of indices for data: 
            # times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]
            rec_neuron_i    = np.zeros(n_neurons_rec_spike[j] + 1)

            # Get corresponding reduced GIDs: nth neuron recorded
            n_spikes_per_neuron = np.unique(senders, return_counts=True)[1]
            max_index       = len(n_spikes_per_neuron) + 1
            nth_neuron      = np.cumsum(n_spikes_per_neuron)
            rec_neuron_i[1 : max_index] = nth_neuron        # leave out 0th index
            rec_neuron_i[max_index : ]  = nth_neuron[-1]    # in case some neurons didn't fire at all

            # sort times
            sorted_times = times[np.argsort(senders)] 
            for i in range(len(rec_neuron_i) - 1):
                i0, i1 = (rec_neuron_i[i], rec_neuron_i[i+1])
                sorted_times[i0:i1] = np.sort(sorted_times[i0:i1])

            # save data to HDF5 file:
            spikes_subgrp   = spikes_grp.create_group(population)
            dset_times      = spikes_subgrp.create_dataset("times", data=sorted_times)
            dset_indices    = spikes_subgrp.create_dataset("rec_neuron_i", data=rec_neuron_i)
            
    if sim.record_voltage:
        multimeters = all_GIDs[2]
        voltage_grp = grp.create_group("voltage")

        # Times can be reconstructed with times = np.arange(start + dt_volt, stop, dt_volt)
        start       = nest.GetStatus(multimeters[0])[0]["start"]   # ms
        stop        = nest.GetStatus(multimeters[0])[0]["stop"]   # ms
        if stop == float("inf"):
            stop = sim.t_sim
        dt_volt     = nest.GetStatus(multimeters[0])[0]["interval"]   # ms
        voltage_grp.attrs["dt_volt"]     = dt_volt 
        voltage_grp.attrs["t_min"]  = start 
        voltage_grp.attrs["t_max"]  = stop 
        voltage_grp.attrs["n_neurons_rec_voltage"] = n_neurons_rec_voltage

        for j, population in enumerate(populations):
            volts       = nest.GetStatus(multimeters[j])[0]["events"]["V_m"]
            senders     = nest.GetStatus(multimeters[j])[0]["events"]["senders"]
            n_events    = nest.GetStatus(multimeters[j])[0]["n_events"]   # number of 
            n_rec       = n_neurons_rec_voltage[j]
            n_times     = n_events / n_rec
            # Create mask in order to get sorted_volts[GID, times_index]
            s_inverse   = np.unique(senders, return_inverse=True)[1]
            volt_mask   = np.sort(np.argsort(s_inverse).reshape(n_rec, n_times))
            sorted_volts = volts[volt_mask]

            # save data to HDF5 file:
            dset_volts      = voltage_grp.create_dataset(population, data=sorted_volts)

