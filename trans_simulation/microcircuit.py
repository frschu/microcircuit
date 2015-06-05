"""microcircuit.py

Naming convention: layer (e.g. L4), type (usually e and i), population (e.g. L4e)

Includes:
check_parameters
prepare_simulation
derive_parameters
create_nodes    (all node parameters should be set only here)
connect_nodes         (synapse parameters are set here)
simulate
"""
from __future__ import print_function
import nest
import numpy as np
import h5py
import sys, os, shutil
import time, datetime

# Import specific moduls
from imp import reload
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import network_params; reload(network_params)

# logging
verbose     = False                     # whether to print every connection made
seed_file   = open(os.path.join(user.log_path, "seeds.log"), "a+")    # save the last used seed number
######################################################

#######################################################
# Instantiation
#######################################################
# Unchanged parameters
area            = 1.0
connection_type = "fixed_indegree"
j02             = 1.0
PSC_rel_sd      = 0.0

# Brunel:
n_neurons       = "brunel"
C_ab            = "brunel"
net_brunel      = network_params.net(area=area, n_neurons=n_neurons, C_ab=C_ab, 
                                     j02=j02, connection_type=connection_type,
                                     PSC_rel_sd=PSC_rel_sd)

# Microcircuit light:
# only some parameters like Potjans" model
# adapt n_neurons AND C_ab!
n_neurons       = "micro"
C_ab            = "micro"
net_micro       = network_params.net(area=area, n_neurons=n_neurons, C_ab=C_ab, 
                                     j02=j02, connection_type=connection_type,
                                     PSC_rel_sd=PSC_rel_sd)

# Differences between the two models
C_brunel    = net_brunel.C_ab
C_micro     = net_micro.C_ab
delta_C     = C_micro - C_brunel
n_brunel    = net_brunel.n_neurons
n_micro     = net_micro.n_neurons
delta_n     = n_micro - n_brunel

# The steps on the way from Brunel to microcircuit
dist_init   = 0.00  # initial point: Brunel
dist_max    = 0.00  # the goal:      microcircuit light
step        = 0.1  # step size towards the goal
dists       = np.arange(dist_init, dist_max + step, step)

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

#######################################################
# Create data file
#######################################################
data_path = user.data_dir
if not os.path.exists(data_path):
    os.makedirs(data_path)

# File name
# contains global conditions of simulations: 
# - area, simulated time, thalamus, background
# e.g. 'a1.0_t20.2_th_dc.hdf5'
file_name= "a%.1f_t%.1f"%(area, sim.t_sim * 1e-3)
if not net_micro.n_th == 0:
    file_name += "_th"
if not net_micro.dc_amplitude == 0:
    file_name += "_dc"
file_name   += ".hdf5"

data_file = h5py.File(os.path.join(data_path, file_name), "w")

data_file.attrs["area"]     = area
data_file.attrs["t_sim"]    = sim.t_sim*1e-3
data_file.attrs["t_trans"]  = sim.t_trans*1e-3
data_file.attrs["n_vp"]     = sim.n_vp
data_file.attrs["dt"]       = sim.dt
data_file.attrs["connection_type"]  = connection_type
data_file.attrs["populations"]      = net_micro.populations 
data_file.attrs["layers"]           = net_micro.layers 
data_file.attrs["types"]            = net_micro.types 
data_file.attrs["n_populations"]    = net_micro.n_populations 
data_file.attrs["n_layers"]         = net_micro.n_layers 
data_file.attrs["n_types"]          = net_micro.n_types 


info_file_dir = os.path.join(data_path, "info.log")
info_file   = open(info_file_dir, "a+")     # save the parameters of the simulation(s)
info_file.write("new simulation\n")
info_str0   = "dist n_syn    area t_sim  T_conn   T_sim    n_vp master_seed  date       time      filename"
info_file.write(info_str0 + "\n")

#######################################################
# Looping
#######################################################
for distance in dists:
    C_ab    = C_brunel + distance * delta_C
    n_ab    = n_brunel + distance * delta_n
    # Get instance of network
    net     = network_params.net(area=area, n_neurons=n_neurons, C_ab=C_ab, 
                                         j02=j02, connection_type=connection_type,
                                         PSC_rel_sd=PSC_rel_sd)

    ######################################################
    # Prepare simulation
    ######################################################
    nest.ResetKernel()
    # set global kernel parameters
    nest.SetKernelStatus(
        {"communicate_allgather": sim.allgather,
        "overwrite_files": True,
        "resolution": sim.dt,
        "total_num_virtual_procs": sim.n_vp})
    #nest.SetKernelStatus({"data_path": data_path})
   
    # Set random seeds
    nest.SetKernelStatus({"grng_seed" : master_seed})
    nest.SetKernelStatus({"rng_seeds" : range(master_seed + 1, master_seed + 1 + sim.n_vp)})
    pyrngs = [np.random.RandomState(s) for s in 
                range(master_seed + 1 + sim.n_vp, master_seed + 1 + 2 * sim.n_vp)]
    pyrngs_rec_spike = [np.random.RandomState(s) for s in 
                range(master_seed + 1 + 2 * sim.n_vp, 
                      master_seed + 1 + 2 * sim.n_vp + net.n_populations)]
    pyrngs_rec_voltage = [np.random.RandomState(s) for s in 
                range(master_seed + 1 + 2 * sim.n_vp + net.n_populations, 
                      master_seed + 1 + 2 * sim.n_vp + 2 * net.n_populations)]

    ######################################################
    # Derive parameters
    ######################################################
    n_neurons       = net.n_neurons
    # Get PSCs used as synaptic weights
    PSCs, PSC_ext, PSC_th = net.get_PSCs()
    
    # numbers of neurons from which to record spikes and membrane potentials
    # either rate of population or simply a fixed number regardless of population size
    if sim.record_fraction_neurons_spike:
        n_neurons_rec_spike = np.rint(net.n_neurons * sim.frac_rec_spike).astype(int)
    else:
        n_neurons_rec_spike = (np.ones_like(net.n_neurons) * sim.n_rec_spike).astype(int)

    if sim.record_fraction_neurons_voltage:
        n_neurons_rec_voltage = np.rint(net.n_neurons * sim.frac_rec_voltage).astype(int)
    else:
        n_neurons_rec_voltage = (np.ones_like(net.n_neurons) * sim._rec_voltage).astype(int)
     
    
    ######################################################
    # Create nodes
    ######################################################
    """
        Creates the following GIDs:
        neuron_GIDs
        ext_poisson
        ext_dc
        th_GIDs
        th_poisson
        spike_detectors
        multimeters
        th_spike_detector
    
        Further initializes the neurons" membrane potentials.
    """
    print("Create nodes")
    
    # Neurons
    neurons     = nest.Create(net.neuron_model, net.n_total, params=net.model_params)
    # Initialize membrane potentials locally
    # drawn from normal distribution with mu=Vm0_mean, sigma=Vm0_std
    neurons_info    = nest.GetStatus(neurons)
    for ni in neurons_info:                 
        if ni["local"]:                         # only adapt local nodes
            nest.SetStatus([ni["global_id"]], 
                {"V_m": pyrngs[ni["vp"]].normal(net.Vm0_mean, net.Vm0_std)})
    # GIDs for neurons on subnets
    GID0        = neurons[0]
    upper_GIDs  = GID0 + np.cumsum(n_neurons) - 1
    lower_GIDs  = np.insert((upper_GIDs[:-1] + 1), 0, GID0) 
    neuron_GIDs = np.array([range(*boundary) for boundary in zip(lower_GIDs, upper_GIDs)])
    
    # External input
    # One poisson generator per population. 
    #Rate is determined by base rate times in-degree[population]
    ext_poisson_params = [{"rate": net.rate_ext * in_degree} for in_degree in net.C_aext]
    ext_poisson = nest.Create("poisson_generator", net.n_populations, 
        params=ext_poisson_params) 
    # One dc generator per population. 
    # Amplitude is determined by base amplitude times in-degree[population]
    ext_dc_params = [{"amplitude": net.dc_amplitude * in_degree} for in_degree in net.C_aext]
    ext_dc = nest.Create("dc_generator", net.n_populations, 
        params=ext_dc_params) 
    
    # Thalamic neurons: parrot neurons and Poisson bg
    if not net.n_th == 0:
        th_GIDs     = nest.Create("parrot_neuron", net.n_th, params=None)
        th_poisson  = nest.Create("poisson_generator", 1, 
            params={"rate": net.th_rate, 
                "start": net.th_start, 
                "stop": net.th_start + net.th_duration})
    
    # Devices
    if sim.record_cortical_spikes:
        spike_detector_dict = [{"label": sim.spike_detector_label + population + "_", 
                                "to_file": False} for population in net.populations]
        spike_detectors = nest.Create("spike_detector", net.n_populations, params=spike_detector_dict)
    if sim.record_voltage:
        multimeter_dict = [{"label": sim.multimeter_label + population + "_", 
                            "to_file": False, 
                            "record_from": ["V_m"]} for population in net.populations]
        multimeters     = nest.Create("multimeter", net.n_populations, params=multimeter_dict)
    if sim.record_thalamic_spikes:
        th_spike_detector_dict = {"label": sim.th_spike_detector_label, 
                                "to_file": False}
        th_spike_detector  = nest.Create("spike_detector", 1, params=th_spike_detector_dict)

    ###################################################
    # Connect
    ###################################################
    t_connect_0 = time.time()

    # Preparation: Thalamic population, if existing.
    print("Connect")
    if not net.n_th == 0:
        if verbose: print("Connect thalamus: poisson to parrots")
        nest.Connect(th_poisson, th_GIDs, "all_to_all")
        if sim.record_thalamic_spikes:
            if verbose: print("Connect thalamus to th_spike_detector")
            nest.Connect(th_GIDs, th_spike_detector, "all_to_all")
    
    # Connect target populations...
    for target_index, target_pop in enumerate(net.populations):
        if verbose: print("Connecting target " + target_pop)
        if verbose: print("with source")
        target_GIDs = neuron_GIDs[target_index]    # transform indices to GIDs of target population
    
        # ...to source populations
        for source_index, source_pop in enumerate(net.populations):
            source_GIDs = neuron_GIDs[source_index] # transform indices to GIDs of source population
            n_synapses  = net.C_ab[target_index, source_index]  # connection probability
            if not n_synapses == 0:
                if verbose: print("\t" + source_pop)
    
                conn_dict       = net.conn_dict.copy()
                if net.connection_type == "fixed_total_number":
                    conn_dict["N"]  = n_synapses
                elif net.connection_type == "fixed_indegree":
                    conn_dict["indegree"]  = n_synapses
    
                mean_weight             = PSCs[target_index, source_index]
                std_weight              = abs(mean_weight * net.PSC_rel_sd)
                if mean_weight >= 0:
                    weight_dict = net.weight_dict_exc.copy()
                else:
                    weight_dict = net.weight_dict_inh.copy()
                weight_dict["mu"]       = mean_weight
                weight_dict["sigma"]    = std_weight
    
                mean_delay              = net.delays[target_index, source_index]
                std_delay               = mean_delay * net.delay_rel_sd 
                delay_dict              = net.delay_dict.copy()
                delay_dict["mu"]        = mean_delay
                delay_dict["sigma"]     = std_delay
    
                syn_dict                = net.syn_dict.copy()
                syn_dict["weight"]      = weight_dict
                syn_dict["delay"]       = delay_dict
    
                nest.Connect(source_GIDs, target_GIDs, conn_dict, syn_dict)
        
        # ...to background
        if not net.rate_ext == 0:
            if verbose: print("\tpoisson background")
            nest.Connect([ext_poisson[target_index]], target_GIDs, 
                conn_spec={"rule": "all_to_all"}, 
                syn_spec={"weight": PSC_ext}) # global delay is unnecessary
        if not net.dc_amplitude == 0:
            if verbose: print("\tDC background" )
            nest.Connect([ext_dc[target_index]], target_GIDs, "all_to_all")
    
        # ...to thalamic population
        if not net.n_th == 0:
            n_synapses_th   = net.C_th_scaled[target_index]
            if not n_synapses_th == 0:
                if verbose: print("\tthalamus")
                conn_dict_th        = net.conn_dict.copy()
                conn_dict_th["N"]   = n_synapses_th
                
                mean_weight_th      = PSC_th
                std_weight_th       = mean_weight_th * net.PSC_rel_sd
                weight_dict_th      = net.weight_dict_exc.copy()
                weight_dict_th["mu"]    = mean_weight_th
                weight_dict_th["sigma"] = std_weight_th
    
                mean_delay_th       = net.delay_th
                std_delay_th        = mean_delay_th * net.delay_th_rel_sd 
                delay_dict_th       = net.delay_dict.copy()
                delay_dict_th["mu"]     = mean_delay_th
                delay_dict_th["sigma"]  = std_delay_th
    
                syn_dict_th             = net.syn_dict.copy()
                syn_dict_th["weight"]   = weight_dict_th
                syn_dict_th["delay"]    = delay_dict_th
    
                nest.Connect(th_GIDs, target_GIDs, conn_dict_th, syn_dict_th)
    
        # ...to spike detector
        if sim.record_cortical_spikes:
            if verbose: print("\tspike detector")
            # Choose only a fixed fraction/number of neurons to record spikes from
            if sim.rand_rec_spike:
                rec_spike_GIDs = np.sort(pyrngs_rec_spike[i].choice(target_GIDs,
                    n_neurons_rec_spike[target_index], replace=False))
            else:
                rec_spike_GIDs = target_GIDs[:n_neurons_rec_spike[target_index]]
            nest.Connect(list(rec_spike_GIDs), [spike_detectors[target_index]], "all_to_all")
    
        # ...to multimeter
        if sim.record_voltage:
            if verbose: print("\tmultimeter")
            # Choose only a fixed fraction/number of neurons to record membrane voltage from
            if sim.rand_rec_voltage:
                rec_voltage_GIDs = np.sort(pyrngs_rec_voltage[i].choice(target_GIDs,
                    n_neurons_rec_voltage[target_index], replace=False))
            else:
                rec_voltage_GIDs = target_GIDs[:n_neurons_rec_voltage[target_index]]
            nest.Connect([multimeters[target_index]], list(rec_voltage_GIDs), "all_to_all")
     
    T_connect   = time.time() - t_connect_0

    ###################################################
    # Simulate
    ###################################################
    print("Simulate")
    t_simulate_0 = time.time()
    nest.Simulate(sim.t_sim)
    T_simulate  = time.time() - t_simulate_0
    
    ###################################################
    # Save recorded data
    ###################################################
    print("Save data")
    t_save_0    = time.time()
    
    now         = str(datetime.datetime.now())[:-7]
    group_name  = "d%.2f_j%.2f_sdJ%.2f"%(distance, j02, PSC_rel_sd)  
    print(group_name)
    grp         = data_file.create_group(group_name)
    grp.attrs["distance"] = distance
    grp.attrs["C_ab"] = net.C_ab
    grp.attrs["master_seed"] = master_seed
    grp.attrs["time_to_connect"] = T_connect
    grp.attrs["time_to_simulate"] = T_simulate
    grp.attrs["date_and_time"] = now

    if sim.record_cortical_spikes:
        spikes_grp = grp.create_group("spikes")
        spikes_grp.attrs["dt"]  = sim.dt 
        spikes_grp.attrs["info"]  = "times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]"
        spikes_grp.attrs["info2"]  = "times in units of dt; dt in ms  =>  times/ms = times * dt"

        senders_all = []
        int_times_all = []
        last_rec_spike = 0
        for i, population in enumerate(net.populations):
            senders = nest.GetStatus((spike_detectors[i],))[0]["events"]["senders"]
            times   = nest.GetStatus((spike_detectors[i],))[0]["events"]["times"]
            sorted_times = (np.uint32(times / sim.dt))[np.argsort(senders)] 

            # Create array of indices for data: 
            # times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]
            rec_neuron_i    = np.zeros(n_neurons_rec_spike[i] + 1)
            rec_neuron_i[0] = last_rec_spike

            # Get corresponding reduced GIDs: nth neuron recorded
            n_spikes_per_neuron = np.unique(senders, return_counts=True)[1]
            max_index       = len(n_spikes_per_neuron) + 1
            nth_neuron      = np.cumsum(n_spikes_per_neuron) + last_rec_spike
            rec_neuron_i[1 : max_index] = nth_neuron
            
            # Update last_rec_spike and fill up the entries for neurons without a single spike
            last_rec_spike  = nth_neuron[-1]
            rec_neuron_i[max_index : ]   = last_rec_spike

            # save data to HDF5 file:
            spikes_subgrp   = spikes_grp.create_group(population)
            dset_times      = spikes_subgrp.create_dataset("times", data=sorted_times)
            dset_indices    = spikes_subgrp.create_dataset("rec_neuron_i", data=rec_neuron_i)

    T_save = time.time() - t_save_0
    grp.attrs["time_to_save"] = T_save
    
    ###################################################
    # Save info, set new seed
    ###################################################

    info_str    = "{0:4.2f} {1:8d} {2:4.1f} {3:6.1f} {4:8.1f} {5:8.1f}   {6:2d}  {7:10d}  ".format(
                    distance, np.sum(net.C_ab), area, sim.t_sim*1e-3, 
                    T_connect, T_simulate, sim.n_vp, master_seed)
    info_str += now + "  " + group_name
    info_file.write(info_str + "\n")

    # save the last seed to file, such that independent realizations are possible
    last_seed = master_seed + 1 + 2 * sim.n_vp + 2 * net.n_populations - 1  # last -1 since range ends beforehand
    seed_file.write("{0:6d}".format(last_seed) + "\t\t" + now + "\t" + group_name + "\n")
    master_seed = last_seed + 1

info_file.close()
seed_file.close()
data_file.close()
####################################################################################
