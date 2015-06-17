"""simulate_transition.py

Naming convention: layer (e.g. L4), type (usually e and i), population (e.g. L4e)

for-loop for simulating the transition from Brunel's to Potjans' model.

Runs functions:
prepare_simulation
derive_parameters
create_nodes    (all node parameters should be set only here)
connect         (synapse parameters are set here)

Saves data to hdf5 file.
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
import functions; reload(functions)

# logging
verbose     = True                     # whether to print every connection made
seed_file   = open(os.path.join(user.log_path, "seeds.log"), "a+")    # save the last used seed number
######################################################

#######################################################
# Instantiation
#######################################################
T0 = time.time()
# Unchanged parameters
area            = 1.0
connection_type = "fixed_indegree"
g               = 4.0
rate_ext        = 8.0 # Hz background rate
PSC_rel_sd      = 0.1 # 0.1 for  Potjans' model
delay_rel_sd    = 0.5 # 0.5 for Potjans' model  

# Brunel:
j02             = 1.0
n_neurons       = "brunel"
C_ab            = "brunel"
model_brunel      = network_params.net(area=area, 
                                           n_neurons=n_neurons, C_ab=C_ab, 
                                           connection_type=connection_type,
                                           j02=j02, g=g, rate_ext=rate_ext,
                                           PSC_rel_sd=PSC_rel_sd, 
                                           delay_rel_sd=delay_rel_sd) 

# Microcircuit light:
# only some parameters like Potjans" model
j02             = 2.0
n_neurons       = "micro"
C_ab            = "micro"
model_micro       = network_params.net(area=area, 
                                           n_neurons=n_neurons, C_ab=C_ab, 
                                           connection_type=connection_type,
                                           j02=j02, g=g, rate_ext=rate_ext,
                                           PSC_rel_sd=PSC_rel_sd, 
                                           delay_rel_sd=delay_rel_sd) 

# Initial Seeds
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
data_sup_path = user.data_dir
data_path = os.path.join(data_sup_path, "vary_d")
if not os.path.exists(data_path):
    os.makedirs(data_path)

# File name
# contains global conditions of simulations: 
# - area, simulated time, thalamus, background
# e.g. 'a1.0_t20.2_th_dc.hdf5'
sim_spec= "a%.1f_t%.1f"%(area, sim.t_sim * 1e-3)
if not model_micro.n_th == 0:
    sim_spec += "_th"
if not model_micro.dc_amplitude == 0:
    sim_spec += "_dc"
if connection_type=="fixed_total_number":
    sim_spec += "_totalN"
file_name   = sim_spec + "_00.hdf5"

# don't overwrite existing files...
if file_name in os.listdir(data_path):
    max_n = 0
    for some_file in os.listdir(data_path):
        if some_file.startswith(sim_spec):
            max_n = max(max_n, int(some_file[len(sim_spec)+1: len(sim_spec) + 3])) 
    file_name = sim_spec + "_" + str(max_n + 1).zfill(2) + ".hdf5"
if verbose: print("Filename: vary_d/" + file_name)

data_file = h5py.File(os.path.join(data_path, file_name), "w")

# Attributes
data_file.attrs["area"]     = area
data_file.attrs["t_sim"]    = sim.t_sim*1e-3
data_file.attrs["t_trans"]  = sim.t_trans*1e-3
data_file.attrs["n_vp"]     = sim.n_vp
data_file.attrs["dt"]       = sim.dt
data_file.attrs["connection_type"]  = connection_type
data_file.attrs["populations"]      = model_micro.populations 
data_file.attrs["layers"]           = model_micro.layers 
data_file.attrs["types"]            = model_micro.types 
data_file.attrs["n_populations"]    = model_micro.n_populations 
data_file.attrs["n_layers"]         = model_micro.n_layers 
data_file.attrs["n_types"]          = model_micro.n_types 
data_file.attrs["delay_e"]          = model_micro.delay_e 
data_file.attrs["delay_i"]          = model_micro.delay_i 


# Save simulation details to info file.
info_file_dir = os.path.join(data_path, "info.log")
info_file   = open(info_file_dir, "a+")     # save the parameters of the simulation(s)
info_file.write("\nfilename: " + file_name + "\n")
info_str0   = "dist area t_sim  T_conn   T_sim      T_save n_vp master_seed  date       time      groupname"
info_file.write(info_str0 + "\n")


#######################################################
# Looping
#######################################################
# The steps on the way from Brunel to microcircuit
model_init      = model_brunel
model_final     = model_micro
dist_init   = 1.00  # initial point: Brunel
dist_final  = 1.00  # the goal:      microcircuit light
step        = 0.05  # step size towards the goal
n_steps     = int(round(abs(dist_final - dist_init) / step)) + 1
dists       = np.linspace(dist_init, dist_final, n_steps)
data_file.attrs["dists"] = dists 

for distance in dists:
    # New model
    area            = (1. - distance) * model_init.area         + distance * model_final.area        
    n_neurons       = (1. - distance) * model_init.n_neurons    + distance * model_final.n_neurons   
    C_ab            = (1. - distance) * model_init.C_ab         + distance * model_final.C_ab        
    j02             = (1. - distance) * model_init.j02          + distance * model_final.j02         
    g               = (1. - distance) * model_init.g            + distance * model_final.g           
    rate_ext        = (1. - distance) * model_init.rate_ext     + distance * model_final.rate_ext    
    PSC_rel_sd      = (1. - distance) * model_init.PSC_rel_sd   + distance * model_final.PSC_rel_sd  
    delay_rel_sd    = (1. - distance) * model_init.delay_rel_sd + distance * model_final.delay_rel_sd
    model = network_params.net(area=area, 
                                     n_neurons=n_neurons, C_ab=C_ab, 
                                     connection_type="fixed_indegree",
                                     j02=j02, g=g, rate_ext=rate_ext,
                                     PSC_rel_sd=PSC_rel_sd, 
                                     delay_rel_sd=delay_rel_sd) 

    ######################################################
    # Prepare simulation
    ######################################################
    pyrngs, pyrngs_rec_spike, pyrngs_rec_voltage = functions.prepare_simulation(master_seed, model=model)

    ######################################################
    # Derive parameters
    ######################################################
    (PSCs, PSC_ext, PSC_th, 
        n_neurons_rec_spike, n_neurons_rec_voltage) = functions.derive_parameters(model)
     
    ######################################################
    # Create nodes
    ######################################################
    print("Create nodes")
    all_GIDs = functions.create_nodes(model, pyrngs)

    ###################################################
    # Connect
    ###################################################
    print("Connect")
    t_connect_0 = time.time()
    functions.connect(model, all_GIDs, PSCs, PSC_ext, PSC_th, 
                      n_neurons_rec_spike, n_neurons_rec_voltage,
                      verbose)
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
    grp.attrs["C_ab"] = model.C_ab
    grp.attrs["master_seed"] = master_seed
    grp.attrs["date_and_time"] = now

    if sim.record_cortical_spikes:
        spike_detectors = all_GIDs[5]
        spikes_grp = grp.create_group("spikes")
        spikes_grp.attrs["dt"]  = sim.dt 
        spikes_grp.attrs["info"]  = "times_{ith neuron} = times[rec_neuron_i[i]:rec_neuron_i[i+1]]"
        spikes_grp.attrs["info2"]  = "times in units of dt; dt in ms  =>  times/ms = times * dt"
        spikes_grp.attrs["n_neurons_rec_spike"] = n_neurons_rec_spike

        for j, population in enumerate(model.populations):
            senders = nest.GetStatus((spike_detectors[j],))[0]["events"]["senders"]
            times   = nest.GetStatus((spike_detectors[j],))[0]["events"]["times"]
            times   = np.uint(times / sim.dt) # in unit of dt!

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
        multimeters = all_GIDs[6]
        voltage_grp = grp.create_group("voltage")

        # Times can be reconstructed with times = np.arange(start + dt_volt, stop, dt_volt)
        start       = nest.GetStatus((multimeters[0],))[0]["start"]   # ms
        stop        = nest.GetStatus((multimeters[0],))[0]["stop"]   # ms
        if stop == float("inf"):
            stop = sim.t_sim
        dt_volt     = nest.GetStatus((multimeters[0],))[0]["interval"]   # ms
        voltage_grp.attrs["dt_volt"]     = dt_volt 
        voltage_grp.attrs["t_min"]  = start 
        voltage_grp.attrs["t_max"]  = stop 
        voltage_grp.attrs["n_neurons_rec_voltage"] = n_neurons_rec_voltage

        for j, population in enumerate(model.populations):
            volts       = nest.GetStatus((multimeters[j],))[0]["events"]["V_m"]
            senders     = nest.GetStatus((multimeters[j],))[0]["events"]["senders"]
            n_events    = nest.GetStatus((multimeters[j],))[0]["n_events"]   # number of 
            n_rec       = n_neurons_rec_voltage[j]
            n_times     = n_events / n_rec
            # Create mask in order to get sorted_volts[GID, times_index]
            s_inverse   = np.unique(senders, return_inverse=True)[1]
            volt_mask   = np.sort(np.argsort(s_inverse).reshape(n_rec, n_times))
            sorted_volts = volts[volt_mask]

            # save data to HDF5 file:
            dset_volts      = voltage_grp.create_dataset(population, data=sorted_volts)
            
    T_save = time.time() - t_save_0

    ###################################################
    # Save info, set new seed
    ###################################################
    print("T_connect    = ", T_connect)
    print("T_simulate   = ", T_simulate)
    print("T_save       = ", T_save)
    grp.attrs["time_to_connect"]    = T_connect
    grp.attrs["time_to_simulate"]   = T_simulate
    grp.attrs["time_to_save"]       = T_save
    

    info_str    = "{0:4.2f} {1:4.1f} {2:6.1f} {3:8.1f} {4:10.1f} {5:6.1f} {6:4d} {7:11d}  ".format(
                    distance, area, sim.t_sim*1e-3, 
                    T_connect, T_simulate, T_save, sim.n_vp, master_seed)
    info_str += now + "  " + group_name
    info_file.write(info_str + "\n")

    # save the last seed to file, such that independent realizations are possible
    last_seed = master_seed + 1 + 2 * sim.n_vp + 2 * model.n_populations - 1  # last -1 since range ends beforehand
    seed_file.write("{0:6d}".format(last_seed) + "\t\t" + 
                    now + "\t" + 
                    os.path.join(file_name, group_name) + "\n")
    master_seed = last_seed + 1

#
T_total = time.time() - T0
print("T_total      = ", T_total)
data_file.attrs["total_time"]    = T_total
info_file.write("total time: %.2f\n"%T_total)

info_file.close()
seed_file.close()
data_file.close()
####################################################################################
