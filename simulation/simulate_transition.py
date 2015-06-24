"""simulate_transition.py

Simulating the transition from Brunel's to Potjans' model.

Structure:
instantiate models                  initial and final; from model_class.py
initialize_data_file                creates file_name and opens HDF5-file
initialize_seeds                    opens seeds.log and looks for previous master seeds 
initialize_info_file                opens and initializes info_file, where basic parameters of the simulation are saved in plain text

Loop over distance(*) from Brunel to Potjans:
    new model                       instantiate model at given distance
    prepare_simulation              prepare random generators such that each simulation is independent 
    derive_parameters               calculate number of neurons to record from (spikes and membrane potentials separately)
    create_nodes                    node parameters are set here (incl. membrane potential initialization)
    connect                         synapse parameters are set here
    nest.simulate
    save_data                       saves data to HDF5-file.

(*) distance = L2(Brunel - model) / L2(Brunel - Potjans)

Naming convention: layer (e.g. L4), type (usually e and i), population (e.g. L4e)
"""
from __future__ import print_function
import nest
import numpy as np
import h5py
import sys, os, shutil
import time, datetime

from imp import reload
import sim_params as sim; reload(sim)
import model_class; reload(model_class)
import functions; reload(functions)
verbose     = True                     # whether to print every connection made

#######################################################
# Instantiate model
#######################################################
T0 = time.time()
# Unchanged parameters
area            = 1.0
connection_rule = "fixed_indegree"
g               = 4.0
rate_ext        = 8.0 # Hz background rate
PSC_rel_sd      = 0.0 # 0.1 for  Potjans' model
delay_rel_sd    = 0.5 # 0.5 for Potjans' model  

# Brunel:
j02             = 1.0
n_neurons       = "brunel"
C_ab            = "brunel"
model_brunel    = model_class.model(area=area, 
                                    n_neurons=n_neurons, C_ab=C_ab, 
                                    connection_rule=connection_rule,
                                    j02=j02, g=g, rate_ext=rate_ext,
                                    PSC_rel_sd=PSC_rel_sd, 
                                    delay_rel_sd=delay_rel_sd) 

# Microcircuit light:
# only some parameters like Potjans" model
j02             = 2.0
n_neurons       = "micro"
C_ab            = "micro"
model_micro     = model_class.model(area=area, 
                                    n_neurons=n_neurons, C_ab=C_ab, 
                                    connection_rule=connection_rule,
                                    j02=j02, g=g, rate_ext=rate_ext,
                                    PSC_rel_sd=PSC_rel_sd, 
                                    delay_rel_sd=delay_rel_sd) 

#######################################################
# Create data file
#######################################################
sub_path = "trans"
data_file, file_name, data_path = functions.initialize_data_file(sub_path, model_micro, verbose)
seed_file, master_seed          = functions.initialize_seeds()
info_file                       = functions.initialize_info_file(file_name, data_path)


#######################################################
# Looping
#######################################################
# The steps on the way from Brunel to microcircuit
model_init      = model_brunel
model_final     = model_micro
dist_init   = 0.10  # initial point: Brunel
dist_final  = 1.00  # the goal:      microcircuit light
step        = 0.05  # step size towards the goal
n_steps     = int(round(abs(dist_final - dist_init) / step)) + 1
dists       = np.linspace(dist_init, dist_final, n_steps)
data_file.attrs["dists"] = dists 

for distance in dists:
    ######################################################
    # New model
    ######################################################
    area            = (1. - distance) * model_init.area         + distance * model_final.area        
    n_neurons       = (1. - distance) * model_init.n_neurons    + distance * model_final.n_neurons   
    C_ab            = (1. - distance) * model_init.C_ab         + distance * model_final.C_ab        
    j02             = (1. - distance) * model_init.j02          + distance * model_final.j02         
    g               = (1. - distance) * model_init.g            + distance * model_final.g           
    rate_ext        = (1. - distance) * model_init.rate_ext     + distance * model_final.rate_ext    
    PSC_rel_sd      = (1. - distance) * model_init.PSC_rel_sd   + distance * model_final.PSC_rel_sd  
    delay_rel_sd    = (1. - distance) * model_init.delay_rel_sd + distance * model_final.delay_rel_sd
    model   = model_class.model(area=area, 
                                n_neurons=n_neurons, C_ab=C_ab, 
                                connection_rule="fixed_indegree",
                                j02=j02, g=g, rate_ext=rate_ext,
                                PSC_rel_sd=PSC_rel_sd, 
                                delay_rel_sd=delay_rel_sd) 

    ######################################################
    # Prepare simulation
    ######################################################
    pyrngs = functions.prepare_simulation(master_seed, n_populations=model.n_populations)

    ######################################################
    # Derive parameters
    ######################################################
    (n_neurons_rec_spike, n_neurons_rec_voltage) = functions.derive_parameters(model)
     
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
    functions.connect(model, all_GIDs, 
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
    group_name  = "d%.2f_j%.2f_sdJ%.2f"%(distance, j02, PSC_rel_sd)  
    print(group_name)
    now         = str(datetime.datetime.now())[:-7]
    grp         = data_file.create_group(group_name)
    grp.attrs["date_and_time"] = now
    grp.attrs["master_seed"] = master_seed
    grp.attrs["C_ab"] = model.C_ab
    grp.attrs["distance"] = distance

    t_save_0    = time.time()
    functions.save_data(grp, all_GIDs, model.populations, n_neurons_rec_spike, n_neurons_rec_voltage)
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

T_total = time.time() - T0
print("T_total      = ", T_total)
data_file.attrs["total_time"]    = T_total
info_file.write("total time: %.2f\n"%T_total)

info_file.close()
seed_file.close()
data_file.close()
####################################################################################
