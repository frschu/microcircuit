"""simulate_brunel.py

Simulation of Brunel balanced random network (Brunel 2000)

Structure:
instantiate model                   from model_class.py
initialize_data_file                creates file_name and opens HDF5-file
initialize_seeds                    opens seeds.log and looks for previous master seeds 
initialize_info_file                opens and initializes info_file, where basic parameters of the simulation are saved in plain text

prepare_simulation              prepare random generators such that each simulation is independent 
derive_parameters               calculate number of neurons to record from (spikes and membrane potentials separately)
create_nodes                    node parameters are set here (incl. membrane potential initialization)
connect                         synapse parameters are set here
nest.simulate
save_data                       saves data to HDF5-file.

"""
from __future__ import print_function
import nest
import numpy as np
import h5py
import sys, os, shutil
import time, datetime

from imp import reload
import brunel_sim_params as sim; reload(sim)
import functions; reload(functions)
import brunel_model_class as model_class; reload(model_class)
verbose     = False                     # whether to print every connection made
append_data = False                     # whether to append the selected data_file

#######################################################
# Instantiate model
#######################################################
T0 = time.time()
# Unchanged parameters
neuron_model     = "iaf_psc_delta" 
connection_rule = "fixed_indegree" # "fixed_indegree", "fixed_total_number"
weight_rel_sd      = 0.1 
model           = model_class.model(neuron_model=neuron_model,
                                    connection_rule=connection_rule,
                                    weight_rel_sd=weight_rel_sd) 

#######################################################
# Create data file
#######################################################
sub_path = "brunel"
data_file, file_name, data_path = functions.initialize_data_file(sub_path, model, verbose, append=append_data)
seed_file, master_seed          = functions.initialize_seeds()
info_file                       = functions.initialize_info_file(file_name, data_path)

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
if append_data: 
    max_grp = 0
    for key in data_file.keys():
        max_grp = max(max_grp, int(key))
    group_name  = "%i"%(max_grp + 1)  
else:
    group_name  = "0"
print(group_name)
now         = str(datetime.datetime.now())[:-7]
grp         = data_file.create_group(group_name)
grp.attrs["date_and_time"] = now
grp.attrs["master_seed"] = master_seed
grp.attrs["C_ab"] = model.C_ab

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

info_str    = "{0:4s} {1:4.1f} {2:6.1f} {3:8.1f} {4:10.1f} {5:6.1f} {6:4d} {7:11d}  ".format(
                group_name, model.area, sim.t_sim*1e-3, 
                T_connect, T_simulate, T_save, sim.n_vp, master_seed)
info_str += now + "  " + group_name
info_file.write(info_str + "\n")

# save the last seed to file, such that independent realizations are possible
last_seed = master_seed + 1 + 2 * sim.n_vp + 2 * model.n_populations - 1  # last -1 since range ends beforehand
seed_file.write("{0:6d}".format(last_seed) + "\t\t" + 
                now + "\t" + 
                os.path.join(file_name, group_name) + "\n")
    
T_total = time.time() - T0
print("T_total      = ", T_total)
data_file.attrs["total_time"]    = T_total
info_file.write("total time: %.2f\n"%T_total)

data_file.close()
seed_file.close()
info_file.close()
####################################################################################
