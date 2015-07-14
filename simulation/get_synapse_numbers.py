"""get_synapse_numbers.py

Retrieve the numbers of synapses (-> distributions) obtain in the 
microcircuit level. This has validation purpose only. 
"""
from __future__ import print_function
import nest
import h5py
import numpy as np
import sys, os, shutil
import time, datetime

from imp import reload
import sim_params as sim; reload(sim)
import functions; reload(functions)
import model_class; reload(model_class)
verbose     = False                     # whether to print every connection made
#######################################################
# Instantiate model
#######################################################
# Unchanged parameters
connection_rule = "fixed_total_number" # "fixed_indegree", "fixed_total_number"
PSC_rel_sd      = 0.1 # 0.1 for  Potjans' model
model           = model_class.model(connection_rule=connection_rule,
                                    PSC_rel_sd=PSC_rel_sd) 
master_seed = 0

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
print("T_connect    = ", T_connect)

###################################################
# Get connection numbers
###################################################
print("Count synapse numbers")
t_count_0 = time.time()

data_sup_path = sim.data_dir
sub_path = "micro"
data_path = os.path.join(data_sup_path, sub_path)
file_name   = "synapse_numbers.hdf5"

neuron_GIDs = all_GIDs[0]
# Histogram template
bin_size        = 10
hist_max        = 3000 * 3
n_bins_hist     = int(hist_max / bin_size) # maximal mean times 3

# Subset of target neurons to take the histogram from:
n_neurons = 1000

with h5py.File(os.path.join(data_path, file_name), "w") as data_file:
    data_file.attrs["structure"] = "n_connection_histogram = data_file[target_pop + '/' + source_pop]"
    data_file.attrs["n_neurons"] = n_neurons
    for target_index, target_pop in enumerate(model.populations):
        if verbose: print("to target " + target_pop)
        if verbose: print("from source")
        target_GIDs = neuron_GIDs[target_index][:n_neurons]
        target_group = data_file.create_group(target_pop)

        for source_index, source_pop in enumerate(model.populations):
            source_GIDs = neuron_GIDs[source_index] 
            
            if target_index == 3 and source_index == 2:
                # Get connections for entire populations
                conns   = nest.GetConnections(source=source_GIDs, target=target_GIDs)
                n_conns = np.unique(nest.GetStatus(conns, "target"), return_counts=True)[1]

                # Calculate histograms
                n_conns_hist = np.histogram(n_conns, bins=n_bins_hist, 
                                            range=(0, hist_max), density=False)[0] / n_neurons
                target_group.create_dataset(source_pop, data=n_conns_hist)

T_count   = time.time() - t_count_0
print("T_count    = ", T_count)
####################################################################################
