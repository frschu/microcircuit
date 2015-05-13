'''
    data_to_npy.py
   
    Transforms raw simulation data to npy files, stored in 
    output_path = user.data_dir/npy_data or npy_data_sli, accordingly. 
    
    Format: 
    n_GIDs:         Contains the upper boundary of spikes for each population;
                    n_GIDs[0] = 0 for convenience.

    times_GID_pop:  One file for each population containing all recorded spikes
                    ordered such that the entries [n_GIDs[i]: n_GIDs[i + 1]] are 
                    the ordered spike times of the i-th recorded neuron. 

    The information about the specific GID is dumped!
    
    Further command line arguments:
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Notes:
    # parameters in net and sim must remain unchanged between simulation and analysis!

'''
from __future__ import print_function
from imp import reload
import numpy as np
import sys, os, time
sys.path.append(os.path.abspath('../')) # include path with style
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specificaitons
sys.path.append(os.path.abspath('./simulation/')) # include path with simulation specificaitons
######################################################

# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import functions_analysis as functions; reload(functions)

# Data path
data_sup_path = user.data_dir
file_name = 'a1.0_t20.0_00/'
print(file_name)

simulation_path = data_sup_path + file_name
pynest_path = simulation_path + 'pynest/'
if 'sli' in sys.argv:
    sli = True
    data_path = simulation_path + 'sli/' 
    output_path = simulation_path + 'npy_data_sli/' 
else:
    sli = False
    data_path = simulation_path + 'pynest/' 
    output_path = simulation_path + 'npy_data/' 
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Get data specified for pynest simulation
populations = net.populations
n_populations = len(populations)
n_rec_spike = np.load(pynest_path + 'n_neurons_rec_spike.npy')

T0 = sim.t_trans # ms; starting point of analysis (avoid transients)

# statistics generated with sli code (not quite optimal, as parameters must correspond to 
# those of the pynest simulation!
if sli:         
    lower_GIDs = np.zeros(n_populations)
    GID_file = open(data_path + 'population_GIDs.dat', 'r')
    for i, line in enumerate(GID_file):
        lower_GIDs[i] = np.int_(line.split()[0])
    GID_file.close()


############################################################################################
for i, population in enumerate(populations):
    print(population)
    if sli:
        rec_spike_GIDs = lower_GIDs[i] + np.arange(n_rec_spike[i])
    else:
        rec_spike_GIDs = np.load(pynest_path + 'rec_spike_GIDs_' + population + '.npy')
    # Get data
    t0 = time.time()
    GIDs, times = functions.get_GIDs_times(population, data_path, T0, sli=sli)
    dt0 = time.time() - t0
    print('time to read data: %.3f s'%dt0)
   
    times_GID_pop = np.zeros(len(times))
    n_GIDs_pop = np.zeros(n_rec_spike[i] + 1).astype(int)
    
    t1 = time.time()
    last_GID = 0
    for j, GID in enumerate(rec_spike_GIDs):
        times_GID = times[GID == GIDs]
        n_times = len(times_GID)
        times_GID_pop[last_GID: last_GID + n_times] = times_GID
        last_GID += n_times
        n_GIDs_pop[j + 1] = last_GID

    dt1 = time.time() - t1
    print('time to sort data: %.3f s'%dt1)

    times_file = 'times_' + population + '.npy'
    n_GIDs_file = 'n_GIDs_' + population + '.npy'
    np.save(output_path + times_file, times_GID_pop)
    np.save(output_path + n_GIDs_file, n_GIDs_pop)

