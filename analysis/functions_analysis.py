'''
    functions.py

    Contains helper functions for analysis.py
'''
import numpy as np
import os
import network_params as net

def get_GIDs_times(population, data_path, T0, sli=False):
    '''
    Get spike data from files.

    Expects: population, data_path, T0
    Option: sli (boolean) -- Get data of sli simulation.

    Returns GIDs and times (not ordered)
    '''
    GIDs = np.int_([])
    times = np.array([])
    file_names  = os.listdir(data_path)
    if sli:
        pop_id      = net.populations.index(population)
        type_id     = int(pop_id % len(net.types))
        layer_id    = int(pop_id / len(net.types))
        prefix = 'spikes_' + str(layer_id) + '_' + str(type_id)
    else:
        prefix = 'spikes_' + population
    for file_name in file_names:
        if file_name.startswith(prefix):
            file = open(data_path + file_name, 'r')
            for line in file:
                GID, time = line.split()
                time = float(time)
                if time > T0:
                    GIDs = np.append(GIDs, int(GID))
                    times = np.append(times, time)
            file.close()
    return GIDs, times

