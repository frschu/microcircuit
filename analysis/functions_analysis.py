'''
    functions.py

    Contains helper functions for analysis.py
'''
import numpy as np
import os, time
import network_params as net

def get_GIDs_times(population, data_path, t_trans, sli=False):
    '''
    Get spike data from files.

    Expects: population, data_path, t_trans
    Option: sli (boolean) -- Get data of sli simulation.

    Returns GIDs and times (not ordered)
    '''
    GIDs = []
    times = []
    if sli:
        pop_id      = np.where(net.populations == population)[0][0]
        type_id     = int(pop_id % len(net.types))
        layer_id    = int(pop_id / len(net.types))
        prefix = 'spikes_' + str(layer_id) + '_' + str(type_id)
    else:
        prefix = 'spikes_' + population
    file_names_all  = os.listdir(data_path)
    file_names = [file_name for file_name in file_names_all if file_name.startswith(prefix)]
    for file_name in file_names:
        print(file_name)
        with open(data_path + file_name, 'r') as file:
            for line in file:
                GID, time = line.split()
                time = float(time)
                if time >= t_trans:
                    GIDs.append(int(GID))
                    times.append(time)
    GIDs = np.array(GIDs)
    times = np.array(times)
    return GIDs, times

def get_voltages(population, data_path, t_trans, sli=False):
    '''
    Get membrane voltages from files.

    Expects: population, data_path, t_trans
    Option: sli (boolean) -- Get data of sli simulation.

    Returns Vs[GID, time_index] and dt. 
    '''
    GIDs = []
    if sli:
        pop_id      = np.where(net.populations == population)[0][0]
        type_id     = int(pop_id % len(net.types))
        layer_id    = int(pop_id / len(net.types))
        prefix = 'voltages_' + str(layer_id) + '_' + str(type_id)
    else:
        prefix = 'voltages_' + population
    file_names_all  = os.listdir(data_path)
    file_names = [file_name for file_name in file_names_all if file_name.startswith(prefix)]
    for i, file_name in enumerate(file_names):
            print(file_name)
            Vs = []
            j  = 0
            t0 = 0
            t1 = 0
            with open(data_path + file_name, 'r') as file:
                for line in file:
                    GID, time, V = line.split()
                    time = float(time)
                    if time >= t_trans:
                        Vs.append(float(V))
                        if t0 == 0:
                            t0 = time
                        elif t1 == 0:
                            j += 1
                            if time != t0:
                                t1 = time
                                n_GIDs = j
            if i == 0:
                dt = t1 - t0
                t_max   = time
                n_times = (t_max - t0) / dt + 1
                Vs_all = np.reshape(Vs, (n_GIDs, n_times))
            else:
                if Vs != []:
                    Vs_all = np.vstack((Vs_all, np.reshape(Vs, (n_GIDs, n_times))))
    return Vs_all, t_max, dt

