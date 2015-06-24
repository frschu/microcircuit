'''functions.py

    Contains helper functions for microcircuit.py
'''
import numpy as np
import os
import network_params as net; reload(net)
import sim_params as sim; reload(sim)

def get_output_path(data_dir, area, sim_t, n_th=0, dc_amplitude=0, overwrite=False, n_digits=2):
    '''
    returns directory where data is to be saved:
    'data_dir/simulation_specifications/pynest/'

    if the constructed filename exists and 'sim.overwrite_existing_files == False',
    the filename's last digit will be raised one above the highest existing number.
    
    'simulation_specifications' contain: 
    a           = area in decimals of 1 mm^2 (full scale);
    t           = simulation time in s;
    th, dc      = whether thalamus or dc background current are connected;
    00, 01, ... = nth experiment of this kind.

    Example: 'a0.1_t10.2_th_dc_00/'
    '''
    sim_spec = 'a%.1f_t%.1f'%(area, sim_t * 1e-3)
    options = ''
    if not n_th == 0:
        options += '_th'
    if not dc_amplitude == 0:
        options += '_dc'
    sim_spec += options
    subdir = sim_spec + '_' + '0' * n_digits
    existing_subdirs = os.listdir(data_dir)
    if subdir in existing_subdirs:
        if overwrite:
            output_path = data_dir + subdir
        else:
            experiment_numbers = [
                name[-n_digits:] for name in existing_subdirs 
                if name[:-(n_digits + 1)] == sim_spec]
            max_number = max(np.int_(experiment_numbers))
            subdir_new = sim_spec + '_' + str.zfill(str(max_number + 1), n_digits)
            output_path = data_dir + subdir_new
    else: 
        output_path = data_dir + subdir
    output_path += '/pynest/'
    return output_path

