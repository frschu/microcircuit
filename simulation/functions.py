'''
    functions.py

    Contains helper functions for microcircuit.py
'''
import numpy as np
import os

def get_output_path(area, sim_t, n_th, dc_amplitude, data_dir, overwrite=False, n_digits=2):
    '''
    returns directory where data is to be saved:
    'data_dir/simulation_specifications/pynest/'

    if the constructed filename exists and 'sim.overwrite_existing_files == False',
    the filename's last digit will be raised one above the highest existing number.
    
    subdirectory example: a0.1_t10.0_th_dc_00/pynest
    simulation specifications include: 
        area, simulation time in s, 
        whether thalamus exists and background is DC current
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


