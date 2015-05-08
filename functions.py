'''
    functions.py

    Contains helper functions for microcircuit.py
'''

def get_output_path(area, sim_t, n_th, dc_amplitude, data_dir, overwrite=False, n_digits=2):
    '''
    returns directory where data is to be saved
    creates the corresponding subdirectory in data_dir, if not existing
    if it exists and 'sim.overwrite_existing_files == False', 
    it creates a new folder with increased number (n_digits).
    
    subdirectory example: a0.1_t10.0_th_dc_00
    simulation specifications include: 
        area, simulation time in s, 
        whether thalamus exists and background is DC current
    '''
    import os
    sim_spec = 'a%.1f_t%.1f_'%(area, sim_t * 1e-3)
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
    return output_path + '/'
