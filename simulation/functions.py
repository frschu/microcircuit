'''
    functions.py

    Contains helper functions for microcircuit.py
'''
import numpy as np
import os
import network_params as net; reload(net)
import sim_params as sim; reload(sim)

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


def get_J_from_PSC():
    n_populations   = len(net.populations)
    n_layers        = len(net.layers)
    matrix_shape    = np.shape(net.conn_probs)  # shape of connection probability matrix

    def PSC(PSP, tau_m, tau_syn, C_m):
        # specify PSP and tau_syn_{ex, in}
        delta_tau   = tau_syn - tau_m
        ratio_tau    = tau_m / tau_syn
        PSC_over_PSP = C_m * delta_tau / (tau_m * tau_syn * \
            (ratio_tau**(tau_m / delta_tau) - ratio_tau**(tau_syn / delta_tau)))
        return PSP * PSC_over_PSP
    tau_m, tau_syn_ex, tau_syn_in, C_m = \
        [net.model_params[key] for key in ['tau_m', 'tau_syn_ex', 'tau_syn_in', 'C_m']]
    PSC_e       = PSC(net.PSP_e, tau_m, tau_syn_ex, C_m)    # excitatory (presynaptic)
    PSC_L4e_to_L23e  = PSC(net.PSP_L4e_to_L23e, tau_m, tau_syn_ex, C_m) # synapses from L4e to L23e
    PSP_i       = net.PSP_e * net.g                         # IPSP from EPSP
    PSC_i       = PSC(PSP_i, tau_m, tau_syn_in, C_m)        # inhibitory (presynaptic)
    PSC_ext     = PSC(net.PSP_ext, tau_m, tau_syn_ex, C_m)  # external poisson
    PSC_th      = PSC(net.PSP_th, tau_m, tau_syn_ex, C_m)   # thalamus

    # Convert PSCs to array, shape of conn_probs
    PSC_neurons = [[PSC_e, PSC_i] * n_layers] * n_populations
    PSC_neurons = np.reshape(PSC_neurons, matrix_shape)
    PSC_neurons[0, 2] = PSC_L4e_to_L23e

    R = net.model_params['C_m'] / net.model_params['tau_m']

    J_ab = R * PSC_neurons
    J_ext = R * PSC_ext
    return J_ab, J_ext

