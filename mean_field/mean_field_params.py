"""mean_field_functions.py"""
from imp import reload
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specificaitons
# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
######################################################

# Global parameters
choose_params = ['microcircuit', 'brunel'][1]
n_pop = 2
# Initial guess
v_guess = np.array([2, 2, 2, 2, 2, 2, 1, 2])


if choose_params == 'microcircuit':
    ######################################################
    # Microcircuit model parameters
    ######################################################
    # Neuron model
    # Reset voltage and threshold (set V_r to zero)
    V_reset, V_th= [net.model_params[key] for key in ('V_reset', 'V_th')]
    V_r     = 0.0
    theta   = V_th - V_reset 
    # All times should be in seconds!
    t_ref, tau_m = [net.model_params[key] * 1e-3 for key in ('t_ref', 'tau_m')]
    # Weights
    def get_J():
        n_populations   = len(net.populations)
        n_layers        = len(net.layers)
        matrix_shape    = np.shape(net.conn_probs)  # shape of connection probability matrix
        
        PSP_i       = net.PSP_e * net.g                         # IPSP from EPSP
        PSP_neurons = [[net.PSP_e, PSP_i] * n_layers] * n_populations
        PSP_neurons = np.reshape(PSP_neurons, matrix_shape)
        PSP_neurons[0, 2] = net.PSP_L4e_to_L23e
        J_ab = PSP_neurons
        J_ext = net.PSP_ext
        return J_ab, J_ext
    J_ab, J_ext = get_J()
    # Synapse numbers
    n_neurons   = net.full_scale_n_neurons
    K_ab        = np.log(1. - net.conn_probs   ) / np.log(1. - 1. / np.outer(n_neurons, n_neurons))
    C_ab        = K_ab / n_neurons
    C_aext      = net.K_bg
    # Background rate
    v_ext       = net.bg_rate
    
elif choose_params == 'brunel':
    ######################################################
    # Brunel's parameters
    ######################################################
    model   = ['A', 'B'][0]
    V_r     = 10.       # mV
    theta   = 20.       # mV
    t_ref   = 0.002     # s
    tau_m   = 0.02      # s
    # Weights
    J     =  0.2      # mV
    g     =  6.
    if model == 'A':
        J_i     =  J      
        g_i     =  g 
    else:
        J_i     =  0.2      # mV
        g_i     =  4. 
    J_ab    = np.array([[J, -g * J], [J_i, -g_i * J_i]])
    J_ext   = J       # In Brunels paper, J_i,ext = J_i
    # Synapse numbers
    C_e     = 4000.
    gamma   = 0.25
    C_i     = gamma * C_e
    C_ab    = np.array([[C_e, C_i], [C_e, C_i]]) # depends only on presynaptic population
    C_aext  = np.array([C_e, C_e])
    # Background rate
    # External frequency in order to reach threshold without recurrence
    v_thr   = theta / (C_e * J * tau_m)
    v_ext   = 1.5 * v_thr
    if n_pop > 2:
        n_pop   = 2         # This must be 2 for Brunel's parameters
    
######################################################
# Test for the case of 1 and two populations!
######################################################
populations = net.populations[:n_pop]
J_ab = J_ab[:n_pop, :n_pop]
C_ab = C_ab[:n_pop, :n_pop]
C_aext = C_aext[:n_pop]
v_guess = v_guess[:n_pop]

######################################################
# Predefine matrices
######################################################
mat1 = C_ab * J_ab
mat2 = C_ab * J_ab ** 2
mu_ext = J_ext * C_aext * v_ext
sd_ext = J_ext ** 2 * C_aext * v_ext
jac_mat1 = np.pi * tau_m**2 * mat1.T
jac_mat2 = np.pi * tau_m**2 * 0.5 * mat2.T
