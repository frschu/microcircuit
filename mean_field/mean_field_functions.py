"""mean_field_functions.py"""
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import sys, os
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specificaitons
import network_params as net; reload(net)
#from mean_field_params import *

# Mean and deviations of synaptic current of one neuron
mu  = lambda v: tau_m * (np.dot(mat1, v) + mu_ext)
sd  = lambda v: np.sqrt(tau_m * (np.dot(mat2, v) + var_ext))    

# The derived integrand of the integral equation
integrand   = lambda u: np.exp(u**2) * (1. + erf(u))
summand1    = lambda v: (-1. / v + t_ref) / (tau_m * np.pi)

# Function to be solved
def func(v):
    mu_v  = mu(v)
    sd_v  = sd(v)
    low = (V_r - mu_v) / sd_v
    up  = (theta - mu_v) / sd_v
    bounds      = np.array([low, up]).T
    integral    = np.array([quad(integrand, lower, upper)[0] for lower, upper in bounds])
    zero        = - 1. / v + t_ref + np.pi * tau_m * integral
    return zero

# Corresponding Jacobian
def jacobian(v):
    mu_v  = mu(v)
    sd_v  = sd(v)
    low = (V_r - mu_v) / sd_v
    up  = (theta - mu_v) / sd_v
    f_low   = integrand(low)
    f_up    = integrand(up)
    jac_T = np.diag(v) - (jac_mat1 * (f_up - f_low) + jac_mat2 * (up * f_up - low * f_low) / sd_v**2)
    return jac_T.T

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours


def get_J_from_PSC():
    """Not applied - wrong approach!"""
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

