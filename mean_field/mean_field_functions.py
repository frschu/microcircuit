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
