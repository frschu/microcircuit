"""mean_field_functions.py"""
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from mean_field_params import *

# Mean and deviations of synaptic current of one neuron
mu  = lambda v: np.dot(mat1, v) + mu_ext
sd  = lambda v: np.sqrt(np.dot(mat2, v) + var_ext)    

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

# Brunel's analytic approximations
if choose_params == 'brunel':
    # Asymptotic for model A:
    #v0 for g < 4, high-activity regime
    v0_g_lt_4 = lambda g: (1. - (theta - V_r) / (C_e * J * (1 - g * gamma))) / t_ref
    #v0 for g > 4, low-activity regime
    v0_g_gt_4 = lambda g: (v_ext - v_thr) / (g * gamma - 1.)

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
