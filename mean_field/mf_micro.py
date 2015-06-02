"""mf_micro.py

Class for mean field approximation of microcircuit model. 
Contains parameters and functions of stationary frequency v.
"""
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specificaitons
# Import specific moduls
from imp import reload
import network_params as net; reload(net)
import sim_params as sim; reload(sim)

######################################################
# Main class
######################################################
class mf_net:
    def __init__(self, n_layer=4):
        '''Initialize according to the model chosen
        '''
        n_types = 2
        n_pop   = n_layer * n_types

        ######################################################
        # Microcircuit model parameters
        ######################################################
        self.populations = np.array(net.populations)[:n_pop]
        # Neuron model
        # Reset voltage and threshold (set V_r to zero)
        V_reset, V_th= [net.model_params[key] for key in ('V_reset', 'V_th')]
        self.E_L    = net.model_params['E_L']
        self.V_r    = V_reset - self.E_L
        self.theta  = V_th - self.E_L 
        # All times should be in seconds!
        self.t_ref  = net.model_params['t_ref'] * 1e-3
        self.tau_m  = net.model_params['tau_m'] * 1e-3
        # Weights
        self.J_ab   = net.PSPs[:n_pop, :n_pop]
        self.J_ext  = net.PSP_ext
        # Synapse numbers
        n_neurons   = net.full_scale_n_neurons
        K_ab        = np.log(1. - net.conn_probs) / np.log(1. - 1. / np.outer(n_neurons, n_neurons))
        self.C_ab   = (K_ab / n_neurons)[:n_pop, :n_pop]
        self.C_aext = net.K_bg[:n_pop]
        # Background rate
        self.v_ext  = net.bg_rate

        ######################################################
        # Predefine matrices
        ######################################################
        self.mu_ext = self.J_ext * self.C_aext * self.v_ext
        self.var_ext = self.J_ext ** 2 * self.C_aext * self.v_ext
        self.mat1 = self.C_ab * self.J_ab
        self.mat2 = self.C_ab * self.J_ab ** 2
        self.jac_mat1 = np.pi * self.tau_m**2 * self.mat1.T
        self.jac_mat2 = np.pi * self.tau_m**2 * 0.5 * self.mat2.T


    ######################################################
    # Methods
    ######################################################
    def mu(self, v):
        return self.tau_m * (np.dot(self.mat1, v) + self.mu_ext)
        
    def sd(self, v):
        return np.sqrt(self.tau_m * (np.dot(self.mat2, v) + self.var_ext))
    
    def integrand(self, u):
        from scipy.special import erf
        return np.exp(u**2) * (1. + erf(u))
    
    def summand1(self, v):
        return (-1. / v + self.t_ref) / (self.tau_m * np.pi)

    def root_v0(self, v):
        """The integral equations to be solved
        Returns the array 'root', each entry corresponding to one population.
        Solve for root == 0.
        """
        from scipy.integrate import quad
        mu_v  = self.mu(v)
        sd_v  = self.sd(v)
        low = (self.V_r - mu_v) / sd_v
        up  = (self.theta - mu_v) / sd_v
        bounds      = np.array([low, up]).T
        integral    = np.array([quad(self.integrand, lower, upper)[0] for lower, upper in bounds])
        root        = - 1. / v + self.t_ref + np.pi * self.tau_m * integral
        return root

    def jacobian(self, v):
        """The Jacobian of root_v0.
        Used to ease the process of solving.
        The calculations are done transposed to avoid unnecessary transposes (adding axes to mu and sd)
        """
        mu_v  = self.mu(v)
        sd_v  = self.sd(v)
        low = (self.V_r - mu_v) / sd_v
        up  = (self.theta - mu_v) / sd_v
        f_low   = self.integrand(low)
        f_up    = self.integrand(up)
        jac_T = np.diag(v) - \
            (self.jac_mat1 * (f_up - f_low) + self.jac_mat2 * (up * f_up - low * f_low) / sd_v**2)
        return jac_T.T

######################################################
