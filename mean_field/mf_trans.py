"""mf_trans.py

Class for the transformation from Brunel in 8D to microcircuit model. 
Contains parameters and functions of stationary frequency v.
"""
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specificaitons
sys.path.append(os.path.abspath("../mean_field/")) # include path with mean field approximation
# Import specific moduls
from imp import reload
import network_params as net_params; reload(net_params)
import sim_params as sim; reload(sim)

######################################################
# Main class
######################################################
class mf_net:
    def __init__(self, g=4., j02=1., C_ab=None):
        '''Initialize according to the model chosen
        Specify g, j02 and C_ab as approaching the microcircuit model.
        Note that V_r, theta, tau_m and v_ext are already changed to values of 
        the microcircuit model.
        '''
        n_layer = 4
        n_types = 2
        n_pop   = n_layer * n_types
        
        ######################################################
        # Brunel's parameters
        ######################################################
        self.populations = np.array(net_params.populations)[:n_pop]
        self.V_r    = 0.       # mV
        self.theta  = 15.       # mV
        self.t_ref  = 0.002     # s
        self.tau_m  = 0.01      # s
        # Weights
        J     =  0.15      # mV
        self.J_ab   = np.tile([J, -g * J], (n_pop, n_layer))
        self.J_ab[0, 2] = J * j02
        self.J_ext  = J
        # Synapse numbers
        if C_ab == None:
            C_e     = 500. # mean for microcircuit = 501
            gamma   = 0.25
            C_i     = gamma * C_e
            self.C_ab   = np.tile([C_e, C_i], (n_pop, n_layer)) # depends only on presynaptic population
        else:
            self.C_ab   = C_ab
        #n_neurons   = net_params.full_scale_n_neurons
        #K_ab        = np.log(1. - net_params.conn_probs) / np.log(1. - 1. / np.outer(n_neurons, n_neurons))
        #self.C_ab   = (K_ab / n_neurons)[:n_pop, :n_pop]
        #self.C_aext = np.tile([2000], n_pop)
        self.C_aext = net_params.K_bg[:n_pop]
        # Background rate
        # External frequency in order to reach threshold without recurrence
        self.v_ext  = net_params.bg_rate

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
