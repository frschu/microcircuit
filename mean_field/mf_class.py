"""mf_class.py

Contains main class
"""
from imp import reload
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../simulation/")) # include path with simulation specificaitons
# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)

######################################################
# Main class
######################################################
class mf_net:
    def __init__(self, choose_model='brunelA', n_pop=2, g=6., v_ext_factor=2.):
        '''Initialize according to the model chosen'''
        self.choose_model = choose_model
        if choose_model.startswith('brunel'):
            ######################################################
            # Brunel's parameters
            ######################################################
            if n_pop > 2:
                raise Exception("Brunel's model excepts n_pop < 3!")
            self.populations = ['e', 'i']
            self.V_r     = 10.       # mV
            self.theta   = 20.       # mV
            self.t_ref   = 0.002     # s
            self.tau_m   = 0.02      # s
            # Weights
            J     =  0.2      # mV
            if choose_model.endswith('A'):
                J_i     =  J      
                g_i     =  g 
            elif choose_model.endswith('B'):
                J_i     =  0.2      # mV
                g_i     =  4. 
            J_ab    = np.array([[J, -g * J], [J_i, -g_i * J_i]])
            self.J_ext   = J       # In Brunels paper, J_i,ext = J_i
            # Synapse numbers
            C_e     = 4000.
            gamma   = 0.25
            C_i     = gamma * C_e
            C_ab    = np.array([[C_e, C_i], [C_e, C_i]]) # depends only on presynaptic population
            C_aext  = np.array([C_e, C_e])
            # Background rate
            # External frequency in order to reach threshold without recurrence
            self.v_thr   = self.theta / (C_e * J * self.tau_m)
            self.v_ext   = self.v_thr * v_ext_factor 
            
        else:
            ######################################################
            # Microcircuit model parameters
            ######################################################
            self.populations = net.populations
            # Neuron model
            # Reset voltage and threshold (set V_r to zero)
            V_reset, V_th= [net.model_params[key] for key in ('V_reset', 'V_th')]
            self.V_r    = 0.0
            self.theta  = V_th - V_reset 
            # All times should be in seconds!
            self.t_ref  = net.model_params['t_ref'] * 1e-3
            self.tau_m  = net.model_params['tau_m'] * 1e-3
            # Weights
            n_populations   = len(net.populations)
            n_layers        = len(net.layers)
            matrix_shape    = np.shape(net.conn_probs)  # shape of connection probability matrix
            
            J           = net.PSP_e
            J_ab        = [[J, -g * J] * n_layers] * n_populations
            J_ab        = np.reshape(J_ab, matrix_shape)
            J_ab[0, 2]  = net.PSP_L4e_to_L23e
            self.J_ext  = net.PSP_ext
            # Synapse numbers
            n_neurons   = net.full_scale_n_neurons
            K_ab        = np.log(1. - net.conn_probs) / np.log(1. - 1. / np.outer(n_neurons, n_neurons))
            C_ab        = K_ab / n_neurons
            C_aext      = net.K_bg
            # Background rate
            self.v_ext       = net.bg_rate * v_ext_factor

        ######################################################
        # Rescale to n_pop populations!
        ######################################################
        self.populations = self.populations[:n_pop]
        self.J_ab = J_ab[:n_pop, :n_pop]
        self.C_ab = C_ab[:n_pop, :n_pop]
        self.C_aext = C_aext[:n_pop]

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

    def brunel_v0(self, gs, v_ext=None):
        """Brunel's analytic approximations for firing rates for model A 
        in the regimes g < 4 and g > 4. 
        Expects numpy array with gs as entries.
        For g > 4., provide v_ext!
        """
        if v_ext == None:
            v_ext = self.v_ext
        v0_theo = np.zeros(gs.shape)
        if self.choose_model == 'brunelA':
            C_e     = self.C_ab[0, 0]
            gamma   = self.C_ab[0, 1] / C_e
            J       = self.J_ab[0, 0]
            mask = gs <= 4.
            v0_theo[mask]   = (1. - (self.theta - self.V_r) / \
                    (C_e * J * (1. - gs[mask] * gamma))) / self.t_ref
            v0_theo[~mask]  = (v_ext - self.v_thr) / (gs[~mask] * gamma - 1.)
            v0_theo[v0_theo < 0] = 0
        return v0_theo

######################################################
