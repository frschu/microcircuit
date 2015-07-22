"""brunel_model_class.py

    Brunel A with two populations.
    
"""
from imp import reload
import numpy as np
import brunel_network_params as net; reload(net)

class model:
    def __init__(self, 
                 neuron_model=net.neuron_model,               # "iaf_psc_delta" or "iaf_psc_exp"
                 n_neurons="brunel", C_ab="brunel",           # else: "brunel" or arrays
                 connection_rule=net.connection_rule,      # "fixed_total_number" or "fixed_indegree"
                 weight_rel_sd=net.weight_rel_sd, 
                 delay_rel_sd=net.delay_rel_sd,  
                 g=net.g, 
                 rate_ext_factor=net.rate_ext_factor):

        """Class of network parameters.

        Contains:
        - network parameters
        - single-neuron parameters
        - stimulus parameters
        
        Specify area, neuron_model, neuron_numbers, C_ab, 
            j02, connection_rule, weight_rel_sd, g, rate_ext_factor.
        Default values correspond to Potjans' model.

        Neuron numbers and synapse numbers (C_ab) can be specified separately.
        Both except either a string in {'brunel'} or an array of 
        the corresponding shape.

        Naming: C_ab, C_aext, J_ab, J_ext, rate_ext
        Don't use conn_probs, K_bg, PSPs, PSP_ext, v_ext, any more!
        """
        ###################################################
        ###     	Network parameters		###        
        ###################################################
        self.area    = 1.0 # remnant
        
        self.layers         = net.layers    #np.array(["L23", "L4", "L5", "L6"])
        self.types          = net.types     #np.array(["e", "i"]) 
        self.populations    = np.array([layer + typus for layer in self.layers for typus in self.types])
        self.n_populations  = len(self.populations)
        self.n_layers       = len(self.layers)
        self.n_types        = len(self.types)
        
        # Neuron numbers
        if n_neurons == "brunel":
            self.n_neurons  = net.n_neurons
        else:
            if type(n_neurons) == np.ndarray:
                if n_neurons.shape == (self.n_populations, ):
                    self.n_neurons   = np.int_(n_neurons)
                else:
                    raise Exception("'n_neurons' has wrong shape. "+
                                    "Expects (%i,)"%self.n_populations)
            else: 
                raise Exception("'n_neurons' expects either numpy.ndarray or string "+
                                "in {'micro', 'brunel'}")
        self.n_total    = np.sum(self.n_neurons)

        # Synapse numbers
        # Number for fixed indegree
        C_ab    = net.epsilon * np.tile(self.n_neurons, (2, 1))
        # Number for fixed total number
        K_ab            = (C_ab.T * self.n_neurons).T
        # Connection probabilities: conn_probs[post, pre] = conn_probs[target, source]
        self.conn_probs = 1. - (1. - 1. / np.outer(self.n_neurons, self.n_neurons)) ** K_ab

        self.connection_rule = connection_rule
        if C_ab == "brunel":
            if self.connection_rule == "fixed_total_number":
                self.C_ab = K_ab   # total number 
            elif self.connection_rule == "fixed_indegree":
                self.C_ab = C_ab
            else:
                raise Exception("Unexpected connection type. Use 'fixed_total_number' for microcircuit " + 
                                "model or 'fixed_indegree' for Brunel's model!")
        else:
            if type(C_ab) == np.ndarray:
                if C_ab.shape == (self.n_populations, self.n_populations):
                    self.C_ab   = np.int_(C_ab)
                else:
                    raise Exception("'C_ab' has wrong shape. "+
                                    "Expects (%i, %i)"%(self.n_populations, self.n_populations))
            else: 
                raise Exception("'C_ab' expects either numpy.ndarray or string "+
                                "in {'micro', 'brunel'}")


        ###################################################
        ###          Single-neuron parameters		###        
        ###################################################
        self.neuron_model   = neuron_model
        self.Vm0_mean       = net.Vm0_mean            # mean of initial membrane potential (mV)
        self.Vm0_std        = net.Vm0_std            # std of initial membrane potential (mV)
        self.model_params   = net.model_params
        if not self.neuron_model=="iaf_psc_delta":
            self.model_params["tau_syn_ex"] = net.tau_syn_ex # excitatory synaptic time constant (ms)
            self.model_params["tau_syn_in"] = net.tau_syn_in # inhibitory synaptic time constant (ms)
            self.tau_syn_ex = net.tau_syn_ex * 1e-3             # s
            self.tau_syn_in = net.tau_syn_in * 1e-3             # s
            self.tau_syn    = np.tile([self.tau_syn_ex, self.tau_syn_in], (self.n_populations, self.n_layers))
        # Rescaling for model calculations: these values are not used in the simulation!
        self.tau_m  = self.model_params["tau_m"] * 1e-3          # s
        self.t_ref  = self.model_params["t_ref"] * 1e-3          # s
        self.E_L    = self.model_params["E_L"]                   # mV
        self.V_r    = self.model_params["V_reset"] - self.E_L    # mV
        self.theta  = self.model_params["V_th"] - self.E_L       # mV
        self.C_m    = self.model_params["C_m"]                   # pF


        ######################################################
        # Synaptic weights. Depend on neuron_model!         ##
        ######################################################
        self.g      = g
        g_all       = np.tile([1., -self.g], (self.n_populations, self.n_layers))
        
        self.J      = net.PSP_e           # mv; mean EPSP, used as reference PSP
        self.J_ab   = self.J * g_all
        self.weight_rel_sd = weight_rel_sd # Standard deviation of weight relative to mean weight
        
        # Actual weights have to be adapted, as a postsynaptic POTENTIAL is used
        if   self.neuron_model=="iaf_psc_delta":
            self.weights    = self.J_ab     # neuron populations
        elif self.neuron_model=="iaf_psc_exp": # PSCs calculated from PSP amplitudes
            delta_tau       = self.tau_syn - self.tau_m
            ratio_tau       = self.tau_m / self.tau_syn
            PSC_over_PSP    = self.C_m * delta_tau / (self.tau_m * self.tau_syn * \
                (ratio_tau**(self.tau_m / delta_tau) - ratio_tau**(self.tau_syn / delta_tau))) * 1e-3
            self.weights    = self.J_ab  * PSC_over_PSP     # neuron populations
        elif self.neuron_model=="iaf_psc_alpha": # PSCs calculated from PSP amplitudes
            raise Exception("Neuron model: iaf_psc_alpha. Check units of weights before applying!")
            self.weights = self.J_ab * np.exp(1) # see Sadeh 2014
        else:
            raise Exception("Neuron model should be iaf_psc_ - {delta, exp, alpha}!")


        ###################################################
        ###          Delays and dicts                   ###        
        ###################################################
        # mean dendritic delays for excitatory and inhibitory transmission (ms)
        self.delay_e = net.delay_e   # ms, excitatory synapses
        self.delay_i = net.delay_i   # ms, inhibitory synapses
        self.delays  = np.tile([self.delay_e, self.delay_i], (self.n_populations, self.n_layers)) # adapt...
        self.delay_rel_sd = delay_rel_sd 
        
        # Synapse dictionaries
        # default connection dictionary
        self.conn_dict   = {"rule": connection_rule}
        # weight distribution of connections between populations
        self.weight_dict_exc = net.weight_dict_exc
        self.weight_dict_inh = net.weight_dict_inh
        # delay distribution of connections between populations
        self.delay_dict  = net.delay_dict
        # default synapse dictionary
        self.syn_dict = net.syn_dict
       

        ###################################################
        ###          External stimuli                    ##        
        ###################################################
        # rate of background Poisson input at each external input synapse (spikes/s) 
        self.J_ext      = net.PSP_ext   # external synaptic weight
        self.rate_ext_factor   = rate_ext_factor 
        self.rate_theta = self.theta / (self.J_ext * self.C_ab[0, 0] * self.tau_m) # Hz; threshold rate without feedback (tau_m in s...)
        self.rate_ext   = self.rate_ext_factor * self.rate_theta    # actual background rate
        self.delay_ext  = self.delay_e  # ms;  mean delay of external input
        self.dc_amplitude = net.dc_amplitude  # constant bg amplitude
        self.C_aext     = net.C_aext        # in-degrees for background input
        # Adapt weights
        if self.neuron_model=="iaf_psc_delta":
            self.weight_ext = self.J_ext    
        elif self.neuron_model=="iaf_psc_exp": # PSCs calculated from PSP amplitudes
            self.weight_ext = self.J_ext * PSC_over_PSP[0, 0] 
        elif self.neuron_model=="iaf_psc_alpha": # PSCs calculated from PSP amplitudes
            self.weight_ext = self.J_ext * np.exp(1) 

        # optional additional thalamic input (Poisson)
        self.n_th           = 0     # size of thalamic population


        ######################################################
        # Predefine matrices for mean field                 ##
        ######################################################
        if self.neuron_model=="iaf_psc_delta":
            self.J_mu       = self.weights    
            self.J_sd       = self.weights    
            self.J_mu_ext   = self.weight_ext     
            self.J_sd_ext   = self.weight_ext 
        elif self.neuron_model=="iaf_psc_exp":
            pA_to_mV            = 1e3 / self.C_m # Factor for conversion from pA to mV
            self.J_mu       = self.weights    * self.tau_syn    * pA_to_mV
            self.J_sd       = self.weights    * self.tau_syn    * pA_to_mV / np.sqrt(2.)
            self.J_mu_ext   = self.weight_ext * self.tau_syn_ex * pA_to_mV
            self.J_sd_ext   = self.weight_ext * self.tau_syn_ex * pA_to_mV / np.sqrt(2.)
        elif self.neuron_model=="iaf_psc_alpha":
            pA_to_mV            = 1e3 / self.C_m # Factor for conversion from pA to mV
            self.J_mu       = self.weights    * self.tau_syn    * pA_to_mV
            self.J_sd       = self.weights    * self.tau_syn    * pA_to_mV / 2.
            self.J_mu_ext   = self.weight_ext * self.tau_syn_ex * pA_to_mV
            self.J_sd_ext   = self.weight_ext * self.tau_syn_ex * pA_to_mV / 2.
        self.mat_mu     = self.J_mu        * self.C_ab
        self.mat_sigma  = self.J_sd**2     * self.C_ab
        self.mu_ext     = self.J_mu_ext    * self.C_aext * self.rate_ext
        self.var_ext    = self.J_sd_ext**2 * self.C_aext * self.rate_ext

    ######################################################
    # Methods                                           ##
    ######################################################
    def mu(self, v):
        """Mean input in Brunel's model"""
        return self.tau_m * (np.dot(self.mat_mu, v) + self.mu_ext)

    def sd(self, v):
        """Fluctuation of input in Brunel's model"""
        return np.sqrt(self.tau_m * (1 + self.weight_rel_sd ** 2) * (np.dot(self.mat_sigma, v) + self.var_ext))

    def root_v0(self, v):
        """The integral equations to be solved
        Returns the array 'root', each entry corresponding to one population.
        Solve for root == 0.
        """
        from scipy.integrate import quad
        from scipy.special import erf
        mu_v  = self.mu(v)
        sd_v  = self.sd(v)
        low = (self.V_r - mu_v) / sd_v
        up  = (self.theta - mu_v) / sd_v
        bounds      = np.array([low, up]).T

        def integrand(u):
            if u < -4.0:
                return -1. / np.sqrt(np.pi) * (1.0 / u - 1.0 / (2.0 * u**3) + 
                                            3.0 / (4.0 * u**5) - 
                                            15.0 / (8.0 * u**7))
            else:
                return np.exp(u**2) * (1. + erf(u))
        integral    = np.array([quad(integrand, lower, upper)[0] for lower, upper in bounds])
        root        = - 1. / v + self.t_ref + np.sqrt(np.pi) * self.tau_m * integral
        return root


