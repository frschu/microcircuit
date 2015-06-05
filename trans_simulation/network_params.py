"""network_params.py

    Network parameters for the network to be changed from Brunel A to microcircuit
"""
import numpy as np

class net:
    def __init__(self, area=0.1, n_neurons="micro", C_ab="micro", 
                 j02=2., connection_type="fixed_total_number", 
                 PSC_rel_sd=0.1, g=4., rate_ext=8.0):
        """Class of network parameters.

        Contains:
        - network parameters
        - single-neuron parameters
        - stimulus parameters
        
        Specify area, neuron_numbers, C_ab, j02, connection_type, PSC_rel_sd, g, rate_ext.
        Default values correspond to Potjans' model.

        Neuron numbers and synapse numbers (C_ab) can be specified separately.
        Both except either a string in {'micro', 'brunel'} or an array of 
        the corresponding shape.

        Note that V_r, theta, tau_m and rate_ext are already changed to values of 
        the microcircuit model, thus not the original values of the Brunel model.
        
        Brunel's model:
        j02     = 1.0
        connection_type = fixed_indegree
        PSC_rel_sd      = 0.0
        g, rate_ext varied accordingly

        Naming: C_ab, C_aext, J_ab, J_ext
        Don"t use conn_probs, K_bg, PSPs, PSP_ext any more!
        """
        ###################################################
        ###     	Network parameters		###        
        ###################################################
        # area of network in mm^2; scales numbers of neurons
        # use 1 for the full-size network (77,169 neurons)
        self.area    = area

        # Whether to scale number of synapses K linearly: K = K_full_scale * area.
        # When scale_K_linearly is false, K is derived from the connection probabilities and
        # scaled neuron numbers according to eq. (1) of the paper. In first order 
        # approximation, this corresponds to K = K_full_scale * area**2.
        # Note that this produces different dynamics compared to the original model.
        self.scale_K_linearly  = True
        
        self.layers         = np.array(["L23", "L4", "L5", "L6"])
        self.types          = np.array(["e", "i"]) 
        self.populations    = np.array([layer + typus for layer in self.layers for typus in self.types])
        self.n_populations  = len(self.populations)
        self.n_layers       = len(self.layers)
        self.n_types       = len(self.types)
        
        # Neuron numbers
        full_scale_n_neurons = np.array( \
          [20683,   # layer 2/3 e
           5834,    # layer 2/3 i
           21915,   # layer 4 e
           5479,    # layer 4 i
           4850,    # layer 5 e
           1065,    # layer 5 i
           14395,   # layer 6 e
           2948])   # layer 6 i
        self.n_total    = int(np.sum(full_scale_n_neurons) * self.area)

        if n_neurons == "micro":
            self.n_neurons  = np.int_(full_scale_n_neurons * self.area)
        elif n_neurons == "brunel":
            # Provide an array of equal number of neurons in each exc./inh. population
            gamma       = 0.25
            inh_factor  = 1. / (gamma + 1.)
            exc_factor  = 1. - inh_factor 
            N_exc       = self.n_total/self.n_populations * exc_factor
            N_inh       = self.n_total/self.n_populations * inh_factor
            self.n_neurons  = np.tile([N_exc, N_inh], self.n_layers).astype(int)
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

        # Weights
        self.g      = g
        self.j02    = j02

        g_all       = np.tile([1., self.g], (self.n_populations, self.n_layers))
        L23e_index  = np.where(self.populations == "L23e")[0][0]
        L4e_index   = np.where(self.populations == "L4e")[0][0]
        g_all[L23e_index, L4e_index] *= self.j02
        J           = 0.15           # mv; mean EPSP, used as reference PSP
        
        self.J_ab   = J * g_all
        self.PSC_rel_sd = PSC_rel_sd # Standard deviation of PSC amplitudes relative to mean PSC amplitudes
        
        # Synapse numbers
        # Connection probabilities
        # conn_probs[post, pre] = conn_prob[target, source]
        # source      L2/3e   L2/3i   L4e     L4i     L5e     L5i     L6e     L6i       
        conn_probs = np.array(
                    [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.    , 0.0076, 0.    ],
                     [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.    , 0.0042, 0.    ],
                     [0.0077, 0.0059, 0.0497, 0.135 , 0.0067, 0.0003, 0.0453, 0.    ],
                     [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.    , 0.1057, 0.    ],
                     [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
                     [0.0548, 0.0269, 0.0257, 0.0022, 0.06  , 0.3158, 0.0086, 0.    ],
                     [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                     [0.0364, 0.001 , 0.0034, 0.0005, 0.0277, 0.008 , 0.0658, 0.1443]])
        # Scale synapse numbers of the C_ab
        if self.scale_K_linearly:
            n_outer_full    = np.outer(full_scale_n_neurons, full_scale_n_neurons)
            K_full_scale    = np.log(1. - conn_probs   ) / np.log(1. - 1. / n_outer_full)
            K_scaled        = np.int_(K_full_scale * self.area)
        else:
            n_outer         = np.outer(self.n_neurons, self.n_neurons)
            K_scaled        = np.int_(np.log(1. - conn_probs   ) / np.log(1. - 1. / n_outer))

        self.connection_type = connection_type
        if self.connection_type == "fixed_total_number":
            C_ab_micro = K_scaled   # total number, do not divide! 
        elif self.connection_type == "fixed_indegree":
            #C_ab_micro = (K_scaled / self.n_neurons) # divided by the actual number of neurons
            C_ab_micro = (K_scaled / (full_scale_n_neurons * self.area) ) # divided by the old numbers 
        else:
            raise Exception("Unexpected connection type. Use 'fixed_total_number' for microcircuit " + 
                            "model or 'fixed_indegree' for Brunel's model!")

        if C_ab == "micro":
            self.C_ab = C_ab_micro.astype(int)
        elif C_ab == "brunel":
            C_e     = np.mean(C_ab_micro) # mean for microcircuit (= 501 in full scale)
            C_i     = gamma * C_e
            self.C_ab   = np.tile([C_e, C_i], (self.n_populations, self.n_layers)).astype(int) 
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
        ###          Delays and dicts                   ###        
        ###################################################

        # mean dendritic delays for excitatory and inhibitory transmission (ms)
        delay_e = 1.5   # ms, excitatory synapses
        delay_i = 0.75  # ms, inhibitory synapses

        self.delays  = np.tile([delay_e, delay_i], (self.n_populations, self.n_layers)) # adapt...
        self.delay_rel_sd = 0.0 # standard deviation relative to mean delays
        
        # Synapse dictionaries
        # default connection dictionary
        self.conn_dict   = {"rule": connection_type}
        # weight distribution of connections between populations
        self.weight_dict_exc = {"distribution": "normal_clipped", "low": 0.0} 
        self.weight_dict_inh = {"distribution": "normal_clipped", "high": 0.0} 
        # delay distribution of connections between populations
        self.delay_dict  = {"distribution": "normal_clipped", "low": 0.1} 
        # default synapse dictionary
        self.syn_dict = {"model": "static_synapse"}
        
        ###################################################
        ###          Single-neuron parameters		###        
        ###################################################
        
        self.neuron_model = "iaf_psc_exp"   # neuron model. For PSP-to-PSC conversion to
                                            # be correct, synapses should be current-based
                                            # with an exponential time course
        self.Vm0_mean    = -58.0            # mean of initial membrane potential (mV)
        self.Vm0_std     = 10.0             # std of initial membrane potential (mV)
        
        # neuron model parameters
        # Reset voltage and threshold (set V_r to zero)
        self.model_params = {"tau_m": 10.,       # membrane time constant (ms)
        	            "tau_syn_ex": 0.5,  # excitatory synaptic time constant (ms)
                            "tau_syn_in": 0.5,  # inhibitory synaptic time constant (ms)
                            "t_ref": 2.,        # absolute refractory period (ms)
        	            "E_L": -65.,        # resting membrane potential (mV)
        	            "V_th": -50.,       # spike threshold (mV)
                            "C_m": 250.,        # membrane capacitance (pF)
        	            "V_reset": -65.     # reset potential (mV)
                            } 
        # Rescaling for model calculations: these values are not used in the simulation!
        self.t_ref  = self.model_params["t_ref"] * 1e-3          # s
        self.tau_m  = self.model_params["tau_m"] * 1e-3          # s
        self.E_L    = self.model_params["E_L"]                  # mV
        self.V_r    = self.model_params["V_reset"] - self.E_L   # mV
        self.theta  = self.model_params["V_th"] - self.E_L      # mV

        

        ###################################################
        ###          Stimuli: Thalamus and DC background ##        
        ###################################################
        # rate of background Poisson input at each external input synapse (spikes/s) 
        self.rate_ext  = rate_ext        # Hz 
        self.J_ext  = J         # external synaptic weight
        # in-degrees for background input
        self.C_aext = np.array([
                                1600,   # 2/3e
                                1500,   # 2/3i
                                2100,   # 4e
                                1900,   # 4i
                                2000,   # 5e
                                1900,   # 5i
                                2900,   # 6e
                                2100])   # 6i

        # DC amplitude at each external input synapse (pA)
        # This is relevant for reproducing Potjans & Diesmann (2012) Fig. 7.
        self.dc_amplitude = 0. 
        
        # optional additional thalamic input (Poisson)
        # Set n_th to 0 to avoid this input.
        # For producing Potjans & Diesmann (2012) Fig. 10, n_th = 902 was used.
        # Note that the thalamic rate here reproduces the simulation results
        # shown in the paper, and differs from the rate given in the text. 
        self.n_th    = 0       # size of thalamic population
        self.th_start    = 700.  # onset of thalamic input (ms)
        self.th_duration = 10.   # duration of thalamic input (ms)
        self.th_rate = 120.      # rate of thalamic neurons (spikes/s)
        self.J_th  = 0.15      # mean EPSP amplitude (mV) for thalamic input
        
        # connection probabilities for thalamic input
        conn_probs_th = np.array(
                [0.0,       # 2/3e
                 0.0,       # 2/3i    
                 0.0983,    # 4e
                 0.0619,    # 4i
                 0.0,       # 5e
                 0.0,       # 5i
                 0.0512,    # 6e
                 0.0196])   # 6i
        if self.scale_K_linearly:
            if not self.n_th == 0:
                K_th_full_scale = np.log(1. - conn_probs_th) / \
                    np.log(1. - 1. / (self.n_th * full_scale_n_neurons))
                self.C_th_scaled     = np.int_(K_th_full_scale * self.area)
        else:
            if not self.n_th == 0:
                self.C_th_scaled     = np.int_(np.log(1. - conn_probs_th) / \
                    np.log(1. - 1. / (self.n_th * self.n_neurons_micro)))
        
        # mean delay of thalamic input (ms)
        self.delay_th    = 1.5  
        # standard deviation relative to mean delay of thalamic input
        self.delay_th_rel_sd = 0.5  
        
    def get_PSCs(self):
        """Compute PSC amplitude from PSP amplitude
        These are used as weights (mean for normal_clipped distribution)
        """
        tau_m, tau_syn, C_m = \
            [self.model_params[key] for key in ["tau_m", "tau_syn_ex", "C_m"]]
        delta_tau   = tau_syn - tau_m
        ratio_tau    = tau_m / tau_syn
        PSC_over_PSP = C_m * delta_tau / (tau_m * tau_syn * \
            (ratio_tau**(tau_m / delta_tau) - ratio_tau**(tau_syn / delta_tau)))
        PSCs    = self.J_ab * PSC_over_PSP     # neuron populations
        PSC_ext = self.J_ext * PSC_over_PSP  # external poisson
        PSC_th  = self.J_th * PSC_over_PSP   # thalamus
        return PSCs, PSC_ext, PSC_th
        
