"""model_class.py

    Class of the network model on a range between Brunel A and 
    Potjans' microcircuit model. 

    Default is Potjans' microcircuit model, parameters are taken from network_params.py
"""
from imp import reload
import numpy as np
import network_params as net; reload(net)

class model:
    def __init__(self, 
                 area=net.area,                             # simulation size
                 neuron_model=net.neuron_model,               # "iaf_psc_delta" or "iaf_psc_exp"
                 n_neurons="micro", C_ab="micro",           # else: "brunel" or arrays
                 connection_rule=net.connection_rule,      # "fixed_total_number" or "fixed_indegree"
                 j02=net.j02, 
                 weight_rel_sd=net.weight_rel_sd, 
                 delay_rel_sd=net.delay_rel_sd,  
                 g=net.g, rate_ext=net.rate_ext):                       # set to zero for Brunel

        """Class of network parameters.

        Contains:
        - network parameters
        - single-neuron parameters
        - stimulus parameters
        
        Specify area, neuron_model, neuron_numbers, C_ab, 
            j02, connection_rule, weight_rel_sd, g, rate_ext_factor.
        Default values correspond to Potjans' model.

        Neuron numbers and synapse numbers (C_ab) can be specified separately.
        Both except either a string in {'micro', 'brunel'} or an array of 
        the corresponding shape.

        Note that V_r, theta, tau_m and rate_ext are already changed to values of 
        the microcircuit model, thus not the original values of the Brunel model.
        
        Brunel's model:
        j02     = 1.0
        connection_rule = fixed_indegree
        weight_rel_sd      = 0.0
        g, rate_ext varied accordingly

        Naming: C_ab, C_aext, J_ab, J_ext, rate_ext
        Don't use conn_probs, K_bg, PSPs, PSP_ext, v_ext, any more!
        """
        ###################################################
        ###     	Network parameters		###        
        ###################################################

        # area of network in mm^2; scales numbers of neurons
        # use 1 for the full-size network (77,169 neurons)
        self.area    = area
        
        self.layers         = net.layers    #np.array(["L23", "L4", "L5", "L6"])
        self.types          = net.types     #np.array(["e", "i"]) 
        self.populations    = np.array([layer + typus for layer in self.layers for typus in self.types])
        self.n_populations  = len(self.populations)
        self.n_layers       = len(self.layers)
        self.n_types        = len(self.types)
        
        # Neuron numbers
        if n_neurons == "micro":
            self.n_neurons  = np.int_(net.full_scale_n_neurons * self.area)
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
        self.n_total    = np.sum(self.n_neurons)

        
        # Synapse numbers
        # Connection probabilities: conn_probs[post, pre] = conn_probs[target, source]
        conn_probs = net.conn_probs
        # Scale synapse numbers of the C_ab
        if net.scale_C_linearly:
            n_outer_full    = np.outer(net.full_scale_n_neurons, net.full_scale_n_neurons)
            C_full_scale    = np.log(1. - conn_probs) / np.log(1. - 1. / n_outer_full)
            C_scaled        = np.int_(C_full_scale * self.area)
        else:
            n_outer         = np.outer(self.n_neurons, self.n_neurons)
            C_scaled        = np.int_(np.log(1. - conn_probs) / np.log(1. - 1. / n_outer))

        self.connection_rule = connection_rule
        if self.connection_rule == "fixed_total_number":
            C_ab_micro = C_scaled   # total number, do not divide! 
        elif self.connection_rule == "fixed_indegree":
            C_ab_micro = (C_scaled.T / (net.full_scale_n_neurons * self.area)).T
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


        ######################################################
        # Synaptic weights. Depend on neuron_model!         ##
        ######################################################
        self.g      = g
        self.j02    = j02

        g_all       = np.tile([1., -self.g], (self.n_populations, self.n_layers))
        L23e_index  = np.where(self.populations == "L23e")[0][0]
        L4e_index   = np.where(self.populations == "L4e")[0][0]
        g_all[L23e_index, L4e_index] *= self.j02
        
        self.J           = net.PSP_e           # mv; mean EPSP, used as reference PSP
        self.J_ab   = self.J * g_all
        self.weight_rel_sd = weight_rel_sd # Standard deviation of weight relative to mean weight
        if neuron_model=="iaf_psc_delta":
            self.weights    = self.J_ab     # neuron populations
        else:
            # PSCs calculated from PSP amplitudes
            tau_m, tau_syn, C_m = \
                [self.model_params[key] for key in ["tau_m", "tau_syn_ex", "C_m"]]
            delta_tau   = tau_syn - tau_m
            ratio_tau    = tau_m / tau_syn
            PSC_over_PSP = C_m * delta_tau / (tau_m * tau_syn * \
                (ratio_tau**(tau_m / delta_tau) - ratio_tau**(tau_syn / delta_tau)))
            self.weights    = self.J_ab  * PSC_over_PSP     # neuron populations


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
        ###          Single-neuron parameters		###        
        ###################################################
        self.neuron_model   = net.neuron_model
        self.Vm0_mean       = net.Vm0_mean            # mean of initial membrane potential (mV)
        self.Vm0_std        = net.Vm0_std            # std of initial membrane potential (mV)
        self.model_params   = net.model_params
        # Rescaling for model calculations: these values are not used in the simulation!
        self.tau_m  = self.model_params["tau_m"] * 1e-3          # s
        self.t_ref  = self.model_params["t_ref"] * 1e-3          # s
        self.E_L    = self.model_params["E_L"]                  # mV
        self.V_r    = self.model_params["V_reset"] - self.E_L   # mV
        self.theta  = self.model_params["V_th"] - self.E_L      # mV


        ###################################################
        ###          External stimuli                    ##        
        ###################################################
        # rate of background Poisson input at each external input synapse (spikes/s) 
        self.rate_ext   = rate_ext      # Hz 
        self.J_ext      = net.PSP_ext   # external synaptic weight
        self.delay_ext  = self.delay_e  # ms;  mean delay of external input
        self.dc_amplitude = net.dc_amplitude  # constant bg amplitude
        self.C_aext     = net.C_aext        # in-degrees for background input
        if neuron_model=="iaf_psc_delta":
            self.weight_ext = self.J_ext    # external poisson
        else:
            self.weight_ext = self.J_ext * PSC_over_PSP  # external poisson

        # optional additional thalamic input (Poisson)
        self.n_th           = net.n_th      # size of thalamic population
        self.th_start       = net.th_start  # onset of thalamic input (ms)
        self.th_duration    = net.th_duration   # duration of thalamic input (ms)
        self.th_rate        = net.th_rate      # rate of thalamic neurons (spikes/s)
        self.J_th           = net.PSP_th      # mean EPSP amplitude (mV) for thalamic input
        if neuron_model=="iaf_psc_delta":
            self.weight_th  = self.J_th     # thalamus
        else:
            self.weight_th  = self.J_th  * PSC_over_PSP   # thalamus
        
        # connection probabilities for thalamic input
        conn_probs_th = net.conn_probs_th
        if net.scale_C_linearly:
            if not self.n_th == 0:
                C_th_full_scale = np.log(1. - conn_probs_th) / \
                    np.log(1. - 1. / (self.n_th * net.full_scale_n_neurons))
                self.C_th_scaled     = np.int_(C_th_full_scale * self.area)
        else:
            if not self.n_th == 0:
                self.C_th_scaled     = np.int_(np.log(1. - conn_probs_th) / \
                    np.log(1. - 1. / (self.n_th * self.n_neurons_micro)))
        if self.n_th == 0:
            self.C_th_scaled = None
        
        # mean delay of thalamic input (ms)
        self.delay_th    = net.delay_th
        # standard deviation relative to mean delay of thalamic input
        self.delay_th_rel_sd = net.delay_th_rel_sd


        ######################################################
        # Predefine matrices for mean field                 ##
        ######################################################
        self.mu_ext     = self.J_ext    * self.C_aext * self.rate_ext
        self.var_ext    = self.J_ext**2 * self.C_aext * self.rate_ext
        self.mat1       = self.C_ab * self.J_ab
        self.mat2       = self.C_ab * self.J_ab**2
        self.jac_mat1   = np.pi * self.tau_m**2 * self.mat1.T
        self.jac_mat2   = np.pi * self.tau_m**2 * 0.5 * self.mat2.T
        

    ######################################################
    # Methods                                           ##
    ######################################################
    def mu(self, v):
        """Mean input in Brunel's model"""
        return self.tau_m * (np.dot(self.mat1, v) + self.mu_ext)

    def sd(self, v):
        """Fluctuation of input in Brunel's model"""
        return np.sqrt(self.tau_m * (np.dot(self.mat2, v) + self.var_ext))

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

    def root_v0_siegert(self, v):
        """The integral equations to be solved
        Returns the array 'root', each entry corresponding to one population.
        Solve for root == 0.
        
        This is a different version. Might be more stable numerically.
        """
        from scipy.integrate import quad
        from scipy.special import erf
        def enum(arr1, *args):
            i_range = range(len(arr1))
            return zip(i_range, arr1 ,*args)

        max_err = 1e-16
        def integral1(mu, sigma, y_r, y_theta):
            """Integral for mu < theta"""
            def integrand(u):
                if u == 0:
                    return np.exp(-y_theta**2) * 2 * (y_theta - y_r)
                else:
                    return np.exp(-(u - y_theta)**2) * (1.0 - np.exp(2 * (y_r - y_theta) * u)) / u
        
            lower_bound = y_theta
            err_dn = 1.
            while err_dn > max_err and lower_bound > 1e-16:
                err_dn = integrand(lower_bound)
                if err_dn > max_err:            
                    lower_bound /= 2
        
            upper_bound = y_theta
            err_up = 1.
            while err_up > max_err:
               err_up = integrand(upper_bound)
               if err_up > max_err:
                   upper_bound *= 2
        
            return np.exp(y_theta**2) * quad(integrand, lower_bound, upper_bound)[0] 
         
        def integral2(mu, sigma, y_r, y_theta):
            """Integral for mu > theta"""
            def integrand(u):
                if u == 0:
                    return 2 * (y_theta - y_r)
                else:
                    return (np.exp(2 * y_theta * u - u**2) - np.exp(2 * y_r * u - u**2)) / u
        
            upper_bound = 1.0
            err = 1.0
            while err > max_err:
                err = integrand(upper_bound)
                upper_bound *= 2
        
            return quad(integrand, 0.0, upper_bound)[0] 
       
        mus         = self.tau_m * (np.dot(self.mat1, v) + self.mu_ext)
        sigmas      = np.sqrt(self.tau_m * (np.dot(self.mat2, v) + self.var_ext))
        y_rs        = (self.V_r - mus) / sigmas
        y_thetas    = (self.theta - mus) / sigmas
        
        integrals    = np.zeros(len(v))
        for i, mu, sigma, y_r, y_theta in enum(mus, sigmas, y_rs, y_thetas):
            if mu <= self.theta * 0.95:
                integrals[i] = integral1(mu, sigma, y_r, y_theta)
            else:
                integrals[i] = integral2(mu, sigma, y_r, y_theta)

        root        = - 1. / v + self.t_ref + self.tau_m * integrals
        return root


    def jacobian(self, v):
        """The Jacobian of root_v0.

        NEEDS TO BE REVIEWED FIRST            

        Used to ease the process of solving.
        The calculations are done transposed to avoid unnecessary transposes (adding axes to mu and sd)
        """
        raise Exception("REVIEW FIRST")
        mu_v  = self.mu(v)
        sd_v  = self.sd(v)
        low = (self.V_r - mu_v) / sd_v
        up  = (self.theta - mu_v) / sd_v
        f_low   = self.integrand(low)
        f_up    = self.integrand(up)
        jac_T = np.diag(v) - \
            (self.jac_mat1 * (f_up - f_low) + self.jac_mat2 * (up * f_up - low * f_low) / sd_v**2)
        return jac_T.T

    def prob_V(self, V_array, mu, sd, v):
        """Membrane potential probability distribution P(V_m) according to Brunel"""
        from scipy.integrate import quad
        step        = lambda x: 0.5 * (np.sign(x) + 1)  # Heaviside step function
        red         = lambda V: (V - mu) / sd           # reduced voltage
        P_integrand = lambda u: step(u - red(self.V_r)) * np.exp(u**2) # integrand
        
        low = red(V_r)
        up  = (self.theta - mu) / sd
        integral    = quad(P_integrand, low, up)[0]
        
        P_V_array = 2 * v * self.tau_m / sd * np.exp(- ((V_array - self.E_L) - mu)**2 / sd**2) * integral
        return step(-(V_array - self.E_L) + self.theta) * P_V_array
