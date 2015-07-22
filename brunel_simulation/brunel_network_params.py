"""brunel_network_params.py

    Network parameters. 
    Based on network_params.sli (Potjans 2014)

    Contains:
    - network parameters
    - single-neuron parameters
    - stimulus parameters

Connection probabilities as well as mean values of synaptic weights and 
delays are implemented as numpy arrays with shape (n_pop, n_pop), where
n_pop = number of populations. Each layer contains neurons of all types. 

In order to introduce further types, all arrays have to be carefully adapted. 
However, the script running the simulation should not depend on these adaptations 
but work without further configuration (not yet tested).
"""
import numpy as np

###################################################
###     	Network parameters		###        
###################################################

layers  = np.array(["Lb"])
types = np.array(["e", "i"]) 

n_neurons = np.array( \
  [40000,   # e
   10000])    # i

# Synaptic weights
g     = 5.           # weight for inhibitory synapses
# Mean reference PSP (EPSP amplitude except for L4e->L2/3e)
PSP_e   = .1          # mv
# Standard deviation of weight  relative to mean weight
weight_rel_sd = 0.0

# Connections
# Synapse numbers for "fixed_indegree":
epsilon = 0.1
	     
# mean dendritic delays for excitatory and inhibitory transmission (ms)
delay_e = 1.5   # ms, excitatory synapses
delay_i = 1.5  # ms, inhibitory synapses
# standard deviation relative to mean delays
delay_rel_sd = 0.5 

# Synapse dictionaries
# default connection dictionary
connection_rule   = "fixed_indegree"
# weight distribution of connections between populations
weight_dict_exc = {"distribution": "normal_clipped", "low": 0.0} 
weight_dict_inh = {"distribution": "normal_clipped", "high": 0.0} 
# delay distribution of connections between populations
delay_dict  = {"distribution": "normal_clipped", "low": 0.1} 
# default synapse dictionary
syn_dict = {"model": "static_synapse"}


###################################################
###          Single-neuron parameters		###        
###################################################

neuron_model = "iaf_psc_delta"  # "iaf_psc_delta" or "iaf_psc_exp"
Vm0_mean    =  0.0             # mean of initial membrane potential (mV)
Vm0_std     = 0.1              # std of initial membrane potential (mV)

# neuron model parameters
model_params = {"tau_m": 20.,       # membrane time constant (ms)
                "t_ref": 2.,        # absolute refractory period (ms)
                "C_m":  1.,          # specific membrane capacitance (pF/ mum^2)
                "E_L": 0.,          # resting membrane potential (mV)
                "V_th": 20.,        # spike threshold (mV)
                "V_reset": 10.,      # reset potential (mV)
                "V_m":        0.0,
               } 

# Synaptic time constants are only applied if the neuron model is not iaf_psc_delta
tau_syn_ex = 0.5 # excitatory synaptic time constant (ms)
tau_syn_in = 0.5 # inhibitory synaptic time constant (ms)

###################################################
###           Stimulus parameters		###        
###################################################
 
# rate of background Poisson input at each external input synapse (spikes/s) 
rate_ext_factor    = 2.0        # in units of rate_theta = theta / (J * C_E * tau_m)
PSP_ext     = PSP_e             # mean EPSP amplitude (mV) for external input
C_aext = epsilon * np.tile(n_neurons[0], 2) # in-degrees for background input

# PREVIOUSLY USED, NOT IMPLEMENTED AT THIS POINT!
# DC amplitude at each external input synapse (pA)
dc_amplitude = 0. 
# optional additional thalamic input (Poisson)
n_th    = 0       # size of thalamic population
