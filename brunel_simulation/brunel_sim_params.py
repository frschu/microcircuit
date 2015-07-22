"""brunel_sim_params.py

    Contains:
    - data_path
    - simulation parameters
    - recording parameters
"""

###################################################
###     	Data Path		                    ###        
###################################################

data_dir    = "/users/schuessler/uni/microcircuit/data"
log_path    = "/users/schuessler/uni/microcircuit/simulation"

###################################################
###     	Simulation parameters		###        
###################################################

t_trans = 0.2e3     # ms; transitional period in order to reach equilibrium
t_measure = 5.0e3   # ms; time actually measured
t_sim = t_measure + t_trans    # ms; simulated time 
dt = 0.1            # ms; simulation step; default is 0.1 ms. (resolution of kernel)
allgather = True    # communication protocol

# master seed for random number generators
# actual seeds will be master_seed ... master_seed + 2*n_vp
#  ==>> different master seeds must be spaced by at least 2*n_vp + 1
# see Gewaltig et al. '2012' for details       
master_seed = 0    # changes rng_seeds and grng_seed

n_vp = 8 	# number of virtual processes

###################################################
###     	Recording parameters		###        
################################################### 

# Recordables: Cortical spikes and voltages, thalamic spikes
# For cortical spikes and voltages, you can choose to measured at all
# and, if so, whether to measure a fixed fraction or a fixed number of each population
# For recording all cortical neurons, set frac_rec_spike == 1.

# Cortical spikes
record_cortical_spikes = True
record_fraction_neurons_spike = False
frac_rec_spike = 0.1
n_rec_spike = 1000

# Cortical voltages
record_voltage = False
record_fraction_neurons_voltage = False
frac_rec_voltage = 0.02 
n_rec_voltage = 100 
t_rec_volt_start  = t_trans         # ms; time up to which voltages are recorded
t_rec_volt_stop   = 1e3 + t_trans   # ms; time up to which voltages are recorded

