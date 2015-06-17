"""sim_params.py

    Contains:
    - simulation parameters
    - recording parameters
"""

###################################################
###     	Simulation parameters		###        
###################################################

t_trans = 0.2e3     # ms; transitional period in order to reach equilibrium
t_measure = 20.e3   # ms; time actually measured
t_sim = t_measure + t_trans    # ms; simulated time 
dt = 0.1            # ms; simulation step; default is 0.1 ms. (resolution of kernel)
allgather = True    # communication protocol

# master seed for random number generators
# actual seeds will be master_seed ... master_seed + 2*n_vp
#  ==>> different master seeds must be spaced by at least 2*n_vp + 1
# see Gewaltig et al. '2012' for details       
master_seed = 000000    # changes rng_seeds and grng_seed

n_mpi_procs = 1         # number of MPI processes

n_threads_per_proc = 8 	# number of threads per MPI process
                            # use for instance 24 for a full-scale simulation


n_vp = int(n_threads_per_proc * n_mpi_procs)# number of virtual processes
                                            # This should be an integer multiple of 
                                            # the number of MPI processes 
                                            # See Morrison et al. '2005' Neural Comput
walltime = '8:0:0' 		# walltime for simulation

memory = '500mb' 	    # total memory
                            # use for instance 4gb for a full-scale simulation

###################################################
###     	Recording parameters		###        
################################################### 

# Recordables: Cortical spikes and voltages, thalamic spikes
# For cortical spikes and voltages, you can choose to measured at all
# and, if so, whether to measure a fixed fraction or a fixed number of each population
# For recording all cortical neurons, set frac_rec_spike == 1.

# Whether to create a new directory to save the data. If True, no such directory is 
# created and the data in the corresponding existing file is overwritten. 
# Data path is described in user_params.py.
overwrite_existing_files = True

# Cortical spikes
record_cortical_spikes = True 
record_fraction_neurons_spike = True 
frac_rec_spike = 0.1
n_rec_spike = 100 

# Cortical voltages
record_voltage = True
record_fraction_neurons_voltage = True
frac_rec_voltage = 0.02 
n_rec_voltage = 20 
t_rec_volt_start  = 0.0 # time up to which voltages are recorded
t_rec_volt_stop   = 1e3 + t_trans # time up to which voltages are recorded

# Thalamic spikes
# Relevant only if network_params.n_thal > 0
record_thalamic_spikes = True 

# stem for spike detector file labels
spike_detector_label = 'spikes_' 

# stem for multimeter file labels
multimeter_label = 'voltages_' 

# stem for thalamic spike detector file labels
th_spike_detector_label = 'th_spikes_' 

# file name for standard output
std_out = 'output.txt' 

# file name for error output
error_out = 'errors.txt' 
