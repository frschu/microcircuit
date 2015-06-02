'''user_params.sli
    
    adapt if necessary
'''
# Absolute path to which the output files should be written.
# Only used if sim_params.run_mode == 'production'
# The data is saved to 'data_dir/simulation_specifications/pynest', 
# as specified in 'functions.py'. 
#
# 'simulation_specifications' contain: 
#   a           = area in decimals of 1 mm^2 (full scale);
#   t           = simulation time in s;
#   th, dc      = whether thalamus or dc background current are connected;
#   00, 01, ... = nth experiment of this kind.
#
# Example: 'a0.1_t10.2_th_dc_00/'
data_dir    = '/users/schuessler/uni/microcircuit/trans_data/'

# Simulation to analyze.
simulation_spec = 'a0.5_t10.0_00/'

