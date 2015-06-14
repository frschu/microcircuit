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
data_dir    = '/users/schuessler/uni/microcircuit/data/micro'
test_path   = data_dir + 'test'

# Simulation to analyze.
simulation_spec = 'a1.0_t20.0_00'


# path to the mpi shell script
# can be left out if set beforehand
mpi = '/path_to_mpi_script/my_mpi_script.sh'

# path to NEST
nest_path = '/path_to_nest_install_folder/bin/nest'
