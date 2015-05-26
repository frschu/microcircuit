'''
    user_params.sli
    
    adapt if necessary
'''
import os

# Import specific moduls
import network_params as net
import sim_params as sim

# absolute path to which the output files should be written
# only used if sim_params.run_mode == 'production'
data_dir    = '/users/schuessler/uni/microcircuit/data/'
test_path   = data_dir + 'test'

# path to the mpi shell script
# can be left out if set beforehand
mpi = '/path_to_mpi_script/my_mpi_script.sh'

# path to NEST
nest_path = '/path_to_nest_install_folder/bin/nest'
