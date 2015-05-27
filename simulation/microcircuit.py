'''
    microcircuit.py

Naming convention: layer (e.g. L4), type (usually e and i), population (e.g. L4e)

Includes:
check_parameters
prepare_simulation
derive_parameters
create_nodes    (all node parameters should be set only here)
connect_nodes         (synapse parameters are set here)
simulate
'''
from __future__ import print_function
from imp import reload
import nest
import numpy as np
from itertools import chain
import os, time
import logging  # Write warnings to 'microcircuit.log'
logging.basicConfig(filename='errors.log', filemode='w', level=logging.DEBUG)

# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import user_params as user; reload(user)
import functions
######################################################


######################################################
# Check parameters 
######################################################

# Not exhaustive.
if net.neuron_model != 'iaf_psc_exp':
    logging.warning('Unexpected neuron type: script is tuned to \'iaf_psc_exp\' neurons.')
# Check whether there are only the neuron types 'e' and 'i'
# If you want to adapt, you further need to adapt sections in this script:
    # derive_parameters: PSP to PSC
    # connect_nodes: weight dictionaries, if-statement 
if len(net.types) != 2:
    raise Exception('Unexpected neuron types: script is tuned to (\'e\', \'i\')-neurons')


######################################################
# Prepare simulation
######################################################

# Set data path
if sim.run_mode == 'test':
    data_path = user.test_path
elif sim.run_mode == 'production':
    data_path = functions.get_output_path(net.area, sim.t_sim, net.n_th, net.dc_amplitude, 
                    user.data_dir, sim.overwrite_existing_files, n_digits=2)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
else: 
    raise Exception('Unexpected sim_params.run_mode: expects \'test\' or \'production\'')

nest.ResetKernel()
# set global kernel parameters
nest.SetKernelStatus(
    {'communicate_allgather': sim.allgather,
    'overwrite_files': True,
    'resolution': sim.dt,
    'total_num_virtual_procs': sim.n_vp})
if sim.run_mode:
    nest.SetKernelStatus({'data_path': data_path})

# Set random seeds
nest.SetKernelStatus({'grng_seed' : sim.master_seed})
nest.SetKernelStatus({'rng_seeds' : range(sim.master_seed + 1, sim.master_seed + sim.n_vp + 1)})
pyrngs = [np.random.RandomState(s) for s in 
            range(sim.master_seed + sim.n_vp + 1, sim.master_seed + 2 * sim.n_vp + 1)]
pyrngs_rec_spike = [np.random.RandomState(s) for s in 
            range(sim.master_seed + 2 * sim.n_vp + 1, 
                  sim.master_seed + 2 * sim.n_vp + 1 + len(net.populations))]
pyrngs_rec_voltage = [np.random.RandomState(s) for s in 
            range(sim.master_seed + 2 * sim.n_vp + 1 + len(net.populations), 
                  sim.master_seed + 2 * sim.n_vp + 1 + 2 * len(net.populations))]
np.save(data_path + 'seed_numbers.npy', 
            np.array([sim.master_seed, 
                      sim.master_seed + 2 * sim.n_vp + 1 + 2 * len(net.populations)]))

######################################################
# Derive parameters
######################################################

# Scale size of network
n_neurons       = np.rint(net.full_scale_n_neurons * net.area).astype(int)
n_populations   = len(net.populations)
n_layers        = len(net.layers)
n_types         = len(net.types)
n_total         = np.sum(n_neurons)
matrix_shape    = np.shape(net.conn_probs)  # shape of connection probability matrix

# Scale synapse numbers
if net.scale_K_linearly:
    n_outer_full    = np.outer(net.full_scale_n_neurons, net.full_scale_n_neurons)
    K_full_scale    = np.log(1. - net.conn_probs   ) / np.log(1. - 1. / n_outer_full)
    K_scaled        = np.int_(K_full_scale * net.area)
    if not net.n_th == 0:
        K_th_full_scale = np.log(1. - net.conn_probs_th) / \
            np.log(1. - 1. / (net.n_th * net.full_scale_n_neurons))
        K_th_scaled     = np.int_(K_th_full_scale * net.area)
else:
    n_outer         = np.outer(n_neurons, n_neurons)
    K_scaled        = np.int_(np.log(1. - net.conn_probs   ) / np.log(1. - 1. / n_outer))
    if not net.n_th == 0:
        K_th_scaled = np.int_(np.log(1. - net.conn_probs_th) / \
            np.log(1. - 1. / (net.n_th * n_neurons)))


# numbers of neurons from which to record spikes and membrane potentials
# either rate of population or simply a fixed number regardless of population size
if sim.record_fraction_neurons_spike:
    n_neurons_rec_spike = np.rint(n_neurons * sim.frac_rec_spike).astype(int)
else:
    n_neurons_rec_spike = (np.ones_like(n_neurons) * n_rec_spike).astype(int)
np.save(data_path + 'n_neurons_rec_spike.npy', n_neurons_rec_spike)

if sim.record_fraction_neurons_voltage:
    n_neurons_rec_voltage = np.rint(n_neurons * sim.frac_rec_voltage).astype(int)
else:
    n_neurons_rec_voltage = (np.ones_like(n_neurons) * n_rec_voltage).astype(int)
np.save(data_path + 'n_neurons_rec_voltge.npy', n_neurons_rec_voltage)


# Compute PSC amplitude from PSP amplitude
# These are used as weights (mean for normal_clipped distribution)
def PSC_over_PSP():
    '''Calculates factor for transformation from PSP to PSC'''
    tau_m, tau_syn_ex, tau_syn_in, C_m = \
        [net.model_params[key] for key in ['tau_m', 'tau_syn_ex', 'tau_syn_in', 'C_m']]
    delta_tau   = tau_syn - tau_m
    ratio_tau    = tau_m / tau_syn
    PSC_over_PSP = C_m * delta_tau / (tau_m * tau_syn * \
        (ratio_tau**(tau_m / delta_tau) - ratio_tau**(tau_syn / delta_tau)))
    return PSC_over_PSP
PSCs    = net.PSPs * PSC_over_PSP()     # neuron populations
PSC_ext = net.PSP_ext * PSC_over_PSP()  # external poisson
PSC_th  = net.PSP_th * PSC_over_PSP()   # thalamus

######################################################
# Create nodes
######################################################
'''
    Creates the following GIDs:
    neuron_GIDs
    ext_poisson
    ext_dc
    th_GIDs
    th_poisson
    spike_detectors
    multimeters
    th_spike_detector

    Further initializes the neurons' membrane potentials.
'''
print('Create nodes')

# Neurons
neurons     = nest.Create(net.neuron_model, n_total, params=net.model_params)
# Initialize membrane potentials locally
# drawn from normal distribution with mu=Vm0_mean, sigma=Vm0_std
neurons_info    = nest.GetStatus(neurons)
for ni in neurons_info:                 
    if ni['local']:                         # only adapt local nodes
        nest.SetStatus([ni['global_id']], 
            {'V_m': pyrngs[ni['vp']].normal(net.Vm0_mean, net.Vm0_std)})
# GIDs for neurons on subnets
GID0        = neurons[0]
upper_GIDs  = GID0 + np.cumsum(n_neurons) - 1
lower_GIDs  = np.insert((upper_GIDs[:-1] + 1), 0, GID0) 
neuron_GIDs = np.array([range(*boundary) for boundary in zip(lower_GIDs, upper_GIDs)])

# External input
# One poisson generator per population. 
#Rate is determined by base rate times in-degree[population]
ext_poisson_params = [{'rate': net.bg_rate * in_degree} for in_degree in net.K_bg]
ext_poisson = nest.Create('poisson_generator', n_populations, 
    params=ext_poisson_params) 
# One dc generator per population. 
# Amplitude is determined by base amplitude times in-degree[population]
ext_dc_params = [{'amplitude': net.dc_amplitude * in_degree} for in_degree in net.K_bg]
ext_dc = nest.Create('dc_generator', n_populations, 
    params=ext_dc_params) 

# Thalamic neurons: parrot neurons and Poisson bg
if not net.n_th == 0:
    th_GIDs     = nest.Create('parrot_neuron', net.n_th, params=None)
    th_poisson  = nest.Create('poisson_generator', 1, 
        params={'rate': net.th_rate, 
            'start': net.th_start, 
            'stop': net.th_start + net.th_duration})

# Devices
if sim.record_cortical_spikes:
    spike_detector_dict = [{'label': sim.spike_detector_label + population + '_', 
                            'to_file': True} for population in net.populations]
    spike_detectors = nest.Create('spike_detector', n_populations, params=spike_detector_dict)
if sim.record_voltage:
    multimeter_dict = [{'label': sim.multimeter_label + population + '_', 
                        'to_file': True, 
                        'record_from': ['V_m']} for population in net.populations]
    multimeters     = nest.Create('multimeter', n_populations, params=multimeter_dict)
if sim.record_thalamic_spikes:
    th_spike_detector_dict = {'label': sim.th_spike_detector_label, 
                            'to_file': True}
    th_spike_detector  = nest.Create('spike_detector', 1, params=th_spike_detector_dict)


###################################################
# Connect
###################################################
t_connect_0 = time.time()

# Preparation: Thalamic population, if existing.
if not net.n_th == 0:
    print('\nConnect thalamus: poisson to parrots')
    nest.Connect(th_poisson, th_GIDs, 'all_to_all')
    if sim.record_thalamic_spikes:
        print('Connect thalamus to th_spike_detector')
        nest.Connect(th_GIDs, th_spike_detector, 'all_to_all')

# Connect target populations...
for target_index, target_pop in enumerate(net.populations):
    print('\nConnecting target ' + target_pop)
    print('with source')
    target_GIDs = neuron_GIDs[target_index]    # transform indices to GIDs of target population

    # ...to source populations
    for source_index, source_pop in enumerate(net.populations):
        source_GIDs = neuron_GIDs[source_index]       # transform indices to GIDs of source population
        n_synapses  = K_scaled[target_index, source_index]  # connection probability
        if not n_synapses == 0:
            print('\t' + source_pop)

            conn_dict       = net.conn_dict.copy()
            conn_dict['N']  = n_synapses

            mean_weight             = PSCs[target_index, source_index]
            std_weight              = abs(mean_weight * net.PSC_rel_sd)
            if mean_weight >= 0:
                weight_dict = net.weight_dict_exc.copy()
            else:
                weight_dict = net.weight_dict_inh.copy()
            weight_dict['mu']       = mean_weight
            weight_dict['sigma']    = std_weight

            mean_delay              = delays[target_index, source_index]
            std_delay               = mean_delay * net.delay_rel_sd 
            delay_dict              = net.delay_dict.copy()
            delay_dict['mu']        = mean_delay
            delay_dict['sigma']     = std_delay

            syn_dict                = net.syn_dict.copy()
            syn_dict['weight']      = weight_dict
            syn_dict['delay']       = delay_dict

            nest.Connect(source_GIDs, target_GIDs, conn_dict, syn_dict)
    
    # ...to background
    if not net.bg_rate == 0:
        print('\tpoisson background')
        nest.Connect([ext_poisson[target_index]], target_GIDs, 
            conn_spec={'rule': 'all_to_all'}, 
            syn_spec={'weight': PSC_ext}) # global delay is unnecessary
    if not net.dc_amplitude == 0:
        print('\tDC background' )
        nest.Connect([ext_dc[target_index]], target_GIDs, 'all_to_all')

    # ...to thalamic population
    if not net.n_th == 0:
        n_synapses_th   = K_th_scaled[target_index]
        if not n_synapses_th == 0:
            print('\tthalamus')
            conn_dict_th        = net.conn_dict.copy()
            conn_dict_th['N']   = n_synapses_th
            
            mean_weight_th      = PSC_th
            std_weight_th       = mean_weight_th * net.PSC_rel_sd
            weight_dict_th      = net.weight_dict_exc.copy()
            weight_dict_th['mu']    = mean_weight_th
            weight_dict_th['sigma'] = std_weight_th

            mean_delay_th       = net.delay_th
            std_delay_th        = mean_delay_th * net.delay_th_rel_sd 
            delay_dict_th       = net.delay_dict.copy()
            delay_dict_th['mu']     = mean_delay_th
            delay_dict_th['sigma']  = std_delay_th

            syn_dict_th             = net.syn_dict.copy()
            syn_dict_th['weight']   = weight_dict_th
            syn_dict_th['delay']    = delay_dict_th

            nest.Connect(th_GIDs, target_GIDs, conn_dict_th, syn_dict_th)

    # ...to spike detector
    if sim.record_cortical_spikes:
        print('\tspike detector')
        # Choose only a fixed fraction/number of neurons to record spikes from
        if sim.rand_rec_spike:
            rec_spike_GIDs = np.sort(pyrngs_rec_spike[i].choice(target_GIDs,
                n_neurons_rec_spike[target_index], replace=False))
        else:
            rec_spike_GIDs = target_GIDs[:n_neurons_rec_spike[target_index]]
        nest.Connect(list(rec_spike_GIDs), [spike_detectors[target_index]], 'all_to_all')
        np.save(data_path + 'rec_spike_GIDs_' + target_pop + '.npy', rec_spike_GIDs)

    # ...to multimeter
    if sim.record_voltage:
        print('\tmultimeter')
        # Choose only a fixed fraction/number of neurons to record membrane voltage from
        if sim.rand_rec_voltage:
            rec_voltage_GIDs = np.sort(pyrngs_rec_voltage[i].choice(target_GIDs,
                n_neurons_rec_voltage[target_index], replace=False))
        else:
            rec_voltage_GIDs = target_GIDs[:n_neurons_rec_voltage[target_index]]
        nest.Connect([multimeters[target_index]], list(rec_voltage_GIDs), 'all_to_all')
        np.save(data_path + 'rec_voltage_GIDs_' + target_pop + '.npy', rec_voltage_GIDs)

t_connect = time.time() - t_connect_0

###################################################
# Simulate
###################################################
t_simulate_0 = time.time()
nest.Simulate(sim.t_sim)
t_simulate = time.time() - t_simulate_0
print('Simulation of network with area = %.2f mm for %.1f ms took:'%(net.area, sim.t_sim))
print('Time for connecting nodes: %.2f s'%t_connect)
print('Time for simulating: %.2f s'%t_simulate)
