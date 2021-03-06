"""functions.py

Contains functions:
prepare_simulation
derive_parameters
create_nodes    (all node parameters should be set only here)
connect         (synapse parameters are set here)

"""
from __future__ import print_function
import nest
import numpy as np
import os
# Import specific moduls
from imp import reload
import sim_params as sim; reload(sim)
data_path_test = os.path.join(sim.data_dir, "test", "pynest_only_seeds")
if not os.path.exists(data_path_test):
    os.makedirs(data_path_test)


#######################################################
# Functions
#######################################################
def prepare_simulation(master_seed, n_populations):
    """Prepare random generators with master seed."""
    nest.ResetKernel()
    # set global kernel parameters
    nest.SetKernelStatus(
        {"communicate_allgather": sim.allgather,
        "overwrite_files": sim.overwrite_existing_files,
        "resolution": sim.dt,
        "total_num_virtual_procs": sim.n_vp})
    if sim.to_text_file:
        nest.SetKernelStatus({"data_path": data_path_test})
   
    # Set random seeds
    
    # PYNEST
    #nest.sli_run('0 << /rngs [%i %i] Range { rngdict/gsl_mt19937 :: exch CreateRNG } Map >> SetStatus'%(
    #             master_seed, master_seed + sim.n_vp - 1))
    #nest.SetKernelStatus({"rng_seeds" : range(master_seed, master_seed + sim.n_vp)})
    #nest.sli_run('0 << /grng rngdict/gsl_mt19937 :: %i CreateRNG >> SetStatus'%(master_seed + sim.n_vp))
    #nest.SetKernelStatus({"grng_seed" : master_seed + sim.n_vp})
    #pyrngs = [np.random.RandomState(s) for s in 
    #            range(master_seed + sim.n_vp + 1, master_seed + 2 * sim.n_vp + 1)]

    # SLI VERSION
    sli_str  = "0 << \n"
    #sli_str += "/rngs %i [0 %i 1 sub] add Range { rngdict/gsl_mt19937 :: exch CreateRNG } Map\n"%(master_seed, sim.n_vp) # local RNG, seeded
    #sli_str += "/grng rngdict/gsl_mt19937 :: %i %i add CreateRNG\n"%(master_seed, sim.n_vp) # global RNG
    sli_str += "/rng_seeds %i [0 %i 1 sub] add Range\n"%(master_seed, sim.n_vp) # local RNG seeds
    sli_str += "/grng_seed %i %i add\n"%(master_seed, sim.n_vp) # global RNG seed
    sli_str += ">> SetStatus"
    nest.sli_run(sli_str)
    sli_str2  = "/script_rngs [%i]\n"%sim.n_vp
    sli_str2 += "{%i add rngdict /gsl_mt19937 get exch CreateRNG } Table def\n"%(master_seed + sim.n_vp)
    sli_str2 += "/normal_rdvs script_rngs { rdevdict /normal get CreateRDV } Map def"
    nest.sli_run(sli_str2)
    pyrngs = None
    return pyrngs


def derive_parameters(model):
    # numbers of neurons from which to record spikes and membrane potentials
    # either rate of population or simply a fixed number regardless of population size
    if sim.record_fraction_neurons_spike:
        n_neurons_rec_spike = (model.n_neurons * sim.frac_rec_spike).astype(int)
    else:
        n_neurons_rec_spike = (np.ones_like(model.n_neurons) * sim.n_rec_spike).astype(int)

    if sim.record_fraction_neurons_voltage:
        n_neurons_rec_voltage = (model.n_neurons * sim.frac_rec_voltage).astype(int)
    else:
        n_neurons_rec_voltage = (np.ones_like(model.n_neurons) * sim._rec_voltage).astype(int)

    return n_neurons_rec_spike, n_neurons_rec_voltage


def create_nodes(model, pyrngs):
    """
        Creates the following GIDs:
        neuron_GIDs
        ext_poisson
        ext_dc
        th_parrots
        th_poisson
        spike_detectors
        multimeters
        th_spike_detector
    
        Further initializes the neurons" membrane potentials.
    """
    neuron_GIDs     = []
    spike_detectors = []
    multimeters     = []
    ext_poisson     = []
    ext_dc          = []
    print(data_path_test)
    Vm0_file = open(os.path.join(data_path_test, "Vm0_pynest"), "w")
    for pop_index, population in enumerate(model.populations):
        # Neurons
        neuron_GIDs.append(nest.Create(model.neuron_model, model.n_neurons[pop_index], params=model.model_params))
        # Initialize membrane potentials locally
        # drawn from normal distribution with mu=Vm0_mean, sigma=Vm0_std
        neurons_info    = nest.GetStatus(neuron_GIDs[pop_index])
        for ni in neurons_info:                 
            if ni["local"]:                         # only adapt local nodes
                sli_str3 = "%i << /V_m normal_rdvs %i get Random %.1f mul %.1f add >> SetStatus"%(ni["global_id"], ni["vp"], model.Vm0_std, model.Vm0_mean)
                nest.sli_run(sli_str3)
                Vm_init = nest.GetStatus([ni["global_id"]])[0]["V_m"]
                #Vm_init = pyrngs[ni["vp"]].normal(model.Vm0_mean, model.Vm0_std)
                #nest.SetStatus([ni["global_id"]], {"V_m": Vm_init})
                Vm0_file.write(str(ni["global_id"]) + "\t" + str(Vm_init) + "\n")

        # Devices
        if sim.record_cortical_spikes:
            spike_detector_dict = {"label": sim.spike_detector_label + population + "_", 
                                    "to_file": sim.to_text_file}
            spike_detectors.append(nest.Create("spike_detector", 1, params=spike_detector_dict))

        if sim.record_voltage:
            multimeter_dict = {"label": sim.multimeter_label + population + "_", 
                                "to_file": sim.to_text_file, 
                                "start": sim.t_rec_volt_start,   
                                #"stop": sim.t_rec_volt_stop, 
                                "interval": 1.0, # ms
                                "withtime": True, 
                                "record_from": ["V_m"]}
            multimeters.append(nest.Create("multimeter", 1, params=multimeter_dict))
        
        # External input
        # One poisson generator per population. 
        #Rate is determined by base rate times in-degree[population]
        ext_poisson_params = {"rate": model.rate_ext * model.C_aext[pop_index]}
        ext_poisson.append(nest.Create("poisson_generator", 1, params=ext_poisson_params))
        # One dc generator per population. 
        # Amplitude is determined by base amplitude times in-degree[population]
        ext_dc_params = {"amplitude": model.dc_amplitude * model.C_aext[pop_index]}
        ext_dc.append(nest.Create("dc_generator", 1, params=ext_dc_params))
        
    # Thalamic neurons: parrot neurons and Poisson bg
    if not model.n_th == 0:
        th_parrots  = nest.Create("parrot_neuron", model.n_th, params=None)
        th_poisson  = nest.Create("poisson_generator", 1, 
            params={"rate": model.th_rate, 
                "start": model.th_start, 
                "stop": model.th_start + model.th_duration})
        if sim.record_thalamic_spikes:
            th_spike_detector_dict = {"label": sim.th_spike_detector_label, 
                                    "to_file": sim.to_text_file}
            th_spike_detector  = nest.Create("spike_detector", 1, params=th_spike_detector_dict)
        else:
            th_spike_detector = None
    else:
        th_parrots, th_poisson, th_spike_detector = (None, None, None)


    Vm0_file.close()
        
    return (neuron_GIDs, 
            spike_detectors, multimeters,
            ext_poisson, ext_dc, 
            th_parrots, th_poisson, th_spike_detector)

def connect(model, all_GIDs, 
            n_neurons_rec_spike, n_neurons_rec_voltage,
            verbose):
    (neuron_GIDs, 
            spike_detectors, multimeters,
            ext_poisson, ext_dc, 
            th_parrots, th_poisson, th_spike_detector) = all_GIDs
    
    # Connect target populations...
    for target_index, target_pop in enumerate(model.populations):
        if verbose: print("Connecting target " + target_pop)
        if verbose: print("with source")
        target_GIDs = neuron_GIDs[target_index]    # transform indices to GIDs of target population
    
        # ...to source populations
        for source_index, source_pop in enumerate(model.populations):
            source_GIDs = neuron_GIDs[source_index] # transform indices to GIDs of source population
            n_synapses  = model.C_ab[target_index, source_index]  # connection probability
            if not n_synapses == 0:
                if verbose: print("\t" + source_pop)
    
                conn_dict       = model.conn_dict.copy()
                if model.connection_rule == "fixed_total_number":
                    conn_dict["N"]  = n_synapses
                elif model.connection_rule == "fixed_indegree":
                    conn_dict["indegree"]  = n_synapses
    
                mean_weight             = model.PSCs[target_index, source_index]
                std_weight              = abs(mean_weight * model.PSC_rel_sd)
                if mean_weight >= 0:
                    weight_dict = model.weight_dict_exc.copy()
                else:
                    weight_dict = model.weight_dict_inh.copy()
                weight_dict["mu"]       = mean_weight
                weight_dict["sigma"]    = std_weight
    
                mean_delay              = model.delays[target_index, source_index]
                std_delay               = mean_delay * model.delay_rel_sd 
                delay_dict              = model.delay_dict.copy()
                delay_dict["mu"]        = mean_delay
                delay_dict["sigma"]     = std_delay
    
                syn_dict                = model.syn_dict.copy()
                syn_dict["weight"]      = weight_dict
                syn_dict["delay"]       = delay_dict
    
                nest.Connect(source_GIDs, target_GIDs, conn_dict, syn_dict)
        
        # ...to thalamic population
        if not model.n_th == 0:
            n_synapses_th   = model.C_th_scaled[target_index]
            if not n_synapses_th == 0:
                if verbose: print("\tthalamus")
                conn_dict_th        = model.conn_dict.copy()
                conn_dict_th["N"]   = n_synapses_th
                
                mean_weight_th      = model.PSC_th
                std_weight_th       = mean_weight_th * model.PSC_rel_sd
                weight_dict_th      = model.weight_dict_exc.copy()
                weight_dict_th["mu"]    = mean_weight_th
                weight_dict_th["sigma"] = std_weight_th
    
                mean_delay_th       = model.delay_th
                std_delay_th        = mean_delay_th * model.delay_th_rel_sd 
                delay_dict_th       = model.delay_dict.copy()
                delay_dict_th["mu"]     = mean_delay_th
                delay_dict_th["sigma"]  = std_delay_th
    
                syn_dict_th             = model.syn_dict.copy()
                syn_dict_th["weight"]   = weight_dict_th
                syn_dict_th["delay"]    = delay_dict_th
    
                nest.Connect(th_parrots, target_GIDs, conn_dict_th, syn_dict_th)
    
        # ...to spike detector
        if sim.record_cortical_spikes:
            if verbose: print("\tspike detector")
            # Choose only a fixed fraction/number of neurons to record spikes from
            rec_spike_GIDs = target_GIDs[:n_neurons_rec_spike[target_index]]
            nest.Connect(list(rec_spike_GIDs), spike_detectors[target_index], "all_to_all")
    
        # ...to multimeter
        if sim.record_voltage:
            if verbose: print("\tmultimeter")
            # Choose only a fixed fraction/number of neurons to record membrane voltage from
            rec_voltage_GIDs = target_GIDs[:n_neurons_rec_voltage[target_index]]
            nest.Connect(multimeters[target_index], list(rec_voltage_GIDs), "all_to_all")
     
        # ...to background
        if not model.rate_ext == 0:
            if verbose: print("\tpoisson background")
            nest.Connect(ext_poisson[target_index], target_GIDs, 
                conn_spec={"rule": "all_to_all"}, 
                syn_spec={"weight": model.PSC_ext, "delay": model.delay_ext}) 
        if not model.dc_amplitude == 0:
            if verbose: print("\tDC background" )
            nest.Connect(ext_dc[target_index], target_GIDs, "all_to_all")
    
    # Connect Thalamic population, if existing.
    if not model.n_th == 0:
        if verbose: print("Connect thalamus: poisson to parrots")
        nest.Connect(th_poisson, th_parrots, "all_to_all")
        if sim.record_thalamic_spikes:
            if verbose: print("Connect thalamus to th_spike_detector")
            nest.Connect(th_parrots, th_spike_detector, "all_to_all")
