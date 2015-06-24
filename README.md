# microcircuit
Pynest implementation of a layered model of the neocortical microcircuit.

The implementation is based on a script written by the 
'microcircuit model example (Potjans&Diesmann, doi:10.1093/cercor/bhs358)' 
found in the current nest version. 

## Paramters/
Adapt parameters in 
  network_params.py
  sim_params.py
  
## Model class
For the simulation, the network parameters are incorporated into 
a class (model_class.py) which further contains derived parameters
and a number of methods related to the mean field approximation of 
are network under consideration. 

## Simulation:
  simulate_microcircuit.py      -- for Potjans' model
  simulate_transition.py        -- for transition from Brunel's to Potjans' model
  
## Data structure
All data is save to HDF5 files, each run of the simulation creates 
a new file, with one group for each loop. Spikes and/or membrane potentials are saved according to sim_params.py. 

## Analysis
prep_data.py creates a results file, containing (some of) the most
important parameters (mean rates, mean membrane potentials, ...).
For further analysis, take a look at the corresponding notebooks. 

Data is not uploaded.
