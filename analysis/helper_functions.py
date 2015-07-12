"""helper_functions.py
Some functions used in many analysis steps.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from imp import reload
import numpy as np
import h5py
import sys, os
sys.path.append(os.path.abspath('../')) # include path with style
import style; reload(style)

def basic_data(path_res_file):
    with h5py.File(path_res_file, "r") as res_file:
        # Simulation attributes
        area    = res_file.attrs["area"]   
        t_sim   = res_file.attrs["t_sim"]  
        t_trans = res_file.attrs["t_trans"]
        dt      = res_file.attrs["dt"]    
        populations   = res_file.attrs["populations"].astype("|U4")
        layers        = res_file.attrs["layers"].astype("|U4")        
        types         = res_file.attrs["types"].astype("|U4")     
        n_populations = res_file.attrs["n_populations"]
        n_layers      = res_file.attrs["n_layers"]       
        n_types       = res_file.attrs["n_types"]     

        t_measure = t_sim - t_trans

    return (area, t_sim, t_trans, t_measure, dt, 
            populations, layers, types, 
            n_populations, n_layers, n_types)

def add_subplot(fig, n_rows_cols=(1, 1), index_row_col=(0, 0), rowspan=1, colspan=1):
    """Add subplot specific to figure."""
    gridspec=plt.GridSpec(*n_rows_cols)
    subplotspec=gridspec.new_subplotspec(index_row_col, rowspan=rowspan, colspan=colspan)
    ax = fig.add_subplot(subplotspec)
    return ax

def saving_fig(fig, figure_path, fig_name, verbose=True):
    if verbose:
        print("save figure to " + fig_name) 
    fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
            bbox_inches='tight', format="pdf")
    fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
            bbox_inches='tight', format="png")      

def rlbl(str_or_array):
    """Relabel array of or single population(s)"""
    def single_rlbl(pop_or_layer):
        """Relabel population or layer L23 to L2/3"""
        if pop_or_layer.startswith("L23"):
            if pop_or_layer.endswith("3"):  # is layer
                return "L2/3"
            else:                           # is population
                return "L2/3" + pop_or_layer[-1]
        else:
            return pop_or_layer
    if type(str_or_array)==np.ndarray:      # is array of layers/populations
        return_obj = []
        for pop_or_layer in str_or_array:
            return_obj.append(single_rlbl(pop_or_layer))
        return_obj = np.array(return_obj).astype("<U5")
    else:                                   # is single string (layer or population)
        return_obj = single_rlbl(str_or_array) 
    return return_obj
    
