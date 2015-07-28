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

def pn(name): 
    """label name printing for hdf5.File.visit(pn)"""
    print(name) 

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

def add_subplot(fig, 
        n_rows_cols=(1, 1), index_row_col=(0, 0), 
        rowspan=1, colspan=1, 
        width_ratios=None, 
        axisbg=None):
    """Add subplot specific to figure."""
    gridspec=plt.GridSpec(n_rows_cols[0], n_rows_cols[1], width_ratios=width_ratios)
    subplotspec=gridspec.new_subplotspec(index_row_col, rowspan=rowspan, colspan=colspan)
    if axisbg==None:
        ax = fig.add_subplot(subplotspec)
    else:
        ax = fig.add_subplot(subplotspec, axisbg=axisbg)
    return ax

def saving_fig(fig, figure_path, fig_name, verbose=True):
    if verbose:
        print("save figure to " + fig_name) 
    fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
            dpi=1000, # This is simple recommendation for publication plots
            bbox_inches='tight', format="pdf")
    fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
            dpi=1000, # This is simple recommendation for publication plots
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
        return_obj = np.array(return_obj).astype("<U8")
    else:                                   # is single string (layer or population)
        return_obj = single_rlbl(str_or_array) 
    return return_obj
    
def adjust_steps(data):
    """Drawstyle "steps" shifts data to the left by one bin. 
    This function takes data and slides it by on bin to the right. 
    Size is unchanged, s.t. the last entry is dropped.
    """
    data_out = np.append(data[0], data[:-1])
    return data_out


def resadjust(ax, xres=None, yres=None):
    """
    Send in an axis and I fix the resolution as desired.
    """

    if xres:
        start, stop = ax.get_xlim()
        ticks = np.arange(start, stop + xres, xres)
        ax.set_xticks(ticks)
    if yres:
        start, stop = ax.get_ylim()
        ticks = np.arange(start, stop + yres, yres)
        ax.set_yticks(ticks)
