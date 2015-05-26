"""mean_field.py

Numerical analysis of the equations corresponding to the 
stationary solutions of the mean field approximation of the 
cortical microcircuit model. 

8 coupled integral equations are solved numerically. 
"""
from __future__ import print_function
from imp import reload
import numpy as np
from scipy.optimize import root
import time
import mf_class; reload(mf_class)
plotting = True
iterate_g = True                # iterate g for fixed v_ext
iterate_v_ext = False           # iterate v_ext for fixed g

# Global parameters
n_pop = 2
choose_model = "brunelA"  # brunelA, brunelB for corresponding models!
mf_net  = mf_class.mf_net()
populations = mf_net.populations
populations = populations[:1]

######################################################
# Functions
######################################################

def v0_g(v_exts, gs, root_method=None, options=None):
    """Iterate over g for fixed v_ext (fig. 1 B.1)
    Returns v0s[v_ext, g, population]
    """
    print("Iterate over g")
    v0s = np.zeros([len(v_exts), len(gs), n_pop])
    #print("g\tv_ext\tv0\t\td(v0)")
    for i, v_ext_factor in enumerate(v_exts):
        v_guess = np.array([2.]*n_pop)  # initial guess
        for j, g in enumerate(gs):
            # create instance of class:
            mf_net = mf_class.mf_net(choose_model=choose_model, n_pop=n_pop, 
                g=g, v_ext_factor=v_ext_factor)
            sol = root(mf_net.root_v0, v_guess, jac=mf_net.jacobian, 
                method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                d   = np.linalg.norm(mf_net.root_v0(v0))
                v0s[i, j] = v0
                v_guess = v0
            else:
                v0s[i, j] = -1
    return v0s

def v0_v_ext(v_exts, gs, root_method=None, options=None):
    """Iterate over v_ext for fixed g (fig. 1 B.2)
    Returns v0s[g, v_ext, population]
    """
    print("Iterate over v_ext")
    v0s = np.zeros([len(gs), len(v_exts), n_pop])
    #print("g\tv_ext\tv0\t\td(v0)")
    for i, g in enumerate(gs):
        v_guess = np.array([2.]*n_pop)  # initial guess
        for j, v_ext_factor in enumerate(v_exts):
            # create instance of class:
            mf_net = mf_class.mf_net(choose_model=choose_model, n_pop=n_pop, 
                g=g, v_ext_factor=v_ext_factor)
            sol = root(mf_net.root_v0, v_guess, jac=mf_net.jacobian, 
                method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                d   = np.linalg.norm(mf_net.root_v0(v0))
                v0s[i, j] = v0
                v_guess = v0
            else:
                v0s[i, j] = -1
    return v0s

######################################################
# Solving
######################################################

# v_0 over g (fig. 1 B.1)
root_method     = ['hybr', 'lm', 'broyden1', 'anderson', 'krylov'][2]
# broyden1 works for g >~ 3.8, v_ext > 0.7
print("Method: ", root_method)
options         = None

if iterate_g:
    v_exts_g    = np.array([1.1, 2., 4., 8.])
    gs_g        = np.arange(8., 3.5, -0.1)
    v_exts_g    = np.array([2.])
    gs_g        = np.array([6.])

    t_int0      = time.time()
    v0s_g       = v0_g(v_exts_g, gs_g, root_method, options)
    t_int1      = time.time() - t_int0
    print("Integration time: %.2f"%(t_int1))
    goodness    = np.sum(v0s_g[:, :, 0] == -1, axis=1)
    print("v_ext\t#failure")
    for v_ext, gn in zip(v_exts_g, goodness):
        print("%.1f\t%i"%(v_ext, gn))

if iterate_v_ext:
    gs_v        = np.array([4.5, 5, 6, 8])
    v_exts_v    = np.arange(4.0, 0.7, -0.1)

    t_int0      = time.time()
    v0s_v       = v0_v_ext(v_exts_v, gs_v, root_method, options)
    t_int2      = time.time() - t_int0
    print("Integration time: %.2f"%(t_int2))
    goodness = np.sum(v0s_v[:, :, 0] == -1, axis=1)
    print("v_ext\t#failure")
    for v_ext, gn in zip(v_exts_v, goodness):
        print("%.1f\t%i"%(v_ext, gn))


######################################################
# Plotting
######################################################
if plotting:
    import mf_plot; reload(mf_plot)
    plot = mf_plot.mf_plot(mf_net, choose_model, populations) 

    if iterate_g:
        plot.plot_v0_g_full(gs_g, v_exts_g, v0s_g)
    if iterate_v_ext:
        plot.plot_v0_v_ext(gs_v, v_exts_v, v0s_v)

    fig_name = 'mean_field_v0'
    #plot.fig.savefig(figure_path + fig_name + picture_format)
    plot.fig.show()

