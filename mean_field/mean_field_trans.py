"""mean_field_trans.py

Imports mf_trans and mf_plot.

Numerical analysis of the equations corresponding to the 
stationary solutions of the mean field approximation of the 
cortical microcircuit model. 

8 coupled integral equations are solved numerically. 
"""
from __future__ import print_function
import numpy as np
from scipy.optimize import root
import time
from imp import reload
import mf_trans as model; reload(model)
import mf_micro as mf_micro; reload(mf_micro)
import mf_plot; reload(mf_plot)
plotting = True
iterate_g = False              # iterate g for fixed v_ext
iterate_j02 = False             # iterate j02 for g=4
iterate_C = False

# Global parameters
choose_model = "brunelA"  # brunelA, brunelB for corresponding models!
n_layer = 4
n_types = 2
n_pop = n_layer * n_types
print("Model: ", choose_model)
print("n layers: ", n_layer)
# Create reference instance containing parameters and functions:
# (used mostly for plotting)
mf_net0  = model.mf_net()
plot_pops= mf_net0.populations    # These populations are plotted
if not type(plot_pops) == np.ndarray:
    plot_pops = np.array([plot_pops])

######################################################
# Functions
######################################################

def v0_g(v_guess0, gs, jacobian=False, root_method=None, options=None):
    """Iterate over g, everything else fixed.
    log: array for diagnostics
    Returns v0s[g, population], log
    """
    print("Iterate over g")
    v_guess = v_guess0 # initial guess
    v0s = np.zeros([len(gs), n_pop])
    log = np.zeros(len(gs))
    #print("g\tv_ext\tv0\t\td(v0)")
    for j, g in enumerate(gs):
        # create instance of class:
        mf_net = model.mf_net(g=g)
        if jacobian:
            jac = mf_net.jacobian
        else:
            jac = False
        try:
            sol = root(mf_net.root_v0, v_guess, jac=jac, method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                v0s[j]  = v0
                if np.all(v0 < 1./mf_net.t_ref): # converged unphysically (v0 > 1/t_ref)
                    log[j]  = 1
                    v_guess = v0
                else:
                    log[j]  = -3
            else:
                v0s[j]  = -1
                log[j]  = -1
        except:
            break
            v0s[j]  = -2
            log[j]  = -2
    return v0s, log

def v0_j02(v_guess0, j02s, jacobian=False, root_method=None, options=None):
    """Iterate over J[0, 2] for fixed g
    log: array for diagnostics
    Returns v0s[j02, population], log
    """
    print("Iterate over j")
    v_guess = v_guess0 # initial guess
    g = 4.
    v0s = np.zeros([len(j02s), n_pop])
    log = np.zeros(len(j02s))
    #print("g\tv_ext\tv0\t\td(v0)")
    for i, j02 in enumerate(j02s):
        # create instance of class:
        mf_net = model.mf_net(g=g, j02=j02)
        if jacobian:
            jac = mf_net.jacobian
        else:
            jac = False
        try:
            sol = root(mf_net.root_v0, v_guess, jac=jac, method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                v0s[i]  = v0
                if np.all(v0 < 1./mf_net.t_ref): 
                    log[i]  = 1
                    v_guess = v0
                else:       # converged unphysically (v0 >= 1/t_ref)
                    log[i]  = -3
            else:
                v0s[i]  = -1
                log[i]  = -1
        except:
            break
            v0s[i]  = -2
            log[i]  = -2
    return v0s, log
    

def cool_C_ab(v_guess_0, step_init = 0.01, d_step=0.5, vary_j02=False, tolerance=5, jacobian=False, root_method=None, options=None):
    """Iteratively change C_ab from C_B = Brunel's to C_M = microcircuit 
    on the straight line connecting C_B and C_M.
    If not disabled, j02 varied as well.
    Returns distances, v0s[distance, population], failures
    """
    print("Iterate over C_ab")
    # initiate
    g   = 4.
    j02_init = 1.
    j02 =   j02_init
    v_guess = v_guess_0
    C_B     = model.mf_net().C_ab
    distances   = []
    failures    = np.array([])
    mf_net = model.mf_net(g=g, j02=j02, C_ab=C_B)
    if jacobian:
        jac = mf_net.jacobian
    else:
        jac = False
    sol = root(mf_net.root_v0, v_guess, jac=jac, method=root_method, options=options)
    if sol["success"]:
        v0  = sol["x"]
        if np.all(v0 < 1./mf_net.t_ref): 
            v0s     = v0
            distances.append(0.)
            v_guess = v0
        else:       # converged unphysically (v0 >= 1/t_ref)
            raise Exception("Converged unphysically for v_guess_0")
    else:   
        raise Exception("No solution found for v_guess_0")
    # Define steps and matrices
    step    = step_init     # initial step size
    C_M     = mf_micro.mf_net().C_ab
    delta_C = C_M - C_B
    #delta_C[deltaC > 0] = 0     # only increasing
    #delta_C[:, 0::2] = 0        # only excitatory
    delta_j02    = 1.
    distance    = 0.
    n_fails = 0
    n_succ  = 0
    # Go on
    print("\t")
    while distance <= 1.:
        distance += step
        C_ab    = C_B + distance * delta_C
        if vary_j02:
            j02     = j02_init + distance * delta_j02
        mf_net  = model.mf_net(g=g, j02=j02, C_ab=C_ab)
        try:
            sol = root(mf_net.root_v0, v_guess, jac=jac, method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                if np.all(v0 < 1./mf_net.t_ref): 
                    v0s = np.vstack((v0s, v0))
                    distances.append(distance)
                    v_guess = v0
                    n_fails = 0
                    n_succ  +=1
                    if n_succ >= tolerance and step < step_init:
                        print("succ\t%.5f\t%i %i"%(distance, n_succ, np.log(step)/np.log(d_step)))
                        step /= d_step
                else:       # converged unphysically (v0 >= 1/t_ref)
                    raise Exception("unphysical")
            else:   
                raise Exception("no solution")
        except: # no (good) solution found
            failures = np.append(failures, distance)
            if n_fails == 0:
                distances.pop()
                v0s = np.delete(v0s, -1, axis=0)
            n_fails += 1
            n_succ   = 0
            print("fail\t%.5f\t%i %i"%(distance, n_fails, np.log(step)/np.log(d_step)))
            distance = distances[-1]
            step    *= d_step
            if n_fails >= tolerance:
                print("Tolerance exceeded at distance = %.3f"%distance)
                break
    distances = np.array(distances)
    return(distances, v0s, failures, C_ab, C_M, step)     

######################################################
# Solving
######################################################
jacobian        = False     # whether to use calculated jacobian
root_method     = ['hybr', 'lm', 'broyden1', 'anderson', 'krylov'][0]
print("Method: ", root_method)
options         = None

iterate_C = True
if iterate_C:
    v_guess0    = np.array([131, 127, 147, 141, 144, 141, 172, 147])
    d_step      = 0.1   # ratio by which step is reduced
    step_init   = d_step**3  # initial step size
    tolerance   = 10     # number of fails accepted at one distance
    t_int0      = time.time()
    dists, v0s, fails, C_ab, C_M, final_step  = cool_C_ab(v_guess0, step_init, d_step, tolerance, jacobian, root_method, options)
    t_int1      = time.time() - t_int0
    print("Integration time: %.2f"%(t_int1))

def diagnostics(log):
    """Currently not working"""
    successes       = np.sum(log ==  1) # converged successful     
    nots            = np.sum(log ==  0) # after break
    failures        = np.sum(log == -1) # algorithm stopped unsuccessfull
    exceptions      = np.sum(log == -2) # algorithm yielded NaN/overflow
    unphysicals     = np.sum(log == -3) # converged unphysically (v0 > 1/t_ref)
    print("%i\t%i\t%i\t%i\t%i"%(successes, nots, failures, exceptions, unphysicals))

if iterate_g:
    v_guess0 = np.array([2.]*n_pop)  # initial guess
    gs        = np.arange(10., 4., -0.1)
    #v_guess0 = np.array([131, 127, 147, 141, 144, 141, 172, 147])
    #gs        = np.array([4.])

    t_int0      = time.time()
    v0s, log  = v0_g(v_guess0, gs, jacobian, root_method, options)
    t_int1      = time.time() - t_int0
    print("Integration time: %.2f"%(t_int1))
    print("succ.\tnots\tfail\texcept\tv >= 500 Hz")
    diagnostics(log)
   
# iterate J for g = 4.
if iterate_j02:
    v_guess0    = np.array([131, 127, 147, 141, 144, 141, 172, 147])
    j02s        = np.arange(1, 2.0, 0.001)

    t_int0      = time.time()
    v0s, log    = v0_j02(v_guess0, j02s, jacobian, root_method, options)
    t_int1      = time.time() - t_int0
    print("Integration time: %.2f"%(t_int1))
    print("succ.\tnots\tfail\texcept\tv >= 500 Hz")
    diagnostics(log)
   

######################################################
# Plotting
######################################################
if plotting:
    if iterate_C:
        suptitle = "Step by step transforming BrunelA to Microcircuit: transform $C_{ab}$" + \
            "\nmethod: " + root_method
    if iterate_j02:
        suptitle = "Step by step transforming BrunelA to Microcircuit: transform $J_{L23e, L4e}$" + \
            "\nmethod: " + root_method
    plot = mf_plot.mf_plot(mf_net0, suptitle, plot_pops) 

    if iterate_C:
        plot.plot_transform(dists, v0s, xlabel="$d(C_{brunel}, C_{micro})$")
    
    if iterate_g:
        plot.plot_transform(gs, v0s, xlabel="$g$")
    
    if iterate_j02:
        plot.plot_transform(j02s, v0s, xlabel="$\\frac{J_{L23e, L4e}}{J}$")
    
    #vs0, lows0, ups0 = plot.plot_bounds()

    fig_name = 'mean_field_v0'
    #plot.fig.savefig(figure_path + fig_name + picture_format)
    plot.fig.show()

