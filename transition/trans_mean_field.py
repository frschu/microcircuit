"""trans_mean_field.py
    Iteratively solve consistency equation for v, starting at 
    model_init, changing towards model_final.

    Contains: 
    function transition: adaptive step size transition

    Global boundaries: model_init, model_final

    Plotting results.
"""

"""Further command line arguments:
        c       script will close all open plots
        sli     data of the original simulation written in sli will be analyzed. 
                Note that at this point, the data must be of the same simulation type, 
                as specifications are loaded from .npy-files of the pynest simulation. 

    Overview over all populations: Raster plot, mean rates, mean CV of ISI per population.
"""
from imp import reload
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import sys, os
import time
sys.path.append(os.path.abspath('../presentation')) # include path with style
sys.path.append(os.path.abspath('../transition/')) # include path with simulation specifications
import network_params_trans; reload(network_params_trans)
import user_params_trans as user; reload(user)
import pres_style as style; reload(style)

show_fig = False
save_fig = True
xfactor = 2.6
style.rcParams['figure.figsize'] = (xfactor*6.2, xfactor*3.83) 
figure_path = os.path.join(".", "figures")

# Close other plots by adding 'c' after 'run <script>' 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'

######################################################
# Functions
######################################################
def transition(model_init, model_final,  
               v_guess_0=np.array([110, 107, 122, 117, 120, 117, 141, 122]),
               step_init=0.01, d_step=0.5, tolerance=5,
               root_function='siegert',
               jacobian=False, root_method='hybr', options=None):
    """Iteratively solve consistency equation for v, starting at 
    model_init, changing towards model_final.

    Initial guess should not be too far off. Default for Brunel model A in 8D.
    
    Step size is adapted, starting with step_init, 
    adapting size with step *= d_step.
    Maximal number of step decreases = tolerance. 
    
    Further arguments:
    jacobian    ~ whether to use calculated jacobian
    root_method ~ method used by scipy.optimize.root
    options     ~ options for this method

    Returns distances, v0s[distance, population], failures, last model
    """
    # Instantiation
    if jacobian:
        jac = mf_net.jacobian
    else:
        jac = False

    # Run on initial guess v_guess_0
    if root_function=="siegert":
        sol = root(model_init.root_v0_siegert, v_guess_0, jac=jac, method=root_method, options=options)
    else:
        sol = root(model_init.root_v0, v_guess_0, jac=jac, method=root_method, options=options)
    if sol["success"]:
        print("intial success")
        v0  = sol["x"]
        if np.all(v0 < 1./model_init.t_ref): 
            v0s     = v0
            distances = [0]
            v_guess = v0
        else:       # converged unphysically (v0 >= 1/t_ref)
            raise Exception("Converged unphysically for v_guess_0")
    else:   
        raise Exception("No solution found for v_guess_0")
    
    # Define steps and matrices
    step    = step_init     # initial step size
    dist    = 0.
    n_fails = 0
    n_succ  = 0
    failures    = []

    # Looping
    while dist <= 1.:
        dist += step
    
        # New model
        area            = (1. - dist) * model_init.area         + dist * model_final.area        
        n_neurons       = (1. - dist) * model_init.n_neurons    + dist * model_final.n_neurons   
        C_ab            = (1. - dist) * model_init.C_ab         + dist * model_final.C_ab        
        j02             = (1. - dist) * model_init.j02          + dist * model_final.j02         
        g               = (1. - dist) * model_init.g            + dist * model_final.g           
        rate_ext        = (1. - dist) * model_init.rate_ext     + dist * model_final.rate_ext    
        PSC_rel_sd      = (1. - dist) * model_init.PSC_rel_sd   + dist * model_final.PSC_rel_sd  
        delay_rel_sd    = (1. - dist) * model_init.delay_rel_sd + dist * model_final.delay_rel_sd
        model = network_params_trans.net(area=area, 
                                         n_neurons=n_neurons, C_ab=C_ab, 
                                         connection_type="fixed_indegree",
                                         j02=j02, g=g, rate_ext=rate_ext,
                                         PSC_rel_sd=PSC_rel_sd, 
                                         delay_rel_sd=delay_rel_sd) 
        try:
            if root_function=="siegert":
                sol = root(model.root_v0_siegert, v_guess, jac=jac, method=root_method, options=options)
            else:
                sol = root(model.root_v0, v_guess, jac=jac, method=root_method, options=options)
            if sol["success"]:
                v0  = sol["x"]
                if np.all(v0 < 1./model.t_ref): 
                    v0s = np.vstack((v0s, v0))
                    distances.append(dist)
                    v_guess = v0
                    n_fails = 0
                    n_succ  +=1
                    if n_succ >= tolerance and step < step_init:
                        print("succ\t%.5f\t%i %i"%(dist, n_succ, np.log(step)/np.log(d_step)))
                        step /= d_step
                else:       # converged unphysically (v0 >= 1/t_ref)
                    raise Exception("unphysical")
            else:   
                raise Exception("no solution")
        except: # no (good) solution found
            failures.append(dist)
            n_fails += 1
            n_succ   = 0
            print("fail\t%.5f\t%i %i"%(dist, n_fails, np.log(step)/np.log(d_step)))
            dist = distances[-1]
            step    *= d_step
            if n_fails >= tolerance:
                print("Tolerance exceeded at distance = %.3f"%dist)
                break
    distances = np.array(distances)
    failures  = np.array(failures)

    return(distances, v0s, failures, model)     


#######################################################
# Global boundaries
#######################################################
# Unchanged parameters
area            = 1.0
connection_type = "fixed_indegree"


g               = 4.0
rate_ext        = 8.0 # Hz background rate
PSC_rel_sd      = 0.0
delay_rel_sd    = 0.0

# Brunel:
j02             = 1.0
n_neurons       = "brunel"
C_ab            = "brunel"
net_brunel      = network_params_trans.net(area=area, 
                                           n_neurons=n_neurons, C_ab=C_ab, 
                                           connection_type=connection_type,
                                           j02=j02, g=g, rate_ext=rate_ext,
                                           PSC_rel_sd=PSC_rel_sd, 
                                           delay_rel_sd=delay_rel_sd) 

# Microcircuit light:
# only some parameters like Potjans" model
# adapt n_neurons AND C_ab!
j02             = 1.0
n_neurons       = "micro"
C_ab            = "micro"
net_micro       = network_params_trans.net(area=area, 
                                           n_neurons=n_neurons, C_ab=C_ab, 
                                           connection_type=connection_type,
                                           j02=j02, g=g, rate_ext=rate_ext,
                                           PSC_rel_sd=PSC_rel_sd, 
                                           delay_rel_sd=delay_rel_sd) 


#######################################################
# Run the loop
#######################################################
model_init      = net_brunel
model_final     = net_micro
v_guess_0       = np.array([110, 107, 122, 117, 120, 117, 141, 122])
d_step      = 0.1   # ratio by which step is reduced
step_init   = d_step**3  # initial step size
tolerance   = 5     # number of fails accepted at one distance
root_function = "siegert" # very slow and no difference...
root_function = None
jacobian=False
root_method='hybr'
options=None

t_int0      = time.time()
dists, v0s, fails, last_model = transition(model_init, model_final, 
                                           v_guess_0, step_init, d_step, tolerance,
                                           root_function, jacobian, root_method, options)
t_int1      = time.time() - t_int0
print("Integration time: %.2f"%(t_int1))


######################################################
# Plotting
######################################################
plot_pops=np.array(['L4e', 'L4i'])
plot_pops= last_model.populations    # These populations are plotted
if not type(plot_pops) == np.ndarray:
    plot_pops = np.array([plot_pops])
i_pop  = np.array([np.where(plot_pop == last_model.populations)[0][0] 
                   for plot_pop in plot_pops])

fig = plt.figure()
if not save_fig:
    suptitle = "Step by step transforming BrunelA to Microcircuit: transform $C_{ab}$" + \
        "\nmethod: " + root_method
    suptitle += "\nfile: " + sim_spec
    fig.suptitle(suptitle, y=0.98)

ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
colors = style.colors[:len(plot_pops)]
for i, population in zip(i_pop, plot_pops):
    ax.plot(dists, v0s[:, i], '.', color=colors[i], 
        label=population)
ax.set_xlabel("$d(C_{brunel}, C_{micro})$")
ax.set_ylabel("$\\nu_0$ / Hz")
#ax.set_yscale("log")
ax.grid(True)
ax.legend(loc="best")
ax.set_ylim(0, 20)
ax.set_xlim(0, min(1.0, dists[-1]*1.1))

fig_name = "numerical_approach_num_only"
fig_name += picture_format
    
if save_fig:
    print("save figure to " + fig_name)
    fig.savefig(os.path.join(figure_path,fig_name))
    
if show_fig:
    fig.show()
