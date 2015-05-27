"""mean_field.py

Simple 2d plots in order to get a feeling for the problem. 
The frequencies v are only diagonal: v = [v1, v1, v1, ...].

Choose parameters in separate file. 
"""
from __future__ import print_function
import numpy as np
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.abspath("../")) # include path with style
import style
# Close other plots by adding "c" after "run <script>" 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'
figure_path = './'
# Import specific moduls
#from mean_field_params import *
#from mean_field_functions import *

import mf_class; reload(mf_class)
n_pop = 2
choose_model = "brunelA"  # brunelA, brunelB for corresponding models!
# Create reference instance containing parameters and functions:
mf_net0  = mf_class.mf_net(choose_model=choose_model, n_pop=n_pop)
plot_pop = mf_net0.populations[1]    # These populations are plotted
if not type(plot_pop) == np.ndarray:
    plot_pop = np.array([plot_pop])
i_pop  = np.where(plot_pop == mf_net0.populations)[0] 

######################################################
# Plotting
######################################################
colors = style.colors[:n_pop]
mu_plot = lambda v: mf_net0.tau_m * (np.dot(mf_net0.mat1, v) + mf_net0.mu_ext[:,None])    
sd_plot = lambda v: np.sqrt(mf_net0.tau_m * (np.dot(mf_net0.mat2, v) + mf_net0.var_ext[:,None]))    

fig = plt.figure()
suptitle = "Mean field approach: model: " + choose_model
fig.suptitle(suptitle, y=0.98)
# Mean
ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
# SD
ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=1, rowspan=1)
# First summand
ax2 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
# Lower and upper boundary
ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
# Distance between lower and upper boundary
ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=1, rowspan=1)
# Integrand
ax5 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)

# Set up grids, variables, etc.
v_min = 0.5
v_max = 10.
v_plot = np.repeat([np.linspace(v_min, v_max, 50)], n_pop, axis=0)
mus = mu_plot(v_plot)
sds = sd_plot(v_plot)
lows= (mf_net0.V_r - mus) / sds
ups = (mf_net0.theta - mus) / sds
u_min   = np.min(lows)
u_max    = np.max(ups)
u_min, u_max = (-2, 5)
us  = np.linspace(u_min, u_max, 100) 

# Plotting everything
ax0.plot([v_min, v_max], [0]*2, "--", c="gray")
ax2.plot(v_plot[0], mf_net0.summand1(v_plot[0]), "-", color="gray", label="$1^{st}$ summand")
ax3.plot([0], [0], "--", c="gray", label="upper bound")
ax3.plot([0], [0], "-", c="gray", label="lower bound")
ax5.plot(us, mf_net0.integrand(us), "-", color="gray")
for i, population in zip(i_pop, plot_pop):
    ax0.plot(v_plot[i], mus[i], "-", color=colors[i], label=population)
    ax1.plot(v_plot[i], sds[i], "-", color=colors[i])
    ax3.plot(v_plot[i], lows[i],"-", color=colors[i])
    ax3.plot(v_plot[i], ups[i], "--", color=colors[i])
    ax4.plot(v_plot[i], ups[i] - lows[i], "-", color=colors[i])

# Set axes labels, etc.   
for ax in fig.axes[:5]:
    ax.set_xlabel("population frequency $\\nu$ / Hz")
    ax.grid(False)
    ax.legend(loc="best")
ax0.set_ylabel("$\mu$")
ax1.set_ylabel("$\sigma$")
ax2.set_ylabel("Unit")
ax3.set_ylabel("Bounds: $\\frac{V_r - \mu(\\nu)}{\sigma(\\nu)}$;" +\
    "$\, \\frac{\\theta - \mu(\\nu)}{\sigma(\\nu)}$")
ax3.set_xlim(v_min, v_max)
ax4.set_ylabel("up($\\nu$) - low($\\nu$)")
ax5.set_xlabel("reduced voltage $u = \\frac{V - \mu}{\sigma}$")
ax5.set_ylabel("$e^{u^2} (1 + \mathrm{erf(u)})$")
for ax in fig.axes:
    style.fixticks(ax)

fig_name = 'mean_field'
#fig.savefig(figure_path + fig_name + picture_format)
fig.show()


