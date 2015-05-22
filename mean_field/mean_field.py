"""mean_field.py

Numerical analysis of the equations corresponding to the 
stationary solutions of the mean field approximation of the 
cortical microcircuit model. 

8 coupled integral equations are solved numerically. 
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
plotting = False
# Import specific moduls
from mean_field_params import *
from mean_field_functions import *

######################################################
# Plotting
######################################################
colors = style.colors[:n_pop]
mu_plot = lambda v: tau_m * (np.dot(mat1, v) + mu_ext[:,None])    
sd_plot = lambda v: np.sqrt(tau_m * (np.dot(mat2, v) + sd_ext[:,None]))    
if plotting:
    fig = plt.figure()
    suptitle = "Mean field approach: model: " + choose_params 
    fig.suptitle(suptitle, y=0.98)
    # Mean
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=1, rowspan=1)
    # SD
    ax1 = plt.subplot2grid((3, 3), (0, 1), colspan=1, rowspan=1)
    # First summand
    ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
    # Lower and upper boundary
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1, rowspan=1)
    # Distance between lower and upper boundary
    ax4 = plt.subplot2grid((3, 3), (1, 1), colspan=1, rowspan=1)
    # Integrand
    ax5 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
    # Brunel's approximations for stationary frequencies
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
    ax7 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)
    ax8 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
    
    # Set up grids, variables, etc.
    v_min = 0.5
    v_max = 10.
    v_plot = np.repeat([np.linspace(v_min, v_max, 50)], n_pop, axis=0)
    mus = mu_plot(v_plot)
    sds = sd_plot(v_plot)
    lows= (V_r - mus) / sds
    ups = (theta - mus) / sds
    u_min   = np.min((V_r - mus) / sds)
    u_max    = np.max((theta - mus) / sds)
    u_min, u_max = (-2, 5)
    us  = np.linspace(u_min, u_max, 100) 
    
    # Plotting everything
    ax0.plot([v_min, v_max], [0]*2, "--", c="gray")
    ax2.plot(v_plot[0], summand1(v_plot[0]), "-", color="gray", label="$1^{st}$ summand")
    ax3.plot([0], [0], "--", c="gray", label="upper bound")
    ax3.plot([0], [0], "-", c="gray", label="lower bound")
    ax5.plot(us, integrand(us), "-", color="gray")
    for i, population in enumerate(populations):
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
    ax5.set_ylabel("$e^{u^2} (1 - \mathrm{erf(u)})$")
    for ax in fig.axes:
        style.fixticks(ax)

    # Brunel
    if choose_params == 'brunel':
        g_lower = np.linspace(0., 3.7, 100)
        g_upper = np.linspace(4.1, 8., 100)
        ax6.plot(g_lower, v0_g_lt_4(g_lower), "-", color="gray")
        ax6.plot(g_upper, v0_g_gt_4(g_upper), "--", color="gray")
        ax6.set_xlabel("weight factor $g$")
        ax6.set_ylabel("$\\nu_0$ / Hz")
        ax7.plot(g_lower, v0_g_lt_4(g_lower), "-", color="gray")
        ax7.set_xlabel("weight factor $g$")
        ax7.set_ylabel("$\\nu_0$ / Hz")
        ax8.plot(g_upper, v0_g_gt_4(g_upper), "--", color="gray")
        ax8.set_xlabel("weight factor $g$")
        ax8.set_ylabel("$\\nu_0$ / Hz")

    fig_name = 'mean_field'
    #fig.savefig(figure_path + fig_name + picture_format)
    fig.show()


######################################################
# Solving
######################################################
print("initial guess:\n", "v_guess = ", v_guess)
print("result for initial guess:\n", func(v_guess), "\n")
#sol = fsolve(func, v_guess, fprime=jacobian)
sol = root(func, v_guess, jac=jacobian)
v0  = sol["x"]
print("Succeeded:", sol["success"])
print("obtained solution:\n", "v0 = ", v0)
print("result for obtained solution:\n", func(v0))
