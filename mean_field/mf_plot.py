"""mf_plot.py

Initializes fig to be plotted. 
Contains functions to plot v0(g) and v0(v_ext)

Insert mf_net object for calculation of Brunel's approximation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.abspath("../")) # include path with style
import style
import mf_class; reload(mf_class)

class mf_plot:
    def __init__(self, mf_net, suptitle='suptitle', plot_pops=np.array(['e', 'i'])):
        # close other plots by adding "c" after "run <script>" 
        if 'c' in sys.argv:
            plt.close('all')
        self.picture_format = '.pdf'
        self.figure_path = './'

        self.fig = plt.figure()
        self.fig.suptitle(suptitle, y=0.98)

        self.mf_net = mf_net
        self.plot_pops = plot_pops
        # indices of plotted plot_pops:
        self.i_pop  = np.array([np.where(plot_pop == self.mf_net.populations)[0][0] 
            for plot_pop in plot_pops])

        self.markers = [".", "v", "^", "*", "p", "<", ">", "s"]

    def plot_v0_g_full(self, gs, v_exts, v0s):
        ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
        colors = style.colors[:len(v_exts)]
        gs_lo   = np.linspace(0, 4, 100)
        gs_up   = np.linspace(4.01, 8, 100)
        for i, population in zip(self.i_pop, self.plot_pops):
            #ax.plot(gs_lo, self.mf_net.brunel_v0(gs_lo), '-.', color="lightgray", 
            #    label="$v_{0,\, approx,\, g < 4}$")
            #ax.plot([0], [0], '--', color="lightgray", 
            #    label="$v_{0,\, approx,\, g > 4}$")
            for j, v_ext_factor in enumerate(v_exts):
                v_ext = self.mf_net.v_thr * v_ext_factor
            #    ax.plot(gs_up, self.mf_net.brunel_v0(gs_up, v_ext), '--', color=colors[j], alpha=0.6) 
                ax.plot(gs, v0s[j, :, i], self.markers[i], color=colors[j], 
                    label="$\\frac{v_{ext}}{v_{thr}} = %.1f$"%v_ext_factor)
        ax.set_xlabel("$g$")
        ax.set_ylabel("$\\nu_0$ / Hz")
        ax.grid(True)
        ax.legend()
        ax.set_ylim(0, 520)

    def plot_v0_g(self, gs, v_exts, v0s):
        ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
        colors = style.colors[:len(v_exts)]
        for i, population in zip(self.i_pop, self.plot_pops):
            ax.plot([0], [0], '--', color="lightgray", 
                label="$v_{0,\, approx,\, g > 4}$")
            for j, v_ext_factor in enumerate(v_exts):
                v_ext = self.mf_net.v_thr * v_ext_factor
                ax.plot(gs, v0s[j, :, i], self.markers, color=colors[j], 
                    label="$\\frac{v_{ext}}{v_{thr}} = %.1f$"%v_ext_factor)
                ax.plot(gs, self.mf_net.brunel_v0(gs, v_ext), '--', color=colors[j], alpha=0.6) 
        ax.set_xlabel("$g$")
        ax.set_ylabel("$\\nu_0$ / Hz")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(4, 8)
        ax.set_ylim(0, 40)

    def plot_v0_v_ext(self, gs, v_exts, v0s):
        ax = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
        colors = style.colors[:len(gs)]
        for i, population in zip(self.i_pop, self.plot_pops):
            for j, g in enumerate(gs):
                ax.plot(v_exts, v0s[j, :, i], self.markers[i], color=colors[j], label="$g = %.1f$"%g)
        ax.set_xlabel("$\\frac{v_{ext}}{v_{thr}}$")
        ax.set_ylabel("$\\nu_0$ / Hz")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 40)

    def plot_bounds(self):
        n_pop = 2
        vs = np.repeat([np.linspace(1.0, 500, 2000)], n_pop, axis=0)
        mus = self.mf_net.tau_m * \
            (np.dot(self.mf_net.mat1, vs) + self.mf_net.mu_ext[:,None])    
        sds = np.sqrt(self.mf_net.tau_m * \
            (np.dot(self.mf_net.mat2, vs) + self.mf_net.var_ext[:,None]))    
        lows= ((self.mf_net.V_r - mus) / sds)[0]
        ups = ((self.mf_net.theta - mus) / sds)[0]

        ax = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
        colors = style.colors[:2]
        ax.plot(lows, vs[0], '-', color=colors[0], label="lower bound")
        ax.plot(ups, vs[0], '-', color=colors[1], label="upper bound")
        #ax.plot(ups - lows, vs[0], '-', color=colors[1], label="upper bound")
        ax.set_xlabel("boundaries of integration")
        ax.set_ylabel("$\\nu_0$ / Hz")
        ax.grid(True)
        ax.legend()
        ax.set_ylim(0, 520)
        #ax.set_xscale('log')
        return vs, lows, ups
        
    def plot_transform(self, xs, v0s, xlabel="$g$"):
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
        colors = style.colors[:len(self.plot_pops)]
        for i, population in zip(self.i_pop, self.plot_pops):
            ax.plot(xs, v0s[:, i], '.', color=colors[i], 
                label=population)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$\\nu_0$ / Hz")
        ax.grid(True)
        ax.legend()
        ax.set_ylim(0, 520)

