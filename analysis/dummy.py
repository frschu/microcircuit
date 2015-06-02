fig = plt.figure()
suptitle = 'Simulation for: area = %.1f, time = %ims'%(area, t_sim)
suptitle += '\nfile: ' + simulation_spec
if sli: 
    suptitle += '  SLI'
fig.suptitle(suptitle, y=0.98)
# Membrane pot over time
ax0 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
# Histogram of membrane pot
ax1 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)


for i, population in enumerate(populations):
    if population[-1] == "e":
        ax = ax0
    else:
        ax = ax1
    ax.plot(bin_edges[:-1], Vs_mean, linewidth=3., color=colors[i], label=population)
    ax.hist(Vs_all[:n_hist].T, bins=n_bins_Vs, normed=True, histtype='step', 
    fill=False, linewidth=1.0, color=[colors[i]]*n_hist, alpha=0.2)
    

# Potential over time
xlim = (0, t_sim * 1e-3)
ax0.set_xlabel('simulation time / s')
ax0.set_ylabel('Membrane potential / V')
ax0.set_xlim(*xlim)
ax0.grid(True)

for ax in fig.axes:
    ax.set_ylabel('Probability P(V)')
    ax.set_xlabel('Membrane potential $V$ / V')
    ax.set_xlim(V_min, V_max)
    ax.grid(True)
    ax.legend(loc='best')

for ax in fig.axes:
    style.fixticks(ax)
fig_name = 'potential'
if sli:
    fig_name += '_sli'
#fig.savefig(figure_path + fig_name + picture_format)

fig.show()
