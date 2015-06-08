    fig = plt.figure()
    suptitle = 'Simulation for: area = %.1f, time = %ims'%(area, t_sim)
    suptitle += '\nfile: ' + sim_spec2
    
    fig.suptitle(suptitle, y=0.98)
    # Raster plot
    ax0 = plt.subplot2grid((1, 4), (0, 0), colspan=2, rowspan=1)
    # Rates
    ax1 = plt.subplot2grid((1, 4), (0, 2), colspan=1, rowspan=1)
    # CV of interspike interval (ISI)
    ax2 = plt.subplot2grid((1, 4), (0, 3), colspan=1, rowspan=1)
    
    y_mean = np.arange(n_populations) + 0.1
    bar_height = 0.8 
    for i, population in enumerate(populations):
        print(population)
        for times, neuron_ids in zip(times_all[i], neuron_ids_all[i]):
            ax0.plot(times, neuron_ids, '.', ms=3, color=colors[i], label=population)
        ax1.barh(y_mean[i], rates_mean[i], height=bar_height, color=colors[i], linewidth=0)
        ax2.barh(y_mean[i], cv_isi_mean[i], height=bar_height, color=colors[i], linewidth=0)
    
    ylim_mean = (0, n_populations)
    yticks_mean = np.arange(n_types * 0.5, n_populations, n_types)
    
    # Raster Plot
    xlim = (0, t_sim)
    ymax_raster = offsets[-1]
    ylim = (0, ymax_raster)
    ax0.set_yticks(yticks)
    ax0.set_yticklabels(populations)
    ax0.set_xlabel('simulation time / s')
    ax0.set_ylabel('Layer')
    ax0.set_xlim(*xlim)
    ax0.set_ylim(*ylim)
    ax0.grid(False)
    
    # Rates
    ax1.set_yticks(yticks_mean)
    ax1.set_yticklabels(layers)
    ax1.set_xlabel('firing rate / Hz')
    #ax1.set_ylabel('Layer')
    ax1.set_ylim(*ylim_mean)
    #ax1.set_xlim(0, 20)
    ax1.grid(False)
      
    # CV of ISI
    ax2.set_yticks(yticks_mean)
    ax2.set_yticklabels(layers)
    ax2.set_xlabel('CV of interspike intervals / Hz')
    #ax2.set_ylabel('Layer')
    ax2.set_ylim(*ylim_mean)
    #ax2.set_xlim(0, 20)
    ax2.grid(False)
    
    # Legend; order is reversed, such that labels appear correctly
    for i in range(n_types):
        ax1.barh(0, 0, 0, color=colors[-(i+1)], label=types[-(i+1)], linewidth=0)
    ax1.legend(loc='best')
    
    for ax in fig.axes:
        style.fixticks(ax)
    fig_name = 'rates_etc'
    
    #fig.savefig(figure_path + fig_name + picture_format):
    
