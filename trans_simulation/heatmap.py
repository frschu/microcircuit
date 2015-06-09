from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import sys, os
sys.path.append(os.path.abspath('../')) # include path with style
sys.path.append(os.path.abspath('../trans_simulation')) # include path with style
from imp import reload
import pres_style as style; reload(style)
import network_params_trans as net_par
# close other plots by adding 'c' after 'run <script>' 
if 'c' in sys.argv:
    plt.close('all')
picture_format = '.pdf'
######################################################

conn_probs = np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.    , 0.0076, 0.    ],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.    , 0.0042, 0.    ],
             [0.0077, 0.0059, 0.0497, 0.135 , 0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.    , 0.1057, 0.    ],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06  , 0.3158, 0.0086, 0.    ],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001 , 0.0034, 0.0005, 0.0277, 0.008 , 0.0658, 0.1443]])
	     

net = net_par.net()
data = net.C_ab
data = conn_probs

fig = plt.figure()
ax = fig.add_subplot(111)
extent = [0, 8, 0, 8]
heatmap = ax.imshow(data, interpolation='none', extent=extent, cmap=cm.afmhot_r)
cbar = plt.colorbar(heatmap)
#cbar.set_clim(0, 0.5)
cbar.set_label("Connection probability")


ticks=np.arange(0.5, 8., 1)
ax.set_xticks(ticks)
ax.set_xticklabels(net.populations)
ax.set_xlabel('Presynaptic population')
ax.xaxis.set_tick_params(labeltop='on')
ax.xaxis.set_tick_params(labelbottom='off')
ax.xaxis.set_label_position("top")
ax.set_yticks(ticks)
ax.set_yticklabels(net.populations[::-1])
ax.set_ylabel('Postsynaptic population')
ax.grid(False)

for ax in fig.axes:
    style.fixticks(ax)
fig.show()
