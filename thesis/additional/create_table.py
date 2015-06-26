import numpy as np
import sys, os
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specificaitons
import network_params as net

# Populations
f = net.C_aext
tabstr = ""
for fi in f:
    tabstr += "& %i "%fi
tabstr += r"\\"

# Connectivity
populations    = np.array([layer + typus for layer in net.layers for typus in net.types])
cstr = ""
for i, line in enumerate(net.conn_probs):
    cstr += populations[i] + "\n" + "    "
    for entry in line:
        cstr += "& %.3f "%entry
    cstr += "\\tn \n"


with open("./created_table", "a") as tab_file:
    tab_file.write("\n" + cstr + "\n")


