from imp import reload
import subprocess
import sys, os, time
sys.path.append(os.path.abspath('../analysis/')) # include path with simulation specifications
import text_to_hdf5_append as tth; reload(tth)

sim_spec = "a1.0_t60.2"

n_populations = 8
sim_params = "./sim_params.sli"
with open(sim_params, 'r') as file:
    lines = file.readlines()

# Obtain master seed and n_vp from sim_params.sli
for i, line in enumerate(lines):
    if line.startswith(r"/master_seed"): # Master seed
        i_ms = i
        ms_line     = line.split(" ")
    if line.startswith(r"/n_threads_per_proc"): # Number of virtual processes
        n_vp        = int(line.split(" ")[1])

master_seed = int(ms_line[1])

t0 = time.time()
n_runs = 10
for run_i in range(n_runs):
    print(run_i)
    # Simulate
    t0sim = time.time()
    subprocess.call('nest microcircuit.sli', shell=True)
    tsim = time.time() - t0sim

    # Save data
    start_file = False # do not overwrite existing hdf5 file but append! ("r+")
    t0save = time.time()
    tth.save_sli_to_hdf5(sim_spec=sim_spec, start_file=start_file)
    tsave = time.time() - t0save

    # Change master seed
    master_seed = master_seed + 1 \
            + 2 * n_vp + 2 * n_populations \
            - 1  # last -1 since range ends beforehand
    ms_line[1] = str(master_seed)
    ms_line_complete = ""
    for snip in ms_line:
        ms_line_complete += snip + " "
    lines[46] = ms_line_complete
    with open(sim_params, 'w') as file:
        file.writelines(lines)

    print("Master seed: %i"%master_seed)
    print("Time for simulation : %.2f s"%tsim)
    print("Time for saving data: %.2f s"%tsave)

ttotal = time.time() - t0
print("Total time: %.2f s"%ttotal)
    
