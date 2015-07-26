from imp import reload
import numpy as np
import h5py
import sys, os
import time
sys.path.append(os.path.abspath('../simulation/')) # include path with simulation specifications

# Import specific moduls
import model_class; reload(model_class)

def solve_model(model,
               v_guess_0=np.array([ 0.6,  2.5,  4. ,  5.6,  8.2,  8. ,  1.6,  7.6]),
               jacobian=False, root_method='hybr', options=None):
    """Solve for model, given the initial guess v_guess_0.
    Returns solution = v0.
    """
    from scipy.optimize import root

    if jacobian==False:
        jac = False
    else:
        print("Use jacobian")
        jac = model.jacobian
        
    # Run on initial guess v_guess_0
    sol = root(model.root_v0, v_guess_0, jac=jac, method=root_method, options=options)
    if sol["success"]:
        v0  = sol["x"]
        if np.all(v0 < 1./model.t_ref): 
            return v0
        else:       # converged unphysically (v0 >= 1/t_ref)
            raise Exception("Converged unphysically for v_guess_0")
    else:   
        raise Exception("No solution found for v_guess_0")
        

def check_solve(model,
               v_guess_0=np.array([ 0.6,  2.5,  4. ,  5.6,  8.2,  8. ,  1.6,  7.6]),
               jacobian=False, root_method='hybr', options=None):
    """Solve for model, given the initial guess v_guess_0.
    Returns solution = v0.
    """
    from scipy.optimize import root
    # Instantiation
    if jacobian:
        jac = model.jacobian
    else:
        jac = False

    # Run on initial guess v_guess_0
    sol = root(model.root_v0, v_guess_0, jac=jac, method=root_method, options=options)
    v0  = sol["x"]
    return v0, sol


# Unchanged parameters
area            = 1.0
neuron_model    = "iaf_psc_exp"
connection_rule = "fixed_indegree"
g               = 4.0
rate_ext        = 8.0 # Hz background rate
weight_rel_sd   = 0.1
delay_rel_sd    = 0.0

# Brunel:
j02             = 1.0
weight_rel_sd   = 0.0
n_neurons       = "brunel"
C_ab            = "brunel"
model_brunel      = model_class.model(area=area, 
                                      neuron_model=neuron_model,
                                       n_neurons=n_neurons, C_ab=C_ab, 
                                       connection_rule=connection_rule,
                                       j02=j02, g=g, rate_ext=rate_ext,
                                       weight_rel_sd=weight_rel_sd, 
                                       delay_rel_sd=delay_rel_sd) 


# Microcircuit light:
# only some parameters like Potjans" model
# adapt n_neurons AND C_ab!
j02             = 2.0
weight_rel_sd   = 0.1
n_neurons       = "micro"
C_ab            = "micro"
model_micro       = model_class.model(area=area, 
                                      neuron_model=neuron_model,
                                      n_neurons=n_neurons, C_ab=C_ab, 
                                      connection_rule=connection_rule,
                                      j02=j02, g=g, rate_ext=rate_ext,
                                      weight_rel_sd=weight_rel_sd, 
                                      delay_rel_sd=delay_rel_sd) 

model_names = ["Brunel", "Micro"]

n = model_micro.n_populations # Dimension
r_max = 2.0       # Radius around which solutions are sampled
r_rel = np.linspace(0., 1., n_rs + 1)

root_method='hybr'
options= {"xtol": 1e-13, "maxfev": 10**4}
jacobian = False

n_rs = 10
n_sims = 10

tints_mean = np.zeros((2, n_rs))
nfevs_mean = np.zeros((2, n_rs))
success_rate = np.zeros((2, n_rs))

t0 = time.time()
for k, model in enumerate([model_micro, model_brunel]):
    print(model_names[k])
    v_sol = solve_model(model)
    print("r_max\tsuccess_rate\tt_int/s\tn_fevs")
    for j in range(n_rs):
        n_success = 0
        tints = np.zeros(n_sims)
        nfevs = np.zeros(n_sims) 
        for i in range(n_sims):
            # Sample initial conditions        
            x = np.random.normal(0, 1, n)
            r = np.random.uniform(r_rel[j], r_rel[j+1]) # THIS IS NOT EQUIVALENT TO THE RADIUS!
            offset = r_max * r**(1 / n) / np.linalg.norm(x) * x
            v_guess_0 = v_sol + offset   

            t_int0      = time.time()
            rate_mf, sol = check_solve(model, v_guess_0,
                            jacobian=jacobian, root_method=root_method, options=options)
            t_int1      = time.time() - t_int0

            n_success += sol["success"]
            tints[i] = t_int1
            nfevs[i] = sol["nfev"]

        tints_mean[k, j] = np.mean(tints)
        nfevs_mean[k, j] = np.mean(nfevs)
        success_rate[k, j] = float(n_success) / float(n_sims)
        res_str = "{0:6.2f}\t\t{1:6.2f}\t{2:6.2f}\t{3:6.2f}".format(
            r_max * r_max_all[j+1], success_rate[k, j], tints_mean[k, j], nfevs_mean[k, j])
        print(res_str)
t_calc = time.time() - t0
        
print("Time for calculation: %.2f sec"%t_calc)
data_path        = "/users/schuessler/uni/microcircuit/analysis"
res_file_name   = "check_numerics.hdf5"
path_res_file   = os.path.join(data_path, res_file_name)
with h5py.File(path_res_file, "w") as  res_file:
    res_file.create_dataset("r_max_all", data=r_max * r_max_all[1:])
    res_file.create_dataset("tints_mean", data=tints_mean)
    res_file.create_dataset("nfevs_mean", data=nfevs_mean)
    res_file.create_dataset("success_rate", data=success_rate)
