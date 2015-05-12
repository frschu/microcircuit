import os

data_sup_path = '/users/schuessler/uni/microcircuit/data/'
#data_path = data_sup_path + 'a0.5_t10.0_01/'
for suffix in os.listdir(data_sup_path):
    data_path = data_sup_path + suffix + '/'
    for name in os.listdir(data_path):
        if name.startswith('rec_spikes'):
            new_name = name.replace('spikes', 'spike')
            os.rename(data_path + name, data_path + new_name)
            print(name, new_name)
        elif name.endswith('rec_spikes.npy'):
            new_name = name.replace('spikes', 'spike')
            os.rename(data_path + name, data_path + new_name)
            print(name, new_name)

