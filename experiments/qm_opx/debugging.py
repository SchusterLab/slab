import numpy as np
import matplotlib.pyplot as plt
from h5py import File

seq_data_file = './data/tof_debug.h5'

with File(seq_data_file,'r') as a:
    ts = np.array(a['ts'])
    in1 = np.array(a['in1'])
    in2 = np.array(a['in2'])

ts_flat = ts.flatten()
print(int(len(ts_flat))/10000)
print(np.unique(np.diff(ts_flat)))
print(np.shape(in1))
print(np.shape(in2))





