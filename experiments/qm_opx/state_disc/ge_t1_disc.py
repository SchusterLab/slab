# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# this file works with version 0.7.411 & gateway configuration of a single controller #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator import TwoStateDiscriminator
from configuration_IQ import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params.npz')

wait_time = 500000
N = 1000
dt = 1000
T_max = 100000
T_min = 1000
times = np.arange(T_min, T_max + dt/2, dt)

with program() as t1:

    n = declare(int)
    t = declare(int)
    res = declare(bool)
    statistic = declare(fixed)

    res_st = declare_stream()
    statistic_st = declare_stream()

    with for_(n, 0, n < N, n + 1):
        with for_(t, T_min, t < T_max + dt/2, t + dt):
            wait(wait_time//4, "qubit")
            play("pi", "qubit")
            wait(t, "qubit")
            align('qubit', 'rr')
            discriminator.measure_state("long_readout", "out1", "out2", res, statistic=statistic)
            save(res, res_st)
            save(statistic, statistic_st)


    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        statistic_st.buffer(len(times)).average().save('statistic')

qm = qmm.open_qm(config)
job = qm.execute(t1, duration_limit=0, data_limit=0)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
statistic = result_handles.get('statistic').fetch_all()

plt.plot(res, 'b*')
plt.figure()
plt.plot(statistic, 'r*')


path = os.getcwd()
data_path = "../data"
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 't1', suffix='.h5'))
print(seq_data_file)
times = 4*times #actual clock time
with File(seq_data_file, 'w') as f:
    f.create_dataset("res", data=res)
    f.create_dataset("st", data=statistic)
    f.create_dataset("time", data=times)