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
    duration=15000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)


qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz')

wait_time = 500000
N = 1000

with program() as active_reset:

    n = declare(int)
    a = declare(fixed)
    res = declare(bool)
    statistic = declare(fixed)

    res_st = declare_stream()
    statistic_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        wait(500000//4, 'rr')
        align('qubit', 'rr')
        discriminator.measure_state("clear", "out1", "out2", res, statistic=statistic)
        with if_(~res):
            align('qubit', 'rr')
            wait(10000//4, 'qubit')
            play('pi', 'qubit')
        align('qubit', 'rr')
        # wait(4000//4, 'rr')
        discriminator.measure_state("clear", "out1", "out2", res, statistic=statistic)
        save(res, res_st)
        save(statistic, statistic_st)


    with stream_processing():
        res_st.boolean_to_int().save_all('res')
        statistic_st.save_all('statistic')

qm = qmm.open_qm(config)
job = qm.execute(active_reset, duration_limit=0, data_limit=0)
# job = qm.simulate(active_reset, simulation_config)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()['value']
statistic = result_handles.get('statistic').fetch_all()['value']

print(np.mean(res))
plt.figure()
plt.hist(statistic, bins=50)
