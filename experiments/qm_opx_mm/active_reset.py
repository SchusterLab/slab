# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# this file works with version 0.7.411 & gateway configuration of a single controller #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, disc_file
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
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

wait_time = 500000
N = 5000

with program() as active_reset:

    n = declare(int)
    a = declare(fixed)
    res1 = declare(bool)
    res2 = declare(bool)
    I = declare(fixed)
    Q = declare(fixed)

    res1_st = declare_stream()
    res2_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < N, n + 1):

        discriminator.measure_state("readout", "out1", "out2", res1, I=I)
        save(res1, res1_st)
        # with if_(~res1):  # |g> = 0 = False
        #     align('qubit_mode0', 'rr')
        #     play('pi', 'qubit_mode0')
        #     align('qubit_mode0', 'rr')
        play('pi', 'qubit_mode0', condition=~res1)
        align('qubit_mode0', 'rr')
        play('pi2', 'qubit_mode0')
        discriminator.measure_state("readout", "out1", "out2", res2, I=I, Q=Q)
        save(res2, res2_st)
        save(I, I_st)
        save(Q, Q_st)

    with stream_processing():
        res1_st.boolean_to_int().save_all('res1')
        res2_st.boolean_to_int().save_all('res2')
        I_st.save_all('I')
        Q_st.save_all('Q')

qm = qmm.open_qm(config)
job = qm.execute(active_reset, duration_limit=0, data_limit=0)
# job = qm.simulate(active_reset, simulation_config)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res1 = result_handles.get('res1').fetch_all()['value']
res2 = result_handles.get('res2').fetch_all()['value']
I = result_handles.get('I').fetch_all()['value']
Q = result_handles.get('Q').fetch_all()['value']

plt.plot(I, Q, '.')

gg = 0
ge = 0
eg = 0
ee = 0

#################
# measured date #
#################

# for i in range(len(res1)):
#     if res1[i]==0 and res2[i]==0:
#         gg+=1
#     elif res1[i]==0 and res2[i]==1:
#         ge+=1
#     elif res1[i]==1 and res2[i]==0:
#         eg+=1
#     elif res1[i]==1 and res2[i]==1:
#         ee+=1
# gg = gg/len(res1)
# ge = ge/len(res1)
# eg = eg/len(res1)
# ee = ee/len(res1)
#
# print('gg ge eg ee')
# print(round(gg, 2), round(ge, 3), round(eg, 2), round(ee,2))
# ##########
# # expect #
# ##########
# T = 0.03
# Fgg = 0.922
# Fee = 0.884
# Fge = 0.077
# Feg = 0.116
#
# # For preparing in the ground:
# gg = (1-T)*(Fgg * Fgg) + T * (Feg * Feg)
# ee = (1-T)*(Fge * Fge) + T * (Fee * Fge)
# ge = (1-T)*(Fgg * Fge) + T * (Feg * Fee)
# eg = (1-T)*(Fge * Feg) + T * (Fee * Fgg)
#
# # For preparing in the excited
# # gg = (1-T)*(Fgg * Feg) + T * (Feg * Fgg)
# # ee = (1-T)*(Fge * Fge) + T * (Fee * Fee)
# # ge = (1-T)*(Fgg * Fee) + T * (Feg * Fge)
# # eg = (1-T)*(Fge * Fgg) + T * (Fee * Feg)
#
# print(round(gg, 2), round(ge, 3), round(eg, 2), round(ee,2))

# print(np.mean(res))
# plt.figure()
# plt.hist(I, bins=50)
