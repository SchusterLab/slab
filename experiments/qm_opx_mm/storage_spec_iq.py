from configuration_IQ import config, storage_IF, biased_th_g_jpa, disc_file
from qm.qua import *
from qm import SimulationConfig
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)
###############
# qubit_spec_prog:
###############
f_min = -100e3
f_max = 100e3
df = 5e3

f_vec = np.arange(f_min, f_max + df/2, df)

avgs = 1000
reset_time = 5000000
simulation = 0
with program() as storage_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, storage_IF[1] + f_min, f < storage_IF[1] + f_max + df/2, f + df):

            update_frequency("storage_mode1", f)
            wait(reset_time// 4, "storage_mode1")# wait for the storage to relax, several T1s
            play("saturation"*amp(0.05), "storage_mode1", duration=1000)
            align("storage_mode1", "qubit_mode0")
            play("res_pi", "qubit_mode0")
            align("qubit_mode0", "rr")
            discriminator.measure_state("readout", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
        I_st.buffer(len(f_vec)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(storage_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(storage_spec, duration_limit=0, data_limit=0)
    print("Experiment done")

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    # plt.figure()
    # plt.plot(f_vec, res, '.-')
    # plt.show()
    # job.halt()
    #

