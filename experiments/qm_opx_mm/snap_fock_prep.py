"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, storage_cal_file, two_chi_2, disc_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import scipy
import os
from slab.dataanalysis import get_next_filename
from fock_state_prep import snap_seq, snap_seq_test
"""Using SNAP pulses to create Fock states in the storage cavity"""

readout = 'readout' #'clear'

if readout=='readout':
    disc = disc_file
else:
    disc = disc_file_opt

f_min = -8.5e6
f_max = 0.5e6
df = (f_max-f_min)/180
f_vec = np.arange(f_min, f_max + df/2, df)
#Fock0 + Fock1: D(-1.31268847) S(0, pi) * D(1.88543008) |0>
# Fock1: D(-0.580) * S(0,pi) * D(1.143) * |0>
# Fock2: D(0.432) * S(1,pi) * D(-1.133) * S(0,pi) * D(0.497) * |0>
# Fock3: D(0.344) * S(2,pi) * D(-1.072) * S(1,pi) * D(-1.125) * S(0,pi) * D(1.878) * |0>
# Fock4: D(-0.284) * S(3,pi) * D(0.775) * S(2,pi) * D(-0.632) * S(1,pi) * D(-0.831) * S(0,pi) * D(1.555) * |0>
# 0.61593606 -1.68684431  1.05855258 -0.55647545  0.62277445 -0.63506165
# 0.27666705

# [ 0.56683305 -0.92097535  0.44288571 -0.38538039  0.735419   -0.27754585]
avgs = 100
reset_time = int(5*7.5e6)
simulation = 0

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc, lsb=True)

opx_amp = 1.0

def fock_prep(f_target=1, phase=0.0):

    with program() as exp:
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

            with for_(f, ge_IF[0] + f_min, f < ge_IF[0] + f_max + df/2, f + df):

                update_frequency("qubit_mode0", ge_IF[0])
                wait(reset_time// 4, "storage_mode1")# wait for the storage to relax, several T1s
                snap_seq_test(fock_state=f_target, phase=phase)
                update_frequency("qubit_mode0", f)
                align("storage_mode1", "qubit_mode0")
                play("res_pi", "qubit_mode0")
                align('qubit_mode0', 'rr')
                discriminator.measure_state(readout, "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)

        with stream_processing():

            res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
            I_st.buffer(len(f_vec)).average().save('I')

    qm = qmm.open_qm(config)
    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(exp, simulation_config)
        # samples = job.get_simulated_samples()
        # samples.con1.plot()
        # result_handles = job.result_handles
        # result_handles.wait_for_all_values()
        # res = result_handles.get('res').fetch_all()
        # I = result_handles.get('I').fetch_all()

    else:
        """To run the actual experiment"""
        print("Experiment execution Done")
        job = qm.execute(exp, duration_limit=0, data_limit=0)

    return job

for i in range(4, 5, 1):
    job = fock_prep(f_target=i, phase=0.05)
    result_handles = job.result_handles
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    plt.figure()
    plt.plot(f_vec, res, '.--')
    plt.show()
    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'snap_fock_prep', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=res)
        f.create_dataset("freq", data=f_vec)
        f.create_dataset("two_chi", data=two_chi)


# plt.figure()
# plt.pcolormesh(f_vec, phase_vec,  res, cmap='RdBu', shading='auto')
# plt.colorbar()
# plt.show()
#
# job.halt()
#
# path = os.getcwd()
# data_path = os.path.join(path, "data/thesis")
# seq_data_file = os.path.join(data_path,
#                              get_next_filename(data_path, 'snap_fock_prep', suffix='.h5'))
# print(seq_data_file)
#
# with File(seq_data_file, 'w') as f:
#     f.create_dataset("I", data=I)
#     f.create_dataset("Q", data=res)
#     f.create_dataset("freq", data=f_vec)
#     f.create_dataset("two_chi", data=two_chi[1])