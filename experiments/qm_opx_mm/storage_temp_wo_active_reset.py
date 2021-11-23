"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config,  ge_IF, two_chi, disc_file_opt, pi_len_resolved, storage_mode
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename

"""Qubit temperature"""

a_min = 0.0
a_max = 0.05
da = 0.0005
a_vec= np.arange(a_min, a_max + da/2, da)

avgs = 20000
reset_time = int(2.5e6)
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)


with program() as storage_temp:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    a = declare(fixed) #array of time delays
    I = declare(fixed)
    res = declare(bool)

    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(a, a_min, a < a_max + da/2, a + da):

            update_frequency('qubit_mode0', ge_IF[0])
            discriminator.measure_state("readout", "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            wait(reset_time//10, 'qubit_mode0')
            #####
            align('qubit_mode0', 'rr')
            play('gaussian'*amp(a), 'qubit_mode0', duration=pi_len_resolved//4)
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, Qg_st)
            save(I, Ig_st)

            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            #####
            update_frequency('qubit_mode0', ge_IF[0]+two_chi[storage_mode])
            wait(reset_time//10, 'qubit_mode0')
            play('gaussian'*amp(a), 'qubit_mode0', duration=pi_len_resolved//4)
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, Qe_st)
            save(I, Ie_st)

    with stream_processing():
        Ig_st.buffer(len(a_vec)).average().save('Ig')
        Qg_st.boolean_to_int().buffer(len(a_vec)).average().save('Qg')
        Ie_st.buffer(len(a_vec)).average().save('Ie')
        Qe_st.boolean_to_int().buffer(len(a_vec)).average().save('Qe')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(storage_temp, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(storage_temp, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    Ig = res_handles.get('Ig').fetch_all()
    Qg = res_handles.get('Qg').fetch_all()
    Ie = res_handles.get('Ie').fetch_all()
    Qe = res_handles.get('Qe').fetch_all()

    plt.figure()
    plt.plot(a_vec, Qg, '.-')
    plt.plot(a_vec, Qe, '.-')

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_temp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Ig", data=Ig)
        f.create_dataset("Qg", data=Qg)
        f.create_dataset("Ie", data=Ie)
        f.create_dataset("Qe", data=Qe)
        f.create_dataset("amps", data=a_vec)
