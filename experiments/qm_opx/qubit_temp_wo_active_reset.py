"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config,  ge_IF, qubit_freq, biased_th_g_jpa
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
a_max = 1.0
da = 0.01
a_vec= np.arange(a_min, a_max + da/2, da)

avgs = 2000
reset_time = 500000
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(5000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')
    # save(I, "check")

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as qubit_temp:

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

            # active_reset(biased_th_g_jpa)
            # align('qubit', 'rr', 'jpa_pump')
            wait(reset_time//4, 'qubit')
            play('pi', 'qubit')
            align('qubit', 'qubit_ef')
            play("gaussian"*amp(a), "qubit_ef")
            align('qubit', 'qubit_ef')
            play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, Qg_st)
            save(I, Ig_st)

            align('qubit', "qubit_ef", 'rr', 'jpa_pump')
            wait(reset_time//4, 'qubit_ef')
            # active_reset(biased_th_g_jpa)
            # align('qubit_ef', 'rr', 'jpa_pump')
            play("gaussian"*amp(a), "qubit_ef")
            align('qubit', 'qubit_ef')
            play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
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
    job = qm.simulate(qubit_temp, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(qubit_temp, duration_limit=0, data_limit=0)

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
                                 get_next_filename(data_path, 'qubit_temp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Ig", data=Ig)
        f.create_dataset("Qg", data=Qg)
        f.create_dataset("Ie", data=Ie)
        f.create_dataset("Qe", data=Qe)
        f.create_dataset("amps", data=a_vec)
