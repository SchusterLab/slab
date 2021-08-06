"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi, disc_file
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
import time

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(1000//4, "jpa_pump")
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

###############
# qubit_spec_prog:
###############

t_min = 50
t_max = 100
dt = 10
t_vec = np.arange(t_min, t_max + dt/2, dt)
print(len(t_vec))

cav_amp = 0.4
t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time

avgs = 2000
reset_time = int(3.5e6)
simulation = 0
with program() as storage_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    t = declare(int)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()
    bit1_st = declare_stream()
    bit2_st = declare_stream()
    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(t, t_min, t < t_max + dt/2, t + dt):

            wait(reset_time//4, 'storage')
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            play('CW'*amp(cav_amp), 'storage', duration=t)
            align('storage', 'qubit')
            play("pi2", "qubit") # unconditional
            wait(t_chi//4, "qubit")
            frame_rotation(np.pi, 'qubit') #
            play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            save(res, bit1_st)

            reset_frame("qubit")
            wait(1000//4, "rr")
            align("qubit", "rr", 'jpa_pump')

            play("pi2", "qubit") # unconditional
            wait(t_chi//4//2-3, "qubit")# subtracted 3 to make the simulated waveforms accurate
            with if_(res==0):
                frame_rotation(np.pi, 'qubit')
                play("pi2", "qubit")
            with else_():
                frame_rotation(3/2*np.pi, 'qubit')
                play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            save(res, bit2_st)

    with stream_processing():
        bit1_st.boolean_to_int().buffer(len(t_vec)).save_all('bit1')
        bit2_st.boolean_to_int().buffer(len(t_vec)).save_all('bit2')

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
    result_handles.wait_for_all_values()
    bit1 = result_handles.get('bit1').fetch_all()['value']
    bit2 = result_handles.get('bit2').fetch_all()['value']

    num = bit1 + 2*bit2

    p_cav = [np.sum(num==0)/avgs, np.sum(num==1)/avgs, np.sum(num==2)/avgs, np.sum(num==3)/avgs]
    print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))

    job.halt()

    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'alpha_cal_binary_decomp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("bit1", data=bit1)
        f.create_dataset("bit2", data=bit2)
        f.create_dataset("amp", data=cav_amp)
        f.create_dataset("times", data=t_vec)
        f.create_dataset("avgs", data=avgs)

