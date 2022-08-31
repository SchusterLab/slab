"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt
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

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

###############
# qubit_spec_prog:
###############

def amp_to_tvec(c_amp):
    cav_amp = 1.0
    t_min = int(6*(cav_amp/c_amp))
    t_max = int(10*(cav_amp/c_amp))
    dt = (int(1*(cav_amp/c_amp)))
    return t_min, t_max, dt

t_chi = int(abs(0.5*1e9/two_chi[1])) #qubit rotates by pi in this time

avgs = 1000
reset_time = int(7.5e6)
simulation = 0

def storage_bd(cav_amp):

    t_min, t_max, dt = amp_to_tvec(cav_amp)

    t_vec = np.arange(t_min, t_max + dt/2, dt)

    with program() as bd:

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

            with for_(t, t_min, t <= t_max, t + dt):

                wait(reset_time//4, 'storage_mode1')
                play('CW'*amp(cav_amp), 'storage_mode1', duration=t)
                align('storage_mode1', 'qubit_mode0')
                play("pi2", "qubit_mode0") # unconditional
                wait(t_chi//4+1, "qubit_mode0")
                frame_rotation(np.pi, 'qubit_mode0') #
                play("pi2", "qubit_mode0")
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)
                save(res, bit1_st)

                reset_frame("qubit_mode0")
                wait(250, "rr")
                align("qubit_mode0", "rr")

                play("pi2", "qubit_mode0") # unconditional
                wait(t_chi//4//2-4, "qubit_mode0")# subtracted 3 to make the simulated waveforms accurate
                with if_(res==0):
                    frame_rotation(np.pi, 'qubit_mode0')
                    play("pi2", "qubit_mode0")
                with else_():
                    frame_rotation(3/2*np.pi, 'qubit_mode0')
                    play("pi2", "qubit_mode0")
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)
                save(res, bit2_st)

        with stream_processing():
            bit1_st.boolean_to_int().buffer(len(t_vec)).save_all('bit1')
            bit2_st.boolean_to_int().buffer(len(t_vec)).save_all('bit2')

    qm = qmm.open_qm(config)

    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(bd, SimulationConfig(150000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        """To run the actual experiment"""
        job = qm.execute(bd, duration_limit=0, data_limit=0)
        print("Experiment done")
        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        bit1 = result_handles.get('bit1').fetch_all()['value']
        bit2 = result_handles.get('bit2').fetch_all()['value']

        num = bit1 + 2*bit2

        p_cav = [np.sum(num==0)/avgs, np.sum(num==1)/avgs, np.sum(num==2)/avgs, np.sum(num==3)/avgs]
        print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))

        job.halt()

        path = os.getcwd()
        data_path = os.path.join(path, "data/")
        # data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'alpha_cal_binary_decomp', suffix='.h5'))
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("bit1", data=bit1)
            f.create_dataset("bit2", data=bit2)
            f.create_dataset("amp", data=cav_amp)
            f.create_dataset("times", data=t_vec)
            f.create_dataset("avgs", data=avgs)
    return

st_amp = list(np.arange(0.001, 0.01, 0.001))
st_amp.extend(np.arange(0.01, 0.1, 0.01))
st_amp.extend(np.arange(0.1, 1.05, 0.1))
for a in st_amp:
    print(a)
    storage_bd(a)