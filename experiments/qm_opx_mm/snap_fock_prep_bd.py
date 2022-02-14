"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, storage_IF, disc_file
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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq

"""Using analytic SNAP pulses to create Fock states in the storage cavity followed by Binary Decomposition"""

# Fock0 + Fock1: D(-1.31268847) S(0, pi) * D(1.88543008) |0>
# Fock1: D(-0.580) * S(0,pi) * D(1.143) * |0>
# Fock2: D(0.432) * S(1,pi) * D(-1.133) * S(0,pi) * D(0.497) * |0>
# Fock3: D(0.344) * S(2,pi) * D(-1.072) * S(1,pi) * D(-1.125) * S(0,pi) * D(1.878) * |0>
# Fock4: D(-0.284) * S(3,pi) * D(0.775) * S(2,pi) * D(-0.632) * S(1,pi) * D(-0.831) * S(0,pi) * D(1.555) * |0>


simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

avgs = 2000
simulation = 0
t_chi = int((abs(0.5*1e9/two_chi[1]))) # in FPGA clock cycles, qubit rotates by pi in this time
opx_amp = 1.0

def fock_prep(f_state=1):

    reset_time = int((f_state+0.5)*7.5e6)

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)        # Averaging
        num = declare(int)
        res = declare(bool)
        bit1 = declare(bool)
        bit2 = declare(bool)
        I = declare(fixed)

        num_st = declare_stream()
        ###############
        # the sequence:
        ###############
        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage_mode1')
            reset_frame('qubit_mode0')
            # update_frequency('qubit_mode0', ge_IF[0])
            # update_frequency('storage_mode1', storage_IF[1])
            # align('storage_mode1', 'qubit_mode0')
            ########################
            """Analytic SNAP pulses to create Fock states"""
            # snap_seq(fock_state=f_state)
            """Known displacement drive on the cavity"""
            play('CW'*amp(1.0), 'storage_mode1', duration=18)
            ########################
            align('qubit_mode0','storage_mode1')
            """BD starts here"""
            play("pi2", 'qubit_mode0') # unconditional
            wait(t_chi//4, 'qubit_mode0')
            frame_rotation(np.pi, 'qubit_mode0') #
            play("pi2", 'qubit_mode0')
            # wait(10, 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", bit1, I=I)

            reset_frame('qubit_mode0')
            wait(500, "rr")
            align('qubit_mode0', "rr")

            play("pi2", 'qubit_mode0') # unconditional
            wait(t_chi//4//2-4, 'qubit_mode0') # subtracted 3 to make the simulated waveforms accurate
            with if_(bit1==0):
                frame_rotation(np.pi, 'qubit_mode0')
                play("pi2", 'qubit_mode0')
            with else_():
                frame_rotation(3/2*np.pi, 'qubit_mode0')
                play("pi2", 'qubit_mode0')
            # wait(10, 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", bit2, I=I)

            assign(num, Cast.to_int(bit1) + 2*Cast.to_int(bit2))
            save(num, num_st)

        with stream_processing():

            num_st.save_all('num')

    qm = qmm.open_qm(config)
    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(exp, simulation_config)
        samples = job.get_simulated_samples()
        samples.con1.plot()
        result_handles = job.result_handles
        # result_handles.wait_for_all_values()
        num = result_handles.get('num').fetch_all()['value']

    else:
        """To run the actual experiment"""
        print("Experiment execution Done")
        job = qm.execute(exp, duration_limit=0, data_limit=0)

    return job

job = fock_prep(f_state=2)

result_handles = job.result_handles

result_handles.wait_for_all_values()
num = result_handles.get('num').fetch_all()['value']

p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

print("n=0 => {}, n=1 => {}, n=2 => {}, n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))

# path = os.getcwd()
# data_path = os.path.join(path, "data/")
# seq_data_file = os.path.join(data_path,
#                              get_next_filename(data_path, 'snap_fock_prep', suffix='.h5'))
# print(seq_data_file)
#
# with File(seq_data_file, 'w') as f:
#     f.create_dataset("num", data=num)
#     f.create_dataset("freq", data=f_vec)
#     f.create_dataset("two_chi", data=two_chi)