"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, st_self_kerr, storage_IF, storage_cal_file, opt_len, two_chi_2, disc_file, two_chi_vec
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
import scipy
from slab.dataanalysis import get_next_filename
from slab.dsfit import*
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq_test

"""Stimulated emission experiment with varying coherent drive
    Sequence => State prep -> repeated pi's at n+1 -> Coherent drive -> repeated pi's at n+1
    This can be used to calibrate the background at n+1
    
"""
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
readout = 'readout' #'clear'

if readout=='readout':
    disc = disc_file
else:
    disc = disc_file_opt

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc, lsb=True)
##################

dt = int(10e3)
T_max = int(0.8e6)
T_min = 250
t_vec = np.arange(T_min, T_max + dt/2, dt)

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, fstate_cal=0, n_pi_m=10, n_pi_n=30, avgs=20000):

    reset_time = int((f_state+0.5)*10e6)
    simulation = 0

    dt = int(25e3)
    T_max = int(1e6)
    T_min = 250
    t_vec = np.arange(T_min, T_max + dt/2, dt)

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        j = declare(int)
        t = declare(int)

        num = declare(int)

        I = declare(fixed)
        bit = declare(bool)

        bit_sum = declare(int)

        I_st = declare_stream()
        bit_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            assign(n, 0)

            with while_(n < avgs):

                assign(bit_sum, 0)

                wait(reset_time//4, 'storage_mode1')
                update_frequency('qubit_mode0', ge_IF[0])
                align('storage_mode1', 'qubit_mode0')
                ########################
                """Analytic SNAP pulses to create Fock states"""
                snap_seq_test(fock_state=f_state)
                ########################
                """Repeated pi pulses at n+1"""
                # ########################
                # ########################
                align('qubit_mode0','storage_mode1')
                update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[f_state+1])
                with for_(i, 0, i < 3, i+1):
                    align('qubit_mode0', "rr")
                    play("res_pi", 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("readout", "out1", "out2", bit, I=I)
                    assign(bit_sum, bit_sum + Cast.to_int(bit))
                    wait(500, "rr")
                #########################
                with if_(bit_sum==0):
                    assign(n, n+1)
                    #########################
                    update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[f_state])
                    align('qubit_mode0', "rr")
                    wait(t, "qubit_mode0")
                    play("res_pi", 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state(readout, "out1", "out2", bit, I=I)
                    save(I, I_st)
                    save(bit, bit_st)
                ########################

        with stream_processing():
            bit_st.boolean_to_int().buffer(avgs).map(FUNCTIONS.average()).save_all('res')
            I_st.buffer(avgs).map(FUNCTIONS.average()).save_all('I')

    qm = qmm.open_qm(config)
    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(exp, simulation_config)
        samples = job.get_simulated_samples()
        samples.con1.plot()
        result_handles = job.result_handles

    else:
        """To run the actual experiment"""
        job = qm.execute(exp, duration_limit=0, data_limit=0)
        print("Experiment execution Done")

        return job

for f_num in range(4, 5, 1):

    avgs = 200
    fock_number = f_num
    job = stim_em(coh_amp=0.0, coh_len=10, f_state=fock_number, fstate_cal=fock_number, n_pi_m=3, n_pi_n=30, avgs=avgs)

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    # time.sleep(5)
    I = result_handles.get('I').fetch_all()
    Q = result_handles.get('res').fetch_all()

    dt = int(25e3)
    T_max = int(1e6)
    T_min = 250
    t_vec = np.arange(T_min, T_max + dt/2, dt)
    #
    times = 4*t_vec + 3*7.3e3

    plt.figure()
    plt.plot(times[:len(Q)]/1e3, Q, '.--')
    plt.show()

    path = os.getcwd()
    data_path = os.path.join(path, "data")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_t1', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("res", data=Q)
        f.create_dataset("time", data=times[:len(Q)])