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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq, snap_seq_test

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

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)
##################

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, fstate_cal=0, n_pi_m=10, n_pi_n=30, avgs=20000):

    reset_time = int(10e6)
    simulation = 0

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        j = declare(int)

        num = declare(int)

        I = declare(fixed)
        bit = declare(bool)

        bit_sum = declare(int)

        I_st = declare_stream()
        bit_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            assign(bit_sum, 0)

            wait(reset_time//4, 'storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0])
            update_frequency('storage_mode1', storage_IF[1])
            align('storage_mode1', 'qubit_mode0')
            ########################
            """Analytic SNAP pulses to create Fock states"""
            snap_seq_test(fock_state=f_state)
            ########################
            """Repeated pi pulses at n+1"""
            # ########################
            align('qubit_mode0','storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[f_state+1])
            with for_(i, 0, i < n_pi_m, i+1):
                align('qubit_mode0', "rr")
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("readout", "out1", "out2", bit, I=I)
                # save(I, I_st)
                # save(bit, bit_st)
                assign(bit_sum, bit_sum + Cast.to_int(bit))
                wait(500, "rr")
            #########################
            update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fstate_cal])
            with if_(bit_sum==0):
                align('qubit_mode0', "rr")
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("readout", "out1", "out2", bit, I=I)
                save(I, I_st)
                save(bit, bit_st)
                ########################

        with stream_processing():
            bit_st.boolean_to_int().buffer(1).save_all('bit')
            I_st.buffer(1).save_all('I')

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

data = {}

data_path = 'S:\\_Data\\20220222 - 3DMM - StimEm - OPX\\data\\stim_em_snap_rp_n+1\\20220226\\state_prep_fid_oct_matrix.json'

import json
for f_num in range(3, 5, 1):

    data['n%i'%(f_num)] = {}

    n_list = np.arange(max(0, f_num-2), f_num+3, 1)

    for n in (n_list):
        data['n%i'%(f_num)]['n%i'%(n)] = {}
        avgs = 10000
        fock_number = f_num
        job = stim_em(coh_amp=0.0, coh_len=10, f_state=fock_number, fstate_cal=n, n_pi_m=3, n_pi_n=30, avgs=avgs)

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        time.sleep(5)
        I = result_handles.get('I').fetch_all()['value']
        Q = result_handles.get('bit').fetch_all()['value']

        Q = [int(q) for q in Q.flatten()]

        data['n%i'%(f_num)]['n%i'%(n)] = Q

        print(len(Q), fock_number, n, np.mean(Q), np.median(Q))

with open(data_path, "w")as outfile:
    json.dump(data, outfile, indent=3)
