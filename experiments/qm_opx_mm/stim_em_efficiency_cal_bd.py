"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, st_self_kerr, storage_IF, storage_cal_file, opt_len, two_chi_2
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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq

"""Stimulated emission experiment to estimate the state preparation 
efficiency with a coherent drive followed by alternate sequence of resolved pi and parity
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
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
##################
# w_plus = [(1.0, opt_len)]
# w_minus = [(-1.0, opt_len)]
# w_zero = [(0.0, opt_len)]
#
# b = (30.0/180)*np.pi
# w_plus_cos = [(np.cos(b), opt_len)]
# w_minus_cos = [(-np.cos(b), opt_len)]
# w_plus_sin = [(np.sin(b), opt_len)]
# w_minus_sin = [(-np.sin(b), opt_len)]
# config['integration_weights']['clear_integW1']['cosine'] = w_plus_cos
# config['integration_weights']['clear_integW1']['sine'] = w_minus_sin
# config['integration_weights']['clear_integW2']['cosine'] = w_plus_sin
# config['integration_weights']['clear_integW2']['sine'] = w_plus_cos
# config['integration_weights']['clear_integW3']['cosine'] = w_minus_sin
# config['integration_weights']['clear_integW3']['sine'] = w_minus_cos


t_chi = int((abs(0.5*1e9/two_chi[1]))) # in FPGA clock cycles, qubit rotates by pi in this time

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, n_pi_m=10, n_pi_n=30, avgs=20000):

    reset_time = int((f_state+0.5)*7.5e6)
    simulation = 0

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)
        I = declare(fixed)
        bit1 = declare(bool)
        bit2 = declare(bool)
        num = declare(int)

        num_st = declare_stream()
        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage_mode1')
            align('storage_mode1', 'qubit_mode0')
            ########################
            """Known displacement drive on the cavity"""
            play('CW'*amp(1.0), 'storage_mode1', duration=18)
            ########################
            """Repeated BD"""
            ########################
            align('qubit_mode0','storage_mode1')
            with for_(i, 0, i < n_pi_m, i+1):
                wait(250, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                play("pi2", 'qubit_mode0') # unconditional
                wait(t_chi//4, 'qubit_mode0')
                frame_rotation(np.pi, 'qubit_mode0') #
                play("pi2", 'qubit_mode0')
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
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit2, I=I)

                reset_frame('qubit_mode0')
                wait(250, "rr")
                align('qubit_mode0', 'rr')
                play('pi', 'qubit_mode0', condition=bit2)

                assign(num, Cast.to_int(bit1) + 2*Cast.to_int(bit2))
                save(num, num_st)

        #########################

        with stream_processing():
            num_st.buffer(n_pi_m+n_pi_n).save_all('num')

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

import json
analysis_params = { 'qubit_params': {'t1':118, 't2':209, 'nth':3e-2},
                    'cavity_params' : {'t1':1.352e3, 'nth':0.0006},
                    'readout_params':{'length':3.3, 'trigger':7.356, 'pi_pulse':3, 'g_inf':0.0288, 'e_inf':0.0392},
                    }

path = os.getcwd()

data_path = os.path.join(path, "data/stim_em_snap_rp/20220130")
filename = "analysis_params"
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, filename, suffix='.json'))


with open(seq_data_file, "w")as outfile:
    json.dump(analysis_params, outfile)

# camp = list(np.round(np.arange(0.001, 0.0095, 0.002).tolist(), 6))
# camp.extend(np.round(np.arange(0.01, 0.10, 0.005).tolist(), 6))
# camp.extend(np.round(np.arange(0.1, 0.95, 0.2).tolist(), 6))
# camp.append(0.0)

camp = [0.0]

for ii in range(len(camp)):

    l = 10
    avgs = 100000
    coh_amp = np.round(camp[ii], 4)
    job = stim_em(coh_amp=coh_amp, coh_len=l, f_state=3, n_pi_m=10, n_pi_n=0, avgs=avgs)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    time.sleep(5)
    Q = result_handles.get('num').fetch_all()['value']
    alpha = opx_amp_to_alpha(cav_amp=1.0, cav_len=4*18)

    path = os.getcwd()

    data_path = os.path.join(path, "data/stim_em_snap_rp/20220130/")

    filename = "stim_em_repeated_bd"

    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, filename, suffix='.h5'))

    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("amp", data=coh_amp)
        f.create_dataset("time", data=l)
        f.create_dataset("alpha", data=alpha)
        f.create_dataset("pi_m", data=10)
        f.create_dataset("pi_n", data=0)
        f.create_dataset("bit", data=Q)