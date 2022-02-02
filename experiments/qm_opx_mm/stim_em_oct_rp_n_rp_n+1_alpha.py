"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, st_self_kerr, storage_IF, storage_cal_file, opt_len
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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha

"""Stimulated emission experiment with varying coherent drive
    Sequence => State prep -> BD -> Coherent drive -> repeated pi's at n+1
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
simulation = 0

w_plus = [(1.0, opt_len)]
w_minus = [(-1.0, opt_len)]
w_zero = [(0.0, opt_len)]

b = (30.0/180)*np.pi
w_plus_cos = [(np.cos(b), opt_len)]
w_minus_cos = [(-np.cos(b), opt_len)]
w_plus_sin = [(np.sin(b), opt_len)]
w_minus_sin = [(-np.sin(b), opt_len)]
config['integration_weights']['clear_integW1']['cosine'] = w_plus_cos
config['integration_weights']['clear_integW1']['sine'] = w_minus_sin
config['integration_weights']['clear_integW2']['cosine'] = w_plus_sin
config['integration_weights']['clear_integW2']['sine'] = w_plus_cos
config['integration_weights']['clear_integW3']['cosine'] = w_minus_sin
config['integration_weights']['clear_integW3']['sine'] = w_minus_cos

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, n_pi_m=0, n_pi_n=30, avgs=20000):

    pulse_len = oct_to_opx_amp(opx_config=config, fock_state=f_state)//2

    reset_time = int((f_state/2+1)*7.5e6)

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes

        num = declare(int)

        I = declare(fixed)
        bit = declare(bool)

        I_st = declare_stream()
        bit_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0])
            update_frequency('storage_mode1', storage_IF[1])
            align('storage_mode1', 'qubit_mode0')
            ########################
            """Analytic SNAP pulses to create Fock states"""
            play("soct", 'storage_mode1', duration=pulse_len)
            play("qoct", 'qubit_mode0', duration=pulse_len)
            ########################
            """Repeated pi pulses at n"""
            ########################
            align('qubit_mode0','storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0] + (f_state)*two_chi[1])
            with for_(i, 0, i < n_pi_m, i+1):
                align('qubit_mode0', 'rr')
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                # measure("clear", "rr", None,
                #                 dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                #                 dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))
                save(I, I_st)
                save(bit, bit_st)
                wait(250, 'rr')
            #########################
            update_frequency('storage_mode1', storage_IF[1] + (f_state*(f_state-1))*st_self_kerr)
            update_frequency('qubit_mode0', ge_IF[0] + (f_state+1)*two_chi[1])
            align('rr', 'qubit_mode0', 'storage_mode1')
            play('CW'*amp(coh_amp), 'storage_mode1', duration=coh_len)
            align('qubit_mode0','storage_mode1')
            ########################
            """Repeated pi pulses at n+1"""
            ########################
            with for_(i, 0, i < n_pi_n, i+1):
                align('qubit_mode0', 'rr')
                play('res_pi', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                # measure("clear", "rr", None,
                #                 dual_demod.full('clear_integW1', 'out1', 'clear_integW3', 'out2', I),
                #                 dual_demod.full('clear_integW2', 'out1', 'clear_integW1', 'out2', Q))
                save(I, I_st)
                save(bit, bit_st)

                wait(250, 'rr')
            ########################

        with stream_processing():
            bit_st.boolean_to_int().buffer(n_pi_m+n_pi_n).save_all('bit')
            I_st.buffer(n_pi_m+n_pi_n).save_all('I')
            # Q_st.buffer(n_pi_m+n_pi_n).save_all('Q')

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

camp = list(np.round(np.arange(0.001, 0.0095, 0.001).tolist(), 6))
camp.extend(np.round(np.arange(0.01, 0.095, 0.01).tolist(), 6))
camp.extend(np.round(np.arange(0.1, 0.95, 0.1).tolist(), 6))
camp.append(0.0)

# camp = [0.01]

for f_num in [0, 1, 3, 6]:

    for ii in range(len(camp)):

        l = 10
        avgs = 20000
        coh_amp = np.round(camp[ii], 4)
        fock_number = f_num
        job = stim_em(coh_amp=coh_amp, coh_len=l, f_state=fock_number, n_pi_m=10, n_pi_n=30, avgs=avgs)

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        time.sleep(5)
        I = result_handles.get('I').fetch_all()['value']
        Q = result_handles.get('bit').fetch_all()['value']
        alpha = opx_amp_to_alpha(cav_amp=coh_amp, cav_len=4*l)
        #
        path = os.getcwd()

        data_path = os.path.join(path, "data/stim_em_oct_rp/20220122/n" + str(fock_number))

        filename = "stim_em_n" + str(fock_number) +"_camp_" + str(coh_amp)+"_len_"+str(l)

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, filename, suffix='.h5'))
        #
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("amp", data=coh_amp)
            f.create_dataset("time", data=l)
            f.create_dataset("alpha", data=alpha)
            f.create_dataset("pi_m", data=10)
            f.create_dataset("pi_n", data=30)
            f.create_dataset("I", data=I)
            f.create_dataset("bit", data=Q)

import json
analysis_params = { 'qubit_params': {'t1':129, 't2':201, 'nth':3e-2},
                    'cavity_params' : {'t1':1.352e3, 'nth':0.0006},
                    'readout_params':{'length':3.3, 'trigger':7.356, 'pi_pulse':3, 'g_inf':0.0344, 'e_inf':0.0348},
                    }
path = os.getcwd()

data_path = os.path.join(path, "data/stim_em_oct_rp/20220122")
filename = "analysis_params"
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, filename, suffix='.json'))


with open(seq_data_file, "w")as outfile:
    json.dump(analysis_params, outfile)
