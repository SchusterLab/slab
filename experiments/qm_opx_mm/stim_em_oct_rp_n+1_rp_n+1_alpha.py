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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, snap_seq, oct_to_opx_amp_test

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
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
##################

cav_t1s = [1444e3, 1444e3, 712e3, 527e3, 395e3, 313e3]

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, n_pi_m=10, n_pi_n=30, avgs=20000, phase=0.0):

    reset_time = int((f_state+0.5)*12e6)
    simulation = 0

    pulse_filename = './oct_pulses/g'+str(f_state)+'.h5'

    if f_state==0 or f_state==1 or f_state==2:
        pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)
    else:
        pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)//2

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
            """OCT pulses to create Fock states"""
            play("soct", 'storage_mode1', duration=pulse_len)
            play("qoct", 'qubit_mode0', duration=pulse_len)
            ########################
            """Repeated pi pulses at n+1"""
            # ########################
            align('qubit_mode0','storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0] + (f_state+1)*two_chi[1]+ (f_state+1)*(f_state+1-1)*two_chi_2)
            with for_(i, 0, i < n_pi_m, i+1):
                align('qubit_mode0', "rr")
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                # save(I, I_st)
                # save(bit, bit_st)
                assign(bit_sum, bit_sum + Cast.to_int(bit))
                wait(250, "rr")
            #########################
            with if_(bit_sum==0):
                update_frequency('storage_mode1', storage_IF[1] + (f_state*(f_state-1))*st_self_kerr)
                align('rr', 'qubit_mode0', 'storage_mode1')
                frame_rotation_2pi(phase, 'storage_mode1')
                play('CW'*amp(coh_amp), 'storage_mode1', duration=coh_len)
                align('rr', 'qubit_mode0', 'storage_mode1')
                wait(740, "rr") # to match the same time as the time between each repeated pi pulse sequence
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                save(I, I_st)
                save(bit, bit_st)
                wait(250, "rr") #same time as the time between each repeated pi pulse sequence
                update_frequency('qubit_mode0', ge_IF[0] + (f_state+1)*two_chi[1]+ (f_state+1)*(f_state+1-1)*two_chi_2)
                align('rr', 'qubit_mode0', 'storage_mode1')
                ########################
                """Repeated pi pulses at n+1"""
                ########################
                with for_(j, 0, j < n_pi_n, j+1):
                    align('qubit_mode0', "rr")
                    play("res_pi", 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                    save(I, I_st)
                    save(bit, bit_st)
                    wait(250, "rr")
                ########################

        with stream_processing():
            bit_st.boolean_to_int().buffer(n_pi_n+1).save_all('bit')
            I_st.buffer(n_pi_n+1).save_all('I')

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
analysis_params = { 'qubit_params': {'t1':123, 't2':230, 'nth':3e-2},
                    'cavity_params' : {'t1':1.412e3, 'nth':0.0006},
                    'readout_params':{'length':3.3, 'trigger':7.356, 'pi_pulse':3, 'g_inf':0.035, 'e_inf':0.0378},
                    }

# path = os.getcwd()
data_path = 'S:\\_Data\\20220222 - 3DMM - StimEm - OPX\\data\\stim_em_snap_rp_n+1\\20220224\\'
# data_path = os.path.join(path, "data/stim_em_snap_rp_n+1/20220222")
filename = "analysis_params"
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, filename, suffix='.json'))


with open(seq_data_file, "w")as outfile:
    json.dump(analysis_params, outfile)

camp = list(np.round(np.arange(0.001, 0.0095, 0.002).tolist(), 6))
camp.extend(np.round(np.arange(0.01, 0.10, 0.005).tolist(), 6))
camp.extend(np.round(np.arange(0.1, 0.95, 0.2).tolist(), 6))
camp.append(0.0)

camp = [0.0, 0.005, 0.01, 0.05, 0.08, 0.1]

phase = [0, 0.25, 0.5, 1.0]

for kk in range(3):

    ph = phase[kk+1]

    for ii in range(len(camp)):

        for f_num in range(7):

            l = 10
            avgs = 15000
            coh_amp = np.round(camp[ii], 4)
            fock_number = f_num
            job = stim_em(coh_amp=coh_amp, coh_len=l, f_state=fock_number, n_pi_m=3, n_pi_n=30, avgs=avgs, phase=ph)

            result_handles = job.result_handles
            result_handles.wait_for_all_values()
            time.sleep(5)
            I = result_handles.get('I').fetch_all()['value']
            Q = result_handles.get('bit').fetch_all()['value']
            alpha = opx_amp_to_alpha(cav_amp=coh_amp, cav_len=4*l)

            print(fock_number, len(Q))

            path = os.getcwd()

            data_path = 'S:\\_Data\\20220222 - 3DMM - StimEm - OPX\\data\\stim_em_snap_rp_n+1\\20220224\\n'+ str(fock_number)

            filename = "stim_em_n" + str(fock_number) +"_camp_" + str(coh_amp)+"_len_"+str(l)

            seq_data_file = os.path.join(data_path,
                                         get_next_filename(data_path, filename, suffix='.h5'))

            print(seq_data_file)
            with File(seq_data_file, 'w') as f:
                f.create_dataset("amp", data=coh_amp)
                f.create_dataset("time", data=l)
                f.create_dataset("alpha", data=alpha)
                f.create_dataset("pi_m", data=0)
                f.create_dataset("pi_n", data=31)
                f.create_dataset("I", data=I)
                f.create_dataset("bit", data=Q)