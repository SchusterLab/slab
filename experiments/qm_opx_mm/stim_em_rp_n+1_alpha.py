"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt, st_self_kerr, storage_IF, storage_cal_file
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
reset_time = int(7.5e6)
simulation = 0

def alpha_awg_cal(alpha, cav_amp=1.0, cal_file=storage_cal_file[1]):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    with File(cal_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = omegas
    a_s = amps

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(a_s, o_s)

    omega_desired = transfer_fn(cav_amp)

    pulse_length = (alpha/omega_desired)
    """Returns time in units of 4ns for FPGA"""
    return abs(pulse_length)//4+1

t_chi = int(abs(0.5*1e9/two_chi[1])) #qubit rotates by pi in this time

opx_amp = 1.0

def snap_seq(fock_state=0):

    if fock_state==0:
        play("CW"*amp(0.0),'storage_mode1', duration=alpha_awg_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(0.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-0.0),'storage_mode1', duration=alpha_awg_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==1:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==2:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(0.497, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(1.133, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi[1])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(0.432, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==3:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(0.531, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(0.559, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi[1])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(0.946, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + 2*two_chi[1])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(0.358, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0])

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, n_pi_m=0, n_pi_n=30, avgs=20000):

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        res = declare(bool)
        bit4 = declare(bool)

        num = declare(int)

        I = declare(fixed)

        num_st = declare_stream()
        bit_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage_mode1')
            update_frequency('qubit_mode0', ge_IF[0])
            update_frequency('storage_mode1', storage_IF[1])
            ########################
            """Analytic SNAP pulses to create Fock states"""
            snap_seq(fock_state=f_state)
            align('storage_mode1', 'qubit_mode0')
            ########################
            ##########################
            """Resolved pi at f_target"""
            ##########################
            update_frequency("qubit_mode0", ge_IF[0]+f_state*two_chi[1])
            play("res_pi", "qubit_mode0")
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            with if_(res):
                align('qubit_mode0', 'rr')
                """If the qubit flips then we need to bring it back to |g>"""
                update_frequency('qubit_mode0', ge_IF[0])
                play('pi', 'qubit_mode0')
                save(f_state, num_st)
                update_frequency('storage_mode1', storage_IF[1] + (f_state*(f_state-1))*st_self_kerr)
                # ########################
                update_frequency('qubit_mode0', ge_IF[0] + (f_state+1)*two_chi[1])
                align('storage_mode1', 'rr', 'qubit_mode0')
                play('CW'*amp(coh_amp), 'storage_mode1', duration=coh_len)
                align('qubit_mode0','storage_mode1')
                ########################
                """Repeated pi pulses at n+1"""
                ########################
                with for_(i, 0, i < n_pi_n, i+1):
                    align('qubit_mode0', "rr")
                    play("res_pi", 'qubit_mode0')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", bit4, I=I)
                    save(bit4, bit_st)
                    wait(250, "rr")
            ########################

        with stream_processing():
            num_st.save_all('num')
            bit_st.boolean_to_int().buffer(n_pi_m+n_pi_n).save_all('bit')

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
analysis_params = { 'qubit_params': {'t1':118, 't2':203, 'nth':3e-2},
                    'cavity_params' : {'t1':1.352e3, 'nth':0.0006},
                    'readout_params':{'length':3.3, 'trigger':7.356, 'pi_pulse':3, 'g_inf':0.0402, 'e_inf':0.0394},
                    }

path = os.getcwd()

data_path = os.path.join(path, "data/stim_em_nobd")
filename = "analysis_params"
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, filename, suffix='.json'))


with open(seq_data_file, "w")as outfile:
    json.dump(analysis_params, outfile)


camp = list(np.round(np.arange(0.001, 0.0095, 0.001).tolist(), 6))
camp.extend(np.round(np.arange(0.01, 0.095, 0.01).tolist(), 6))
camp.extend(np.round(np.arange(0.1, 0.95, 0.1).tolist(), 6))
camp.append(0.0)

# camp = [0.01]

for f_num in range(4):

    for ii in range(len(camp)):

        l = 10
        avgs = 10000
        coh_amp = np.round(camp[ii], 4)
        fock_number = f_num
        job = stim_em(coh_amp=coh_amp, coh_len=l, f_state=fock_number, n_pi_m=0, n_pi_n=30, avgs=avgs)

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        time.sleep(10)
        num = result_handles.get('num').fetch_all()['value']
        bit = result_handles.get('bit').fetch_all()['value']

        p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

        print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))

        path = os.getcwd()

        data_path = os.path.join(path, "data/stim_em_nobd/n" + str(fock_number))

        filename = "stim_em_n" + str(fock_number) +"_camp_" + str(coh_amp)+"_len_"+str(l)

        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, filename, suffix='.h5'))

        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("amp", data=coh_amp)
            f.create_dataset("time", data=l)
            f.create_dataset("pi_m", data=0)
            f.create_dataset("pi_n", data=30)
            f.create_dataset("num", data=num)
            f.create_dataset("bit", data=bit)