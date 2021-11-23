"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, two_chi, disc_file_opt
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
"""Repeated parity measurements followed by coherent drive"""
def alpha_awg_cal(alpha, cav_amp=0.5):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx_mm\drive_calibration'

    fn_file = cal_path + '\\00000_2021_11_09_cavity_square_mode_2.h5'

    with File(fn_file, 'r') as f:
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
##################
simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)
t_chi = int(abs(0.5*1e9/two_chi[1]) //4 +1) #FPGA clock cycle unit, qubit rotates by pi in this time

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

""""Coherent drive to simulate the dark matter push"""
coh_len = 10

camp = np.round(np.arange(0.0001, 0.0009, 0.0001).tolist(), 6)
qm = qmm.open_qm(config)

avgs = 10000
reset_time = int(5e6)
simulation = 0

num_pi_pulses_m = 30 #need even number to bring the qubit back to 'g' before coherent drive
num_pi_pulses_n = 0

# camp = [0.5]

def photon_counting(a):

    with program() as repeated_parity:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        bit3 = declare(bool)
        I = declare(fixed)
        res = declare(bool)
        bit3_st = declare_stream()

        ###############
        # the sequence:
        ###############

        update_frequency("qubit_mode0", ge_IF[0])

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage_mode1')
            align('rr', 'storage_mode1')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            wait(reset_time//100, "qubit_mode0")
            align('storage_mode1', 'qubit_mode0')
            ########################
            play('CW'*amp(a), 'storage_mode1', duration=coh_len)
            ########################
            align('storage_mode1', 'qubit_mode0')

            with for_(i, 0, i < num_pi_pulses_m, i+1):
                # reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                play("pi2", "qubit_mode0") # unconditional
                wait(t_chi, "qubit_mode0")
                frame_rotation(np.pi, 'qubit_mode0') #
                play("pi2", "qubit_mode0")
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
                save(bit3, bit3_st)
                wait(250, "rr")

        with stream_processing():
            bit3_st.boolean_to_int().buffer(num_pi_pulses_m+num_pi_pulses_n).save_all('bit3')

    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(repeated_parity, simulation_config)
        samples = job.get_simulated_samples()
        samples.con1.plot()
        result_handles = job.result_handles

    else:

        job = qm.execute(repeated_parity, duration_limit=0, data_limit=0)

        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        bit3 = result_handles.get('bit3').fetch_all()['value']

        job.halt()

        path = os.getcwd()
        data_path = os.path.join(path, "data/")
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'photon_counting_parity_alpha', suffix='.h5'))
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("amp", data=a)
            f.create_dataset("time", data=coh_len*4)
            f.create_dataset("bit3", data=bit3)
            f.create_dataset("pi_m", data=num_pi_pulses_m)
            f.create_dataset("pi_n", data=num_pi_pulses_n)

camp = list(np.round(np.arange(0.001, 0.009, 0.001).tolist(), 6))
camp.extend(np.round(np.arange(0.1, 0.9, 0.1).tolist(), 6))
camp.append(0.0)
for cav_amp in camp[:]:
    print(cav_amp)
    photon_counting(cav_amp)
