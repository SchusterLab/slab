"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi, disc_file
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
def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_7_14_cavity_square.h5'

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

t_chi = int(abs(0.5*1e9/two_chi) //4 +1) #FPGA clock cycle unit, qubit rotates by pi in this time

simulation_config = SimulationConfig(
    duration=30000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

""""Coherent drive to simulate the dark matter push"""
coh_len = 100
coh_amp = 0.00

camp = np.round(np.arange(0.1, 0.9, 0.1).tolist(), 6)
qm = qmm.open_qm(config)

avgs = 25000
reset_time = int(3.75e6)
simulation = 0

num_pi_pulses_m = 30 #need even number to bring the qubit back to 'g' before coherent drive
num_pi_pulses_n = 0

# camp = [0.0]

for a in camp:

    coh_amp = a

    with program() as repeated_parity:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        t = declare(int) #array of time delays
        bit3 = declare(bool)
        I = declare(fixed)

        bit3_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage')
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ########################

            play('CW'*amp(coh_amp), 'storage', duration=coh_len)
            ########################
            align('storage', 'qubit')

            with for_(i, 0, i < num_pi_pulses_m, i+1):
                wait(1000//4, "rr")
                # reset_frame('qubit')
                align("qubit", "rr", 'jpa_pump')
                play("pi2", "qubit") # unconditional
                wait(t_chi, "qubit")
                frame_rotation(np.pi, 'qubit') #
                play("pi2", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
                save(bit3, bit3_st)

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
            f.create_dataset("amp", data=coh_amp)
            f.create_dataset("time", data=coh_len*4)
            f.create_dataset("bit3", data=bit3)
            f.create_dataset("pi_m", data=num_pi_pulses_m)
            f.create_dataset("pi_n", data=num_pi_pulses_n)