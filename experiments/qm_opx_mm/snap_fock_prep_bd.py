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
import scipy
import os
from slab.dataanalysis import get_next_filename
"""Using analytic SNAP pulses to create Fock states in the storage cavity followed by Binary Decomposition"""

def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_08_17_cavity_square.h5'

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
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

avgs = 1000
reset_time = int(3.75e6)
simulation = 0
t_chi = int((abs(0.5*1e9/two_chi))//4 + 1) # in FPGA clock cycles, qubit rotates by pi in this time
opx_amp = 0.40

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    I  = declare(fixed)

    wait(1000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

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

def snap_seq(fock_state=0):

    if fock_state==0:
        play("CW"*amp(0.0), "storage", duration=alpha_awg_cal(1.143))
        align("storage", "qubit")
        play("res_pi"*amp(0.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-0.0), "storage", duration=alpha_awg_cal(-0.58))

    elif fock_state==1:
        play("CW"*amp(opx_amp), "storage", duration=alpha_awg_cal(1.143))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-opx_amp), "storage", duration=alpha_awg_cal(-0.58))

    elif fock_state==2:
        play("CW"*amp(opx_amp), "storage", duration=alpha_awg_cal(0.497))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-opx_amp), "storage", duration=alpha_awg_cal(1.133))
        update_frequency("qubit", ge_IF + two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(opx_amp), "storage", duration=alpha_awg_cal(0.432))
        update_frequency("qubit", ge_IF)

    elif fock_state==3:
        play("CW"*amp(opx_amp), "storage", duration=alpha_awg_cal(0.531))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-opx_amp), "storage", duration=alpha_awg_cal(0.559))
        update_frequency("qubit", ge_IF + two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(opx_amp), "storage", duration=alpha_awg_cal(0.946))
        update_frequency("qubit", ge_IF + 2*two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-opx_amp), "storage", duration=alpha_awg_cal(0.358))
        update_frequency("qubit", ge_IF)

def fock_prep(f_target=1):

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)        # Averaging
        num = declare(int)
        bit1 = declare(bool)
        bit2 = declare(bool)
        I = declare(fixed)

        num_st = declare_stream()
        ###############
        # the sequence:
        ###############
        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            update_frequency('qubit', ge_IF)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ##########################
            snap_seq(fock_state=f_target)
            ##########################
            align('storage', 'qubit')

            # wait(int(25e3), 'qubit')
            """BD starts here"""

            play("pi2", "qubit") # unconditional
            wait(t_chi, "qubit")
            frame_rotation(np.pi, 'qubit') #
            play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", bit1, I=I)

            reset_frame("qubit")
            wait(1000//4, "rr")
            align("qubit", "rr", 'jpa_pump')

            play("pi2", "qubit") # unconditional
            wait(t_chi//2-3, "qubit") # subtracted 3 to make the simulated waveforms accurate
            with if_(bit1==0):
                frame_rotation(np.pi, 'qubit')
                play("pi2", "qubit")
            with else_():
                frame_rotation(3/2*np.pi, 'qubit')
                play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
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

        result_handles = job.result_handles

        result_handles.wait_for_all_values()
        num = result_handles.get('num').fetch_all()['value']

    return num

num = fock_prep(f_target=3)

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