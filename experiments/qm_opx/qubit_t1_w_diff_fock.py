"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi, disc_file, st_self_kerr, storage_IF
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
"""Qubit decay as a function of different Fock state in the storage cavity"""
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


t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time

def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_6_14_cavity_square.h5'

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
    return abs(pulse_length)//4

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    I = declare(fixed)

    wait(1000//4, "rr")
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
        play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(1.143))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(-0.58))

    elif fock_state==2:
        play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(0.497))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(1.133))
        update_frequency("qubit", ge_IF + two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(0.432))
        update_frequency("qubit", ge_IF)

    elif fock_state==3:
        play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(0.531))
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(0.559))
        update_frequency("qubit", ge_IF + two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(0.946))
        update_frequency("qubit", ge_IF + 2*two_chi)
        align("storage", "qubit")
        play("res_pi"*amp(2.0), "qubit")
        align("storage", "qubit")
        play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(0.358))
        update_frequency("qubit", ge_IF)

dt = 1000
T_max = 50000
T_min = 4
times = np.arange(T_min, T_max + dt/2, dt)

dl = 4
l_max = 250
l_min = 10
lengths = np.arange(l_min, l_max + dl/2, dl)

avgs = 1000
reset_time = int(3.5e6)
simulation = 0
cav_amp = 0.4

with program() as exp:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    t = declare(int)      # time
    l = declare(int)

    bit = declare(bool)

    num = declare(int)

    I = declare(fixed)

    bit_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(l, l_min, l < l_max + dl/2, l+dl):

            with for_(t, T_min, t < T_max + dt/2, t+dt):

                wait(reset_time//4, 'storage')
                align('storage', 'rr', 'jpa_pump', 'qubit')
                active_reset(biased_th_g_jpa)
                align('storage', 'rr', 'jpa_pump', 'qubit')
                ########################
                """Coherent drive on the storage"""
                play("CW"*amp(cav_amp), "storage", duration=l)
                align('storage', 'qubit')
                ########################
                """Qubit T1 sequence"""
                ########################
                play("pi", "qubit")
                wait(t, "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit, I=I)
                save(bit, bit_st)
                ########################

    with stream_processing():
        bit_st.boolean_to_int().buffer(len(lengths), len(times)).average().save('bit')

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

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    bit = result_handles.get('bit').fetch_all()

    plt.figure()
    plt.pcolormesh(4*times/1e3, 4*lengths/1e3, bit, cmap='RdBu', shading='auto')
    plt.colorbar()
    plt.show()

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'qubit_t1_alpha', suffix='.h5'))

    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("cav_amp", data=cav_amp)
        f.create_dataset("cav_len", data=4*lengths/1e3)
        f.create_dataset("wait_time", data=4*times/1e3)
        f.create_dataset("bit", data=bit)
