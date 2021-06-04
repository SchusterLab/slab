"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os, scipy
from slab.dataanalysis import get_next_filename
"""Binary decomposition"""
def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_05_20_cavity_square.h5'

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

t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time
cav_len = 10
cav_amp = 0.4 # 0.08

avgs = 1000
reset_time = int(3.5e6)
simulation = 0
snap = False
simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(5000//4, "jpa_pump")
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

with program() as binary_decomposition:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    bit1_st = declare_stream()
    bit2_st = declare_stream()
    I_st = declare_stream()
    n_st = declare_stream()
    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):
        wait(reset_time//4, 'storage')
        update_frequency('qubit', ge_IF)
        align('storage', 'rr', 'jpa_pump', 'qubit')
        active_reset(biased_th_g_jpa)
        align('storage', 'rr', 'jpa_pump', 'qubit')
        with if_(snap):
            play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(1.143))
            align("storage", "qubit")
            play("res_pi"*amp(2.0), "qubit")
            align("storage", "qubit")
            play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(-0.58)) #249
        with else_():
            play('CW'*amp(cav_amp), 'storage', duration=cav_len)
        align('storage', 'qubit')
        play("pi2", "qubit") # unconditional
        wait(t_chi//4, "qubit")
        frame_rotation(np.pi, 'qubit') #
        play("pi2", "qubit")
        align('qubit', 'rr', 'jpa_pump')
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", res, I=I)
        save(res, bit1_st)

        reset_frame("qubit")
        wait(1000//4, "rr")
        align("qubit", "rr", 'jpa_pump')

        play("pi2", "qubit") # unconditional
        wait(t_chi//4//2-3, "qubit") # subtracted 3 to make the simulated waveforms accurate
        with if_(res==0):
            frame_rotation(np.pi, 'qubit')
            play("pi2", "qubit")
        with else_():
            frame_rotation(3/2*np.pi, 'qubit')
            play("pi2", "qubit")
        align('qubit', 'rr', 'jpa_pump')
        play('pump_square', 'jpa_pump')
        discriminator.measure_state("clear", "out1", "out2", res, I=I)
        save(res, bit2_st)

    with stream_processing():
        bit1_st.boolean_to_int().save_all('bit1')
        bit2_st.boolean_to_int().save_all('bit2')

qm = qmm.open_qm(config)
if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(binary_decomposition, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    result_handles = job.result_handles

else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(binary_decomposition, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    bit1 = result_handles.get('bit1').fetch_all()['value']
    bit2 = result_handles.get('bit2').fetch_all()['value']

    num = bit1 + 2*bit2

    p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

    print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))
    job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'binary_decomp', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("p_cav", data=p_cav)
    #     f.create_dataset("amp", data=cav_amp)
    #     f.create_dataset("time", data=cav_len*4)
