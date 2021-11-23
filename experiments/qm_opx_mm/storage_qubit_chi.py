from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi, disc_file
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
"""Qubit-Storage chi calibration with Ramsey """
def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_08_13_cavity_square.h5'

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

ramsey_freq = 100e3
omega = 2*np.pi*ramsey_freq

dt = 250

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 4
T_max = 75000
times = np.arange(T_min, T_max + dt/2, dt)
avgs = 1000
reset_time = int(3.5e6)
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

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

with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        assign(phi, 0)

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            update_frequency('qubit', ge_IF)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ##########################
            play("CW"*amp(0.4), "storage", duration=40)
            # play("CW"*amp(0.4), "storage", duration=alpha_awg_cal(1.143))
            # align("storage", "qubit")
            # play("res_pi"*amp(2.0), "qubit")
            # align("storage", "qubit")
            # play("CW"*amp(-0.4), "storage", duration=alpha_awg_cal(-0.58)) #249
            ##########################
            update_frequency('qubit', ge_IF+two_chi)
            align('storage', 'qubit')
            play("pi2", "qubit")
            wait(t, "qubit")
            frame_rotation_2pi(phi, "qubit") #2pi is already multiplied to the phase
            play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            assign(phi, phi + dphi)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        I_st.buffer(len(times)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    result_handles = job.result_handles

    # result_handles.wait_for_all_values()

    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    plt.figure()
    plt.plot( 4*times/1e3, res, '.-')
    plt.show()

    # job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_qubit_chi_ramsey', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Q", data=res)
        f.create_dataset("I", data=I)
        f.create_dataset("time", data= 4*times/1e3)
        f.create_dataset("ramsey_freq", data=ramsey_freq)
        f.create_dataset("qubit_freq", data=qubit_freq)
        f.create_dataset("two_chi", data=two_chi)
