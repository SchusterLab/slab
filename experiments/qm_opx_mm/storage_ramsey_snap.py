"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, storage_freq, two_chi, disc_file_opt    
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from h5py import File
import os, scipy
from slab.dataanalysis import get_next_filename

"""Storage cavity ramsey experiment using analytic SNAP pulses to create |0>+|1> """
def alpha_awg_cal(alpha, cav_amp=1.0):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx_mm\drive_calibration'

    fn_file = cal_path + '\\00000_2021_11_24_cavity_square_mode_2.h5'

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

ramsey_freq = 20e3
omega = 2*np.pi*ramsey_freq

dt = 2500
T_min = 4
T_max = 300000
t_vec = np.arange(T_min, T_max + dt/2, dt)

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

avgs = 1000
reset_time = int(7.5e6)
simulation = 0 #1 to simulate the pulses

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
opx_amp = 1.0
with program() as storage_t2:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    res = declare(bool)
    phi = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        assign(phi, 0)

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            # update_frequency('qubit_mode0', ge_IF)
            wait(reset_time// 4, 'storage_mode1')# wait for the storage to relax, several T1s
            ########################
            play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(1.143, cav_amp=opx_amp))
            align('storage_mode1', 'qubit_mode0')
            play("res_pi"*amp(2.0), 'qubit_mode0')
            align('storage_mode1', 'qubit_mode0')
            play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(-0.58, cav_amp=opx_amp))
            ########################
            wait(t, 'storage_mode1')
            frame_rotation_2pi(phi, 'storage_mode1')
            ########################
            play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_awg_cal(1.143, cav_amp=opx_amp))
            align('storage_mode1', 'qubit_mode0')
            play("res_pi"*amp(2.0), 'qubit_mode0')
            align('storage_mode1', 'qubit_mode0')
            play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_awg_cal(-0.58, cav_amp=opx_amp))
            ########################
            align('storage_mode1', 'qubit_mode0')
            play("res_pi", 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            assign(phi, phi + dphi)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
        I_st.buffer(len(t_vec)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(storage_t2, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(storage_t2, duration_limit=0, data_limit=0)
    print ("Execution done")

    result_handles = job.result_handles

    # result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    #
    # times = 4*t_vec/1e3
    #
    # plt.figure()
    # plt.plot(times, res, '.-')
    # plt.show()

    #
    # job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'storage_ramsey', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("time", data=times)
    #     f.create_dataset("ramsey_freq", data=ramsey_freq)
    #     f.create_dataset("cavity_freq", data=storage_freq)
