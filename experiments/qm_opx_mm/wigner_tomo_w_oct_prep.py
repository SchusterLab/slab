"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, two_chi, disc_file_opt, storage_cal_file, disc_file
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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, oct_to_opx_amp_test

"""Wigner tomography of the cavity state with binary decomposition"""

t_chi = int(abs(0.5*1e9/two_chi[1]//4)) #qubit rotates by pi in this time

avgs = 2000
reset_time = int(3*7.5e6)
simulation = 0

def wigner_alpha_awg_amp(wigner_pt_file = None, cavity_cal_file = None, cav_len=200):
    cal_path = 'C:\\_Lib\\python\\slab\\experiments\\qm_opx_mm'
    wigner_pt_file = cal_path + '\\wigner_function_points\\'+ wigner_pt_file

    with File(wigner_pt_file, 'r') as f:
        xs = np.array(f['alphax'])
        ys = np.array(f['alphay'])

    tom_amp = np.sqrt(xs ** 2 + ys ** 2)
    tom_phase = np.arctan2(ys, xs)

    omega_desired = tom_amp/cav_len #keeping the cavity length fixed
    cavity_cal_file = cal_path + '\\drive_calibration\\'+ cavity_cal_file
    with File(cavity_cal_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, -omegas)
    amps = np.append(amps, -amps)
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)
    o_s = omegas
    a_s = amps
    max_interp_index = np.argmax(omegas)

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(o_s, a_s)
    awg_amps = []
    for i in range(len(omega_desired)):
        # if frequency greater than calibrated range, assume a proportional relationship (high amp)
        if np.abs(omega_desired[i]) > omegas[max_interp_index]:
            awg_amps.append(omega_desired[i] * amps[max_interp_index] / omegas[max_interp_index])
            # output_amps.append(amps[max_interp_index])
        else:  # otherwise just use the interpolated transfer function
            awg_amps.append(transfer_fn((omega_desired[i])))

    # awg_amp = transfer_fn(omega_desired)

    """Returns time in units of 4ns for FPGA"""

    x_awg = awg_amps * np.cos(tom_phase)
    y_awg = awg_amps * np.sin(tom_phase)

    return x_awg, y_awg, cav_len

cav_file = '00000_2021_12_28_cavity_square_mode_2.h5'
wigner_file = '00000_wigner_points_nmax_3_nexpt_50_kappa_1pt0_gauss.h5'

ax, ay, cav_len = wigner_alpha_awg_amp(wigner_pt_file=wigner_file, cavity_cal_file=cav_file)

n_disp = len(ax)

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

readout = 'readout' #'clear'

if readout=='readout':
    disc = disc_file
else:
    disc = disc_file_opt

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc, lsb=True)

fock_state = 3
pulse_filename = './oct_pulses/g'+str(fock_state)+'.h5'

reset_time = int((fock_state+0.5)*10e6)

if fock_state < 3:
    pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)
else:
    pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse_filename)//2

with program() as wigner_tomo:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      #
    res = declare(bool)
    I = declare(fixed)
    x = declare(fixed)
    y = declare(fixed)
    axqua = declare(fixed, value=[ax[i] for i in range(len(ax))])
    ayqua = declare(fixed, value=[ay[i] for i in range(len(ay))])

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(i, 0, i < n_disp, i + 1):

            wait(reset_time// 4, 'storage_mode1')# wait for the storage to relax, several T1s
            align('storage_mode1', 'qubit_mode0')
            play("soct", 'storage_mode1', duration=pulse_len)
            play("qoct", 'qubit_mode0', duration=pulse_len)
            align('storage_mode1', 'qubit_mode0', 'rr')

            # measure and do a qubit reset if the qubit is in e
            discriminator.measure_state(readout, "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            align('storage_mode1', 'qubit_mode0', 'rr')

            """Do a dispacement drive of the cavity in the phase space"""
            wait(5, "storage_mode1")
            play("CW"*amp(axqua[i], 0, ayqua[i], 0), "storage_mode1", duration=cav_len//4)
            align("storage_mode1", "qubit_mode0")

            # parity
            play("pi2", "qubit_mode0") # unconditional
            wait(t_chi, "qubit_mode0")
            frame_rotation(np.pi, 'qubit_mode0') #
            play("pi2", "qubit_mode0")
            align('qubit_mode0', 'rr',)
            discriminator.measure_state(readout, "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(n_disp).average().save('res')
        I_st.buffer(n_disp).average().save('I')

qm = qmm.open_qm(config)
if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(wigner_tomo, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    result_handles = job.result_handles

else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(wigner_tomo, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    # #
    # # job.halt()
    cal_path = 'C:\\_Lib\\python\\slab\\experiments\\qm_opx_mm'
    wigner_file = '00000_wigner_points_nmax_3_nexpt_50_kappa_1pt0_gauss.h5'
    wigner_pt_file = cal_path + '\\wigner_function_points\\'+ wigner_file

    with File(wigner_pt_file, 'r') as f:
        xs = np.array(f['alphax'])
        ys = np.array(f['alphay'])
        f.close()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'wigner_tomo_oct', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("res", data=res)
        f.create_dataset("alphax", data=xs)
        f.create_dataset("alphay", data=ys)