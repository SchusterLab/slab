from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa
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

"""Wigner tomography of the cavity state with binary decomposition"""

t_chi = int(0.5*1e9/1.118e6) #qubit rotates by pi in this time
cav_len = 5000
cav_amp = 0#0.04 # 0.08

avgs = 500
reset_time = int(5e6)
simulation = 0

def wigner_alpha_awg_amp(wigner_pt_file = None, cavity_cal_file = None, cav_len=2000):
    cal_path = 'C:\\_Lib\\python\\slab\\experiments\\qm_opx'
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

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(o_s, a_s)

    awg_amp = transfer_fn(omega_desired)

    """Returns time in units of 4ns for FPGA"""

    x_awg = awg_amp * np.cos(tom_phase)
    y_awg = awg_amp * np.sin(tom_phase)

    return x_awg, y_awg, cav_len


cav_file = '00000_2021_05_10_cavity_square.h5'
wigner_file = '00000_wigner_points_nmax_3_nexpt_25_kappa_1pt0_gauss.h5'

ax, ay, cav_len = wigner_alpha_awg_amp(wigner_pt_file=wigner_file, cavity_cal_file=cav_file)

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
    x =declare(fixed)
    y = declare(fixed)
    bit1 = declare(bool)
    bit2 = declare(bool)
    num =declare(int)

    # bit1_st = declare_stream()
    # bit2_st = declare_stream()

    wigner_st = declare_stream()
    x_st = declare_stream()
    y_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):


        with for_(x, x_min, x < x_max + dx/2, x +dx):

            with for_(y, y_min, y < y_max + dy/2, y +dy):
                reset_frame('storage')
                wait(reset_time//4, 'storage')
                # align('storage', 'rr', 'jpa_pump', 'qubit')
                # active_reset(biased_th_g_jpa)
                # align('storage', 'rr', 'jpa_pump', 'qubit')
                play('CW'*amp(cav_amp), 'storage', duration=cav_len)
                align('storage', 'qubit')
                play("pi2", "qubit") # unconditional
                wait(t_chi//4, "qubit")
                frame_rotation(np.pi, 'qubit') #
                play("pi2", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit1, I=I)

                # save(res, bit1_st)

                reset_frame("qubit")
                wait(1000//4, "rr")
                align("qubit", "rr", 'jpa_pump')

                play("pi2", "qubit") # unconditional
                wait(t_chi//4//2, "qubit")
                with if_(res==0):
                    frame_rotation(np.pi, 'qubit')
                    play("pi2", "qubit")
                with else_():
                    frame_rotation(3/2*np.pi, 'qubit')
                    play("pi2", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit2, I=I)
                # save(res, bit2_st)

                align("storage", "rr", 'jpa_pump')

                """Do a dispacement drive of the cavity in the phase space"""

                wait(1000//4, "storage")
                play("CW"*amp(x, 0, 0, y), "storage", duration=250)
                align("storage", "qubit")

                reset_frame("qubit")

                # parity
                play("pi2", "qubit") # unconditional
                wait(t_chi//4, "qubit")
                frame_rotation(np.pi, 'qubit') #
                play("pi2", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                # assign(num, bit1 + 2*bit2)
                with if_(bit1==0 and bit2==0):
                    save(res, wigner_st)
                    save(x, x_st)
                    save(y, y_st)

    with stream_processing():
        # bit1_st.boolean_to_int().save_all('bit1')
        # bit2_st.boolean_to_int().save_all('bit2')
        wigner_st.boolean_to_int().save_all('wigner')
        x_st.save_all('x')
        y_st.save_all('y')

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
    res = result_handles.get('wigner').fetch_all()['value']
    x = result_handles.get('x').fetch_all()['value']
    y = result_handles.get('y').fetch_all()['value']
    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'wigner_tomo_binary_decomp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("res", data=res)
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)


