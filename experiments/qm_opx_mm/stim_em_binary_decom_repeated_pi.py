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
"""Binary decomposition followed by repeated resolved pi pulses"""
def alpha_awg_cal(alpha, cav_amp=1.0):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_11_24_cavity_square.h5'

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

t_chi = int(abs(0.5*1e9/two_chi[1])) #qubit rotates by pi in this time

"""Coherent drive to create Fock state in the cavity"""
cav_len = 400
cav_amp = 0.00
""""Coherent drive to simulate the dark matter push"""
coh_len = 400
coh_amp = 0.00

# Fock1: D(-0.580) * S(0,pi) * D(1.143) * |0>
# Fock2: D(0.432) * S(1,pi) * D(-1.133) * S(0,pi) * D(0.497) * |0>
# Fock3: D(0.344) * S(2,pi) * D(-1.072) * S(1,pi) * D(-1.125) * S(0,pi) * D(1.878) * |0>
# Fock4: D(-0.284) * S(3,pi) * D(0.775) * S(2,pi) * D(-0.632) * S(1,pi) * D(-0.831) * S(0,pi) * D(1.555) * |0>

avgs = 1000
reset_time = int(7.5e6)
simulation = 0

num_pi_pulses_m = 10 #need even number to bring the qubit back to 'g' before coherent drive
num_pi_pulses_n = 30

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
##################
path = os.getcwd()

with program() as binary_decomposition:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    bit1 = declare(bool)
    bit2 = declare(bool)
    bit3 = declare(bool)
    bit4 = declare(bool)
    num = declare(int)
    I = declare(fixed)

    bit1_st = declare_stream()
    bit2_st = declare_stream()
    bit3_st = declare_stream()
    bit4_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        wait(reset_time//4, 'storage_mode1')
        update_frequency('qubit_mode0', ge_IF)
        ########################
        play("CW" * amp(0.4), 'storage_mode1', duration=alpha_awg_cal(1.143))
        align('storage_mode1', 'qubit_mode0')
        play("res_pi"*amp(2.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW" * amp(-0.4), 'storage_mode1', duration=alpha_awg_cal(-0.58)) #249
        ########################
        align('storage_mode1', 'qubit_mode0')
        play("pi2", 'qubit_mode0') # unconditional
        wait(t_chi//4+1, 'qubit_mode0')
        frame_rotation(np.pi, 'qubit_mode0') #
        play("pi2", 'qubit_mode0')
        align('qubit_mode0', 'rr')
        discriminator.measure_state("clear", "out1", "out2", bit1, I=I)
        save(bit1, bit1_st)

        reset_frame('qubit_mode0')
        wait(250, "rr")
        align('qubit_mode0', "rr")

        play("pi2", 'qubit_mode0') # unconditional
        wait(t_chi//4//2-4, 'qubit_mode0') # subtracted 3 to make the simulated waveforms accurate
        with if_(bit1==0):
            frame_rotation(np.pi, 'qubit_mode0')
            play("pi2", 'qubit_mode0')
        with else_():
            frame_rotation(3/2*np.pi, 'qubit_mode0')
            play("pi2", 'qubit_mode0')
        align('qubit_mode0', 'rr')
        discriminator.measure_state("clear", "out1", "out2", bit2, I=I)
        save(bit2, bit2_st)

        assign(num, Cast.to_int(bit1) + 2*Cast.to_int(bit2))

        update_frequency('qubit_mode0', ge_IF[0] + (num+1)*two_chi)

        with for_(i, 0, i < num_pi_pulses_m, i+1):
            wait(250, "rr")
            align('qubit_mode0', "rr", 'jpa_pump')
            play("res_pi", 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
            save(bit3, bit3_st)

        align('storage_mode1', 'rr')

        play('CW'*amp(coh_amp), 'storage_mode1', duration=coh_len)

        align('qubit_mode0', 'storage_mode1')

        with for_(i, 0, i < num_pi_pulses_n, i+1):
            wait(250, "rr")
            align('qubit_mode0', "rr", 'jpa_pump')
            play("res_pi", 'qubit_mode0')
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", bit4, I=I)
            save(bit4, bit3_st)

    with stream_processing():
        bit1_st.boolean_to_int().save_all('bit1')
        bit2_st.boolean_to_int().save_all('bit2')
        bit3_st.boolean_to_int().buffer(num_pi_pulses_m+num_pi_pulses_n).save_all('bit3')

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
    bit3 = result_handles.get('bit3').fetch_all()['value']

    num = bit1 + 2*bit2

    p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

    print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))
    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'stim_em_bd_rp', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("p_cav", data=p_cav)
        f.create_dataset("amp", data=cav_amp)
        f.create_dataset("time", data=cav_len*4)
        f.create_dataset("bit1", data=bit1)
        f.create_dataset("bit2", data=bit2)
        f.create_dataset("bit3", data=bit3)
        f.create_dataset("pi_m", data=num_pi_pulses_m)
        f.create_dataset("pi_n", data=num_pi_pulses_n)