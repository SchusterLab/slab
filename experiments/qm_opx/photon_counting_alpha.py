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
import os
import scipy
from slab.dataanalysis import get_next_filename
"""Binary decomposition followed by repeated parity measurements"""
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
##################
filename = 'oct_pulses/g1.h5'

# filename = "S:\\Ankur\\Stimulated Emission\\pulses\\picollo\\2021-03-23\\00001_g0_to_g1_2.0us_qamp_7.5_camp_0.2_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5"
# filename = 'S:\\_Data\\210326 - QM_OPX\oct_pulses\\00000_g0_to_g1_2.0us_qamp_24.0_camp_0.25_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5'

with File(filename,'r') as a:
    Iq = np.array(a['uks'][-1][0], dtype=float)
    Qq = np.array(a['uks'][-1][1], dtype=float)
    Ic = np.array(a['uks'][-1][2], dtype=float)
    Qc = np.array(a['uks'][-1][3], dtype=float)
    a.close()

path = os.getcwd()
cal_path = os.path.join(path, "drive_calibration")

def transfer_function(omegas_in, cavity=False, qubit=True, pulse_length=2000):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity

    if cavity==True:
        fn_file = cal_path + '\\00000_2021_05_20_cavity_square.h5'
    elif qubit==True:
        fn_file = cal_path + '\\00000_2021_05_21_qubit_square.h5'

    with File(fn_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, -omegas)
    amps = np.append(amps, -amps)
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = [x for y, x in sorted(zip(amps, omegas))]
    a_s = np.sort(amps)

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(o_s, a_s)
    output_amps = []
    max_interp_index = np.argmax(omegas)

    for i in range(len(omegas_in)):
        # if frequency greater than calibrated range, assume a proportional relationship (high amp)
        if np.abs(omegas_in[i]) > omegas[max_interp_index]:
            output_amps.append(omegas_in[i] * amps[max_interp_index] / omegas[max_interp_index])
            # output_amps.append(amps[max_interp_index])
        else:  # otherwise just use the interpolated transfer function
            output_amps.append(transfer_fn((omegas_in[i])))
    return np.array(output_amps)

Iq = transfer_function(Iq, qubit=True)
Qq = transfer_function(Qq, qubit=True)
Ic = transfer_function(Ic, qubit=False, cavity=True)
Qc = transfer_function(Qc, qubit=False, cavity=True)

a_max = 0.45 #Max peak-peak amplitude out of OPX

Iq = [float(x*a_max) for x in Iq]
Qq = [float(x*a_max) for x in Qq]
Ic = [float(x*a_max) for x in Ic]
Qc = [float(x*a_max) for x in Qc]

config['pulses']['qoct_pulse']['length'] = len(Iq)
config['pulses']['soct_pulse']['length'] = len(Ic)

config['waveforms']['qoct_wf_i']['samples'] = Iq
config['waveforms']['qoct_wf_q']['samples'] = Qq
config['waveforms']['soct_wf_i']['samples'] = Ic
config['waveforms']['soct_wf_q']['samples'] = Qc

pulse_len = 500

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

t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time

simulation_config = SimulationConfig(
    duration=30000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_jpa.npz', lsb=True)

""""Coherent drive to simulate the dark matter push"""
coh_len = 100
coh_amp = 0.00

camp = np.round(np.arange(0.0005, 0.001, 0.0001).tolist(), 6)
qm = qmm.open_qm(config)

avgs = 75000
reset_time = int(3.5e6)
simulation = 0

num_pi_pulses_m = 30 #need even number to bring the qubit back to 'g' before coherent drive
num_pi_pulses_n = 0

simulation = 0
camp = [0.0002]

for a in camp:

    coh_amp = a

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

            wait(reset_time//4, 'storage')
            # update_frequency('qubit', ge_IF)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ########################
            # play("soct"*amp(0.0), "storage", duration=pulse_len)
            # play("qoct"*amp(0.0), "qubit", duration=pulse_len)
            # ########################
            # align('storage', 'qubit')
            # play("pi2", "qubit") # unconditional
            # wait(t_chi//4, "qubit")
            # frame_rotation(np.pi, 'qubit') #
            # play("pi2", "qubit")
            # align('qubit', 'rr', 'jpa_pump')
            # play('pump_square', 'jpa_pump')
            # discriminator.measure_state("clear", "out1", "out2", bit1, I=I)
            # save(bit1, bit1_st)
            #
            # reset_frame("qubit")
            # wait(1000//4, "rr")
            # align("qubit", "rr", 'jpa_pump')
            #
            # play("pi2", "qubit") # unconditional
            # wait(t_chi//4//2-3, "qubit") # subtracted 3 to make the simulated waveforms accurate
            # with if_(bit1==0):
            #     frame_rotation(np.pi, 'qubit')
            #     play("pi2", "qubit")
            # with else_():
            #     frame_rotation(3/2*np.pi, 'qubit')
            #     play("pi2", "qubit")
            # align('qubit', 'rr', 'jpa_pump')
            # play('pump_square', 'jpa_pump')
            # discriminator.measure_state("clear", "out1", "out2", bit2, I=I)
            # save(bit2, bit2_st)
            #
            # assign(num, Cast.to_int(bit1) + 2*Cast.to_int(bit2))
            # update_frequency('qubit', ge_IF + (num)*two_chi)
            # align('storage', 'rr', 'jpa_pump')
            assign(bit1, 0)
            assign(bit2, 0)
            save(bit1, bit1_st)
            save(bit2, bit2_st)

            play('CW'*amp(coh_amp), 'storage', duration=coh_len)

            with for_(i, 0, i < num_pi_pulses_m, i+1):
                wait(1000//4, "rr")
                align("qubit", "rr", 'jpa_pump')
                play("pi2", "qubit") # unconditional
                wait(t_chi//4, "qubit")
                frame_rotation(np.pi, 'qubit') #
                play("pi2", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
                save(bit3, bit3_st)

        with stream_processing():
            bit1_st.boolean_to_int().save_all('bit1')
            bit2_st.boolean_to_int().save_all('bit2')
            bit3_st.boolean_to_int().buffer(num_pi_pulses_m+num_pi_pulses_n).save_all('bit3')

    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(binary_decomposition, simulation_config)
        samples = job.get_simulated_samples()
        samples.con1.plot()
        result_handles = job.result_handles

    else:

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
                                     get_next_filename(data_path, 'photon_counting_parity_alpha', suffix='.h5'))
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("p_cav", data=p_cav)
            f.create_dataset("amp", data=coh_amp)
            f.create_dataset("time", data=coh_len*4)
            f.create_dataset("bit1", data=bit1)
            f.create_dataset("bit2", data=bit2)
            f.create_dataset("bit3", data=bit3)
            f.create_dataset("pi_m", data=num_pi_pulses_m)
            f.create_dataset("pi_n", data=num_pi_pulses_n)