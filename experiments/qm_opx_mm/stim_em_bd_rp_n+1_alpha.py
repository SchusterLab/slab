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
"""Stimulated emission experiment with varying coherent drive
    Sequence => State prep -> BD -> Coherent drive -> repeated pi's at n+1
"""
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
filename = 'oct_pulses/g1.h5'

# filename = "S:\\Ankur\\Stimulated Emission\\pulses\\picollo\\2021-03-23\\00001_g0_to_g1_2.0us_qamp_7.5_camp_0.2_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5"
# filename = 'S:\\_Data\\210326 - QM_OPX\oct_pulses\\00000_g0_to_g3_2.0us_qamp_18.75_camp_2.0_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5'

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
        fn_file = cal_path + '\\00000_2021_7_30_cavity_square.h5'
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
avgs = 10000
reset_time = int(5e6)
simulation = 0

t_chi = int(abs(0.5*1e9/two_chi)) #qubit rotates by pi in this time

def alpha_awg_cal(alpha, cav_amp=0.4):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    cal_path = 'C:\_Lib\python\slab\experiments\qm_opx\drive_calibration'

    fn_file = cal_path + '\\00000_2021_7_30_cavity_square.h5'

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

def ramsey():
    ramsey_freq = 100e3
    omega = 2*np.pi*ramsey_freq
    dt = 250
    dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

    T_min = 0
    T_max = 30000
    times = np.arange(T_min, T_max + dt/2, dt)
    avgs = 1000
    reset_time = 500000

    with program() as exp:

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

                active_reset(biased_th_g_jpa)
                align('qubit', 'rr', 'jpa_pump')
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
    job = qm.execute(exp, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    t = 4*times/1e3
    P = res
    p = fitdecaysin(t[:], P[:],fitparams = None, showfit=False)
    offset = ramsey_freq - p[1]
    nu_q_new = qubit_freq + offset/1e3
    t2 = p[3]
    return t2, qubit_freq

def stim_em(coh_amp=0.0, coh_len=100, f_state=0, n_pi_m=0, n_pi_n=30, avgs=1000):

    with program() as exp:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        bit1 = declare(bool)
        bit2 = declare(bool)
        bit3 = declare(bool)
        bit4 = declare(bool)

        num = declare(int)

        I = declare(fixed)

        num_st = declare_stream()
        bit_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            wait(reset_time//4, 'storage')
            update_frequency('qubit', ge_IF)
            update_frequency('storage', storage_IF)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ########################
            """Analytic SNAP pulses to create Fock states"""
            snap_seq(fock_state=f_state)
            align('storage', 'qubit')
            ########################

            ##########################
            """First Bd starts here"""
            ##########################

            play("pi2", "qubit") # unconditional
            wait(t_chi//4, "qubit")
            frame_rotation(np.pi, 'qubit') #
            play("pi2", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", bit1, I=I)

            reset_frame("qubit")
            wait(1000//4, "rr")
            align("qubit", "rr", 'jpa_pump')

            play("pi2", "qubit") # unconditional
            wait(t_chi//4//2-3, "qubit") # subtracted 3 to make the simulated waveforms accurate
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

            """First BD after the state prep ends here"""
            ########################
            # """Flip the qubit to |g> before repeated pi pulses"""
            # align('rr', 'jpa_pump', 'qubit')
            # with if_(bit2):
            #     play('pi', 'qubit')
            # align('rr', 'jpa_pump', 'qubit')
            ########################
            # reset_frame("qubit")
            # update_frequency('qubit', ge_IF + (num)*two_chi)

            ########################
            """Repeated pi pulses at n"""
            ########################

            # with for_(i, 0, i < n_pi_m, i+1):
            #     wait(1000//4, "rr")
            #     align("qubit", "rr", 'jpa_pump')
            #     play("res_pi", "qubit")
            #     align('qubit', 'rr', 'jpa_pump')
            #     play('pump_square', 'jpa_pump')
            #     discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
            #     save(bit3, bit_st)
            # ########################
            update_frequency('storage', storage_IF + (num*(num-1))*st_self_kerr)

            """active reset to bring qubit back to 'g' before the next BD"""
            with if_(bit2):
                align('rr', 'jpa_pump', 'qubit')
                update_frequency('qubit', ge_IF)
                play('pi', 'qubit')

            update_frequency('qubit', ge_IF + (num+1)*two_chi)

            align('storage', 'rr', 'qubit', 'jpa_pump')

            play('CW'*amp(coh_amp), 'storage', duration=coh_len)

            align("qubit", "storage")

            ########################
            """Repeated pi pulses at n+1"""
            ########################

            with for_(i, 0, i < n_pi_n, i+1):
                align("qubit", "rr", 'jpa_pump')
                play("res_pi", "qubit")
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", bit4, I=I)
                save(bit4, bit_st)
                wait(1000//4, "rr")

            ########################

        with stream_processing():
            num_st.save_all('num')
            bit_st.boolean_to_int().buffer(n_pi_m+n_pi_n).save_all('bit')

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
        result_handles.wait_for_all_values()
        num = result_handles.get('num').fetch_all()['value']
        bit = result_handles.get('bit').fetch_all()['value']

    p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

    print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))

    return coh_amp, coh_len, n_pi_m, n_pi_n, num, bit

# cav_amp = [x for x in np.linspace(0.0005, 0.0009, 5)]
# cav_amp.extend([x for x in np.linspace(0.001, 0.005, 5)])
cav_amp = [x for x in np.linspace(0.001, 0.009, 9)]

cav_amp = [0.1]

for ii in range(len(cav_amp)):
    # t2 = 0
    # while(t2<90):
    #     t2, q_f = ramsey()
    #     print(t2)
    #     time.sleep(600)

    l = 400
    coh_amp = np.round(cav_amp[ii], 4)
    fock_number = 3
    data = stim_em(coh_amp=coh_amp, coh_len=l/4, f_state=fock_number, n_pi_m=0, n_pi_n=20, avgs=20000)

    path = os.getcwd()

    data_path = os.path.join(path, "data/stim_em/n" + str(fock_number)) + '_reset_self_kerr'

    filename = "stim_em_n" + str(fock_number) +"_camp_" + str(coh_amp)+"_len_"+str(l)

    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, filename, suffix='.h5'))

    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("amp", data=coh_amp)
        f.create_dataset("time", data=l)
        f.create_dataset("pi_m", data=data[2])
        f.create_dataset("pi_n", data=data[3])
        f.create_dataset("num", data=data[4])
        f.create_dataset("bit", data=data[5])
