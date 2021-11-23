"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
import hmm
from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g_jpa, two_chi, disc_file
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
"""Repeated parity measurements followed by coherent drive"""
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
    return abs(pulse_length)//4+1
##################

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)

    wait(1000//4, "jpa_pump")
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

t_chi = int(abs(0.5*1e9/two_chi) //4 +1) #FPGA clock cycle unit, qubit rotates by pi in this time

simulation_config = SimulationConfig(
    duration=30000,
    include_analog_waveforms=True,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

########
qubit_params = {'t1':100, 't2':130, 'nth':5e-2}
cavity_params = {'t1':5e2, 'nth':0.001}
readout_params = {'length':3.2, 'trigger':7.2, 'pi_pulse':3, 'g_inf':0.02, 'e_inf':0.05}

qubit_t1 = qubit_params['t1']
qubit_t2 = qubit_params['t2']
qubit_nth = qubit_params['nth']
cavity_t1 = cavity_params['t1']
cavity_nth = cavity_params['nth']

readout_len = readout_params['length']
trigger_period = readout_params['trigger']
pi_length = readout_params['pi_pulse']

"""Readout fidelities from an independent measurement"""
g_infidelity, e_infidelity = 0.0194, 0.06647

fstate_in = 0

if fstate_in == 0:
    cavity_t1 = cavity_t1/1
    Pnm =  cavity_nth * (1-np.exp(-trigger_period/cavity_t1))
else:
    cavity_t1 = cavity_t1/fstate_in
    Pnm =  (1-np.exp(-trigger_period/cavity_t1)) + cavity_nth * (1-np.exp(-strigger_period/cavity_t1))

Pmn = 0 + 0 #assuming that the population at (n+1) is negligible and (n-1) we will estimate\
Pge = qubit_nth * (1-np.exp(-trigger_period/qubit_t1)) + \
      (1-np.exp(-pi_length/qubit_t2))
Peg = (1-np.exp(-trigger_period/qubit_t1)) + \
      (1-np.exp(-pi_length/qubit_t2))

T = np.asarray([[(1-Pmn)*(1-Pge), (1-Pmn)*Pge, Pmn*Pge, Pmn*(1-Pge)],
                [(1-Pmn)*Peg, (1-Pmn)*(1-Peg), Pmn*(1-Peg), Pmn*Peg],
                [Pnm*(1-Pge), Pnm*Pge, (1-Pnm)*Pge, (1-Pnm)*(1-Pge)],
                [Pnm*Peg, Pnm*(1-Peg), (1-Pnm)*(1-Peg), (1-Pnm)*Peg]])

E = 0.5*np.asarray([[1-g_infidelity, g_infidelity],
                    [e_infidelity, 1- e_infidelity],
                    [1-g_infidelity, g_infidelity],
                    [e_infidelity, 1- e_infidelity]])

#######

""""Coherent drive to simulate the dark matter push"""
coh_len = 100
coh_amp = 0.00

camp = np.round(np.arange(0.1, 0.5, 0.1).tolist(), 6)
qm = qmm.open_qm(config)

avgs = 1000
reset_time = int(3.75e6)
simulation = 0

num_pi_pulses_m = 30 #need even number to bring the qubit back to 'g' before coherent drive
num_pi_pulses_n = 0

camp = [0.1]

for a in camp:

    coh_amp = a

    with program() as repeated_parity:

        ##############################
        # declare real-time variables:
        ##############################

        k = declare(int)      # Averaging
        i = declare(int)      # Amplitudes
        t = declare(int) #array of time delays
        bit3 = declare(bool)
        I = declare(fixed)
        bit3_st = declare_stream()
        meas_vec = declare(int, size=num_pi_pulses_m)

        #############################
        # variables delcaration HMM #
        #############################
        t = declare(int)
        n = declare(int)
        j = declare(int)
        E_0 = declare(fixed, value=E[:, 0].tolist())
        E_1 = declare(fixed, value=E[:, 1].tolist())
        T_vec0 = declare(fixed, value=T[0, :].tolist())
        T_vec1 = declare(fixed, value=T[1, :].tolist())
        T_vec2 = declare(fixed, value=T[2, :].tolist())
        T_vec3 = declare(fixed, value=T[3, :].tolist())
        beta = declare(fixed, size=4)
        b0_stream = declare_stream()
        b1_stream = declare_stream()
        b2_stream = declare_stream()
        b3_stream = declare_stream()
        E_temp = declare(fixed)
        beta0_temp = declare(fixed)
        beta1_temp = declare(fixed)
        beta2_temp = declare(fixed)
        beta3_temp = declare(fixed)
        norm = declare(fixed)


        ###############
        # the sequence:
        ###############
        with for_(k, 0, k < 1, k + 1):
            wait(reset_time//4, 'storage')
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            ########################
            play('CW'*amp(coh_amp), 'storage', duration=coh_len)
            ########################
            align('storage', 'qubit')

            with for_(i, 0, i < num_pi_pulses_m, i+1):
                # reset_frame('qubit')
                with qrun_():
                    align("qubit", "rr", 'jpa_pump')
                    play("pi2", "qubit") # unconditional
                    wait(t_chi, "qubit")
                    frame_rotation(np.pi, 'qubit') #
                    play("pi2", "qubit")
                    align('qubit', 'rr', 'jpa_pump')
                    play('pump_square', 'jpa_pump')
                    discriminator.measure_state("clear", "out1", "out2", bit3, I=I)
                save(bit3, bit3_st)
                wait(1000//4, "rr")
                assign(meas_vec[i],Cast.to_int(bit3))




            #################
            # backward algo #
            #################
            with for_(j, 0, j<4, j+1):
                assign(beta[j], 0.25)
            with for_(t, meas_vec.length()  - 1, t >= 0, t - 1):
                assign(beta0_temp, 0)
                assign(beta1_temp, 0)
                assign(beta2_temp, 0)
                assign(beta3_temp, 0)
                with for_(n, 0, n < T.shape[1], n + 1):
                    assign(E_temp, Util.cond(meas_vec [t] == 1, E_1[n], E_0[n]))
                    assign(beta0_temp, beta0_temp + (beta[n] * T_vec0[n] * E_temp))
                    assign(beta1_temp, beta1_temp + (beta[n] * T_vec1[n] * E_temp))
                    assign(beta2_temp, beta2_temp + (beta[n] * T_vec2[n] * E_temp))
                    assign(beta3_temp, beta3_temp + (beta[n] * T_vec3[n] * E_temp))
                assign(norm, Math.div(1,beta0_temp+beta1_temp+beta2_temp+beta3_temp))
                assign(beta[0], beta0_temp*norm)
                assign(beta[1], beta1_temp*norm)
                assign(beta[2], beta2_temp*norm)
                assign(beta[3], beta3_temp*norm)
                save(beta[0], b0_stream)
                save(beta[1], b1_stream)
                save(beta[2], b2_stream)
                save(beta[3], b3_stream)

        with stream_processing():
            bit3_st.boolean_to_int().buffer(num_pi_pulses_m+num_pi_pulses_n).save_all('bit3')
            b0_stream.save_all('b0')
            b1_stream.save_all('b1')
            b2_stream.save_all('b2')
            b3_stream.save_all('b3')


    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(repeated_parity, simulation_config)
        samples = job.get_simulated_samples()
        samples.con1.plot()
        result_handles = job.result_handles

    else:

        job = qm.execute(repeated_parity, duration_limit=0, data_limit=0)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        bit3 = res_handles.get('bit3').fetch_all()['value']
        back_prob_qua = np.zeros((T.shape[0], num_pi_pulses_m))
        back_prob_qua[0, :] = res_handles.b0.fetch_all()['value']
        back_prob_qua[1, :] = res_handles.b1.fetch_all()['value']
        back_prob_qua[2, :] = res_handles.b2.fetch_all()['value']
        back_prob_qua[3, :] = res_handles.b3.fetch_all()['value']
        job.halt()

        path = os.getcwd()
        data_path = os.path.join(path, "data/")
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'photon_counting_parity_test', suffix='.h5'))
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("amp", data=coh_amp)
            f.create_dataset("time", data=coh_len*4)
            f.create_dataset("bit3", data=bit3)
            f.create_dataset("g0", data=back_prob_qua[0, :])
            f.create_dataset("e0", data=back_prob_qua[1, :])
            f.create_dataset("g1", data=back_prob_qua[2, :])
            f.create_dataset("e1", data=back_prob_qua[3, :])
            f.create_dataset("pi_m", data=num_pi_pulses_m)
            f.create_dataset("pi_n", data=num_pi_pulses_n)

def backward(meas_seq, T, E):
    N = T.shape[0]
    num_meas = len(meas_seq)
    beta = np.zeros((N,num_meas))
    beta[:,-1] = 0.25 #No information in the last measuremnt
    for t in reversed(range(num_meas-1)):
        for n in range(N):
            beta[n,t] = sum(beta[:,t+1] * T[n,:] * E[:, meas_seq[t+1]])
        norm = beta[0,t]+beta[1,t]+beta[2,t]+beta[3,t]
        norm = 1/norm
        for i in range(N):
            beta[i,t]=beta[i,t]*norm
    return beta

plt.figure()
for i in range(4):
    plt.plot(np.arange(28,-2, -1),backward(bit3[0], T, E)[i])
    plt.plot(np.arange(0,30, 1),back_prob_qua[i, :], '*')

# print(job.simulated_analog_waveforms()['elements']['qubit'][16]['timestamp']-job.simulated_analog_waveforms()['elements']['qubit'][14]['duration']-job.simulated_analog_waveforms()['elements']['qubit'][14]['timestamp'])