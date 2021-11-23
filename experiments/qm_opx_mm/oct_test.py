from configuration_IQ import config, ge_IF, biased_th_g_jpa, two_chi, disc_file
from qm.qua import *
from qm import SimulationConfig
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
import time

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
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

###############
# qubit_spec_prog:
###############
f_min = -5.5e6
f_max = 0.5e6
df = 40e3
f_vec = np.arange(f_min, f_max + df/2, df)

filename = 'oct_pulses/g1.h5'

# filename = "S:\\Ankur\\Stimulated Emission\\pulses\\picollo\\2021-03-23\\00001_g0_to_g1_2.0us_qamp_7.5_camp_0.2_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5"
# filename = 'S:\\_Data\\210326 - QM_OPX\oct_pulses\\00000_g0_to_g1_2.0us_qamp_24.0_camp_0.12_gamp_0.1_dwdt_1.0_dw2dt2_0.1.h5'

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
        fn_file = cal_path + '\\00000_2021_08_13_cavity_square.h5'
    elif qubit==True:
        fn_file = cal_path + '\\00000_2021_08_14_qubit_square.h5'

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

avgs = 1000
reset_time = int(3.75e6)
simulation = 0
with program() as oct_test:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ge_IF + f_min, f < ge_IF + f_max + df/2, f + df):

            update_frequency("qubit", ge_IF)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            align('storage', 'rr', 'jpa_pump', 'qubit')
            active_reset(biased_th_g_jpa)
            align('storage', 'rr', 'jpa_pump', 'qubit')
            #########################
            play("soct", "storage", duration=pulse_len)
            play("qoct", "qubit", duration=pulse_len)
            #########################
            align("storage", "qubit")
            update_frequency("qubit", f)
            play("res_pi", "qubit")
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
        I_st.buffer(len(f_vec)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qmm.simulate(config, oct_test, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(oct_test, duration_limit=0, data_limit=0)
    print("Experiment done")

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    #
    # plt.figure()
    # plt.plot(f_vec, res, '.-')
    # plt.show()

    # job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'oct_fock', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("freq", data=f_vec)
    #     f.create_dataset("two_chi", data=two_chi)