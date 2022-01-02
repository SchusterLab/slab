from configuration_IQ import config, ge_IF, two_chi, disc_file_opt, storage_cal_file, qubit_cal_file
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
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)
###############
# qubit_spec_prog:
###############
def transfer_function(omegas_in, cavity=False, qubit=True, storage_cal_file=storage_cal_file[1], qubit_cal_file=qubit_cal_file):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity

    if cavity==True:
        fn_file = storage_cal_file
    elif qubit==True:
        fn_file = qubit_cal_file

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

def oct_to_opx_amp(opx_config=config, fock_state=fock_state):
    """Given the Fock state to prepare, it obtains the oct amplitudes in terms of the opx amp and
    updates the waveforms in the opx config"""
    path = os.getcwd()
    pulse_filename = path+"//oct_pulses//g"+str(fock_state)+".h5"

    with File(pulse_filename,'r') as a:
        Iq = np.array(a['uks'][-1][0], dtype=float)
        Qq = np.array(a['uks'][-1][1], dtype=float)
        Ic = np.array(a['uks'][-1][2], dtype=float)
        Qc = np.array(a['uks'][-1][3], dtype=float)
        a.close()

    Iq = transfer_function(Iq, qubit=True)
    Qq = transfer_function(Qq, qubit=True)
    Ic = transfer_function(Ic, qubit=False, cavity=True)
    Qc = transfer_function(Qc, qubit=False, cavity=True)
    a_max = 0.45 #Max peak-peak amplitude out of OPX

    Iq = [float(x*a_max) for x in Iq]
    Qq = [float(x*a_max) for x in Qq]
    Ic = [float(x*a_max) for x in Ic]
    Qc = [float(-x*a_max) for x in Qc] #We need to multiply this with a -ve sign since we are using the LSB with a +ve IF frequency

    config['pulses']['qoct_pulse']['length'] = len(Iq)
    config['pulses']['soct_pulse']['length'] = len(Ic)

    config['waveforms']['qoct_wf_i']['samples'] = Iq
    config['waveforms']['qoct_wf_q']['samples'] = Qq
    config['waveforms']['soct_wf_i']['samples'] = Ic
    config['waveforms']['soct_wf_q']['samples'] = Qc

    return len(Iq)

def oct_test(scale=1.0, fock_state=1):

    avgs = 500
    reset_time = int(7.5e6)
    simulation = 0

    pulse_len = oct_to_opx_amp(opx_config=config, fock_state=fock_state)//2

    f_min = (fock_state+1)*two_chi[1]
    f_max = (fock_state-1)*two_chi[1]
    df = (f_max - f_min)/100

    f_vec = np.arange(f_min, f_max + df/2, df)

    with program() as expt:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)        # Averaging
        f = declare(int)        # Frequencies
        res = declare(bool)
        I = declare(fixed)
        f = declare(int)

        res_st = declare_stream()
        I_st = declare_stream()
        f_st = declare_stream()

        ###############
        # the sequence:
        ###############
        with for_(n, 0, n < avgs, n + 1):

            with for_(f, ge_IF[0] + f_min, f < ge_IF[0] + f_max + df/2, f + df):

                update_frequency('qubit_mode0', ge_IF[0])
                wait(reset_time// 4, 'storage_mode1')# wait for the storage to relax, several T1s
                align('storage_mode1', 'qubit_mode0')
                #########################
                play("soct"*amp(scale), 'storage_mode1', duration=pulse_len)
                play("qoct"*amp(scale), 'qubit_mode0', duration=pulse_len)
                #########################
                align('storage_mode1', 'qubit_mode0')
                update_frequency('qubit_mode0', f)
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)
                save(f, f_st)

        with stream_processing():

            res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
            I_st.buffer(len(f_vec)).average().save('I')
            f_st.buffer(len(f_vec)).average().save('f')

    qm = qmm.open_qm(config)
    if simulation:
        """To simulate the pulse sequence"""
        job = qmm.simulate(config, expt, SimulationConfig(15000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        """To run the actual experiment"""
        job = qm.execute(expt, duration_limit=0, data_limit=0)
        print("Experiment done")

    return job

for s in arange(1, 7, 1):
    job = oct_test(fock_state=s)
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    f_vec = result_handles.get('f').fetch_all()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    # data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'fock_state_oct', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=res)
        f.create_dataset("freq", data=f_vec-ge_IF[0])

# plt.figure()
# plt.pcolormesh(f_vec, scale_vec, data, cmap='RdBu')
# plt.colorbar()
# plt.xlabel('Frequency (MHz')
# plt.ylabel('Scale factor')
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