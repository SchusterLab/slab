"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, two_chi, disc_file_opt, storage_mode
from qm.qua import *
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

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

###############
# qubit_spec_prog:
###############

f_min = -14.5e6
f_max = 0.5e6
df = 100e3
f_vec = np.arange(f_min, f_max + df/2, df)

avgs = 1000
reset_time = int(5e6)
simulation = 0

cav_len = 10
cav_amp = 1.0

with program() as storage_spec:

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

        with for_(f, ge_IF[0] + f_min, f < ge_IF[0] + f_max + df/2, f + df):

            wait(reset_time// 4, "storage_mode0")# wait for the storage to relax, several T1s
            # update_frequency("qubit_mode0", ge_IF[0])
            # discriminator.measure_state("readout", "out1", "out2", res, I=I)
            # align('qubit_mode0', 'rr')
            # play('pi', 'qubit_mode0', condition=res)
            # wait(reset_time//10, "qubit_mode0")

            # align('storage_mode0', 'qubit_mode0')
            update_frequency("qubit_mode0", f)

            play("CW"*amp(cav_amp), "storage_mode0", duration=cav_len)
            align("storage_mode0", "qubit_mode0")
            play("res_pi", "qubit_mode0")
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():

        res_st.boolean_to_int().buffer(len(f_vec)).average().save('res')
        I_st.buffer(len(f_vec)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(storage_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(storage_spec, duration_limit=0, data_limit=0)

    result_handles = job.result_handles

    # res_handle = result_handles.get("res")
    # res_handle.wait_for_values(1)
    # plt.figure()
    # while(result_handles.is_processing()):
    #     res = res_handle.fetch_all()
    #     plt.plot(f_vec, res, '.-')
    #     # plt.xlabel(r'Time ($\mu$s)')
    #     # plt.ylabel(r'$\Delta \nu$ (kHz)')
    #     plt.pause(5)
    #     plt.clf()

    # result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    plt.figure()
    plt.plot(f_vec, res, '.-')
    plt.show()

    job.halt()
    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    # data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'number_splitting', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=res)
        f.create_dataset("freq", data=f_vec)
        f.create_dataset("amp", data=cav_amp)
        f.create_dataset("time", data=cav_len*4)
        f.create_dataset("two_chi", data=two_chi[storage_mode])
