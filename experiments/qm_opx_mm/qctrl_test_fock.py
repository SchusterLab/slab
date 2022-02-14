from configuration_IQ import config, ge_IF, biased_th_g_jpa, two_chi, disc_file_opt, disc_file
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
from fock_state_prep import oct_to_opx_amp, opx_amp_to_alpha, oct_to_opx_amp_test

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

###############
# qubit_spec_prog:
###############
def oct_test(scale=1.0, fock_state=1, pulse=None):

    avgs = 100
    reset_time = int((fock_state/2+1)*7.5e6)
    simulation = 0

    if fock_state == 0:
        scale=0.0

    pulse_len = oct_to_opx_amp_test(opx_config=config, pulse_filename=pulse, qctrl=True)

    print(pulse_len)

    f_min = (fock_state+2.5)*two_chi[1]
    f_max = (fock_state-2.5)*two_chi[1]
    df = (f_max - f_min)/100

    f_vec = np.arange(f_min, f_max + df/2, df)

    # f_min = -6.5e6
    # f_max = 0.5e6
    # df = 70e3
    # f_vec = np.arange(f_min, f_max + df/2, df)

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
                discriminator.measure_state("readout", "out1", "out2", res, I=I)

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

print("Experiment running")
pulse_path = './oct_pulses/qctrl/'

filenames = os.listdir(pulse_path)

for file in filenames[0:1]:
    filename = pulse_path+file
    print(filename)
    job = oct_test(fock_state=1, pulse=filename)
    #
    result_handles = job.result_handles
    # # result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    # f_vec = result_handles.get('f').fetch_all()
    #
    # plt.figure()
    # plt.plot(f_vec, res, '.-')
    # plt.show()
    #
    # job.halt()
    #
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'qctrl_fock', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("freq", data=f_vec + ge_IF[0])
    #     f.create_dataset("two_chi", data=two_chi)