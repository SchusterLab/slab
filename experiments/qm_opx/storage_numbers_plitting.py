from configuration_IQ import config, qubit_freq, rr_LO, qubit_LO, ge_IF, storage_IF, storage_freq, storage_LO
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
im = InstrumentManager()
LO_q = im['RF5']
LO_r = im['RF8']
LO_s  = im['sccav']

# LO_s.set_frequency(storage_LO)
# LO_s.set_power(13.0)
# LO_q.set_frequency(qubit_LO)
# LO_q.set_ext_pulse(mod=False)
# LO_q.set_power(18)
# LO_r.set_frequency(rr_LO)
# LO_r.set_ext_pulse(mod=False)
# LO_r.set_power(18)

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)
biased_th = 0.0012

###############
# qubit_spec_prog:
###############

f_min = -9e6
f_max = 1e6
df = 100e3
f_vec = np.arange(f_min, f_max + df/2, df)

# n_max = 10
# df = -1.118361e6
# f_vec = df*np.arange(n_max)
# f_min = np.min(f_vec)
# f_max = np.max(f_vec)

avgs = 1000
reset_time = 2500000
simulation = 0

cav_len = 1000
cav_amp = 0.15
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

        with for_(f, ge_IF + f_min, f < ge_IF + f_max + df/2, f + df):

            update_frequency("qubit", f)
            wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
            play("CW"*amp(cav_amp), "storage", duration=cav_len)
            align("storage", "qubit")
            play("res_pi", "qubit")
            align("qubit", "rr")
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
    print("Experiment done")
    start_time = time.time()

    result_handles = job.result_handles

    I_handle = result_handles.get("res")
    I_handle.wait_for_values(1)
    plt.figure()
    while(result_handles.is_processing()):
        I = I_handle.fetch_all()
        plt.plot(f_vec, I, '.-')
        # plt.xlabel(r'Time ($\mu$s)')
        # plt.ylabel(r'$\Delta \nu$ (kHz)')
        plt.pause(5)
        plt.clf()

    # result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    # I = result_handles.get('I').fetch_all()
    #
    # job.halt()
    # stop_time = time.time()
    # print(f"Time taken: {stop_time-start_time}")
    # #
    # # path = os.getcwd()
    # # data_path = os.path.join(path, "data/")
    # data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'number_splitting', suffix='.h5'))
    # print(seq_data_file)
    #
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("freq", data=f_vec)
    #     f.create_dataset("amp", data=cav_amp)
    #     f.create_dataset("time", data=cav_len*4)
