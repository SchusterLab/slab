from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, qubit_freq, rr_LO, qubit_LO, ge_IF, storage_IF, storage_freq, storage_LO
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
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

###############
# qubit_spec_prog:
###############
f_min = -9e6
f_max = 1e6
df = 100e3
f_vec = np.arange(f_min, f_max + df/2, df)
# #

# n_max = 10
# df = -1.118361e6
# f_vec = df*np.arange(n_max)
# f_min = np.min(f_vec)
# f_max = np.max(f_vec)

t_min = 25000
t_max = 75000
dt = 10000
t_vec = np.arange(t_min, t_max + dt/2, dt)

cav_amp = 0.001

avgs = 1000
reset_time = 5000000
simulation = 0
with program() as storage_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    t = declare(int)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()
    n_st = declare_stream()
    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):
        save(n, n_st)

        with for_(t, t_min, t < t_max + dt/2, t + dt):

            with for_(f, ge_IF + f_min, f < ge_IF + f_max + df/2, f + df):

                update_frequency("qubit", f)
                wait(reset_time// 4, "storage")# wait for the storage to relax, several T1s
                play("CW"*amp(cav_amp), "storage", duration=t)
                align("storage", "qubit")
                play("res_pi", "qubit")
                align("qubit", "rr")
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)
    with stream_processing():

        res_st.boolean_to_int().buffer(len(t_vec), len(f_vec)).average().save('res')
        I_st.buffer(len(t_vec), len(f_vec)).average().save('I')
        n_st.save('n')

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

    res_handle = result_handles.get("res")
    n_handle = result_handles.get("n")

    res_handle.wait_for_values(1)
    plt.figure()
    while(result_handles.is_processing()):
        res = res_handle.fetch_all()
        plt.pcolor(res, cmap='RdBu')
        plt.colorbar()
        # plt.xlabel(r'Time ($\mu$s)')
        # plt.ylabel(r'$\Delta \nu$ (kHz)')
        plt.pause(5)
        plt.clf()
    stop_time = time.time()
    print(f"Time taken: {stop_time-start_time}")

    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    job.halt()
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'alpha_cal', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("res", data=res)
        f.create_dataset("amp", data=cav_amp)
        f.create_dataset("times", data=t_vec)
        f.create_dataset("freq", data=f_vec)
