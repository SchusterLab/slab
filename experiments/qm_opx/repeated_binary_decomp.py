from configuration_IQ import config, ge_IF, qubit_freq, biased_th_g
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from tqdm import tqdm
from h5py import File
import os
from slab.dataanalysis import get_next_filename
"""Repeated binary decomposition"""

t_chi = int(0.5*1e9/1.118e6) #qubit rotates by pi in this time
cav_len = 750
cav_amp = 0.04 # 0.08

avgs = 1000
reset_time = 5000000
simulation = 0

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params_opt.npz', lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    wait(5000//4, 'rr')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

with program() as binary_decomposition:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    bit1_st = declare_stream()
    bit2_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        wait(reset_time//4, 'storage')
        play('CW'*amp(cav_amp), 'storage', duration=cav_len)
        align('storage', 'qubit')
        play("pi2", "qubit") # unconditional
        wait(t_chi//4, "qubit")
        frame_rotation(np.pi, 'qubit') #
        play("pi2", "qubit")
        align("qubit", "rr")
        discriminator.measure_state("clear", "out1", "out2", res, I=I)
        save(res, bit1_st)

        reset_frame("qubit")
        wait(1000//4, "rr")
        align("qubit", "rr")

        play("pi2", "qubit") # unconditional
        wait(t_chi//4//2, "qubit")
        with if_(res==0):
            frame_rotation(np.pi, 'qubit')
            play("pi2", "qubit")
        with else_():
            frame_rotation(3/2*np.pi, 'qubit')
            play("pi2", "qubit")
        align("qubit", "rr")
        discriminator.measure_state("clear", "out1", "out2", res, I=I)
        save(res, bit2_st)

    with stream_processing():
        bit1_st.boolean_to_int().save_all('bit1')
        bit2_st.boolean_to_int().save_all('bit2')

qm = qmm.open_qm(config)
if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(binary_decomposition, SimulationConfig(15000))
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

    num = bit1 + 2*bit2

    p_cav = [np.sum(num==0)*100/avgs, np.sum(num==1)*100/avgs, np.sum(num==2)*100/avgs, np.sum(num==3)*100/avgs]

    print("n=0 => {}, n=1 => {}, n=2 => {},n=3 => {}".format(p_cav[0], p_cav[1], p_cav[2], p_cav[3]))
    job.halt()
    #
    # plt.plot(res)
    #
    # times = 4*times/1e3
    # path = os.getcwd()
    # data_path = os.path.join(path, "data/")
    # seq_data_file = os.path.join(data_path,
    #                              get_next_filename(data_path, 'binar_decomp', suffix='.h5'))
    # print(seq_data_file)
    # with File(seq_data_file, 'w') as f:
    #     f.create_dataset("Q", data=res)
    #     f.create_dataset("I", data=I)
    #     f.create_dataset("time", data=times)
    #     f.create_dataset("ramsey_freq", data=ramsey_freq)
    #     f.create_dataset("qubit_freq", data=qubit_freq)