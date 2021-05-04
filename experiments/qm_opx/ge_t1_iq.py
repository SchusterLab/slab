from configuration_IQ import config, biased_th_g
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
##################
# t1:
##################
dt = 1000
T_max = 50000
T_min = 25
times = np.arange(T_min, T_max + dt/2, dt)

avgs = 1000
reset_time = 500000
simulation = 0 #1 to simulate the pulses

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.00**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', 'ge_disc_params.npz', lsb=True)

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


with program() as ge_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    # res = declare(bool)
    I = declare(fixed)
    Q = declare(fixed)

    # res_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, T_min, t < T_max + dt/2, t + dt):
            wait(reset_time//4, "qubit")
            # active_reset(biased_th_g)
            # align("qubit", 'rr')
            play("pi", "qubit")
            wait(t, "qubit")
            align("qubit", "rr")
            # discriminator.measure_state("clear", "out1", "out2", res, I=I)
            #
            # save(res, res_st)
            # save(I, I_st)
            measure("clear", "rr", None,
                    demod.full("clear_integW1", I1, 'out1'),
                    demod.full("clear_integW2", Q1, 'out1'),
                    demod.full("clear_integW1", I2, 'out2'),
                    demod.full("clear_integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        # res_st.boolean_to_int().buffer(len(times)).average().save('res')
        I_st.buffer(len(times)).average().save('I')
        Q_st.buffer(len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(ge_t1, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(ge_t1, duration_limit=0, data_limit=0)
    print ("Execution done")

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    # res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    Q = result_handles.get('Q').fetch_all()

    print ("Data collection done")

    job.halt()

    plt.plot(times, Q)

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 't1', suffix='.h5'))
    print(seq_data_file)

    times = 4*times #actual clock time
    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("res", data=Q)
        f.create_dataset("time", data=times)