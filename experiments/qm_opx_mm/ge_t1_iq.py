from configuration_IQ import config
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
T_max = 125000
T_min = 4
times = np.arange(T_min, T_max + dt/2, dt)

avgs = 1000
reset_time = 500000
simulation = 0 #1 to simulate the pulses

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

    I = declare(fixed)
    Q = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, T_min, t < T_max + dt/2, t + dt):
            wait(reset_time//4, "qubit_mode0")
            play("pi", "qubit_mode0")
            wait(t, "qubit_mode0")
            align("qubit_mode0", "rr")
            measure("readout", "rr", None,
                    demod.full("integW1", I1, 'out1'),
                    demod.full("integW2", Q1, 'out1'),
                    demod.full("integW1", I2, 'out2'),
                    demod.full("integW2", Q2, 'out2'))

            assign(I, I1-Q2)
            assign(Q, I2+Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
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

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done")

    job.halt()

    times = 4*times/1e3
    plt.figure()
    plt.plot(times, Q, '.-')
    plt.plot(times, I, '.-')

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 't1', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=Q)
        f.create_dataset("time", data=times)
