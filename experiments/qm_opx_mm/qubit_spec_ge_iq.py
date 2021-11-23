from configuration_IQ import config, qubit_freq, rr_LO, qubit_LO, ge_IF
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
###############
# qubit_spec_prog:
###############
f_min = -100e3
f_max = 100e3
df = 2e3

f_vec = np.arange(f_min, f_max + df/2, df)
f_vec = f_vec + qubit_freq[0]

avgs = 500
reset_time = 500000
simulation = 0
with program() as qubit_spec:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    f = declare(int)        # Frequencies
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    I_st = declare_stream()
    Q_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(f, ge_IF[0] + f_min, f <= ge_IF[0] + f_max, f + df):

            update_frequency("qubit_mode0", f)
            wait(reset_time// 4, "qubit_mode0")# wait for the qubit to relax, several T1s
            play("CW"*amp(0.0002), "qubit_mode0", duration=2700)
            align("qubit_mode0", "rr")
            measure("readout", "rr", None,
                    demod.full("integW1", I1, 'out1'),
                    demod.full("integW2", Q1, 'out1'),
                    demod.full("integW1", I2, 'out2'),
                    demod.full("integW2", Q2, 'out2'))

            assign(I, I1 - Q2)
            assign(Q, I2 + Q1)

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():

        I_st.buffer(len(f_vec)).average().save('I')
        Q_st.buffer(len(f_vec)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(qubit_spec, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    """To run the actual experiment"""
    job = qm.execute(qubit_spec, duration_limit=0, data_limit=0)
    print("Experiment done")

    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    print("Data collection done!")

    job.halt()

    plt.figure()
    plt.plot(I, '.-')
    plt.plot(Q, '.-')
    plt.show()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'ge_spectroscopy', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=Q)
        f.create_dataset("freqs", data=f_vec)