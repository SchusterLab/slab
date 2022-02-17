from configuration_IQ import config, disc_file_opt, readout_len
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
T_max = 75000
T_min = 4
times = np.arange(T_min, T_max + dt/2, dt)

l_min = 5
l_max = 45
dl = 2

l_vec = np.arange(l_min, l_max + dl/2, dl)

avgs = 1000
reset_time = int(7.5e6)

simulation = 0 #1 to simulate the pulses

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

pulse_len = readout_len

w_plus = [(1.0, pulse_len)]
w_minus = [(-1.0, pulse_len)]
w_zero = [(0.0, pulse_len)]

b = (30.0/180)*np.pi
w_plus_cos = [(np.cos(b), pulse_len)]
w_minus_cos = [(-np.cos(b), pulse_len)]
w_plus_sin = [(np.sin(b), pulse_len)]
w_minus_sin = [(-np.sin(b), pulse_len)]

config['integration_weights']['cos']['cosine'] = w_plus_cos
config['integration_weights']['cos']['sine'] = w_minus_sin
config['integration_weights']['sin']['cosine'] = w_plus_sin
config['integration_weights']['sin']['sine'] = w_plus_cos
config['integration_weights']['minus_sin']['cosine'] = w_minus_sin
config['integration_weights']['minus_sin']['sine'] = w_minus_cos

with program() as ge_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    Q = declare(fixed)

    l = declare(int)

    Q_st = declare_stream()
    I_st = declare_stream()
    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(l, l_min, l < l_max + dl/2, l+dl):

            with for_(t, T_min, t < T_max + dt/2, t + dt):

                wait(reset_time//4, "storage_mode1")
                play("CW"*amp(1.0), "storage_mode1", duration=l)
                align("qubit_mode0", "storage_mode1")
                play("pi", "qubit_mode0")
                wait(t, "qubit_mode0")
                align('qubit_mode0', 'rr')
                measure("readout", "rr", None,
                        dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                        dual_demod.full('sin', 'out1', 'cos', 'out2', Q))
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        Q_st.buffer(len(l_vec), len(times)).average().save('Q')
        I_st.buffer(len(l_vec), len(times)).average().save('I')

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

    res = result_handles.get('Q').fetch_all()
    I = result_handles.get('I').fetch_all()
    plt.figure()
    plt.pcolormesh(4*times/1e3, 4*l_vec, res, cmap='RdBu', shading='auto')
    plt.colorbar()
    plt.xlabel('Wait times ($\mu$s)')
    plt.ylabel('Storage coherent len (ns)')
    plt.show()


    job.halt()
    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 't1_alpha', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=res)
        f.create_dataset("time", data=4*times/1e3)
        f.create_dataset("len", data=4*l_vec)
