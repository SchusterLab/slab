from configuration_IQ import config, disc_file_opt, disc_file
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
readout = 'readout' #'clear'

if readout=='readout':
    disc = disc_file
else:
    disc = disc_file_opt

dt = 1000
T_max = 125000
T_min = 4
times = np.arange(T_min, T_max + dt/2, dt)

avgs = 1000
reset_time = 500000
simulation = 0 #1 to simulate the pulses

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc, lsb=True)

with program() as ge_t1:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    res = declare(bool)

    I_st = declare_stream()
    res_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            discriminator.measure_state(readout, "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            wait(reset_time//10, "qubit_mode0")
            play("pi", "qubit_mode0")
            wait(t, "qubit_mode0")
            align('qubit_mode0', 'rr')
            discriminator.measure_state(readout, "out1", "out2", res, I=I)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        I_st.buffer(len(times)).average().save('I')

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
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    job.halt()

    times = 4*times/1e3
    plt.figure()
    plt.plot(times, res, '.-')

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'ge_t1', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("I", data=I)
        f.create_dataset("Q", data=res)
        f.create_dataset("time", data=times)
