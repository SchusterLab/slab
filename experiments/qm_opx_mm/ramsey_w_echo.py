from configuration_IQ import config, ge_IF, qubit_freq, disc_file_opt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
"""Ramsey phase"""

ramsey_freq = 25e3
omega = 2*np.pi*ramsey_freq

dt = 1000

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 8
T_max = 150000
times = np.arange(T_min, T_max + dt/2, dt)
avgs = 1000
reset_time = 500000
simulation = 0

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # Amplitudes
    t = declare(int) #array of time delays
    phi = declare(fixed)
    res = declare(bool)
    I = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        assign(phi, 0)

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            align('qubit_mode0', 'rr')
            play('pi', 'qubit_mode0', condition=res)
            wait(reset_time//10, "qubit_mode0")
            play("pi2", "qubit_mode0")
            wait(t/2, "qubit_mode0")
            play('pi', 'qubit_mode0')
            frame_rotation_2pi(phi, "qubit_mode0") #2pi is already multiplied to the phase
            wait(t/2, "qubit_mode0")
            play("pi2", "qubit_mode0")
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            assign(phi, phi + dphi)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(times)).average().save('res')
        I_st.buffer(len(times)).average().save('I')

qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    job.halt()

    times = 4*times/1e3

    plt.plot(times, res, '.-')

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'ramsey_phase_echo', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Q", data=res)
        f.create_dataset("I", data=I)
        f.create_dataset("time", data=times)
        f.create_dataset("ramsey_freq", data=ramsey_freq)
        f.create_dataset("qubit_freq", data=qubit_freq)
