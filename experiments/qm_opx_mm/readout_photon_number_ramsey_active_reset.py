from configuration_IQ import config, qubit_LO, rr_LO, ge_IF, qubit_freq, disc_file
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
##################
# ramsey_prog:
##################

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)


ramsey_freq = 1000e3
omega = 2*np.pi*ramsey_freq

dt = 25

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

T_min = 0
T_max = 750
times = np.arange(T_min, T_max + dt/2, dt)

wait_tmin = 25
wait_tmax = 1000
wait_dt = 25
wait_tvec = np.arange(wait_tmin, wait_tmax + wait_dt/2, wait_dt)
t_buffer = 500

avgs = 1000
reset_time = 500000
simulation = 0
with program() as ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)      # Averaging
    i = declare(int)      # wait_times after CLEAR
    t = declare(int)        #array of time delays
    I = declare(fixed)
    res = declare(bool)
    phi = declare(fixed)

    I_st = declare_stream()
    res_st = declare_stream()

    ###############
    # the sequence:
    ###############
    with for_(n, 0, n < avgs, n + 1):

        with for_(i, wait_tmin, i < wait_tmax + wait_dt/2 , i + wait_dt):

            assign(phi, 0)

            with for_(t, T_min, t < T_max + dt/2, t + dt):

                # discriminator.measure_state("readout", "out1", "out2", res, I=I)
                # wait(reset_time//40, "rr")
                # play('pi', 'qubit_mode0', condition=res)
                wait(reset_time//4, "rr")
                play('readout', 'rr')
                align("rr", "qubit_mode0")
                wait(i, "qubit_mode0")
                play("pi2", "qubit_mode0")
                wait(t, "qubit_mode0")
                frame_rotation_2pi(phi, "qubit_mode0") #2pi is already multiplied to the phase
                play("pi2", "qubit_mode0")
                align("qubit_mode0", "rr")
                wait(t_buffer, "rr")
                discriminator.measure_state("readout", "out1", "out2", res, I=I)
                assign(phi, phi + dphi)

                save(I, I_st)
                save(res, res_st)

    with stream_processing():
        I_st.buffer(len(wait_tvec), len(times)).average().save('I')
        res_st.boolean_to_int().buffer(len(wait_tvec), len(times)).average().save('Q')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    """To simulate the pulse sequence"""
    job = qm.simulate(ramsey, SimulationConfig(150000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    start_time = time.time()
    """To run the actual experiment"""
    print("Experiment execution Done")
    job = qm.execute(ramsey, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    # res_handles.wait_for_all_values()

    I = res_handles.get('I').fetch_all()
    Q = res_handles.get('Q').fetch_all()
    plt.figure()
    plt.pcolormesh(4*times*1e6, 4*wait_tvec*1e6, Q, shading='auto')
    plt.colorbar()
    plt.xlabel('Ramsey time (μs)')
    plt.ylabel('Wait time (μs)')

    print("Data collection done")

    """Stop the output from OPX,heats up the fridge"""
    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'ramsey_square', suffix='.h5'))
    print(seq_data_file)

    wait_tvec = 4*wait_tvec
    times = 4*times
    with File(seq_data_file, 'w') as f:
        dset = f.create_dataset("I", data=I)
        dset = f.create_dataset("Q", data=Q)
        dset = f.create_dataset("wait_time", data=wait_tvec)
        dset = f.create_dataset("ramsey_times", data=times)