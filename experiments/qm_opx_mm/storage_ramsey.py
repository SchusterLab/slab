from configuration_IQ import config, ge_IF, storage_freq, disc_file_opt, two_chi
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
from h5py import File
import os
from slab.dsfit import*
from slab.dataanalysis import get_next_filename

"""Storage cavity ramsey experiment"""
ramsey_freq = 10e3
omega = 2*np.pi*ramsey_freq

dt = 2500
T_min = 4
T_max = 480000
t_vec = np.arange(T_min, T_max + dt/2, dt)

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

avgs = 2000
reset_time = int(7.5e6)
simulation = 0 #1 to simulate the pulses

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

opx_amp = 1.0
cav_len = 8

with program() as storage_ramsey:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    I = declare(fixed)
    Q = declare(fixed)
    res = declare(bool)
    I = declare(fixed)
    phi = declare(fixed)

    res_st = declare_stream()
    I_st = declare_stream()

    ###############
    # the sequence:
    ###############

    with for_(n, 0, n < avgs, n + 1):

        assign(phi, 0)

        with for_(t, T_min, t < T_max + dt/2, t + dt):

            wait(reset_time// 4, "storage_mode1")# wait for the storage to relax, several T1s
            play("CW"*amp(opx_amp), "storage_mode1", duration=cav_len)
            wait(t, 'storage_mode1')
            frame_rotation_2pi(phi, 'storage_mode1')
            play("CW"*amp(-opx_amp), "storage_mode1", duration=cav_len)
            align("storage_mode1", "qubit_mode0")
            play("res_pi", "qubit_mode0")
            align('qubit_mode0', 'rr')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)
            assign(phi, phi + dphi)

            save(res, res_st)
            save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
        I_st.buffer(len(t_vec)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(storage_ramsey, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(storage_ramsey, duration_limit=0, data_limit=0)
    print ("Execution done")

    result_handles = job.result_handles
    # result_handles.wait_for_all_values()
    #
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()
    plt.figure()
    plt.plot(4*t_vec/1e3, res, '.-')
    plt.show()
    print ("Data collection done")

    job.halt()
    path = os.getcwd()
    data_path = os.path.join(path, "data/thesis/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_ramsey', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Q", data=res)
        f.create_dataset("I", data=I)
        f.create_dataset("time", data=4*t_vec/1e3)
        f.create_dataset("ramsey_freq", data=ramsey_freq)
        f.create_dataset("cavity_freq", data=storage_freq)