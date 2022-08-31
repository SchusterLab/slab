"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, storage_freq, two_chi, disc_file_opt
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
from fock_state_prep import opx_amp_to_alpha
"""Storage cavity self-Kerr calibration 
    1. Variable coherent drive on the storage
    2. Wait time (t)
    3. Same alpha drive with a phase advanced by (2*pi*ramsey_freq*t) 
    3. Number resolved pi pulse on the qubit at n=0
    +> Results in a 2D spectrum which can be fitted to extract self_Kerr
    ramsey experiment using analytic SNAP pulses to create |0>+|1> 
"""
ramsey_freq = 100e3
omega = 2*np.pi*ramsey_freq

dt = 250
T_min = 4
T_max = 30000
t_vec = np.arange(T_min, T_max + dt/2, dt)

dphi = omega*dt*1e-9/(2*np.pi)*4 #to convert to ns

avgs = 1000
reset_time = int(7.5e6)
simulation = 0 #1 to simulate the pulses

#Coherent drive length
l_min = 5
l_max = 25
dl = 2
l_vec = np.arange(l_min, l_max + dl/2, dl)

opx_amp = 1.0

alpha_vec = opx_amp_to_alpha(cav_amp=opx_amp, cav_len=4*l_vec)

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

with program() as storage_self_kerr:

    ##############################
    # declare real-time variables:
    ##############################

    n = declare(int)        # Averaging
    t = declare(int)        # Wait time
    l = declare(int)
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

        with for_(l, l_min, l < l_max + dl/2, l + dl):

            assign(phi, 0)

            with for_(t, T_min, t < T_max + dt/2, t + dt):

                wait(reset_time// 4, 'storage_mode1')# wait for the storage to relax, several T1s
                ########################
                play("CW"*amp(opx_amp), 'storage_mode1', duration=l)
                ########################
                wait(t, 'storage_mode1')
                frame_rotation_2pi(phi, 'storage_mode1')
                ########################
                play("CW"*amp(-opx_amp), 'storage_mode1', duration=l)
                ########################
                align('storage_mode1', 'qubit_mode0')
                play("res_pi", 'qubit_mode0')
                align('qubit_mode0', 'rr')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)
                assign(phi, phi + dphi)

                save(res, res_st)
                save(I, I_st)

    with stream_processing():
        res_st.boolean_to_int().buffer(len(l_vec), len(t_vec)).average().save('res')
        I_st.buffer(len(l_vec), len(t_vec)).average().save('I')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(storage_self_kerr, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()

else:
    job = qm.execute(storage_self_kerr, duration_limit=0, data_limit=0)
    print ("Execution done")

    result_handles = job.result_handles

    # res_handle = result_handles.get("res")
    # res_handle.wait_for_values(1)
    #
    # plt.figure()
    # while(result_handles.is_processing()):
    #     res = res_handle.fetch_all()
    #     plt.pcolormesh(res, cmap='RdBu')
    #     # plt.xlabel(r'Time ($\mu$s)')
    #     # plt.ylabel(r'$\Delta \nu$ (kHz)')
    #     plt.pause(5)
    #     plt.clf()
    #

    # result_handles.wait_for_all_values()
    res = result_handles.get('res').fetch_all()
    I = result_handles.get('I').fetch_all()

    plt.figure()
    plt.pcolormesh(res, cmap='RdBu', shading='auto')
    plt.colorbar()
    plt.show()

    job.halt()

    path = os.getcwd()
    data_path = os.path.join(path, "data/thesis/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'storage_self_kerr', suffix='.h5'))
    print(seq_data_file)
    with File(seq_data_file, 'w') as f:
        f.create_dataset("Q", data=res)
        f.create_dataset("I", data=I)
        f.create_dataset("time", data=4*t_vec/1e3)
        f.create_dataset("ramsey_freq", data=ramsey_freq)
        f.create_dataset("cavity_freq", data=storage_freq)
        f.create_dataset("cav_len", data=4*l_vec)
        f.create_dataset("cav_amp", data=1.0)
        f.create_dataset("alpha", data=alpha_vec)

