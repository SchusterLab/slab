"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, two_chi, disc_file_opt, disc_file, disc_file_gauss
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm import SimulationConfig, LoopbackInterface
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os, scipy
from slab.dataanalysis import get_next_filename
from fock_state_prep import snap_seq
"""Qubit t1 experiment with different rate of readout pulses to """


t_pi = int(4.2e3//4)
dt = int(t_pi)
T_max = int(t_pi*75)
T_min = int(0)
t_vec = np.arange(T_min, T_max + dt/2, dt)

wait_times = np.arange(20)*t_pi
wait_min = 0
wait_dt = t_pi
wait_max = np.max(wait_times)

avgs = 500
simulation = 0 #1 to simulate the pulses

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

opx_amp = 1.0

def storage_t1_snap(f_state=1.0):

    reset_time = int(1e5)

    with program() as storage_t1:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)        # Averaging
        t = declare(int)        # Wait time
        I = declare(fixed)
        Q = declare(fixed)
        res = declare(bool)
        I = declare(fixed)
        n_pi_n = declare(int)
        j = declare(int)
        r = declare(int)
        n_pi = declare(int)
        w = declare(int)
        m = declare(int)

        res_st = declare_stream()
        I_st = declare_stream()

        ###############
        # the sequence:
        ###############

        with for_(n, 0, n < avgs, n + 1):

            with for_(w, wait_min, w < wait_max + wait_dt/2, w + wait_dt):

                with for_(t, T_min, t < T_max + dt/2, t + dt):

                    wait(reset_time, "qubit_mode0")
                    # play('pi', 'qubit_mode0')
                    assign(n_pi, t/(w+t_pi))
                    assign(m, t-n_pi*(w+t_pi))
                    align('qubit_mode0', 'rr')
                    with for_(j, 0, j < n_pi, j+1):
                        play('clear', 'rr')
                        wait(250+w, 'rr')
                    wait(m+4, 'rr')
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)
                    save(I, I_st)

        with stream_processing():
            res_st.boolean_to_int().buffer(len(wait_times), len(t_vec)).average().save('res')
            I_st.buffer(len(wait_times), len(t_vec)).average().save('I')

    qmm = QuantumMachinesManager()
    qm = qmm.open_qm(config)

    if simulation:
        job = qm.simulate(storage_t1, SimulationConfig(25000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        job = qm.execute(storage_t1, duration_limit=0, data_limit=0)
        print ("Execution done")

    return job

# plt.figure()

job =  storage_t1_snap(f_state=1)
result_handles = job.result_handles
# result_handles.wait_for_all_values()
res = result_handles.get('res').fetch_all()
I = result_handles.get('I').fetch_all()

plt.figure()
plt.pcolormesh(4*t_vec/1e3, 4*(wait_times+t_pi)/1e3,  res, shading='auto')
plt.colorbar()
plt.show()
# # # #
# plt.figure()
# plt.plot(4*t_vec/1e3,  res[0], '.--')
# plt.plot(4*t_vec/1e3,  res[-1], '.--')
# plt.show()

# print ("Data collection done")

# job.halt()

path = os.getcwd()
data_path = os.path.join(path, "data/")
seq_data_file = os.path.join(data_path,
                             get_next_filename(data_path, 'qubit_t1_readout_heating', suffix='.h5'))
print(seq_data_file)

with File(seq_data_file, 'w') as f:
    f.create_dataset("I", data=I)
    f.create_dataset("res", data=res)
    f.create_dataset("time", data=4*t_vec)
    f.create_dataset("inv_rate", data=4*(wait_times+t_pi))