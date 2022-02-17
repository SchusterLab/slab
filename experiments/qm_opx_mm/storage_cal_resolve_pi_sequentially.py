"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, ge_IF, biased_th_g_jpa, two_chi, disc_file_opt
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename
import time

simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt, lsb=True)

###############
# qubit_mode0_spec_prog:
###############

def amp_to_tvec(c_amp):
    cav_amp = 1.0
    t_min = int(5*(cav_amp/c_amp))
    t_max = int(13*(cav_amp/c_amp))
    dt = (int(2*(cav_amp/c_amp)))
    return t_min, t_max, dt

t_chi = int(abs(0.5*1e9/two_chi[1])) #qubit_mode0 rotates by pi in this time

avgs = 200
reset_time = int(7.5e6)
simulation = 0

f_min = -5.5e6
f_max = 0.5e6
df = 200e3
f_vec = np.arange(f_min, f_max + df/2, df)

def storage_mode1_bd(cav_amp):

    t_min, t_max, dt = amp_to_tvec(cav_amp)

    t_vec = np.arange(t_min, t_max + dt/2, dt)

    with program() as bd:

        ##############################
        # declare real-time variables:
        ##############################

        n = declare(int)        # Averaging
        f = declare(int)        # Frequencies
        t = declare(int)
        res = declare(bool)
        I = declare(fixed)

        res_st = declare_stream()
        ###############
        # the sequence:
        ###############
        with for_(n, 0, n < avgs, n + 1):

            with for_(t, t_min, t <= t_max, t + dt):

                with for_(f, ge_IF[0] + f_min, f < ge_IF[0] + f_max + df/2, f + df):

                    wait(reset_time//4, 'storage_mode1')
                    align('storage_mode1', 'qubit_mode0')
                    update_frequency("qubit_mode0", f)
                    play('CW'*amp(cav_amp), 'storage_mode1', duration=t)
                    align('storage_mode1', 'qubit_mode0')
                    play("res_pi", "qubit_mode0")
                    align('qubit_mode0', 'rr')
                    discriminator.measure_state("clear", "out1", "out2", res, I=I)

                    save(res, res_st)

        with stream_processing():
            res_st.boolean_to_int().buffer(len(t_vec), len(f_vec)).average().save('res')

    qm = qmm.open_qm(config)

    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(bd, SimulationConfig(150000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        """To run the actual experiment"""
        job = qm.execute(bd, duration_limit=0, data_limit=0)
        print("Experiment done")
        result_handles = job.result_handles
        result_handles.wait_for_all_values()
        res = result_handles.get('res').fetch_all()

        job.halt()

        path = os.getcwd()
        data_path = os.path.join(path, "data/")
        # data_path = 'S:\\_Data\\210326 - QM_OPX\\data\\'
        seq_data_file = os.path.join(data_path,
                                     get_next_filename(data_path, 'alpha_cal_storage_mode1_spec', suffix='.h5'))
        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("res", data=res)
            f.create_dataset("amp", data=cav_amp)
            f.create_dataset("times", data=t_vec)
            f.create_dataset("freqs", data=f_vec)
            f.create_dataset("two_chi", data=two_chi[1])

    return

st_amp = list(np.arange(0.001, 0.01, 0.001))
st_amp.extend(np.arange(0.01, 0.1, 0.01))
st_amp.extend(np.arange(0.1, 1.05, 0.1))

# b = storage_mode1_bd(0.4)

for a in st_amp[5:]:
    print(a)
    storage_mode1_bd(a)