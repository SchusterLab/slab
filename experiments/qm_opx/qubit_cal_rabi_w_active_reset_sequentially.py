from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
from configuration_IQ import config, biased_th_g_jpa, disc_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slab import*
from slab.instruments import instrumentmanager
from slab.dsfit import*
from h5py import File
import os
from slab.dataanalysis import get_next_filename


simulation_config = SimulationConfig(
    duration=60000,
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=230, noisePower=0.5**2
    )
)

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file, lsb=True)

def active_reset(biased_th, to_excited=False):
    res_reset = declare(bool)
    I = declare(fixed)

    wait(1000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')

    if to_excited == False:
        with while_(I < biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(~res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')
    else:
        with while_(I > biased_th):
            align('qubit', 'rr', 'jpa_pump')
            with if_(res_reset):
                play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
            wait(1000//4, 'rr')

def amp_to_tvec(c_amp):
    cav_amp = 1.0
    t_min = int(4*(cav_amp/c_amp))
    t_max = int(50*(cav_amp/c_amp))
    dt = int(2*(cav_amp/c_amp))
    return t_min, t_max, dt

simulation = 0
wait_time = 500000
avgs = 1000

def qubit_rabi(q_amp):

    t_min, t_max, dt = amp_to_tvec(q_amp)

    t_vec = np.arange(t_min, t_max + dt/2, dt)

    with program() as power_rabi:

        n = declare(int)
        t = declare(int)

        res = declare(bool)
        I = declare(fixed)

        res_st = declare_stream()
        I_st = declare_stream()

        with for_(n, 0, n < avgs, n + 1):

            with for_(t, t_min, t < t_max + dt/2, t + dt):

                active_reset(biased_th_g_jpa)
                align('qubit', 'rr', 'jpa_pump')
                play('CW'*amp(q_amp), 'qubit', duration=t)
                align('qubit', 'rr', 'jpa_pump')
                play('pump_square', 'jpa_pump')
                discriminator.measure_state("clear", "out1", "out2", res, I=I)

                save(res, res_st)
                save(I, I_st)

        with stream_processing():
            res_st.boolean_to_int().buffer(len(t_vec)).average().save('res')
            I_st.buffer(len(t_vec)).average().save('I')

    qm = qmm.open_qm(config)

    if simulation:
        """To simulate the pulse sequence"""
        job = qm.simulate(power_rabi, SimulationConfig(15000))
        samples = job.get_simulated_samples()
        samples.con1.plot()

    else:
        """To run the actual experiment"""

        job = qm.execute(power_rabi, duration_limit=0, data_limit=0)

        result_handles = job.result_handles

        t_vec = 4*t_vec/1e3

        result_handles.wait_for_all_values()
        res = result_handles.get('res').fetch_all()
        I = result_handles.get('I').fetch_all()
        data_path = "S:\_Data\\210326 - QM_OPX\\data"
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'qubit_square_cal', suffix='.h5'))


        print(seq_data_file)
        with File(seq_data_file, 'w') as f:
            f.create_dataset("Q", data=res)
            f.create_dataset("I", data=I)
            f.create_dataset("amps", data=q_amp)
            f.create_dataset("times", data=t_vec)

    return

st_amp = np.arange(0.001, 0.01, 0.001)

for a in st_amp:
    print(a)
    qubit_rabi(a)