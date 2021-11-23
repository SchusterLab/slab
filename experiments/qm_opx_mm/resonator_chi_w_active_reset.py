"""
Created on May 2021

@author: Ankur Agrawal, Schuster Lab
"""
from configuration_IQ import config, rr_IF, rr_freq, biased_th_g_jpa, two_chi, disc_file
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from TwoStateDiscriminator_2103 import TwoStateDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from slab import*
from h5py import File
import scipy
import os
from slab.dataanalysis import get_next_filename
"""Cross-Kerr (Chi shift of the readout resonator with qubit in |g> and in |e>"""

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

    wait(1000//4, "jpa_pump")
    align("rr", "jpa_pump")
    play('pump_square', 'jpa_pump')
    discriminator.measure_state("clear", "out1", "out2", res_reset, I=I)
    wait(1000//4, 'rr')
    # save(I, "check")

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

f_min = -2.5e6
f_max = 2.5e6
df = 25e3
f_vec = rr_freq - np.arange(f_min, f_max + df/2, df)

reset_time = int(5e5)

avgs = 1000
simulation = 0
with program() as resonator_spectroscopy:

    f = declare(int)
    i = declare(int)
    I = declare(fixed)
    res = declare(bool)

    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(i, 0, i < avgs, i+1):

        with for_(f, f_min + rr_IF, f <=f_max + rr_IF, f + df):

            update_frequency("rr", f)
            wait(reset_time//4, 'rr')
            align('rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(I, Ig_st)
            save(res, Qg_st)

            align('qubit', 'rr', 'jpa_pump')

            wait(reset_time//4, 'qubit')
            play('pi', 'qubit')
            align('qubit', 'rr', 'jpa_pump')
            play('pump_square', 'jpa_pump')
            discriminator.measure_state("clear", "out1", "out2", res, I=I)

            save(I, Ie_st)
            save(res, Qe_st)

    with stream_processing():

        Ig_st.buffer(len(f_vec)).average().save('Ig')
        Qg_st.boolean_to_int().buffer(len(f_vec)).average().save('Qg')
        Ie_st.buffer(len(f_vec)).average().save('Ie')
        Qe_st.boolean_to_int().buffer(len(f_vec)).average().save('Qe')

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

if simulation:
    job = qm.simulate(resonator_spectroscopy, SimulationConfig(15000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = qm.execute(resonator_spectroscopy, duration_limit=0, data_limit=0)

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    Ig = np.array(res_handles.get("Ig").fetch_all())
    Qg = np.array(res_handles.get("Qg").fetch_all())

    Ie = np.array(res_handles.get("Ie").fetch_all())
    Qe = np.array(res_handles.get("Qe").fetch_all())

    print ("Data collection done")

    job.halt()
    plt.figure()
    plt.plot(Ig**2)
    plt.plot(Ie**2)
    plt.show()


    path = os.getcwd()
    data_path = os.path.join(path, "data/")
    seq_data_file = os.path.join(data_path,
                                 get_next_filename(data_path, 'resonator_chi_opt', suffix='.h5'))
    print(seq_data_file)

    with File(seq_data_file, 'w') as f:
        f.create_dataset("Ig", data=Ig)
        f.create_dataset("Qg", data=Qg)
        f.create_dataset("Ie", data=Ie)
        f.create_dataset("Qe", data=Qe)
        f.create_dataset("freqs", data=f_vec)